"""FEN-style Matthew-effect GRIDWORLD + fairness-exploitability test.

A 2D grid with N agents and one resource. An agent collects the resource by
standing on it (+1 utility); the resource then respawns at a random cell. The
MATTHEW EFFECT: an agent's movement speed scales with its wealth -- richer agents
move every step, poorer agents move only intermittently -> the rich reach
resources first and get richer. Fair play therefore needs the rich to YIELD so
the poor can collect.

Allocators (collection rule when an agent is on the resource):
  matthew : the agent on the resource collects (ties -> richest). Exploitable.
  fairneed: only an agent BELOW its fair share collects (need-based). Robust.

Two-phase test: train a GGF-welfare fair team (dense GGF-increment reward, so
navigation is learnable), freeze it, train a best-response self-interested
defector, measure free-ride + Jain.

    python -m smax_bench.fair_grid --alloc matthew fairneed
"""
import argparse
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

DR = jnp.array([0, -1, 1, 0, 0])
DC = jnp.array([0, 0, 0, -1, 1])


class Pol(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(64)(x))
        x = nn.tanh(nn.Dense(64)(x))
        return nn.Dense(5)(x)


def ggf(u):
    n = u.shape[-1]
    s = jnp.sort(u, -1)
    w = jnp.arange(n, 0, -1).astype(jnp.float32); w = w / w.sum()
    return jnp.sum(s * w, -1)


def jain(u):
    return (u.sum(-1) ** 2) / (u.shape[-1] * jnp.clip((u ** 2).sum(-1), 1e-9, None))


def gini(u):
    s = jnp.sort(u, -1); n = u.shape[-1]
    c = jnp.cumsum(s, -1); tot = jnp.clip(c[..., -1], 1e-9, None)
    return (n + 1 - 2 * c.sum(-1) / tot) / n


class GridEnv:
    def __init__(self, n=4, G=6, T=80, alloc="matthew"):
        self.n, self.G, self.T, self.alloc = n, G, T, alloc

    def obs(self, pos, u, rpos, t):
        B, n = u.shape
        mean = u.mean(-1, keepdims=True)
        fair_share = t / self.n                          # expected fair cumulative
        below = (u < fair_share).astype(jnp.float32)
        d = (rpos[:, None, :] - pos) / self.G            # (B,n,2) resource offset
        return jnp.concatenate([
            pos / self.G, d, (u / self.T)[..., None],
            ((u - mean) / self.T)[..., None], below[..., None],
            jnp.full((B, n, 1), t / self.T)], -1)         # (B,n,8)

    def step(self, pos, u, rpos, t, acts, key):
        B, n = u.shape
        # Matthew speed: above-mean wealth -> move every step; else p=0.5
        mean = u.mean(-1, keepdims=True)
        move_p = jnp.where(u >= mean, 1.0, 0.5)           # (B,n)
        km, kr = jax.random.split(key)
        do_move = jax.random.bernoulli(km, move_p)        # (B,n)
        delta = jnp.stack([DR[acts], DC[acts]], -1)       # (B,n,2)
        npos = jnp.clip(pos + delta * do_move[..., None], 0, self.G - 1)
        on_res = jnp.all(npos == rpos[:, None, :], -1)    # (B,n)
        fair_share = (t + 1) / self.n
        if self.alloc == "matthew":
            elig = on_res
            winner = jnp.argmax(jnp.where(elig, u, -1e9), -1)
        else:                                             # fairneed
            elig = on_res & (u < fair_share)
            winner = jnp.argmin(jnp.where(elig, u, 1e9), -1)
        collected = elig.any(-1)
        u2 = u + jax.nn.one_hot(winner, n) * collected[:, None]
        # respawn resource where collected
        kk = jax.random.split(kr, B)
        new_r = jax.vmap(lambda k: jax.random.randint(k, (2,), 0, self.G))(kk)
        rpos2 = jnp.where(collected[:, None], new_r, rpos)
        return npos, u2, rpos2


def make_rollout(env, B):
    pol = Pol()
    n, G, T = env.n, env.G, env.T

    def rollout(fair_p, def_p, def_idx, key):
        key, kp, kr = jax.random.split(key, 3)
        pos = jax.random.randint(kp, (B, n, 2), 0, G)
        rpos = jax.random.randint(kr, (B, 2), 0, G)
        u = jnp.zeros((B, n))

        def stp(carry, t):
            pos, u, rpos, key = carry
            o = env.obs(pos, u, rpos, t)
            logits = jnp.where((jnp.arange(n) == def_idx)[None, :, None],
                               pol.apply(def_p, o), pol.apply(fair_p, o))
            key, ka, ks = jax.random.split(key, 3)
            acts = jax.random.categorical(ka, logits)
            u_before = u
            pos, u, rpos = env.step(pos, u, rpos, t, acts, ks)
            r_team = ggf(u) - ggf(u_before)               # dense GGF increment
            r_def = u[:, 0] - u_before[:, 0]
            return (pos, u, rpos, key), (o, acts, r_team, r_def)
        (_, uF, _, _), (obs, acts, rt, rd) = jax.lax.scan(
            stp, (pos, u, rpos, key), jnp.arange(T))
        return uF, obs, acts, rt, rd

    return pol, rollout


def returns_to_go(r, gamma=0.99):                          # r (T,B) -> (T,B)
    def f(carry, x):
        g = x + gamma * carry
        return g, g
    _, G = jax.lax.scan(f, jnp.zeros(r.shape[1:]), r, reverse=True)
    return G


def logp(pol, p, obs, acts):
    lsm = jax.nn.log_softmax(pol.apply(p, obs))
    return jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]


def train(alloc, n=4, G=6, T=80, B=512, iters=800, lr=3e-3, seed=0):
    env = GridEnv(n=n, G=G, T=T, alloc=alloc)
    pol, rollout = make_rollout(env, B)
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    fair_p = pol.init(ki, jnp.zeros((1, 8)))
    txf = optax.adam(lr); optf = txf.init(fair_p)

    @jax.jit
    def fair_upd(fair_p, optf, key):
        uF, obs, acts, rt, _ = rollout(fair_p, fair_p, -1, key)
        G_t = returns_to_go(rt)                            # (T,B)
        adv = G_t - G_t.mean(1, keepdims=True)
        def loss(p):
            lp = logp(pol, p, obs, acts)                   # (T,B,n)
            return -(adv[:, :, None] * lp).mean()
        g = jax.grad(loss)(fair_p); upd, optf = txf.update(g, optf)
        return optax.apply_updates(fair_p, upd), optf, jain(uF).mean()
    for _ in range(iters):
        key, k = jax.random.split(key)
        fair_p, optf, _ = fair_upd(fair_p, optf, k)

    key, k = jax.random.split(key)
    uA, _, _, _, _ = rollout(fair_p, fair_p, -1, k)
    fair_jain = float(jain(uA).mean()); fair_u = np.asarray(uA.mean(0))
    fair_eff = float(uA.sum(-1).mean()) / T

    # best-response defector vs frozen fair team
    key, kd = jax.random.split(key)
    def_p = pol.init(kd, jnp.zeros((1, 8)))
    txd = optax.adam(lr); optd = txd.init(def_p)

    @jax.jit
    def def_upd(def_p, optd, key):
        uF, obs, acts, _, rd = rollout(fair_p, def_p, 0, key)
        G_t = returns_to_go(rd)
        adv = G_t - G_t.mean(1, keepdims=True)
        def loss(p):
            return -(adv * logp(pol, p, obs, acts)[:, :, 0]).mean()
        g = jax.grad(loss)(def_p); upd, optd = txd.update(g, optd)
        return optax.apply_updates(def_p, upd), optd
    for _ in range(iters):
        key, k = jax.random.split(key)
        def_p, optd = def_upd(def_p, optd, k)

    key, k = jax.random.split(key)
    uE, _, _, _, _ = rollout(fair_p, def_p, 0, k)
    ex_u = np.asarray(uE.mean(0)); ex_jain = float(jain(uE).mean())
    share = float(uE.sum(-1).mean()) / n
    fr = ex_u[0] / max(share, 1e-9)
    print(f"\n=== FEN gridworld [alloc={alloc}, n={n}, G={G}] ===")
    print(f"  fair team: util={np.round(fair_u,1)} Jain={fair_jain:.3f} "
          f"eff(collect/step)={fair_eff:.3f}")
    print(f"  +defector: util={np.round(ex_u,1)} Jain={ex_jain:.3f}")
    print(f"  free-ride = {fr:.2f}x  (fair share={share:.1f})")
    return dict(alloc=alloc, fair_jain=fair_jain, ex_jain=ex_jain, free_ride=fr,
                fair_eff=fair_eff)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alloc", nargs="+", default=["matthew", "fairneed"])
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--G", type=int, default=6)
    p.add_argument("--T", type=int, default=80)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    res = [train(a, n=args.n, G=args.G, T=args.T, iters=args.iters, seed=args.seed)
           for a in args.alloc]
    print("\n=== FEN GRIDWORLD SUMMARY ===")
    for r in res:
        tag = "EXPLOITED" if r["free_ride"] > 1.5 else "robust"
        print(f"  {r['alloc']:>8}: fair Jain={r['fair_jain']:.3f} eff={r['fair_eff']:.2f}"
              f" -> +defector free-ride={r['free_ride']:.2f}x [{tag}]")


if __name__ == "__main__":
    main()
