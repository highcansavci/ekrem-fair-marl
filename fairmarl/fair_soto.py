"""Faithful SOTO (Zimmer et al. 2021) baseline + exploit test.

Validity check: is the fairness exploit an artifact of our simplified GGF-REINFORCE
team, or does it hit the ACTUAL SOTO architecture? SOTO = per-agent Self-Oriented
sub-net (own-return/efficiency) + Team-Oriented sub-net (GGF welfare/equity), with
the behaviour annealing SO->TO over training (be competent first, then fair).

We anneal beta: 1 (pure self-oriented) -> 0 (pure team-oriented) over training, so
the converged team is fair. Then freeze it (beta=0) and train a best-response
defector. If free-ride >> 1, the exploit holds against real SOTO.

    python -m smax_bench.fair_soto --env donate fairmech
"""
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .fair_train import Pol, ggf, gini, jain, env_step, obs_fn


def make_rollout(env, n, T, B):
    pol = Pol()

    def rollout(so_p, to_p, beta, def_p, def_idx, key):
        def step(carry, t):
            u, key = carry
            o = obs_fn(u, t, T)
            so_l, to_l = pol.apply(so_p, o), pol.apply(to_p, o)
            key, km, ks, kt, kd = jax.random.split(key, 5)
            use_so = jax.random.bernoulli(km, beta, (B, n))   # which sub-net acts
            a_so = jax.random.categorical(ks, so_l)
            a_to = jax.random.categorical(kt, to_l)
            a_team = jnp.where(use_so, a_so, a_to)
            a_def = jax.random.categorical(kd, pol.apply(def_p, o))
            acts = jnp.where((jnp.arange(n) == def_idx)[None, :], a_def, a_team)
            return (env_step(env, u, acts), key), (o, acts, use_so)
        (uF, _), (obs, acts, m) = jax.lax.scan(step, (jnp.zeros((B, n)), key),
                                               jnp.arange(T))
        return uF, obs, acts, m

    return pol, rollout


def logp(pol, p, obs, acts):
    lsm = jax.nn.log_softmax(pol.apply(p, obs))
    return jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]


def train(env, n=4, T=200, B=512, iters=600, lr=3e-3, seed=0):
    pol, rollout = make_rollout(env, n, T, B)
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)
    so_p = pol.init(k1, jnp.zeros((1, 5)))
    to_p = pol.init(k2, jnp.zeros((1, 5)))
    txs = optax.adam(lr); opts = txs.init(so_p)
    txt = optax.adam(lr); optt = txt.init(to_p)
    anneal = int(0.7 * iters)

    @jax.jit
    def upd(so_p, to_p, opts, optt, beta, key):
        uF, obs, acts, m = rollout(so_p, to_p, beta, so_p, -1, key)
        own = uF - uF.mean(0, keepdims=True)              # (B,n) per-agent own adv
        gadv = ggf(uF) - ggf(uF).mean()                   # (B,) team welfare adv

        def so_loss(p):                                   # SO maximizes OWN return
            return -(own[None] * logp(pol, p, obs, acts) * m).mean()
        def to_loss(p):                                   # TO maximizes GGF welfare
            return -(gadv[None, :, None] * logp(pol, p, obs, acts) * (1 - m)).mean()
        us, opts = txs.update(jax.grad(so_loss)(so_p), opts)
        so_p = optax.apply_updates(so_p, us)
        ut, optt = txt.update(jax.grad(to_loss)(to_p), optt)
        to_p = optax.apply_updates(to_p, ut)
        return so_p, to_p, opts, optt, jain(uF).mean()

    for it in range(iters):
        beta = max(0.0, 1.0 - it / anneal)                # 1 -> 0
        key, k = jax.random.split(key)
        so_p, to_p, opts, optt, _ = upd(so_p, to_p, opts, optt, beta, k)

    # eval converged team (beta=0 -> pure Team-Oriented = fair)
    key, k = jax.random.split(key)
    uA, _, _, _ = rollout(so_p, to_p, 0.0, so_p, -1, k)
    fair_jain = float(jain(uA).mean())

    # best-response defector vs frozen SOTO team (beta=0)
    key, kd = jax.random.split(key)
    dp = pol.init(kd, jnp.zeros((1, 5)))
    txd = optax.adam(lr); optd = txd.init(dp)

    @jax.jit
    def dupd(dp, optd, key):
        uF, obs, acts, _ = rollout(so_p, to_p, 0.0, dp, 0, key)
        adv = uF[:, 0] - uF[:, 0].mean()
        def loss(p):
            return -(adv[None, :] * logp(pol, p, obs, acts)[:, :, 0]).mean()
        u, optd = txd.update(jax.grad(loss)(dp), optd)
        return optax.apply_updates(dp, u), optd
    for _ in range(iters):
        key, k = jax.random.split(key)
        dp, optd = dupd(dp, optd, k)

    key, k = jax.random.split(key)
    uE, _, _, _ = rollout(so_p, to_p, 0.0, dp, 0, k)
    ex_u = np.asarray(uE.mean(0))
    ex_jain = float(jain(uE).mean())
    fair_share = float(uE.sum(-1).mean()) / n
    free_ride = ex_u[0] / max(fair_share, 1e-9)
    print(f"\n=== faithful SOTO [env={env}, n={n}] ===")
    print(f"  converged SOTO team (beta=0, pure TO): Jain={fair_jain:.3f}  "
          f"util={np.round(uA.mean(0),1)}")
    print(f"  + best-response defector: util={np.round(ex_u,1)}  Jain={ex_jain:.3f}")
    print(f"  free-ride = {free_ride:.2f}x  (fair share = {fair_share:.1f})")
    return dict(env=env, fair_jain=fair_jain, ex_jain=ex_jain, free_ride=free_ride)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", nargs="+", default=["donate", "fairmech"])
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    res = [train(e, n=args.n, T=args.T, iters=args.iters, seed=args.seed)
           for e in args.env]
    print("\n=== SOTO VALIDITY CHECK ===")
    for r in res:
        tag = "EXPLOITED" if r["free_ride"] > 1.5 else "robust"
        print(f"  {r['env']:>8}: SOTO Jain={r['fair_jain']:.3f} -> +defector "
              f"free-ride={r['free_ride']:.2f}x  [{tag}]")


if __name__ == "__main__":
    main()
