"""RECFAIR + adversarial co-training: a robust-fair MARL policy that stays fair
among cooperators yet denies a self-interested free-rider.

Two ingredients vs the plain GGF-fair baseline (fair_train.py):
  (1) RECIPROCITY feature: each agent observes the running claim-rate of the
      OTHER agents (max/mean over j!=i). This lets the shared policy detect an
      over-claimer and switch from turn-taking to contesting.
  (2) ADVERSARIAL co-training: the fair team trains WITH a best-response defector
      in the loop; in defector episodes the fair reward is GGF over the
      COOPERATORS only (slots != defector), so it learns to deny the free-rider
      rather than subsidize it.

Kill criteria (pre-committed): success iff after training (a) all-fair Jain>=0.9
AND (b) a FRESHLY trained best-response defector free-rides <= ~1.3x.

    python -m smax_bench.fair_robust --env donate claim
"""
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .fair_train import Pol, ggf, gini, jain, env_step


def obs7(u, t, T, cr):
    """(B,n) utils + (B,n) claim-rate -> (B,n,7) with reciprocity features."""
    B, n = u.shape
    mean = u.mean(-1, keepdims=True); mn = u.min(-1, keepdims=True)
    is_min = (u == mn).astype(jnp.float32)
    tf = jnp.full_like(u, t / T)
    eye = jnp.eye(n, dtype=bool)[None]                 # (1,n,n)
    cr_b = jnp.broadcast_to(cr[:, None, :], (B, n, n)) # (B,i,j)
    max_other = jnp.where(eye, -1.0, cr_b).max(-1)     # (B,n) max_{j!=i} cr_j
    mean_other = jnp.where(eye, 0.0, cr_b).sum(-1) / (n - 1)
    return jnp.stack([u / T, (u - mean) / T, (u - mn) / T, is_min, tf,
                      max_other, mean_other], -1)


def make_rollout(env, n, T, B):
    pol = Pol()

    def rollout(fair_p, def_p, def_idx, key):
        def step(carry, t):
            u, cc, key = carry
            cr = cc / jnp.clip(t, 1.0, None)
            o = obs7(u, t, T, cr)
            logits = jnp.where((jnp.arange(n) == def_idx)[None, :, None],
                               pol.apply(def_p, o), pol.apply(fair_p, o))
            key, ka = jax.random.split(key)
            acts = jax.random.categorical(ka, logits)
            return (env_step(env, u, acts), cc + (acts == 1), key), (o, acts)
        init = (jnp.zeros((B, n)), jnp.zeros((B, n)), key)
        (uF, _, _), (obs, acts) = jax.lax.scan(step, init, jnp.arange(T))
        return uF, obs, acts

    return pol, rollout


def logp_of(pol, p, obs, acts):
    lsm = jax.nn.log_softmax(pol.apply(p, obs))
    return jnp.take_along_axis(lsm, acts[..., None], -1)[..., 0]


def best_response_defector(pol, rollout, fair_p, n, iters, lr, key):
    """Train a fresh self-interested defector (agent 0) vs frozen fair_p."""
    key, ki = jax.random.split(key)
    dp = pol.init(ki, jnp.zeros((1, 7)))
    tx = optax.adam(lr); opt = tx.init(dp)

    @jax.jit
    def upd(dp, opt, key):
        uF, obs, acts = rollout(fair_p, dp, 0, key)
        R = uF[:, 0]; adv = R - R.mean()
        def loss(p):
            lp = logp_of(pol, p, obs, acts)[:, :, 0]
            return -(adv[None, :] * lp).mean()
        g = jax.grad(loss)(dp); u, opt = tx.update(g, opt)
        return optax.apply_updates(dp, u), opt
    for _ in range(iters):
        key, k = jax.random.split(key)
        dp, opt = upd(dp, opt, k)
    return dp


def train(env, n=4, T=100, B=512, iters=700, lr=3e-3, seed=0):
    pol, rollout = make_rollout(env, n, T, B)
    key = jax.random.PRNGKey(seed)
    key, kf, kd = jax.random.split(key, 3)
    fair_p = pol.init(kf, jnp.zeros((1, 7)))
    def_p = pol.init(kd, jnp.zeros((1, 7)))
    txf = optax.adam(lr); optf = txf.init(fair_p)
    txd = optax.adam(lr); optd = txd.init(def_p)
    coop = (jnp.arange(n) != 0).astype(jnp.float32)

    @jax.jit
    def step(fair_p, def_p, optf, optd, key):
        # (i) defector best-responds to current fair policy
        key, k = jax.random.split(key)
        uF, obs, acts = rollout(fair_p, def_p, 0, k)
        advd = uF[:, 0] - uF[:, 0].mean()
        def dloss(p):
            return -(advd[None, :] * logp_of(pol, p, obs, acts)[:, :, 0]).mean()
        ud, optd = txd.update(jax.grad(dloss)(def_p), optd)
        def_p = optax.apply_updates(def_p, ud)

        # (ii) fair update: all-fair batch + defector-present batch (coop reward)
        key, k1, k2 = jax.random.split(key, 3)
        uA, obsA, actsA = rollout(fair_p, fair_p, -1, k1)
        advA = ggf(uA) - ggf(uA).mean()
        uD, obsD, actsD = rollout(fair_p, def_p, 0, k2)
        RD = ggf(uD[:, 1:]); advD = RD - RD.mean()       # cooperators' welfare

        def floss(p):
            lA = -(advA[None, :, None] * logp_of(pol, p, obsA, actsA)).mean()
            lpD = logp_of(pol, p, obsD, actsD) * coop[None, None, :]
            lD = -(advD[None, :, None] * lpD).sum(-1).mean()
            return lA + lD
        uf, optf = txf.update(jax.grad(floss)(fair_p), optf)
        fair_p = optax.apply_updates(fair_p, uf)
        return fair_p, def_p, optf, optd, key, jain(uA).mean()

    for it in range(iters):
        fair_p, def_p, optf, optd, key, _ = step(fair_p, def_p, optf, optd, key)

    # ---- evaluation ----
    key, k = jax.random.split(key)
    uA, _, _ = rollout(fair_p, fair_p, -1, k)
    allfair_jain = float(jain(uA).mean()); allfair_eff = float(uA.sum(-1).mean()) / T
    # FRESH best-response defector vs the frozen robust fair policy
    key, k = jax.random.split(key)
    dp = best_response_defector(pol, rollout, fair_p, n, iters // 2, lr, k)
    key, k = jax.random.split(key)
    uE, _, _ = rollout(fair_p, dp, 0, k)
    ex_u = np.asarray(uE.mean(0))
    ex_jain = float(jain(uE).mean()); ex_gini = float(gini(uE).mean())
    fair_share = float(uE.sum(-1).mean()) / n
    free_ride = ex_u[0] / max(fair_share, 1e-9)

    ok = (allfair_jain >= 0.9) and (free_ride <= 1.3)
    print(f"\n=== ROBUST-FAIR (RECFAIR+adversarial) [env={env}, n={n}] ===")
    print(f"  all-fair: util={np.round(uA.mean(0),1)}  Jain={allfair_jain:.3f}  "
          f"eff={allfair_eff:.3f}")
    print(f"  +fresh best-response defector: util={np.round(ex_u,1)}  "
          f"Jain={ex_jain:.3f}  Gini={ex_gini:.3f}")
    print(f"  free-ride = {free_ride:.2f}x  (baseline plain-GGF was ~3-4x)")
    print(f"  KILL CRITERIA: all-fair Jain>=0.9 -> {allfair_jain>=0.9}; "
          f"free-ride<=1.3 -> {free_ride<=1.3}   ==> {'PASS' if ok else 'FAIL'}")
    return dict(env=env, allfair_jain=allfair_jain, free_ride=free_ride,
                ex_jain=ex_jain, pass_=ok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", nargs="+", default=["claim", "donate"])
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--iters", type=int, default=700)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    res = [train(e, n=args.n, T=args.T, iters=args.iters, seed=args.seed)
           for e in args.env]
    print("\n=== SUMMARY ===")
    for r in res:
        print(f"  {r['env']:>7}: all-fair Jain={r['allfair_jain']:.3f}  "
              f"free-ride={r['free_ride']:.2f}x  -> {'PASS' if r['pass_'] else 'FAIL'}")


if __name__ == "__main__":
    main()
