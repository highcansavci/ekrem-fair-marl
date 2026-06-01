"""Solidify the fairness-exploitability arc: scaling (team size) x multi-seed.

For each n and seed, run the two-phase exploit test under:
  - baseline 'donate' (plain GGF-welfare fair team) -> how exploitable?
  - 'fairmech' (fairness-aware allocator, our fix)   -> exploit closed?

Reports mean+/-std free-ride and post-defector Jain vs team size, and a figure.
Hypotheses: baseline free-ride GROWS with n (more cooperators to free-ride on);
the allocator holds free-ride ~1 and Jain ~1 at every n.
"""
import argparse
import numpy as np

from .fair_train import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--iters", type=int, default=500)
    args = p.parse_args()

    rows = []                                          # (n, env, fr, exj)
    for n in args.ns:
        for env in ("donate", "fairmech"):
            frs, exjs = [], []
            for s in args.seeds:
                r = train(env, n=n, T=args.T, fair_iters=args.iters,
                          def_iters=args.iters, seed=s)
                frs.append(r["free_ride"]); exjs.append(r["ex_jain"])
            rows.append((n, env, np.mean(frs), np.std(frs),
                         np.mean(exjs), np.std(exjs)))

    print("\n\n=== SOLIDIFY SUMMARY (mean +/- std over seeds) ===")
    print(f"  {'n':>3} {'condition':>10} | {'free-ride':>16} | {'post-defector Jain':>18}")
    for n, env, fr, frs, exj, exjs in rows:
        name = "baseline" if env == "donate" else "fair-alloc"
        print(f"  {n:>3} {name:>10} | {fr:>7.2f} +/- {frs:>4.2f}     | "
              f"{exj:>7.3f} +/- {exjs:>4.3f}")

    # CSV + figure
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/fair_solidify.csv", "w") as f:
        f.write("n,condition,free_ride_mean,free_ride_std,exjain_mean,exjain_std\n")
        for n, env, fr, frs, exj, exjs in rows:
            f.write(f"{n},{env},{fr:.4f},{frs:.4f},{exj:.4f},{exjs:.4f}\n")
    print("  wrote -> results/fair_solidify.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ns = args.ns
        base = {n: None for n in ns}; fix = {n: None for n in ns}
        bse = {}; fse = {}
        for n, env, fr, frs, exj, exjs in rows:
            (base if env == "donate" else fix)[n] = fr
            (bse if env == "donate" else fse)[n] = frs
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.errorbar(ns, [base[n] for n in ns], yerr=[bse[n] for n in ns],
                    marker="o", label="baseline GGF-fair (exploitable)", color="tab:red")
        ax.errorbar(ns, [fix[n] for n in ns], yerr=[fse[n] for n in ns],
                    marker="s", label="fairness-aware allocator (ours)", color="tab:green")
        ax.plot(ns, ns, ls=":", color="grey", label="max free-ride (= n)")
        ax.axhline(1.0, ls="--", color="k", lw=0.8, label="fair share (1x)")
        ax.set_xlabel("team size n"); ax.set_ylabel("defector free-ride factor")
        ax.set_title("Fairness exploitability vs team size")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig("results/fair_solidify.png", dpi=140)
        print("  wrote -> results/fair_solidify.png")
    except Exception as e:
        print(f"  (figure skipped: {e})")


if __name__ == "__main__":
    main()
