"""Per-environment exploit bar chart: baseline vs EKREM free-ride factor."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

envs = ["donate", "claim", "matthew", "SOTO", "gridworld"]
baseline = [4.0, 3.0, 4.0, 4.0, 1.90]
ekrem = [1.0, 1.0, 1.0, 1.0, 1.22]

x = np.arange(len(envs)); w = 0.38
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.bar(x - w / 2, baseline, w, label="baseline (welfare-fair)", color="tab:red")
ax.bar(x + w / 2, ekrem, w, label="EKREM (ours)", color="tab:green")
ax.axhline(1.0, ls="--", c="k", lw=0.8, label="fair share (1$\\times$)")
ax.set_xticks(x); ax.set_xticklabels(envs)
ax.set_ylabel("defector free-ride factor $\\rho$")
ax.set_title("Exploitability across environments ($N{=}4$)")
ax.legend(fontsize=8)
for i, (b, e) in enumerate(zip(baseline, ekrem)):
    ax.text(i - w / 2, b + 0.08, f"{b:.1f}", ha="center", fontsize=7)
    ax.text(i + w / 2, e + 0.08, f"{e:.2f}", ha="center", fontsize=7)
fig.tight_layout()
fig.savefig("paper/fair_exploit_bars.png", dpi=140)
print("wrote paper/fair_exploit_bars.png")
