"""Architecture / algorithm schematic for the paper:
 (a) the per-agent policy (SOTO: Self-Oriented + Team-Oriented sub-nets, mixed);
 (b) the allocation mechanism: baseline (grab -> claimant) vs EKREM (need -> worst-off).
Drawn with matplotlib (no TikZ dependency) -> paper/fair_arch.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.3))


def box(ax, x, y, w, h, text, fc="#eef2ff", ec="#3b5bdb", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.2, color="#333"))


# ---- (a) SOTO policy architecture ----
ax1.set_title("(a) Fair agent policy (SOTO)", fontsize=9)
box(ax1, 0.05, 0.42, 0.18, 0.16, "obs\n$o_i$", fc="#f1f3f5", ec="#868e96")
box(ax1, 0.34, 0.62, 0.30, 0.18, "Self-Oriented net\n(own return)", fc="#fff0f0", ec="#e03131")
box(ax1, 0.34, 0.20, 0.30, 0.18, "Team-Oriented net\n(GGF welfare)", fc="#ebfbee", ec="#2f9e44")
box(ax1, 0.72, 0.42, 0.22, 0.16, r"mix $\beta{:}1{\to}0$" + "\n$\\to$ action", fc="#eef2ff", ec="#3b5bdb")
arrow(ax1, 0.23, 0.52, 0.34, 0.70)
arrow(ax1, 0.23, 0.48, 0.34, 0.30)
arrow(ax1, 0.64, 0.71, 0.79, 0.58)
arrow(ax1, 0.64, 0.29, 0.79, 0.50)
ax1.text(0.5, 0.04, "self-interested early $\\to$ fair at convergence",
         ha="center", fontsize=7, style="italic", color="#555")

# ---- (b) allocation mechanism ----
ax2.set_title("(b) Allocation mechanism", fontsize=9)
box(ax2, 0.02, 0.60, 0.30, 0.18, "fair team\n($N{-}1$ agents)", fc="#ebfbee", ec="#2f9e44")
box(ax2, 0.02, 0.22, 0.30, 0.18, "defector\n(best response)", fc="#fff0f0", ec="#e03131")
box(ax2, 0.40, 0.40, 0.24, 0.20, "ALLOCATOR", fc="#fff9db", ec="#f08c00", fs=9)
box(ax2, 0.72, 0.42, 0.26, 0.16, "utilities $\\mathbf{u}$\n$\\to$ Jain / GGF", fc="#f1f3f5", ec="#868e96")
arrow(ax2, 0.32, 0.69, 0.40, 0.55)
arrow(ax2, 0.32, 0.31, 0.40, 0.45)
arrow(ax2, 0.64, 0.50, 0.72, 0.50)
ax2.text(0.52, 0.30, "baseline: claimant wins\n(exploitable)", ha="center",
         fontsize=6.5, color="#e03131")
ax2.text(0.52, 0.74, "EKREM: worst-off wins\n(robust)", ha="center",
         fontsize=6.5, color="#2f9e44")

for ax in (ax1, ax2):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.tight_layout()
fig.savefig("paper/fair_arch.png", dpi=160)
print("wrote paper/fair_arch.png")
