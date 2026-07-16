#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results"

bins_main = ["<30%", "30–50%", "50–70%", "70–90%", ">90%"]
n_main =    [2911,   17013,    15838,    11234,    19801]
delta_main = [1.13,  0.58,     0.36,     0.08,     0.25]
corr_per_1k_main = [17.5, 8.1, 4.5, 1.1, 2.8]
corr_main = [51,     138,      71,       12,       55]
regr_main = [18,     39,       14,       3,        5]

x = np.arange(len(bins_main))

# Teal/blue-green color scheme
TEAL = "#4A90D9"
TEAL_DARK = "#2E6DB4"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                gridspec_kw={"width_ratios": [1, 1]})

# === Panel (a): ΔTop-1 ===
bars1 = ax1.bar(x, delta_main, width=0.55, color=TEAL, edgecolor=TEAL_DARK,
                linewidth=0.6, zorder=3)

for i in range(len(bins_main)):
    ax1.text(x[i], delta_main[i] + 0.03,
             f"+{delta_main[i]:.2f}",
             ha="center", va="bottom", fontsize=8.5, color="black")

ax1.set_xticks(x)
ax1.set_xticklabels(bins_main, fontsize=9.5)
ax1.set_xlabel("Max. sequence identity to training set", fontsize=10)
ax1.set_ylabel(r"$\Delta\,$Top-1 accuracy (%)", fontsize=10)
ax1.set_ylim(0, 1.45)
ax1.grid(axis="y", alpha=0.25, linestyle="-", zorder=0)
ax1.set_title(r"$\Delta\,$Top-1 improvement", fontsize=11, fontweight="bold")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# === Panel (b): Corrections per 1k with corr/regr labels ===
bars2 = ax2.bar(x, corr_per_1k_main, width=0.55, color=TEAL,
                edgecolor=TEAL_DARK, linewidth=0.6, zorder=3)

for i in range(len(bins_main)):
    ax2.text(x[i], corr_per_1k_main[i] + 0.3,
             f"{corr_main[i]}/{regr_main[i]}",
             ha="center", va="bottom", fontsize=8.5, color="black")

ax2.set_xticks(x)
ax2.set_xticklabels(bins_main, fontsize=9.5)
ax2.set_xlabel("Max. sequence identity to training set", fontsize=10)
ax2.set_ylabel("Corrections per 1,000 proteins", fontsize=10)
ax2.set_ylim(0, 21)
ax2.grid(axis="y", alpha=0.25, linestyle="-", zorder=0)
ax2.set_title("Correction rate (corr/regr)", fontsize=11, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout(w_pad=2.5)
fig.savefig(f"{OUT_DIR}/seq_identity_dpz_figure.pdf", bbox_inches="tight", dpi=300)
fig.savefig(f"{OUT_DIR}/seq_identity_dpz_figure.png", bbox_inches="tight", dpi=150)
print(f"Saved: {OUT_DIR}/seq_identity_dpz_figure.pdf")
print(f"Saved: {OUT_DIR}/seq_identity_dpz_figure.png")
