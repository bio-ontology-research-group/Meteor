#!/usr/bin/env python3
"""FIGURE-GENERATING SCRIPT --- produces Figure 2 of the main text.

Draws figures/fig_decoy.pdf: spurious reactions introduced when the evidence
for 60 reactions is deleted and the network is rebuilt under four regimes.
The first three share METEOR's single-stage architecture and differ only in
the cost; the fourth applies METEOR's own cost in the two-stage shape that
probabilistic gap-filling takes, isolating the architecture's contribution.

Reads:  results/recovery_4arm/*.json   (eval/recovery_ablation.py, 108 genomes)
Writes: figures/fig_decoy.pdf
Numbers: 34.6 / 78.2 / 142.9 / 2006.6 mean spurious reactions.

Label placement note: the mean annotations used to sit at each group's 88th
percentile, which is inside the box, so they landed on top of the box, the
whisker and the fliers, and the highest one ran into the axes edge. They now
clear the topmost drawn element of their own group, and the y-limit is
derived from those label positions rather than from the data.
"""
import json, glob, os, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as st

BASE = "/ibex/scratch/projects/c2014/kexin/funcarve"
ap = argparse.ArgumentParser()
ap.add_argument("--datadir", default=f"{BASE}/meteor_v8/results/recovery_4arm")
ap.add_argument("--outdir", default=f"{BASE}/meteor_v8/paper/paper_main/figures")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)

ARMS = ["true", "twostage", "uniform", "shuffled"]
cols = {x: [] for x in ARMS}
for f in sorted(glob.glob(f"{a.datadir}/*.json")):
    d = json.load(open(f))
    if not all(x in d and "decoy" in d.get(x, {}) for x in ARMS):
        continue
    for x in ARMS:
        cols[x].append(d[x]["decoy"])
t, w, u, s = [np.array(cols[x]) for x in ARMS]
n = len(t)

# Categorical slots 1-4 of the validated palette. Identity is carried by the
# x-axis labels, so colour only separates the groups; it encodes nothing extra.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, AXIS = "#52514e", "#898781", "#c3c2b7"

fig, ax = plt.subplots(figsize=(5.2, 3.9))
data = [t, w, u, s]
labs = ["METEOR\n(evw)", "Two-stage\n(evw gap-fill)", "Uniform\n($p{=}0$)", "Shuffled"]

bp = ax.boxplot(data, tick_labels=labs, patch_artist=True, widths=0.5,
                showfliers=True, showmeans=True,
                flierprops=dict(marker="o", ms=2.6, alpha=0.35,
                                markerfacecolor=MUTED, markeredgecolor="none"),
                medianprops=dict(color=INK, lw=1.4),
                whiskerprops=dict(color=INK, lw=1.0),
                capprops=dict(color=INK, lw=1.0),
                meanprops=dict(marker="D", ms=4.2, markerfacecolor="white",
                               markeredgecolor=INK, markeredgewidth=1.0))
for patch, c in zip(bp["boxes"], COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor(INK)
    patch.set_linewidth(1.0)

ax.set_yscale("log")
ax.set_ylabel("Spurious reactions", fontsize=10, color=INK)
ax.set_title(f"Deletion-recovery ({n} genomes)", fontsize=10.5, color=INK, pad=8)
ax.yaxis.grid(True, alpha=0.25, color=AXIS, lw=0.7)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)
ax.tick_params(axis="x", labelsize=8, colors=INK, length=0)
ax.tick_params(axis="y", labelsize=8, colors=MUTED)

# Each label clears its own group: above the upper cap and above any flier.
means = [x.mean() for x in data]
tops = []
for i, x in enumerate(data):
    cap = bp["caps"][2 * i + 1].get_ydata()[0]
    fli = bp["fliers"][i].get_ydata()
    tops.append(max(cap, fli.max() if len(fli) else cap))

label_y = [tp * 1.45 for tp in tops]
for i, (m, ly) in enumerate(zip(means, label_y), start=1):
    ax.text(i, ly, f"$\\mu$={m:.0f}", ha="center", va="bottom",
            fontsize=8.5, color=INK)

# Fit the range to the data plus the labels. A floor at 1 wasted the lower
# third of the panel on empty decades and flattened every box.
lo = min(float(x.min()) for x in data)
ax.set_ylim(bottom=lo * 0.55, top=max(label_y) * 1.7)

tm = means[0]
fig.text(0.5, 0.02,
         "relative to METEOR:   two-stage %.1f$\\times$    uniform %.1f$\\times$"
         "    shuffled %.0f$\\times$" % (means[1]/tm, means[2]/tm, means[3]/tm),
         ha="center", fontsize=8, color=MUTED)

plt.tight_layout(rect=[0, 0.055, 1, 1])
out = f"{a.outdir}/fig_decoy.pdf"
plt.savefig(out, dpi=300); plt.savefig(out.replace(".pdf", ".png"), dpi=200)
print(f"-> {out}")
for x_, d_ in zip(ARMS, data):
    print(f"  {x_:10s} mean={d_.mean():7.1f}  ratio={d_.mean()/tm:6.2f}x")
print(f"  two-stage vs METEOR: worse in "
      f"{int((w > t).sum())}/{n}  p={st.wilcoxon(w, t).pvalue:.3g}")
print(f"  y range: {lo*0.55:.1f} to {max(label_y)*1.7:.0f}")
