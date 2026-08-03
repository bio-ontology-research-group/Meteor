#!/usr/bin/env python3
"""FIGURE-GENERATING SCRIPT --- produces Figure 2 of the main text.

Draws figures/fig_decoy.pdf: off-reference reactions introduced when the evidence
for 60 reactions is deleted and the network is rebuilt under four regimes.
The first three share METEOR's single-stage architecture and differ only in
the cost; the fourth applies METEOR's own cost in the two-stage shape that
probabilistic gap-filling takes, isolating the architecture's contribution.

Reads:  results/recovery_4arm/*.json   (eval/recovery_ablation.py, 108 genomes)
Writes: figures/fig_decoy.pdf
Numbers: 34.6 / 78.2 / 142.9 / 2006.6 mean off-reference reactions.

Orientation: horizontal. The measured quantity spans three decades, so the
log axis wants the long side of the panel, and four regime names read as
plain single-line labels on the category axis instead of wrapping onto two
lines under a vertical one. The panel then fills the full text width for
about the same height a half-width vertical panel cost.

Label placement: the mean annotation for each regime sits to the right of
whatever that regime draws furthest right --- upper cap or outermost flier
--- so labels cannot land on the data, and the x-limit is derived from the
label positions rather than from the data. The vertical version placed them
at the 88th percentile, inside the box.
"""
import json, glob, os, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--datadir", default=os.path.join(ROOT, "results", "recovery_4arm"))
ap.add_argument("--outdir", default=os.path.join(ROOT, "figures"))
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
# category labels, so colour only separates the groups; it encodes nothing.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, AXIS = "#52514e", "#898781", "#c3c2b7"

# Bottom-to-top on a horizontal boxplot, so reverse for top-to-bottom reading.
data = [t, w, u, s]
labs = ["METEOR (evw)", "Two-stage (evw gap-fill)", "Uniform ($p{=}0$)", "Shuffled"]
order = list(range(len(data)))[::-1]
data_p = [data[i] for i in order]
labs_p = [labs[i] for i in order]
cols_p = [COLORS[i] for i in order]

fig, ax = plt.subplots(figsize=(5.2, 1.85))
bp = ax.boxplot(data_p, tick_labels=labs_p, patch_artist=True, widths=0.55,
                orientation="horizontal",
                showfliers=True, showmeans=True,
                flierprops=dict(marker="o", ms=2.4, alpha=0.35,
                                markerfacecolor=MUTED, markeredgecolor="none"),
                medianprops=dict(color=INK, lw=1.4),
                whiskerprops=dict(color=INK, lw=1.0),
                capprops=dict(color=INK, lw=1.0),
                meanprops=dict(marker="D", ms=4.0, markerfacecolor="white",
                               markeredgecolor=INK, markeredgewidth=1.0))
for patch, c in zip(bp["boxes"], cols_p):
    patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor(INK)
    patch.set_linewidth(1.0)

ax.set_xscale("log")
ax.set_xlabel("Off-reference reactions (log scale)", fontsize=9.5, color=INK)
ax.set_title(f"Perturbation stability ({n} genomes)", fontsize=10, color=INK, pad=6)
ax.xaxis.grid(True, alpha=0.25, color=AXIS, lw=0.7)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)
ax.tick_params(axis="y", labelsize=8, colors=INK, length=0)
ax.tick_params(axis="x", labelsize=8, colors=MUTED)

# Each label clears its own group: right of the upper cap and of any flier.
rights = []
for i, x in enumerate(data_p):
    cap = bp["caps"][2 * i + 1].get_xdata()[0]
    fli = bp["fliers"][i].get_xdata()
    rights.append(max(cap, fli.max() if len(fli) else cap))
label_x = [r * 1.35 for r in rights]
# The ratio to METEOR rides with each mean, so the panel carries it without a
# separate footer line; the reference arm needs no ratio.
tm = t.mean()
for i, (x, lx) in enumerate(zip(data_p, label_x), start=1):
    m = x.mean()
    txt = f"mean {m:.0f}" if abs(m - tm) < 1e-9 else \
          f"mean {m:.0f}   {m/tm:.1f}\u00d7" if m / tm < 10 else \
          f"mean {m:.0f}   {m/tm:.0f}\u00d7"
    ax.text(lx, i, txt, ha="left", va="center", fontsize=8.5, color=INK)

lo = min(float(x.min()) for x in data)
ax.set_xlim(left=lo * 0.6, right=max(label_x) * 4.0)

plt.tight_layout()
out = f"{a.outdir}/fig_decoy.pdf"
plt.savefig(out, dpi=300); plt.savefig(out.replace(".pdf", ".png"), dpi=200)
print(f"-> {out}")
for x_, d_ in zip(ARMS, data):
    print(f"  {x_:10s} mean={d_.mean():7.1f}  ratio={d_.mean()/tm:6.2f}x")
print(f"  two-stage vs METEOR: worse in "
      f"{int((w > t).sum())}/{n}  p={st.wilcoxon(w, t).pvalue:.3g}")
print(f"  x range: {lo*0.6:.1f} to {max(label_x)*2.6:.0f}")
