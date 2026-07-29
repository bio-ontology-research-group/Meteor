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
"""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as st

BASE = "/ibex/scratch/projects/c2014/kexin/funcarve"
DATADIR = f"{BASE}/meteor_v8/results/recovery_4arm"
OUTDIR = f"{BASE}/meteor_v8/paper/paper_main/figures"
os.makedirs(OUTDIR, exist_ok=True)

ARMS = ["true", "twostage", "uniform", "shuffled"]
cols = {a: [] for a in ARMS}
for f in sorted(glob.glob(f"{DATADIR}/*.json")):
    d = json.load(open(f))
    if not all(a in d and "decoy" in d.get(a, {}) for a in ARMS):
        continue
    for a in ARMS:
        cols[a].append(d[a]["decoy"])
t, w, u, s = [np.array(cols[a]) for a in ARMS]
n = len(t)

fig, ax = plt.subplots(figsize=(5.0, 3.8))
data = [t, w, u, s]
labs = ["METEOR\n(evw)", "Two-stage\n(evw gap-fill)", "Uniform\n($p{=}0$)", "Shuffled"]
colors = ["#4477AA", "#66AA88", "#EE6677", "#BBBBBB"]

bp = ax.boxplot(data, labels=labs, patch_artist=True, widths=0.55,
                showfliers=True, flierprops=dict(marker="o", ms=3, alpha=0.4))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)

ax.set_yscale("log")
ax.set_ylabel("Spurious reactions", fontsize=10)
ax.set_title(f"Deletion-recovery ({n} genomes)", fontsize=10)
ax.yaxis.grid(True, alpha=0.3)
ax.set_ylim(bottom=0.8)
ax.tick_params(axis="x", labelsize=8)

means = [x.mean() for x in data]
for i, (x, m, c) in enumerate(zip(data, means, colors), start=1):
    ax.text(i, np.percentile(x, 88), f"μ={m:.0f}", ha="center", fontsize=8.5,
            color=c if c != "#BBBBBB" else "#666666")

tm = means[0]
fig.text(0.5, 0.015,
         "relative to METEOR:  two-stage %.1fx    uniform %.1fx    shuffled %.0fx"
         % (means[1]/tm, means[2]/tm, means[3]/tm),
         ha="center", fontsize=8, color="#444444")

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f"{OUTDIR}/fig_decoy.pdf", dpi=300)
print(f"-> {OUTDIR}/fig_decoy.pdf")
for a, x in zip(ARMS, data):
    print(f"  {a:10s} mean={x.mean():7.1f}  ratio={x.mean()/tm:6.2f}x")
print(f"  two-stage vs METEOR: worse in "
      f"{int((w > t).sum())}/{n}  p={st.wilcoxon(w, t).pvalue:.3g}")
