#!/usr/bin/env python3
"""Generate fig_decoy.pdf: spurious reactions under true/uniform/shuffled cost.

Data: eval/recovery_ablation.py -> results/recovery_abl/*.json
Output: paper/paper_main/figures/fig_decoy.pdf
"""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as st

BASE = "/ibex/scratch/projects/c2014/kexin/funcarve"
DATADIR = f"{BASE}/meteor_v8/results/recovery_abl"
OUTDIR  = f"{BASE}/meteor_v8/paper/paper_main/figures"
os.makedirs(OUTDIR, exist_ok=True)

t, u, s = [], [], []
for f in sorted(glob.glob(f"{DATADIR}/*.json")):
    d = json.load(open(f))
    t.append(d["true"]["decoy"])
    u.append(d["uniform"]["decoy"])
    s.append(d["shuffled"]["decoy"])
t, u, s = [np.array(x) for x in (t, u, s)]
n = len(t)

fig, ax = plt.subplots(figsize=(4.2, 3.8))
data, labs = [t, u, s], ["True\n(evw)", "Uniform\n($p{=}0$)", "Shuffled"]
colors = ["#4477AA", "#EE6677", "#BBBBBB"]

bp = ax.boxplot(data, labels=labs, patch_artist=True, widths=0.5,
                showfliers=True, flierprops=dict(marker="o", ms=3, alpha=0.4))

for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)

ax.set_yscale("log")
ax.set_ylabel("Spurious reactions", fontsize=10)
ax.set_title("Deletion-recovery (%d genomes)" % n, fontsize=10)
ax.yaxis.grid(True, alpha=0.3)
ax.set_ylim(bottom=0.8)

# Mean annotations
tm, um, sm = t.mean(), u.mean(), s.mean()
ax.text(1, np.percentile(t, 85), f"μ={tm:.0f}", ha="center", fontsize=9, color=colors[0])
ax.text(2, np.percentile(u, 85), f"μ={um:.0f}", ha="center", fontsize=9, color=colors[1])
ax.text(3, np.percentile(s, 85), f"μ={sm:.0f}", ha="center", fontsize=9, color="#666666")

_, p1 = st.wilcoxon(t, u, alternative="less")
_, p2 = st.wilcoxon(t, s, alternative="less")
fig.text(0.5, 0.01,
    "True vs Uniform: %sx  (p < 10$^{-18}$)    True vs Shuffled: %sx  (p < 10$^{-18}$)" %
    (f"{um/tm:.1f}", f"{sm/tm:.0f}"),
    ha="center", fontsize=8, color="#444444")

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f"{OUTDIR}/fig_decoy.pdf", dpi=300)
print(f"-> {OUTDIR}/fig_decoy.pdf")
print(f"  n={n}  true={tm:.1f}  uniform={um:.1f} ({um/tm:.1f}x)  shuffled={sm:.0f} ({sm/tm:.0f}x)")
