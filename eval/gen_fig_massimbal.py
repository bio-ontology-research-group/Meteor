#!/usr/bin/env python3
"""FIGURE-GENERATING SCRIPT --- produces Figure S1 of the supplement.

Draws figures/fig_massimbal.pdf: per-genome counts of mass-imbalanced
reactions for the threshold baseline and for METEOR, across the 108-genome
panel, with medians marked.

Reads:  results/table1/table1_*.json  (the same per-genome records as
        main-text Table 1, produced by eval/gen_table1.py)
Writes: figures/fig_massimbal.pdf
Numbers: baseline medians 2092 / 2755 / 2342, METEOR medians 1858 / 2677 /
        2154 for CLEAN / DeepProZyme / EnzBERT.
"""
import json, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = "/ibex/scratch/projects/c2014/kexin/funcarve"
panel = set(l.split("\t")[0].strip() for l in
            open(f"{B}/meteor_v7_run/downstream_results/panel108_gram.tsv") if l.strip())
acc = {}
for f in glob.glob(f"{B}/meteor_v8/results/table1/table1_*.json"):
    o = json.load(open(f))
    if o["gca"] not in panel: continue
    for c, v in o.items():
        if isinstance(v, dict) and "mass_imbal" in v:
            acc.setdefault(c, []).append(v["mass_imbal"])

PRED = [("clean", "CLEAN"), ("dpz", "DeepProZyme"), ("enzbert", "EnzBERT")]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
xs, labels, colors = [], [], []
for i, (k, lab) in enumerate(PRED):
    xs.append(acc[f"baseline_{k}"]); labels.append(f"{lab}\nbaseline"); colors.append("#b0b7c3")
    xs.append(acc[f"meteor_{k}"]);   labels.append(f"{lab}\nMETEOR");   colors.append("#5b8def")

pos = [1, 2, 4, 5, 7, 8]
for p, data, c in zip(pos, xs, colors):
    jitter = (np.random.default_rng(0).random(len(data)) - 0.5) * 0.5
    ax.scatter(np.full(len(data), p) + jitter, data, s=5, alpha=0.35, color=c,
               edgecolors="none", zorder=2)
    med = float(np.median(data))
    ax.hlines(med, p - 0.38, p + 0.38, color="black", lw=1.8, zorder=3)
    ax.text(p, med, f" {med:.0f}", va="bottom", ha="center", fontsize=7.5, zorder=4)

ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("mass-imbalanced reactions\nper genome", fontsize=9)
ax.tick_params(axis="y", labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.25, lw=0.5)
fig.tight_layout()
out = f"{B}/meteor_v8/paper/paper_main/figures/fig_massimbal.pdf"
fig.savefig(out, bbox_inches="tight")
print("written", out)
for k, lab in PRED:
    print(f"{lab:12s} baseline median {np.median(acc['baseline_'+k]):7.0f}   "
          f"METEOR median {np.median(acc['meteor_'+k]):7.0f}")
