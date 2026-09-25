"""Figure S1: per-genome dead-end and mass-imbalance distributions, threshold vs METEOR.
Run from the repository root: python eval/gen_fig_massimbal.py (reads results/table1/)."""
import json, glob, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
panel = {l.split("\t")[0].strip() for l in open(f"{ROOT}/cohorts/panel108_gram.tsv") if l.strip()}
acc = {}
for f in glob.glob(f"{ROOT}/results/table1/table1_*.json"):
    o = json.load(open(f))
    if o["gca"] not in panel: continue
    for c, v in o.items():
        if isinstance(v, dict) and "deadends" in v:
            for k in ("n_selected","deadends","mi_frac"): acc.setdefault((c,k), []).append(v[k])
PRED = [("clean","CLEAN"),("dpz","DeepProZyme"),("enzbert","EnzBERT")]
METRIC = [("n_selected","selected reactions"),("deadends","dead-end metabolites"),("mi_frac","mass-imbalanced fraction")]
# Three panels stacked (not side-by-side) so they share one x-axis: the six
# predictor x method groups are identical across (a)/(b)/(c), so only the
# bottom panel needs tick labels, and each panel gets the full figure width
# instead of a cramped third of it. Per-point "thr"/"METEOR" sub-labels are
# dropped in favour of a single legend, since two adjacent long predictor
# names (e.g. "DeepProZyme thr" / "DeepProZyme METEOR") no longer have to
# fit side by side at 1-unit spacing.
fig, axes = plt.subplots(3, 1, figsize=(6.2, 8.6), sharex=True)
pos = [1,2,4,5,7,8]; group_pos = [1.5,4.5,7.5]; rng = np.random.default_rng(0)
for ax, (mk, mlab), tag in zip(axes, METRIC, "abc"):
    data, colors = [], []
    for k, lab in PRED:
        data += [acc[(f"baseline_{k}",mk)], acc[(f"meteor_{k}",mk)]]
        colors += ["#b0b7c3","#5b8def"]
    for p, d, c in zip(pos, data, colors):
        ax.scatter(np.full(len(d), p) + (rng.random(len(d))-0.5)*0.5, d, s=5, alpha=0.35, color=c, edgecolors="none", zorder=2)
        ax.hlines(np.median(d), p-0.38, p+0.38, color="black", lw=1.6, zorder=3)
    ax.set_xticks(pos, minor=True)
    ax.set_ylabel(mlab, fontsize=9.5)
    ax.set_title(f"({tag})", loc="left", fontsize=10)
    ax.tick_params(axis="y", labelsize=8.5); ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y", alpha=0.25, lw=0.5)
axes[-1].set_xticks(group_pos)
axes[-1].set_xticklabels([lab for _, lab in PRED], fontsize=9)
axes[-1].tick_params(axis="x", length=0)
legend_handles = [Line2D([0],[0], marker="o", color="none", markerfacecolor="#b0b7c3", markersize=6, label="threshold ($\\tau{=}0.5$)"),
                  Line2D([0],[0], marker="o", color="none", markerfacecolor="#5b8def", markersize=6, label="METEOR")]
fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=[0,0,1,0.98])
fig.savefig(os.path.join(ROOT, "figures", "fig_massimbal.pdf"), bbox_inches="tight"); print("written")
