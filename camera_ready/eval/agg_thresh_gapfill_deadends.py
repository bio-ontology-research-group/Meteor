"""Aggregate T9 (growing threshold baseline dead ends, DeepProZyme-vanilla) and check controls vs Table 1 v2."""
import json, glob, statistics as st
from scipy.stats import wilcoxon
R = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/results"
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/thresh_gapfill_deadends/tgd_*.json"))]
ms = lambda v: dict(mean=round(st.mean(v), 3), sd=round(st.stdev(v), 3), median=st.median(v), min=min(v), max=max(v))
ctl_bad = []
for r in rows:
    t = json.load(open(f"{R}/table1_v2/table1_{r['gca']}.json"))
    for k, tk in (("raw_threshold", "baseline_dpz"), ("meteor", "meteor_dpz")):
        for f in ("n_selected", "n_rxn", "deadends", "mi_frac"):
            if r[k][f] != t[tk][f]:
                ctl_bad.append((r["gca"], k, f, r[k][f], t[tk][f]))
rep = [r["repaired"]["deadends"] for r in rows]; met = [r["meteor"]["deadends"] for r in rows]
raw = [r["raw_threshold"]["deadends"] for r in rows]
d = [m - b for m, b in zip(met, rep)]
w = wilcoxon(met, rep) if any(d) else None
out = dict(
    description="Growing threshold baseline: tau=0.5 DeepProZyme-vanilla draft minus structurally excluded reactions, repaired with grow_support (gamma_min=0.1), as the Section 3.2 comparator; structural metrics via the Table 1 profile().",
    n=len(rows),
    repaired_grows=sum(r["repaired_grows"] for r in rows), repair_found=sum(r["repair_found"] for r in rows),
    draft_grows=sum(r["draft_grows"] for r in rows),
    deadends=dict(repaired_baseline=ms(rep), meteor=ms(met), raw_threshold=ms(raw)),
    meteor_minus_repaired=dict(fewer=sum(x < 0 for x in d), equal=sum(x == 0 for x in d), more=sum(x > 0 for x in d),
                               wilcoxon_stat=None if w is None else float(w.statistic),
                               wilcoxon_p=None if w is None else float(w.pvalue)),
    n_selected=dict(repaired_baseline=ms([r["repaired"]["n_selected"] for r in rows]),
                    meteor=ms([r["meteor"]["n_selected"] for r in rows]),
                    raw_threshold=ms([r["raw_threshold"]["n_selected"] for r in rows])),
    n_core=ms([r["n_core"] for r in rows]), n_added_by_repair=ms([r["n_added_by_repair"] for r in rows]),
    mi_frac=dict(repaired_baseline=ms([r["repaired"]["mi_frac"] for r in rows]),
                 meteor=ms([r["meteor"]["mi_frac"] for r in rows])),
    fba_growth_table1_procedure=dict(repaired_baseline_nonzero=sum(r["repaired"]["fba_growth"] > 1e-6 for r in rows),
                                     meteor_nonzero=sum(r["meteor"]["fba_growth"] > 1e-6 for r in rows)),
    controls_vs_table1_v2=dict(mismatches=len(ctl_bad), examples=ctl_bad[:10]),
)
json.dump(out, open(f"{R}/thresh_gapfill_deadends_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
