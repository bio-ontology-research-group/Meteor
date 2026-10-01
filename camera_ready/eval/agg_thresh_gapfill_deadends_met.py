"""Aggregate T10: dead ends as a fraction of submodel metabolites, growing threshold baseline vs METEOR."""
import json, glob, statistics as st
from scipy.stats import wilcoxon
R = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/results"
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/thresh_gapfill_deadends_met/tgd_*.json"))]
ms = lambda v: dict(mean=round(st.mean(v), 5), sd=round(st.stdev(v), 5), median=round(st.median(v), 5))
unchanged_bad = []
for r in rows:
    o = json.load(open(f"{R}/thresh_gapfill_deadends/tgd_{r['gca']}.json"))
    for k in ("repaired", "raw_threshold", "meteor"):
        for f in ("deadends", "n_rxn"):
            if r[k][f] != o[k][f]: unchanged_bad.append((r["gca"], k, f, r[k][f], o[k][f]))
def frac(r, k, internal=False):
    x = r[k]
    return (x["deadends_internal"] / x["n_metabolites_internal"]) if internal else (x["deadends"] / x["n_metabolites"])
def compare(internal):
    b = [frac(r, "repaired", internal) for r in rows]; m = [frac(r, "meteor", internal) for r in rows]
    d = [y - x for x, y in zip(b, m)]; w = wilcoxon(m, b)
    return dict(repaired_baseline=ms(b), meteor=ms(m), raw_threshold=ms([frac(r, "raw_threshold", internal) for r in rows]),
                meteor_lower=sum(x < 0 for x in d), equal=sum(x == 0 for x in d), meteor_higher=sum(x > 0 for x in d),
                wilcoxon_stat=float(w.statistic), wilcoxon_p=float(w.pvalue))
out = dict(
    definition="fraction = dead-end metabolites / len(model.metabolites) of the exact submodel passed to MEMOTE find_deadends; "
               "internal = same restricted to metabolites whose compartment is not 'extracellular' (submodel compartments: cytosol, extracellular).",
    n=len(rows),
    fraction_all_metabolites=compare(False),
    fraction_internal_metabolites=compare(True),
    n_metabolites=dict((k, ms([r[k]["n_metabolites"] for r in rows])) for k in ("repaired", "meteor", "raw_threshold")),
    n_metabolites_internal=dict((k, ms([r[k]["n_metabolites_internal"] for r in rows])) for k in ("repaired", "meteor", "raw_threshold")),
    deadends_internal_equals_deadends=sum(r[k]["deadends_internal"] == r[k]["deadends"] for r in rows for k in ("repaired", "meteor", "raw_threshold")),
    unchanged_vs_first_run=dict(checked=len(rows) * 3 * 2, mismatches=len(unchanged_bad), examples=unchanged_bad[:10]),
)
json.dump(out, open(f"{R}/thresh_gapfill_deadends_met_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
