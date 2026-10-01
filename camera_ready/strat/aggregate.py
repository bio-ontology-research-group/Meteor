"""Aggregate results/strat/{gem}_{pred}.json -> results/strat_summary.json (+ strat_summary.md).
mean +- sd over organisms per predictor x arm x stratum; paired exact Wilcoxon (6 pairs) for
M-rxn vs B in-vocab, M-rxn vs S out-of-vocab, M-rxn vs S in-vocab (and a few extra pairs, labelled)."""
import os, sys, json, glob, argparse
import numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["B", "M-rxn", "M-act", "S", "F", "M-rxn-allSEED", "S-allSEED"]
STRATA = ["all", "in_vocab", "out_vocab", "bin0", "bin(0,0.1)", "bin[0.1,0.5)", "bin[0.5,1]", "bin0_in_vocab",
          "reach_skeleton", "reach_universal_only", "reach_not_in_seed",
          "out_vocab_reach_skeleton", "out_vocab_reach_universal_only", "out_vocab_reach_not_in_seed",
          "H3_universal_only_score_lt_0.01", "H3_universal_only_score_0", "no_candidate_rxn", "out_vocab_no_candidate_rxn", "in_vocab_no_candidate_rxn"]
PAIRS = [("M-rxn", "B", "in_vocab"), ("M-rxn", "S", "out_vocab"), ("M-rxn", "S", "in_vocab"),
         ("M-rxn", "B", "all"), ("M-rxn", "S", "all"), ("M-act", "S", "out_vocab"),
         ("M-rxn", "S", "bin(0,0.1)"), ("M-rxn", "S", "bin[0.1,0.5)"), ("M-rxn", "B", "bin[0.5,1]"),
         ("M-act", "S-allSEED", "out_vocab"), ("M-act", "S-allSEED", "in_vocab"), ("M-act", "B", "in_vocab")]

def msd(v):
    v = [x for x in v if x is not None]
    return dict(mean=round(float(np.mean(v)), 4), sd=round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None, n_org=len(v)) if v else dict(mean=None, sd=None, n_org=0)

def wtest(x, y):
    d = np.array(x, float) - np.array(y, float)
    if len(d) < 2 or np.all(d == 0): return dict(p=None, note="all differences zero" if len(d) else "no data", n=len(d), mean_diff=float(d.mean()) if len(d) else None)
    r = wilcoxon(x, y, method="exact", zero_method="wilcox")
    return dict(p=round(float(r.pvalue), 4), stat=float(r.statistic), n=len(d), mean_diff=round(float(d.mean()), 4), n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()))

ap = argparse.ArgumentParser(); ap.add_argument("--indir", default=f"{HERE}/results/strat"); ap.add_argument("--out", default=f"{HERE}/results/strat_summary.json")
a = ap.parse_args()
recs = [json.load(open(p)) for p in sorted(glob.glob(f"{a.indir}/*_*.json")) if not os.path.basename(p).startswith("skeleton_")]
preds = sorted({r["pred"] for r in recs}); summ = dict(n_records=len(recs), predictors={}, gates_failed=[])
for r in recs:
    for k, v in r.get("gates", {}).items():
        if not v["ok"]: summ["gates_failed"].append(dict(gem=r["gem"], pred=r["pred"], gate=k, vals=v["vals"]))
for p in preds:
    rs = sorted([r for r in recs if r["pred"] == p], key=lambda r: r["gem"]); orgs = [r["organism"] for r in rs]
    P = dict(organisms=orgs, vocab_size=rs[0]["vocab_size"], gem_ec=[r["gem_ec"] for r in rs],
             n_in_vocab=msd([r["n_in_vocab"] for r in rs]), n_out_vocab=msd([r["n_out_vocab"] for r in rs]),
             frac_out_vocab=msd([r["n_out_vocab"] / r["gem_ec"] for r in rs]),
             bin_sizes={b: msd([r["bin_sizes"][b] for r in rs]) for b in rs[0]["bin_sizes"]},
             reach_counts_all={k: msd([r["reach_counts_all"][k] for r in rs]) for k in rs[0]["reach_counts_all"]},
             reach_counts_out_vocab={k: msd([r["reach_counts_out_vocab"][k] for r in rs]) for k in rs[0]["reach_counts_out_vocab"]},
             H3_count_lt_001=msd([r["table"]["B"]["H3_universal_only_score_lt_0.01"]["n"] for r in rs]),
             no_candidate_rxn=msd([r["n_no_candidate_rxn"] for r in rs]), out_vocab_no_candidate_rxn=msd([r["n_out_vocab_no_candidate_rxn"] for r in rs]),
             n_candidate=msd([r["n_candidate"] for r in rs]),
             dark={k: msd([r["dark"][k] for r in rs]) for k in rs[0]["dark"]},
             arm_sizes={arm: msd([r["arm_sizes"][arm] for r in rs]) for arm in ARMS},
             recall={arm: {x: dict(**msd([r["table"][arm][x]["recall"] for r in rs]), n_mean=msd([r["table"][arm][x]["n"] for r in rs])["mean"]) for x in STRATA} for arm in ARMS},
             per_organism={r["organism"]: {arm: {x: r["table"][arm][x]["recall"] for x in ("all", "in_vocab", "out_vocab")} for arm in ARMS} for r in rs},
             wilcoxon={})
    for a1, a2, x in PAIRS:
        xs = [r["table"][a1][x]["recall"] for r in rs]; ys = [r["table"][a2][x]["recall"] for r in rs]
        if None in xs or None in ys: continue
        P["wilcoxon"][f"{a1}_vs_{a2}__{x}"] = wtest(xs, ys)
    summ["predictors"][p] = P
json.dump(summ, open(a.out, "w"), indent=1)
# markdown digest
L = [f"# strat summary ({len(recs)} records)", ""]
if summ["gates_failed"]: L += ["GATES FAILED: " + json.dumps(summ["gates_failed"]), ""]
for p, P in summ["predictors"].items():
    L += [f"## {p}  vocab={P['vocab_size']}  orgs={P['organisms']}  out-of-vocab frac={P['frac_out_vocab']['mean']}±{P['frac_out_vocab']['sd']}",
          "", "| arm | all | in_vocab | out_vocab | bin0 | (0,0.1) | [0.1,0.5) | [0.5,1] | reach_skel | reach_univ_only | not_in_seed |", "|" + "---|" * 11]
    for arm in ARMS:
        c = lambda x: f"{P['recall'][arm][x]['mean']}±{P['recall'][arm][x]['sd']}" if P['recall'][arm][x]['mean'] is not None else "-"
        L.append(f"| {arm} | " + " | ".join(c(x) for x in ["all", "in_vocab", "out_vocab", "bin0", "bin(0,0.1)", "bin[0.1,0.5)", "bin[0.5,1]", "reach_skeleton", "reach_universal_only", "reach_not_in_seed"]) + " |")
    L.append("| n (mean) | " + " | ".join(str(P['recall']['B'][x]['n_mean']) for x in ["all", "in_vocab", "out_vocab", "bin0", "bin(0,0.1)", "bin[0.1,0.5)", "bin[0.5,1]", "reach_skeleton", "reach_universal_only", "reach_not_in_seed"]) + " |")
    L += ["", f"dark: {P['dark']}", f"H3 count (universal-only & score<0.01): {P['H3_count_lt_001']}  |  no-candidate-reaction ECs (true ceiling): {P['no_candidate_rxn']} (out-of-vocab {P['out_vocab_no_candidate_rxn']})  n_candidate={P['n_candidate']}", f"reach out-of-vocab: {P['reach_counts_out_vocab']}",
          "wilcoxon: " + json.dumps(P["wilcoxon"]), ""]
open(a.out.replace(".json", ".md"), "w").write("\n".join(L)); print("\n".join(L)); print("->", a.out)
