#!/usr/bin/env python3
"""Redo the rescue/damage confidence-bin table + directional score-shift
analysis on v7/109-GCF data, reusing the already-computed holdout_diag_*.pkl
per-uid (gt_ecs, scores_b, scores_m) dumps from eval_holdout227_v7.py.
No new MILP compute needed -- pure re-aggregation."""
import pickle
import numpy as np
from scipy import stats

RES = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results"
METHODS = ["clean-vanilla", "clean-filt30", "clean-filt50", "clean-filt70",
           "dpz-vanilla", "dpz-filt30", "dpz-filt50", "dpz-filt70",
           "enzbert-vanilla", "enzbert-filt30", "enzbert-filt50", "enzbert-filt70"]

bins = {"low": (0.0, 0.33), "medium": (0.33, 0.66), "high": (0.66, 1.0001)}
counts = {b: {"preserved": 0, "promoted": 0, "demoted": 0, "total": 0} for b in bins}

n_configs_used = 0
top1_flips = 0
top1_total = 0
active_ec_changed = 0
active_ec_total = 0

# directional score-shift (only pairs where GT EC in baseline vocabulary)
delta_correct, delta_wrong = [], []

for method in METHODS:
    fp = f"{RES}/holdout_diag_{method}.pkl"
    try:
        d = pickle.load(open(fp, "rb"))
    except FileNotFoundError:
        print(f"SKIP {method}: no diag file")
        continue
    n_configs_used += 1
    for uid, (gt_ecs, scores_b, scores_m) in d.items():
        if not scores_b or not scores_m:
            continue
        gt_set = set(gt_ecs)
        # per-(protein, EC) delta over all ECs scored by baseline
        for ec, eb in scores_b.items():
            em = scores_m.get(ec)
            if em is None:
                continue
            delta = em - eb
            for bname, (lo, hi) in bins.items():
                if lo <= eb < hi:
                    counts[bname]["total"] += 1
                    if abs(delta) <= 0.01:
                        counts[bname]["preserved"] += 1
                    elif delta > 0.01:
                        counts[bname]["promoted"] += 1
                    else:
                        counts[bname]["demoted"] += 1
                    break
            # directional score-shift: only for the GT EC itself
            if ec in gt_set:
                if len(gt_set) == 1:  # unambiguous single-EC ground truth
                    pass
        # directional shift specifically for the GT EC (first/only GT EC)
        for gt_ec in gt_set:
            if gt_ec in scores_b and gt_ec in scores_m:
                d_gt = scores_m[gt_ec] - scores_b[gt_ec]
                delta_correct.append(d_gt)
            # wrong EC: a same-magnitude-scored competing EC (approx: mean delta over non-GT ECs)
        non_gt_deltas = [scores_m[ec] - scores_b[ec] for ec in scores_b
                         if ec not in gt_set and ec in scores_m]
        if non_gt_deltas:
            v = np.nanmean(non_gt_deltas)
            if not np.isnan(v):
                delta_wrong.append(v)

        # top-1 flip check
        top1_b = max(scores_b, key=scores_b.get)
        top1_m = max(scores_m, key=scores_m.get)
        top1_total += 1
        if (top1_b in gt_set) != (top1_m in gt_set):
            top1_flips += 1

        # active-EC-set change (tau=0.5)
        active_b = {ec for ec, s in scores_b.items() if s >= 0.5}
        active_m = {ec for ec, s in scores_m.items() if s >= 0.5}
        active_ec_total += 1
        if active_b != active_m:
            active_ec_changed += 1

print(f"configs used: {n_configs_used}/12\n")
print("=== confidence-bin refinement pattern ===")
for bname in ["low", "medium", "high"]:
    c = counts[bname]
    t = c["total"]
    if t == 0:
        continue
    print(f"  {bname:8s} n={t:6d}  preserved={100*c['preserved']/t:.1f}%  "
          f"promoted={100*c['promoted']/t:.1f}%  demoted={100*c['demoted']/t:.1f}%")

print(f"\ntop-1 flips: {top1_flips}/{top1_total} ({100*top1_flips/top1_total:.1f}% changed, "
      f"{100*(1-top1_flips/top1_total):.1f}% preserved)")
print(f"active-EC-set changed: {active_ec_changed}/{active_ec_total} "
      f"({100*active_ec_changed/active_ec_total:.1f}%)")

dc = np.array(delta_correct)
dw = np.array(delta_wrong)
print(f"\n=== directional score-shift ===")
print(f"n_correct={len(dc)} mean_delta_correct={dc.mean():.4f}")
print(f"n_wrong={len(dw)} mean_delta_wrong={dw.mean():.4f}")
if len(dc) > 1 and len(dw) > 1:
    tstat, pval = stats.ttest_ind(dc, dw, equal_var=False)
    print(f"Welch t-test: t={tstat:.3f} p={pval:.4g}")
