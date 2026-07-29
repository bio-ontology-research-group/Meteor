#!/usr/bin/env python3
"""Evaluate METEOR on ALL NCBI EC-annotated proteins per genome.
Includes Fmax (numpy-optimized). Uses cached GBFF EC annotations.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import os, re, json, pickle, sys, time
import numpy as np
import pandas as pd

W = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026"
V7_RUN = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run"
CACHE_DIR = f"{V7_RUN}/downstream_results/ncbi_ec_cache"
OUT_DIR = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/downstream_results"

from baseline_io import resolve_baseline_pkl
from meteor_v8.utils import extract_pred as _XP, load_ec as _LE
_ANC=_LE(data_path('all_ancestors.txt'))



def fmax_fast(gt_indices, scores):
    """Compute Fmax using numpy. gt_indices = set of column indices for GT ECs."""
    n_gt = len(gt_indices)
    if n_gt == 0:
        return 0.0
    order = np.argsort(-scores)
    gt_mask = np.zeros(len(scores), dtype=bool)
    for i in gt_indices:
        gt_mask[i] = True
    sorted_mask = gt_mask[order]
    cum_tp = np.cumsum(sorted_mask).astype(float)
    n_pred = np.arange(1, len(scores) + 1, dtype=float)
    precision = cum_tp / n_pred
    recall = cum_tp / n_gt
    f1 = np.where((precision + recall) > 0,
                  2 * precision * recall / (precision + recall), 0.0)
    return float(np.max(f1))


BASELINE_NAME = "dpz"
VARIANT = "vanilla"
meteor_dir = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla"

manifest = json.load(open(f"{W}/scripts/genome_collection_manifest.json"))
all_gcfs = sorted(g for g in manifest["assemblies"].keys() if g.startswith("GCF"))
print(f"GCF assemblies: {len(all_gcfs)}", flush=True)

total_eval = 0
total_gcf_eval = 0
top1_base_hits = 0
top1_met_hits = 0
fmax_base_all = []
fmax_met_all = []
score_changes = []
per_gcf_results = []

t_start = time.time()

for gi, gcf in enumerate(all_gcfs):
    cache_file = os.path.join(CACHE_DIR, f"{gcf}_ec.json")
    if not os.path.exists(cache_file):
        continue
    met_pkl = os.path.join(meteor_dir, f"meteor_df_{gcf}.pkl")
    if not os.path.exists(met_pkl):
        continue
    base_pkl_path = resolve_baseline_pkl(BASELINE_NAME, VARIANT, gcf)
    if not base_pkl_path:
        continue

    ec_map = json.load(open(cache_file))
    ec_map = {k: set(v) for k, v in ec_map.items()}
    if not ec_map:
        continue

    try:
        base_df = _XP(base_pkl_path, _ANC)
        met_df = pickle.load(open(met_pkl, "rb"))
    except:
        continue

    if any(str(c).startswith("EC:") for c in base_df.columns[:5]):
        base_df.columns = [str(c).replace("EC:", "") for c in base_df.columns]

    b_cols = list(base_df.columns)
    m_cols = list(met_df.columns)
    b_col2idx = {c: i for i, c in enumerate(b_cols)}
    m_col2idx = {c: i for i, c in enumerate(m_cols)}

    n_matched = 0
    gcf_t1_b, gcf_t1_m = 0, 0
    gcf_fmax_b, gcf_fmax_m = [], []

    for pid, gt_ecs in ec_map.items():
        if pid not in base_df.index or pid not in met_df.index:
            continue
        n_matched += 1

        b_vals = base_df.loc[pid].values.astype(float)
        m_vals = met_df.loc[pid].values.astype(float)

        bt1_idx = int(np.argmax(b_vals))
        mt1_idx = int(np.argmax(m_vals))
        bt1 = b_cols[bt1_idx]
        mt1 = m_cols[mt1_idx]
        bh = bt1 in gt_ecs
        mh = mt1 in gt_ecs
        if bh: top1_base_hits += 1; gcf_t1_b += 1
        if mh: top1_met_hits += 1; gcf_t1_m += 1

        # Fmax
        b_gt_idx = {b_col2idx[ec] for ec in gt_ecs if ec in b_col2idx}
        m_gt_idx = {m_col2idx[ec] for ec in gt_ecs if ec in m_col2idx}
        fb = fmax_fast(b_gt_idx, b_vals)
        fm = fmax_fast(m_gt_idx, m_vals)
        fmax_base_all.append(fb)
        fmax_met_all.append(fm)
        gcf_fmax_b.append(fb)
        gcf_fmax_m.append(fm)

        for ec in gt_ecs:
            bsc = float(b_vals[b_col2idx[ec]]) if ec in b_col2idx else 0.0
            msc = float(m_vals[m_col2idx[ec]]) if ec in m_col2idx else 0.0
            d = msc - bsc
            score_changes.append({
                "uid": pid, "ec": ec, "gcf": gcf,
                "base": bsc, "meteor": msc, "delta": d,
                "base_top1_hit": bh, "met_top1_hit": mh,
                "base_top1": bt1, "met_top1": mt1,
                "fmax_base": fb, "fmax_meteor": fm,
            })

    if n_matched > 0:
        total_eval += n_matched
        total_gcf_eval += 1
        per_gcf_results.append({
            "gcf": gcf, "n_ec": len(ec_map), "n_matched": n_matched,
            "top1_base": gcf_t1_b, "top1_meteor": gcf_t1_m,
            "fmax_base": np.mean(gcf_fmax_b), "fmax_meteor": np.mean(gcf_fmax_m),
        })

    if (gi + 1) % 10 == 0 or gi == len(all_gcfs) - 1:
        print(f"  [{gi+1}/{len(all_gcfs)}] {total_eval} proteins, "
              f"top1 {top1_base_hits}/{top1_met_hits}, "
              f"{time.time()-t_start:.0f}s", flush=True)

# ── Results ──
print(f"\n{'='*70}", flush=True)
print(f"RESULTS: {BASELINE_NAME}_{VARIANT}", flush=True)
print(f"{'='*70}", flush=True)
print(f"GCFs evaluated: {total_gcf_eval}/{len(all_gcfs)}", flush=True)
print(f"Total proteins with NCBI EC matched: {total_eval}", flush=True)

if total_eval == 0:
    sys.exit(0)

print(f"\nTop-1 accuracy:", flush=True)
print(f"  Baseline: {top1_base_hits}/{total_eval} ({100*top1_base_hits/total_eval:.1f}%)", flush=True)
print(f"  METEOR:   {top1_met_hits}/{total_eval} ({100*top1_met_hits/total_eval:.1f}%)", flush=True)

print(f"\nFmax:", flush=True)
print(f"  Baseline: {np.mean(fmax_base_all):.4f} (median={np.median(fmax_base_all):.4f})", flush=True)
print(f"  METEOR:   {np.mean(fmax_met_all):.4f} (median={np.median(fmax_met_all):.4f})", flush=True)

# Per-protein Fmax changes
fmax_deltas = np.array(fmax_met_all) - np.array(fmax_base_all)
n_fmax_up = int(np.sum(fmax_deltas > 0.001))
n_fmax_down = int(np.sum(fmax_deltas < -0.001))
n_fmax_same = int(np.sum(np.abs(fmax_deltas) <= 0.001))
print(f"  Fmax improved: {n_fmax_up}, degraded: {n_fmax_down}, unchanged: {n_fmax_same}", flush=True)
print(f"  Mean Fmax delta: {np.mean(fmax_deltas):+.5f}", flush=True)

deltas = [s["delta"] for s in score_changes]
n_up = sum(1 for d in deltas if d > 0.001)
n_down = sum(1 for d in deltas if d < -0.001)
n_same = sum(1 for d in deltas if abs(d) <= 0.001)
print(f"\nGT EC score changes: boosted={n_up}, dampened={n_down}, unchanged={n_same}", flush=True)
print(f"Mean delta: {np.mean(deltas):+.5f}", flush=True)

# Corrections / regressions
seen = set()
corrections, regressions = [], []
for s in score_changes:
    key = (s["uid"], s["gcf"])
    if key in seen:
        continue
    seen.add(key)
    if not s["base_top1_hit"] and s["met_top1_hit"]:
        corrections.append(s)
    elif s["base_top1_hit"] and not s["met_top1_hit"]:
        regressions.append(s)

print(f"\nTop-1 corrections: {len(corrections)}, regressions: {len(regressions)}", flush=True)

if corrections:
    print("\nCorrections (top 15):", flush=True)
    for c in sorted(corrections, key=lambda x: -x["delta"])[:15]:
        print(f"  {c['gcf']}:{c['uid']} EC={c['ec']}: "
              f"{c['base']:.4f}->{c['meteor']:.4f} ({c['delta']:+.4f})", flush=True)

if regressions:
    print("\nRegressions (top 15):", flush=True)
    for r in sorted(regressions, key=lambda x: x["delta"])[:15]:
        print(f"  {r['gcf']}:{r['uid']} EC={r['ec']}: "
              f"{r['base']:.4f}->{r['meteor']:.4f} ({r['delta']:+.4f})", flush=True)

# Per-GCF summary
print(f"\nPer-GCF breakdown (top 20 by protein count):", flush=True)
for r in sorted(per_gcf_results, key=lambda x: -x["n_matched"])[:20]:
    dt = r["top1_meteor"] - r["top1_base"]
    df = r["fmax_meteor"] - r["fmax_base"]
    print(f"  {r['gcf']}: {r['n_matched']} prots, "
          f"top1 {r['top1_base']}->{r['top1_meteor']} ({dt:+d}), "
          f"Fmax {r['fmax_base']:.4f}->{r['fmax_meteor']:.4f} ({df:+.4f})", flush=True)

# Save
out_pkl = os.path.join(OUT_DIR, f"full_genome_eval_gbff_fmax_{BASELINE_NAME}_{VARIANT}.pkl")
pickle.dump({
    "baseline": f"{BASELINE_NAME}_{VARIANT}",
    "total_eval": total_eval,
    "total_gcf_eval": total_gcf_eval,
    "top1_base": top1_base_hits,
    "top1_meteor": top1_met_hits,
    "fmax_base_mean": float(np.mean(fmax_base_all)),
    "fmax_meteor_mean": float(np.mean(fmax_met_all)),
    "fmax_base_median": float(np.median(fmax_base_all)),
    "fmax_meteor_median": float(np.median(fmax_met_all)),
    "n_fmax_improved": n_fmax_up,
    "n_fmax_degraded": n_fmax_down,
    "n_corrections": len(corrections),
    "n_regressions": len(regressions),
    "corrections": corrections,
    "regressions": regressions,
    "score_changes": score_changes,
    "per_gcf": per_gcf_results,
    "fmax_base_all": fmax_base_all,
    "fmax_met_all": fmax_met_all,
}, open(out_pkl, "wb"))
print(f"\nSaved: {out_pkl}", flush=True)
print(f"Total time: {time.time()-t_start:.0f}s", flush=True)
