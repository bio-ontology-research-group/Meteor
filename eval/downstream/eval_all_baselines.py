#!/usr/bin/env python3
"""Evaluate METEOR on ALL NCBI EC-annotated proteins, ALL 12 baseline×cutoff configs.
Reports Top-1, Fmax, micro-F1 at τ=0.5 for each config.
Uses cached GBFF EC annotations (already downloaded).
"""
import os, re, json, pickle, sys, time
import numpy as np
import pandas as pd

W = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026"
V7_RUN = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run"
CACHE_DIR = f"{V7_RUN}/downstream_results/ncbi_ec_cache"
OUT_DIR = f"{V7_RUN}/downstream_results"

sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7/eval/downstream")
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX

manifest = json.load(open(f"{W}/scripts/genome_collection_manifest.json"))
all_gcfs = sorted(g for g in manifest["assemblies"].keys() if g.startswith("GCF"))


def fmax_fast(gt_indices, scores):
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


CONFIGS = [
    ("clean", "vanilla"), ("clean", "filt30"), ("clean", "filt50"), ("clean", "filt70"),
    ("dpz", "vanilla"), ("dpz", "filt30"), ("dpz", "filt50"), ("dpz", "filt70"),
    ("enzbert", "vanilla"), ("enzbert", "filt30"), ("enzbert", "filt50"), ("enzbert", "filt70"),
]

all_results = []

for baseline, variant in CONFIGS:
    meteor_dir = f"{V7_RUN}/meteor_out/{baseline}_{variant}"
    config_name = f"{baseline}_{variant}"
    print(f"\n{'='*60}", flush=True)
    print(f"Config: {config_name}", flush=True)
    t0 = time.time()

    total_eval = 0
    top1_b, top1_m = 0, 0
    fmax_b_all, fmax_m_all = [], []
    # micro-F1 at τ=0.5
    b_tp, b_fp, b_fn = 0, 0, 0
    m_tp, m_fp, m_fn = 0, 0, 0
    n_boosted, n_dampened = 0, 0
    n_corrections, n_regressions = 0, 0
    n_gcf = 0

    for gcf in all_gcfs:
        cache_file = os.path.join(CACHE_DIR, f"{gcf}_ec.json")
        if not os.path.exists(cache_file):
            continue
        met_pkl = os.path.join(meteor_dir, f"meteor_df_{gcf}.pkl")
        if not os.path.exists(met_pkl):
            continue
        base_pkl_path = resolve_baseline_pkl(baseline, variant, gcf)
        if not base_pkl_path:
            continue

        ec_map = json.load(open(cache_file))
        ec_map = {k: set(v) for k, v in ec_map.items()}
        if not ec_map:
            continue

        try:
            base_df = pickle.load(open(base_pkl_path, "rb"))
            met_df = pickle.load(open(met_pkl, "rb"))
        except:
            continue

        if any(str(c).startswith("EC:") for c in base_df.columns[:5]):
            base_df.columns = [str(c).replace("EC:", "") for c in base_df.columns]

        b_cols = list(base_df.columns)
        m_cols = list(met_df.columns)
        b_col2idx = {c: i for i, c in enumerate(b_cols)}
        m_col2idx = {c: i for i, c in enumerate(m_cols)}
        b_cols_arr = np.array(b_cols)
        m_cols_arr = np.array(m_cols)

        gcf_matched = False
        for pid, gt_ecs in ec_map.items():
            if pid not in base_df.index or pid not in met_df.index:
                continue
            if not gcf_matched:
                gcf_matched = True
                n_gcf += 1
            total_eval += 1

            b_vals = base_df.loc[pid].values.astype(float)
            m_vals = met_df.loc[pid].values.astype(float)

            # Top-1
            bt1 = b_cols[int(np.argmax(b_vals))]
            mt1 = m_cols[int(np.argmax(m_vals))]
            bh = bt1 in gt_ecs
            mh = mt1 in gt_ecs
            if bh: top1_b += 1
            if mh: top1_m += 1
            if not bh and mh: n_corrections += 1
            if bh and not mh: n_regressions += 1

            # Fmax
            b_gt_idx = {b_col2idx[ec] for ec in gt_ecs if ec in b_col2idx}
            m_gt_idx = {m_col2idx[ec] for ec in gt_ecs if ec in m_col2idx}
            fmax_b_all.append(fmax_fast(b_gt_idx, b_vals))
            fmax_m_all.append(fmax_fast(m_gt_idx, m_vals))

            # micro-F1 at τ=0.5
            b_pred = set(b_cols_arr[b_vals >= 0.5])
            m_pred = set(m_cols_arr[m_vals >= 0.5])
            b_tp += len(gt_ecs & b_pred)
            b_fp += len(b_pred - gt_ecs)
            b_fn += len(gt_ecs - b_pred)
            m_tp += len(gt_ecs & m_pred)
            m_fp += len(m_pred - gt_ecs)
            m_fn += len(gt_ecs - m_pred)

            # Score changes
            for ec in gt_ecs:
                bsc = float(b_vals[b_col2idx[ec]]) if ec in b_col2idx else 0.0
                msc = float(m_vals[m_col2idx[ec]]) if ec in m_col2idx else 0.0
                d = msc - bsc
                if d > 0.001: n_boosted += 1
                elif d < -0.001: n_dampened += 1

    if total_eval == 0:
        print(f"  No proteins evaluated", flush=True)
        continue

    # Compute metrics
    fmax_b_mean = np.mean(fmax_b_all)
    fmax_m_mean = np.mean(fmax_m_all)
    b_prec = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0
    b_rec = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0
    b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec) if (b_prec + b_rec) > 0 else 0
    m_prec = m_tp / (m_tp + m_fp) if (m_tp + m_fp) > 0 else 0
    m_rec = m_tp / (m_tp + m_fn) if (m_tp + m_fn) > 0 else 0
    m_f1 = 2 * m_prec * m_rec / (m_prec + m_rec) if (m_prec + m_rec) > 0 else 0

    result = {
        "config": config_name, "baseline": baseline, "variant": variant,
        "n_gcf": n_gcf, "n_eval": total_eval,
        "top1_b": top1_b, "top1_m": top1_m,
        "top1_b_pct": 100 * top1_b / total_eval,
        "top1_m_pct": 100 * top1_m / total_eval,
        "fmax_b": fmax_b_mean, "fmax_m": fmax_m_mean,
        "micro_f1_b": b_f1, "micro_f1_m": m_f1,
        "prec_b": b_prec, "prec_m": m_prec,
        "rec_b": b_rec, "rec_m": m_rec,
        "corrections": n_corrections, "regressions": n_regressions,
        "boosted": n_boosted, "dampened": n_dampened,
    }
    all_results.append(result)

    print(f"  GCFs: {n_gcf}, proteins: {total_eval}", flush=True)
    print(f"  Top-1: {top1_b}/{total_eval} ({result['top1_b_pct']:.1f}%) -> "
          f"{top1_m}/{total_eval} ({result['top1_m_pct']:.1f}%)", flush=True)
    print(f"  Fmax:  {fmax_b_mean:.4f} -> {fmax_m_mean:.4f} ({fmax_m_mean-fmax_b_mean:+.4f})", flush=True)
    print(f"  micro-F1: {b_f1:.4f} -> {m_f1:.4f} ({m_f1-b_f1:+.4f})", flush=True)
    print(f"  Precision: {b_prec:.4f} -> {m_prec:.4f}", flush=True)
    print(f"  Corrections: {n_corrections}, Regressions: {n_regressions}", flush=True)
    print(f"  Boosted: {n_boosted}, Dampened: {n_dampened}", flush=True)
    print(f"  Time: {time.time()-t0:.0f}s", flush=True)

# ── Summary table ──
print(f"\n{'='*70}", flush=True)
print(f"SUMMARY TABLE", flush=True)
print(f"{'='*70}", flush=True)
print(f"{'Config':<20} {'n':>6} {'Top1-B':>7} {'Top1-M':>7} {'dTop1':>6} "
      f"{'Fmax-B':>7} {'Fmax-M':>7} {'dFmax':>7} "
      f"{'uF1-B':>7} {'uF1-M':>7} {'duF1':>7} "
      f"{'Corr':>5} {'Regr':>5}", flush=True)
print("-" * 120, flush=True)
for r in all_results:
    dt1 = r["top1_m_pct"] - r["top1_b_pct"]
    dfm = r["fmax_m"] - r["fmax_b"]
    df1 = r["micro_f1_m"] - r["micro_f1_b"]
    print(f"{r['config']:<20} {r['n_eval']:>6} "
          f"{r['top1_b_pct']:>6.1f}% {r['top1_m_pct']:>6.1f}% {dt1:>+5.1f}% "
          f"{r['fmax_b']:>7.4f} {r['fmax_m']:>7.4f} {dfm:>+7.4f} "
          f"{r['micro_f1_b']:>7.4f} {r['micro_f1_m']:>7.4f} {df1:>+7.4f} "
          f"{r['corrections']:>5} {r['regressions']:>5}", flush=True)

# Save
out_pkl = os.path.join(OUT_DIR, "full_genome_eval_all_baselines.pkl")
pickle.dump(all_results, open(out_pkl, "wb"))
print(f"\nSaved: {out_pkl}", flush=True)
