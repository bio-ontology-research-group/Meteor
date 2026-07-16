#!/usr/bin/env python3
"""Extract biologically interesting case studies from METEOR full-proteome eval.

For each baseline config, identifies:
1. Top corrected proteins (baseline wrong → METEOR right) with EC details
2. Pathway-level changes (KEGG pathway gains/losses)
3. Genomes with strongest METEOR effects

Outputs a structured report for paper inclusion.
"""
import os, json, pickle, sys, time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

W = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026"
V7_RUN = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run"
CACHE_DIR = f"{V7_RUN}/downstream_results/ncbi_ec_cache"

sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7/eval/downstream")
from baseline_io import resolve_baseline_pkl

manifest = json.load(open(f"{W}/scripts/genome_collection_manifest.json"))
all_gcfs = sorted(g for g in manifest["assemblies"].keys() if g.startswith("GCF"))

BASELINE = "dpz"
VARIANT = "vanilla"
meteor_dir = f"{V7_RUN}/meteor_out/{BASELINE}_{VARIANT}"

corrections = []   # (gcf, protein_id, true_ec, baseline_top1, meteor_top1, base_score_true, met_score_true)
regressions = []
per_genome_stats = []  # (gcf, n_eval, n_correct_b, n_correct_m, top1_delta)
ec_corrections = Counter()  # EC → count of corrections involving this EC
ec_regressions = Counter()

t0 = time.time()
for gcf in all_gcfs:
    cache_file = os.path.join(CACHE_DIR, f"{gcf}_ec.json")
    if not os.path.exists(cache_file):
        continue
    met_pkl = os.path.join(meteor_dir, f"meteor_df_{gcf}.pkl")
    if not os.path.exists(met_pkl):
        continue
    base_pkl_path = resolve_baseline_pkl(BASELINE, VARIANT, gcf)
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

    gcf_n = 0
    gcf_b_correct = 0
    gcf_m_correct = 0

    for pid, gt_ecs in ec_map.items():
        if pid not in base_df.index or pid not in met_df.index:
            continue
        gcf_n += 1

        b_vals = base_df.loc[pid].values.astype(float)
        m_vals = met_df.loc[pid].values.astype(float)

        bt1 = b_cols[int(np.argmax(b_vals))]
        mt1 = m_cols[int(np.argmax(m_vals))]
        bh = bt1 in gt_ecs
        mh = mt1 in gt_ecs
        if bh: gcf_b_correct += 1
        if mh: gcf_m_correct += 1

        if not bh and mh:
            # Correction! Get score details
            best_gt_ec = max(gt_ecs, key=lambda e: float(b_vals[b_col2idx[e]]) if e in b_col2idx else 0.0)
            b_score_gt = float(b_vals[b_col2idx[best_gt_ec]]) if best_gt_ec in b_col2idx else 0.0
            m_score_gt = float(m_vals[m_col2idx[best_gt_ec]]) if best_gt_ec in m_col2idx else 0.0
            b_score_top = float(np.max(b_vals))
            m_score_top = float(np.max(m_vals))
            corrections.append({
                "gcf": gcf, "pid": pid, "gt_ecs": list(gt_ecs),
                "baseline_top1": bt1, "meteor_top1": mt1,
                "best_gt_ec": best_gt_ec,
                "base_score_gt": b_score_gt, "met_score_gt": m_score_gt,
                "base_score_top": b_score_top, "met_score_top": m_score_top,
                "score_delta_gt": m_score_gt - b_score_gt,
            })
            for ec in gt_ecs:
                ec_corrections[ec] += 1

        if bh and not mh:
            best_gt_ec = max(gt_ecs, key=lambda e: float(b_vals[b_col2idx[e]]) if e in b_col2idx else 0.0)
            b_score_gt = float(b_vals[b_col2idx[best_gt_ec]]) if best_gt_ec in b_col2idx else 0.0
            m_score_gt = float(m_vals[m_col2idx[best_gt_ec]]) if best_gt_ec in m_col2idx else 0.0
            regressions.append({
                "gcf": gcf, "pid": pid, "gt_ecs": list(gt_ecs),
                "baseline_top1": bt1, "meteor_top1": mt1,
                "best_gt_ec": best_gt_ec,
                "base_score_gt": b_score_gt, "met_score_gt": m_score_gt,
            })
            for ec in gt_ecs:
                ec_regressions[ec] += 1

    if gcf_n > 0:
        per_genome_stats.append({
            "gcf": gcf, "n_eval": gcf_n,
            "top1_b": gcf_b_correct, "top1_m": gcf_m_correct,
            "delta": gcf_m_correct - gcf_b_correct,
        })

print(f"Processed in {time.time()-t0:.0f}s", flush=True)
print(f"\nTotal corrections: {len(corrections)}", flush=True)
print(f"Total regressions: {len(regressions)}", flush=True)

# ── Top corrected ECs ──
print(f"\n{'='*60}", flush=True)
print("TOP 20 CORRECTED EC NUMBERS (wrong→right)", flush=True)
print(f"{'='*60}", flush=True)
for ec, cnt in ec_corrections.most_common(20):
    print(f"  {ec:<15} {cnt:>4} corrections", flush=True)

# ── Top regressed ECs ──
print(f"\n{'='*60}", flush=True)
print("TOP 20 REGRESSED EC NUMBERS (right→wrong)", flush=True)
print(f"{'='*60}", flush=True)
for ec, cnt in ec_regressions.most_common(20):
    print(f"  {ec:<15} {cnt:>4} regressions", flush=True)

# ── Genomes with most corrections ──
print(f"\n{'='*60}", flush=True)
print("TOP 15 GENOMES BY NET CORRECTION COUNT", flush=True)
print(f"{'='*60}", flush=True)
per_genome_stats.sort(key=lambda x: -x["delta"])
for g in per_genome_stats[:15]:
    pct_b = 100 * g["top1_b"] / g["n_eval"] if g["n_eval"] > 0 else 0
    pct_m = 100 * g["top1_m"] / g["n_eval"] if g["n_eval"] > 0 else 0
    print(f"  {g['gcf']:<25} n={g['n_eval']:>5} "
          f"top1: {pct_b:.1f}%→{pct_m:.1f}% (Δ={g['delta']:>+3})", flush=True)

# ── Example corrections with score details ──
print(f"\n{'='*60}", flush=True)
print("INTERESTING CORRECTIONS (large score boost on GT EC)", flush=True)
print(f"{'='*60}", flush=True)
corrections.sort(key=lambda x: -x["score_delta_gt"])
for c in corrections[:30]:
    print(f"  {c['gcf']} | {c['pid']}", flush=True)
    print(f"    GT: {c['gt_ecs']} | best_gt: {c['best_gt_ec']}", flush=True)
    print(f"    Baseline top1: {c['baseline_top1']} (score={c['base_score_top']:.4f})", flush=True)
    print(f"    METEOR  top1: {c['meteor_top1']}  (score={c['met_score_top']:.4f})", flush=True)
    print(f"    GT EC score: {c['base_score_gt']:.4f} → {c['met_score_gt']:.4f} "
          f"(Δ={c['score_delta_gt']:+.4f})", flush=True)

# ── Example regressions ──
print(f"\n{'='*60}", flush=True)
print("TOP REGRESSIONS (for honest reporting)", flush=True)
print(f"{'='*60}", flush=True)
for r in regressions[:15]:
    print(f"  {r['gcf']} | {r['pid']}", flush=True)
    print(f"    GT: {r['gt_ecs']} | Baseline: {r['baseline_top1']} → METEOR: {r['meteor_top1']}", flush=True)
    print(f"    GT EC score: {r['base_score_gt']:.4f} → {r['met_score_gt']:.4f}", flush=True)

# Save
out_pkl = os.path.join(V7_RUN, "downstream_results", "case_studies_dpz_vanilla.pkl")
pickle.dump({
    "corrections": corrections,
    "regressions": regressions,
    "per_genome_stats": per_genome_stats,
    "ec_corrections": dict(ec_corrections),
    "ec_regressions": dict(ec_regressions),
}, open(out_pkl, "wb"))
print(f"\nSaved: {out_pkl}", flush=True)
