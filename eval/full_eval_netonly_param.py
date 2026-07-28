#!/usr/bin/env python3
"""Parameterized full-proteome net-only eval. Reads BASELINE and VARIANT from env."""
import os, re, json, pickle, sys, time
import numpy as np
import pandas as pd
BASELINE_NAME = os.environ["BASELINE"]
VARIANT = os.environ["VARIANT"]
W = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026"
V7_RUN = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run"
CACHE_DIR = f"{V7_RUN}/downstream_results/ncbi_ec_cache"
OUT_DIR = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/downstream_results"
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/eval")
from baseline_io import resolve_baseline_pkl
import sys as _s2; _s2.path.insert(0,"/ibex/user/niuk0a/funcarve/cobra/v6")
from src.v6utils import extract_pred as _XP, load_ec as _LE
_ANC=_LE("/ibex/user/niuk0a/funcarve/cobra/v6/data/all_ancestors.txt")
def fmax_fast(gt_indices, scores):
    n_gt = len(gt_indices)
    if n_gt == 0: return 0.0
    order = np.argsort(-scores)
    gt_mask = np.zeros(len(scores), dtype=bool)
    for i in gt_indices: gt_mask[i] = True
    sorted_mask = gt_mask[order]
    cum_tp = np.cumsum(sorted_mask).astype(float)
    n_pred = np.arange(1, len(scores) + 1, dtype=float)
    precision = cum_tp / n_pred
    recall = cum_tp / n_gt
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(np.max(f1))
meteor_dir = f"/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/{BASELINE_NAME}_{VARIANT}"
manifest = json.load(open(f"{W}/scripts/genome_collection_manifest.json"))
all_gcfs = sorted(g for g in manifest["assemblies"].keys() if g.startswith("GCF"))
print(f"[{BASELINE_NAME}_{VARIANT}] GCF assemblies: {len(all_gcfs)}", flush=True)
total_eval = 0; total_gcf_eval = 0
top1_base_hits = 0; top1_met_hits = 0
fmax_base_all = []; fmax_met_all = []
score_changes = []; per_gcf_results = []
t_start = time.time()
for gi, gcf in enumerate(all_gcfs):
    cache_file = os.path.join(CACHE_DIR, f"{gcf}_ec.json")
    if not os.path.exists(cache_file): continue
    met_pkl = os.path.join(meteor_dir, f"meteor_df_{gcf}.pkl")
    if not os.path.exists(met_pkl): continue
    base_pkl_path = resolve_baseline_pkl(BASELINE_NAME, VARIANT, gcf)
    if not base_pkl_path: continue
    ec_map = json.load(open(cache_file))
    ec_map = {k: set(v) for k, v in ec_map.items()}
    if not ec_map: continue
    try:
        base_df = _XP(base_pkl_path, _ANC)
        met_df = pickle.load(open(met_pkl, "rb"))
    except: continue
    if any(str(c).startswith("EC:") for c in base_df.columns[:5]):
        base_df.columns = [str(c).replace("EC:", "") for c in base_df.columns]
    b_cols = list(base_df.columns); m_cols = list(met_df.columns)
    b_col2idx = {c: i for i, c in enumerate(b_cols)}
    m_col2idx = {c: i for i, c in enumerate(m_cols)}
    n_matched = 0; gcf_t1_b, gcf_t1_m = 0, 0; gcf_fmax_b, gcf_fmax_m = [], []
    for pid, gt_ecs in ec_map.items():
        if pid not in base_df.index or pid not in met_df.index: continue
        n_matched += 1
        b_vals = base_df.loc[pid].values.astype(float)
        m_vals = met_df.loc[pid].values.astype(float)
        bt1_idx = int(np.argmax(b_vals)); mt1_idx = int(np.argmax(m_vals))
        bt1 = b_cols[bt1_idx]; mt1 = m_cols[mt1_idx]
        bh = bt1 in gt_ecs; mh = mt1 in gt_ecs
        if bh: top1_base_hits += 1; gcf_t1_b += 1
        if mh: top1_met_hits += 1; gcf_t1_m += 1
        b_gt_idx = {b_col2idx[ec] for ec in gt_ecs if ec in b_col2idx}
        m_gt_idx = {m_col2idx[ec] for ec in gt_ecs if ec in m_col2idx}
        fb = fmax_fast(b_gt_idx, b_vals); fm = fmax_fast(m_gt_idx, m_vals)
        fmax_base_all.append(fb); fmax_met_all.append(fm)
        gcf_fmax_b.append(fb); gcf_fmax_m.append(fm)
        for ec in gt_ecs:
            bsc = float(b_vals[b_col2idx[ec]]) if ec in b_col2idx else 0.0
            msc = float(m_vals[m_col2idx[ec]]) if ec in m_col2idx else 0.0
            score_changes.append({"uid": pid, "ec": ec, "gcf": gcf,
                "base": bsc, "meteor": msc, "delta": msc - bsc,
                "base_top1_hit": bh, "met_top1_hit": mh,
                "base_top1": bt1, "met_top1": mt1, "fmax_base": fb, "fmax_meteor": fm})
    if n_matched > 0:
        total_eval += n_matched; total_gcf_eval += 1
        per_gcf_results.append({"gcf": gcf, "n_ec": len(ec_map), "n_matched": n_matched,
            "top1_base": gcf_t1_b, "top1_meteor": gcf_t1_m,
            "fmax_base": float(np.mean(gcf_fmax_b)), "fmax_meteor": float(np.mean(gcf_fmax_m))})
    if (gi + 1) % 20 == 0 or gi == len(all_gcfs) - 1:
        elapsed = time.time() - t_start
        print(f"  [{gi+1}/{len(all_gcfs)}] {total_eval} prots, "
              f"top1 {top1_base_hits}/{top1_met_hits}, {elapsed:.0f}s", flush=True)
fmax_deltas = np.array(fmax_met_all) - np.array(fmax_base_all)
n_fmax_up = int(np.sum(fmax_deltas > 0.001))
n_fmax_down = int(np.sum(fmax_deltas < -0.001))
deltas = [s["delta"] for s in score_changes]
seen = set(); corrections, regressions = [], []
for s in score_changes:
    key = (s["uid"], s["gcf"])
    if key in seen: continue
    seen.add(key)
    if not s["base_top1_hit"] and s["met_top1_hit"]: corrections.append(s)
    elif s["base_top1_hit"] and not s["met_top1_hit"]: regressions.append(s)
out_pkl = os.path.join(OUT_DIR, f"full_genome_eval_gbff_fmax_{BASELINE_NAME}_{VARIANT}.pkl")
pickle.dump({"baseline": f"{BASELINE_NAME}_{VARIANT}", "total_eval": total_eval,
    "total_gcf_eval": total_gcf_eval, "top1_base": top1_base_hits,
    "top1_meteor": top1_met_hits,
    "fmax_base_mean": float(np.mean(fmax_base_all)),
    "fmax_meteor_mean": float(np.mean(fmax_met_all)),
    "fmax_base_median": float(np.median(fmax_base_all)),
    "fmax_meteor_median": float(np.median(fmax_met_all)),
    "n_fmax_improved": n_fmax_up, "n_fmax_degraded": n_fmax_down,
    "n_corrections": len(corrections), "n_regressions": len(regressions),
    "corrections": corrections, "regressions": regressions,
    "score_changes": score_changes, "per_gcf": per_gcf_results,
    "fmax_base_all": fmax_base_all, "fmax_met_all": fmax_met_all,
}, open(out_pkl, "wb"))
elapsed = time.time() - t_start
print(f"[{BASELINE_NAME}_{VARIANT}] DONE: {total_eval} prots, "
      f"top1 {top1_base_hits}/{top1_met_hits} "
      f"corr+{len(corrections)}/regr-{len(regressions)} "
      f"Fmax {np.mean(fmax_base_all):.4f}/{np.mean(fmax_met_all):.4f} "
      f"saved {out_pkl} [{elapsed:.0f}s]", flush=True)
