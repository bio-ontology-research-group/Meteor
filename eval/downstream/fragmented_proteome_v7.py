#!/usr/bin/env python3
"""Fragmented-proteome recovery test (replaces the stale L3 empirical claim).

For each of the 22 Price-149 genomes, simulate an incomplete/fragmented
proteome: zero out the baseline evidence for ALL Price-149 ground-truth
proteins in that genome (as if those genes were never called), plus an
additional rho fraction of randomly-chosen OTHER proteins in the same
genome. Re-run the full v7 pipeline (noisy-OR -> logodds cost -> hard-
biomass MILP -> posterior) on this modified matrix (an in-memory copy;
the original prediction files on disk are never touched) and check, for
each deleted ground-truth protein:
  (1) network-level recovery: is its true EC's mapped reaction MILP-active
      (in active_ecs), inferred purely from the REST of the genome's
      evidence and network connectivity?
  (2) protein-level recovery: does ITS OWN posterior score for the true EC
      rise above zero? (Expected: no, by construction of the multiplicative
      posterior E_new = E + alpha*E*(1-E), which cannot lift an exact zero.)

DPZ baseline only, matching the paper's other ablation/negative-control
experiments. rho in {0.0, 0.10, 0.20, 0.30}, 5 seeds per rho (seed=0 is
reserved for rho=0.0, which is deterministic).
"""
import sys, os, pickle, argparse, time
import numpy as np
import pandas as pd

V6 = "/ibex/user/niuk0a/funcarve/cobra/v6"
F = "/ibex/scratch/projects/c2014/kexin/funcarve"
ECONTO = "/ibex/user/niuk0a/funcarve/econto"
sys.path.insert(0, V6); os.chdir(V6)
sys.path.insert(0, "/ibex/user/niuk0a/meteor_v7/src")
from src.v6utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                         apply_media, find_excluded_reactions, aggregate_confidence,
                         compute_costs, build_candidate_mask, build_rxn_ec_mask,
                         extract_pred, load_refmapping, load_ec, _detect_solver,
                         posterior_calibrated)
from meteor.milp_hard import biomass_feasible_skeleton, build_milp_hard


def _dpz_price_path(gca):
    import glob as _g
    gns = gca.rsplit(".", 1)[0]
    exact = f"{F}/dpec2_result/result_price/{gns}_DeepECv2_t5.pkl"
    if os.path.exists(exact):
        return exact
    num = gca.split("_")[1].split(".")[0]
    hits = _g.glob(f"{F}/dpec2_result/result_price/GC?_{num}_DeepECv2_t5.pkl")
    return hits[0] if hits else exact


ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True)
ap.add_argument("--rho", type=float, required=True, choices=[0.0, 0.10, 0.20, 0.30])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--outdir", default=f"{F}/meteor_v7_run/ablation_out/frag_proteome")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
t0 = time.time()
outfile = f"{a.outdir}/{a.gca}_rho{int(a.rho*100)}_seed{a.seed}.pkl"
if os.path.exists(outfile):
    print(f"skip {outfile}", flush=True); sys.exit(0)

# --- ground truth + protein-id mapping (same as eval_price149_v7.py) ---
price = pd.read_csv(f"{ECONTO}/data/test/price.csv", sep="\t")
gt_ecs_all = {r["Entry"]: {e.strip() for e in str(r["EC number"]).split(";")
                           if e.strip().count(".") == 3 and "-" not in e.strip()}
              for _, r in price.iterrows()}
gt_ecs_all = {k: v for k, v in gt_ecs_all.items() if v and len(v) == 1}  # unambiguous single-EC only
meta = pd.read_csv(f"{ECONTO}/data/processed/geno/price_proteomes/Price_genome_metainfo.csv")
meta_map = {r["input_id"]: (r["protein_id"], r.get("nucleotide_id"), r["assembly"]) for _, r in meta.iterrows()}

# target proteins in THIS genome
targets = []  # list of (entry, protein_id_in_pred, gt_ec)
for entry, gt_set in gt_ecs_all.items():
    if entry not in meta_map:
        continue
    pid, nid, assembly = meta_map[entry]
    if assembly != a.gca:
        continue
    targets.append((entry, pid, next(iter(gt_set))))
if not targets:
    print(f"[frag] ABORT {a.gca}: no Price-149 target proteins in this genome", flush=True)
    sys.exit(3)
print(f"[frag] {a.gca}: {len(targets)} target proteins", flush=True)

pred_path = _dpz_price_path(a.gca)
if not os.path.exists(pred_path):
    print(f"[frag] ABORT {a.gca}: no DPZ pred file", flush=True); sys.exit(3)

seedr2ec, _ = load_refmapping(f"{V6}/data"); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, allmet = load_universal()
anc = load_ec(f"{V6}/data/all_ancestors.txt")
pred = extract_pred(pred_path, anc)

# resolve target protein_ids to actual index labels in pred (tolerate ID variants)
def resolve_idx(df, pid):
    if pid in df.index:
        return pid
    base = str(pid).rsplit(".", 1)[0]
    hits = [i for i in df.index if i == base or str(i).startswith(base + ".")]
    return hits[0] if hits else None

resolved_targets = []
for entry, pid, gt_ec in targets:
    ridx = resolve_idx(pred, pid)
    if ridx is not None and gt_ec in pred.columns:
        resolved_targets.append((entry, ridx, gt_ec))
if not resolved_targets:
    print(f"[frag] ABORT {a.gca}: no targets resolved into pred matrix", flush=True)
    sys.exit(3)
print(f"[frag] {a.gca}: {len(resolved_targets)}/{len(targets)} targets resolved", flush=True)

# --- build the fragmented (in-memory) prediction matrix ---
pred_frag = pred.copy()
target_rows = {ridx for _, ridx, _ in resolved_targets}
for ridx in target_rows:
    pred_frag.loc[ridx] = 0.0

rng = np.random.default_rng(a.seed)
other_rows = [i for i in pred.index if i not in target_rows]
n_extra = int(round(a.rho * len(other_rows)))
extra_deleted = set(rng.choice(other_rows, size=n_extra, replace=False).tolist()) if n_extra else set()
for ridx in extra_deleted:
    pred_frag.loc[ridx] = 0.0
print(f"[frag] {a.gca} rho={a.rho} seed={a.seed}: deleted {len(target_rows)} target + "
      f"{len(extra_deleted)} extra rows ({len(target_rows)+len(extra_deleted)}/{pred.shape[0]} total)", flush=True)

# --- standard v7 pipeline on the fragmented matrix ---
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lb_t, ub_t = load_tight_bounds(f"{V6}/data/tight_bounds_v6_neg.pkl")
if lb_t is not None: lb = np.maximum(lb, lb_t); ub = np.minimum(ub, ub_t)
obj_idx = allrxns.index("biomass_GmNeg")
excludes = find_excluded_reactions(S, lb, ub, allrxns, "biomass_GmNeg")
lb, ub, mm, media_rxns = apply_media(["default"], allrxns, lb, ub)
solver, sname = _detect_solver(threads=4, time_limit=600)

feas, skel, bm_max = biomass_feasible_skeleton(S, lb, ub, obj_idx, 0.1, solver=solver)
w = aggregate_confidence(pred_frag, mask, allrxns, method="noisy_or")
w = np.clip(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 1e-6, 1.0 - 1e-6)
c = compute_costs(w, mode="logodds")["c"]
cand = build_candidate_mask(w, allrxns, excludes, media_rxns, essential_skeleton=skel, w_min=0.01)
m, y, vp, vn, _ = build_milp_hard(S, lb, ub, c, obj_idx, excludes, media_rxns, cand, 0.1, 2.5, 1e-4)
m.solve(solver)
yv = np.array([y[j].value() or 0 for j in range(len(y))])
v_vals = np.array([(vp[j].value() or 0) - (vn[j].value() or 0) for j in range(len(y))])
bm = v_vals[obj_idx]
n_active = int((yv > 0.5).sum())
print(f"[frag] {a.gca} rho={a.rho} seed={a.seed}: biomass={bm:.4f} n_active={n_active}", flush=True)

opt_df, active_ecs, muted_ecs = posterior_calibrated(pred_frag, yv, seedr2ec, allrxns, anc, beta=a.beta)

# --- per-target recovery check ---
rows = []
for entry, ridx, gt_ec in resolved_targets:
    network_recovered = gt_ec in active_ecs
    protein_score_new = float(opt_df.loc[ridx, gt_ec]) if gt_ec in opt_df.columns else 0.0
    protein_score_orig = float(pred.loc[ridx, gt_ec]) if gt_ec in pred.columns else 0.0
    rows.append(dict(entry=entry, gca=a.gca, rho=a.rho, seed=a.seed, gt_ec=gt_ec,
                      network_recovered=network_recovered,
                      protein_score_orig=protein_score_orig,
                      protein_score_new=protein_score_new,
                      protein_recovered=protein_score_new > 0.01))

df = pd.DataFrame(rows)
df.to_pickle(outfile)
print(f"[frag] {a.gca} rho={a.rho} seed={a.seed}: network_recovered={df.network_recovered.sum()}/{len(df)} "
      f"protein_recovered={df.protein_recovered.sum()}/{len(df)}  {time.time()-t0:.0f}s", flush=True)
