#!/usr/bin/env python3
"""v7 METEOR-output emitter for the Price-149 benchmark (22 genomes, 6 baselines).
Same hard-biomass MILP + v6 posterior as emit_meteor_out.py; only the baseline
prediction path differs (per-baseline GCA-named price preds, scattered dirs).
gram = negative for all Price genomes. Saves meteor_sol/df/preds per (baseline,gca)
to meteor_out_price/{baseline}/."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys, os, pickle, argparse, time, glob
import numpy as np
import pandas as pd

F = "/ibex/scratch/projects/c2014/kexin/funcarve"

from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                         apply_media, find_excluded_reactions, aggregate_confidence,
                         compute_costs, build_candidate_mask, build_rxn_ec_mask,
                         extract_pred, load_refmapping, load_ec, _detect_solver,
                         posterior_calibrated)
from meteor_v8.milp_hard import biomass_feasible_skeleton, build_milp_hard
from meteor_v8.repair import verify_and_repair


def _dpz_price_path(gca):
    import glob as _g, os as _os
    gns = gca.rsplit(".", 1)[0]
    exact = f"{F}/dpec2_result/result_price/{gns}_DeepECv2_t5.pkl"
    if _os.path.exists(exact):
        return exact
    num = gca.split("_")[1].split(".")[0]
    hits = _g.glob(f"{F}/dpec2_result/result_price/GC?_{num}_DeepECv2_t5.pkl")
    return hits[0] if hits else exact


def price_pred_path(baseline, gca):
    """gca like GCA_000006965.1 ; DPZ uses the accession WITHOUT the .N suffix."""
    gca_ns = gca.rsplit(".", 1)[0]
    m = {
        "clean":   f"{F}/paperA_2026/baseline_preds/price22_CLEAN_clean/{gca}/{gca}_CLEAN_confidence.pkl",
        "dpz":     _dpz_price_path(gca),
        "enzbert": f"{F}/tfpc/resultprice_newg/{gca}_enzbert_predictions.pkl",
        "graphec": f"{F}/graphec_price_new/{gca}_GraphEC.pkl",
        "mapred":  f"{F}/mapred_price_new/{gca}_MAPred.pkl",
        "topec":   f"{F}/topec_price_new/{gca}_TopEC.pkl",
    }
    p = m[baseline]
    if os.path.exists(p):
        return p
    # tolerate suffix variants
    for cand in (p, p.replace(gca, gca_ns), p.replace(gca_ns, gca)):
        if os.path.exists(cand):
            return cand
    return p


ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True)
ap.add_argument("--baseline", choices=["clean", "dpz", "enzbert", "graphec", "mapred", "topec"], required=True)
ap.add_argument("--outroot", default=f"{F}/meteor_v8_evw_p2mu3_run/meteor_out_price")
ap.add_argument("--beta", type=float, default=1.0)
a = ap.parse_args()
t0 = time.time()
MOUT = os.path.join(a.outroot, a.baseline)
os.makedirs(MOUT, exist_ok=True)
if os.path.exists(f"{MOUT}/meteor_preds_{a.gca}.pkl"):
    print(f"skip {a.gca}", flush=True); sys.exit(0)
biomass_id = "biomass_GmNeg"          # all 22 Price genomes are gram-negative

pred_path = price_pred_path(a.baseline, a.gca)
if not os.path.exists(pred_path):
    print(f"[ERR] no pred: {pred_path}", flush=True); sys.exit(3)

seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, allmet = load_universal()
anc = load_ec(data_path('all_ancestors.txt'))
pred = extract_pred(pred_path, anc)
_P = pred.values.astype(float).copy(); _ne=_P.shape[1]
if 5 < _ne:
    _dr=np.argpartition(_P,_ne-5,axis=1)[:,:_ne-5]; np.put_along_axis(_P,_dr,0.0,axis=1)
    pred = pd.DataFrame(_P, index=pred.index, columns=pred.columns)
print(f"[emit_price] {a.baseline} {a.gca} pred={pred.shape} src={os.path.basename(pred_path)}", flush=True)

mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lb_t, ub_t = load_tight_bounds(data_path('tight_bounds_v6_neg.pkl'))
if lb_t is not None: lb = np.maximum(lb, lb_t); ub = np.minimum(ub, ub_t)
obj_idx = allrxns.index(biomass_id)
excludes = find_excluded_reactions(S, lb, ub, allrxns, biomass_id)
lb, ub, mm, media_rxns = apply_media(["default"], allrxns, lb, ub)
solver, sname = _detect_solver(threads=4, time_limit=600)

feas, skel, bm_max = biomass_feasible_skeleton(S, lb, ub, obj_idx, 0.1, solver=solver)
w = aggregate_confidence(pred, mask, allrxns, method="noisy_or")
w = np.clip(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 1e-6, 1.0 - 1e-6)
c = compute_costs(w, mode="logodds")["c"]
cand = build_candidate_mask(w, allrxns, excludes, media_rxns, essential_skeleton=skel, w_min=0.01)
c = np.asarray(c, dtype=float) + 3.0*np.power(np.clip(1.0-w,0.0,1.0),2.0)*np.asarray(cand,dtype=float)  # evw p=2 mu=3
m, y, vp, vn, _ = build_milp_hard(S, lb, ub, c, obj_idx, excludes, media_rxns, cand, 0.1, 2.5, 1e-4)
m.solve(solver)
yv = np.array([y[j].value() or 0 for j in range(len(y))])
v_vals = np.array([(vp[j].value() or 0) - (vn[j].value() or 0) for j in range(len(y))])
yv, n_active, n_repaired, _mb_core = verify_and_repair(S, lb, ub, obj_idx, yv, v_vals, cand)
print(f"[emit_price] biomass={v_vals[obj_idx]:.4f} n_active={n_active} "
      f"mb_core={_mb_core:.3f} repaired={n_repaired}", flush=True)

pickle.dump({"y_vals": yv, "v_vals": v_vals, "biomass_flux": float(v_vals[obj_idx]),
             "n_active": n_active, "status": "MILP_hard",
             "n_repaired": int(n_repaired), "maxbio_core": float(_mb_core)}, open(f"{MOUT}/meteor_sol_{a.gca}.pkl", "wb"))
opt_df, active_ecs, muted_ecs = posterior_calibrated(pred, yv, seedr2ec, allrxns, anc, beta=a.beta)
opt_df.to_pickle(f"{MOUT}/meteor_df_{a.gca}.pkl")
pickle.dump({"active_ecs": active_ecs, "muted_ecs": muted_ecs}, open(f"{MOUT}/meteor_preds_{a.gca}.pkl", "wb"))
print(f"[emit_price] saved: {len(active_ecs)} active ECs, {time.time()-t0:.0f}s", flush=True)
