#!/usr/bin/env python3
"""Lightweight v7 METEOR-output emitter (no MEMOTE build): runs the hard-biomass
MILP + v6 posterior and saves v6_sol / v6_df / v6_preds(active_ecs) per genome,
in the format the downstream evals (pathway/bgc/capability/per-protein) read.
Same MILP as build_grow_memote.py; just skips the model build + MEMOTE."""
import sys, os, pickle, json, argparse, time
import numpy as np
import pandas as pd

V6 = "/ibex/user/niuk0a/funcarve/cobra/v6"
PA = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026"
sys.path.insert(0, V6); os.chdir(V6)
sys.path.insert(0, "/ibex/user/niuk0a/meteor_v7/src")
from src.v6utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                         apply_media, find_excluded_reactions, aggregate_confidence,
                         compute_costs, build_candidate_mask, build_rxn_ec_mask,
                         extract_pred, load_refmapping, load_ec, _detect_solver,
                         posterior_calibrated)
from meteor.milp_hard import biomass_feasible_skeleton, build_milp_hard

ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True)
ap.add_argument("--gram", choices=["negative", "positive"], required=True)
ap.add_argument("--baseline", choices=["clean", "dpz", "enzbert"], default="dpz")
ap.add_argument("--variant", choices=["vanilla", "filt30", "filt50", "filt70"], default="vanilla")
ap.add_argument("--outroot", default="/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/meteor_out")
ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--min_frac", type=float, default=0.9)
a = ap.parse_args()
t0 = time.time()
MOUT = os.path.join(a.outroot, f"{a.baseline}_{a.variant}")
os.makedirs(MOUT, exist_ok=True)
if os.path.exists(f"{MOUT}/meteor_preds_{a.gca}.pkl"):
    print(f"skip {a.gca}", flush=True); sys.exit(0)
biomass_id = "biomass_GmPos" if a.gram == "positive" else "biomass_GmNeg"

seedr2ec, _ = load_refmapping(f"{V6}/data"); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, allmet = load_universal()
anc = load_ec(f"{V6}/data/all_ancestors.txt")

sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7/eval/downstream")
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
_SUF = BASELINE_SUFFIX[a.baseline]
pred_path = resolve_baseline_pkl(a.baseline, a.variant, a.gca, _SUF)
if not pred_path:
    print(f"[GUARD] no pred {a.gca}", flush=True); sys.exit(3)
pred = extract_pred(pred_path, anc)
_ep = resolve_baseline_pkl("enzbert", "vanilla", a.gca, "enzbert")
_ref = pd.read_pickle(_ep).shape[0] if _ep else None
if _ref and pred.shape[0] < a.min_frac * _ref:
    print(f"[GUARD] incomplete {a.gca}: {pred.shape[0]}<{a.min_frac}*{_ref}", flush=True); sys.exit(3)

mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lb_t, ub_t = load_tight_bounds(f"{V6}/data/tight_bounds_v6_{a.gram[:3]}.pkl")
if lb_t is not None: lb = np.maximum(lb, lb_t); ub = np.minimum(ub, ub_t)
obj_idx = allrxns.index(biomass_id)
excludes = find_excluded_reactions(S, lb, ub, allrxns, biomass_id)
lb, ub, media_mask, media_rxns = apply_media(["default"], allrxns, lb, ub)
solver, sname = _detect_solver(threads=4, time_limit=600)

feas, skel, bm_max = biomass_feasible_skeleton(S, lb, ub, obj_idx, 0.1, solver=solver)
w = aggregate_confidence(pred, mask, allrxns, method="noisy_or")
w = np.clip(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 1e-6, 1.0 - 1e-6)
c = compute_costs(w, mode="logodds")["c"]
cand = build_candidate_mask(w, allrxns, excludes, media_rxns, essential_skeleton=skel, w_min=0.01)
m, y, vp, vn, _ = build_milp_hard(S, lb, ub, c, obj_idx, excludes, media_rxns, cand, 0.1, 2.5, 1e-4)
m.solve(solver)
yv = np.array([y[j].value() or 0 for j in range(len(y))])
v_vals = np.array([(vp[j].value() or 0) - (vn[j].value() or 0) for j in range(len(y))])
bm = v_vals[obj_idx]
n_active = int((yv > 0.5).sum())
print(f"[emit] {a.baseline}_{a.variant} {a.gca}: biomass={bm:.4f} n_active={n_active}", flush=True)

pickle.dump({"y_vals": yv, "v_vals": v_vals, "biomass_flux": float(bm),
             "n_active": n_active, "status": "MILP_hard"},
            open(f"{MOUT}/meteor_sol_{a.gca}.pkl", "wb"))
opt_df, active_ecs, muted_ecs = posterior_calibrated(pred, yv, seedr2ec, allrxns, anc, beta=a.beta)
opt_df.to_pickle(f"{MOUT}/meteor_df_{a.gca}.pkl")
pickle.dump({"active_ecs": active_ecs, "muted_ecs": muted_ecs},
            open(f"{MOUT}/meteor_preds_{a.gca}.pkl", "wb"))
print(f"[emit] saved sol/df/preds: {len(active_ecs)} active ECs, {time.time()-t0:.0f}s", flush=True)
