#!/usr/bin/env python3
"""Lightweight v7 METEOR-output emitter (no MEMOTE build): runs the hard-biomass
MILP + v6 posterior and saves v6_sol / v6_df / v6_preds(active_ecs) per genome,
in the format the downstream evals (pathway/bgc/capability/per-protein) read.
Same MILP as build_grow_memote.py; just skips the model build + MEMOTE."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys, os, pickle, json, argparse, time
import numpy as np
import pandas as pd

PA = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026"

from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                         apply_media, find_excluded_reactions, aggregate_confidence,
                         compute_costs, build_candidate_mask, build_rxn_ec_mask,
                         extract_pred, load_refmapping, load_ec, _detect_solver,
                         posterior_calibrated)
from meteor_v8.milp_hard import biomass_feasible_skeleton
from meteor_v8.milp_v8 import build_milp_v8
from meteor_v8.repair import verify_and_repair

ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True)
ap.add_argument("--gram", choices=["negative", "positive"], required=True)
ap.add_argument("--baseline", choices=["clean", "dpz", "enzbert"], default="dpz")
ap.add_argument("--variant", choices=["vanilla", "filt30", "filt50", "filt70"], default="vanilla")
ap.add_argument("--outroot", default="/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_run/meteor_out")
ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--min_frac", type=float, default=0.9)
ap.add_argument("--mu", type=float, default=3.0)          # PSB2027 paper config
ap.add_argument("--eps", type=float, default=0.0)         # forced-flux floor OFF in the paper
ap.add_argument("--penalty", choices=["uniform","evw"], default="evw")
ap.add_argument("--pexp", type=float, default=2.0)
ap.add_argument("--gmin", type=float, default=0.1)          # biomass floor (0 = no-biomass ablation)
ap.add_argument("--no_repair", action="store_true")         # skip verify-and-repair (no-repair ablation)
a = ap.parse_args()
t0 = time.time()
MOUT = os.path.join(a.outroot, f"{a.baseline}_{a.variant}")
os.makedirs(MOUT, exist_ok=True)
if os.path.exists(f"{MOUT}/meteor_preds_{a.gca}.pkl"):
    print(f"skip {a.gca}", flush=True); sys.exit(0)
biomass_id = "biomass_GmPos" if a.gram == "positive" else "biomass_GmNeg"

seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, allmet = load_universal()
anc = load_ec(data_path('all_ancestors.txt'))

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
lb_t, ub_t = load_tight_bounds(data_path(f'tight_bounds_v6_{a.gram[:3]}.pkl'))
if lb_t is not None: lb = np.maximum(lb, lb_t); ub = np.minimum(ub, ub_t)
obj_idx = allrxns.index(biomass_id)
excludes = find_excluded_reactions(S, lb, ub, allrxns, biomass_id)
lb, ub, media_mask, media_rxns = apply_media(["default"], allrxns, lb, ub)
solver, sname = _detect_solver(threads=4, time_limit=600)

feas, skel, bm_max = biomass_feasible_skeleton(S, lb, ub, obj_idx, 0.1, solver=solver)
_P=pred.values.astype(np.float32).copy(); _ne=_P.shape[1]
if 5<_ne:
    _dr=np.argpartition(_P,_ne-5,axis=1)[:,:_ne-5]; np.put_along_axis(_P,_dr,0.0,axis=1)
_l1m=np.log(np.clip(1.0-_P,1e-9,1.0))
w=np.zeros(len(allrxns),dtype=np.float32)
for _j in range(len(allrxns)):
    _ei=np.where(mask[_j]==1)[0]
    if len(_ei): w[_j]=1.0-np.exp(float(_l1m[:,_ei].sum()))
w = np.clip(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 1e-6, 1.0 - 1e-6)
c = compute_costs(w, mode="logodds")["c"]
_mu_eff = a.mu
if a.penalty == "evw":
    # evidence-weighted parsimony: mu_j = mu*(1-w_j)^p folded into cost; call build with mu=0
    _pen = a.mu * np.power(np.clip(1.0 - w, 0.0, 1.0), a.pexp)
    c = np.asarray(c, dtype=float) + _pen
    _mu_eff = 0.0
cand = build_candidate_mask(w, allrxns, excludes, media_rxns, essential_skeleton=skel, w_min=0.01)
import time as _time, pulp as _pulp
m, y, vp, vn, _ = build_milp_v8(S, lb, ub, c, obj_idx, excludes, media_rxns, cand, a.gmin, 2.5, 1e-4, mu=_mu_eff, eps=a.eps)
_t0 = _time.time()
m.solve(solver)
_solve_sec = round(_time.time() - _t0, 2)
_solver_status = _pulp.LpStatus[m.status]
yv = np.array([y[j].value() or 0 for j in range(len(y))])
v_vals = np.array([(vp[j].value() or 0) - (vn[j].value() or 0) for j in range(len(y))])
bm = v_vals[obj_idx]
# verify-and-repair the v8 big-M gap-fill leak (degenerate solutions); no-op if clean
if a.no_repair:
    n_active = int((yv > 0.5).sum()); n_repaired = 0; _mb_core = float(bm)
else:
    yv, n_active, n_repaired, _mb_core = verify_and_repair(S, lb, ub, obj_idx, yv, v_vals, cand)
print(f"[emit] {a.baseline}_{a.variant} {a.gca}: biomass={bm:.4f} n_active={n_active} "
      f"mb_core={_mb_core:.3f} repaired={n_repaired}", flush=True)

pickle.dump({"y_vals": yv, "v_vals": v_vals, "biomass_flux": float(bm),
             "n_active": n_active, "status": _solver_status, "solver": "MILP_hard_evw", "solve_sec": _solve_sec,
             "n_repaired": int(n_repaired), "maxbio_core": float(_mb_core)},
            open(f"{MOUT}/meteor_sol_{a.gca}.pkl", "wb"))
opt_df, active_ecs, muted_ecs = posterior_calibrated(pred, yv, seedr2ec, allrxns, anc, beta=a.beta)
opt_df.to_pickle(f"{MOUT}/meteor_df_{a.gca}.pkl")
pickle.dump({"active_ecs": active_ecs, "muted_ecs": muted_ecs},
            open(f"{MOUT}/meteor_preds_{a.gca}.pkl", "wb"))
print(f"[emit] saved sol/df/preds: {len(active_ecs)} active ECs, {time.time()-t0:.0f}s", flush=True)
