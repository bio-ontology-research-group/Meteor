"""E3: skeleton ablation, DPZ vanilla, published flags. Two arms:
  noskel   - candidate mask = evidence (w>=0.01) U medium; skeleton not protected
  skelonly - candidate mask = skeleton U medium; no evidence reactions selectable
Structural metrics use profile() copied from meteor_v8/eval/gen_table1.py so they
are comparable with Table 1 / Table S5. Writes only under psb_revision/results."""
import os, sys, json, pickle, time, argparse, numpy as np, pulp
from _env import *
from meteor_v8.utils import (data_path, data_dir, load_universal, extract_fba_matrices, load_tight_bounds, apply_media,
    find_excluded_reactions, compute_costs, build_candidate_mask, build_rxn_ec_mask, extract_pred, load_refmapping,
    load_ec, _detect_solver, build_submodel)
from meteor_v8.milp_hard import biomass_feasible_skeleton
from meteor_v8.milp_v8 import build_milp_v8
from meteor_v8.repair import verify_and_repair
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
import memote.support.consistency as cons

ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True); ap.add_argument("--gram", choices=["negative", "positive"], required=True)
ap.add_argument("--arm", choices=["noskel", "skelonly", "full", "nomask"], required=True,
                help="'full' re-solves the published mask as an in-run control; 'nomask' makes every non-excluded universal reaction selectable")
ap.add_argument("--time_limit", type=int, default=600)
ap.add_argument("--outdir", default=f"{RESULTS}/skeleton_abl")
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
OUT = f"{a.outdir}/{a.gca}_{a.arm}.json"
if os.path.exists(OUT): print("done", OUT); sys.exit(0)
F = MILP_FLAGS; bid = "biomass_GmPos" if a.gram == "positive" else "biomass_GmNeg"

seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, allmet = load_universal(); anc = load_ec(data_path("all_ancestors.txt"))
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
pred = extract_pred(resolve_baseline_pkl("dpz", "vanilla", a.gca, BASELINE_SUFFIX["dpz"]), anc)
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{a.gram[:3]}.pkl"))
if lt is not None: lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
obj_idx = allrxns.index(bid); excludes = find_excluded_reactions(S, lb, ub, allrxns, bid)
lb, ub, media_mask, media_rxns = apply_media(["default"], allrxns, lb, ub)
solver, _ = _detect_solver(threads=4, time_limit=a.time_limit)
feas, skel, bm_max = biomass_feasible_skeleton(S, lb, ub, obj_idx, F["gmin"], solver=solver)

# ---- w and c exactly as emit_v8.py
P = pred.values.astype(np.float32).copy(); ne = P.shape[1]
if ne > 5:
    dr = np.argpartition(P, ne-5, axis=1)[:, :ne-5]; np.put_along_axis(P, dr, 0.0, axis=1)
l1m = np.log(np.clip(1.0-P, 1e-9, 1.0)); w = np.zeros(len(allrxns), dtype=np.float32)
for j in range(len(allrxns)):
    ei = np.where(mask[j] == 1)[0]
    if len(ei): w[j] = 1.0 - np.exp(float(l1m[:, ei].sum()))
w = np.clip(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 1e-6, 1-1e-6)
c = np.asarray(compute_costs(w, mode="logodds")["c"], dtype=float) + F["mu"] * np.power(np.clip(1.0-w, 0, 1), F["pexp"])

# ---- the only thing that differs between arms
if a.arm == "full":
    cand = build_candidate_mask(w, allrxns, excludes, media_rxns, essential_skeleton=skel, w_min=F["wmin"])
elif a.arm == "nomask":
    cand = np.ones(len(allrxns), dtype=bool)
    for j in excludes: cand[j] = False
elif a.arm == "noskel":
    cand = build_candidate_mask(w, allrxns, excludes, media_rxns, essential_skeleton=None, w_min=F["wmin"])
else:
    cand = np.zeros(len(allrxns), dtype=bool)
    for j in skel: cand[j] = True
    for j in media_rxns: cand[j] = True
    for j in excludes: cand[j] = False
n_cand = int(cand.sum())

m, y, vp, vn, _ = build_milp_v8(S, lb, ub, c, obj_idx, excludes, media_rxns, cand, F["gmin"], F["gmax"], F["lam"], mu=0.0, eps=F["eps"])
t0 = time.time(); m.solve(solver); solve_sec = round(time.time()-t0, 2); status = pulp.LpStatus[m.status]
yv = np.array([y[j].value() or 0 for j in range(len(y))]); vv = np.array([(vp[j].value() or 0)-(vn[j].value() or 0) for j in range(len(y))])
rec = dict(gca=a.gca, gram=a.gram, arm=a.arm, status=status, solve_sec=solve_sec, n_candidate=n_cand, n_skeleton=len(skel), flags=F,
           time_limit=a.time_limit, objective=float(pulp.value(m.objective)) if m.status in (1,) or (yv > 0.5).any() else None,
           cost_selected_raw=float(c[yv > 0.5].sum()) if (yv > 0.5).any() else None)

def profile(keep_flags):
    """verbatim logic of meteor_v8/eval/gen_table1.py::profile"""
    keep = [r for r, k in zip(universal.reactions, keep_flags) if k]
    if not keep: return {"err": "empty"}
    model = build_submodel(universal, keep, a.gca, biomass_id=bid); model.objective = bid
    ix = {rid: i for i, rid in enumerate(allrxns)}; BND, UPTAKE_LB = 100.0, -10.0
    for r in model.reactions:
        if r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower(): continue
        i = ix.get(r.id)
        r.lower_bound = max(-BND, float(lb[i])) if i is not None else -BND
        r.upper_bound = min(BND, float(ub[i])) if i is not None else BND
    exs = [r for r in model.reactions if r.id.startswith("EX_") and r.id != "EX_biomass"]
    for r in exs: r.lower_bound, r.upper_bound = -1000.0, 1000.0
    so = model.optimize(); uptake = {r.id for r in exs if so.fluxes.get(r.id, 0.0) < -1e-6}
    for r in exs: r.lower_bound = UPTAKE_LB if r.id in uptake else 0.0; r.upper_bound = 1000.0
    bm = model.slim_optimize(); de = len(cons.find_deadends(model))
    mi = len(cons.find_mass_unbalanced_reactions(model.reactions)); nr = len(model.reactions)
    return dict(n_selected=int(keep_flags.sum()), n_rxn=nr, fba_growth=round(float(bm), 4), deadends=de,
                mass_imbal=mi, mi_frac=round(mi/max(1, nr), 4), n_uptake=len(uptake))

if status in ("Optimal", "Not Solved") and (yv > 0.5).any():
    yv, n_active, n_repaired, mb_core = verify_and_repair(S, lb, ub, obj_idx, yv, vv, cand)
    rec.update(n_repaired=int(n_repaired), maxbio_core=float(mb_core), milp_biomass=float(vv[obj_idx]))
    rec.update(profile(yv > 0.5))
    pickle.dump({"y_vals": yv, "arm": a.arm}, open(f"{a.outdir}/{a.gca}_{a.arm}_y.pkl", "wb"))
else:
    rec.update(err=status, n_selected=0)
print(json.dumps(rec), flush=True)
json.dump(rec, open(OUT, "w"))
