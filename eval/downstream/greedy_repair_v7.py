#!/usr/bin/env python3
"""v7/109-GCF greedy topology-only dead-end repair vs METEOR MILP.

Re-run of the "Greedy topology-only repair: a direct comparison" case study
(draft.tex sec:claim2) on the v7 hard-biomass-MILP / 109-GCF genome panel,
replacing the legacy 121-organism taxid panel. Reuses the SAME Step1/Step2
math as the main v7 pipeline (noisy-OR aggregation + log-odds cost from
src/v6utils.py) and the SAME submodel/medium logic as the panel builder
(build_grow_memote.py) so the FBA-feasibility check is apples-to-apples with
METEOR's own MILP active set.

Algorithm (unchanged from the original 121-organism
greedy_repair_array.sbatch found under /ibex/user/niuk0a/paperA_jobs/):
  1. w_j = noisy-OR aggregated per-reaction confidence from raw baseline
     scores (DeepProZyme-vanilla) through the EC->reaction map.
  2. y_init = 1[w_j > 0.5]  (per-protein threshold network, tau=0.5)
  3. cost c_j = -log((w_j+eps)/(1-w_j+eps))  (log-odds, same as METEOR Step 2)
  4. Greedy loop, up to --max_iter (20000): find all metabolites that are
     currently a dead end (no producer or no consumer among ACTIVE
     reactions, accounting for reversibility), gather every currently
     INACTIVE reaction that touches >=1 such metabolite, activate the one
     with lowest cost. Stop early if no dead-end metabolites remain
     (converged=1) or no candidate reaction exists to fix any of them
     (converged=0, "stuck").
  5. FBA check: build a COBRA submodel from the final active set with
     v6utils.build_submodel (same helper used by build_grow_memote.py --
     it auto-adds boundary reactions ONLY for metabolites already reachable
     by the active set, and force-adds the biomass reaction as objective),
     then derive a defined/bounded medium the same way build_grow_memote.py
     does (open all exchanges, run FBA once to find the uptake set, then
     re-bound to a realistic -10..1000 medium) and record the resulting
     biomass flux.

No MILP is solved here -- this loop is purely topological/greedy and does
not touch the meteor_v7/meteor/milp_hard.py MILP at all. METEOR's own
comparator numbers are read from the already-cached
meteor_v7_run/meteor_out/dpz_vanilla/meteor_sol_*.pkl files (n_active,
biomass_flux) produced by the main panel run -- NOT recomputed here.
"""
import sys, os, time, pickle, argparse
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

V6 = "/ibex/user/niuk0a/funcarve/cobra/v6"
sys.path.insert(0, V6)
os.chdir(V6)
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7/eval/downstream")
from src.v6utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                          apply_media, find_excluded_reactions, aggregate_confidence,
                          compute_costs, build_rxn_ec_mask, extract_pred, load_refmapping,
                          load_ec, build_submodel)
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX

ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True)
ap.add_argument("--gram", choices=["negative", "positive"], required=True)
ap.add_argument("--baseline", default="dpz", choices=["clean", "dpz", "enzbert"])
ap.add_argument("--variant", default="vanilla", choices=["vanilla", "filt30", "filt50", "filt70"])
ap.add_argument("--outdir", required=True)
ap.add_argument("--max_iter", type=int, default=20000)
ap.add_argument("--min_frac", type=float, default=0.9)
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)

SOLOUT = f"{a.outdir}/greedy_sol_{a.gca}.pkl"
METOUT = f"{a.outdir}/greedy_metrics_{a.gca}.tsv"
if os.path.exists(METOUT):
    print(f"already done: {METOUT}", flush=True)
    sys.exit(0)

biomass_id = "biomass_GmPos" if a.gram == "positive" else "biomass_GmNeg"
t0 = time.time()

_suf = BASELINE_SUFFIX[a.baseline]
pred_path = resolve_baseline_pkl(a.baseline, a.variant, a.gca, _suf)
if not pred_path:
    print(f"[GUARD] no pred file for {a.gca}", flush=True)
    sys.exit(3)

seedr2ec, _ = load_refmapping(f"{V6}/data")
seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, allmet = load_universal()
anc = load_ec(f"{V6}/data/all_ancestors.txt")
pred = extract_pred(pred_path, anc)

_ep = resolve_baseline_pkl("enzbert", "vanilla", a.gca, "enzbert")
_ref = pd.read_pickle(_ep).shape[0] if _ep else None
if _ref and pred.shape[0] < a.min_frac * _ref:
    print(f"[GUARD] incomplete pred for {a.gca}: {pred.shape[0]} < {a.min_frac}*{_ref}", flush=True)
    sys.exit(3)

print(f"[greedy] gca={a.gca} gram={a.gram} baseline={a.baseline}_{a.variant} "
      f"src={pred_path} pred={pred.shape}", flush=True)

mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lb_t, ub_t = load_tight_bounds(f"{V6}/data/tight_bounds_v6_{a.gram[:3]}.pkl")
if lb_t is not None:
    lb = np.maximum(lb, lb_t)
    ub = np.minimum(ub, ub_t)
obj_idx = allrxns.index(biomass_id)
excludes = find_excluded_reactions(S, lb, ub, allrxns, biomass_id)
lb, ub, media_mask, media_rxns = apply_media(["default"], allrxns, lb, ub)

n_rxns, n_mets = len(allrxns), len(allmet)

# --- Step 1-2: noisy-OR aggregation + log-odds cost (identical formula/code
#     path to the main v7 MILP pipeline, see emit_meteor_out.py) -----------
w = aggregate_confidence(pred, mask, allrxns, method="noisy_or")
w = np.clip(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0, posinf=1.0, neginf=0.0),
            1e-6, 1.0 - 1e-6)
cost = compute_costs(w, mode="logodds")["c"]

y_init = (w > 0.5).astype(np.int8)
n_init = int(y_init.sum())

rev_mask = (lb < 0) & (ub > 0)
S_sp = csc_matrix(S)


def dead_ends(y):
    active = np.where(y > 0.5)[0]
    no_p = np.ones(n_mets, dtype=bool)
    no_c = np.ones(n_mets, dtype=bool)
    for j in active:
        col = S_sp.getcol(j)
        is_rev = rev_mask[j]
        for i, c in zip(col.indices, col.data):
            if is_rev or c > 0:
                no_p[i] = False
            if is_rev or c < 0:
                no_c[i] = False
    return no_p, no_c


no_p0, no_c0 = dead_ends(y_init)
de_frac_init = float((no_p0 | no_c0).sum()) / n_mets
print(f"[greedy] {a.gca} threshold: {n_init} active rxns, w>0.5, "
      f"dead-end fraction={de_frac_init:.4f}", flush=True)

# met -> candidate reaction sets (topology only, same as legacy 121-panel script)
met_prodby = [set() for _ in range(n_mets)]
met_consby = [set() for _ in range(n_mets)]
for j in range(n_rxns):
    col = S_sp.getcol(j)
    is_rev = rev_mask[j]
    for i, c in zip(col.indices, col.data):
        if is_rev or c > 0:
            met_prodby[i].add(j)
        if is_rev or c < 0:
            met_consby[i].add(j)

# --- Step 3: greedy dead-end repair loop, capped at --max_iter -------------
t0_repair = time.perf_counter()
y_greedy = y_init.copy()
n_added = 0
converged = 0
n_iter_run = 0
for iteration in range(a.max_iter):
    n_iter_run = iteration
    no_p, no_c = dead_ends(y_greedy)
    target_mets = np.where(no_p | no_c)[0]
    if len(target_mets) == 0:
        converged = 1
        break

    candidates = set()
    for m in target_mets:
        if no_p[m]:
            candidates |= met_prodby[m]
        if no_c[m]:
            candidates |= met_consby[m]
    candidates = {j for j in candidates if y_greedy[j] == 0}

    if not candidates:
        print(f"[greedy] {a.gca} no candidates at iter {iteration}, stopping", flush=True)
        break

    cand_arr = np.array(list(candidates))
    best_j = cand_arr[np.argmin(cost[cand_arr])]
    y_greedy[best_j] = 1
    n_added += 1

    if iteration % 500 == 0:
        print(f"[greedy] {a.gca} iter={iteration} added={n_added} dead={len(target_mets)}",
              flush=True)

t_repair = time.perf_counter() - t0_repair
n_greedy = int(y_greedy.sum())
no_p1, no_c1 = dead_ends(y_greedy)
de_frac_greedy = float((no_p1 | no_c1).sum()) / n_mets
print(f"[greedy] {a.gca} repair done: added={n_added} total={n_greedy} "
      f"dead-end={de_frac_greedy:.4f} converged={converged} time={t_repair:.1f}s", flush=True)

# --- Step 4: FBA feasibility, SAME submodel + defined-medium logic as
#     build_grow_memote.py (the panel builder) -----------------------------
biomass_flux = float("nan")
n_rxn_submodel = 0
n_exchange = 0
n_uptake = 0
try:
    keep = [r for r, k in zip(universal.reactions, y_greedy > 0.5) if k]
    model = build_submodel(universal, keep, f"greedy_{a.gca}", biomass_id=biomass_id)
    model.objective = biomass_id

    _ix = {rid: i for i, rid in enumerate(allrxns)}
    for r in model.reactions:
        if r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower():
            continue
        i = _ix.get(r.id)
        if i is not None:
            r.lower_bound = float(lb[i])
            r.upper_bound = float(ub[i])

    UPTAKE_LB = -10.0
    exs = [r for r in model.reactions if r.id.startswith("EX_") and r.id != "EX_biomass"]
    for r in exs:
        r.lower_bound = -1000.0
        r.upper_bound = 1000.0
    sol_open = model.optimize()
    uptake = {r.id for r in exs if (sol_open.fluxes.get(r.id, 0.0) < -1e-6)}
    for r in exs:
        r.lower_bound = UPTAKE_LB if r.id in uptake else 0.0
        r.upper_bound = 1000.0

    biomass_flux = float(model.slim_optimize() or 0.0)
    n_rxn_submodel = len(model.reactions)
    n_exchange = len(exs)
    n_uptake = len(uptake)
    print(f"[greedy] {a.gca} submodel={n_rxn_submodel} rxns, exchanges={n_exchange}, "
          f"uptake={n_uptake}, FBA biomass={biomass_flux:.6f}", flush=True)
except Exception as e:
    print(f"[greedy] {a.gca} FBA check failed: {e}", flush=True)

pickle.dump({
    "gca": a.gca, "y_init": y_init, "y_greedy": y_greedy, "allrxns": allrxns,
    "n_init": n_init, "n_greedy": n_greedy, "n_added": n_added,
    "de_frac_init": de_frac_init, "de_frac_greedy": de_frac_greedy,
    "converged": converged, "n_iter_run": n_iter_run,
    "biomass_flux": biomass_flux, "n_rxn_submodel": n_rxn_submodel,
    "n_exchange": n_exchange, "n_uptake": n_uptake,
    "t_repair_s": t_repair,
}, open(SOLOUT, "wb"))
print(f"[greedy] {a.gca} sol saved: {SOLOUT}", flush=True)

metrics = {
    "gca": a.gca,
    "n_rxn_thresh": n_init,
    "n_rxn_greedy": n_greedy,
    "n_added": n_added,
    "de_frac_thresh": round(de_frac_init, 6),
    "de_frac_greedy": round(de_frac_greedy, 6),
    "biomass_flux_greedy": round(biomass_flux, 6) if biomass_flux == biomass_flux else -1.0,
    "n_rxn_submodel": n_rxn_submodel,
    "n_exchange": n_exchange,
    "n_uptake": n_uptake,
    "converged": converged,
    "n_iter_run": n_iter_run,
    "t_repair_s": round(t_repair, 2),
    "elapsed_s": round(time.time() - t0, 1),
}
pd.DataFrame([metrics]).to_csv(METOUT, sep="\t", index=False)
print(f"[greedy] {a.gca} metrics saved: {METOUT}", flush=True)
print(f"DONE greedy {a.gca}", flush=True)
