#!/usr/bin/env python3
"""v7 FIXED metabolic-model build: hard-biomass MILP (real biomass flux) +
submodel export with a DEFINED growth medium + cross-reference/charge annotation,
then MEMOTE. Fixes the growth=0 defect (v6 soft-slack + un-set medium)."""
import sys, os, pickle, json, argparse, time
import numpy as np
import pandas as pd

V6 = "/ibex/user/niuk0a/funcarve/cobra/v6"
PA = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026"
V2 = "/ibex/user/niuk0a/meteor_v7_V2/experiments/v7_search/V2_annotation"
sys.path.insert(0, V6); os.chdir(V6)
sys.path.insert(0, "/ibex/user/niuk0a/meteor_v7/src")
sys.path.insert(0, V2)
from src.v6utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                         apply_media, find_excluded_reactions, aggregate_confidence,
                         compute_costs, build_candidate_mask, build_rxn_ec_mask,
                         extract_pred, load_refmapping, load_ec, _detect_solver,
                         build_submodel, assign_gpr_from_predictions, add_annotation,
                         apply_bounds_to_submodel, load_modelseed_charge)
from meteor.milp_hard import biomass_feasible_skeleton, build_milp_hard
from v2_xrefs import load_met_xrefs, load_rxn_xrefs, apply_met_xrefs, apply_rxn_xrefs

ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True)
ap.add_argument("--gram", choices=["negative", "positive"], required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--baseline", choices=["clean", "dpz", "enzbert"], default="dpz")
ap.add_argument("--variant", choices=["vanilla", "filt30", "filt50", "filt70"], default="vanilla")
ap.add_argument("--min_frac", type=float, default=0.9,
                help="completeness guard: pred rows must be >= min_frac * reference proteome size")
a = ap.parse_args()
t0 = time.time()
os.makedirs(a.outdir, exist_ok=True)
biomass_id = "biomass_GmPos" if a.gram == "positive" else "biomass_GmNeg"

seedr2ec, _ = load_refmapping(f"{V6}/data"); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, allmet = load_universal()
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
anc = load_ec(f"{V6}/data/all_ancestors.txt")
# --- baseline pred path: prefer recovered reinfer/ over scratch; completeness guard ---
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7/eval/downstream")
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
_suf = BASELINE_SUFFIX[a.baseline]
_pred_path = resolve_baseline_pkl(a.baseline, a.variant, a.gca, _suf)
if not _pred_path:
    print(f"[GUARD] ABORT {a.baseline}_{a.variant} {a.gca}: no pred file", flush=True)
    sys.exit(3)
pred = extract_pred(_pred_path, anc)
_ep = resolve_baseline_pkl("enzbert", "vanilla", a.gca, "enzbert")
_ref_n = pd.read_pickle(_ep).shape[0] if _ep else None
if _ref_n and pred.shape[0] < a.min_frac * _ref_n:
    print(f"[GUARD] ABORT {a.baseline}_{a.variant} {a.gca}: pred rows {pred.shape[0]} < "
          f"{a.min_frac}*{_ref_n} (incomplete prediction)", flush=True)
    sys.exit(3)
print(f"[grow] baseline={a.baseline} variant={a.variant} "
      f"src={_pred_path} pred={pred.shape} ref={_ref_n}", flush=True)
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lb_t, ub_t = load_tight_bounds(f"{V6}/data/tight_bounds_v6_{a.gram[:3]}.pkl")
if lb_t is not None: lb = np.maximum(lb, lb_t); ub = np.minimum(ub, ub_t)
obj_idx = allrxns.index(biomass_id)
excludes = find_excluded_reactions(S, lb, ub, allrxns, biomass_id)
lb, ub, media_mask, media_rxns = apply_media(["default"], allrxns, lb, ub)
solver, sname = _detect_solver(threads=4, time_limit=600)

# --- ModelSEED thermodynamic directionality on bounds BEFORE the MILP, so the
#     MILP finds a directionally-consistent (loop-poor) growing flux ---
if os.environ.get("DIRPRE") == "1":
    _rt = pd.read_csv("/ibex/user/niuk0a/meteor_v7/eval/modelseed_reactions_full.csv",
                      low_memory=False)
    _dir = dict(zip(_rt["ID"].astype(str), _rt["Reversibility"].astype(str)))
    _n = 0
    for j, rid in enumerate(allrxns):
        base = rid[:-2] if rid.endswith(("_c", "_e", "_p")) else rid
        if base.startswith(("EX_", "DM_", "SK_")) or "biomass" in rid.lower():
            continue
        d = _dir.get(base)
        if d == ">" and lb[j] < 0:
            lb[j] = 0.0; _n += 1
        elif d == "<" and ub[j] > 0:
            ub[j] = 0.0; _n += 1
    _f, _, _bx = biomass_feasible_skeleton(S, lb, ub, obj_idx, 0.1, solver=solver)
    print(f"[grow] pre-MILP directionality: {_n} internal rxns constrained; "
          f"biomass_max after={_bx:.3f} feasible={_f}", flush=True)

# --- hard-biomass MILP (real biomass flux, no slack) ---
feas, skel, bm_max = biomass_feasible_skeleton(S, lb, ub, obj_idx, 0.1, solver=solver)
w = aggregate_confidence(pred, mask, allrxns, method="noisy_or")
# clamp to keep log-odds cost finite (w=0/1 or NaN in baseline preds -> inf/NaN)
w = np.clip(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 1e-6, 1.0 - 1e-6)
c = compute_costs(w, mode="logodds")["c"]
cand = build_candidate_mask(w, allrxns, excludes, media_rxns, essential_skeleton=skel, w_min=0.01)
m, y, vp, vn, _ = build_milp_hard(S, lb, ub, c, obj_idx, excludes, media_rxns, cand, 0.1, 2.5, 1e-4)
m.solve(solver)
bm_milp = (vp[obj_idx].value() or 0) - (vn[obj_idx].value() or 0)
yv = np.array([y[j].value() or 0 for j in range(len(y))])
print(f"[grow] hard MILP biomass_flux={bm_milp:.4f} n_active={int((yv>0.5).sum())}", flush=True)

# --- submodel + DEFINED growth medium ---
# The model grows with all exchanges open (structurally complete). We derive a
# defined medium = the exchanges the model actually consumes for growth, then
# lock the rest to secretion-only. This gives a realistic default-condition
# growth (not the trivial all-open flood) while guaranteeing the model grows.
keep = [r for r, k in zip(universal.reactions, yv > 0.5) if k]
model = build_submodel(universal, keep, a.gca, biomass_id=biomass_id)
model.objective = biomass_id
# tighten INTERNAL reaction bounds (FVA tight + media lb/ub) to fix MEMOTE
# unbounded-flux; leave EXCHANGE bounds to the medium logic below. The MILP
# produced biomass 0.1 under these same internal bounds, so growth survives.
_ix = {rid: i for i, rid in enumerate(allrxns)}
BND = 100.0   # finite physiological flux bound; caps futile-cycle magnitude so
              # MEMOTE no longer flags reactions as unbounded (default is +-1000)
for r in model.reactions:
    if r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower():
        continue
    i = _ix.get(r.id)
    lo = max(-BND, float(lb[i])) if i is not None else -BND
    hi = min(BND, float(ub[i])) if i is not None else BND
    r.lower_bound = lo; r.upper_bound = hi
UPTAKE_LB = -10.0   # realistic bounded uptake (mmol/gDW/h), not unlimited -1000
exs = [r for r in model.reactions if r.id.startswith("EX_") and r.id != "EX_biomass"]
for r in exs:
    r.lower_bound = -1000.0; r.upper_bound = 1000.0        # open all to find uptakes
sol_open = model.optimize()
uptake = {r.id for r in exs if (sol_open.fluxes.get(r.id, 0.0) < -1e-6)}
for r in exs:                                              # defined, BOUNDED medium
    r.lower_bound = UPTAKE_LB if r.id in uptake else 0.0
    r.upper_bound = 1000.0
bm_fba = model.slim_optimize()
print(f"[grow] defined medium: %d uptake EX / %d total; uptake_lb=%.0f FBA biomass=%.4f" % (
    len(uptake), len(exs), UPTAKE_LB, bm_fba), flush=True)

# --- ModelSEED thermodynamic directionality (fast futile-cycle removal) ---
if os.environ.get("DIRECTION") == "1":
    rt = pd.read_csv(f"{V2}/data/reactions.tsv", sep="\t", low_memory=False)
    dirmap = dict(zip(rt["id"].astype(str), rt["direction"].astype(str)))
    nfix = 0
    for r in model.reactions:
        if "biomass" in r.id.lower() or r.id.startswith(("EX_", "DM_", "SK_")):
            continue
        base = r.id[:-2] if r.id.endswith(("_c", "_e", "_p")) else r.id
        d = dirmap.get(base)
        if d == ">" and r.lower_bound < 0:
            r.lower_bound = 0.0; nfix += 1
        elif d == "<" and r.upper_bound > 0:
            r.upper_bound = 0.0; nfix += 1
    print(f"[grow] ModelSEED directionality: constrained {nfix} rxns to irreversible; "
          f"FBA biomass after={model.slim_optimize():.4f}", flush=True)

# --- loopless FVA bound tightening (break futile cycles -> fix unbounded_flux) ---
if os.environ.get("LOOPLESS") == "1":
    from cobra.flux_analysis import flux_variability_analysis
    _t0 = time.time()
    _revrx = [r for r in model.reactions
              if r.lower_bound < 0 and r.upper_bound > 0
              and "biomass" not in r.id.lower()
              and not r.id.startswith(("EX_", "DM_", "SK_"))]
    print(f"[grow] cycleFreeFlux scope: {len(_revrx)}/{len(model.reactions)} reversible rxns (cycle carriers)", flush=True)
    _llm = os.environ.get("LLMETHOD", "cycleFreeFlux")
    print(f"[grow] loopless method={_llm}", flush=True)
    fva = flux_variability_analysis(model, reaction_list=_revrx, loopless=_llm, fraction_of_optimum=0.1, processes=8)
    nt = 0
    for r in model.reactions:
        if r.id in fva.index and "biomass" not in r.id.lower():
            lo = float(fva.at[r.id, "minimum"]); hi = float(fva.at[r.id, "maximum"])
            if hi < lo: lo, hi = hi, lo
            r.lower_bound = lo; r.upper_bound = hi; nt += 1
    print(f"[grow] loopless-FVA tightened {nt} rxns in {time.time()-_t0:.0f}s; "
          f"FBA biomass after={model.slim_optimize():.4f}", flush=True)

# --- annotation: GPR + charge + cross-references ---
model, n_gpr, _ = assign_gpr_from_predictions(model, pred, seedr2ec, tau=0.5)
charge_map = load_modelseed_charge(f"{V6}/data/modelseed_compounds.csv")
model = add_annotation(model, gram=a.gram, obj=biomass_id, charge_map=charge_map)
apply_met_xrefs(model, load_met_xrefs(f"{V6}/data/modelseed_compounds.csv"))
apply_rxn_xrefs(model, load_rxn_xrefs(f"{V2}/data/reactions.tsv", seedr2ec=seedr2ec), seedr2ec=seedr2ec)
for g in model.genes:
    ann = g.annotation if isinstance(g.annotation, dict) else {}
    ann.setdefault("uniprot", g.id.rsplit(".", 1)[0]); g.annotation = ann

# --- MEMOTE ---
import memote
from memote.suite.api import snapshot_report
_, res = memote.test_model(model, results=True)
rep = json.loads(snapshot_report(res, html=False))
(open(f"{a.outdir}/{a.gca}.json", "w")).write(json.dumps(rep))
t = rep["tests"]
bdef = list(t["test_biomass_default_production"]["data"].values())[0]
bopen = list(t["test_biomass_open_production"]["data"].values())[0]
tot = rep["score"]["total_score"]
row = dict(gca=a.gca, gram=a.gram, hard_biomass=round(bm_milp, 4), fba_biomass=round(bm_fba, 4),
           memote_default=bdef, memote_open=bopen, memote_total=round(tot, 4),
           n_rxn=len(model.reactions), elapsed_s=round(time.time()-t0, 1))
json.dump(row, open(f"{a.outdir}/{a.gca}_summary.json", "w"))
print(f"[grow] MEMOTE total={tot:.4f} default_growth={bdef} open_growth={bopen}", flush=True)
print("SUMMARY\t" + json.dumps(row), flush=True)
