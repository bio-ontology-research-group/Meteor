"""K07: how many selected reactions can actually carry flux?

Selection sets y_j=1; the constraints do not force v_j!=0, so a negative-cost
reaction can enter the set at zero flux. This measures, per genome:
  - carrying    : |v| > tol in the biomass-maximising FBA solution
  - consistent  : |v| > tol somewhere in the feasible space (FVA), i.e. not blocked
Same submodel construction as eval/gen_table1.py so the counts line up with Table 1.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys, os, json, pickle, argparse, numpy as np
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7/src")

ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True); ap.add_argument("--gram", required=True)
ap.add_argument("--tol", type=float, default=1e-9)
ap.add_argument("--arm", default="meteor", choices=["meteor","baseline"])
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--outdir", default="/ibex/scratch/projects/c2014/kexin/funcarve/"
                                    "meteor_v8/results/fva_selected")
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
OUT = os.path.join(a.outdir, "fva_%s_%s.json" % (a.arm, a.gca))
if os.path.exists(OUT): print("done"); sys.exit(0)

from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                         apply_media, build_submodel)
from cobra.flux_analysis import flux_variability_analysis

F = "/ibex/scratch/projects/c2014/kexin/funcarve"
SOL = f"{F}/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla/meteor_sol_{a.gca}.pkl"
if not os.path.exists(SOL): print("no sol"); sys.exit(3)

universal, allrxns, allmet = load_universal()
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
biomass_id = "biomass_GmPos" if a.gram == "positive" else "biomass_GmNeg"
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lt, ut = load_tight_bounds(data_path('tight_bounds_v6_%s.pkl' % a.gram[:3]))
if lt is not None: lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
lb, ub, _, _ = apply_media(["default"], allrxns, lb, ub)
ix = {rid: i for i, rid in enumerate(allrxns)}
BND, UPTAKE_LB = 100.0, -10.0

if a.arm == "meteor":
    yv = np.array(pickle.load(open(SOL, "rb")).get("y_vals", [])) > 0.5
else:
    import glob as _g

    from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
    from meteor_v8.utils import load_refmapping, load_ec, extract_pred, build_rxn_ec_mask
    _p = resolve_baseline_pkl("dpz", "vanilla", a.gca, BASELINE_SUFFIX["dpz"])
    if not _p: print("no pred"); sys.exit(3)
    _s2e, _ = load_refmapping(data_dir()); _s2e = {k: v for k, v in _s2e.items() if v}
    _anc = load_ec(data_path('all_ancestors.txt'))
    _mask = build_rxn_ec_mask(allrxns, _s2e, _anc)
    _pred = extract_pred(_p, _anc)
    _hit = (_pred.values >= a.tau).any(axis=0)
    yv = np.array([bool((_mask[j] == 1).any() and _hit[_mask[j] == 1].any()) for j in range(len(allrxns))])
sel_ids = {allrxns[j] for j in np.where(yv)[0]}
keep = [r for r, k in zip(universal.reactions, yv) if k]
model = build_submodel(universal, keep, a.gca, biomass_id=biomass_id)
model.objective = biomass_id
for r in model.reactions:
    if r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower(): continue
    i = ix.get(r.id)
    r.lower_bound = max(-BND, float(lb[i])) if i is not None else -BND
    r.upper_bound = min(BND, float(ub[i])) if i is not None else BND
exs = [r for r in model.reactions if r.id.startswith("EX_") and r.id != "EX_biomass"]
for r in exs: r.lower_bound, r.upper_bound = -1000.0, 1000.0
so = model.optimize()
uptake = {r.id for r in exs if so.fluxes.get(r.id, 0.0) < -1e-6}
for r in exs:
    r.lower_bound = UPTAKE_LB if r.id in uptake else 0.0; r.upper_bound = 1000.0

sol = model.optimize()
targets = [r for r in model.reactions if r.id in sel_ids]
carrying = sum(1 for r in targets if abs(sol.fluxes.get(r.id, 0.0)) > a.tol)

fva = flux_variability_analysis(model, reaction_list=targets,
                                fraction_of_optimum=0.0, loopless=False)
consistent = int(((fva["maximum"].abs() > a.tol) | (fva["minimum"].abs() > a.tol)).sum())

res = dict(gca=a.gca, gram=a.gram, arm=a.arm, n_selected=len(targets),
           n_carrying_fba=int(carrying), n_flux_consistent=int(consistent),
           biomass=round(float(sol.objective_value or 0.0), 4))
res["frac_carrying"] = round(carrying / max(1, len(targets)), 4)
res["frac_consistent"] = round(consistent / max(1, len(targets)), 4)
json.dump(res, open(OUT, "w"))
print(f"{a.gca}: selected={res['n_selected']} carrying={carrying} "
      f"({res['frac_carrying']:.1%}) consistent={consistent} ({res['frac_consistent']:.1%})", flush=True)
