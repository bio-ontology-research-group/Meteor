"""Table 1 v2 — per-genome structural record with the corrected mass-imbalance metric.

Copy of psb_revision/code_snapshot/eval/gen_table1.py (itself a copy of meteor_v8/eval/gen_table1.py).
Every existing field and code path is untouched so n_selected/n_rxn/deadends/mass_imbal/mi_frac/
fba_growth reproduce meteor_v8/results/table1/table1_{gca}.json exactly. profile() ADDS:
  n_internal               = # reactions not boundary (EX_/DM_/SK_/SNK_ prefix or cobra r.boundary) and not biomass
  mi_int                   = # internal reactions failing memote's elemental balance
  mi_int_missing_formula   = of those, # with >=1 metabolite lacking elements (memote returns False for these)
  mi_int_genuine           = mi_int - mi_int_missing_formula
  mi_frac_int              = mi_int / n_internal
Extra arms (same profile): abl_full, abl_skelonly (psb_revision/results/skeleton_abl/{gca}_{arm}_y.pkl),
abl_uniform (meteor_v8_mu3e0_run/meteor_out/dpz_vanilla/meteor_sol_{gca}.pkl = uniform-cost mu3 eps0 run).
Reads everything read-only; writes only psb_revision/results/table1_v2/table1_{gca}.json.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _env  # noqa: E402  -> METEOR_DATA=code_snapshot/data, sys.path += code_snapshot/{src,eval}
import sys, os, json, pickle, argparse, numpy as np
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7/src")

ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True)
ap.add_argument("--gram", required=True, choices=["negative", "positive"])
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--outdir", default=f"{_env.RESULTS}/table1_v2")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
OUT = os.path.join(a.outdir, "table1_%s.json" % a.gca)
if os.path.exists(OUT):
    print("done", flush=True); sys.exit(0)

from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                         apply_media, build_submodel, load_refmapping, load_ec,
                         extract_pred, build_rxn_ec_mask, data_path, data_dir)
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
import memote.support.consistency as cons
import memote.support.consistency_helpers as con_helpers

MET = _env.RUNS  # meteor_v8_evw_p2mu3_run/meteor_out
SKEL = f"{_env.RESULTS}/skeleton_abl"
UNIF = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_mu3e0_run/meteor_out/dpz_vanilla"
BASELINES = ["clean", "dpz", "enzbert"]
BOUNDARY_PREF = ("EX_", "DM_", "SK_", "SNK_")  # SEED universal names sinks SNK_cpd*_c

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

seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path('all_ancestors.txt'))
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)


def is_internal(r):
    """Internal = not a boundary reaction (EX_/DM_/SK_/SNK_ prefix or single-metabolite
    cobra boundary) and not the biomass objective. build_submodel adds one SNK_ per kept
    metabolite; those are one-sided by construction and excluded here."""
    return not (r.id.startswith(BOUNDARY_PREF) or r.boundary or "biomass" in r.id.lower())


def profile(keep_flags, label):
    """keep_flags: bool array over allrxns. Returns the structural record."""
    keep = [r for r, k in zip(universal.reactions, keep_flags) if k]
    if not keep:
        return {"err": "empty"}
    model = build_submodel(universal, keep, a.gca, biomass_id=biomass_id)
    model.objective = biomass_id
    for r in model.reactions:
        if r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower():
            continue
        i = ix.get(r.id)
        r.lower_bound = max(-BND, float(lb[i])) if i is not None else -BND
        r.upper_bound = min(BND, float(ub[i])) if i is not None else BND
    exs = [r for r in model.reactions if r.id.startswith("EX_") and r.id != "EX_biomass"]
    for r in exs: r.lower_bound, r.upper_bound = -1000.0, 1000.0
    so = model.optimize()
    uptake = {r.id for r in exs if so.fluxes.get(r.id, 0.0) < -1e-6}
    for r in exs:
        r.lower_bound = UPTAKE_LB if r.id in uptake else 0.0
        r.upper_bound = 1000.0
    bm = model.slim_optimize()
    de = len(cons.find_deadends(model))
    mi = len(cons.find_mass_unbalanced_reactions(model.reactions))
    nr = len(model.reactions)
    rec = dict(n_selected=int(keep_flags.sum()), n_rxn=nr,
               fba_growth=round(float(bm), 4), deadends=de, mass_imbal=mi,
               mi_frac=round(mi / max(1, nr), 4), n_uptake=len(uptake))
    # ---- v2 additions (pure read of the same model object; nothing above changes) ----
    internal = [r for r in model.reactions if is_internal(r)]
    mi_int_rxns = [r for r in internal if not con_helpers.is_mass_balanced(r)]
    missing = [r for r in mi_int_rxns
               if any(m.elements is None or len(m.elements) == 0 for m in r.metabolites)]
    rec.update(n_internal=len(internal), mi_int=len(mi_int_rxns),
               mi_int_missing_formula=len(missing),
               mi_int_genuine=len(mi_int_rxns) - len(missing),
               mi_frac_int=round(len(mi_int_rxns) / max(1, len(internal)), 5),
               mi_int_ids=sorted(r.id for r in mi_int_rxns))
    print("[%s] %s sel=%d rxn=%d growth=%.3f deadend=%d mi=%d | int=%d mi_int=%d (missing %d, genuine %d)"
          % (label, a.gca, rec["n_selected"], nr, bm, de, mi, rec["n_internal"], rec["mi_int"],
             rec["mi_int_missing_formula"], rec["mi_int_genuine"]), flush=True)
    return rec


def yvals_from_pkl(path):
    if not os.path.exists(path):
        return None
    return np.array(pickle.load(open(path, "rb")).get("y_vals", []))


res = {"gca": a.gca, "gram": a.gram, "tau": a.tau}

for b in BASELINES:
    # --- baseline network: reaction kept if any EC of it scores >= tau ---
    p = resolve_baseline_pkl(b, "vanilla", a.gca, BASELINE_SUFFIX[b])
    if not p:
        res["baseline_" + b] = {"err": "no_pred"}
    else:
        pred = extract_pred(p, anc)
        hit = (pred.values >= a.tau).any(axis=0)          # EC columns above tau
        flags = np.array([bool((mask[j] == 1).any() and hit[mask[j] == 1].any())
                          for j in range(len(allrxns))])
        res["baseline_" + b] = profile(flags, "base_" + b)

    # --- METEOR selected set ---
    f = "%s/%s_vanilla/meteor_sol_%s.pkl" % (MET, b, a.gca)
    yv = yvals_from_pkl(f)
    res["meteor_" + b] = {"err": "no_sol"} if yv is None else profile(yv > 0.5, "meteor_" + b)

# --- extra arms: candidate-mask ablation (T4) and uniform-cost ablation ---
for arm in ("full", "skelonly"):
    yv = yvals_from_pkl(f"{SKEL}/{a.gca}_{arm}_y.pkl")
    res["abl_" + arm] = {"err": "no_sol"} if yv is None else profile(yv > 0.5, "abl_" + arm)
yv = yvals_from_pkl(f"{UNIF}/meteor_sol_{a.gca}.pkl")
res["abl_uniform"] = {"err": "no_sol"} if yv is None else profile(yv > 0.5, "abl_uniform")

json.dump(res, open(OUT, "w"))
print("written", OUT, flush=True)
