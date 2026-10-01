"""Which uptakes does the METEOR E. coli model need beyond GS_MM_glc? Read-only inputs; writes results/diag_medium.json."""
import sys, json, pickle, numpy as np
R = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision"; sys.path.insert(0, f"{R}/eval")
from _env import *
from meteor_v8.utils import load_universal, extract_fba_matrices, load_tight_bounds, apply_media, data_path, build_submodel
import warnings, logging; warnings.filterwarnings("ignore"); logging.getLogger("cobra").setLevel(logging.ERROR)
GLC = {"cpd00001","cpd00007","cpd00009","cpd00027","cpd00030","cpd00034","cpd00048","cpd00058","cpd00063","cpd00067","cpd00099","cpd00149","cpd00205","cpd00254","cpd00531","cpd00971","cpd01012","cpd01048","cpd10515","cpd10516","cpd11595","cpd00013","cpd00011","cpd11574"}
gca = "GCF_058436375.1"; universal, allrxns, _ = load_universal()
for x in list(universal.reactions)+list(universal.metabolites): 
    if not hasattr(x,"_annotation"): x._annotation = {}
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lt, ut = load_tight_bounds(data_path("tight_bounds_v6_neg.pkl")); lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
lb, ub, _, _ = apply_media(["default"], allrxns, lb, ub); ix = {r: i for i, r in enumerate(allrxns)}
sol = pickle.load(open(f"{RUNS}/dpz_vanilla/meteor_sol_{gca}.pkl", "rb")); act = np.array(sol["y_vals"]) > 0.5
m = build_submodel(universal, [r for r, k in zip(universal.reactions, act) if k], gca, biomass_id="biomass_GmNeg"); m.objective = "biomass_GmNeg"; m.solver = "glpk"
for r in m.reactions:
    if r.id.startswith("EX_") or "biomass" in r.id.lower(): continue
    i = ix.get(r.id); r.lower_bound = max(-100, float(lb[i])); r.upper_bound = min(100, float(ub[i]))
exs = [r for r in m.reactions if r.id.startswith("EX_") and r.id != "EX_biomass"]
names = {mt.id: mt.name for mt in m.metabolites}
for r in exs: r.bounds = (-1000, 1000)
so = m.optimize(); up = {r.id for r in exs if so.fluxes[r.id] < -1e-6}
# stated-medium ('default' minimal_uptake) exchanges present in the model
stated = {r.id for r in exs if lb[ix[r.id]] < 0}
# minimal extra set: start from glc_min, greedily add stated-medium uptakes until growth
def growth(allowed):
    for r in exs: r.bounds = (-10.0 if r.id[3:-2] in allowed else 0.0, 1000.0)
    return m.slim_optimize() or 0.0
out = dict(n_exchanges=len(exs), posthoc_uptake=sorted(up), posthoc_not_in_glc_min=sorted(f"{u} {names.get(u[3:], '')}" for u in up if u[3:-2] not in GLC),
           stated_medium_ex_in_model=len(stated), growth_glc_min=growth(GLC), growth_stated=growth({r[3:-2] for r in stated}))
# which glc_min compounds have no EX in the model at all
out["glc_min_missing_ex"] = sorted(c for c in GLC if f"EX_{c}_e" not in {r.id for r in exs})
# greedy: add one stated-medium compound at a time on top of glc_min
extra = []; cur = set(GLC)
cands = [r.id[3:-2] for r in exs if r.id in stated and r.id[3:-2] not in GLC]
for c in cands:
    if growth(cur | {c}) > 1e-6: extra.append((c, names.get(c + "_e", ""), growth(cur | {c})))
out["single_additions_that_rescue_glc_min"] = extra
print(json.dumps(out, indent=1)); json.dump(out, open(f"{R}/feasibility_essentiality/results/diag_medium.json", "w"), indent=1)
