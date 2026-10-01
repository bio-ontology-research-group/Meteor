"""One-off sanity check: is fba_growth_under_this_medium==0 (despite MILP
milp_biomass_flux>0) an artifact of assigning matrix lb/ub directly to EX
reactions without accounting for the reversed_trans EX sign flip in
extract_fba_matrices? Rebuilds the GCF_058436375.1/dpz/GS_MM_glc selected
model with EX bounds sign-corrected (cobra_lb = -ub_matrix, cobra_ub =
-lb_matrix for EX reactions except EX_biomass) and compares slim_optimize()
against the naive (buggy, as used in verify_selected_growth.py and my own
resolve_minmed.py) assignment. Read-only; no files modified outside stdout.
"""
import sys, json, pickle, numpy as np
R = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision"; sys.path.insert(0, f"{R}/eval")
from _env import *
from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds,
    find_excluded_reactions, build_submodel, data_path)
import warnings, logging; warnings.filterwarnings("ignore"); logging.getLogger("cobra").setLevel(logging.ERROR)

GS_MM_GLC = {"cpd00001","cpd00007","cpd00009","cpd00027","cpd00030","cpd00034","cpd00048",
             "cpd00058","cpd00063","cpd00067","cpd00099","cpd00149","cpd00205","cpd00254",
             "cpd00531","cpd00971","cpd01012","cpd01048","cpd10515","cpd10516","cpd11595",
             "cpd00013","cpd00011","cpd11574"}
gca = "GCF_058436375.1"; gram = "negative"; bid = "biomass_GmNeg"
universal, allrxns, allmet = load_universal()
for x in list(universal.reactions) + list(universal.metabolites):
    if not hasattr(x, "_annotation"): x._annotation = {}
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{gram[:3]}.pkl")); lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
allowed = {"EX_" + c + "_e" for c in GS_MM_GLC}
for i, r in enumerate(allrxns):
    if not r.startswith("EX_"): continue
    if r in allowed: lb[i] = -100.0; ub[i] = 100.0
    else: lb[i] = 0.0; ub[i] = 1000.0
ix = {rid: i for i, rid in enumerate(allrxns)}

# use the dpz-combo saved y vector from this run
d = json.load(open(f"{R}/feasibility_essentiality/results/minmed/dpz_vanilla_{gca}_GS_MM_glc.json"))
print("MILP said milp_biomass_flux =", d["milp_biomass_flux"], "n_selected_after_repair =", d["n_selected_after_repair"])

# reconstruct which reactions were selected isn't saved (no y-pkl for this script yet) -- rerun the same
# candidate/cost/MILP quickly is expensive; instead just verify the SIGN-FIX HYPOTHESIS on the FULL
# universal-under-medium skeleton FBA (bm_max reported =250 for the unconstrained skeleton LP, so any
# selected submodel that is a superset of the skeleton flux support should also grow under a matching
# cobra reconstruction if bounds are assigned correctly). Build cobra model = ALL non-excluded reactions
# (i.e. "nomask" arm), apply BOTH naive and sign-fixed bounds, compare growth.
obj_idx = allrxns.index(bid)
excludes = find_excluded_reactions(S, lb, ub, allrxns, bid)
keep = [r for j, r in enumerate(universal.reactions) if j not in excludes]
m = build_submodel(universal, keep, gca, biomass_id=bid); m.objective = bid; m.solver = "glpk"
for r in m.reactions:
    if r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower(): continue
    i = ix.get(r.id)
    if i is not None: r.lower_bound = float(lb[i]); r.upper_bound = float(ub[i])
exs = [r for r in m.reactions if r.id.startswith("EX_") and r.id != "EX_biomass"]
# naive (as in verify_selected_growth.py / resolve_minmed.py)
for r in exs:
    i = ix.get(r.id)
    if i is not None: r.lower_bound = float(lb[i]); r.upper_bound = float(ub[i])
naive = m.slim_optimize()
print("naive (matrix-bounds-as-is) growth on FULL non-excluded network:", naive)
# sign-corrected: cobra_lb = -ub_matrix, cobra_ub = -lb_matrix (reversed_trans flips EX stoich sign)
for r in exs:
    i = ix.get(r.id)
    if i is not None: r.lower_bound = -float(ub[i]); r.upper_bound = -float(lb[i])
fixed = m.slim_optimize()
print("sign-corrected growth on FULL non-excluded network:", fixed)
print("skeleton LP bm_max (ground truth, matrix frame) reported earlier: 250.0")
