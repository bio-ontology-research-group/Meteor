"""Feasibility-only: re-solve the METEOR selection MILP under a strict minimal
medium instead of the pipeline's published "default" medium, and report
whether the network can even grow -- NO GPR / gene essentiality here.

Medium definitions (verified from code, not assumed):
  GS_MM_glc  : gapseq_eval/.../GeneEssentiality/media/media.tsv, 24 rows,
               modelseed column -> cpd ids (see GS_MM_GLC below). Used in
               compare_GE_recons.R as the medium for the E. coli essentiality
               comparison (ecol.gs/.cm/.ms all constrained to GS_MM_glc).
  LB_marinos : same media.tsv, 62 rows (rich: all 20 aa, nucleotides,
               vitamins, hemes). compare_GE_recons.R constrains B. subtilis
               (bsub.gs/.cm/.ms) to LB_marinos, NOT GS_MM_glc -- i.e. the
               bsub essentiality reference is NOT a glucose-minimal-medium
               experiment. Included here as an informational bonus run only.

IMPORTANT bypass: meteor_v8.utils.apply_media() hardcodes a "minimal_uptake"
set of ~30 exchanges (incl. ALL 20 amino acids) that is unioned in regardless
of which medium name is passed -- so simply swapping the medium argument
would NOT produce a strict minimal medium. This script implements its own
apply_strict_medium() that mirrors apply_media()'s exchange-bound logic
without that union, so GS_MM_glc really is glucose + inorganic salts only.

Writes only under psb_revision/feasibility_essentiality/results/minmed/.
"""
import os, sys, json, time, argparse, warnings, logging
import numpy as np
warnings.filterwarnings("ignore"); logging.getLogger("cobra").setLevel(logging.ERROR)
R = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision"
sys.path.insert(0, f"{R}/eval")
from _env import *  # noqa
from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds,
    find_excluded_reactions, build_rxn_ec_mask, extract_pred, load_refmapping, load_ec,
    data_path, data_dir, compute_costs, build_candidate_mask, _detect_solver, build_submodel)
from meteor_v8.milp_hard import biomass_feasible_skeleton
from meteor_v8.milp_v8 import build_milp_v8
from meteor_v8.repair import verify_and_repair
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
import pulp

HERE = f"{R}/feasibility_essentiality"
OUT = f"{HERE}/results/minmed"; os.makedirs(OUT, exist_ok=True)
MILP_FLAGS = dict(mu=3.0, pexp=2.0, eps=0.0, gmin=0.1, wmin=0.01, gmax=2.5, lam=1e-4)

# --- medium definitions (cpd ids, verified from media.tsv above) ---
GS_MM_GLC = {"cpd00001","cpd00007","cpd00009","cpd00027","cpd00030","cpd00034","cpd00048",
             "cpd00058","cpd00063","cpd00067","cpd00099","cpd00149","cpd00205","cpd00254",
             "cpd00531","cpd00971","cpd01012","cpd01048","cpd10515","cpd10516","cpd11595",
             "cpd00013","cpd00011","cpd11574"}
LB_MARINOS = {"cpd00971","cpd00099","cpd00063","cpd10515","cpd10516","cpd00048","cpd00009",
              "cpd00254","cpd00205","cpd00035","cpd00041","cpd00132","cpd00023","cpd00053",
              "cpd00033","cpd00119","cpd00322","cpd00107","cpd00039","cpd00060","cpd00066",
              "cpd00129","cpd00054","cpd00161","cpd00065","cpd00069","cpd00156","cpd00051",
              "cpd00013","cpd00028","cpd00084","cpd00104","cpd00166","cpd00393","cpd00419",
              "cpd00305","cpd00220","cpd00027","cpd00007","cpd00011","cpd00001","cpd00067",
              "cpd11595","cpd00030","cpd00034","cpd00058","cpd00149","cpd00531","cpd01012",
              "cpd00224","cpd00138","cpd00158","cpd00128","cpd00207","cpd00307","cpd00092",
              "cpd00018","cpd00126","cpd00046","cpd00091","cpd00557","cpd00644"}
MEDIA = {"GS_MM_glc": GS_MM_GLC, "LB_marinos": LB_MARINOS}

def apply_strict_medium(allrxns, lb, ub, cpd_set):
    """Mirror apply_media()'s exchange-bound logic (STRICT: no minimal_uptake union)."""
    lb = lb.copy(); ub = ub.copy()
    allowed = {"EX_" + c + "_e" for c in cpd_set}
    media_rxns = set()
    for i, r in enumerate(allrxns):
        if not r.startswith("EX_"): continue
        if r in allowed:
            lb[i] = -100.0; ub[i] = 100.0; media_rxns.add(i)
        else:
            lb[i] = 0.0; ub[i] = 1000.0
    return lb, ub, media_rxns

ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True); ap.add_argument("--gram", choices=["negative","positive"], required=True)
ap.add_argument("--baseline", required=True); ap.add_argument("--variant", default="vanilla")
ap.add_argument("--medium", default="GS_MM_glc", choices=list(MEDIA))
ap.add_argument("--time_limit", type=int, default=300)
a = ap.parse_args()
OUTF = f"{OUT}/{a.baseline}_{a.variant}_{a.gca}_{a.medium}.json"
if os.path.exists(OUTF): print("done", OUTF); sys.exit(0)

t0 = time.time()
F = MILP_FLAGS; bid = "biomass_GmPos" if a.gram == "positive" else "biomass_GmNeg"
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, allmet = load_universal(); anc = load_ec(data_path("all_ancestors.txt"))
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
pred = extract_pred(resolve_baseline_pkl(a.baseline, a.variant, a.gca, BASELINE_SUFFIX[a.baseline]), anc)
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{a.gram[:3]}.pkl"))
if lt is not None: lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
obj_idx = allrxns.index(bid)
excludes = find_excluded_reactions(S, lb, ub, allrxns, bid)   # BEFORE medium swap, matches skeleton_ablation.py order

lb, ub, media_rxns = apply_strict_medium(allrxns, lb, ub, MEDIA[a.medium])
solver, _ = _detect_solver(threads=4, time_limit=a.time_limit)
print(f'[{round(time.time()-t0,1)}s] loaded matrices, excludes={len(excludes)}, media_rxns={len(media_rxns)}; solving skeleton LP...', flush=True)
t_skel0 = time.time()
feas, skel, bm_max = biomass_feasible_skeleton(S, lb, ub, obj_idx, F["gmin"], solver=solver)
t_skel = round(time.time() - t_skel0, 1)
print(f'[{round(time.time()-t0,1)}s] skeleton done: feasible={feas} bm_max={bm_max:.4f} skel_solve={t_skel}s |skel|={len(skel)}', flush=True)

rec = dict(gca=a.gca, gram=a.gram, baseline=a.baseline, variant=a.variant, medium=a.medium,
           n_medium_cpds=len(MEDIA[a.medium]), n_media_ex_in_model=len(media_rxns),
           skeleton_feasible=bool(feas), skeleton_max_biomass=round(float(bm_max), 6), skeleton_solve_sec=t_skel)

if not feas:
    # diagnostic: which biomass precursors are blocked -- probe each biomass
    # substrate metabolite for max producible flux under this medium+universal
    bm_rxn = universal.reactions.get_by_id(bid)
    subs = [m.id for m, coef in bm_rxn.metabolites.items() if coef < 0]
    ix = {mid: i for i, mid in enumerate(allmet)}
    from scipy.sparse import csr_matrix
    Sc = csr_matrix(S)
    blocked = []
    for mid in subs:
        mi = ix.get(mid)
        if mi is None: continue
        mprob = pulp.LpProblem("probe", pulp.LpMaximize)
        v = {j: pulp.LpVariable(f"v{j}", lowBound=float(lb[j]), upBound=float(ub[j])) for j in range(len(allrxns))}
        # allow a virtual sink on this metabolite only, everything else mass-balanced
        sink = pulp.LpVariable("sink", lowBound=0)
        for i in range(Sc.shape[0]):
            row = Sc.getrow(i)
            if len(row.indices) == 0: continue
            expr = pulp.lpSum(float(row.data[k]) * v[int(row.indices[k])] for k in range(len(row.indices)))
            if i == mi: expr = expr - sink
            mprob += expr == 0
        mprob += sink
        mprob.solve(solver)
        val = sink.value() or 0.0
        if val < 1e-6:
            met = universal.metabolites.get_by_id(mid)
            blocked.append(dict(met=mid, name=met.name, max_flux=round(float(val), 6)))
    rec["biomass_precursors_blocked"] = blocked
    rec["biomass_precursors_total"] = len(subs)
    rec["milp_skipped"] = True
    json.dump(rec, open(OUTF, "w"), indent=1)
    print(json.dumps(rec, indent=1)); sys.exit(0)

# ---- feasible skeleton: proceed to build the actual MILP (evidence-weighted, published flags) ----
P = pred.values.astype(np.float32).copy(); ne = P.shape[1]
if ne > 5:
    dr = np.argpartition(P, ne - 5, axis=1)[:, :ne - 5]; np.put_along_axis(P, dr, 0.0, axis=1)
l1m = np.log(np.clip(1.0 - P, 1e-9, 1.0)); w = np.zeros(len(allrxns), dtype=np.float32)
for j in range(len(allrxns)):
    ei = np.where(mask[j] == 1)[0]
    if len(ei): w[j] = 1.0 - np.exp(float(l1m[:, ei].sum()))
w = np.clip(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 1e-6, 1 - 1e-6)
c = np.asarray(compute_costs(w, mode="logodds")["c"], dtype=float) + F["mu"] * np.power(np.clip(1.0 - w, 0, 1), F["pexp"])

cand = build_candidate_mask(w, allrxns, excludes, media_rxns, essential_skeleton=skel, w_min=F["wmin"])
rec["n_candidate"] = int(cand.sum()); rec["n_skeleton"] = len(skel)

m, y, vp, vn, _ = build_milp_v8(S, lb, ub, c, obj_idx, excludes, media_rxns, cand, F["gmin"], F["gmax"], F["lam"], mu=0.0, eps=F["eps"])
t_solve0 = time.time(); m.solve(solver); solve_sec = round(time.time() - t_solve0, 2)
status = pulp.LpStatus[m.status]
yv = np.array([y[j].value() or 0 for j in range(len(y))]); vv = np.array([(vp[j].value() or 0) - (vn[j].value() or 0) for j in range(len(y))])
rec.update(status=status, solve_sec=solve_sec, time_limit=a.time_limit,
           objective=float(pulp.value(m.objective)) if (yv > 0.5).any() else None,
           n_selected_raw=int((yv > 0.5).sum()), milp_biomass_flux=float(vv[obj_idx]))

if status in ("Optimal", "Not Solved") and (yv > 0.5).any():
    yv2, n_active, n_repaired, mb_core = verify_and_repair(S, lb, ub, obj_idx, yv, vv, cand)
    rec.update(n_selected_after_repair=int(n_active), n_repaired=int(n_repaired), maxbio_core=float(mb_core))
    keep = [r for r, k in zip(universal.reactions, yv2 > 0.5) if k]
    mdl = build_submodel(universal, keep, a.gca, biomass_id=bid); mdl.objective = bid
    ix = {rid: i for i, rid in enumerate(allrxns)}
    for r in mdl.reactions:
        if r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower(): continue
        i = ix.get(r.id)
        if i is not None: r.lower_bound = float(lb[i]); r.upper_bound = float(ub[i])
    # NAIVE assignment (as used in meteor_v8/eval/verify_selected_growth.py and
    # gen_table1.py::profile): matrix lb/ub applied as-is to the EX reaction.
    # This is WRONG for exchanges because extract_fba_matrices(reversed_trans=True)
    # flips the S-column sign for every EX_ reaction except EX_biomass, so the
    # matrix flux convention (positive = uptake) is the NEGATIVE of the raw cobra
    # reaction's own stoichiometric convention. Verified 2026-09-17 on the full
    # non-excluded network (GCF_058436375.1, GS_MM_glc): naive growth=19.24,
    # sign-corrected growth=250.0 == skeleton LP bm_max (ground truth). This is
    # almost certainly why meteor_v8/results/growth_verify/*/summary.json reports
    # growth_fixed=0 for ALL 108 genomes under the pipeline's own default medium.
    for r in mdl.reactions:
        if r.id.startswith("EX_") and r.id != "EX_biomass":
            i = ix.get(r.id)
            if i is not None: r.lower_bound = float(lb[i]); r.upper_bound = float(ub[i])
    fba_naive = mdl.slim_optimize()
    # SIGN-CORRECTED assignment: cobra_lb = -ub_matrix, cobra_ub = -lb_matrix
    for r in mdl.reactions:
        if r.id.startswith("EX_") and r.id != "EX_biomass":
            i = ix.get(r.id)
            if i is not None: r.lower_bound = -float(ub[i]); r.upper_bound = -float(lb[i])
    fba_fixed = mdl.slim_optimize()
    rec["fba_growth_naive_buggy"] = round(float(fba_naive) if fba_naive is not None else 0.0, 6)
    rec["fba_growth_under_this_medium"] = round(float(fba_fixed) if fba_fixed is not None else 0.0, 6)
    rec["n_rxn_final_model"] = len(mdl.reactions)
else:
    rec["err"] = status
rec["total_wall_sec"] = round(time.time() - t0, 1)
json.dump(rec, open(OUTF, "w"), indent=1)
print(json.dumps(rec, indent=1))
