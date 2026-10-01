"""Round 11: mechanistic trace of why METEOR wrongly calls purine/pyrimidine
de novo synthesis genes essential while threshold correctly calls them
non-essential (Salmonella/dpz, group_a genes from forensic.py's deepdive).

Reproduces essentiality_correct_medium.py's exact MILP (meteor) and
threshold+gapfill (thresh) reaction selection for organism=salmonella,
baseline=dpz -- but this time SAVES the full boolean keep_flags (y-vector)
for both arms, then traces network-level connectivity for the salvage
pathway and its immediate neighbor reactions to find the actual
differentiator between the two arms' selected networks.

Writes only under psb_revision/feasibility_essentiality/results/forensic/.
"""
import os, sys, json, time
import numpy as np
warnings_off = True
import warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("cobra").setLevel(logging.ERROR)

R = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision"
HERE = f"{R}/feasibility_essentiality"
OUT = f"{HERE}/results/forensic"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{R}/eval")
from _env import *  # noqa
from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds,
    find_excluded_reactions, build_rxn_ec_mask, extract_pred, load_refmapping, load_ec,
    data_path, data_dir, compute_costs, build_candidate_mask, _detect_solver, build_submodel)
from meteor_v8.milp_hard import biomass_feasible_skeleton
from meteor_v8.milp_v8 import build_milp_v8
from meteor_v8.repair import verify_and_repair, grow_support
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
import pulp

MILP_FLAGS = dict(mu=3.0, pexp=2.0, eps=0.0, gmin=0.1, wmin=0.01, gmax=2.5, lam=1e-4)
LB_MARINOS = {"cpd00971","cpd00099","cpd00063","cpd10515","cpd10516","cpd00048","cpd00009",
              "cpd00254","cpd00205","cpd00035","cpd00041","cpd00132","cpd00023","cpd00053",
              "cpd00033","cpd00119","cpd00322","cpd00107","cpd00039","cpd00060","cpd00066",
              "cpd00129","cpd00054","cpd00161","cpd00065","cpd00069","cpd00156","cpd00051",
              "cpd00013","cpd00028","cpd00084","cpd00104","cpd00166","cpd00393","cpd00419",
              "cpd00305","cpd00220","cpd00027","cpd00007","cpd00011","cpd00001","cpd00067",
              "cpd11595","cpd00030","cpd00034","cpd00058","cpd00149","cpd00531","cpd01012",
              "cpd00224","cpd00138","cpd00158","cpd00128","cpd00207","cpd00307","cpd00092",
              "cpd00018","cpd00126","cpd00046","cpd00091","cpd00557","cpd00644"}
GCA = "GCF_000006945.2"; GRAM = "negative"; BASELINE = "dpz"; VARIANT = "vanilla"
CUTOFF = 0.5

def apply_strict_medium(allrxns, lb, ub, cpd_set):
    lb = lb.copy(); ub = ub.copy(); allowed = {"EX_" + c + "_e" for c in cpd_set}; media_rxns = set()
    for i, r in enumerate(allrxns):
        if not r.startswith("EX_"): continue
        if r in allowed: lb[i] = -100.0; ub[i] = 100.0; media_rxns.add(i)
        else: lb[i] = 0.0; ub[i] = 1000.0
    return lb, ub, media_rxns

T0 = time.time()
def tick(k): print(f"[{round(time.time()-T0,1):7.1f}s] {k}", flush=True)

bid = "biomass_GmNeg"
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, allmet = load_universal(); anc = load_ec(data_path("all_ancestors.txt"))
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
pred = extract_pred(resolve_baseline_pkl(BASELINE, VARIANT, GCA, BASELINE_SUFFIX[BASELINE]), anc)
prots = [str(p).split()[0] for p in pred.index]; P = pred.values
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
S, lb0, ub0 = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{GRAM[:3]}.pkl"))
if lt is not None: lb0 = np.maximum(lb0, lt); ub0 = np.minimum(ub0, ut)
obj_idx = allrxns.index(bid)
excludes = find_excluded_reactions(S, lb0, ub0, allrxns, bid)
lb, ub, media_rxns = apply_strict_medium(allrxns, lb0, ub0, LB_MARINOS)
ix = {rid: i for i, rid in enumerate(allrxns)}
solver, _ = _detect_solver(threads=4, time_limit=250)
tick("loaded universal + evidence + medium")

def gpr_for(j):
    ei = np.where(mask[j] == 1)[0]
    if len(ei) == 0: return []
    sc = P[:, ei].max(axis=1)
    return [prots[i] for i in np.where(sc >= CUTOFF)[0]]

# ---- METEOR arm ----
feas, skel, bm_max = biomass_feasible_skeleton(S, lb, ub, obj_idx, MILP_FLAGS["gmin"], solver=solver)
tick(f"skeleton feasible={feas} bm_max={bm_max:.2f}")
P5 = P.copy(); ne = P5.shape[1]
if ne > 5:
    dr = np.argpartition(P5, ne - 5, axis=1)[:, :ne - 5]; np.put_along_axis(P5, dr, 0.0, axis=1)
l1m = np.log(np.clip(1.0 - P5, 1e-9, 1.0)); w = np.zeros(len(allrxns), dtype=np.float32)
for j in range(len(allrxns)):
    ei = np.where(mask[j] == 1)[0]
    if len(ei): w[j] = 1.0 - np.exp(float(l1m[:, ei].sum()))
w = np.clip(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 1e-6, 1 - 1e-6)
c = np.asarray(compute_costs(w, mode="logodds")["c"], dtype=float) + MILP_FLAGS["mu"] * np.power(np.clip(1.0 - w, 0, 1), MILP_FLAGS["pexp"])
cand = build_candidate_mask(w, allrxns, excludes, media_rxns, essential_skeleton=skel, w_min=MILP_FLAGS["wmin"])
m_, y, vp, vn, _ = build_milp_v8(S, lb, ub, c, obj_idx, excludes, media_rxns, cand, MILP_FLAGS["gmin"], MILP_FLAGS["gmax"], MILP_FLAGS["lam"], mu=0.0, eps=MILP_FLAGS["eps"])
t0 = time.time(); m_.solve(solver); solve_sec = round(time.time() - t0, 2)
status = pulp.LpStatus[m_.status]
yv = np.array([y[j].value() or 0 for j in range(len(y))]); vv = np.array([(vp[j].value() or 0) - (vn[j].value() or 0) for j in range(len(y))])
tick(f"MILP solved status={status} solve_sec={solve_sec}")
yv2, n_active, n_repaired, mb_core = verify_and_repair(S, lb, ub, obj_idx, yv, vv, cand)
meteor_keep = (yv2 > 0.5)

# ---- threshold arm ----
ecmax = P.max(axis=0); hot = set(np.where(ecmax >= 0.5)[0]); NR = len(allrxns)
draft = np.zeros(NR, bool)
for j in range(NR):
    ei = np.where(mask[j] == 1)[0]
    if len(ei) and any(e in hot for e in ei): draft[j] = True
avail = np.ones(NR, bool); avail[list(excludes)] = False; core = draft & avail
sup = grow_support(S, lb, ub, obj_idx, avail, MILP_FLAGS["gmin"], core=core)
if sup is None: sup = grow_support(S, lb, ub, obj_idx, np.ones(NR, bool), MILP_FLAGS["gmin"], core=core)
thresh_keep = core | (sup if sup is not None else np.zeros(NR, bool))
tick(f"thresh built, n_gapfilled={(thresh_keep & ~core).sum()}")

# ---- save the y-vectors (this is the missing artifact) ----
np.savez(f"{OUT}/yvectors_salmonella_dpz.npz",
         allrxns=np.array(allrxns, dtype=object), meteor_keep=meteor_keep, thresh_keep=thresh_keep)
tick("y-vectors saved")

# ---- salvage pathway reactions: find by name keyword in universal ----
KEYWORDS = ["hypoxanthine phosphoribosyltransferase", "xanthine phosphoribosyltransferase",
            "adenine phosphoribosyltransferase", "uracil phosphoribosyltransferase",
            "purine-nucleoside phosphorylase", "purine nucleoside phosphorylase",
            "nucleoside permease", "xanthine permease", "uracil permease", "adenine permease",
            "hypoxanthine permease", "purine permease"]
salvage_idx = {}
for r in universal.reactions:
    nm = (r.name or "").lower()
    for kw in KEYWORDS:
        if kw in nm:
            salvage_idx.setdefault(kw, []).append(r.id)

print("\n=== salvage-pathway reactions found by name, keep-flag per arm ===")
salvage_report = {}
for kw, rids in salvage_idx.items():
    for rid in rids:
        i = ix.get(rid)
        if i is None: continue
        row = dict(keyword=kw, rxn=rid, in_meteor=bool(meteor_keep[i]), in_thresh=bool(thresh_keep[i]))
        salvage_report[rid] = row
        print(row)

# ---- neighbor trace: for each salvage reaction, find other reactions sharing
# a metabolite, and flag any neighbor present in thresh but absent in meteor
# (candidate "missing link" reactions) ----
met_ids = [m.id for m in universal.metabolites]
mix = {mid: k for k, mid in enumerate(met_ids)}
neighbor_report = {}
for rid, row in salvage_report.items():
    if not (row["in_meteor"] and row["in_thresh"]):
        continue
    j = ix[rid]
    col = S[:, j] if S.shape[1] == len(allrxns) else None
    if col is None:
        continue
    met_rows = np.where(np.abs(col) > 1e-9)[0]
    diffs = []
    for mr in met_rows:
        touching = np.where(np.abs(S[mr, :]) > 1e-9)[0]
        for tj in touching:
            if tj == j: continue
            only_in_thresh = bool(thresh_keep[tj]) and not bool(meteor_keep[tj])
            only_in_meteor = bool(meteor_keep[tj]) and not bool(thresh_keep[tj])
            if only_in_thresh or only_in_meteor:
                diffs.append(dict(rxn=allrxns[tj], only_in_thresh=only_in_thresh, only_in_meteor=only_in_meteor,
                                   shared_met=met_ids[mr]))
    if diffs:
        neighbor_report[rid] = diffs

print("\n=== neighbor reactions of salvage rxns that DIFFER between arms ===")
for rid, diffs in neighbor_report.items():
    print(rid, "->", len(diffs), "differing neighbors")
    for d in diffs[:20]:
        print("   ", d)

json.dump(dict(salvage_report=salvage_report, neighbor_report=neighbor_report,
               n_meteor_selected=int(meteor_keep.sum()), n_thresh_selected=int(thresh_keep.sum()),
               n_only_meteor=int((meteor_keep & ~thresh_keep).sum()),
               n_only_thresh=int((thresh_keep & ~meteor_keep).sum())),
          open(f"{OUT}/yvector_trace_salmonella_dpz.json", "w"), indent=1)
print("\n-> ", f"{OUT}/yvector_trace_salmonella_dpz.json")
print("-> ", f"{OUT}/yvectors_salmonella_dpz.npz")
