"""Round 12: check whether the eukaryote-only candidate-pool artifact found in
Salmonella/DPZ (rxn34494_c/rxn34495_c, deoxynucleotide carriers labeled
"Mitochondrial") generalizes across all 9 organism x predictor combinations
where threshold beats METEOR on essentiality MCC (Salmonella, K. pneumoniae,
P. putida x CLEAN/DPZ/EnzBERT).

For each combo: reproduce essentiality_correct_medium.py's exact MILP (meteor)
and threshold+gap-fill (thresh) reaction selection, save y-vectors, then scan
ALL thresh-only reactions for eukaryote-only annotation keywords (not just the
two already found) -- Mitochondrial, Golgi, Nucleus, Peroxisome, Lysosome,
Endoplasmic, Chloroplast, Vacuole, Vesicle.

Writes only under psb_revision/feasibility_essentiality/results/forensic/.
"""
import os, sys, json, time
import numpy as np
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

ORG = {
    "salmonella": dict(gca="GCF_000006945.2", gram="negative"),
    "kpneumoniae": dict(gca="GCF_058435815.1", gram="negative"),
    "pputida": dict(gca="GCF_045571375.1", gram="negative"),
}
PREDICTORS = ["clean", "dpz", "enzbert"]
EUK_KEYWORDS = ["mitochondrial", "golgi", "nucleus", "peroxisome", "lysosome",
                "endoplasmic", "chloroplast", "vacuole", "vesicle", "nuclear"]

def apply_strict_medium(allrxns, lb, ub, cpd_set):
    lb = lb.copy(); ub = ub.copy(); allowed = {"EX_" + c + "_e" for c in cpd_set}; media_rxns = set()
    for i, r in enumerate(allrxns):
        if not r.startswith("EX_"): continue
        if r in allowed: lb[i] = -100.0; ub[i] = 100.0; media_rxns.add(i)
        else: lb[i] = 0.0; ub[i] = 1000.0
    return lb, ub, media_rxns

T0 = time.time()
def tick(k): print(f"[{round(time.time()-T0,1):7.1f}s] {k}", flush=True)

# load universal once, reused across all combos
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, allmet = load_universal(); anc = load_ec(data_path("all_ancestors.txt"))
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
ix = {rid: i for i, rid in enumerate(allrxns)}
tick("universal loaded")

# precompute euk-flagged reaction indices once
euk_flag = np.zeros(len(allrxns), dtype=bool)
euk_name = {}
for j, r in enumerate(universal.reactions):
    nm = (r.name or "").lower()
    if any(kw in nm for kw in EUK_KEYWORDS):
        euk_flag[j] = True
        euk_name[allrxns[j]] = r.name
tick(f"euk-flagged candidate reactions in universal: {int(euk_flag.sum())}")

results = {}
for org, ocfg in ORG.items():
    gca = ocfg["gca"]; gram = ocfg["gram"]
    bid = "biomass_GmPos" if gram == "positive" else "biomass_GmNeg"
    S, lb0, ub0 = extract_fba_matrices(universal, allrxns, reversed_trans=True)
    lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{gram[:3]}.pkl"))
    if lt is not None: lb0 = np.maximum(lb0, lt); ub0 = np.minimum(ub0, ut)
    obj_idx = allrxns.index(bid)
    excludes = find_excluded_reactions(S, lb0, ub0, allrxns, bid)
    lb, ub, media_rxns = apply_strict_medium(allrxns, lb0, ub0, LB_MARINOS)
    solver, _ = _detect_solver(threads=4, time_limit=250)

    for pred_name in PREDICTORS:
        key = f"{org}_{pred_name}"
        t_combo = time.time()
        pred = extract_pred(resolve_baseline_pkl(pred_name, "vanilla", gca, BASELINE_SUFFIX[pred_name]), anc)
        prots = [str(p).split()[0] for p in pred.index]; P = pred.values

        # ---- METEOR ----
        feas, skel, bm_max = biomass_feasible_skeleton(S, lb, ub, obj_idx, MILP_FLAGS["gmin"], solver=solver)
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
        m_.solve(solver)
        yv = np.array([y[j].value() or 0 for j in range(len(y))]); vv = np.array([(vp[j].value() or 0) - (vn[j].value() or 0) for j in range(len(y))])
        yv2, n_active, n_repaired, mb_core = verify_and_repair(S, lb, ub, obj_idx, yv, vv, cand)
        meteor_keep = (yv2 > 0.5)

        # ---- threshold ----
        ecmax = P.max(axis=0); hot = set(np.where(ecmax >= 0.5)[0]); NR = len(allrxns)
        draft = np.zeros(NR, bool)
        for j in range(NR):
            ei = np.where(mask[j] == 1)[0]
            if len(ei) and any(e in hot for e in ei): draft[j] = True
        avail = np.ones(NR, bool); avail[list(excludes)] = False; core = draft & avail
        mdl_raw, _ = None, None
        sup = grow_support(S, lb, ub, obj_idx, avail, MILP_FLAGS["gmin"], core=core)
        if sup is None: sup = grow_support(S, lb, ub, obj_idx, np.ones(NR, bool), MILP_FLAGS["gmin"], core=core)
        thresh_keep = core | (sup if sup is not None else np.zeros(NR, bool))

        n_only_thresh = int((thresh_keep & ~meteor_keep).sum())
        n_only_meteor = int((meteor_keep & ~thresh_keep).sum())
        euk_only_thresh = [allrxns[j] for j in np.where(euk_flag & thresh_keep & ~meteor_keep)[0]]
        euk_only_meteor = [allrxns[j] for j in np.where(euk_flag & meteor_keep & ~thresh_keep)[0]]
        euk_names_thresh = {rid: euk_name[rid] for rid in euk_only_thresh}

        row = dict(n_meteor=int(meteor_keep.sum()), n_thresh=int(thresh_keep.sum()),
                   n_only_thresh=n_only_thresh, n_only_meteor=n_only_meteor,
                   n_euk_only_thresh=len(euk_only_thresh), n_euk_only_meteor=len(euk_only_meteor),
                   euk_only_thresh_rxns=euk_names_thresh, elapsed_s=round(time.time()-t_combo,1))
        results[key] = row
        print(key, "->", row, flush=True)

        np.savez(f"{OUT}/yvectors_{key}.npz", allrxns=np.array(allrxns, dtype=object),
                 meteor_keep=meteor_keep, thresh_keep=thresh_keep)

json.dump(results, open(f"{OUT}/yvector_all9_summary.json", "w"), indent=1)
print("\n=== SUMMARY: euk-flagged thresh-only reactions per combo ===")
for k, v in results.items():
    print(f"{k:24s} n_euk_only_thresh={v['n_euk_only_thresh']:3d}  n_only_thresh={v['n_only_thresh']:4d}  {list(v['euk_only_thresh_rxns'].values())}")
print("\n-> ", f"{OUT}/yvector_all9_summary.json")
