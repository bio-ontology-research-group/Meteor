"""Round 13: re-run essentiality (METEOR + threshold) for the 9 organism x
predictor combinations where threshold beat METEOR, with a domain filter
excluding eukaryote-only-annotated reactions (Mitochondrial/Golgi/Nucleus/
Peroxisome/Lysosome/Endoplasmic/Chloroplast/Vacuole/Vesicle -- same keyword
list as yvector_all9.py) from BOTH arms' candidate reaction pools, to test
whether closing this candidate-pool artifact narrows or reverses the
METEOR-vs-threshold gap. Otherwise identical methodology to
essentiality_correct_medium.py (same MILP flags, same medium, same GPR
construction, same reference datasets).

Writes only under psb_revision/feasibility_essentiality/results/domainfilter/.
"""
import os, sys, json, time, warnings, logging
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); logging.getLogger("cobra").setLevel(logging.ERROR)
R = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision"
sys.path.insert(0, f"{R}/eval")
from _env import *  # noqa
from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds,
    find_excluded_reactions, build_rxn_ec_mask, extract_pred, load_refmapping, load_ec,
    data_path, data_dir, compute_costs, build_candidate_mask, _detect_solver, build_submodel)
from meteor_v8.milp_hard import biomass_feasible_skeleton
from meteor_v8.milp_v8 import build_milp_v8
from meteor_v8.repair import verify_and_repair, grow_support
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
import cobra, pulp
from cobra.flux_analysis import single_gene_deletion

HERE = f"{R}/feasibility_essentiality"
OUT = f"{HERE}/results/domainfilter"; os.makedirs(OUT, exist_ok=True)
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
EUK_KEYWORDS = ["mitochondrial", "golgi", "nucleus", "peroxisome", "lysosome",
                "endoplasmic", "chloroplast", "vacuole", "vesicle", "nuclear"]

ORG = {
    "salmonella": dict(gca="GCF_000006945.2", gram="negative",
        ref_csv=f"{HERE}/ref/salmonella/salmonella_binary.csv",
        ref_gene_col="gene", map_tsv=f"{HERE}/ref/salmonella_map/lt2_to_sl1344.tsv"),
    "kpneumoniae": dict(gca="GCF_058435815.1", gram="negative",
        ref_csv=f"{HERE}/ref/kpneumoniae/kpneumoniae_ecl8_binary.csv",
        ref_gene_col="gene", map_tsv=f"{HERE}/ref/kpneumoniae_map/kp0179_to_ecl8.tsv"),
    "pputida": dict(gca="GCF_045571375.1", gram="negative",
        ref_csv=f"{HERE}/ref/pputida/pputida_binary.csv",
        ref_gene_col="gene", map_tsv=f"{HERE}/ref/pputida_map/panel_to_pp.tsv"),
}
PREDICTORS = ["clean", "dpz", "enzbert"]
CUTOFF = 0.5

def apply_strict_medium(allrxns, lb, ub, cpd_set):
    lb = lb.copy(); ub = ub.copy(); allowed = {"EX_" + c + "_e" for c in cpd_set}; media_rxns = set()
    for i, r in enumerate(allrxns):
        if not r.startswith("EX_"): continue
        if r in allowed: lb[i] = -100.0; ub[i] = 100.0; media_rxns.add(i)
        else: lb[i] = 0.0; ub[i] = 1000.0
    return lb, ub, media_rxns

def metrics(pred_ess, ref_ess, mapper):
    tp = fp = fn = tn = 0; n_model = len(pred_ess); n_mapped = 0; n_ref = 0
    for g, e in pred_ess.items():
        b = mapper(g)
        if b is None: continue
        n_mapped += 1
        if b not in ref_ess: continue
        n_ref += 1; t = ref_ess[b]
        if e and t: tp += 1
        elif e and not t: fp += 1
        elif (not e) and t: fn += 1
        else: tn += 1
    P = tp / max(1, tp + fp); Rc = tp / max(1, tp + fn)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / den if den > 0 else 0.0
    return dict(n_genes_model=n_model, n_genes_mapped=n_mapped, n_genes_with_ref=n_ref,
                n_pred_essential=int(sum(pred_ess.values())), TP=tp, FP=fp, FN=fn, TN=tn,
                precision=round(P, 3), recall=round(Rc, 3), MCC=round(mcc, 3),
                accuracy=round((tp + tn) / max(1, n_ref), 3))

def run_sgd(model, label, procs=4):
    wt = model.slim_optimize(); print(f"  {label}: WT growth={wt:.4f}, n_rxn={len(model.reactions)}, n_genes={len(model.genes)}", flush=True)
    if not (wt > 1e-6): return None, wt
    sgd = single_gene_deletion(model, processes=procs)
    ess = {}
    for ids, gr, st in zip(sgd["ids"], sgd["growth"], sgd["status"]):
        g = next(iter(ids)); ess[g] = bool((not np.isfinite(gr)) or gr < 0.05 * wt or st != "optimal")
    return ess, wt

T0 = time.time()
def tick(k): print(f"[{round(time.time()-T0,1):7.1f}s] {k}", flush=True)

seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, allmet = load_universal(); anc = load_ec(data_path("all_ancestors.txt"))
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
ix = {rid: i for i, rid in enumerate(allrxns)}

euk_flag = np.zeros(len(allrxns), dtype=bool)
for j, r in enumerate(universal.reactions):
    nm = (r.name or "").lower()
    if any(kw in nm for kw in EUK_KEYWORDS): euk_flag[j] = True
euk_idx = set(np.where(euk_flag)[0].tolist())
tick(f"universal loaded, {len(euk_idx)} eukaryote-flagged reactions to exclude from candidate pool")

all_results = {}
for org, cfg in ORG.items():
    gca = cfg["gca"]; gram = cfg["gram"]
    bid = "biomass_GmPos" if gram == "positive" else "biomass_GmNeg"
    ref = pd.read_csv(cfg["ref_csv"])
    ref_ess = {str(g): (e == "yes") for g, e in zip(ref[cfg["ref_gene_col"]], ref["ess.experimental"])}
    p2ref = {}
    for l in open(cfg["map_tsv"]):
        parts = l.rstrip("\n").split("\t")
        if len(parts) >= 3: p2ref[parts[0]] = parts[2]

    S, lb0, ub0 = extract_fba_matrices(universal, allrxns, reversed_trans=True)
    lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{gram[:3]}.pkl"))
    if lt is not None: lb0 = np.maximum(lb0, lt); ub0 = np.minimum(ub0, ut)
    obj_idx = allrxns.index(bid)
    excludes_base = find_excluded_reactions(S, lb0, ub0, allrxns, bid)
    excludes = set(excludes_base) | euk_idx  # <-- the domain filter
    lb, ub, media_rxns = apply_strict_medium(allrxns, lb0, ub0, LB_MARINOS)
    solver, _ = _detect_solver(threads=4, time_limit=250)

    for pred_name in PREDICTORS:
        key = f"{org}_{pred_name}"
        t0 = time.time()
        pred = extract_pred(resolve_baseline_pkl(pred_name, "vanilla", gca, BASELINE_SUFFIX[pred_name]), anc)
        prots = [str(p).split()[0] for p in pred.index]; P = pred.values

        def gpr_for(j):
            ei = np.where(mask[j] == 1)[0]
            if len(ei) == 0: return []
            sc = P[:, ei].max(axis=1)
            return [prots[i] for i in np.where(sc >= CUTOFF)[0]]

        def build_model_with_gpr(keep_flags, label):
            keep = [r for r, k in zip(universal.reactions, keep_flags) if k]
            m = build_submodel(universal, keep, label, biomass_id=bid); m.objective = bid
            for r in m.reactions:
                i = ix.get(r.id); r.gene_reaction_rule = ""
                if r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower(): continue
                if i is not None:
                    genes = gpr_for(i)
                    if genes: r.gene_reaction_rule = " or ".join(genes)
            for r in m.reactions:
                if r.id.startswith("EX_") and r.id != "EX_biomass":
                    i = ix.get(r.id)
                    if i is not None: r.lower_bound = -float(ub[i]); r.upper_bound = -float(lb[i])
                elif not (r.id.startswith(("DM_", "SK_")) or "biomass" in r.id.lower()):
                    i = ix.get(r.id)
                    if i is not None: r.lower_bound = float(lb[i]); r.upper_bound = float(ub[i])
            m.solver = "glpk"
            return m

        # ---- METEOR, with domain-filtered excludes ----
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
        m_meteor = build_model_with_gpr(meteor_keep, "meteor")
        ess_m, wt_m = run_sgd(m_meteor, f"{key}/meteor")
        row_m = dict(n_selected=int(meteor_keep.sum()), wt_growth=round(float(wt_m), 4))
        row_m.update(metrics(ess_m, ref_ess, p2ref.get) if ess_m is not None else {"err": "no growth"})

        # ---- threshold, with domain-filtered excludes ----
        ecmax = P.max(axis=0); hot = set(np.where(ecmax >= 0.5)[0]); NR = len(allrxns)
        draft = np.zeros(NR, bool)
        for j in range(NR):
            ei = np.where(mask[j] == 1)[0]
            if len(ei) and any(e in hot for e in ei): draft[j] = True
        avail = np.ones(NR, bool); avail[list(excludes)] = False; core = draft & avail
        sup = grow_support(S, lb, ub, obj_idx, avail, MILP_FLAGS["gmin"], core=core)
        if sup is None: sup = grow_support(S, lb, ub, obj_idx, np.ones(NR, bool) & ~euk_flag, MILP_FLAGS["gmin"], core=core)
        thresh_keep = core | (sup if sup is not None else np.zeros(NR, bool))
        m_thresh = build_model_with_gpr(thresh_keep, "thresh")
        ess_t, wt_t = run_sgd(m_thresh, f"{key}/thresh")
        row_t = dict(n_selected=int(thresh_keep.sum()), wt_growth=round(float(wt_t), 4))
        row_t.update(metrics(ess_t, ref_ess, p2ref.get) if ess_t is not None else {"err": "no growth"})

        all_results[key] = dict(meteor=row_m, thresh=row_t, elapsed_s=round(time.time()-t0, 1))
        print(key, "meteor:", row_m.get("MCC"), "thresh:", row_t.get("MCC"), flush=True)
        json.dump(all_results, open(f"{OUT}/domainfilter_summary.json", "w"), indent=1)

print("\n=== FINAL SUMMARY (domain-filtered candidate pool) ===")
for k, v in all_results.items():
    print(f"{k:24s} meteor MCC={v['meteor'].get('MCC')}  thresh MCC={v['thresh'].get('MCC')}")
print("\n-> ", f"{OUT}/domainfilter_summary.json")
