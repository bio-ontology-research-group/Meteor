"""Feasibility demo: in-silico gene essentiality from METEOR output vs known essential genes.

GPR construction: reaction r <- proteins p with E[p, ec] >= cutoff for any ec mapped to r
(seedr2ec).  OR-only rules (isozymes); complexes (AND) cannot be inferred from E alone.
Arms: meteor (DPZ vanilla y*), thresh (EC>=0.5 draft + parsimony gap-fill, as in
baseline_thresh_gapfill.py), iml1515 (curated BiGG, real GPRs, its own default medium).
Reference: gapseq_eval essentiality.data/gess-ecol.csv (ess.experimental; iML1515 genes).
Gene mapping: strain M25631H protein -> MG1655 b-number via diamond RBH (ref/m25631h_to_bnum.tsv).
Writes only under psb_revision/feasibility_essentiality/results.
"""
import os, sys, json, pickle, re, time, argparse, logging, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); logging.getLogger("cobra").setLevel(logging.ERROR)
R = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision"
sys.path.insert(0, f"{R}/eval")
from _env import *  # noqa  (RUNS, GEMDIR, sys.path for meteor_v8 + baseline_io)
from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds, apply_media,
    find_excluded_reactions, build_rxn_ec_mask, extract_pred, load_refmapping, load_ec, data_path, data_dir,
    build_submodel)
from meteor_v8.repair import grow_support
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
import cobra
from cobra.flux_analysis import single_gene_deletion

HERE = f"{R}/feasibility_essentiality"
ap = argparse.ArgumentParser()
ap.add_argument("--gca", default="GCF_058436375.1"); ap.add_argument("--gram", default="negative")
ap.add_argument("--baseline", default="dpz"); ap.add_argument("--variant", default="vanilla")
ap.add_argument("--cutoff", type=float, default=0.5); ap.add_argument("--procs", type=int, default=4)
ap.add_argument("--arms", default="meteor,thresh,iml1515"); ap.add_argument("--gem", default="iML1515")
a = ap.parse_args()
OUT = f"{HERE}/results"; os.makedirs(OUT, exist_ok=True)
T0 = time.time(); timing = {}
def tick(k): timing[k] = round(time.time() - T0, 1); print(f"[{timing[k]:7.1f}s] {k}", flush=True)

# ---------------- reference data ----------------
ref = pd.read_csv("/ibex/scratch/projects/c2014/kexin/funcarve/gapseq_eval/gapseqEval/GeneEssentiality/essentiality.data/gess-ecol.csv")
ref_ess = {g: (e == "yes") for g, e in zip(ref["gene"], ref["ess.experimental"])}
p2b = {}
for l in open(f"{HERE}/ref/m25631h_to_bnum.tsv"):
    q, s, b = l.rstrip("\n").split("\t"); p2b[q] = b

def metrics(pred_ess: dict, label: str, mapper=lambda g: g):
    """pred_ess: gene -> bool essential (model genes). Evaluate on genes with reference data."""
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
    return dict(arm=label, n_genes_model=n_model, n_genes_mapped_bnum=n_mapped, n_genes_with_ref=n_ref,
                n_pred_essential=int(sum(pred_ess.values())), n_pred_essential_with_ref=tp + fp,
                n_ref_essential_in_eval=tp + fn, TP=tp, FP=fp, FN=fn, TN=tn,
                precision=round(P, 3), recall=round(Rc, 3), MCC=round(mcc, 3), accuracy=round((tp + tn) / max(1, n_ref), 3))

def run_sgd(model, label):
    wt = model.slim_optimize(); print(f"  {label}: WT growth={wt:.4f}, n_rxn={len(model.reactions)}, n_genes={len(model.genes)}", flush=True)
    if not (wt > 1e-6): return None, wt
    sgd = single_gene_deletion(model, processes=a.procs)
    ess = {}
    for ids, gr, st in zip(sgd["ids"], sgd["growth"], sgd["status"]):
        g = next(iter(ids)); ess[g] = bool((not np.isfinite(gr)) or gr < 0.05 * wt or st != "optimal")
    return ess, wt

results = {"gca": a.gca, "strain": "E. coli M25631H (GCF_058436375.1) -> MG1655 b-numbers via diamond RBH",
           "cutoff": a.cutoff, "arms": []}
arms = a.arms.split(",")

# ---------------- universal + evidence ----------------
if "meteor" in arms or "thresh" in arms:
    universal, allrxns, allmet = load_universal()
    for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
        if not hasattr(x, "_annotation"): x._annotation = {}
    print("universal genes:", len(universal.genes), "reactions with GPR:", sum(1 for r in universal.reactions if r.gene_reaction_rule))
    seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
    anc = load_ec(data_path("all_ancestors.txt")); mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
    pred = extract_pred(resolve_baseline_pkl(a.baseline, a.variant, a.gca, BASELINE_SUFFIX[a.baseline]), anc)
    prots = list(pred.index); P = pred.values
    S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
    lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{a.gram[:3]}.pkl"))
    if lt is not None: lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
    bid = "biomass_GmPos" if a.gram == "positive" else "biomass_GmNeg"; oi = allrxns.index(bid)
    exc = find_excluded_reactions(S, lb, ub, allrxns, bid)
    lb, ub, media_mask, media_rxns = apply_media(["default"], allrxns, lb, ub)
    ix = {rid: i for i, rid in enumerate(allrxns)}
    tick("loaded universal + evidence")

    def gpr_for(j, ec_filter=None):
        ei = np.where(mask[j] == 1)[0]
        if ec_filter is not None: ei = np.array([e for e in ei if e in ec_filter], dtype=int)
        if len(ei) == 0: return []
        sc = P[:, ei].max(axis=1)
        return [prots[i] for i in np.where(sc >= a.cutoff)[0]]

    def build_model(keep_flags, label, ec_filter=None):
        keep = [r for r, k in zip(universal.reactions, keep_flags) if k]
        m = build_submodel(universal, keep, label, biomass_id=bid); m.objective = bid
        n_gpr = n_orphan_noec = n_orphan_ec = 0
        for r in m.reactions:
            i = ix.get(r.id)
            r.gene_reaction_rule = ""
            if r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower(): continue
            if i is not None: r.lower_bound = max(-100.0, float(lb[i])); r.upper_bound = min(100.0, float(ub[i]))
            if i is None: continue
            genes = gpr_for(i, ec_filter)
            if genes: r.gene_reaction_rule = " or ".join(genes); n_gpr += 1
            elif mask[i].sum() == 0: n_orphan_noec += 1
            else: n_orphan_ec += 1
        m.solver = "glpk"
        n_int = sum(1 for r in m.reactions if not (r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower()))
        info = dict(n_selected=int(keep_flags.sum()), n_rxn_model=len(m.reactions), n_internal=n_int, n_with_gpr=n_gpr,
                    n_orphan_no_ec=n_orphan_noec, n_orphan_ec_below_cutoff=n_orphan_ec, n_genes=len(m.genes))
        return m, info

GLC_MIN = {"cpd00001":100,"cpd00007":10,"cpd00009":100,"cpd00027":5,"cpd00030":100,"cpd00034":100,"cpd00048":100,
           "cpd00058":100,"cpd00063":100,"cpd00067":100,"cpd00099":100,"cpd00149":100,"cpd00205":100,"cpd00254":100,
           "cpd00531":100,"cpd00971":100,"cpd01012":100,"cpd01048":100,"cpd10515":100,"cpd10516":100,"cpd11595":100,
           "cpd00013":100,"cpd00011":10,"cpd11574":100}   # gapseq media.tsv GS_MM_glc (reference essentiality medium)

def set_medium(m, mode):
    """mode 'posthoc' = published pipeline (open all EX, infer uptake, cap -10); 'glc_min' = GS_MM_glc."""
    exs = [r for r in m.reactions if r.id.startswith("EX_") and r.id != "EX_biomass"]
    if mode == "posthoc":
        for r in exs: r.lower_bound, r.upper_bound = -1000.0, 1000.0
        so = m.optimize(); up = {r.id for r in exs if so.fluxes.get(r.id, 0.0) < -1e-6}
        for r in exs: r.lower_bound = -10.0 if r.id in up else 0.0; r.upper_bound = 1000.0
        return len(up)
    n = 0
    for r in exs:
        c = r.id[3:-2]
        if c in GLC_MIN: r.lower_bound = -float(GLC_MIN[c]); n += 1
        else: r.lower_bound = 0.0
        r.upper_bound = 1000.0
    return n

def eval_arm(keep_flags, label, ec_filter=None):
    m, info = build_model(keep_flags, label, ec_filter); tick(f"{label} model built")
    for mode in ("posthoc", "glc_min"):
        n_up = set_medium(m, mode)
        ess, wt = run_sgd(m, f"{label}/{mode}"); tick(f"{label}/{mode} SGD done")
        row = dict(info, medium=mode, n_uptake_open=n_up, wt_growth=round(float(wt), 4))
        row.update(metrics(ess, f"{label}/{mode}", p2b.get) if ess else {"arm": f"{label}/{mode}", "err": "no growth"})
        results["arms"].append(row); print(json.dumps(row), flush=True)
        if ess: json.dump({g: bool(e) for g, e in ess.items()}, open(f"{OUT}/sgd_{label}_{mode}_{a.gca}.json", "w"))

if "meteor" in arms:
    sol = pickle.load(open(f"{RUNS}/{a.baseline}_{a.variant}/meteor_sol_{a.gca}.pkl", "rb"))
    act = np.array(sol["y_vals"]) > 0.5
    eval_arm(act, "meteor")
    # E^new variant: evidence restricted to ECs METEOR kept active (meteor_preds active_ecs)
    pr = pickle.load(open(f"{RUNS}/{a.baseline}_{a.variant}/meteor_preds_{a.gca}.pkl", "rb"))
    active_cols = {i for i, c in enumerate(pred.columns) if str(c).split(":")[-1] in {str(e).split(":")[-1] for e in pr["active_ecs"]}}
    eval_arm(act, "meteor_Enew", ec_filter=active_cols)

if "thresh" in arms:
    ecmax = P.max(axis=0); hot = set(np.where(ecmax >= 0.5)[0]); NR = len(allrxns)
    draft = np.zeros(NR, bool)
    for j in range(NR):
        ei = np.where(mask[j] == 1)[0]
        if len(ei) and any(e in hot for e in ei): draft[j] = True
    avail = np.ones(NR, bool); avail[list(exc)] = False; core = draft & avail
    sup = grow_support(S, lb, ub, oi, avail, 0.1, core=core)
    if sup is None: sup = grow_support(S, lb, ub, oi, np.ones(NR, bool), 0.1, core=core)
    tm = core | (sup if sup is not None else np.zeros(NR, bool)); tick("thresh gap-fill done")
    results["thresh_n_gapfilled"] = int((tm & ~core).sum())
    eval_arm(tm, "thresh")

if "iml1515" in arms:
    m = cobra.io.read_sbml_model(f"{GEMDIR}/{a.gem}.xml"); m.solver = "glpk"
    ess, wt = run_sgd(m, a.gem); tick("iML1515 SGD done")
    row = dict(n_rxn_model=len(m.reactions), n_genes=len(m.genes), wt_growth=round(float(wt), 4),
               medium="model default (glucose M9 aerobic, as shipped in BiGG)")
    row.update(metrics(ess, a.gem, lambda g: g))
    # cross-check with the essentiality call stored in gess-ecol.csv (curated model column)
    cur = {g: (e == "yes") for g, e in zip(ref["gene"], ref["ess.curated.model"])}
    agree = sum(1 for g in ess if g in cur and cur[g] == ess[g]); n = sum(1 for g in ess if g in cur)
    row["agreement_with_gapseq_curated_call"] = round(agree / max(1, n), 3)
    results["arms"].append(row); print(json.dumps(row), flush=True)
    json.dump({g: bool(e) for g, e in ess.items()}, open(f"{OUT}/sgd_{a.gem}.json", "w"))

results["timing_s"] = timing
json.dump(results, open(f"{OUT}/essentiality_{a.baseline}_{a.variant}_{a.gca}.json", "w"), indent=1)
print("-> ", f"{OUT}/essentiality_{a.baseline}_{a.variant}_{a.gca}.json")
