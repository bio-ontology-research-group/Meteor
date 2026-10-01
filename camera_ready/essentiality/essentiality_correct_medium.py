######################################################################
#  ___ _                ___ _         _                              #
# |_ _| |__  _____ __  / __| |_  _ __| |_ ___ _ _                    #
#  | || '_ \/ -_) \ / | (__| | || (_-<  _/ -_) '_|                   #
# |___|_.__/\___/_\_\  \___|_|\_,_/__/\__\___|_|                     #
#                                                                    #
# Access is only permitted to authorised users.                      #
#                                                                    #
# All access must comply with the acceptable use policy.             #
#                                                                    #
#                                             - Your Ibex Admin Team #
#                                                  ibex@kaust.edu.sa #
#                              https://kaust-ibex.slack.com #general #
######################################################################
"""Round 3: gene essentiality on the 6 correct-medium combos (E. coli x3
under GS_MM_glc, B. subtilis x3 under LB_marinos). NOT using B. subtilis's
GS_MM_glc results (discarded diagnostic, wrong medium for that organism).

Per combo, per arm (meteor / thresh):
  1. Re-solve the selection MILP (meteor) or build the EC>=0.5 threshold
     draft (thresh), under the organism's CORRECT medium (sign-corrected
     apply_strict_medium, matching resolve_minmed.py's fix).
  2. GPR: reaction r <- proteins scoring any EC(r) >= 0.5 (OR-only, via
     seedr2ec), exactly as round-1's gene_essentiality_demo.py.
  3. Build COBRA model (sign-corrected EX bounds under the organism's
     medium), single_gene_deletion (cobrapy).
  4. Map model genes -> reference identifiers via the organism's RBH table,
     compare to the on-disk reference table (gess-ecol.csv by b-number,
     gess-bsub.csv by gene SYMBOL -- gene2 column, 844/844 populated).
Also runs the curated-GEM sanity check per organism (iML1515 for E. coli,
iYO844 for B. subtilis; iYO844 gene ids are BSU locus tags, matched via the
model's own gene.name attribute against gess-bsub's symbol column).
Writes only under psb_revision/feasibility_essentiality/results/essround3/.
"""
import os, sys, json, pickle, time, argparse, warnings, logging
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
OUT = f"{HERE}/results/essround3"; os.makedirs(OUT, exist_ok=True)
MILP_FLAGS = dict(mu=3.0, pexp=2.0, eps=0.0, gmin=0.1, wmin=0.01, gmax=2.5, lam=1e-4)

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

ORG = {
    "ecoli": dict(gca="GCF_058436375.1", gram="negative", medium="GS_MM_glc",
                  ref_csv=f"{HERE}/../../../gapseq_eval/gapseqEval/GeneEssentiality/essentiality.data/gess-ecol.csv"
                          if False else "/ibex/scratch/projects/c2014/kexin/funcarve/gapseq_eval/gapseqEval/GeneEssentiality/essentiality.data/gess-ecol.csv",
                  ref_gene_col="gene", map_tsv=f"{HERE}/ref/m25631h_to_bnum.tsv",
                  curated_gem="iML1515", curated_match="id"),
    "bsub": dict(gca="GCF_058182495.1", gram="positive", medium="LB_marinos",
                 ref_csv="/ibex/scratch/projects/c2014/kexin/funcarve/gapseq_eval/gapseqEval/GeneEssentiality/essentiality.data/gess-bsub.csv",
                 ref_gene_col="gene2", map_tsv=f"{HERE}/ref/knb1_to_symbol.tsv",
                 curated_gem="iYO844", curated_match="name"),
    # Round 4: E. coli under a RICH medium (LB_marinos, same compound set already
    # validated feasible for B. subtilis), scored against PEC (Profiling of E. coli
    # Chromosome, https://shigen.nig.ac.jp/ecoli/pec/) instead of gess-ecol.csv --
    # PEC's essential/non-essential classification is condition-independent
    # (fails to grow under EVERY documented condition, mostly LB-disruption-derived;
    # see ref/ecoli_medium_provenance.txt), unlike gess-ecol.csv which likely
    # reflects Monk et al. 2017's M9-carbon-source screens. Same GCA, same RBH
    # mapping (m25631h_to_bnum.tsv) -- both are MG1655 b-number spaces.
    "ecoli_lb": dict(gca="GCF_058436375.1", gram="negative", medium="LB_marinos",
                     ref_csv=f"{HERE}/ref/pec/pec_ecoli_binary.csv",
                     ref_gene_col="gene", map_tsv=f"{HERE}/ref/m25631h_to_bnum.tsv",
                     curated_gem="iML1515", curated_match="id"),
    # Round 7: B. subtilis under GS_MM_glc (minimal medium), scored against the
    # combined Koo et al. 2017 minimal-medium reference (Table S3 LB-essential UNION
    # Table S4/D auxotrophs, plus gess-bsub.csv-derived non-essential backbone --
    # see build_bsub_minimal_ref.py and ref/koo2017/bsub_minimal_binary.csv).
    "bsub_koomin": dict(gca="GCF_058182495.1", gram="positive", medium="GS_MM_glc",
                        ref_csv=f"{HERE}/ref/koo2017/bsub_minimal_binary.csv",
                        ref_gene_col="gene", map_tsv=f"{HERE}/ref/knb1_to_symbol.tsv",
                        curated_gem="iYO844", curated_match="name"),
    # Round 8: S. aureus (5th core experiment) under TSB (rich medium, matches
    # Santiago et al. 2015's actual Tn-seq screening medium exactly; Chaudhuri
    # et al. 2009 used BHI, a comparable standard rich broth -- see
    # ref/sau/sau_medium_provenance.txt), scored against Koo et al. 2017's own
    # cross-species Table S3 sheet C (392 genes conserved across their 4-species
    # comparison -- NOT genome-wide, smaller denominator, see README). Panel
    # genome is strain RN4220 (GCF_045348045.1); reference is SAOUHSC (NCTC 8325)
    # locus tags -- RBH built fresh (ref/sau/rn4220_to_saouhsc.tsv). Curated GEM
    # iYS854 uses USA300_TCH1516 gene IDs, a THIRD strain space -- needs its own
    # RBH-derived external map (ref/sau/iys854_gene_to_saouhsc.tsv), hence
    # curated_match="external_map".
    "sau": dict(gca="GCF_045348045.1", gram="positive", medium="TSB",
               ref_csv=f"{HERE}/ref/koo2017/sau_binary.csv",
               ref_gene_col="gene", map_tsv=f"{HERE}/ref/sau/rn4220_to_saouhsc.tsv",
               curated_gem="iYS854", curated_match="external_map",
               curated_map_tsv=f"{HERE}/ref/sau/iys854_gene_to_saouhsc.tsv"),
    # Round 9: Salmonella (panel=LT2, GCF_000006945.2), Yasir et al. 2024 mBio
    # TraDIS reference (strain SL1344, LB agar 37C). RBH: panel(LT2 protein,
    # confirmed = LT2's own official NP_ accessions) -> SL1344 locus tag
    # (ref/salmonella_map/lt2_to_sl1344.tsv). Curated GEM STM_v1_0 uses LT2's
    # own STM#### locus tags directly -- external map chains STM#### -> LT2
    # protein -> SL1344 locus (ref/salmonella_map/stm_v1_0_to_sl1344.tsv).
    "salmonella": dict(gca="GCF_000006945.2", gram="negative", medium="LB_marinos",
               ref_csv=f"{HERE}/ref/salmonella/salmonella_binary.csv",
               ref_gene_col="gene", map_tsv=f"{HERE}/ref/salmonella_map/lt2_to_sl1344.tsv",
               curated_gem="STM_v1_0", curated_match="external_map",
               curated_map_tsv=f"{HERE}/ref/salmonella_map/stm_v1_0_to_sl1344.tsv"),
    # Round 9: K. pneumoniae (panel=Kp0179, GCF_058435815.1), same Yasir et al.
    # 2024 mBio TraDIS source, strain Ecl8 (LB agar, 37C) tried first per
    # coordinator instruction. RBH: panel(Kp0179) -> Ecl8 BN373_ locus tag
    # (ref/kpneumoniae_map/kp0179_to_ecl8.tsv). Curated GEM iYL1228 uses
    # MGH 78578's KPN_##### locus tags -- external map chains KPN_ -> MGH78578
    # protein -> Ecl8 protein (RBH) -> BN373_ locus
    # (ref/kpneumoniae_map/iyl1228_to_ecl8.tsv).
    "kpneumoniae": dict(gca="GCF_058435815.1", gram="negative", medium="LB_marinos",
               ref_csv=f"{HERE}/ref/kpneumoniae/kpneumoniae_ecl8_binary.csv",
               ref_gene_col="gene", map_tsv=f"{HERE}/ref/kpneumoniae_map/kp0179_to_ecl8.tsv",
               curated_gem="iYL1228", curated_match="external_map",
               curated_map_tsv=f"{HERE}/ref/kpneumoniae_map/iyl1228_to_ecl8.tsv"),
    # Round 9: P. putida (panel=KT2440, GCF_045571375.1 -- SAME strain as the
    # reference, but a different/modern PGAP re-annotation with different
    # locus tags, so RBH was still needed against the classic PP_-tagged
    # assembly GCF_000007565.2). Royet et al. 2025 Environ Microbiol Tn-seq,
    # LB agar baseline. Binarization (judgment call, not given by the paper):
    # essential = ES only; non-essential = NE+GD+GA pooled (GD=growth defect
    # but viable, GA=growth advantage -- neither implies essentiality).
    # Curated GEM iJN1463 already uses classic PP_#### gene IDs directly
    # (curated_match="id", no external map needed).
    "pputida": dict(gca="GCF_045571375.1", gram="negative", medium="LB_marinos",
               ref_csv=f"{HERE}/ref/pputida/pputida_binary.csv",
               ref_gene_col="gene", map_tsv=f"{HERE}/ref/pputida_map/panel_to_pp.tsv",
               curated_gem="iJN1463", curated_match="id"),
}
TSB = {"cpd00018","cpd00322","cpd00035","cpd00161","cpd00058","cpd00438","cpd00048","cpd00051",
       "cpd00215","cpd00066","cpd01048","cpd00226","cpd00069","cpd00054","cpd00254","cpd00107",
       "cpd00063","cpd00184","cpd00099","cpd00027","cpd00041","cpd00009","cpd00119","cpd00381",
       "cpd00793","cpd01012","cpd00023","cpd00246","cpd00393","cpd00007","cpd09398","cpd00084",
       "cpd11595","cpd00971","cpd00034","cpd00383","cpd00039","cpd00182","cpd00205","cpd00531",
       "cpd10516","cpd00249","cpd03424","cpd10515","cpd00028","cpd00129","cpd00091","cpd00149",
       "cpd00030","cpd00218","cpd00219","cpd00311","cpd00220","cpd00067","cpd00126","cpd00644",
       "cpd00060","cpd19148","cpd00156","cpd00001","cpd00541","cpd00654","cpd00092","cpd00239",
       "cpd00033","cpd00046","cpd00065"}  # METEOR's own data/medium.pkl "TSB" entry (67 cpds) --
# no gapseq media.tsv entry exists for S. aureus (the gapseq paper's 5-organism
# essentiality benchmark did not include S. aureus), so this is sourced from
# METEOR's own built-in medium.pkl instead, matching Santiago et al. 2015's
# actual Tn-seq library growth medium (tryptic soy broth, TSB) exactly, and
# a reasonable proxy for Chaudhuri et al. 2009's BHI (both standard rich
# peptone-based broths) -- see ref/sau/sau_medium_provenance.txt.
MEDIA = {"GS_MM_glc": GS_MM_GLC, "LB_marinos": LB_MARINOS, "TSB": TSB}

def apply_strict_medium(allrxns, lb, ub, cpd_set):
    lb = lb.copy(); ub = ub.copy(); allowed = {"EX_" + c + "_e" for c in cpd_set}; media_rxns = set()
    for i, r in enumerate(allrxns):
        if not r.startswith("EX_"): continue
        if r in allowed: lb[i] = -100.0; ub[i] = 100.0; media_rxns.add(i)
        else: lb[i] = 0.0; ub[i] = 1000.0
    return lb, ub, media_rxns

ap = argparse.ArgumentParser()
ap.add_argument("--organism", required=True, choices=list(ORG))
ap.add_argument("--baseline")
ap.add_argument("--variant", default="vanilla")
ap.add_argument("--cutoff", type=float, default=0.5)
ap.add_argument("--procs", type=int, default=4)
ap.add_argument("--time_limit", type=int, default=300)
ap.add_argument("--curated", action="store_true", help="run only the curated-GEM sanity check for this organism")
a = ap.parse_args()
cfg = ORG[a.organism]
T0 = time.time(); timing = {}
def tick(k): timing[k] = round(time.time() - T0, 1); print(f"[{timing[k]:7.1f}s] {k}", flush=True)

# ---------------- reference + mapping ----------------
ref = pd.read_csv(cfg["ref_csv"])
ref_ess = {str(g): (e == "yes") for g, e in zip(ref[cfg["ref_gene_col"]], ref["ess.experimental"])}
p2ref = {}
for l in open(cfg["map_tsv"]):
    parts = l.rstrip("\n").split("\t")
    if len(parts) >= 3: p2ref[parts[0]] = parts[2]

def metrics(pred_ess: dict, label: str, mapper=lambda g: g):
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
    return dict(arm=label, n_genes_model=n_model, n_genes_mapped=n_mapped, n_genes_with_ref=n_ref,
                n_pred_essential=int(sum(pred_ess.values())), TP=tp, FP=fp, FN=fn, TN=tn,
                precision=round(P, 3), recall=round(Rc, 3), MCC=round(mcc, 3), accuracy=round((tp + tn) / max(1, n_ref), 3))

def run_sgd(model, label):
    wt = model.slim_optimize(); print(f"  {label}: WT growth={wt:.4f}, n_rxn={len(model.reactions)}, n_genes={len(model.genes)}", flush=True)
    if not (wt > 1e-6): return None, wt
    sgd = single_gene_deletion(model, processes=a.procs)
    ess = {}
    for ids, gr, st in zip(sgd["ids"], sgd["growth"], sgd["status"]):
        g = next(iter(ids)); ess[g] = bool((not np.isfinite(gr)) or gr < 0.05 * wt or st != "optimal")
    return ess, wt

# =========================== curated-GEM sanity check ===========================
if a.curated:
    OUTF = f"{OUT}/curated_{cfg['curated_gem']}_{a.organism}.json"
    m = cobra.io.read_sbml_model(f"{GEMDIR}/{cfg['curated_gem']}.xml"); m.solver = "glpk"
    ess, wt = run_sgd(m, cfg["curated_gem"]); tick("curated SGD done")
    if cfg["curated_match"] == "id":
        mapper = lambda g: g
    elif cfg["curated_match"] == "external_map":
        ext = {}
        for l in open(cfg["curated_map_tsv"]):
            parts = l.rstrip("\n").split("\t")
            if len(parts) >= 3: ext[parts[0]] = parts[2]
        mapper = lambda g: ext.get(g)
    else:
        name_of = {g.id: g.name for g in m.genes}
        mapper = lambda g: name_of.get(g)
    row = dict(gem=cfg["curated_gem"], n_rxn_model=len(m.reactions), n_genes=len(m.genes),
               wt_growth=round(float(wt), 4), medium="model default (as shipped in BiGG)")
    row.update(metrics(ess, cfg["curated_gem"], mapper))
    json.dump(row, open(OUTF, "w"), indent=1); print(json.dumps(row, indent=1))
    sys.exit(0)

# =========================== METEOR + threshold arms ===========================
assert a.baseline, "--baseline required unless --curated"
OUTF = f"{OUT}/{a.organism}_{a.baseline}_{a.variant}.json"
bid = "biomass_GmPos" if cfg["gram"] == "positive" else "biomass_GmNeg"
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, allmet = load_universal(); anc = load_ec(data_path("all_ancestors.txt"))
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
pred = extract_pred(resolve_baseline_pkl(a.baseline, a.variant, cfg["gca"], BASELINE_SUFFIX[a.baseline]), anc)
prots = [str(p).split()[0] for p in pred.index]; P = pred.values  # CLEAN baseline index = full FASTA header; strip to accession
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
S, lb0, ub0 = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{cfg['gram'][:3]}.pkl"))
if lt is not None: lb0 = np.maximum(lb0, lt); ub0 = np.minimum(ub0, ut)
obj_idx = allrxns.index(bid)
excludes = find_excluded_reactions(S, lb0, ub0, allrxns, bid)  # before medium swap
lb, ub, media_rxns = apply_strict_medium(allrxns, lb0, ub0, MEDIA[cfg["medium"]])
ix = {rid: i for i, rid in enumerate(allrxns)}
solver, _ = _detect_solver(threads=4, time_limit=a.time_limit)
tick("loaded universal + evidence + medium")

def gpr_for(j):
    ei = np.where(mask[j] == 1)[0]
    if len(ei) == 0: return []
    sc = P[:, ei].max(axis=1)
    return [prots[i] for i in np.where(sc >= a.cutoff)[0]]

def build_model_with_gpr(keep_flags, label):
    keep = [r for r, k in zip(universal.reactions, keep_flags) if k]
    m = build_submodel(universal, keep, label, biomass_id=bid); m.objective = bid
    n_gpr = n_orphan_noec = n_orphan_ec = 0
    for r in m.reactions:
        i = ix.get(r.id); r.gene_reaction_rule = ""
        if r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower(): continue
        if i is not None:
            genes = gpr_for(i)
            if genes: r.gene_reaction_rule = " or ".join(genes); n_gpr += 1
            elif mask[i].sum() == 0: n_orphan_noec += 1
            else: n_orphan_ec += 1
    # sign-corrected exchange bounds (round-2 fix): cobra_lb=-ub_matrix, cobra_ub=-lb_matrix
    for r in m.reactions:
        if r.id.startswith("EX_") and r.id != "EX_biomass":
            i = ix.get(r.id)
            if i is not None: r.lower_bound = -float(ub[i]); r.upper_bound = -float(lb[i])
        elif not (r.id.startswith(("DM_", "SK_")) or "biomass" in r.id.lower()):
            i = ix.get(r.id)
            if i is not None: r.lower_bound = float(lb[i]); r.upper_bound = float(ub[i])
    m.solver = "glpk"
    info = dict(n_selected=int(keep_flags.sum()), n_rxn_model=len(m.reactions), n_with_gpr=n_gpr,
                n_orphan_no_ec=n_orphan_noec, n_orphan_ec_below_cutoff=n_orphan_ec, n_genes=len(m.genes))
    return m, info

results = {"organism": a.organism, "gca": cfg["gca"], "medium": cfg["medium"], "baseline": a.baseline, "arms": []}

def eval_arm(keep_flags, label, extra_info=None):
    m, info = build_model_with_gpr(keep_flags, label); tick(f"{label} model built")
    if extra_info: info.update(extra_info)
    ess, wt = run_sgd(m, label); tick(f"{label} SGD done")
    row = dict(info, wt_growth=round(float(wt), 4))
    row.update(metrics(ess, label, p2ref.get) if ess is not None else {"arm": label, "err": "no growth"})
    results["arms"].append(row); print(json.dumps(row), flush=True)
    if ess: json.dump({g: bool(e) for g, e in ess.items()}, open(f"{OUT}/sgd_{a.organism}_{label}_{a.baseline}.json", "w"))

# ---- METEOR arm: re-solve MILP under the organism's correct medium ----
feas, skel, bm_max = biomass_feasible_skeleton(S, lb, ub, obj_idx, MILP_FLAGS["gmin"], solver=solver)
tick(f"skeleton feasible={feas} bm_max={bm_max:.2f}")
assert feas, f"skeleton infeasible for {a.organism}/{a.baseline} under {cfg['medium']} -- STOP, contradicts round-2 result"
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
assert status in ("Optimal", "Not Solved") and (yv > 0.5).any(), f"MILP did not select anything for {a.organism}/{a.baseline}"
yv2, n_active, n_repaired, mb_core = verify_and_repair(S, lb, ub, obj_idx, yv, vv, cand)
eval_arm(yv2 > 0.5, "meteor", extra_info=dict(solve_sec=solve_sec, n_repaired=int(n_repaired)))

# ---- threshold-baseline arm: EC>=0.5 draft; gap-fill only if it doesn't grow (S7.4) ----
ecmax = P.max(axis=0); hot = set(np.where(ecmax >= 0.5)[0]); NR = len(allrxns)
draft = np.zeros(NR, bool)
for j in range(NR):
    ei = np.where(mask[j] == 1)[0]
    if len(ei) and any(e in hot for e in ei): draft[j] = True
avail = np.ones(NR, bool); avail[list(excludes)] = False; core = draft & avail
# check raw draft growth first (sign-corrected quick FBA probe)
mdl_raw, _ = build_model_with_gpr(core, "thresh_raw_probe")
raw_growth = mdl_raw.slim_optimize() or 0.0
tick(f"threshold raw draft growth (no gap-fill) = {raw_growth:.4f}")
if raw_growth > 1e-6:
    tm = core; gapfilled = 0
else:
    sup = grow_support(S, lb, ub, obj_idx, avail, MILP_FLAGS["gmin"], core=core)
    if sup is None: sup = grow_support(S, lb, ub, obj_idx, np.ones(NR, bool), MILP_FLAGS["gmin"], core=core)
    tm = core | (sup if sup is not None else np.zeros(NR, bool)); gapfilled = int((tm & ~core).sum())
    tick(f"threshold gap-filled (+{gapfilled} rxns, matches S7.4 baseline)")
eval_arm(tm, "thresh", extra_info=dict(raw_growth=round(float(raw_growth), 4), used_gapfill=bool(raw_growth <= 1e-6), n_gapfilled=gapfilled))

results["timing_s"] = timing
json.dump(results, open(OUTF, "w"), indent=1)
print("-> ", OUTF)
