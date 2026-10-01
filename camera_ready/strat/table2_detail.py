"""Table 2 (tab:subthreshold): W-recall and full-set precision, threshold+parsimonious-repair baseline vs METEOR,
mean +- sd over 6 organisms, under none / pkl. Also stores the recomputed baseline EC sets (results/remap_impact/baseline_sets.pkl)."""
import sys, os, re, json, pickle, time, warnings, logging
import numpy as np
warnings.filterwarnings("ignore"); logging.getLogger("cobra").setLevel(logging.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval")
from _env import *
from meteor_v8.utils import (_load_enzyme_obsolete, _resolve_current_ec, extract_pred, extract_fba_matrices, load_tight_bounds,
                             apply_media, find_excluded_reactions, data_path)
from meteor_v8.repair import grow_support
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
import strat_all as SA
FULL = SA.FULL; norm = SA.norm; OUT = f"{HERE}/results/remap_impact"
ORGS = ["E.coli", "Salmonella", "K.pneumoniae", "P.putida", "S.aureus", "B.subtilis"]
pkl = _load_enzyme_obsolete(); MAPS = {"none": {}, "pkl": pkl}
canon_fn = lambda m: (lambda e: e) if not m else (lambda e: _resolve_current_ec(e, m))
def cset(X, c): return {c(e) for e in X}
def cscores(sc, c):
    o = {}
    for e, s in sc.items(): o[c(e)] = max(o.get(c(e), 0.0), s)
    return o
ctx = SA.build_ctx(f"{HERE}/results/strat"); allrxns, anc, mask = ctx["allrxns"], ctx["anc"], ctx["mask"]; NR = len(allrxns)
universal, _, _ = SA.load_universal()
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
S, lb0, ub0 = extract_fba_matrices(universal, allrxns, reversed_trans=True)
seed_idx = {int(e) for j in range(NR) for e in np.where(mask[j] == 1)[0]}
def rxn_ecs_anc(j): return {anc[e] for e in np.where(mask[j] == 1)[0] if FULL.match(anc[e])}
_gc = {}
def setup(gram):
    if gram in _gc: return _gc[gram]
    lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{gram[:3]}.pkl")); lb = np.maximum(lb0, lt); ub = np.minimum(ub0, ut)
    bid = "biomass_GmPos" if gram == "positive" else "biomass_GmNeg"; oi = allrxns.index(bid)
    exc = find_excluded_reactions(S, lb, ub, allrxns, bid); lb, ub, _, _ = apply_media(["default"], allrxns, lb, ub)
    _gc[gram] = (lb, ub, oi, exc); return _gc[gram]
RAW = {}; sets_store = {}
for gem, (gcf, org, gram) in SA.GEM2G.items():
    lb, ub, oi, exc = setup(gram)
    pred = extract_pred(resolve_baseline_pkl("dpz", "vanilla", gcf, BASELINE_SUFFIX["dpz"]), anc); cols = list(pred.columns); mx = pred.values.max(axis=0)
    sc_seed = {cols[j]: float(mx[j]) for j in range(len(cols)) if j in seed_idx and FULL.match(cols[j])}
    sc_all = {cols[j]: float(mx[j]) for j in range(len(cols)) if FULL.match(cols[j])}
    sol = pickle.load(open(f"{RUNS}/dpz_vanilla/meteor_sol_{gcf}.pkl", "rb")); act = np.where(np.asarray(sol["y_vals"]) > 0.5)[0]
    M = set(); [M.update(rxn_ecs_anc(j)) for j in act]
    hot = set(np.where(mx >= 0.5)[0]); draft = np.zeros(NR, bool)
    for j in range(NR):
        ei = np.where(mask[j] == 1)[0]
        if len(ei) and any(e in hot for e in ei): draft[j] = True
    avail = np.ones(NR, bool); avail[list(exc)] = False; core = draft & avail
    sup = grow_support(S, lb, ub, oi, avail, 0.1, core=core)
    if sup is None: sup = grow_support(S, lb, ub, oi, np.ones(NR, bool), 0.1, core=core)
    bmodel = core | (sup if sup is not None else np.zeros(NR, bool))
    B = set(); [B.update(rxn_ecs_anc(j)) for j in np.where(bmodel)[0]]
    Bcore = set(); [Bcore.update(rxn_ecs_anc(j)) for j in np.where(core)[0]]
    RAW[org] = dict(G=SA.gem_ecs(gem), M=M, B=B, Bcore=Bcore, sc_seed=sc_seed, sc_all=sc_all)
    sets_store[org] = dict(gcf=gcf, baseline_rxn_mask=bmodel, baseline_core_mask=core, baseline_ecs=sorted(B), meteor_rxnmapped_ecs=sorted(M))
    print(org, "B", len(B), "Bcore", len(Bcore), "M", len(M), flush=True)
pickle.dump(sets_store, open(f"{OUT}/baseline_sets.pkl", "wb"))
res = {}
for pn, m in MAPS.items():
    c = canon_fn(m); res[pn] = {}
    for wdef in ("seed_restricted", "all_ancestors"):
        rows = []
        for o in ORGS:
            r = RAW[o]; G = cset(r["G"], c); M = cset(r["M"], c); B = cset(r["B"], c); sc = cscores(r["sc_seed" if wdef == "seed_restricted" else "sc_all"], c)
            W = {e for e in G if 0 < sc.get(e, 0.0) < 0.5}
            K = len(M); topK = {e for e, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:K]}
            rows.append(dict(organism=o, W=len(W), hits_meteor=len(M & W), hits_baseline=len(B & W), hits_topK=len(topK & W),
                             recallW_meteor=round(len(M & W) / len(W), 4), recallW_baseline=round(len(B & W) / len(W), 4),
                             precision_meteor=round(len(M & G) / len(M), 4), precision_baseline=round(len(B & G) / len(B), 4),
                             n_meteor=len(M), n_baseline=len(B),
                             baseline_W_hits_via_gapfill_only=len((B - cset(r["Bcore"], c)) & W)))
        ms = lambda k: (round(float(np.mean([x[k] for x in rows])), 4), round(float(np.std([x[k] for x in rows], ddof=1)), 4))
        res[pn][wdef] = dict(rows=rows, pooled_W=sum(x["W"] for x in rows), pooled_hits=dict(meteor=sum(x["hits_meteor"] for x in rows), baseline=sum(x["hits_baseline"] for x in rows), topK=sum(x["hits_topK"] for x in rows)),
                             mean_sd={k: ms(k) for k in ("recallW_meteor", "recallW_baseline", "precision_meteor", "precision_baseline")},
                             n_meteor_gt_baseline_recallW=sum(x["recallW_meteor"] > x["recallW_baseline"] for x in rows))
        print(pn, wdef, res[pn][wdef]["pooled_W"], res[pn][wdef]["pooled_hits"], res[pn][wdef]["mean_sd"], flush=True)
# why baseline W-hits do not change under remap
expl = []
c = canon_fn(pkl)
for o in ORGS:
    r = RAW[o]; G0, Gc = r["G"], cset(r["G"], c); sc0 = r["sc_seed"]; scc = cscores(sc0, c)
    W0 = {e for e in G0 if 0 < sc0.get(e, 0.0) < 0.5}; Wc = {e for e in Gc if 0 < scc.get(e, 0.0) < 0.5}
    newW = Wc - cset(W0, c)                      # ECs that enter W only after remap
    B0, Bc, M0, Mc = r["B"], cset(r["B"], c), r["M"], cset(r["M"], c)
    expl.append(dict(organism=o, W_none=len(W0), W_pkl=len(Wc), new_W_after_remap=len(newW), new_W_hit_by_baseline=len(newW & Bc), new_W_hit_by_meteor=len(newW & Mc),
                     baseline_hits_none=len(B0 & W0), baseline_hits_pkl=len(Bc & Wc), baseline_obsolete_ecs_in_set=len(set(B0) & set(pkl)), meteor_obsolete_ecs_in_set=len(set(M0) & set(pkl)),
                     baseline_hits_lost_by_merge=len(cset(B0 & W0, c)) - len(B0 & W0)))
json.dump(dict(table2=res, why_baseline_constant=expl), open(f"{OUT}/table2_detail.json", "w"), indent=1)
t2 = json.load(open(f"{OUT}/table2.json")); t2["detail"] = res; t2["why_baseline_constant"] = expl; json.dump(t2, open(f"{OUT}/table2.json", "w"), indent=1)
L = ["", "## Table 2 (tab:subthreshold): W-recall and full-set precision, threshold+parsimonious repair vs METEOR (mean +- sd over 6; W = curated ECs with 0 < max DPZ < 0.5)", ""]
for wdef in ("seed_restricted", "all_ancestors"):
    L += [f"W definition: {wdef} (scores over {'SEED-representable ECs only (weakreal_matched.py)' if wdef=='seed_restricted' else 'all all_ancestors columns (weakreal.py / weakreal_ids)'})", "",
          "| cell | published | none | pkl |", "|---|---|---|---|"]
    pub = dict(recallW_baseline="0.480 +- 0.039", recallW_meteor="0.604 +- 0.053", precision_baseline="0.455 +- 0.061", precision_meteor="0.422 +- 0.057")
    for k in ("recallW_baseline", "recallW_meteor", "precision_baseline", "precision_meteor"):
        a, b = res["none"][wdef]["mean_sd"][k], res["pkl"][wdef]["mean_sd"][k]
        L.append(f"| {k} | {pub[k]} | {a[0]:.3f} +- {a[1]:.3f} | {b[0]:.3f} +- {b[1]:.3f} |")
    L.append(f"| pooled W / hits meteor / baseline / topK | 612 | {res['none'][wdef]['pooled_W']} / {res['none'][wdef]['pooled_hits']['meteor']} / {res['none'][wdef]['pooled_hits']['baseline']} / {res['none'][wdef]['pooled_hits']['topK']} | {res['pkl'][wdef]['pooled_W']} / {res['pkl'][wdef]['pooled_hits']['meteor']} / {res['pkl'][wdef]['pooled_hits']['baseline']} / {res['pkl'][wdef]['pooled_hits']['topK']} |")
    L.append(f"| METEOR > baseline recall_W (organisms) | 6/6 | {res['none'][wdef]['n_meteor_gt_baseline_recallW']}/6 | {res['pkl'][wdef]['n_meteor_gt_baseline_recallW']}/6 |")
    L.append("| per organism W: hits M/B (none -> pkl) | | " + "; ".join(f"{x['organism']} {x['W']}: {x['hits_meteor']}/{x['hits_baseline']}" for x in res["none"][wdef]["rows"]) + " | " + "; ".join(f"{x['organism']} {x['W']}: {x['hits_meteor']}/{x['hits_baseline']}" for x in res["pkl"][wdef]["rows"]) + " |")
    L.append("")
L += ["Why the baseline's W hits do not move under remap (seed_restricted W; baseline EC set IS recomputed from the gap-filled reaction mask and canonicalised like every other set, not taken from stored counts; the recomputed sets are saved in results/remap_impact/baseline_sets.pkl):", "",
      "| organism | W none -> pkl | new W members after remap | of which hit by baseline | hit by METEOR | baseline hits none -> pkl | obsolete numbers inside baseline set / METEOR set |", "|---|---|---|---|---|---|---|"]
for x in expl:
    L.append(f"| {x['organism']} | {x['W_none']} -> {x['W_pkl']} | {x['new_W_after_remap']} | {x['new_W_hit_by_baseline']} | {x['new_W_hit_by_meteor']} | {x['baseline_hits_none']} -> {x['baseline_hits_pkl']} | {x['baseline_obsolete_ecs_in_set']} / {x['meteor_obsolete_ecs_in_set']} |")
open(f"{HERE}/results/remap_impact_summary.md", "a").write("\n".join(L) + "\n"); print("\n".join(L))
