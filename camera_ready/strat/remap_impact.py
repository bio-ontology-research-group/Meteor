"""Effect of the EC remap protocol on every published curated-model comparison (6 organisms).
Protocol: none (published) vs pkl (code_snapshot enzymeobsolete.pkl, the canonical dictionary).
Every EC set is canonicalised old->new; partial ECs dropped. Read-only inputs; writes results/remap_impact/."""
import sys, os, re, json, pickle, time, warnings, logging
import numpy as np, pandas as pd, cobra
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore"); logging.getLogger("cobra").setLevel(logging.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval")
from _env import *
from meteor_v8.utils import (_load_enzyme_obsolete, _resolve_current_ec, extract_pred, extract_fba_matrices, load_tight_bounds,
                             apply_media, find_excluded_reactions, data_path)
from meteor_v8.repair import grow_support
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
import strat_all as SA
FULL = SA.FULL; norm = SA.norm
OUT = f"{HERE}/results/remap_impact"; os.makedirs(OUT, exist_ok=True)
CV = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026/results/carveme_gc"
ORGS = ["E.coli", "Salmonella", "K.pneumoniae", "P.putida", "S.aureus", "B.subtilis"]

# ---------------- remap tables
pkl = _load_enzyme_obsolete()
MAPS = {"none": {}, "pkl": pkl}
def canon_fn(m): return (lambda e: e) if not m else (lambda e: _resolve_current_ec(e, m))
def cset(X, c): return {c(e) for e in X}
def cscores(sc, c):
    out = {}
    for e, s in sc.items(): out[c(e)] = max(out.get(c(e), 0.0), s)
    return out
def metrics(P, G):
    i = len(P & G); prec = i / max(1, len(P)); rec = i / max(1, len(G))
    return dict(n=len(P), P=round(prec, 4), R=round(rec, 4), F1=round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0, J=round(i / max(1, len(P | G)), 4))
def wil(x, y):
    d = np.array(x) - np.array(y)
    if np.all(d == 0): return None
    return round(float(wilcoxon(x, y, method="exact").pvalue), 4)

# ---------------- network context
t0 = time.time()
ctx = SA.build_ctx(f"{HERE}/results/strat")
allrxns, anc, mask = ctx["allrxns"], ctx["anc"], ctx["mask"]; NR = len(allrxns)
universal, _, _ = SA.load_universal()
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
S, lb0, ub0 = extract_fba_matrices(universal, allrxns, reversed_trans=True)
seed_idx = {int(e) for j in range(NR) for e in np.where(mask[j] == 1)[0]}          # weakreal_matched convention
def rxn_ecs_anc(j): return {anc[e] for e in np.where(mask[j] == 1)[0] if FULL.match(anc[e])}
_gc = {}
def setup(gram):
    if gram in _gc: return _gc[gram]
    lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{gram[:3]}.pkl")); lb = np.maximum(lb0, lt); ub = np.minimum(ub0, ut)
    bid = "biomass_GmPos" if gram == "positive" else "biomass_GmNeg"; oi = allrxns.index(bid)
    exc = find_excluded_reactions(S, lb, ub, allrxns, bid); lb, ub, _, _ = apply_media(["default"], allrxns, lb, ub)
    _gc[gram] = (lb, ub, oi, exc); return _gc[gram]
def model_ecs_sbml(p):
    m = cobra.io.read_sbml_model(p); ex = set()
    for r in m.reactions:
        a = r.annotation.get("ec-code") if hasattr(r, "annotation") else None
        if not a: continue
        for x in (a if isinstance(a, list) else [a]):
            if (n := norm(x)): ex.add(n)
    return ex
cvfiles = os.listdir(CV)
def find_cv(acc):
    num = acc.split("_")[1]
    return next(os.path.join(CV, f) for f in cvfiles if num in f and f.endswith(".xml"))

# ---------------- protocol-independent raw sets per organism
RAW = {}
for gem, (gcf, org, gram) in SA.GEM2G.items():
    t1 = time.time(); lb, ub, oi, exc = setup(gram)
    pred = extract_pred(resolve_baseline_pkl("dpz", "vanilla", gcf, BASELINE_SUFFIX["dpz"]), anc)
    cols = list(pred.columns); mx = pred.values.max(axis=0)
    scores_seed = {cols[j]: float(mx[j]) for j in range(len(cols)) if j in seed_idx and FULL.match(cols[j])}
    sol = pickle.load(open(f"{RUNS}/dpz_vanilla/meteor_sol_{gcf}.pkl", "rb")); act = np.where(np.asarray(sol["y_vals"]) > 0.5)[0]
    Mrxn = set(); [Mrxn.update(rxn_ecs_anc(j)) for j in act]
    A = {n for e in pickle.load(open(f"{RUNS}/dpz_vanilla/meteor_preds_{gcf}.pkl", "rb"))["active_ecs"] if (n := norm(e))}
    # S7.4 baseline: hard-threshold draft + parsimony gap-fill (baseline_thresh_gapfill.py / weakreal.py verbatim)
    hot = set(np.where(mx >= 0.5)[0]); draft = np.zeros(NR, bool)
    for j in range(NR):
        ei = np.where(mask[j] == 1)[0]
        if len(ei) and any(e in hot for e in ei): draft[j] = True
    avail = np.ones(NR, bool); avail[list(exc)] = False; core = draft & avail
    sup = grow_support(S, lb, ub, oi, avail, 0.1, core=core)
    if sup is None: sup = grow_support(S, lb, ub, oi, np.ones(NR, bool), 0.1, core=core)
    bmodel = core | (sup if sup is not None else np.zeros(NR, bool))
    Bgf = set(); [Bgf.update(rxn_ecs_anc(j)) for j in np.where(bmodel)[0]]
    RAW[org] = dict(gcf=gcf, gem=gem, G=SA.gem_ecs(gem), C=model_ecs_sbml(find_cv(gcf)), A=A, Mrxn=Mrxn, Bgf=Bgf, scores_seed=scores_seed,
                    n_bmodel=int(bmodel.sum()), n_gapfill=int((bmodel & ~core).sum()))
    print(f"  {org}: G={len(RAW[org]['G'])} C={len(RAW[org]['C'])} A={len(A)} Mrxn={len(Mrxn)} Bgf={len(Bgf)} (rxns {RAW[org]['n_bmodel']}, +{RAW[org]['n_gapfill']} gapfilled) [{time.time()-t1:.0f}s]", flush=True)

# ---------------- per protocol
res = {k: {} for k in ("table5", "s74", "s75", "table2", "recall_denominators")}
for pname, m in MAPS.items():
    c = canon_fn(m)
    Gc = {o: cset(r["G"], c) for o, r in RAW.items()}; Cc = {o: cset(r["C"], c) for o, r in RAW.items()}
    Vb = set().union(*Gc.values()) | set().union(*Cc.values())
    Vseed = cset({anc[e] for e in seed_idx if FULL.match(anc[e])}, c)
    t5, s74, s75, t2, rd = [], [], [], [], []
    for o in ORGS:
        r = RAW[o]; G = Gc[o]; C = Cc[o]; A = cset(r["A"], c); M = cset(r["Mrxn"], c); B = cset(r["Bgf"], c); sc = cscores(r["scores_seed"], c)
        t5.append(dict(organism=o, gem_ec=len(G), meteor_full=metrics(A, G), meteor_refvocab=metrics(A & Vb, G & Vb), carveme=metrics(C, G),
                       meteor_fp_outside_refvocab=len((A - G) - Vb), meteor_fp=len(A - G)))
        s74.append(dict(organism=o, gem_ec=len(G), meteor=metrics(M, G), baseline_gapfill=metrics(B, G)))
        W = {e for e in G if 0 < sc.get(e, 0.0) < 0.5}; K = len(M)
        topK = {e for e, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:K]}
        extra = M - topK
        ext_set = M - B
        s75.append(dict(organism=o, K=K, W=len(W), recall_W_meteor=round(len(M & W) / max(1, len(W)), 4), recall_W_topK=round(len(topK & W) / max(1, len(W)), 4),
                        precision_meteor=round(len(M & G) / max(1, K), 4), precision_topK=round(len(topK & G) / max(1, K), 4),
                        recall_G_meteor=round(len(M & G) / len(G), 4), recall_G_topK=round(len(topK & G) / len(G), 4),
                        extra=len(M) - len(B), extra_correct=len(M & G) - len(B & G), extra_correct_subthreshold=len(M & W) - len(B & W),
                        extra_set=len(ext_set), extra_set_outside_refvocab=len(ext_set - Vb), extra_set_correct=len(ext_set & G),
                        extra_outside_refvocab=len(ext_set - Vb),
                        topK_extra=len(extra), topK_extra_correct=len(extra & G), topK_extra_subthreshold=len(extra & W), topK_extra_outside_refvocab=len(extra - Vb),
                        n_W_recovered_meteor=len(M & W), n_W_recovered_topK=len(topK & W), n_W_recovered_baseline_gapfill=len(B & W)))
        t2.append(dict(organism=o, W=len(W), recovered_meteor=len(M & W), recovered_topK=len(topK & W), recovered_baseline_gapfill=len(B & W),
                       recall_W_baseline_gapfill=round(len(B & W) / max(1, len(W)), 4)))
        rd.append(dict(organism=o, gem_ec=len(G), table3_R=round(len(A & G) / len(G), 4), samepred_R=round(len(M & G) / len(G), 4), n_ec_active=len(A), n_ec_rxnmapped=len(M)))
    mean = lambda rows, f: round(float(np.mean([f(x) for x in rows])), 4)
    res["table5"][pname] = dict(V_ref=len(Vb), V_seed=len(Vseed), rows=t5, mean={arm: {k: mean(t5, lambda x: x[arm][k]) for k in ("n", "P", "R", "F1", "J")} for arm in ("meteor_full", "meteor_refvocab", "carveme")},
                                fp_outside_frac=mean(t5, lambda x: x["meteor_fp_outside_refvocab"] / max(1, x["meteor_fp"])))
    res["s74"][pname] = dict(rows=s74, mean=dict(meteor_R=mean(s74, lambda x: x["meteor"]["R"]), baseline_R=mean(s74, lambda x: x["baseline_gapfill"]["R"]),
                                                  meteor_P=mean(s74, lambda x: x["meteor"]["P"]), baseline_P=mean(s74, lambda x: x["baseline_gapfill"]["P"])),
                             n_meteor_gt_baseline_R=sum(x["meteor"]["R"] > x["baseline_gapfill"]["R"] for x in s74),
                             p_R=wil([x["meteor"]["R"] for x in s74], [x["baseline_gapfill"]["R"] for x in s74]))
    res["s75"][pname] = dict(rows=s75, mean={k: mean(s75, lambda x: x[k]) for k in ("recall_W_meteor", "recall_W_topK", "precision_meteor", "precision_topK", "recall_G_meteor", "recall_G_topK")},
                             n_meteor_gt_topK_recallW=sum(x["recall_W_meteor"] > x["recall_W_topK"] for x in s75), n_meteor_gt_topK_precision=sum(x["precision_meteor"] > x["precision_topK"] for x in s75),
                             p_recallW=wil([x["recall_W_meteor"] for x in s75], [x["recall_W_topK"] for x in s75]), p_precision=wil([x["precision_meteor"] for x in s75], [x["precision_topK"] for x in s75]),
                             marginal=dict(extra=sum(x["extra"] for x in s75), correct=sum(x["extra_correct"] for x in s75), correct_subthreshold=sum(x["extra_correct_subthreshold"] for x in s75),
                                           outside_refvocab=sum(x["extra_outside_refvocab"] for x in s75), extra_set=sum(x["extra_set"] for x in s75),
                                           extra_set_correct=sum(x["extra_set_correct"] for x in s75)))
    res["s75"][pname]["marginal_vs_topK"] = dict(extra=sum(x["topK_extra"] for x in s75), correct=sum(x["topK_extra_correct"] for x in s75),
                                                  subthreshold=sum(x["topK_extra_subthreshold"] for x in s75), outside_refvocab=sum(x["topK_extra_outside_refvocab"] for x in s75))
    mg = res["s75"][pname]["marginal"]; mg["marginal_precision"] = round(mg["correct"] / max(1, mg["extra"]), 4); mg["frac_outside_refvocab"] = round(mg["outside_refvocab"] / max(1, mg["extra_set"]), 4); mg["marginal_precision_set"] = round(mg["extra_set_correct"] / max(1, mg["extra_set"]), 4)
    res["table2"][pname] = dict(rows=t2, pooled=dict(W=sum(x["W"] for x in t2), meteor=sum(x["recovered_meteor"] for x in t2), topK=sum(x["recovered_topK"] for x in t2), baseline_gapfill=sum(x["recovered_baseline_gapfill"] for x in t2)))
    res["recall_denominators"][pname] = dict(rows=rd, mean=dict(table3_R=mean(rd, lambda x: x["table3_R"]), samepred_R=mean(rd, lambda x: x["samepred_R"])))
    print(f"[{pname}] T5 full {res['table5'][pname]['mean']['meteor_full']} ref {res['table5'][pname]['mean']['meteor_refvocab']} carveme {res['table5'][pname]['mean']['carveme']} |V_ref|={len(Vb)} "
          f"| S7.4 {res['s74'][pname]['mean']} {res['s74'][pname]['n_meteor_gt_baseline_R']}/6 p={res['s74'][pname]['p_R']} | S7.5 {res['s75'][pname]['mean']} marginal {mg}", flush=True)
for k, v in res.items(): json.dump(v, open(f"{OUT}/{k}.json", "w"), indent=1)

# ---------------- summary markdown with flags
PUB = dict(table5=dict(meteor_full=dict(n=1521, P=0.438, R=0.884, F1=0.585, J=0.415), meteor_refvocab=dict(n=876, P=0.761, R=0.884, F1=0.817, J=0.694), carveme=dict(n=818, P=0.822, R=0.893, F1=0.854, J=0.746)),
           s74=dict(meteor_R=0.784, baseline_R=0.767), s75=dict(recall_W_meteor=0.606, recall_W_topK=0.483, precision_meteor=0.422, precision_topK=0.385),
           marginal=dict(extra=773, correct=77, correct_subthreshold=75, marginal_precision=0.10, frac_outside_refvocab=0.76))
flags = []
def cell(k, a, b, cnt=False):
    d = b - a; thr = 10 if cnt else 0.01
    if abs(d) > thr: flags.append((k, a, b, round(d, 4)))
    return f"{b:.0f}" if cnt else f"{b:.3f}"
L = ["# Remap impact on published curated-model comparisons (dpz vanilla, 6 organisms)", "",
     f"Protocols: none = published pipeline (partial extract_pred remap only); pkl = every EC set through code_snapshot enzymeobsolete.pkl ({len(pkl)} transfers); "
     "Flag: |delta vs none| > 0.01 (ratios) or > 10 (counts).", "",
     "## Table 5 (tab:carveme) — mean over 6", "", "| row | metric | published | none | pkl |", "|---|---|---|---|---|---|"]
for arm in ("meteor_full", "meteor_refvocab", "carveme"):
    for k in ("n", "P", "R", "F1", "J"):
        a = res["table5"]["none"]["mean"][arm][k]
L += ["", "## S7.4 same-predictor: METEOR (rxn-mapped) vs threshold + gap-fill", "", "| metric | published | none | pkl |", "|---|---|---|---|---|"]
for k, pub in (("meteor_R", 0.784), ("baseline_R", 0.767), ("meteor_P", None), ("baseline_P", None)):
    s = res["s74"][pn]; L.append(f"| diff R / direction / p ({pn}) | +0.017, 6/6, p=.031 | {s['mean']['meteor_R']-s['mean']['baseline_R']:+.3f} | {s['n_meteor_gt_baseline_R']}/6 | p={s['p_R']} |")
L += ["", "## S7.5 sub-threshold recovery (size-matched top-K) + marginal precision", "", "| metric | published | none | pkl |", "|---|---|---|---|---|"]
for k in ("recall_W_meteor", "recall_W_topK", "precision_meteor", "precision_topK", "recall_G_meteor", "recall_G_topK"):
    s = res["s75"][pn]; L.append(f"| direction recallW / precision, p ({pn}) | 6/6 | {s['n_meteor_gt_topK_recallW']}/6, {s['n_meteor_gt_topK_precision']}/6 | p_recallW={s['p_recallW']} | p_prec={s['p_precision']} |")
L.append("| (marginal = METEOR rxn-mapped minus S7.4 gap-filled baseline; extra/correct/sub-threshold are summed count differences; outside-ref-vocab is on the set difference M \\ B) | | | | |")
for k in ("extra", "correct", "correct_subthreshold", "marginal_precision", "extra_set", "extra_set_correct", "marginal_precision_set", "outside_refvocab", "frac_outside_refvocab"):
    a = res["s75"]["none"]["marginal"][k]; cnt = k in ("extra", "correct", "correct_subthreshold", "outside_refvocab", "extra_set", "extra_set_correct")
L += ["", "## Table 2 (recovered sub-threshold ECs; pooled counts, if derived from the same W sets)", "", "| | none | pkl |", "|---|---|---|---|"]
for k in ("W", "meteor", "topK", "baseline_gapfill"):
for i, o in enumerate(ORGS):
    g = lambda pn, k: res["recall_denominators"][pn]["rows"][i][k]
    cell(f"RD {o} table3_R", g("none", "table3_R"), g("pkl", "table3_R")); cell(f"RD {o} samepred_R", g("none", "samepred_R"), g("pkl", "samepred_R"))
L += ["", f"## Flags ({len(flags)} cells moved beyond threshold vs none)", ""] + [f"- {k}: {a} -> {b} ({d:+})" for k, a, b, d in flags]
L += ["", "## Per-organism S7.4 / S7.5 (none -> pkl)", "", "| organism | S7.4 R meteor/baseline none | pkl | S7.5 recallW meteor/topK none | pkl |", "|---|---|---|---|---|"]
for i, o in enumerate(ORGS):
    a, b = res["s74"]["none"]["rows"][i], res["s74"]["pkl"]["rows"][i]; c1, c2 = res["s75"]["none"]["rows"][i], res["s75"]["pkl"]["rows"][i]
    L.append(f"| {o} | {a['meteor']['R']:.3f}/{a['baseline_gapfill']['R']:.3f} | {b['meteor']['R']:.3f}/{b['baseline_gapfill']['R']:.3f} | {c1['recall_W_meteor']:.3f}/{c1['recall_W_topK']:.3f} | {c2['recall_W_meteor']:.3f}/{c2['recall_W_topK']:.3f} |")
open(f"{HERE}/results/remap_impact_summary.md", "w").write("\n".join(L)); print("\n".join(L)); print(f"-> {OUT}/*.json, results/remap_impact_summary.md  total {time.time()-t0:.0f}s")
