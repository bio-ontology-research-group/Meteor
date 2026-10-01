"""C3: size-matched control for curated sub-threshold EC recovery (Section 3.2).
top-K baseline = K highest max-over-proteome DPZ scores among SEED-representable
four-digit ECs, K = |METEOR EC set| (reaction-mapped, as in weakreal.py)."""
import json, pickle, re, numpy as np, cobra, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("cobra").setLevel(logging.ERROR)
from _env import *
from meteor_v8.utils import load_universal, load_refmapping, load_ec, data_path, data_dir, build_rxn_ec_mask, extract_pred
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
FULL = re.compile(r"^\d+\.\d+\.\d+\.\d+$"); MO = f"{RUNS}/dpz_vanilla"
GEM2G = {"iML1515": ("GCF_058436375.1", "neg", "E.coli"), "STM_v1_0": ("GCF_000006945.2", "neg", "Salmonella"),
         "iYL1228": ("GCF_058435815.1", "neg", "K.pneumoniae"), "iJN1463": ("GCF_045571375.1", "neg", "P.putida"),
         "iYS854": ("GCF_045348045.1", "pos", "S.aureus"), "iYO844": ("GCF_058182495.1", "pos", "B.subtilis")}
universal, allrxns, _ = load_universal()
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path("all_ancestors.txt")); mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
seed_ecs = set()
for j in range(len(allrxns)):
    for e in np.where(mask[j] == 1)[0]: seed_ecs.add(int(e))
def gem_ecs(gem):
    m = cobra.io.read_sbml_model(f"{GEMDIR}/{gem}.xml"); ex = set()
    for r in m.reactions:
        a = r.annotation.get("ec-code") if hasattr(r, "annotation") else None
        if not a: continue
        for x in (a if isinstance(a, list) else [a]):
            if FULL.match(str(x).strip()): ex.add(str(x).strip())
    return ex
prev = {r["organism"]: r for r in json.load(open(f"{ORIG_RESULTS}/toolcompare/weakreal.json"))}
rows = []
for gem, (gcf, gram, org) in GEM2G.items():
    pred = extract_pred(resolve_baseline_pkl("dpz", "vanilla", gcf, BASELINE_SUFFIX["dpz"]), anc)
    cols = [str(c).split(":")[-1] for c in pred.columns]; mx = pred.values.max(axis=0)
    scores = {cols[j]: float(mx[j]) for j in range(len(cols)) if j in seed_ecs and FULL.match(cols[j])}
    def rxn_ecs(j): return {cols[e] for e in np.where(mask[j] == 1)[0] if e < len(cols) and FULL.match(cols[e])}
    sol = pickle.load(open(f"{MO}/meteor_sol_{gcf}.pkl", "rb")); act = np.where(np.array(sol["y_vals"]) > 0.5)[0]
    M = set(); [M.update(rxn_ecs(j)) for j in act]
    G = gem_ecs(gem); W = {e for e in G if 0 < scores.get(e, 0.0) < 0.5}
    K = len(M); topK = {e for e, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:K]}
    thr = {e for e, s in scores.items() if s >= 0.5}
    rows.append(dict(organism=org, gem=gem, gcf=gcf, K=K, W=len(W), n_thr=len(thr),
        recall_W_meteor=round(len(M & W)/max(1, len(W)), 3), recall_W_topK=round(len(topK & W)/max(1, len(W)), 3),
        recall_W_thr=0.0, recall_W_baseline_gapfill_published=prev.get(org, {}).get("W_recovery", {}).get("baseline"),
        precision_meteor=round(len(M & G)/max(1, K), 3), precision_topK=round(len(topK & G)/max(1, K), 3),
        recall_G_meteor=round(len(M & G)/len(G), 3), recall_G_topK=round(len(topK & G)/len(G), 3),
        n_topK_below_thr=len(topK - thr)))
    print(rows[-1], flush=True)
mean = lambda k: round(float(np.mean([r[k] for r in rows])), 3)
summ = {k: mean(k) for k in ("recall_W_meteor", "recall_W_topK", "precision_meteor", "precision_topK", "recall_G_meteor", "recall_G_topK")}
json.dump(dict(rows=rows, summary=summ), open(f"{RESULTS}/weakreal_matched.json", "w"), indent=1); print(summ)
