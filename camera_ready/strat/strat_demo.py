"""R1 feasibility demo: stratify curated-EC recall by predictor vocabulary /
baseline confidence, and quantify the 'dark proteome' per predictor.
Read-only on everything except this directory. Usage:
    python strat_demo.py [GEM ...]      (default: iML1515 only)
"""
import sys, os, re, json, time, pickle, warnings, logging
import numpy as np, pandas as pd, cobra
warnings.filterwarnings("ignore"); logging.getLogger("cobra").setLevel(logging.ERROR)
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval")
from _env import *
from meteor_v8.utils import load_universal, load_refmapping, load_ec, data_path, data_dir, build_rxn_ec_mask, extract_pred
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
GEM2G = {"iML1515": ("GCF_058436375.1", "E.coli"), "STM_v1_0": ("GCF_000006945.2", "Salmonella"),
         "iYL1228": ("GCF_058435815.1", "K.pneumoniae"), "iJN1463": ("GCF_045571375.1", "P.putida"),
         "iYS854": ("GCF_045348045.1", "S.aureus"), "iYO844": ("GCF_058182495.1", "B.subtilis")}
PREDICTORS = ("clean", "dpz", "enzbert")
VARIANT = "vanilla"
THR = 0.5                       # baseline calling threshold used in the paper's same-predictor comparison
BINS = [("0", lambda s: s <= 0.0), ("(0,0.1)", lambda s: 0.0 < s < 0.1),
        ("[0.1,0.5)", lambda s: 0.1 <= s < 0.5), ("[0.5,1]", lambda s: s >= 0.5)]

def norm(x):
    x = str(x).strip(); x = x.split("EC:")[-1]
    return x if FULL.match(x) else None

def gem_ecs(gem):
    m = cobra.io.read_sbml_model(f"{GEMDIR}/{gem}.xml"); ex = set()
    for r in m.reactions:
        a = r.annotation.get("ec-code") if hasattr(r, "annotation") else None
        if not a: continue
        for x in (a if isinstance(a, list) else [a]):
            n = norm(x)
            if n: ex.add(n)
    return ex

def rec(P, G):
    return round(len(P & G) / len(G), 3) if G else None

t0 = time.time()
universal, allrxns, _ = load_universal()
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path("all_ancestors.txt")); mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
# ECs the skeleton can emit: exactly what rfinal2ec() can put into active_ecs (seedr2ec over universal reactions,
# NOT restricted to all_ancestors.txt -- 509 seedr2ec ECs are absent from that list)
seed_ecs = {n for r in allrxns for x in seedr2ec.get(r.split("_")[0], []) if (n := norm(x))}
print(f"[setup {time.time()-t0:.1f}s] universal rxns={len(allrxns)} skeleton-expressible 4-digit ECs={len(seed_ecs)}", flush=True)

gems = sys.argv[1:] or ["iML1515"]
out = []
for gem in gems:
    gcf, org = GEM2G[gem]; tg = time.time()
    G = gem_ecs(gem); G_seed = G & seed_ecs
    row = dict(organism=org, gem=gem, gcf=gcf, gem_ec=len(G), gem_ec_skeleton_expressible=len(G_seed),
               gem_ec_unreachable_by_skeleton=len(G - seed_ecs), predictors={})
    for b in PREDICTORS:
        tb = time.time()
        pkl = resolve_baseline_pkl(b, VARIANT, gcf, BASELINE_SUFFIX[b])
        raw = pd.read_pickle(pkl)
        if raw.index[0].count(".") == 3: raw = raw.T
        n_prot = raw.shape[0]
        V_raw = {n for c in raw.columns if (n := norm(c))}
        pred = extract_pred(pkl, anc)                     # obsolete-EC remap + reindex to ancestors, as METEOR sees it
        mx = pred.values.max(axis=0); cols = list(pred.columns)
        V = V_raw | {cols[j] for j in range(len(cols)) if mx[j] > 0 and FULL.match(cols[j])}
        score = {cols[j]: float(mx[j]) for j in range(len(cols)) if FULL.match(cols[j])}
        # protein-level: max score over all ECs (incl. partial) and over 4-digit ECs
        pmax = pred.values.max(axis=1)
        full_idx = [j for j in range(len(cols)) if FULL.match(cols[j])]
        pmax4 = pred.values[:, full_idx].max(axis=1)
        # METEOR (Table 3 convention: active_ecs) and baseline (max-over-proteome >= THR)
        pr = pickle.load(open(f"{RUNS}/{b}_{VARIANT}/meteor_preds_{gcf}.pkl", "rb"))
        M = {n for e in pr["active_ecs"] if (n := norm(e))}
        B = {e for e, s in score.items() if s >= THR}
        G_in, G_out = G & V, G - V
        G_out_seed = G_out & seed_ecs                      # only the skeleton could ever propose these
        strat = dict(
            vocab_size_4digit=len(V), n_proteins=n_prot,
            gem_in_vocab=len(G_in), gem_out_vocab=len(G_out),
            gem_out_vocab_but_skeleton_expressible=len(G_out_seed),
            recall_all=dict(meteor=rec(M, G), baseline=rec(B, G)),
            recall_in_vocab=dict(meteor=rec(M, G_in), baseline=rec(B, G_in)),
            recall_out_vocab=dict(meteor=rec(M, G_out), baseline=rec(B, G_out)),
            recall_out_vocab_skeleton_expressible=dict(meteor=rec(M, G_out_seed), baseline=rec(B, G_out_seed)),
            n_gem_recovered_only_via_skeleton=len(M & G_out),
            by_conf_bin={}, dark_proteome={})
        for name, f in BINS:
            Gb = {e for e in G if f(score.get(e, 0.0))}
            strat["by_conf_bin"][name] = dict(n_gem=len(Gb), meteor=rec(M, Gb), baseline=rec(B, Gb),
                                              n_meteor_hit=len(M & Gb))
        for thr in (0.1, 0.3, 0.5):
            strat["dark_proteome"][f"frac_proteins_max_lt_{thr}"] = round(float((pmax < thr).mean()), 3)
            strat["dark_proteome"][f"frac_proteins_max4digit_lt_{thr}"] = round(float((pmax4 < thr).mean()), 3)
        strat["meteor_n_ec"] = len(M); strat["baseline_n_ec"] = len(B)
        strat["meteor_n_ec_out_vocab"] = len(M - V)
        strat["sec"] = round(time.time() - tb, 1)
        row["predictors"][b] = strat
        print(f"  {org} {b:8s} vocab={len(V)} G_in={len(G_in)} G_out={len(G_out)} (skel-expressible {len(G_out_seed)}) "
              f"R_all M/B={strat['recall_all']['meteor']}/{strat['recall_all']['baseline']} "
              f"R_in={strat['recall_in_vocab']['meteor']}/{strat['recall_in_vocab']['baseline']} "
              f"R_out={strat['recall_out_vocab']['meteor']}/{strat['recall_out_vocab']['baseline']} "
              f"bins={ {k:(v['n_gem'],v['meteor'],v['baseline']) for k,v in strat['by_conf_bin'].items()} } "
              f"dark<0.5={strat['dark_proteome']['frac_proteins_max_lt_0.5']} [{strat['sec']}s]", flush=True)
    row["sec"] = round(time.time() - tg, 1); out.append(row)
tag = "_".join(gems) if len(gems) <= 2 else f"{len(gems)}gems"
jp = f"{HERE}/strat_{tag}.json"
json.dump(out, open(jp, "w"), indent=1)
print(f"-> {jp}  total {time.time()-t0:.1f}s")
