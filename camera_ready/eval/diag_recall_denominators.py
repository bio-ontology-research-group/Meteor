"""B1: why Table 3 recall (0.884) differs from the same-predictor recall (0.784).
Table 3 (toolcompare_ec.py) scores meteor_preds[active_ecs]; the same-predictor
comparison (baseline_thresh_gapfill.py / weakreal.py) scores ECs of reactions
with y>0.5. This script counts both sets per curated organism. Read-only inputs."""
import json, pickle, re, numpy as np
from _env import *
from meteor_v8.utils import load_universal, load_refmapping, load_ec, data_path, data_dir, build_rxn_ec_mask, extract_pred
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
FULL = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
MO = f"{RUNS}/dpz_vanilla"
pb = json.load(open(f"{ORIG_RESULTS}/toolcompare/panelB_ec.json"))
ab = {r["organism"]: r for r in json.load(open(f"{ORIG_RESULTS}/toolcompare/ablation_thresh_vs_evw_ec.json"))}
universal, allrxns, _ = load_universal()
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path("all_ancestors.txt")); mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
out = []
for row in pb:
    g = row["gcf"]
    pred = extract_pred(resolve_baseline_pkl("dpz", "vanilla", g, BASELINE_SUFFIX["dpz"]), anc)
    cols = [str(c).split(":")[-1] for c in pred.columns]
    def rxn_ecs(j): return {cols[e] for e in np.where(mask[j] == 1)[0] if e < len(cols) and FULL.match(cols[e])}
    sol = pickle.load(open(f"{MO}/meteor_sol_{g}.pkl", "rb")); act = np.where(np.array(sol["y_vals"]) > 0.5)[0]
    E_rxn = set(); [E_rxn.update(rxn_ecs(j)) for j in act]
    pr = pickle.load(open(f"{MO}/meteor_preds_{g}.pkl", "rb"))
    E_post = {str(e).split(":")[-1] for e in pr["active_ecs"] if FULL.match(str(e).split(":")[-1])}
    out.append(dict(organism=row["organism"], gcf=g, gem_ec=row["gem_ec"],
                    table3_n_ec=row["meteor"]["n_ec"], table3_R=row["meteor"]["R"],
                    samepred_n_ec=ab[row["organism"]]["meteor"]["n_ec"], samepred_R=ab[row["organism"]]["meteor"]["R"],
                    n_ec_active_ecs=len(E_post), n_ec_rxnmapped=len(E_rxn),
                    n_only_active=len(E_post - E_rxn), n_only_rxn=len(E_rxn - E_post)))
    print(out[-1], flush=True)
json.dump(out, open(f"{RESULTS}/recall_denominators.json", "w"), indent=1)
print("-> results/recall_denominators.json")
