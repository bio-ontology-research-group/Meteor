"""Action A: what exactly is meteor_preds['active_ecs']? Exact set arithmetic per organism (dpz vanilla)."""
import sys, re, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval")
from _env import *
from meteor_v8.utils import load_universal, load_refmapping, load_ec, data_path, data_dir, build_rxn_ec_mask, extract_pred
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
FULL = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
def norm(x):
    x = str(x).strip().split("EC:")[-1]; return x if FULL.match(x) else None
GEM2G = {"iML1515": ("GCF_058436375.1", "E.coli"), "STM_v1_0": ("GCF_000006945.2", "Salmonella")}
universal, allrxns, _ = load_universal()
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path("all_ancestors.txt")); anc_set = set(anc); mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
ref = {r["organism"]: r for r in json.load(open(f"{RESULTS}/recall_denominators.json"))}
out = {}
for gem, (gcf, org) in GEM2G.items():
    MO = f"{RUNS}/dpz_vanilla"
    pr = pickle.load(open(f"{MO}/meteor_preds_{gcf}.pkl", "rb")); A_raw = set(map(str, pr["active_ecs"]))
    A = {n for e in A_raw if (n := norm(e))}
    sol = pickle.load(open(f"{MO}/meteor_sol_{gcf}.pkl", "rb")); y = np.asarray(sol["y_vals"]); sel = np.where(y > 0.5)[0]
    # rfinal2ec convention: rkey = rxn.split('_')[0], all ECs in seedr2ec, unrestricted
    U_raw = set(); [U_raw.update(map(str, seedr2ec.get(allrxns[j].split("_")[0], []))) for j in sel]
    U = {n for e in U_raw if (n := norm(e))}
    # anc-restricted mask convention (diag_recall_denominators 'rxnmapped')
    Mk = set(); [Mk.update(anc[e] for e in np.where(mask[j] == 1)[0] if FULL.match(anc[e])) for j in sel]
    pred = extract_pred(resolve_baseline_pkl("dpz", "vanilla", gcf, BASELINE_SUFFIX["dpz"]), anc)
    mx = pred.values.max(axis=0); cols = list(pred.columns)
    H = {cols[j] for j in range(len(cols)) if mx[j] >= 0.5 and FULL.match(cols[j])}
    # ECs in active_ecs with no selected reaction mapped (unrestricted)
    ec2sel = {}
    for j in sel:
        for e in seedr2ec.get(allrxns[j].split("_")[0], []): ec2sel.setdefault(str(e), set()).add(int(j))
    no_rxn = {e for e in A if e not in ec2sel}
    r = dict(organism=org, gcf=gcf, n_selected_rxns=int(len(sel)),
             active_ecs_raw=len(A_raw), active_ecs_4digit=len(A), active_ecs_partial_or_other=len(A_raw) - len(A),
             seedr2ec_selected_unrestricted_raw=len(U_raw), seedr2ec_selected_unrestricted_4digit=len(U),
             symdiff_active_vs_unrestricted_raw=len(A_raw ^ U_raw), symdiff_active_vs_unrestricted_4digit=len(A ^ U),
             ancrestricted_rxnmapped_4digit=len(Mk), active_minus_ancrestricted=len(A - Mk), ancrestricted_minus_active=len(Mk - A),
             active_not_in_all_ancestors=len(A - anc_set), ref_n_only_active=ref[org]["n_only_active"],
             n_score_ge_0p5=len(H), active_and_score_ge_0p5=len(A & H), score_ge_0p5_not_in_active=len(H - A),
             active_ecs_with_no_selected_rxn=len(no_rxn), examples_score_ge_0p5_not_active=sorted(H - A)[:10])
    out[org] = r; print(json.dumps(r, indent=1), flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "verify_active_ecs.json", "w"), indent=1)
