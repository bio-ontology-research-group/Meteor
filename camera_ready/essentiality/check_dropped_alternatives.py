"""For each METEOR-dropped-but->=0.5-confidence reaction (thresh-only,
high-confidence bucket from check_2_and_1.py), test whether METEOR's
selected network actually has a substitute for it:
  (a) same-EC alternative: another reaction covering the same EC(s), selected by METEOR
  (b) metabolite connectivity: are this reaction's metabolites still
      producible/consumable by >=1 other METEOR-selected reaction (not a dead end)
A reaction with neither (a) nor (b) is a case where METEOR's drop has NO
backup in its own network -- a real gap, not a defensible parsimony call.
No MILP re-solve; reuses saved y-vectors + predictor scores.
"""
import sys, json
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, '/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval')
from _env import *  # noqa
from meteor_v8.utils import (load_universal, extract_fba_matrices, build_rxn_ec_mask,
    extract_pred, load_refmapping, load_ec, data_path, data_dir)
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX

HERE = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality"
OUT = f"{HERE}/results/forensic"
GCA = {"salmonella": "GCF_000006945.2", "kpneumoniae": "GCF_058435815.1", "pputida": "GCF_045571375.1"}

universal, allrxns, allmet = load_universal()
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path("all_ancestors.txt"))
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)  # rxn x EC boolean-ish
ix = {rid: i for i, rid in enumerate(allrxns)}
S, lb0, ub0 = extract_fba_matrices(universal, allrxns, reversed_trans=True)
S = S.tocsc() if sp.issparse(S) else sp.csc_matrix(S)

# universal cofactors: exclude from the "metabolite connectivity" check --
# these touch thousands of reactions and would make met_alt trivially True
# for almost anything, defeating the point of the check (same issue found
# in the earlier Salmonella neighbor-trace: ATP/water/phosphate are
# near-universal connectors, not evidence of a real alternate pathway).
COFACTOR_IDS = {"cpd00001","cpd00002","cpd00008","cpd00009","cpd00067","cpd00003","cpd00004",
                 "cpd00006","cpd00005","cpd00011","cpd00007","cpd00012","cpd00013","cpd00010",
                 "cpd00971","cpd15561","cpd15560","cpd11620","cpd11621","cpd11640","cpd11641"}
met_ids = [m.id for m in universal.metabolites]
is_cofactor = np.array([mid.rsplit("_",1)[0] in COFACTOR_IDS for mid in met_ids], dtype=bool)

# EC -> list of reaction indices covering it (for same-EC alternative check)
ec_to_rxns = {}
n_ec = mask.shape[1]
for e in range(n_ec):
    rs = np.where(mask[:, e] == 1)[0]
    if len(rs): ec_to_rxns[e] = rs

combos = [(org, pred) for org in GCA for pred in ("clean", "dpz", "enzbert")]
report = {}

for org, pred_name in combos:
    key = f"{org}_{pred_name}"
    npz = np.load(f"{OUT}/yvectors_{key}.npz", allow_pickle=True)
    allrxns_saved = list(npz["allrxns"]); meteor_keep = npz["meteor_keep"]; thresh_keep = npz["thresh_keep"]
    assert allrxns_saved == allrxns

    predf = extract_pred(resolve_baseline_pkl(pred_name, "vanilla", GCA[org], BASELINE_SUFFIX[pred_name]), anc)
    P = predf.values

    def max_ev(j):
        ei = np.where(mask[j] == 1)[0]
        if len(ei) == 0: return None
        return float(P[:, ei].max())

    only_thresh = np.where(thresh_keep & ~meteor_keep)[0]
    dropped_hi = [j for j in only_thresh if (max_ev(j) is not None and max_ev(j) >= 0.5)]

    n_has_ec_alt = n_has_met_alt = n_neither = n_both = 0
    examples_neither = []
    for j in dropped_hi:
        ei = np.where(mask[j] == 1)[0]
        ec_alt = False
        for e in ei:
            for rj in ec_to_rxns.get(e, []):
                if rj != j and meteor_keep[rj]:
                    ec_alt = True; break
            if ec_alt: break

        col = S[:, j]
        met_rows = [mr for mr in col.nonzero()[0] if not is_cofactor[mr]]
        met_alt = True  # assume connected unless we find a NON-COFACTOR metabolite with no other producer/consumer
        if len(met_rows) == 0:
            met_alt = False  # only cofactors touched -> no real substrate/product connectivity signal
        else:
            for mr in met_rows:
                row = S[mr, :]
                touching = row.nonzero()[1]
                others = [t for t in touching if t != j]
                if not any(meteor_keep[t] for t in others):
                    met_alt = False
                    break

        if ec_alt and met_alt: n_both += 1
        elif ec_alt and not met_alt: n_has_ec_alt += 1
        elif met_alt and not ec_alt: n_has_met_alt += 1
        else:
            n_neither += 1
            if len(examples_neither) < 10:
                examples_neither.append(dict(rxn=allrxns[j], name=universal.reactions[j].name,
                                              score=round(max_ev(j), 3)))

    n_tot = len(dropped_hi)
    report[key] = dict(n_dropped_high_conf=n_tot,
                        pct_both_backup=round(100*n_both/max(1,n_tot),1),
                        pct_ec_alt_only=round(100*n_has_ec_alt/max(1,n_tot),1),
                        pct_met_alt_only=round(100*n_has_met_alt/max(1,n_tot),1),
                        pct_no_backup_at_all=round(100*n_neither/max(1,n_tot),1),
                        n_no_backup_at_all=n_neither,
                        examples_no_backup=examples_neither)
    print(key, report[key], flush=True)

json.dump(report, open(f"{OUT}/check_dropped_alternatives.json", "w"), indent=1)
print("\n=== SUMMARY across 9 combos ===")
tot_dropped = sum(v["n_dropped_high_conf"] for v in report.values())
tot_no_backup = sum(v["n_no_backup_at_all"] for v in report.values())
print(f"total high-conf dropped reactions: {tot_dropped}")
print(f"of those, with NEITHER same-EC alt NOR metabolite connectivity in METEOR's own network: {tot_no_backup} ({100*tot_no_backup/max(1,tot_dropped):.1f}%)")
print(f"\n-> {OUT}/check_dropped_alternatives.json")
