"""Round 11b: fixed neighbor-trace step, reusing the already-saved y-vectors
(no MILP re-solve). Fixes the sparse-matrix bug in yvector_trace.py's
neighbor loop (np.abs(col)>1e-9 on a scipy.sparse column raised
"truth value of an array... is ambiguous").

Writes only under psb_revision/feasibility_essentiality/results/forensic/.
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
import warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("cobra").setLevel(logging.ERROR)

R = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision"
HERE = f"{R}/feasibility_essentiality"
OUT = f"{HERE}/results/forensic"
sys.path.insert(0, f"{R}/eval")
from _env import *  # noqa
from meteor_v8.utils import load_universal, extract_fba_matrices, load_tight_bounds

T0 = time.time()
def tick(k): print(f"[{round(time.time()-T0,1):7.1f}s] {k}", flush=True)

npz = np.load(f"{OUT}/yvectors_salmonella_dpz.npz", allow_pickle=True)
allrxns_saved = list(npz["allrxns"])
meteor_keep = npz["meteor_keep"]
thresh_keep = npz["thresh_keep"]
tick(f"loaded saved y-vectors: n_meteor={meteor_keep.sum()} n_thresh={thresh_keep.sum()}")

universal, allrxns, allmet = load_universal()
assert allrxns == allrxns_saved, "reaction order mismatch vs saved npz"
S, lb0, ub0 = extract_fba_matrices(universal, allrxns, reversed_trans=True)
if not sp.issparse(S):
    S = sp.csc_matrix(S)
else:
    S = S.tocsc()
ix = {rid: i for i, rid in enumerate(allrxns)}
met_ids = [m.id for m in universal.metabolites]
tick("reloaded universal + S (no MILP re-solve)")

KEYWORDS = ["hypoxanthine phosphoribosyltransferase", "xanthine phosphoribosyltransferase",
            "adenine phosphoribosyltransferase", "uracil phosphoribosyltransferase",
            "purine-nucleoside phosphorylase", "purine nucleoside phosphorylase",
            "nucleoside permease", "xanthine permease", "uracil permease", "adenine permease",
            "hypoxanthine permease", "purine permease"]
salvage_idx = {}
for r in universal.reactions:
    nm = (r.name or "").lower()
    for kw in KEYWORDS:
        if kw in nm:
            salvage_idx.setdefault(kw, []).append(r.id)

salvage_report = {}
for kw, rids in salvage_idx.items():
    for rid in rids:
        i = ix.get(rid)
        if i is None: continue
        salvage_report[rid] = dict(keyword=kw, rxn=rid, in_meteor=bool(meteor_keep[i]), in_thresh=bool(thresh_keep[i]))

# ---- de novo purine/pyrimidine pathway reactions: from the group_a gene list
# in forensic.py's salmonella_deepdive_dpz.json (purH/purN/purK/purL/purM/purF/
# purA/purB/pyrE/pyrF-type products), find their reaction IDs via GPR match on
# the protein accessions, so we can trace THEIR neighbors too, not just salvage.
group_a_products = ["adenylosuccinate lyase", "argininosuccinate lyase", "argininosuccinate synthase",
    "orotidine-5'-phosphate decarboxylase", "phosphoribosylaminoimidazolecarboxamide",
    "phosphoribosylamine--glycine ligase", "phosphoribosylaminoimidazolesuccinocarboxamide synthase",
    "phosphoribosylglycinamide formyltransferase", "orotate phosphoribosyltransferase",
    "phosphoribosylformylglycinamidine synthase", "phosphoribosylformylglycinamidine cyclo-ligase",
    "amidophosphoribosyltransferase"]
denovo_idx = {}
for r in universal.reactions:
    nm = (r.name or "").lower()
    for kw in group_a_products:
        if kw.lower() in nm:
            denovo_idx.setdefault(kw, []).append(r.id)
denovo_report = {}
for kw, rids in denovo_idx.items():
    for rid in rids:
        i = ix.get(rid)
        if i is None: continue
        denovo_report[rid] = dict(keyword=kw, rxn=rid, in_meteor=bool(meteor_keep[i]), in_thresh=bool(thresh_keep[i]))

def neighbors_of(rid, S, ix, allrxns, met_ids, meteor_keep, thresh_keep):
    j = ix.get(rid)
    if j is None: return []
    col = S[:, j]
    met_rows = col.nonzero()[0]
    diffs = []
    seen = set()
    for mr in met_rows:
        row = S[mr, :]
        touching = row.nonzero()[1]
        for tj in touching:
            if tj == j or tj in seen: continue
            seen.add(tj)
            om = bool(thresh_keep[tj]) and not bool(meteor_keep[tj])
            oM = bool(meteor_keep[tj]) and not bool(thresh_keep[tj])
            if om or oM:
                diffs.append(dict(rxn=allrxns[tj], name=universal.reactions[tj].name,
                                   only_in_thresh=om, only_in_meteor=oM, shared_met=met_ids[mr]))
    return diffs

neighbor_report = {}
for rid, row in list(salvage_report.items()) + list(denovo_report.items()):
    diffs = neighbors_of(rid, S, ix, allrxns, met_ids, meteor_keep, thresh_keep)
    if diffs:
        neighbor_report[rid] = diffs
tick("neighbor trace done")

# ---- what's actually different network-wide between the two arms ----
only_meteor = meteor_keep & ~thresh_keep
only_thresh = thresh_keep & ~meteor_keep
only_thresh_rxns = [dict(rxn=allrxns[i], name=universal.reactions[i].name) for i in np.where(only_thresh)[0]]
only_meteor_rxns = [dict(rxn=allrxns[i], name=universal.reactions[i].name) for i in np.where(only_meteor)[0]]

out = dict(
    n_meteor_selected=int(meteor_keep.sum()), n_thresh_selected=int(thresh_keep.sum()),
    n_only_meteor=int(only_meteor.sum()), n_only_thresh=int(only_thresh.sum()),
    salvage_report=salvage_report, denovo_report=denovo_report, neighbor_report=neighbor_report,
    only_thresh_rxns=only_thresh_rxns, only_meteor_rxns=only_meteor_rxns,
)
json.dump(out, open(f"{OUT}/yvector_trace_salmonella_dpz.json", "w"), indent=1)
tick("json written")
print("n_only_meteor:", out["n_only_meteor"], "n_only_thresh:", out["n_only_thresh"])
print("neighbor_report keys with diffs:", list(neighbor_report.keys()))
for rid, diffs in neighbor_report.items():
    print(rid, "->", len(diffs), "diffs")
    for d in diffs[:15]:
        print("   ", d)
print("\n-> ", f"{OUT}/yvector_trace_salmonella_dpz.json")
