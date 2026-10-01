"""Round 14: two follow-up checks reusing the 9 saved y-vectors from
yvector_all9.py (no MILP re-solve needed).

Check 2: quantify how many thresh-only reactions are HIGH-confidence
(EC score >= 0.5) reactions that METEOR's MILP actively excluded despite
them clearing the threshold -- i.e. reactions METEOR "gave up" for
parsimony/network reasons, not for lack of evidence.

Check 1: for thresh-only reactions with ZERO evidence (orphan, no EC score
at all) and NOT already caught by the eukaryote-keyword filter, inspect
their names/subsystems for patterns -- other non-bacterial-sounding
reactions, generic "transport via diffusion" reactions, etc. -- to see if
there's a second identifiable category beyond "Mitochondrial/Golgi/...".

Writes only under psb_revision/feasibility_essentiality/results/forensic/.
"""
import sys, json, collections
import numpy as np
sys.path.insert(0, '/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval')
from _env import *  # noqa
from meteor_v8.utils import load_universal, build_rxn_ec_mask, extract_pred, load_refmapping, load_ec, data_path, data_dir
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX

HERE = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality"
OUT = f"{HERE}/results/forensic"
GCA = {"salmonella": "GCF_000006945.2", "kpneumoniae": "GCF_058435815.1", "pputida": "GCF_045571375.1"}
EUK_KEYWORDS = ["mitochondrial", "golgi", "nucleus", "peroxisome", "lysosome",
                "endoplasmic", "chloroplast", "vacuole", "vesicle", "nuclear"]

universal, allrxns, allmet = load_universal()
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path("all_ancestors.txt"))
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
ix = {rid: i for i, rid in enumerate(allrxns)}
rxn_name = {allrxns[j]: (universal.reactions[j].name or "") for j in range(len(allrxns))}
euk_flag = np.zeros(len(allrxns), dtype=bool)
for j, r in enumerate(universal.reactions):
    if any(kw in (r.name or "").lower() for kw in EUK_KEYWORDS): euk_flag[j] = True

combos = [(org, pred) for org in GCA for pred in ("clean", "dpz", "enzbert")]

check2 = {}
check1_samples = collections.Counter()
check1_examples = []

for org, pred_name in combos:
    key = f"{org}_{pred_name}"
    npz = np.load(f"{OUT}/yvectors_{key}.npz", allow_pickle=True)
    allrxns_saved = list(npz["allrxns"]); meteor_keep = npz["meteor_keep"]; thresh_keep = npz["thresh_keep"]
    assert allrxns_saved == allrxns, f"reaction order mismatch for {key}"

    predf = extract_pred(resolve_baseline_pkl(pred_name, "vanilla", GCA[org], BASELINE_SUFFIX[pred_name]), anc)
    P = predf.values

    def max_ev(j):
        ei = np.where(mask[j] == 1)[0]
        if len(ei) == 0: return None
        return float(P[:, ei].max())

    only_thresh = np.where(thresh_keep & ~meteor_keep)[0]
    n_only_thresh = len(only_thresh)
    n_hi = n_lo = n_zero = 0
    hi_names = []
    for j in only_thresh:
        s = max_ev(j)
        if s is None: n_zero += 1
        elif s >= 0.5:
            n_hi += 1
            hi_names.append((allrxns[j], round(s,3), rxn_name.get(allrxns[j],"")))
        else:
            n_lo += 1

    check2[key] = dict(n_only_thresh=n_only_thresh, n_high_conf_dropped_by_meteor=n_hi,
                        n_low_conf=n_lo, n_zero_evidence=n_zero,
                        pct_high_conf_dropped=round(100*n_hi/max(1,n_only_thresh),1),
                        pct_zero_evidence=round(100*n_zero/max(1,n_only_thresh),1),
                        sample_high_conf_dropped=hi_names[:8])

    # check 1: zero-evidence, non-euk-flagged reactions -- collect name keywords
    for j in only_thresh:
        s = max_ev(j)
        if s is not None: continue  # only zero-evidence
        if euk_flag[j]: continue    # already explained by euk filter
        nm = rxn_name.get(allrxns[j], "").strip()
        rid = allrxns[j]
        if nm:
            # crude tokenization: last word often names the process type
            toks = nm.lower().replace(",", " ").replace("(", " ").replace(")", " ").split()
            for t in toks:
                if len(t) > 3:
                    check1_samples[t] += 1
        check1_examples.append(dict(combo=key, rxn=rid, name=nm))

json.dump(check2, open(f"{OUT}/check2_high_conf_dropped.json", "w"), indent=1)
json.dump(dict(top_keywords=check1_samples.most_common(40),
               n_examples=len(check1_examples), examples=check1_examples[:150]),
          open(f"{OUT}/check1_zero_evidence_nonEuk.json", "w"), indent=1)

print("=== CHECK 2: high-confidence (>=0.5) reactions METEOR actively dropped, per combo ===")
for k, v in check2.items():
    print(f"{k:24s} n_only_thresh={v['n_only_thresh']:4d}  high_conf_dropped={v['n_high_conf_dropped_by_meteor']:3d} ({v['pct_high_conf_dropped']:5.1f}%)  zero_evidence={v['n_zero_evidence']:3d} ({v['pct_zero_evidence']:5.1f}%)")

print("\n=== CHECK 1: top name-keywords among zero-evidence, non-eukaryote-flagged thresh-only reactions ===")
for word, cnt in check1_samples.most_common(30):
    print(f"  {word:20s} {cnt}")

print(f"\n-> {OUT}/check2_high_conf_dropped.json")
print(f"-> {OUT}/check1_zero_evidence_nonEuk.json")
