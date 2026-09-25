"""Exact candidate-set partition per genome, averaged (Fig 1b step 3). Run: python eval/gen_fig_pipeline_partition.py
K skeleton, X excluded (blocked), M medium, E evidence (w>=0.01).
From set checks on the candidate masks: M subset of K, M∩X=∅, |K∩X|=4 (2 without EC, 2 EC-mapped: rxn00724_c, rxn03234_c).
k_g = # of the 2 EC-mapped K∩X reactions with evidence in genome g (kx json; 0 if absent)."""
import json, os, statistics as st
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "results/candidate_size_dpz_vanilla.json")))["rows"]
kx = json.load(open(os.path.join(ROOT, "results/candidate_partition_kx_evidence.json")))
rows = []
for r in d:
    k = kx.get(r["gca"], 0); K = r["n_skeleton"]
    a = r["n_skel_without_evidence"] - 4 + k          # K\X, no evidence (includes 36 medium)
    b = K - r["n_skel_without_evidence"] - k          # K\X, with evidence
    c = r["n_candidate"] - (K - 4)                    # E \ (K ∪ X)
    exE = r["n_evidence"] - (K - r["n_skel_without_evidence"]) - c   # evidence reactions dropped as blocked (outside K)
    assert a + b + c == r["n_candidate"] and c >= 0 and exE >= 0
    rows.append(dict(a=a, b=b, c=c, exE=exE, cand=r["n_candidate"], ev=r["n_evidence"], K=K))
m = {k: st.mean(x[k] for x in rows) for k in rows[0]}
print({k: round(v, 1) for k, v in m.items()}, "n =", len(rows), "kx provided:", bool(kx))
json.dump(dict(skel_noev=m["a"], skel_ev=m["b"], ev_only=m["c"], candidate=m["cand"], universal=47880,
               selected=3121, evidence_blocked=m["exE"]), open(os.path.join(ROOT, "figures/fig1_partition.json"), "w"), indent=1)
