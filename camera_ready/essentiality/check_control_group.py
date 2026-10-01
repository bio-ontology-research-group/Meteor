"""Control group for nobackup_curated_precise.json: among reactions METEOR
ITSELF selected (meteor_keep=True) with the SAME confidence bar (predictor
EC score >= 0.5), what fraction are present in the organism's curated GEM
(by exact seed.reaction ID, or EC-code match)? Compare against the 95
'no-backup dropped' set's presence rate (19.0%: 18/95 by ID+EC combined,
or 5.3%/18.9% separately) to know whether METEOR's drops are meaningfully
worse-curated-alignment than its own picks, or indistinguishable.

No MILP re-solve; reuses saved y-vectors + predictor scores.
"""
import sys, json
import numpy as np
sys.path.insert(0, '/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval')
from _env import *  # noqa
from meteor_v8.utils import load_universal, build_rxn_ec_mask, extract_pred, load_refmapping, load_ec, data_path, data_dir
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
import cobra

HERE = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality"
OUT = f"{HERE}/results/forensic"
GCA = {"salmonella": "GCF_000006945.2", "kpneumoniae": "GCF_058435815.1", "pputida": "GCF_045571375.1"}
CURATED_GEM = {"salmonella": "STM_v1_0", "kpneumoniae": "iYL1228", "pputida": "iJN1463"}

universal, allrxns, allmet = load_universal()
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path("all_ancestors.txt"))
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
ix = {rid: i for i, rid in enumerate(allrxns)}

curated_seed_ids = {}; curated_ecs = {}
for org, gem in CURATED_GEM.items():
    m = cobra.io.read_sbml_model(f"{GEMDIR}/{gem}.xml")
    seeds = set(); ecs = set()
    for r in m.reactions:
        ann = r.annotation or {}
        sid = ann.get("seed.reaction")
        if sid: seeds.update(sid if isinstance(sid, list) else [sid])
        ec = ann.get("ec-code")
        if ec: ecs.update(ec if isinstance(ec, list) else [ec])
    curated_seed_ids[org] = seeds; curated_ecs[org] = ecs

combos = [(org, pred) for org in GCA for pred in ("clean", "dpz", "enzbert")]
baseline_rows = []

for org, pred_name in combos:
    key = f"{org}_{pred_name}"
    npz = np.load(f"{OUT}/yvectors_{key}.npz", allow_pickle=True)
    allrxns_saved = list(npz["allrxns"]); meteor_keep = npz["meteor_keep"]
    assert allrxns_saved == allrxns

    predf = extract_pred(resolve_baseline_pkl(pred_name, "vanilla", GCA[org], BASELINE_SUFFIX[pred_name]), anc)
    P = predf.values
    def max_ev(j):
        ei = np.where(mask[j] == 1)[0]
        if len(ei) == 0: return None
        return float(P[:, ei].max())

    meteor_hi = [j for j in np.where(meteor_keep)[0] if (max_ev(j) is not None and max_ev(j) >= 0.5)]
    for j in meteor_hi:
        rid = allrxns[j]
        seed_id = rid[:-2] if rid.endswith(("_c","_e","_p")) else rid
        ecs = {anc[e] for e in np.where(mask[j]==1)[0]}
        in_id = seed_id in curated_seed_ids[org]
        in_ec = bool(ecs & curated_ecs[org])
        baseline_rows.append(dict(combo=key, org=org, rxn=rid, in_curated_by_seedid=in_id, in_curated_by_ec=in_ec))
    print(f"{key}: n_meteor_selected_highconf={len(meteor_hi)}", flush=True)

n = len(baseline_rows)
n_id = sum(1 for r in baseline_rows if r["in_curated_by_seedid"])
n_ec = sum(1 for r in baseline_rows if r["in_curated_by_ec"])
n_either = sum(1 for r in baseline_rows if r["in_curated_by_seedid"] or r["in_curated_by_ec"])

print(f"\n=== CONTROL GROUP: METEOR's own selected high-confidence (>=0.5) reactions, n={n} ===")
print(f"present by exact seed.reaction ID: {n_id} ({100*n_id/n:.1f}%)")
print(f"present by EC-code match: {n_ec} ({100*n_ec/n:.1f}%)")
print(f"present by EITHER: {n_either} ({100*n_either/n:.1f}%)")

print(f"\n=== COMPARISON ===")
print(f"METEOR-dropped 'no-backup' set (n=95): present by ID=5.3%, by EC=18.9%")
print(f"METEOR's own selected set     (n={n}): present by ID={100*n_id/n:.1f}%, by EC={100*n_ec/n:.1f}%")

json.dump(dict(n=n, n_present_by_id=n_id, n_present_by_ec=n_ec, n_present_by_either=n_either,
               pct_present_by_id=round(100*n_id/n,1), pct_present_by_ec=round(100*n_ec/n,1),
               pct_present_by_either=round(100*n_either/n,1), rows=baseline_rows),
          open(f"{OUT}/control_group_meteor_selected.json", "w"), indent=1)
print(f"\n-> {OUT}/control_group_meteor_selected.json")
