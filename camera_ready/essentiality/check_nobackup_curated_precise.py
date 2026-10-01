"""Precise re-check of the 95 'no-backup' high-confidence METEOR-dropped
reactions against each organism's curated GEM, using exact cross-references
(seed.reaction ModelSEED ID, and EC-code as fallback) instead of the flawed
name-token matching from check_nobackup_full.py.

No MILP re-solve; reuses saved y-vectors + predictor scores + the
already-computed nobackup_full_classification.json for the reaction list.
"""
import sys, json
import numpy as np
sys.path.insert(0, '/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval')
from _env import *  # noqa
from meteor_v8.utils import load_universal, build_rxn_ec_mask, load_refmapping, load_ec, data_path, data_dir
import cobra

HERE = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality"
OUT = f"{HERE}/results/forensic"
CURATED_GEM = {"salmonella": "STM_v1_0", "kpneumoniae": "iYL1228", "pputida": "iJN1463"}

universal, allrxns, allmet = load_universal()
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path("all_ancestors.txt"))  # index-aligned EC vocabulary for mask columns
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
ix = {rid: i for i, rid in enumerate(allrxns)}

def ecs_for(rid):
    j = ix[rid]
    ei = np.where(mask[j] == 1)[0]
    return {anc[e] for e in ei}

# build curated-model cross-reference sets per organism: seed reaction IDs + ECs
curated_seed_ids = {}
curated_ecs = {}
for org, gem in CURATED_GEM.items():
    m = cobra.io.read_sbml_model(f"{GEMDIR}/{gem}.xml")
    seeds = set(); ecs = set()
    for r in m.reactions:
        ann = r.annotation or {}
        sid = ann.get("seed.reaction")
        if sid:
            if isinstance(sid, list): seeds.update(sid)
            else: seeds.add(sid)
        ec = ann.get("ec-code")
        if ec:
            if isinstance(ec, list): ecs.update(ec)
            else: ecs.add(ec)
    curated_seed_ids[org] = seeds
    curated_ecs[org] = ecs
    print(f"{org} ({gem}): {len(seeds)} seed.reaction xrefs, {len(ecs)} EC xrefs", flush=True)

nb = json.load(open(f"{OUT}/nobackup_full_classification.json"))["all_reactions"]

results = []
for r in nb:
    rid = r["rxn"]; org = r["org"]
    seed_id = rid[:-2] if rid.endswith(("_c", "_e", "_p")) else rid  # strip compartment suffix
    ecs = ecs_for(rid)
    in_curated_by_id = seed_id in curated_seed_ids[org]
    in_curated_by_ec = bool(ecs & curated_ecs[org]) if ecs else None
    results.append(dict(r, seed_id=seed_id, ecs=sorted(ecs),
                         in_curated_by_seedid=in_curated_by_id,
                         in_curated_by_ec=in_curated_by_ec))

n = len(results)
n_by_id = sum(1 for r in results if r["in_curated_by_seedid"])
n_by_ec_true = sum(1 for r in results if r["in_curated_by_ec"] is True)
n_by_ec_false = sum(1 for r in results if r["in_curated_by_ec"] is False)
n_by_ec_none = sum(1 for r in results if r["in_curated_by_ec"] is None)
n_absent_both = sum(1 for r in results if (not r["in_curated_by_seedid"]) and (r["in_curated_by_ec"] is not True))

domain_flagged = [r for r in results if r["domain_categories"]]
non_flagged = [r for r in results if not r["domain_categories"]]

def summarize(rows, label):
    n = len(rows)
    nid = sum(1 for r in rows if r["in_curated_by_seedid"])
    nec = sum(1 for r in rows if r["in_curated_by_ec"] is True)
    nabs = sum(1 for r in rows if (not r["in_curated_by_seedid"]) and (r["in_curated_by_ec"] is not True))
    print(f"{label}: n={n}  present_by_exact_seedID={nid} ({100*nid/max(1,n):.1f}%)  "
          f"present_by_EC={nec} ({100*nec/max(1,n):.1f}%)  "
          f"ABSENT_from_curated={nabs} ({100*nabs/max(1,n):.1f}%)")

print(f"\n=== precise curated-GEM cross-reference, n={n} 'no-backup' reactions ===")
print(f"present by exact seed.reaction ID match: {n_by_id} ({100*n_by_id/n:.1f}%)")
print(f"present by EC-code match (of those with EC evidence): {n_by_ec_true} / {n_by_ec_true+n_by_ec_false} scored")
print(f"absent from curated model by BOTH criteria: {n_absent_both} ({100*n_absent_both/n:.1f}%)")
print()
summarize(domain_flagged, "domain-flagged subset (n=22 expected)")
summarize(non_flagged, "non-domain-flagged subset")

json.dump(dict(n_total=n, n_present_by_seedid=n_by_id, n_absent_both=n_absent_both,
               domain_flagged_summary=dict(n=len(domain_flagged),
                   present=sum(1 for r in domain_flagged if r["in_curated_by_seedid"] or r["in_curated_by_ec"] is True),
                   absent=sum(1 for r in domain_flagged if (not r["in_curated_by_seedid"]) and (r["in_curated_by_ec"] is not True))),
               non_flagged_summary=dict(n=len(non_flagged),
                   present=sum(1 for r in non_flagged if r["in_curated_by_seedid"] or r["in_curated_by_ec"] is True),
                   absent=sum(1 for r in non_flagged if (not r["in_curated_by_seedid"]) and (r["in_curated_by_ec"] is not True))),
               all_reactions=results),
          open(f"{OUT}/nobackup_curated_precise.json", "w"), indent=1)
print(f"\n-> {OUT}/nobackup_curated_precise.json")
