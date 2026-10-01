"""Full quantification of the 95 'no-backup' high-confidence reactions
METEOR dropped (across 9 combos), classified as domain-inappropriate
(photosynthesis/archaeal/plant-secondary-metabolism/eukaryotic-signaling,
on top of the earlier Mitochondrial/Golgi/etc. keyword set) vs plausible-
but-missing bacterial reactions. Also cross-checks each against the
organism's own curated GEM (by reaction NAME keyword match) to see whether
expert-curated models also lack these reactions.
No MILP re-solve; reuses saved y-vectors + predictor scores.
"""
import sys, json, re
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, '/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval')
from _env import *  # noqa
from meteor_v8.utils import (load_universal, extract_fba_matrices, build_rxn_ec_mask,
    extract_pred, load_refmapping, load_ec, data_path, data_dir)
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
import cobra

HERE = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality"
OUT = f"{HERE}/results/forensic"
GCA = {"salmonella": "GCF_000006945.2", "kpneumoniae": "GCF_058435815.1", "pputida": "GCF_045571375.1"}
CURATED_GEM = {"salmonella": "STM_v1_0", "kpneumoniae": "iYL1228", "pputida": "iJN1463"}

DOMAIN_KEYWORDS = {
    "eukaryote_organelle": ["mitochondrial", "golgi", "nucleus", "peroxisome", "lysosome",
                             "endoplasmic", "chloroplast", "vacuole", "vesicle", "nuclear"],
    "photosynthesis_plant_pigment": ["phytoene", "plastoquinone", "phytofluene", "carotene",
                                      "chlorophyll", "photosystem"],
    "archaeal_methanogenesis": ["coenzyme b", "coenzyme m", "methanophenazine", "methanogen",
                                 "tetrahydromethanopterin"],
    "plant_secondary_metabolism": ["cinnamyl", "coniferyl", "sinapyl", "coumaryl", "coumaroyl",
                                    "flavonol", "flavonoid", "lignin", "anthocyanin", "rhamnosyl"],
    "eukaryotic_signaling": ["inositol", "phosphoinositide", "diacylglycerol kinase"],
}
ALL_KEYWORDS = [(cat, kw) for cat, kws in DOMAIN_KEYWORDS.items() for kw in kws]

def classify(name):
    nm = name.lower()
    hits = [cat for cat, kw in ALL_KEYWORDS if kw in nm]
    return sorted(set(hits))

universal, allrxns, allmet = load_universal()
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path("all_ancestors.txt"))
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
ix = {rid: i for i, rid in enumerate(allrxns)}
S, lb0, ub0 = extract_fba_matrices(universal, allrxns, reversed_trans=True)
S = S.tocsc() if sp.issparse(S) else sp.csc_matrix(S)

COFACTOR_IDS = {"cpd00001","cpd00002","cpd00008","cpd00009","cpd00067","cpd00003","cpd00004",
                 "cpd00006","cpd00005","cpd00011","cpd00007","cpd00012","cpd00013","cpd00010",
                 "cpd00971","cpd15561","cpd15560","cpd11620","cpd11621","cpd11640","cpd11641"}
met_ids = [m.id for m in universal.metabolites]
is_cofactor = np.array([mid.rsplit("_",1)[0] in COFACTOR_IDS for mid in met_ids], dtype=bool)

ec_to_rxns = {}
for e in range(mask.shape[1]):
    rs = np.where(mask[:, e] == 1)[0]
    if len(rs): ec_to_rxns[e] = rs

# load curated GEMs once, build a normalized name-token set per organism
def curated_name_blob(gem_id):
    path = f"{GEMDIR}/{gem_id}.xml"
    m = cobra.io.read_sbml_model(path)
    names = " || ".join((r.name or "") for r in m.reactions).lower()
    return names

curated_blob = {org: curated_name_blob(CURATED_GEM[org]) for org in GCA}
print("curated GEM name blobs loaded", {k: len(v) for k, v in curated_blob.items()}, flush=True)

def curated_has_match(name, blob):
    """crude check: does any distinctive >=5-char token from this reaction's
    name also appear in the curated model's reaction-name text?"""
    toks = [t for t in re.split(r'[^a-z0-9]+', name.lower()) if len(t) >= 6]
    if not toks: return None
    return any(t in blob for t in toks)

combos = [(org, pred) for org in GCA for pred in ("clean", "dpz", "enzbert")]
all_nobackup = []

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

    for j in dropped_hi:
        ei = np.where(mask[j] == 1)[0]
        ec_alt = any(meteor_keep[rj] for e in ei for rj in ec_to_rxns.get(e, []) if rj != j)
        col = S[:, j]
        met_rows = [mr for mr in col.nonzero()[0] if not is_cofactor[mr]]
        met_alt = False if len(met_rows) == 0 else all(
            any(meteor_keep[t] for t in [tt for tt in S[mr, :].nonzero()[1] if tt != j])
            for mr in met_rows
        )
        if ec_alt or met_alt: continue  # only care about the "no backup" set
        nm = universal.reactions[j].name or ""
        cats = classify(nm)
        curated_match = curated_has_match(nm, curated_blob[org])
        all_nobackup.append(dict(combo=key, org=org, rxn=allrxns[j], name=nm,
                                  score=round(max_ev(j),3), domain_categories=cats,
                                  in_curated_gem_by_name=curated_match))

n_tot = len(all_nobackup)
n_domain_flagged = sum(1 for r in all_nobackup if r["domain_categories"])
n_curated_match_true = sum(1 for r in all_nobackup if r["in_curated_gem_by_name"] is True)
n_curated_match_false = sum(1 for r in all_nobackup if r["in_curated_gem_by_name"] is False)
n_curated_match_none = sum(1 for r in all_nobackup if r["in_curated_gem_by_name"] is None)

cat_counts = {}
for r in all_nobackup:
    for c in r["domain_categories"]:
        cat_counts[c] = cat_counts.get(c, 0) + 1

json.dump(dict(n_total_nobackup=n_tot, n_domain_flagged=n_domain_flagged,
               pct_domain_flagged=round(100*n_domain_flagged/max(1,n_tot),1),
               category_counts=cat_counts,
               n_curated_match_true=n_curated_match_true, n_curated_match_false=n_curated_match_false,
               n_curated_match_none=n_curated_match_none, all_reactions=all_nobackup),
          open(f"{OUT}/nobackup_full_classification.json", "w"), indent=1)

print(f"\n=== FULL 'no-backup' set: n={n_tot} ===")
print(f"domain-inappropriate keyword hit: {n_domain_flagged} ({100*n_domain_flagged/max(1,n_tot):.1f}%)")
print("category breakdown:", cat_counts)
print(f"curated GEM name-match: YES={n_curated_match_true}  NO={n_curated_match_false}  no-distinctive-token={n_curated_match_none}")
print(f"\namong domain-flagged reactions, curated-GEM match:")
df = [r for r in all_nobackup if r["domain_categories"]]
print(f"  YES={sum(1 for r in df if r['in_curated_gem_by_name'] is True)}  NO={sum(1 for r in df if r['in_curated_gem_by_name'] is False)}  none={sum(1 for r in df if r['in_curated_gem_by_name'] is None)}")
print(f"\namong NON-domain-flagged reactions, curated-GEM match:")
ndf = [r for r in all_nobackup if not r["domain_categories"]]
print(f"  YES={sum(1 for r in ndf if r['in_curated_gem_by_name'] is True)}  NO={sum(1 for r in ndf if r['in_curated_gem_by_name'] is False)}  none={sum(1 for r in ndf if r['in_curated_gem_by_name'] is None)}")

print("\n-> ", f"{OUT}/nobackup_full_classification.json")
