#!/usr/bin/env python3
"""Pathway-level case study: how METEOR selects different metabolic pathways
across bacteria. Examines MILP active reaction sets across 109 genomes.

Identifies:
1. Differentially activated reactions (high variance across genomes)
2. Biologically meaningful EC classes in those reactions
3. Specific genome pairs with contrasting pathway choices
4. Cross-baseline consistency of pathway selection
"""
import os, json, pickle, sys, time
import numpy as np
from collections import Counter, defaultdict

V7_RUN = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run"
W = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026"
DATA = "/ibex/user/niuk0a/funcarve/cobra/v6/data"

manifest = json.load(open(f"{W}/scripts/genome_collection_manifest.json"))
all_gcfs = sorted(g for g in manifest["assemblies"].keys() if g.startswith("GCF"))

uni = pickle.load(open(f"{DATA}/universal.pickle", "rb"))
rxn_ids = [r.id for r in uni.reactions]
rxn_names = {r.id: r.name for r in uni.reactions}
n_rxns = len(rxn_ids)

seedr2ec = pickle.load(open(f"{DATA}/seedr2ec.pkl", "rb"))
seedec2r = pickle.load(open(f"{DATA}/seedec2r.pkl", "rb"))

EC_CLASS_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}

NOTABLE_ECS = {
    "1.18.6.1": "nitrogenase (nitrogen fixation)",
    "1.9.3.1": "cytochrome c oxidase (aerobic respiration)",
    "1.3.5.1": "succinate dehydrogenase (TCA/Complex II)",
    "7.1.1.1": "NADH:ubiquinone oxidoreductase (Complex I)",
    "7.1.1.2": "NADH:ubiquinone oxidoreductase (Complex I)",
    "1.6.5.5": "NADH:quinone oxidoreductase",
    "1.2.7.1": "pyruvate:ferredoxin oxidoreductase (anaerobic)",
    "1.12.7.2": "ferredoxin hydrogenase",
    "2.7.1.11": "phosphofructokinase (glycolysis)",
    "4.1.1.32": "PEPCK (gluconeogenesis)",
    "6.4.1.1": "pyruvate carboxylase (anaplerosis)",
    "4.2.1.2": "fumarase (TCA cycle)",
    "1.1.1.42": "isocitrate dehydrogenase (TCA)",
    "2.3.3.1": "citrate synthase (TCA)",
    "2.8.3.18": "succinyl-CoA transferase",
    "1.4.1.13": "glutamate synthase (nitrogen assimilation)",
    "6.3.1.2": "glutamine synthetase",
    "1.7.2.1": "nitrite reductase (denitrification)",
    "1.7.99.4": "nitrate reductase",
    "2.4.99.28": "lipopolysaccharide glycosyltransferase",
    "3.4.16.4": "metallocarboxypeptidase",
}

print("Loading MILP solutions for DPZ-vanilla across 109 genomes...", flush=True)
t0 = time.time()

meteor_dir = f"{V7_RUN}/meteor_out/dpz_vanilla"
Y = np.zeros((len(all_gcfs), n_rxns), dtype=np.int8)
gcf_valid = []
gcf_nactive = {}

for i, gcf in enumerate(all_gcfs):
    sol_path = os.path.join(meteor_dir, f"meteor_sol_{gcf}.pkl")
    if not os.path.exists(sol_path):
        continue
    sol = pickle.load(open(sol_path, "rb"))
    y = sol["y_vals"]
    Y[i, :] = (y > 0.5).astype(np.int8)
    gcf_valid.append(i)
    gcf_nactive[gcf] = int(np.sum(Y[i, :]))

Y_valid = Y[gcf_valid, :]
valid_gcfs = [all_gcfs[i] for i in gcf_valid]
print(f"Loaded {len(gcf_valid)} genomes in {time.time()-t0:.0f}s", flush=True)

# ── Reaction activation frequency ──
rxn_freq = Y_valid.mean(axis=0)  # fraction of genomes activating each reaction
rxn_std = Y_valid.std(axis=0)

# Core reactions (active in >95% of genomes)
core = np.sum(rxn_freq > 0.95)
# Absent (active in <5%)
absent = np.sum(rxn_freq < 0.05)
# Variable (5-95%)
variable = np.sum((rxn_freq >= 0.05) & (rxn_freq <= 0.95))

print(f"\n{'='*60}", flush=True)
print("REACTION ACTIVATION LANDSCAPE", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Core reactions (>95% genomes): {core}", flush=True)
print(f"  Variable reactions (5-95%):    {variable}", flush=True)
print(f"  Rare/absent (<5%):             {absent}", flush=True)
print(f"  Mean active per genome:        {np.mean(list(gcf_nactive.values())):.0f} ± "
      f"{np.std(list(gcf_nactive.values())):.0f}", flush=True)

# ── Top variable reactions by EC class ──
var_idx = np.where((rxn_freq >= 0.1) & (rxn_freq <= 0.9))[0]
ec_class_var = Counter()
for idx in var_idx:
    rid = rxn_ids[idx]
    base_rxn = rid.split("_")[0] if "_" in rid else rid
    if base_rxn in seedr2ec and seedr2ec[base_rxn]:
        for ec in seedr2ec[base_rxn]:
            cl = ec.split(".")[0]
            if cl in EC_CLASS_NAMES:
                ec_class_var[cl] += 1

print(f"\n{'='*60}", flush=True)
print("VARIABLE REACTIONS BY EC CLASS", flush=True)
print(f"{'='*60}", flush=True)
for cl, cnt in sorted(ec_class_var.items()):
    print(f"  EC {cl} ({EC_CLASS_NAMES.get(cl, '?'):<20}): {cnt:>4} variable reactions", flush=True)

# ── Notable EC activation patterns ──
print(f"\n{'='*60}", flush=True)
print("NOTABLE EC ACTIVATION PATTERNS ACROSS 109 GENOMES", flush=True)
print(f"{'='*60}", flush=True)

for ec, desc in sorted(NOTABLE_ECS.items()):
    if ec not in seedec2r:
        continue
    rxn_bases = seedec2r[ec]
    # Find matching reaction indices
    active_genomes = set()
    for j, rid in enumerate(rxn_ids):
        base_rxn = rid.split("_")[0] if "_" in rid else rid
        if base_rxn in rxn_bases:
            for gi, g_idx in enumerate(gcf_valid):
                if Y_valid[gi, j] > 0:
                    active_genomes.add(gi)
    pct = 100 * len(active_genomes) / len(gcf_valid) if gcf_valid else 0
    if 5 < pct < 95:
        marker = " ← DIFFERENTIAL"
    elif pct >= 95:
        marker = " (core)"
    else:
        marker = " (rare)"
    print(f"  {ec:<12} {pct:>5.1f}% active  {desc}{marker}", flush=True)

# ── Genomes with extreme active-set sizes ──
sorted_genomes = sorted(gcf_nactive.items(), key=lambda x: x[1])
print(f"\n{'='*60}", flush=True)
print("GENOMES WITH SMALLEST/LARGEST ACTIVE REACTION SETS", flush=True)
print(f"{'='*60}", flush=True)
print("  Smallest:", flush=True)
for gcf, na in sorted_genomes[:5]:
    ftp = manifest["assemblies"].get(gcf, {}).get("ftp", "")
    name = manifest["assemblies"].get(gcf, {}).get("name", "")
    print(f"    {gcf:<25} {na:>5} active rxns  ({name})", flush=True)
print("  Largest:", flush=True)
for gcf, na in sorted_genomes[-5:]:
    name = manifest["assemblies"].get(gcf, {}).get("name", "")
    print(f"    {gcf:<25} {na:>5} active rxns  ({name})", flush=True)

# ── Pairwise genome comparison (most different pair) ──
print(f"\n{'='*60}", flush=True)
print("MOST DIVERGENT GENOME PAIRS (Jaccard distance on active sets)", flush=True)
print(f"{'='*60}", flush=True)

n_valid = len(gcf_valid)
max_dist = 0
max_pair = (0, 0)
# Sample 500 random pairs to avoid O(n^2)
np.random.seed(42)
pairs = [(np.random.randint(n_valid), np.random.randint(n_valid)) for _ in range(2000)]
for i, j in pairs:
    if i == j:
        continue
    intersection = np.sum(Y_valid[i] & Y_valid[j])
    union = np.sum(Y_valid[i] | Y_valid[j])
    if union == 0:
        continue
    dist = 1 - intersection / union
    if dist > max_dist:
        max_dist = dist
        max_pair = (i, j)

gi, gj = max_pair
gcf_i, gcf_j = valid_gcfs[gi], valid_gcfs[gj]
na_i, na_j = gcf_nactive[gcf_i], gcf_nactive[gcf_j]
only_i = np.sum(Y_valid[gi] & ~Y_valid[gj])
only_j = np.sum(~Y_valid[gi] & Y_valid[gj])
both = np.sum(Y_valid[gi] & Y_valid[gj])

print(f"  {gcf_i} ({na_i} active) vs {gcf_j} ({na_j} active)", flush=True)
print(f"  Jaccard distance: {max_dist:.3f}", flush=True)
print(f"  Shared: {both}, Only-A: {only_i}, Only-B: {only_j}", flush=True)

# What ECs differ between these two?
ec_only_i = Counter()
ec_only_j = Counter()
for idx in range(n_rxns):
    if Y_valid[gi, idx] != Y_valid[gj, idx]:
        rid = rxn_ids[idx]
        base_rxn = rid.split("_")[0] if "_" in rid else rid
        if base_rxn in seedr2ec and seedr2ec[base_rxn]:
            for ec in seedr2ec[base_rxn]:
                if Y_valid[gi, idx]:
                    ec_only_i[ec] += 1
                else:
                    ec_only_j[ec] += 1

print(f"\n  Top ECs unique to {gcf_i}:", flush=True)
for ec, cnt in ec_only_i.most_common(10):
    desc = NOTABLE_ECS.get(ec, "")
    print(f"    {ec:<12} ({cnt} rxns) {desc}", flush=True)
print(f"\n  Top ECs unique to {gcf_j}:", flush=True)
for ec, cnt in ec_only_j.most_common(10):
    desc = NOTABLE_ECS.get(ec, "")
    print(f"    {ec:<12} ({cnt} rxns) {desc}", flush=True)

# ── Cross-baseline consistency ──
print(f"\n{'='*60}", flush=True)
print("CROSS-BASELINE PATHWAY CONSISTENCY", flush=True)
print(f"{'='*60}", flush=True)

baselines = ["clean_vanilla", "dpz_vanilla", "enzbert_vanilla"]
gcf_test = valid_gcfs[0]
active_sets = {}
for bl in baselines:
    sol_path = f"{V7_RUN}/meteor_out/{bl}/meteor_sol_{gcf_test}.pkl"
    if os.path.exists(sol_path):
        sol = pickle.load(open(sol_path, "rb"))
        active = set(np.where(sol["y_vals"] > 0.5)[0])
        active_sets[bl] = active
        print(f"  {bl:<20}: {len(active)} active reactions", flush=True)

if len(active_sets) >= 2:
    keys = list(active_sets.keys())
    for a in range(len(keys)):
        for b in range(a+1, len(keys)):
            sa, sb = active_sets[keys[a]], active_sets[keys[b]]
            inter = len(sa & sb)
            union = len(sa | sb)
            jacc = inter / union if union > 0 else 0
            print(f"  {keys[a]} vs {keys[b]}: Jaccard={jacc:.3f} "
                  f"(shared={inter}, A-only={len(sa-sb)}, B-only={len(sb-sa)})", flush=True)

# ── Differential ECs between baselines on same genome ──
if len(active_sets) >= 2:
    print(f"\n  Differential reactions (DPZ vs CLEAN) on {gcf_test}:", flush=True)
    if "dpz_vanilla" in active_sets and "clean_vanilla" in active_sets:
        dpz_only = active_sets["dpz_vanilla"] - active_sets["clean_vanilla"]
        clean_only = active_sets["clean_vanilla"] - active_sets["dpz_vanilla"]
        dpz_ecs = Counter()
        clean_ecs = Counter()
        for idx in dpz_only:
            rid = rxn_ids[idx]
            base_rxn = rid.split("_")[0] if "_" in rid else rid
            if base_rxn in seedr2ec and seedr2ec[base_rxn]:
                for ec in seedr2ec[base_rxn]:
                    dpz_ecs[ec] += 1
        for idx in clean_only:
            rid = rxn_ids[idx]
            base_rxn = rid.split("_")[0] if "_" in rid else rid
            if base_rxn in seedr2ec and seedr2ec[base_rxn]:
                for ec in seedr2ec[base_rxn]:
                    clean_ecs[ec] += 1
        print(f"    DPZ-only reactions: {len(dpz_only)}, CLEAN-only: {len(clean_only)}", flush=True)
        print(f"    Top DPZ-only ECs: {dpz_ecs.most_common(5)}", flush=True)
        print(f"    Top CLEAN-only ECs: {clean_ecs.most_common(5)}", flush=True)

out_pkl = f"{V7_RUN}/downstream_results/pathway_case_study.pkl"
pickle.dump({
    "rxn_freq": rxn_freq,
    "gcf_nactive": gcf_nactive,
    "valid_gcfs": valid_gcfs,
    "core_count": int(core),
    "variable_count": int(variable),
    "absent_count": int(absent),
}, open(out_pkl, "wb"))
print(f"\nSaved: {out_pkl}", flush=True)
