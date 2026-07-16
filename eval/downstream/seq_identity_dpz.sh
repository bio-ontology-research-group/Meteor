#!/bin/bash
#SBATCH --job-name=seqid_dpz
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results/seqid_dpz_%j.log
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results/seqid_dpz_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=batch

module load diamond/2.1.16

WORKDIR=/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results/seq_identity_dpz
DPZ_DB=/ibex/user/niuk0a/funcarve/METEOR/DeepProZyme/model/swissprot_enzyme_diamond
INPUT_DIR=/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/reconstructor_run/input_fasta
MANIFEST=/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026/scripts/genome_collection_manifest.json

mkdir -p $WORKDIR

echo "Step 1: Concatenating test proteomes..."
python3 -c "
import json
m = json.load(open('$MANIFEST'))
gcfs = sorted(g for g in m['assemblies'] if g.startswith('GCF'))
for g in gcfs:
    print(g)
" > $WORKDIR/gcf_list.txt

> $WORKDIR/all_test_proteins.fasta
while read gcf; do
    fasta="${INPUT_DIR}/${gcf}.fasta"
    if [ -f "$fasta" ]; then
        cat "$fasta" >> $WORKDIR/all_test_proteins.fasta
    fi
done < $WORKDIR/gcf_list.txt

n_query=$(grep -c "^>" $WORKDIR/all_test_proteins.fasta)
echo "Total test proteins: $n_query"

echo "Step 2: Running DIAMOND blastp against DPZ training set (226,325 seqs)..."
diamond blastp \
    --query $WORKDIR/all_test_proteins.fasta \
    --db $DPZ_DB \
    --out $WORKDIR/diamond_hits.tsv \
    --outfmt 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovhsp \
    --max-target-seqs 1 \
    --threads 8 \
    --sensitive \
    --evalue 1e-3

n_hits=$(wc -l < $WORKDIR/diamond_hits.tsv)
echo "DIAMOND hits: $n_hits"

echo "Step 3: Running Python analysis..."
source ~/.bashrc
conda activate cobra

python3 -u << 'PYEOF'
import pickle, json, os
import numpy as np
from collections import defaultdict

WORKDIR = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results/seq_identity_dpz"
V7_RUN = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run"

print("Loading DIAMOND hits...", flush=True)
max_identity = {}
with open(os.path.join(WORKDIR, "diamond_hits.tsv")) as f:
    for line in f:
        parts = line.strip().split("\t")
        qid = parts[0]
        pident = float(parts[2])
        if qid not in max_identity or pident > max_identity[qid]:
            max_identity[qid] = pident
print("  Proteins with hits: %d" % len(max_identity), flush=True)

# Identity distribution
idents = list(max_identity.values())
print("  Identity distribution: min=%.1f%% median=%.1f%% mean=%.1f%% max=%.1f%%" % (
    np.min(idents), np.median(idents), np.mean(idents), np.max(idents)), flush=True)

CACHE_DIR = os.path.join(V7_RUN, "downstream_results", "ncbi_ec_cache")
W = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026"
import sys
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7/eval/downstream")
from baseline_io import resolve_baseline_pkl

manifest = json.load(open(os.path.join(W, "scripts/genome_collection_manifest.json")))
all_gcfs = sorted(g for g in manifest["assemblies"] if g.startswith("GCF"))

BASELINE, VARIANT = "dpz", "vanilla"
meteor_dir = os.path.join(V7_RUN, "meteor_out", "%s_%s" % (BASELINE, VARIANT))

print("Evaluating per protein with DPZ training-set identity...", flush=True)
BINS = [(0, 30, "<30%"), (30, 50, "30-50%"), (50, 70, "50-70%"), (70, 90, "70-90%"), (90, 100.01, ">90%")]
bin_stats = {}
for _, _, label in BINS:
    bin_stats[label] = {"n": 0, "b_correct": 0, "m_correct": 0, "corr": 0, "regr": 0,
                        "fmax_b": [], "fmax_m": []}
no_hit_stats = {"n": 0, "b_correct": 0, "m_correct": 0, "corr": 0, "regr": 0,
                "fmax_b": [], "fmax_m": []}

def fmax_fast(gt_indices, scores):
    n_gt = len(gt_indices)
    if n_gt == 0:
        return 0.0
    order = np.argsort(-scores)
    gt_mask = np.zeros(len(scores), dtype=bool)
    for i in gt_indices:
        gt_mask[i] = True
    sorted_mask = gt_mask[order]
    cum_tp = np.cumsum(sorted_mask).astype(float)
    n_pred = np.arange(1, len(scores) + 1, dtype=float)
    precision = cum_tp / n_pred
    recall = cum_tp / n_gt
    f1 = np.where((precision + recall) > 0,
                  2 * precision * recall / (precision + recall), 0.0)
    return float(np.max(f1))

n_processed = 0
for gcf in all_gcfs:
    cache_file = os.path.join(CACHE_DIR, "%s_ec.json" % gcf)
    if not os.path.exists(cache_file):
        continue
    met_pkl = os.path.join(meteor_dir, "meteor_df_%s.pkl" % gcf)
    if not os.path.exists(met_pkl):
        continue
    base_pkl_path = resolve_baseline_pkl(BASELINE, VARIANT, gcf)
    if not base_pkl_path:
        continue

    ec_map = json.load(open(cache_file))
    ec_map = {k: set(v) for k, v in ec_map.items()}
    if not ec_map:
        continue

    try:
        base_df = pickle.load(open(base_pkl_path, "rb"))
        met_df = pickle.load(open(met_pkl, "rb"))
    except:
        continue

    if any(str(c).startswith("EC:") for c in base_df.columns[:5]):
        base_df.columns = [str(c).replace("EC:", "") for c in base_df.columns]

    b_cols = list(base_df.columns)
    m_cols = list(met_df.columns)
    b_col2idx = {c: i for i, c in enumerate(b_cols)}
    m_col2idx = {c: i for i, c in enumerate(m_cols)}

    for pid, gt_ecs in ec_map.items():
        if pid not in base_df.index or pid not in met_df.index:
            continue

        b_vals = base_df.loc[pid].values.astype(float)
        m_vals = met_df.loc[pid].values.astype(float)
        bt1 = b_cols[int(np.argmax(b_vals))]
        mt1 = m_cols[int(np.argmax(m_vals))]
        bh = bt1 in gt_ecs
        mh = mt1 in gt_ecs

        # Fmax
        b_gt_idx = {b_col2idx[ec] for ec in gt_ecs if ec in b_col2idx}
        m_gt_idx = {m_col2idx[ec] for ec in gt_ecs if ec in m_col2idx}

        ident = max_identity.get(pid, None)
        if ident is None:
            stats = no_hit_stats
        else:
            stats = None
            for lo, hi, label in BINS:
                if lo <= ident < hi:
                    stats = bin_stats[label]
                    break
            if stats is None:
                stats = no_hit_stats

        stats["n"] += 1
        if bh:
            stats["b_correct"] += 1
        if mh:
            stats["m_correct"] += 1
        if not bh and mh:
            stats["corr"] += 1
        if bh and not mh:
            stats["regr"] += 1
        stats["fmax_b"].append(fmax_fast(b_gt_idx, b_vals))
        stats["fmax_m"].append(fmax_fast(m_gt_idx, m_vals))

        n_processed += 1

print("\nProcessed %d proteins" % n_processed, flush=True)
print("No DIAMOND hit: %d proteins" % no_hit_stats["n"], flush=True)

print("\n" + "=" * 120, flush=True)
print("METEOR IMPROVEMENT BY SEQUENCE IDENTITY TO DPZ TRAINING SET (226,325 Swiss-Prot enzymes)", flush=True)
print("=" * 120, flush=True)
print("%-12s %7s %8s %8s %8s %8s %8s %8s %6s %6s %6s %10s" % (
    "Identity", "n", "Top1-B", "Top1-M", "dTop1", "Fmax-B", "Fmax-M", "dFmax", "Corr", "Regr", "Net", "Corr/1k"), flush=True)
print("-" * 120, flush=True)

for lo, hi, label in BINS:
    s = bin_stats[label]
    if s["n"] == 0:
        continue
    pct_b = 100.0 * s["b_correct"] / s["n"]
    pct_m = 100.0 * s["m_correct"] / s["n"]
    delta = pct_m - pct_b
    fmax_b = np.mean(s["fmax_b"])
    fmax_m = np.mean(s["fmax_m"])
    corr_rate = 1000.0 * s["corr"] / s["n"]
    print("%-12s %7d %7.1f%% %7.1f%% %+7.2f%% %8.4f %8.4f %+7.4f %6d %6d %+5d %9.1f" % (
        label, s["n"], pct_b, pct_m, delta, fmax_b, fmax_m, fmax_m-fmax_b,
        s["corr"], s["regr"], s["corr"]-s["regr"], corr_rate), flush=True)

s = no_hit_stats
if s["n"] > 0:
    pct_b = 100.0 * s["b_correct"] / s["n"]
    pct_m = 100.0 * s["m_correct"] / s["n"]
    delta = pct_m - pct_b
    fmax_b = np.mean(s["fmax_b"]) if s["fmax_b"] else 0
    fmax_m = np.mean(s["fmax_m"]) if s["fmax_m"] else 0
    corr_rate = 1000.0 * s["corr"] / s["n"]
    print("%-12s %7d %7.1f%% %7.1f%% %+7.2f%% %8.4f %8.4f %+7.4f %6d %6d %+5d %9.1f" % (
        "no hit", s["n"], pct_b, pct_m, delta, fmax_b, fmax_m, fmax_m-fmax_b,
        s["corr"], s["regr"], s["corr"]-s["regr"], corr_rate), flush=True)

# Save (convert fmax lists to means for pickling)
for label in bin_stats:
    bin_stats[label]["fmax_b_mean"] = float(np.mean(bin_stats[label]["fmax_b"])) if bin_stats[label]["fmax_b"] else 0
    bin_stats[label]["fmax_m_mean"] = float(np.mean(bin_stats[label]["fmax_m"])) if bin_stats[label]["fmax_m"] else 0
    del bin_stats[label]["fmax_b"]
    del bin_stats[label]["fmax_m"]
no_hit_stats["fmax_b_mean"] = float(np.mean(no_hit_stats["fmax_b"])) if no_hit_stats["fmax_b"] else 0
no_hit_stats["fmax_m_mean"] = float(np.mean(no_hit_stats["fmax_m"])) if no_hit_stats["fmax_m"] else 0
del no_hit_stats["fmax_b"]
del no_hit_stats["fmax_m"]

out_pkl = os.path.join(V7_RUN, "downstream_results", "seq_identity_dpz_analysis.pkl")
pickle.dump({
    "bin_stats": bin_stats,
    "no_hit_stats": no_hit_stats,
    "n_train_seqs": 226325,
    "n_test_proteins": n_processed,
    "n_with_hits": len(max_identity),
    "diamond_params": "diamond blastp --sensitive --evalue 1e-3 --max-target-seqs 1",
    "diamond_version": "2.1.16 (module), db built with 2.0.11",
    "bins": BINS,
}, open(out_pkl, "wb"))
print("\nSaved: %s" % out_pkl, flush=True)
PYEOF

echo "Done!"
