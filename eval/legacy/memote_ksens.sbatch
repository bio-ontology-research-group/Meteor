#!/bin/bash
# Score-truncation sweep behind Supplementary Table S8.
#
# Legacy: this ran under the log-odds cost alone, before the evidence-weighted
# parsimony term mu*(1-w_j)^p was added, so its selected-reaction counts are
# larger than the ones reported in the main text. It is deposited to document
# the exact k values and flags, not as part of the reproduction path.
#
# The per-genome records it writes are not deposited.
#
# Paths come from the environment so the script carries no site-specific
# locations. Set these before submitting:
#
#   METEOR_RUN   run directory; outputs go to $METEOR_RUN/PANEL109_ksens/k<K>
#   METEOR_SRC   checkout that provides eval/build_grow_memote_topk.py
#   GCA_LIST     text file, one assembly accession per line (53 genomes)
#   CONDA_SH     path to conda's profile.d/conda.sh
#   CONDA_ENV    environment name (default: cobra)
#
#   sbatch --array=0-264%32 eval/legacy/memote_ksens.sbatch
#
#SBATCH --job-name=memote_ksens
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -eo pipefail

: "${METEOR_RUN:?set METEOR_RUN to the run directory}"
: "${METEOR_SRC:?set METEOR_SRC to the checkout providing build_grow_memote_topk.py}"
: "${GCA_LIST:?set GCA_LIST to a file of assembly accessions, one per line}"
: "${CONDA_ENV:=cobra}"

if [ -n "${CONDA_SH:-}" ]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

# 53 genomes x 5 k values = 265 tasks; task_id = genome_idx * 5 + k_idx.
# k=0 means no truncation and is passed to the script as a large proxy value.
K_VALUES=(0 1 3 5 10)
TOPK_ARGS=(99999 1 3 5 10)

GENOME_IDX=$((SLURM_ARRAY_TASK_ID / 5))
K_IDX=$((SLURM_ARRAY_TASK_ID % 5))
K=${K_VALUES[$K_IDX]}
TOPK=${TOPK_ARGS[$K_IDX]}

mapfile -t GCAS < "$GCA_LIST"
GCA=${GCAS[$GENOME_IDX]}
if [ -z "$GCA" ]; then
    echo "No genome at index $GENOME_IDX"
    exit 0
fi

OUTDIR="$METEOR_RUN/PANEL109_ksens/k${K}"
mkdir -p "$OUTDIR"

if [ -f "$OUTDIR/${GCA}_summary.json" ]; then
    echo "Already done: k=$K $GCA"
    exit 0
fi

echo "MEMOTE: genome $GENOME_IDX/${#GCAS[@]} gca=$GCA k=$K (topk=$TOPK)"

python3 "$METEOR_SRC/eval/build_grow_memote_topk.py" \
    --gca "$GCA" \
    --gram negative \
    --baseline dpz \
    --variant vanilla \
    --outdir "$OUTDIR" \
    --topk "$TOPK" \
    --gate_topk 1 \
    --evidence_thresh 0.01

echo "Done: k=$K $GCA"
