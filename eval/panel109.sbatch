#!/bin/bash
#SBATCH --job-name=panel109
#SBATCH --array=1-109%16
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -o /ibex/user/niuk0a/meteor_v7/eval/logs/panel109_%A_%a.out
#SBATCH -e /ibex/user/niuk0a/meteor_v7/eval/logs/panel109_%A_%a.err
set -eo pipefail
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh
conda activate cobra
cd /ibex/user/niuk0a/meteor_v7/eval
G=$(awk -v n=$SLURM_ARRAY_TASK_ID 'NR==n+1{print $1}' gca108_gram.tsv)
GR=$(awk -v n=$SLURM_ARRAY_TASK_ID 'NR==n+1{print $2}' gca108_gram.tsv)
echo "[panel109] task $SLURM_ARRAY_TASK_ID gca=$G gram=$GR"
python build_grow_memote.py --gca $G --gram $GR --outdir /ibex/user/niuk0a/meteor_v7/eval/results/PANEL109
