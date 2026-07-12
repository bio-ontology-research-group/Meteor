#!/bin/bash
#SBATCH --job-name=panelfull
#SBATCH --array=1-1308%40
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH -o /ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/logs/panelfull_%A_%a.out
#SBATCH -e /ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/logs/panelfull_%A_%a.err
set -eo pipefail
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh
conda activate cobra
cd /ibex/user/niuk0a/meteor_v7/eval

LINE=$(awk -v n=$SLURM_ARRAY_TASK_ID 'NR==n' panel_jobs.tsv)
B=$(echo "$LINE" | cut -f1)
V=$(echo "$LINE" | cut -f2)
G=$(echo "$LINE" | cut -f3)
GR=$(echo "$LINE" | cut -f4)
OUT=/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/PANEL109/${B}_${V}
mkdir -p "$OUT"
if [ -f "$OUT/${G}_summary.json" ]; then echo "skip $B $V $G"; exit 0; fi
echo "[panelfull] task $SLURM_ARRAY_TASK_ID $B $V $G $GR"
python build_grow_memote.py --gca "$G" --gram "$GR" --baseline "$B" --variant "$V" --outdir "$OUT"
