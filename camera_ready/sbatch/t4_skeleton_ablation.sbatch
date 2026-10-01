#!/bin/bash
#SBATCH --job-name=t4_skelabl
#SBATCH --account=pi-hohndor
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-323
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/logs/%x_%a.out
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/logs/%x_%a.err
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
R=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision
export METEOR_DATA=$R/code_snapshot/data
cd $R
# 3 arms x 108 genomes: 0-107 noskel, 108-215 skelonly, 216-323 full (in-run control)
ARMS=(noskel skelonly full); ARM=${ARMS[$(( SLURM_ARRAY_TASK_ID / 108 ))]}
IDX=$(( SLURM_ARRAY_TASK_ID % 108 ))
LINE=$(sed -n "$((IDX + 1))p" $R/code_snapshot/panel108_gram.tsv); GCA=$(echo "$LINE" | cut -f1); GRAM=$(echo "$LINE" | cut -f2)
echo "START $GCA $GRAM $ARM $(date)"
python3 -u eval/skeleton_ablation.py --gca "$GCA" --gram "$GRAM" --arm "$ARM"
echo "END $GCA rc=$? $(date)"
