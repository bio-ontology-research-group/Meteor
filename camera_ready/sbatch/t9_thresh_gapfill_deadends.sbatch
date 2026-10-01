#!/bin/bash
#SBATCH --job-name=t9_tgd
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --nice=5000
#SBATCH --array=0-107%20
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/logs/t9_tgd_%A_%a.out
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/logs/t9_tgd_%A_%a.err
# T9: dead ends of the growing threshold baseline (tau=0.5 draft + grow_support), DeepProZyme-vanilla.
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
R=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision
export METEOR_DATA=$R/code_snapshot/data
cd $R
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $R/code_snapshot/panel108_gram.tsv); GCA=$(echo "$LINE" | cut -f1); GRAM=$(echo "$LINE" | cut -f2)
echo "START $GCA $GRAM $(date)"
python3 -u eval/thresh_gapfill_deadends.py --gca "$GCA" --gram "$GRAM"
echo "END $GCA rc=$? $(date)"
