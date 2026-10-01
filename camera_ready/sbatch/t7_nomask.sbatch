#!/bin/bash
#SBATCH --job-name=t7_nomask
#SBATCH --account=pi-hohndor
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-11
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/logs/%x_%a.out
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/logs/%x_%a.err
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
R=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision
export METEOR_DATA=$R/code_snapshot/data
cd $R
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" sbatch/t7_genomes.tsv); GCA=$(echo "$LINE" | cut -f1); GRAM=$(echo "$LINE" | cut -f2)
echo "START $GCA $GRAM nomask $(date)"
# all 47,880 non-excluded reactions selectable; 2 h CBC limit
python3 -u eval/skeleton_ablation.py --gca "$GCA" --gram "$GRAM" --arm nomask --time_limit 7200 --outdir $R/results/nomask_abl
echo "END $GCA rc=$? $(date)"
