#!/bin/bash
#SBATCH --job-name=t2_candsize
#SBATCH --account=pi-hohndor
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/logs/%x_%j.out
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/logs/%x_%j.err
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
R=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision
export METEOR_DATA=$R/code_snapshot/data
cd $R
python3 -u eval/candidate_size.py
