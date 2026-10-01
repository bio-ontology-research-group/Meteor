#!/bin/bash
#SBATCH --job-name=remap_impact
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --nice=5000
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_r1_strat/logs/remap_impact_%j.out
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_r1_strat/logs/remap_impact_%j.err
set -euo pipefail
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
cd /ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_r1_strat
echo "start $(date) host $(hostname)"
python -u remap_impact.py
echo "done $(date)"
