#!/bin/bash
#SBATCH --job-name=yvec_trace
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --nice=5000
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/logs/%x_%j.out
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/logs/%x_%j.err
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
cd /ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality
set -x
timeout 1500 python yvector_trace.py
echo ALL_DONE
