#!/bin/bash
#SBATCH --job-name=ess_ecoli_lb
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --nice=5000
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/logs/%x_%j.out
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/logs/%x_%j.err
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
cd /ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality
set -x
timeout 600 python essentiality_correct_medium.py --organism ecoli_lb --baseline clean   --procs 4 --time_limit 250
timeout 600 python essentiality_correct_medium.py --organism ecoli_lb --baseline dpz     --procs 4 --time_limit 250
timeout 600 python essentiality_correct_medium.py --organism ecoli_lb --baseline enzbert --procs 4 --time_limit 250
timeout 120 python essentiality_correct_medium.py --organism ecoli_lb --curated --procs 4
echo ALL_DONE
