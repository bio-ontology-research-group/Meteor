#!/bin/bash
#SBATCH --job-name=r1_strat
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --nice=5000
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_r1_strat/logs/r1_strat_%j.out
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_r1_strat/logs/r1_strat_%j.err
set -euo pipefail
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
D=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_r1_strat
cd "$D"
echo "start $(date)  host $(hostname)"
# 6 GEMs x 3 predictors in one process (context + skeletons built once)
python -u strat_all.py iML1515 STM_v1_0 iYL1228 iJN1463 iYS854 iYO844 --preds clean dpz enzbert --outdir "$D/results/strat"
python -u aggregate.py --indir "$D/results/strat" --out "$D/results/strat_summary.json"
echo "done $(date)"
