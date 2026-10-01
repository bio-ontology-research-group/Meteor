#!/bin/bash
#SBATCH --job-name=r1_strat_remap
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --nice=5000
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_r1_strat/logs/r1_strat_remap_%j.out
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_r1_strat/logs/r1_strat_remap_%j.err
set -euo pipefail
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
D=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_r1_strat
cd "$D"; mkdir -p "$D/results/strat_remap"
# reuse the skeleton/excludes/media cache from the frozen run (identical network inputs)
cp -n "$D/results/strat/skeleton_negative.json" "$D/results/strat/skeleton_positive.json" "$D/results/strat_remap/" || true
echo "start $(date) host $(hostname)"
python -u strat_all.py iML1515 STM_v1_0 iYL1228 iJN1463 iYS854 iYO844 --preds clean dpz enzbert --remap --outdir "$D/results/strat_remap"
python -u aggregate.py --indir "$D/results/strat_remap" --out "$D/results/strat_remap_summary.json"
echo "done $(date)"
