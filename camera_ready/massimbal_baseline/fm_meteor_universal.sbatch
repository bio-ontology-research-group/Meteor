#!/bin/bash
#SBATCH --job-name=fm_massimbal
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --nice=5000
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_massimbal_baseline/fm_meteor_universal_%j.out
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_massimbal_baseline/fm_meteor_universal_%j.err
# Feasibility demo for R3 #4: SEED universal + METEOR(dpz) Salmonella submodel,
# mass-imbalance decomposed into missing-formula vs genuine; CarveMe + curated for same GCA.
set -euo pipefail
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh
conda activate cobra
D=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_massimbal_baseline
cd "$D"
/usr/bin/time -f "wall=%E maxrss=%MkB" python carveme_structural_metrics.py \
  --gcas GCF_000006945.2 --curated --meteor --out results_meteor_universal_salmonella.json
