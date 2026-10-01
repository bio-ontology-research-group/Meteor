#!/bin/bash
#SBATCH --job-name=t8b_carveme_ref
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --nice=5000
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/logs/t8b_carveme_ref_%j.out
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/logs/t8b_carveme_ref_%j.err
# T8b: reference row — Table 1 metrics on all 108 CarveMe panel models + 6 curated GEMs.
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
R=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision
cd $R/feasibility_massimbal_baseline
python3 -u carveme_structural_metrics.py --curated --gcas $(cut -f1 $R/code_snapshot/panel108_gram.tsv) --out $R/results/table1_v2/carveme_panel108.json
