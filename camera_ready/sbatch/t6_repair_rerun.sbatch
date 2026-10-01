#!/bin/bash
#SBATCH --job-name=t6_repair
#SBATCH --account=pi-hohndor
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-1187
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/logs/%x_%a.out
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/logs/%x_%a.err
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
R=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision
export METEOR_DATA=$R/code_snapshot/data
cd $R
# 11 configs x 108 genomes; dpz_vanilla already has the deposited rerun (solver_status_dpz_vanilla.json)
CFGS=(clean_vanilla clean_filt30 clean_filt50 clean_filt70 dpz_filt30 dpz_filt50 dpz_filt70 enzbert_vanilla enzbert_filt30 enzbert_filt50 enzbert_filt70)
CFG=${CFGS[$(( SLURM_ARRAY_TASK_ID / 108 ))]}; BASE=${CFG%%_*}; VAR=${CFG#*_}
IDX=$(( SLURM_ARRAY_TASK_ID % 108 ))
LINE=$(sed -n "$((IDX + 1))p" $R/code_snapshot/panel108_gram.tsv); GCA=$(echo "$LINE" | cut -f1); GRAM=$(echo "$LINE" | cut -f2)
OUTROOT=$R/meteor_out/repair_rerun
echo "START $GCA $GRAM $CFG $(date)"
# snapshot copy of meteor_v8/eval/emit_v8.py; --outroot keeps every write inside psb_revision
python3 -u code_snapshot/eval/emit_v8.py --gca "$GCA" --gram "$GRAM" --baseline $BASE --variant $VAR \
    --penalty evw --mu 3 --pexp 2 --eps 0.0 --gmin 0.1 --outroot "$OUTROOT"
echo "END $GCA rc=$? $(date)"
