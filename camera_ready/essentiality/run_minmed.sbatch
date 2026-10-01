#!/bin/bash
#SBATCH --job-name=minmed_feas
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --nice=5000
#SBATCH --output=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/logs/%x_%j.out
#SBATCH --error=/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality/logs/%x_%j.err
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
cd /ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality
set -x
# 6 required combos: {E.coli, B.subtilis} x {clean, dpz, enzbert} vanilla, GS_MM_glc
timeout 480 python resolve_minmed.py --gca GCF_058436375.1 --gram negative --baseline clean   --medium GS_MM_glc --time_limit 300
timeout 480 python resolve_minmed.py --gca GCF_058436375.1 --gram negative --baseline dpz     --medium GS_MM_glc --time_limit 300
timeout 480 python resolve_minmed.py --gca GCF_058436375.1 --gram negative --baseline enzbert --medium GS_MM_glc --time_limit 300
timeout 480 python resolve_minmed.py --gca GCF_058182495.1 --gram positive --baseline clean   --medium GS_MM_glc --time_limit 300
timeout 480 python resolve_minmed.py --gca GCF_058182495.1 --gram positive --baseline dpz     --medium GS_MM_glc --time_limit 300
timeout 480 python resolve_minmed.py --gca GCF_058182495.1 --gram positive --baseline enzbert --medium GS_MM_glc --time_limit 300
# bonus: bsub reference medium is actually LB_marinos, not GS_MM_glc (gapseq compare_GE_recons.R) -- test dpz only
timeout 480 python resolve_minmed.py --gca GCF_058182495.1 --gram positive --baseline dpz     --medium LB_marinos --time_limit 300
echo ALL_DONE
