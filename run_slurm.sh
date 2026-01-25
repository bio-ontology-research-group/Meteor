#!/bin/sh
#SBATCH --job-name=test
#SBATCH --time=0:30:00
#SBATCH --mem=20Gb
#SBATCH --cpus-per-task=8
#SBATCH --partition=workq
#SBATCH -e test.%J.err
#SBATCH -o test.%J.out
#SBATCH --mail-type=END


python v5main.py --input_file test/GCF_000236665_DeepECv2_t5.pkl --media default \
        --svfolder test \
        --cpu 8 --gram negative  --name testgenome
        # --buildcobra 1 \ # if you want to generate cobra model, add this argument