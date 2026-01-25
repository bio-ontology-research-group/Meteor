## predict protein potentials for genomes using deepprozyme
conda activate deepectransformer
python DeepProZyme/run_deepprozyme.py  -i test/GCF_000236665.fasta -o test --gpu cuda:0 -b 128

## generate genome-scale metabolic model using METEOR
python v5main.py --input_file test/GCF_000236665_DeepECv2_t5.pkl --media default \
        --svfolder test \
        --cpu 8 --gram negative  --name testgenome
        # --buildcobra 1 \ # if you want to generate cobra model, add this argument