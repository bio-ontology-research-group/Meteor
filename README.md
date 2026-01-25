
# METEOR: Joint Genome-Scale Reconstruction and Enzyme Prediction

This repository contains the official implementation of the framework described in our paper: **"METEOR: joint genome-scale reconstruction and enzyme prediction"**.

METEOR integrates protein sequence-based enzyme function prediction (EC numbers) directly with the metabolic network reconstruction process, improving the accuracy and consistency of Genome-Scale Metabolic Models (GEMs).

## 🌟 Key Features

* **Integrated Pipeline**: Seamlessly transition from raw protein sequences to a draft genome-scale metabolic model.
* **Deep Learning-based Enzyme Prediction**: EC number prediction optimized specifically for metabolic gaps.
* **Consistency-Driven Reconstruction**: Uses network topology and metabolic constraints to refine enzyme annotations.
* **SBML Support**: Exports high-quality models compatible with COBRApy and other standard metabolic engineering tools.

---

## 🛠️ Installation

METEOR requires two separate environments: one for the METEOR reconstruction framework and one for the DeepProZyme prediction tool.

### 1. Install METEOR
```bash
# Clone the repository
git clone [https://github.com/bio-ontology-research-group/Meteor.git](https://github.com/bio-ontology-research-group/Meteor.git)
cd meteor

# Create and activate the METEOR environment
conda create -n meteor python=3.10 -y
conda activate meteor

# Install dependencies
pip install -r requirements.txt

```

### 2. Install DeepProZyme (Dependency)

```bash
# Clone the DeepProZyme repository
git clone [https://github.com/kaistsystemsbiology/DeepProZyme.git](https://github.com/kaistsystemsbiology/DeepProZyme.git)
cd DeepProZyme

# Create the environment for enzyme prediction
conda env create -f environment.yml

```

---

## 🚀 Quick Start

The reconstruction process follows a two-step workflow:

### Step 1: Enzyme Potential Prediction

First, use **DeepProZyme** to predict the full-scale enzyme potential from your genomic data (FASTA format).

```bash
conda activate deepectransformer

# Run prediction to generate the .pkl file
python DeepProZyme/run_deepprozyme.py \
    -i test/GCF_000236665.fasta \
    -o test \
    --gpu cuda:0 \
    -b 128

```

### Step 2: Metabolic Model Reconstruction

Next, use **METEOR** to generate the genome-scale metabolic model using the `.pkl` file generated in the previous step.

```bash
conda activate meteor

# Run METEOR reconstruction
python v5main.py \
    --input_file test/GCF_000236665_DeepECv2_t5.pkl \
    --media default \
    --svfolder test \
    --cpu 8 \
    --gram negative \
    --name testgenome \
    --buildcobra 1 

```

*Note: Use `--buildcobra 1` if you wish to generate a COBRA-compatible model.*

---

## 📊 Evaluation & Benchmarking

The code used to reproduce the benchmarks and figures in the paper (including comparisons against standard enzyme predictors and reconstruction tools) can be found in the `exp/` directory.

### External Data & Genomes

Additional data and pre-processed files are available on **Zenodo**:
[[Link to Zenodo Dataset](https://doi.org/10.5281/zenodo.18367730)]

To reproduce the benchmarks using the **Price dataset**:

1. Download the genome files from the Zenodo link above.
2. Unzip the downloaded archive.
3. Move the contents into the `genomes/` folder within this repository.

---

## 📬 Contact

For questions regarding the paper or the software, please contact:

* **Kexin Niu**: [kexin.niu@kaust.edu.sa](mailto:kexin.niu@kaust.edu.sa)
* **Lab/Group**: [Bio-Ontology Research Group (BORG)](https://borg.kaust.edu.sa/)

