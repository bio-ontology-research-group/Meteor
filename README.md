# METEOR: Network-Consistent Refinement of Enzyme Annotations

METEOR is a model-agnostic post-processing layer that refines any upstream
EC-number predictor's output using organism-level metabolic constraints.
It closes a loop that existing pipelines leave open: sequence evidence
informs metabolic feasibility, and metabolic feasibility feeds back into
per-protein scores.

> **Paper:** K. Niu and R. Hoehndorf, "METEOR: Network-consistent refinement
> of enzyme annotations," submitted, 2026.

## How It Works

METEOR takes a per-protein EC probability matrix from any baseline predictor
(CLEAN, DeepProZyme, EnzBERT, etc.) and refines it through four steps:

1. **Noisy-OR aggregation** — per-(protein, EC) probabilities → per-reaction
   confidence *w_j*
2. **Log-odds cost** — *w_j* → reaction cost *c_j* (favoring high-confidence
   reactions)
3. **Hard-biomass MILP** — selects a biomass-feasible active reaction set over
   the SEED universal (~48k reactions) by minimizing total cost subject to
   stoichiometric constraints
4. **Confidence update** — MILP active/muted decisions propagate back to
   per-protein scores via a bounded monotone adjustment

No retraining. No threshold tuning. The same pipeline works on any monotone
[0,1] baseline.

## Installation

```bash
git clone https://github.com/bio-ontology-research-group/Meteor.git
cd Meteor
pip install -e .

# Or with conda
conda env create -f environments/meteor.yml
conda activate meteor
pip install -e .
```

### Data files

METEOR requires the SEED universal metabolic model and precomputed reaction
bounds. Download from Zenodo:

```bash
wget -O meteor_data.tar.gz https://doi.org/10.5281/zenodo.21321228
tar xzf meteor_data.tar.gz -C data/
```

Required files in `data/`:
- `universal.pickle` — SEED universal bacterial reaction set (~48k reactions)
- `tight_bounds_v6_neg.pkl` / `tight_bounds_v6_pos.pkl` — precomputed flux bounds
- `seedr2ec.pkl` / `seedec2r.pkl` — reaction ↔ EC mappings
- `all_ancestors.txt` / `all_ec.txt` — EC vocabulary (included in repo)

## Quick Start

A sample genome is included in `examples/` so you can run METEOR immediately
after downloading the data files:

```bash
python -m meteor \
    --input examples/sample_baseline_pred.pkl \
    --output_dir results/ \
    --name sample \
    --data_dir data/ \
    --gram negative \
    --gamma_min 0.01 \
    --time_limit 1800
```

The sample is a 174-protein proteome (`sample_proteome.fasta`) with
DeepProZyme baseline predictions (`sample_baseline_pred.pkl`).

**Input format:** a pickled pandas DataFrame with shape `(n_proteins, n_ECs)`,
values in [0, 1].

**Output:**
- `meteor_df_{name}.pkl` — refined prediction matrix (same shape as input)
- `meteor_meta_{name}.json` — run metadata (solver status, active-set size)

## Python API

```python
from meteor.core import (
    aggregate_confidence,
    logodds_cost,
    build_milp,
    solve_milp,
    posterior_calibrated,
    MILPParams,
    run_meteor,
)

refined_df, meta = run_meteor(
    pred_df=baseline_df,
    universal_S=S,
    lb=lb, ub=ub,
    rxn_ec_mask=rxn_ec_mask,
    biomass_idx=biomass_idx,
    rxn_to_ec=rxn_to_ec,
    allrxns=allrxns,
    params=MILPParams(gamma_min=0.01),
    beta=1.0,
)
```

## Repository Structure

```
src/meteor/
  core.py          # 4-step algorithm (noisy-OR, log-odds, MILP, confidence update)
  cli.py           # Command-line interface
tests/
  test_meteor_core.py
examples/
  sample_proteome.fasta        # 174-protein sample genome
  sample_baseline_pred.pkl     # DeepProZyme predictions for sample genome
eval/
  emit_meteor_out.py           # Per-genome METEOR driver
  build_grow_memote.py         # MILP + COBRA submodel + MEMOTE evaluation
  downstream/
    baseline_io.py             # Baseline prediction path resolution
    pathway_completion.py      # KEGG pathway completeness analysis
    bgc_specificity.py         # BGC indicator recovery
    calibration_v7.py          # ECE + Brier calibration
    ec_propagation_analysis.py # Confidence update verification
    ...
environments/                  # Conda environment files
data/                          # EC vocabulary + KEGG pathway maps
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma_min` | 0.01 | Minimum biomass flux (hard constraint) |
| `gamma_max` | 2.5 | Maximum biomass flux (sanity cap) |
| `lambda` | 1e-4 | Flux parsimony penalty |
| `alpha` | 0.4 | Confidence update step size |
| `beta` | 1.0 | Confidence update temperature |
| `time_limit` | 1800s | CBC solver time limit per genome |

## Reproducing Paper Results

The paper evaluates METEOR on a 109-GCF bacterial genome panel with three
baselines (CLEAN, DeepProZyme, EnzBERT) at four retraining cutoffs each.

The 109 genome proteomes are available at Zenodo alongside the model data.

```bash
# Step 1: Run baseline EC predictors on the 109-GCF panel
# (requires GPU; see environments/ for per-baseline conda envs)

# Step 2: Run METEOR across the panel
python eval/emit_meteor_out.py --baseline clean --variant filt30 --panel 109

# Step 3: Evaluate
python eval/downstream/pathway_completion.py
python eval/downstream/calibration_v7.py
python eval/downstream/bgc_specificity.py
python eval/downstream/ec_propagation_analysis.py
```

## Tests

```bash
pytest tests/ -v
```

## Citation

If you use METEOR in your research, please cite:

```bibtex
@article{niu2026meteor,
  title={METEOR: Network-Consistent Refinement of Enzyme Annotations},
  author={Niu, Kexin and Hoehndorf, Robert},
  year={2026}
}
```
