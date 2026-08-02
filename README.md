# METEOR — evidence-weighted reaction selection reconciles enzyme prediction with growth feasibility

Code and deposited results for the PSB 2027 submission *METEOR:
Evidence-weighted reaction selection reconciles enzyme prediction with
growth feasibility*.

METEOR takes a per-protein EC confidence matrix, aggregates it into
reaction-level evidence, selects a biomass-feasible reaction set from the SEED
universal database with a MILP, and writes the result back to the protein score
matrix through a ranking-preserving update. The cost grades each reaction by the
confidence behind it, `mu*(1-w_j)^p`, rather than charging every reaction the
same: a uniform penalty prunes by count and discards genuine isozymes with the
noise.

This is not weighted gap-filling. There is no thresholded draft and no
gap-filling stage; every reaction in the universal database competes on the same
evidence-derived cost and the whole set is chosen in one optimisation.

## Contents

```
src/meteor_v8/   method: milp_v8.py (MILP), repair.py (verify-and-repair),
                 milp_hard.py (biomass-feasible skeleton)
eval/            one script per table or figure, plus the aggregators
data/            reaction database, EC maps, media -- everything the code
                 loads at run time (see data/README.md)
cohorts/         COHORTS.json/tsv, panel108_gram.tsv, solver status
results/         the records every reported number is computed from
env/             conda environment
```

## Reproducing the published numbers

Four scripts recompute every main-text number from the deposited records. They
need a Python 3.9 interpreter and nothing else — the standard library only, no
environment to build, no input beyond this checkout:

```bash
python eval/agg_table1.py    # Table 1; dead-end reduction 75.8-90.6%
python eval/agg_decoy.py     # Figure 2; 34.5 / 142.4 / 2007.7 spurious
                             # reactions, 4.13x and 58.3x
python eval/agg_fva.py       # Section 3.1; 54.5% of selected reactions
                             # flux-consistent vs 40.5% for the baseline
python eval/agg_4arm.py      # Section 3.4; 34.6 / 78.2 / 142.9 / 2006.6
                             # spurious reactions across the four arms
```

Install SciPy to get the paired Wilcoxon p-values as well; without it each
script prints its counts and says the tests were skipped. The reported values
are `p=1.9e-19` for uniform and for shuffled against METEOR in `agg_decoy.py`,
`1.9e-19` for the flux-consistent fraction in `agg_fva.py`, and `9.0e-18`,
`1.9e-19`, `1.9e-19` for the three comparison arms in `agg_4arm.py`.

Verified from a fresh clone on macOS 15 (arm64) against the system
`/usr/bin/python3` 3.9.6: all four exit 0 and print the values above.

## Environments

`env/environment.yml` is portable and solves on Linux, macOS and Windows. Build
it to re-run the pipeline itself — `emit_v8.py`, the FVA and recovery scripts,
the figures — which needs COBRApy, PuLP, NumPy, pandas, SciPy, Matplotlib and
MEMOTE:

```bash
conda env create -f env/environment.yml && conda activate meteor
```

`env/meteor_cobra.yml` is the exact solve from our linux-64 cluster with build
strings pinned. It reproduces our environment bit for bit and, being pinned to
linux-64 builds, will not solve on another platform.

## Reconstructing a genome

```bash
python eval/emit_v8.py --gca GCF_000006945.2 --gram negative \
    --baseline dpz --variant vanilla \
    --penalty evw --pexp 2 --mu 3 --eps 0.0 --gmin 0.1 \
    --outroot <output directory>
```

Pass the flags explicitly; the argparse defaults match the paper.

That command needs nothing outside this directory.
`meteor_v8.utils.data_path` resolves the reaction database and the EC maps
from `data/`; set `$METEOR_DATA` to read them from somewhere else.

## Where each number comes from

| paper item | script | records |
|---|---|---|
| Table 1 (structural) | `eval/gen_table1.py` → `eval/agg_table1.py` | `results/table1/` |
| §3.1 flux consistency | `eval/fva_selected.py` → `eval/agg_fva.py` | `results/fva_selected/`, `results/fva_baseline/` |
| Table 2, §3.2 curated sub-threshold ECs | `eval/weakreal.py` | `results/toolcompare/weakreal.json` |
| §3.3 size-matched pathway control | `eval/kegg_matched.py` | `results/toolcompare/kegg_pathway_108_matched.json` |
| Figure 2, §3.4 | `eval/recovery_ablation.py` → `eval/agg_4arm.py`; figure drawn by `eval/gen_fig_decoy.py` | `results/recovery_4arm/` |
| §3.4 two-stage arm | `eval/twostage.py` (called by `recovery_ablation.py`) | `results/recovery_4arm/` |
| §3.5 holdout | `eval/holdout_cfg.py` | `results/holdout_cfg/` |
| §3.5 full proteome | `eval/net_seqident.py` | `results/toolcompare/net_seqident.json` |
| §3.5 CarveMe (Table 3) | `eval/toolcompare_ec.py`, `eval/toolcompare_ec_vocab.py` | `results/toolcompare/panelB_ec.json`, `panelB_ec_vocab.json` |
| §3.5 EC vocabularies | `eval/export_ec_vocab.py` | `results/toolcompare/ec_vocabulary_map.tsv`, `ec_fp_by_organism.tsv` |
| §3.5 threshold baseline | `eval/baseline_thresh_gapfill.py` | `results/toolcompare/ablation_thresh_vs_evw_ec.json` |
| S1.1 solver status, timings, repair counts | `eval/emit_v8.py` (rerun at the published flags) | `results/solver_status_dpz_vanilla.json` |
| Figure S1 | `eval/gen_fig_massimbal.py` (draws the figure) | `results/table1/` |
| S3.5 pathway detection | `eval/kegg_detect.py`, `eval/kegg_select.py`, `eval/kegg_price.py` | `results/toolcompare/kegg_pathway_108_*.json` |
| S8.5 curated sub-threshold detail | `eval/weakreal.py` | `results/toolcompare/weakreal.json` |
| S8 MEMOTE | `eval/memote_evw.py` | per-genome MEMOTE JSONs (not deposited, see below) |

Scripts whose name starts with `gen_fig_` draw a figure in the paper and say
so in their docstring; `gen_table1.py` and the two `agg_` scripts produce
tables.

`src/meteor_v8/` contains no absolute paths. The analysis scripts under `eval/`
are deposited as they were run and reference our cluster layout for the inputs
listed below — 139 such paths across 69 scripts, all of them under one root.
`meteor_v8.utils.run_path` joins onto that root and `$METEOR_RUNS` overrides
it, so a copy of the inputs elsewhere needs one environment variable rather
than an edit per script. The scripts that reproduce the published numbers do
not use it: they read only `results/` and `data/` from this checkout.

## Parameters

`p=2`, `mu=3`, `k=5` (score truncation), `w_min=0.01` (candidate mask),
`gamma_min=0.1`, `gamma_max=2.5`, `lambda=1e-4`, `epsilon=1e-6` (log-odds
smoothing, `utils.EPS_SMOOTH`), `delta=0` (forced-flux floor, disabled),
`alpha=0.4` (score update).
CBC via PuLP 3.3.1, four threads, 600 s limit, `gapRel=0.05`. The manuscript
calls the forced-flux floor `delta`; the flag is `--eps`.

## Inputs not deposited here

The baseline EC prediction matrices (~5 GB per predictor-cutoff
configuration), the genome proteomes, and the full 12-configuration MILP
output (~336 GB). This repository carries the 108 `dpz_vanilla` MILP solutions
the main-text tables are computed from, plus every aggregated result.
Assemblies are on NCBI RefSeq under the accessions in
`cohorts/panel108_gram.tsv`.

The SEED universal database is in `data/` and no longer has to be fetched
separately. It and the Reconstructor-derived EC mappings carry their own
licences, which govern that content rather than this repository's MIT
licence; see `data/README.md`.

## Cohorts

- **panel108** — 108 bacterial genomes, 63 Gram-negative and 45 Gram-positive;
  every organism-level analysis.
- **holdout-66** — 66 Swiss-Prot proteins across 53 of those genomes with a
  strain-level source assembly; the protein-level set.
- **curated-GEM 6** — six species with published BiGG models; EC-level
  comparison against CarveMe and the threshold baseline.

`cohorts/COHORTS.json` gives the membership of each.

## Licence

MIT, see `LICENSE`.
