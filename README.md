# METEOR — evidence-weighted reaction selection reconciles enzyme prediction with growth feasibility

Code and deposited results for the PSB 2027 submission *METEOR:
Evidence-weighted reaction selection reconciles enzyme prediction with growth
feasibility*, Kexin Niu and Robert Hoehndorf, King Abdullah University of
Science and Technology.

The submitted version is tag `v8.2-psb2027`; the camera-ready version is tag
`v8.3-psb2027`, which the manuscript cites and from which it links
`supplementary_v8.pdf`. Zenodo archives the repository at
[10.5281/zenodo.21716682](https://doi.org/10.5281/zenodo.21716682), which
resolves to the most recent deposited version.

METEOR takes a per-protein EC confidence matrix, aggregates it into
reaction-level evidence, selects a growth-feasible reaction set from the SEED
universal database with a MILP, and writes the result back to the protein score
matrix through a bounded per-EC update that preserves protein ordering within
each EC column. The cost grades each reaction by the confidence behind it,
`mu*(1-w_j)^p`, rather than charging every reaction the same, so it can retain
evidence-supported redundant reactions and alternative routes while
discouraging unsupported ones. The optimisation contains no protein-level
variables and makes no claim about isozymes.

This is not weighted gap-filling. There is no thresholded draft: candidate
reactions -- those with reaction confidence >= 0.01, the medium, and a shared
biomass-feasible skeleton -- compete on one evidence-derived cost in a single
optimisation, followed by post-solve growth verification and, where needed, a
parsimony repair.

## Contents

```
src/meteor_v8/   method: milp_v8.py (MILP), repair.py (verify-and-repair),
                 milp_hard.py (biomass-feasible skeleton)
eval/            one script per table or figure, plus the aggregators
data/            reaction database, EC maps, media -- everything the code
                 loads at run time (see data/README.md)
cohorts/         COHORTS.json/tsv, panel108_gram.tsv, solver status
results/         the records every reported number is computed from
camera_ready/    scripts and records added for the camera-ready revision
                 (see "Camera-ready additions" below)
env/             conda environment
```

## Reproducing the published numbers

Four scripts recompute the Table 1, Section 3.1, Section 3.4 and Figure 2
numbers from the deposited records. They need a Python 3.9 interpreter and
nothing else — the standard library only, no
environment to build, no input beyond this checkout:

```bash
python eval/agg_table1.py    # Table 1; dead-end reduction 75.8-90.6%
python eval/agg_decoy.py     # three-arm decoy summary (not a paper figure)
                             # 34.5 / 142.4 / 2007.7, 4.13x and 58.3x
python eval/agg_fva.py       # Section 3.1; 54.5% of selected reactions
                             # flux-consistent vs 40.5% for the baseline
python eval/agg_4arm.py      # Figure 2 and Section 3.4; 34.6 / 78.2 / 142.9
                             # / 2006.6 off-reference reactions, four arms
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

One genome's inputs are bundled so the pipeline can be run end to end from this
checkout alone:

```bash
conda env create -f env/environment.yml && conda activate meteor
PYTHONPATH=src:eval python eval/emit_v8.py \
    --gca GCF_000017425.1 --gram negative \
    --baseline dpz --variant vanilla \
    --penalty evw --pexp 2 --mu 3 --eps 0.0 --gmin 0.1 \
    --preds data/demo/GCF_000017425.1_DPZ_top5.pkl.gz
```

About three minutes on four cores. It writes `meteor_sol_`, `meteor_df_` and
`meteor_preds_` for that genome under `results/meteor_out/dpz_vanilla/` and
reports `n_active=3119`. See `data/demo/README.md` for how that input was
reduced from 43 MB to 0.27 MB without changing the result.

For any other genome, `--preds` takes the per-protein EC score matrix, a
pickled proteins × ECs DataFrame. Without `--preds` the matrix is looked up in
the run directories described below, which are not deposited.

Pass the flags explicitly; the argparse defaults match the paper.
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
| S1.1 run-to-run stability | `eval/emit_v8.py` (second solve of the same panel) | `results/rerun_stability_dpz_vanilla.json` |
| Figure S1 | `eval/gen_fig_massimbal.py` (draws the figure) | `results/table1/` |
| S3 pathway detection | `eval/kegg_detect.py`, `eval/kegg_select.py`, `eval/kegg_price.py` | `results/toolcompare/kegg_pathway_108_*.json` |
| S7.5 curated sub-threshold detail | `eval/weakreal.py` | `results/toolcompare/weakreal.json` |
| S7.1, S7.2 MEMOTE | `eval/memote_evw.py` | per-genome MEMOTE JSONs (not deposited, see below) |

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

### Camera-ready additions (tag `v8.3-psb2027`)

The table above follows the section numbering of the submitted version
(tag `v8.2-psb2027`). The camera-ready revision added experiments for the
reviewers; their scripts and records are deposited under `camera_ready/`
exactly as they were run (paths in `camera_ready/eval/_env.py`). The paper and
supplement name no files; this table is the map from each camera-ready item
to its script and records. Section numbers are those of the camera-ready.

| camera-ready item | script | records |
|---|---|---|
| Figure 1 (toy example and pipeline; TikZ) | `figures/fig1_pipeline.tex`; step-3 numbers from `eval/gen_fig_pipeline_partition.py` | `results/candidate_size_dpz_vanilla.json`, `results/candidate_partition_kx_evidence.json` |
| Figure 2a, 2d (dead ends; skeleton/candidate-mask ablation) | `eval/gen_fig_results.py` | `results/table1_v2/pergenome/`, `results/skeleton_ablation_summary.json` |
| Figure 2b (sub-threshold recall vs precision, per organism) | `eval/gen_fig_results.py` | `results/toolcompare/weakreal.json` |
| Figure 2c (deletion-recovery stability) | `eval/gen_fig_results.py` | `results/recovery_4arm/` |
| Figure S1 | `eval/gen_fig_s1_panels.py` | `results/table1/` |
| §2.2, S1.1 candidate-set size and composition (Figure 1b, step 3) | `camera_ready/eval/candidate_size.py`, `candidate_partition.py`, `kx_ids.py`, `kx_evidence.py` | `camera_ready/results/candidate_partition_sets.json`, `results/candidate_*.json` |
| §2.2, S1.1, Table S1 repair counts and solve-call wall times, all twelve configurations | `camera_ready/eval/agg_repair_by_config.py`; wall-time recount `camera_ready/eval/recount_time_limit.py` | `camera_ready/results/repair_by_config.json`, `camera_ready/results/time_limit_recount.json` (per-genome wall times; the `not_optimal`/`time_limit` fields elsewhere are status-based and under-count, see the script docstring) |
| §3.1 dead ends of a growing threshold+repair baseline (count and per-metabolite fraction) | `camera_ready/eval/thresh_gapfill_deadends.py`, `thresh_gapfill_deadends_met.py`, `agg_thresh_gapfill_deadends*.py` | `camera_ready/results/thresh_gapfill_deadends/`, `thresh_gapfill_deadends_met/`, `*_summary.json` |
| §3.1, Figure 2a, S7.7 structural properties, mean ± s.d. | `camera_ready/eval/gen_table1_v2.py`, `agg_table1_v2.py` | `results/table1_v2/pergenome/`, `camera_ready/results/table1_v2_meansd.json` |
| §3.3, S3, Table S7 skeleton-only / no-skeleton / candidate-mask ablation | `camera_ready/eval/skeleton_ablation.py`, `agg_skeleton_ablation.py` | `camera_ready/results/skeleton_abl/` (per-genome JSON and y vectors), `nomask_abl/`, `results/skeleton_ablation_summary.json` |
| §3.2, S7.5 size-matched top-K control | `camera_ready/eval/weakreal_matched.py` | `camera_ready/results/weakreal_matched.json` |
| §3.4, S7.4 recall denominators (0.884 vs 0.784) | `camera_ready/eval/diag_recall_denominators.py`; EC-version audit `camera_ready/strat/ecversion/ec_audit.py` | `camera_ready/results/recall_denominators.json`, `camera_ready/strat/ecversion/ec_audit.json` |
| §3.4, S7.6 gene essentiality (eight organism–medium–reference combinations, 2×2 decomposition, eukaryotic-reaction sensitivity) | `camera_ready/essentiality/gene_essentiality_demo.py`, `medium_flip_analysis.py`, `rescoring_2x2.py`, `essentiality_domainfilter.py`, `forensic.py`, `joyce_crosscheck.py`; `run_*.sbatch` | `camera_ready/essentiality/results/{essround3,medflip,rescoring2x2,domainfilter,forensic,minmed}/`, references under `camera_ready/essentiality/ref/` (see its `README.md`) |
| §4.1, S7.8, Table S26 recovery stratified by predictor vocabulary | `camera_ready/strat/strat_all.py --remap`, `aggregate.py` | `camera_ready/strat/results/strat_remap/`, `strat_remap_summary.json`; unmapped protocol in `strat/` and `strat_summary.json` |
| S4 paired gap-filler timing (COBRApy MILP vs flux-parsimony LP) | `eval/cgf2.py`, `eval/agg_gapfill.py` | `results/gapfill_paired/` |
| S7.1 CarveMe structural metrics on the 108-genome panel | `camera_ready/massimbal_baseline/carveme_structural_metrics.py`, `gate_table1_v2.py` | `results/table1_v2/carveme_panel108.json`, `carveme_reference_summary.json`, `camera_ready/massimbal_baseline/carveme_memote_panel108_summary.json` |
| Batch launchers for the above | `camera_ready/sbatch/t*.sbatch` | |

`camera_ready/PROVENANCE_code_snapshot.txt` records the frozen copy of `src/`
and `eval/` that the revision scripts import (identical to the tagged code in
the MILP, repair and mapping logic); `camera_ready/conda_cobra_list.txt` is
the environment they ran in. Run the figure scripts from the repository root;
outputs go to `figures/`. The figures use Arial (Liberation Sans is
metric-compatible); with a wider fallback font the layout self-check in
`gen_fig_results.py` flags the panel c note as touching data.
`supplementary_v8.pdf` at the repository root is the camera-ready supplement.
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
