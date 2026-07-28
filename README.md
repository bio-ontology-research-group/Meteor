# METEOR — evidence-weighted parsimony for network-consistent enzyme annotation

Code and deposited results for the PSB 2027 submission *METEOR:
Network-consistent enzyme annotation while preserving per-protein predictions*.

METEOR takes a per-protein EC confidence matrix, aggregates it into
reaction-level evidence, selects a biomass-feasible reaction set with a MILP
whose cost carries an evidence-weighted parsimony penalty, and writes the result
back to the protein score matrix through a ranking-preserving update.

## Contents

```
src/meteor_v8/   method: milp_v8.py (MILP), repair.py (verify-and-repair),
                 milp_hard.py (biomass-feasible skeleton)
eval/            one script per table or figure, plus the aggregators
cohorts/         COHORTS.json/tsv, panel108_gram.tsv, solver status
results/         the records every reported number is computed from
env/             conda environment
```

## Reproducing the published numbers

These run against this checkout with no further input:

```bash
conda env create -f env/meteor_cobra.yml && conda activate cobra
python eval/agg_table1.py    # Table 1; dead-end reduction 75.8-90.6%
python eval/agg_decoy.py     # Figure 2; 34.5 / 142.4 / 2007.7 spurious
                             # reactions, 4.13x and 58.3x, p=1.9e-19
```

## Reconstructing a genome

```bash
python eval/emit_v8.py --gca GCF_000006945.2 --gram negative \
    --baseline dpz --variant vanilla \
    --penalty evw --pexp 2 --mu 3 --eps 0.0 --gmin 0.1 \
    --outroot <output directory>
```

Pass the flags explicitly. The argparse defaults now match the paper, but
earlier runs that omitted `--eps` silently used a forced-flux floor of 0.01,
which is a different optimisation problem.

## Where each number comes from

| paper item | script | records |
|---|---|---|
| Table 1 (structural) | `eval/gen_table1.py` → `eval/agg_table1.py` | `results/table1/` |
| Table 2 (KEGG detection) | `eval/kegg_detect.py` | `results/toolcompare/kegg_pathway_108_detection.json` |
| §3.2 selectivity | `eval/kegg_select.py` | `results/toolcompare/kegg_pathway_108_selectivity.json` |
| §3.2 Price-149 | `eval/kegg_price.py` | `results/toolcompare/kegg_price_matched_detect.json` |
| Figure 2, §3.3 | `eval/recovery_ablation.py` → `eval/agg_decoy.py`; figure drawn by `eval/gen_fig_decoy.py` | `results/recovery_abl/` |
| §3.4 holdout | `eval/holdout_cfg.py` | `results/holdout_cfg/` |
| §3.4 full proteome | `eval/net_seqident.py` | `results/toolcompare/net_seqident.json` |
| §3.4 CarveMe | `eval/toolcompare_ec.py` | `results/toolcompare/panelB_ec.json` |
| §3.4 threshold baseline | `eval/baseline_thresh_gapfill.py` | `results/toolcompare/ablation_thresh_vs_evw_ec.json` |
| Figure S1 | `eval/gen_fig_massimbal.py` (draws the figure) | `results/table1/` |
| S8 MEMOTE | `eval/memote_evw.py` | per-genome MEMOTE JSONs (not deposited, see below) |

Scripts whose name starts with `gen_fig_` draw a figure in the paper and say
so in their docstring; `gen_table1.py` and the two `agg_` scripts produce
tables.

`src/meteor_v8/` contains no absolute paths. The analysis scripts under `eval/`
are deposited as they were run and reference input locations on our cluster; to
re-run one, point the path constants at your own copies of the inputs below.

## Parameters

`p=2`, `mu=3`, `k=5` (score truncation), `w_min=0.01` (candidate mask),
`gamma_min=0.1`, `gamma_max=2.5`, `lambda=1e-4`, `epsilon=0` (log-odds
smoothing), `delta=0` (forced-flux floor, disabled), `alpha=0.4` (score update).
CBC via PuLP 3.3.1, four threads, 600 s limit, `gapRel=0.05`. The manuscript
calls the forced-flux floor `delta`; the flag is `--eps`.

## Inputs not deposited here

The SEED universal database (47,880 reactions), the baseline EC prediction
matrices (~5 GB per predictor-cutoff configuration), the genome proteomes, and
the full 12-configuration MILP output (~336 GB). This repository carries the 108
`dpz_vanilla` MILP solutions the main-text tables are computed from, plus every
aggregated result. ModelSEED is available from the ModelSEED project and
Reconstructor v1.1.0; assemblies are on NCBI RefSeq under the accessions in
`cohorts/panel108_gram.tsv`.

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
