# Raw MILP solutions

`milp_solutions.tar.gz` holds the 108 per-genome MILP solutions for the
DeepProZyme-vanilla configuration, one pickle per assembly:

```
tar xzf results/milp_solutions.tar.gz -C results/
```

They expand to about 80 MB from 3 MB, since the flux vectors are mostly zero.
Each pickle holds `y_vals` (the binary selection), `v_vals` (fluxes),
`n_active`, `biomass_flux`, and `status`. The three genomes that needed a
post-solve repair also carry `n_repaired` and `maxbio_core`.

`status` here is the formulation label `MILP_hard`, not a solver termination
status: these solutions were written by an earlier revision of `emit_v8.py`
that did not persist the CBC status or the wall time. The current script does,
and `results/solver_status_dpz_vanilla.json` holds those fields for all 108
genomes from a rerun at the published flags. Selected-set sizes there agree
with these solutions in 107 of 108 genomes; the remaining genome differs by 6
reactions, which is CBC choosing among degenerate optima.

Nothing in the paper is computed from these directly. Every reported number
comes from the aggregated per-genome records also deposited here --
`results/table1/`, `results/recovery_abl/`, `results/holdout_cfg/`,
`results/fva_selected/`, `results/fva_baseline/`, `results/toolcompare/` -- and
the three aggregation scripts reproduce the published figures from those alone:

```
python eval/agg_table1.py
python eval/agg_decoy.py
python eval/agg_fva.py
```

To regenerate a solution rather than unpack it, run `eval/emit_v8.py` for that
accession with the flags in README.md.
