# Raw MILP solutions

`milp_solutions.tar.gz` holds the 108 per-genome MILP solutions for the
DeepProZyme-vanilla configuration, one pickle per assembly:

```
tar xzf results/milp_solutions.tar.gz -C results/
```

They expand to about 80 MB from 3 MB, since the flux vectors are mostly zero.
Each pickle holds `y_vals` (the binary selection), `v_vals` (fluxes),
`n_active`, `n_repaired`, `biomass_flux`, `status` and `solve_sec`.

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
