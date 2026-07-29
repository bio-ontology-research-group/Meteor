# Raw MILP solutions

The 108 per-genome MILP solutions for the DeepProZyme-vanilla configuration
(`meteor_sol_{accession}.pkl`, ~80 MB in total) are not kept in this repository:
they would make up 86% of the archive while nothing in the paper is computed
from them directly.

Every reported number comes from the aggregated per-genome records that *are*
here — `results/table1/`, `results/recovery_abl/`, `results/holdout_cfg/`,
`results/fva_selected/`, `results/fva_baseline/` and `results/toolcompare/` —
and the three aggregation scripts reproduce the published figures from those
alone:

```
python eval/agg_table1.py
python eval/agg_decoy.py
python eval/agg_fva.py
```

Each solution pickle holds `y_vals` (the binary selection), `v_vals` (fluxes),
`n_active`, `n_repaired`, `biomass_flux`, `status` and `solve_sec`. To
regenerate one, run `eval/emit_v8.py` for that accession with the flags in
README.md; to obtain the exact solutions used here, contact the authors.
