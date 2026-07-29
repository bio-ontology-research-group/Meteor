# Data files

Everything `meteor_v8.utils.data_path` resolves. With this directory present
the pipeline runs from a clean checkout; nothing reaches outside the
repository. Point `$METEOR_DATA` elsewhere to override the location.

## Inputs

| file | what it is |
|---|---|
| `universal.pickle` | ModelSEED `Prokaryote_Universal` reaction database (47,880 reactions, 23,013 metabolites), as distributed with Reconstructor v1.1.0. The MILP selects from this. |
| `seedr2ec.pkl`, `seedec2r.pkl` | SEED reaction to EC mapping and its inverse, derived from the Reconstructor distribution. 20,055 of the universal's reactions carry at least one EC; 1,650 carry more than one. |
| `all_ancestors.txt` | The EC vocabulary predictions are reindexed onto, each four-digit EC with its ancestors. Anything a predictor emits outside this list is dropped. |
| `all_ec.txt` | Flat EC list, without the ancestor expansion. |
| `enzymeobsolete.pkl` | Obsolete or renumbered EC to its current equivalent. `extract_pred` walks this before reindexing, so a baseline score under a retired number is not silently discarded. |
| `medium.pkl`, `medium_info.pkl` | Medium definitions. Every genome in the paper is solved under `default`, a single minimal medium. |

## Derived, and regenerable

These are caches. Delete them and the code recomputes them from
`universal.pickle`; they are shipped because recomputing costs minutes per
run and because shipping them guarantees the matrices are bit-identical to
the ones behind the published numbers.

| file | how to regenerate |
|---|---|
| `fba_matrices_v6RE.pkl` | `extract_fba_matrices(universal, allrxns, reversed_trans=True)` writes it on first call. Holds S, lb and ub with exchange columns sign-flipped. |
| `tight_bounds_v6_pos.pkl`, `tight_bounds_v6_neg.pkl` | FVA-tightened bounds, one file per Gram group. Optional: `load_tight_bounds` warns and falls back to the universal's own bounds if absent, which is slower and gives a looser relaxation. |

The `noRE` variant of the matrix cache is not shipped. No script in this
release calls `extract_fba_matrices` with `reversed_trans=False`.

## Provenance

ModelSEED and the Reconstructor distribution carry their own licences; this
directory redistributes their content for reproducibility, and their terms
govern that content rather than this repository's MIT licence.
