# Runnable example input

`GCF_000017425.1_DPZ_top5.pkl.gz` is one genome's per-protein EC score matrix,
enough to run `eval/emit_v8.py` end to end without the undeposited baseline
prediction matrices.

- **Source**: the DeepProZyme-vanilla prediction matrix for GCF_000017425.1,
  3,846 proteins.
- **Processing**: passed through `meteor_v8.utils.extract_pred` (obsolete-EC
  remapping and reindexing to `data/all_ancestors.txt`, giving 7,433 EC
  columns), then reduced to each protein's five highest-scoring ECs.
- **Why that is lossless**: `emit_v8.py` applies the same top-5 truncation
  before aggregating, so the reduced matrix drives the pipeline exactly as the
  full one does. Verified: the full 43 MB matrix and this 0.27 MB file select
  the same 3,119 reactions.

The selection differs from the deposited solution for this genome (3,123
reactions) by 56 reactions, Jaccard 0.982. That is CBC choosing among
degenerate optima, not an effect of the reduced input; both inputs above give
the same 3,119.
