# R1 feasibility: recall stratified by predictor vocabulary / confidence + dark proteome

Demo run 2026-09-16, E. coli iML1515 (GCF_058436375.1), login node, 3:38 wall, 1.9 GB RSS.
Script: strat_demo.py (read-only on all inputs; writes only strat_<gem>.json here).
Log: demo_iML1515.log ; JSON: strat_iML1515.json

Reproduce:
  source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
  cd /ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_r1_strat
  python -u strat_demo.py iML1515                       # one organism, ~3.5 min
  python -u strat_demo.py iML1515 STM_v1_0 iYL1228 iJN1463 iYS854 iYO844   # all 6, ~20 min -> sbatch 1 cpu 8G 40 min

Definitions:
  vocab       = 4-digit ECs in the raw predictor score matrix columns (fixed per predictor: CLEAN 5235, DPZ 2829, EnzBERT 4746)
  METEOR set  = meteor_preds_<gcf>.pkl['active_ecs'] of the matching {predictor}_vanilla run (Table 3 convention; dpz R_all=0.905 == Table 3)
  baseline    = ECs with max-over-proteome score >= 0.5 (same-predictor threshold convention)
  conf bin    = max-over-proteome baseline score of the curated EC: 0 (incl. out-of-vocab) / (0,0.1) / [0.1,0.5) / [0.5,1]
  skeleton-expressible = ECs that rfinal2ec() can emit = seedr2ec over the 47,880 universal reactions (6041 4-digit ECs)
  dark        = fraction of proteins whose max score over all ECs is < t (t=0.1/0.3/0.5). Threshold-dependent proxy, not the predictors' native callers.

Sanity: METEOR recall_all for dpz = 0.905 reproduces Table 3 exactly. 863/865 iML1515 ECs are skeleton-expressible (2 unreachable by anyone).

## P1/P2 (2026-09-16) — strat_all.py / aggregate.py / run_strat.sbatch
Decisions taken with the coordinator:
- Two EC-mapping conventions are kept side by side in every JSON. `M-rxn`, `S`, `F` = seedr2ec over selected
  reactions restricted to all_ancestors.txt (S7.4 convention; reproduces 0.816 for E. coli). `M-rxn-allSEED`,
  `S-allSEED` = same without the restriction (rfinal2ec convention). `M-act` (= preds pkl active_ecs) equals
  `M-rxn-allSEED` exactly. Compare like with like: M-act vs S-allSEED, M-rxn vs S; never across conventions.
- Vocab = 4-digit ECs in the RAW score-matrix columns (frozen). One DPZ column is an obsolete EC that
  extract_pred remaps to its current number, so counts differ by 1 from strat_demo.py (744/121 vs 745/120 for E. coli).
- "M-act ⊇ B by construction" is false; per-organism counterexample counts are in sanity.b_not_in_mact
  (all ECs) and sanity.b_curated_not_in_mact (curated only; 8/596 for E. coli/DPZ).
- H3 = `no_candidate_rxn`: curated ECs none of whose seedr2ec reactions is in the published candidate mask
  (w>=0.01 U skeleton U media minus excludes, w rebuilt exactly as emit_v8.py). Gate: n_candidate == T4 full n_candidate.
  The earlier "universal-only & score<0.01" set is NOT a ceiling (kept in JSON as H3_universal_only_* for the record).
- S is ONE skeleton-only solve per organism under DPZ cost flags (T4 skelonly y*), reused for all three predictor
  rows; no per-predictor re-solve (P2b dropped). Footnote in the paper.
- Gate "S n_selected ≈ 1,985 ± 30" was the 108-panel mean; replaced by "S n_selected == T4 skelonly n_selected for that GCF".
Memory: strat_all.py peaks at ~4.2 GB (dense 47,880 x 7,433 float64 mask matmul for w); sbatch asks 8 GB.
