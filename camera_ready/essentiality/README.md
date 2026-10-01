# feasibility_essentiality — in-silico gene essentiality from METEOR output (R2/R3 item)

Created 2026-09-16. Everything here is new; nothing outside this directory was modified.
Read-only inputs: psb_revision/code_snapshot, meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla,
paperA_2026/{baseline_preds,benchmark/proteomes}, gapseq_eval/gapseqEval/GeneEssentiality.

## Files
- gene_essentiality_demo.py   GPR construction (protein p -> reaction r if E[p,ec]>=0.5 for an ec of r, OR-only),
                              COBRA model from selected set (build_submodel), single_gene_deletion (cobrapy 0.31, glpk),
                              metrics vs gess-ecol.csv ess.experimental. Arms: meteor, meteor_Enew, thresh, iml1515.
- diag_medium.py              why the SEED-based models do not grow on GS_MM_glc (auxotrophies list).
- run_essentiality.sbatch     1 job, 4 cpu, 24G, 1h (job 51930975: 46 s, 1.75 GB peak). Second run done on login node (3m12s, 1.1 GB).
- ref/mg1655.faa              MG1655 RefSeq proteome (copied from gapseq_eval FermentationTest)
- ref/GCF_000005845.2_ASM584v2_feature_table.txt, ref/np2b.tsv   NP_ accession -> b-number (fetched from NCBI FTP)
- ref/fwd.tsv, rev.tsv, m25631h_to_bnum.tsv   diamond RBH (id>=60, cov>=70): 3762/5231 strain proteins -> b-number
- results/essentiality_dpz_vanilla_GCF_058436375.1.json   all metrics; results/sgd_*.json per-gene calls; results/diag_medium.json
- logs/

## Reproduce
source /ibex/user/niuk0a/anaconda3/etc/profile.d/conda.sh && conda activate cobra
module load diamond/2.1.16
cd /ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality
# (ref/ already built; commands in the report)
python gene_essentiality_demo.py --gca GCF_058436375.1 --gram negative --procs 2     # ~3 min on login node
python diag_medium.py
# or: sbatch run_essentiality.sbatch

## Caveats to carry into the response letter
1. Panel E. coli is GCF_058436375.1 = strain M25631H (chicken isolate), not K-12; genes mapped to b-numbers by RBH.
2. GPRs are OR-only (isozymes); complexes (AND) are not inferable from E -> essentiality underestimated.
3. Reference (gapseq compilation, glucose-minimal, iML1515 gene set) vs METEOR model medium (post-hoc inferred, rich:
   CoA, biotin, thiamine-P, fatty acids, Gln/Met/Cys/His uptaken). Glucose-minimal growth = 0 for METEOR and threshold models.
4. On this metric METEOR (MCC 0.32) == threshold baseline (MCC 0.32); iML1515 = 0.76. It is a sanity check, not a differentiator.

## Round 2 (2026-09-17): does the MILP re-solve under a strict minimal medium? (feasibility only, no GPR/SGD)

Scope: swap the pipeline's "default" medium for the strict minimal medium the
essential-gene reference tables assume, re-solve the same evidence-weighted
MILP (published flags), and check status/growth. NO gene essentiality run in
this round.

Files added: `resolve_minmed.py`, `run_minmed.sbatch` (job 51997259: 21m08s,
4.93 GB peak, 4 cpu; superseded job 51996240 cancelled — first version had a
sign-convention bug in the FBA rebuild, see below), `verify_signfix.py`
(one-off diagnostic), `results/minmed/*.json`.

**Medium definitions used (verified from code, not assumed):**
- `GS_MM_glc`: `gapseq_eval/.../GeneEssentiality/media/media.tsv`, 24 rows
  (glucose + inorganic salts + NH4/CO2/O2/H2O/phosphate/sulfate). Used in
  `compare_GE_recons.R` for the **E. coli** essentiality comparison only
  (`ecol.gs/.cm/.ms` all constrained to `GS_MM_glc`).
- **`GS_MM_glc` is NOT the medium bsub's reference table assumes.**
  `compare_GE_recons.R` constrains `bsub.gs/.cm/.ms` to **`LB_marinos`** (62
  compounds: all 20 amino acids, nucleotides, hemes, vitamins — a rich
  medium, not minimal). Ran the required GS_MM_glc combo for bsub anyway
  (as instructed) plus a bonus LB_marinos run for the biologically correct
  comparison.
- Bypassed `meteor_v8.utils.apply_media()`: it hardcodes a `minimal_uptake`
  set (~30 exchanges including **all 20 amino acids**) unioned into every
  medium regardless of name, so simply passing "GS_MM_glc" through it would
  not have been a strict test. Wrote `apply_strict_medium()` to replicate
  its exchange-bound logic without that union.

**Bug found and fixed mid-round:** naively assigning matrix `lb[i]/ub[i]`
to cobra EX-reaction bounds (exactly what `meteor_v8/eval/verify_selected_growth.py`
and `gen_table1.py::profile()` already do) is WRONG, because
`extract_fba_matrices(reversed_trans=True)` flips the S-column sign for
every EX reaction except `EX_biomass`. Verified with `verify_signfix.py` on
the full non-excluded E. coli network under GS_MM_glc: naive growth=19.24,
sign-corrected growth=250.0 == skeleton LP `bm_max` (ground truth). This is
almost certainly why `meteor_v8/results/growth_verify/*/summary.json`
reports `growth_fixed=0` for **all 108 genomes** under the pipeline's own
published default medium — a pre-existing cobra-reconstruction bug, not
genuine metabolic infeasibility. `resolve_minmed.py` reports both
`fba_growth_naive_buggy` (always 0.0, reproducing the known artifact) and
`fba_growth_under_this_medium` (sign-corrected, trustworthy).

**Result table** (all `status=Optimal`, `milp_biomass_flux=0.1` = hard gmin floor, all feasible):

| organism | predictor | medium | solve_sec | n_selected | fba_growth (sign-fixed) |
|---|---|---|---|---|---|
| E. coli GCF_058436375.1 | clean | GS_MM_glc | 148.0 | 3056 | 6.00 |
| E. coli | dpz | GS_MM_glc | 114.3 | 3686 | 4.40 |
| E. coli | enzbert | GS_MM_glc | 172.1 | 2991 | 11.67 |
| B. subtilis GCF_058182495.1 | clean | GS_MM_glc | 348.3 | 2154 | 4.33 |
| B. subtilis | dpz | GS_MM_glc | 94.9 | 3213 | 3.76 |
| B. subtilis | enzbert | GS_MM_glc | 111.0 | 2368 | 11.23 |
| B. subtilis (bonus) | dpz | LB_marinos (correct ref medium) | 96.3 | 3253 | 0.95 |

Skeleton LP (unconstrained, any universal reaction) always feasible with
`bm_max=250` (the LP's upper bound cap), confirming GS_MM_glc/LB_marinos can
in principle support growth given the full reaction universe — the question
was whether the evidence-restricted MILP selection could too, and it can, in
every one of the 7 combos tested.

**Verdict per combination: FEASIBLE (MILP solves to Optimality, positive
growth under the swapped medium) for all 6 required combos + the bonus.**
No infeasible case arose, so the "which precursors are blocked" diagnostic
branch in `resolve_minmed.py` was never exercised in this round.

**Caveat carried forward:** growth magnitudes (0.95–11.7) are not directly
comparable to the reference tables' own growth assumptions and this round
does **not** establish that gene essentiality calls will be sensible — it
only clears the prerequisite ("can the selected model grow under this
medium at all"). Proceed to GPR/deletion only after this handback.

## Round 3 (2026-09-17): gene essentiality on the 6 correct-medium combos

Scope: E. coli x3 predictors under GS_MM_glc (its correct reference medium),
B. subtilis x3 predictors under LB_marinos (its correct reference medium —
NOT GS_MM_glc; the earlier GS_MM_glc/bsub results in `results/minmed/` are
discarded diagnostics, not used here). METEOR arm + threshold-baseline arm
(EC>=0.5 draft, gap-filled only if the raw draft doesn't grow — it never did,
+58 to +97 reactions in all 6 cases, matching S7.4) + curated-GEM sanity
check (iML1515 for E. coli, iYO844 for B. subtilis — both present in
`curated_gems/`).

Files added: `essentiality_correct_medium.py`, `run_ess_ecoli.sbatch`
(job 51998676: 8m59s, 4.19GB), `run_ess_bsub.sbatch` (job 51998677: 10m43s,
4.20GB), `run_ess_clean_fix.sbatch` (job 51998966: 9m09s, 3.55GB — reran
just the 2 "clean" combos after a bug fix, see below), `ref/GCF_000009045.1_ASM904v1_*`,
`ref/bsub168.faa`, `ref/np2symbol_bsub.tsv`, `ref/fwd_bsub.tsv`, `ref/rev_bsub.tsv`,
`ref/knb1_to_symbol.tsv` (RBH: panel B. subtilis KnB1 -> strain-168 reference,
3739 RBH / 3704 with a gene symbol, out of 3967 panel proteins),
`results/essround3/*.json`.

**B. subtilis strain identity (verified from code, not assumed):**
panel genome GCF_058182495.1 = strain **KnB1** (Kasetsart University, PGAP,
2026), NOT strain 168. `gess-bsub.csv`'s reference is strain 168 (the genome
in `gapseq_eval/.../genomes/bsub.fna.gz` header is explicitly "str. 168";
iYO844's BSU-locus-tag genes are also strain-168 nomenclature). Built the
same RBH strategy as E. coli, but matched by **gene SYMBOL** (gess-bsub.csv's
`gene2` column is 844/844 populated with standard symbols like `asd`,
`dapA`, `glyA` — much more robust than E. coli's empty `gene2`, which forced
b-number matching there) against strain-168's own NCBI feature-table
symbol column, itself linked to the KnB1 panel proteome via diamond RBH.
iYO844's curated model gene IDs are BSU locus tags, not symbols, so its
comparison uses each `cobra.Gene.name` (already a symbol, e.g. `atpF`) as
the match key.

**Bug found and fixed mid-round:** the `clean` (CLEAN_confidence) baseline's
prediction-matrix row index is the FULL FASTA description line (e.g.
`"YFJ07105.1 DeoR/GlpR family DNA-binding transcription regulator
[Escherichia coli]"`), not a bare accession — unlike `dpz`/`enzbert`, whose
index is already a bare accession. Passing that string straight into
`gene_reaction_rule` breaks cobra's GPR parser (its AST-based tokenizer
chokes on spaces/`/`/`[]`), and cobra **silently drops every gene** with
only a `SyntaxWarning` printed to stderr, leaving `model.genes` empty (0
genes despite thousands of reactions carrying a nonempty
`gene_reaction_rule` string). Combined with a separate truthiness bug in my
own script (`if ess else ...` treated a valid-but-empty `single_gene_deletion`
result the same as `ess is None`), this made both `ecoli/clean` and
`bsub/clean` silently report `"err": "no growth"` in the first pass despite
positive WT growth. Fixed by stripping to `pred.index[i].split()[0]` before
using it as a gene ID (harmless no-op for dpz/enzbert, whose index has no
whitespace) and by checking `ess is not None`. Reran only the 2 affected
combos (job 51998966); dpz/enzbert results from the first pass (jobs
51998676/51998677) were already correct and were kept.

### Result table (6 required combos)

| organism | predictor | arm | n_selected | n_genes(model) | n_pred_ess | precision | recall | MCC |
|---|---|---|---|---|---|---|---|---|
| E. coli (GS_MM_glc) | clean | meteor | 3056 | 1105 | 55 | 0.764 | 0.207 | 0.314 |
| E. coli | clean | thresh (+59 gapfill) | 3056/3115 | 1111 | 65 | 0.769 | 0.246 | 0.347 |
| E. coli | dpz | meteor | 3686 | 1728 | 45 | 0.767 | 0.157 | 0.283 |
| E. coli | dpz | thresh (+58) | 3345 | 1732 | 66 | 0.730 | 0.219 | 0.322 |
| E. coli | enzbert | meteor | 2991 | 1415 | 71 | 0.697 | 0.232 | 0.308 |
| E. coli | enzbert | thresh (+69) | 2932 | 1419 | 62 | 0.614 | 0.177 | 0.231 |
| B. subtilis (LB_marinos) | clean | meteor | 2182 | 827 | 65 | 0.451 | 0.411 | 0.332 |
| B. subtilis | clean | thresh (+97) | 2167 | 829 | 76 | 0.352 | 0.339 | 0.229 |
| B. subtilis | dpz | meteor | 3253 | 1312 | 50 | 0.405 | 0.279 | 0.247 |
| B. subtilis | dpz | thresh (+82) | 2470 | 1318 | 51 | 0.410 | 0.262 | 0.242 |
| B. subtilis | enzbert | meteor | 2413 | 1105 | 56 | 0.564 | 0.379 | 0.389 |
| B. subtilis | enzbert | thresh (+77) | 2346 | 1110 | 55 | 0.429 | 0.305 | 0.269 |

**Curated-GEM sanity check:** iML1515 P=0.838 R=0.755 MCC=0.758 (n=1516
genes, matches round-1 exactly, medium-independent since it uses the GEM's
own default bounds). iYO844 P=0.340 R=0.688 MCC=0.390 (n=844 genes) — a much
weaker sanity anchor than iML1515; B. subtilis's own curated model doesn't
reproduce its reference essentiality table nearly as well as E. coli's does,
so the ceiling for what METEOR/threshold can be expected to achieve on
B. subtilis is intrinsically lower than the iML1515 comparison suggests.

**Head-to-head, mean MCC across 3 predictors:** E. coli METEOR 0.302 vs
threshold 0.300 (statistical tie, as in round 1). B. subtilis METEOR 0.323
vs threshold 0.247 — METEOR is ahead here, driven mostly by `enzbert`
(0.389 vs 0.269) and `clean` (0.332 vs 0.229); `dpz` is a near-tie (0.247 vs
0.242). This is the first result across all three rounds where METEOR shows
a real margin over the threshold baseline on an external validation metric,
though it rests on a much smaller reference set (425/391/364 B. subtilis
genes with both a model gene and a reference call, vs 763-898 for E. coli)
and a weaker curated-model ceiling (MCC 0.39 vs 0.76), so the margin should
be reported cautiously.

## Round 4 (2026-09-17): E. coli under a RICH medium (LB_marinos) vs PEC — rich-vs-minimal control

Purpose: test whether METEOR's apparent B. subtilis edge (round 3) is a
"rich medium is easier" artifact rather than an organism effect, by running
E. coli under the same LB_marinos compound set already validated for
B. subtilis, scored against a genuinely rich/condition-independent E. coli
reference (PEC) instead of gess-ecol.csv (which most likely reflects
Monk et al. 2017's M9-carbon-source screens, not a rich-medium call — see
`ref/ecoli_medium_provenance.txt` for full verbatim sourcing from Baba 2006,
Yamamoto 2009 and Monk 2017).

**Medium used: `LB_marinos`** (gapseq's own compound set, 62 cpds, the same
one validated feasible for B. subtilis in round 3) — chosen for consistency
with that run rather than `data/medium.pkl`'s own `'LB'` entry, so the two
organisms' rich-medium results are directly comparable; both use identical
medium-construction code (`apply_strict_medium`, no `minimal_uptake` union).

**Reference: PEC** (Profiling of E. coli Chromosome,
https://shigen.nig.ac.jp/ecoli/pec/), downloaded directly (no gate) to
`ref/pec/PECData.dat`, reduced to `ref/pec/pec_ecoli_binary.csv` (gene
[b-number from `Alternative name`], ess.experimental [yes/no from
Class 1/2], pec_class, pmid) — 287 essential / 4029 non-essential / 3
unknown (dropped) among 4319 protein-coding-gene rows. Matched via the
existing `m25631h_to_bnum.tsv` RBH mapping (no new mapping needed — PEC is
MG1655-based, same as gess-ecol.csv).

Files added: `resolve_minmed.py` reused unmodified; `essentiality_correct_medium.py`
extended with an `"ecoli_lb"` organism config (medium=LB_marinos,
ref=pec_ecoli_binary.csv, curated_gem=iML1515); `run_minmed_ecoli_lb.sbatch`
(feasibility, job 52002498: 8m19s, 3.90GB — all 3 predictors Optimal,
growing); `run_ess_ecoli_lb.sbatch` (job 52002731: 8m43s, 4.09GB — job
52002661 superseded after a CSV-quoting bug in the first PEC-derivation
pass, embedded commas in the PMID column broke the naive f-string writer;
fixed with `csv.writer`/`QUOTE_MINIMAL`).

### Feasibility (all 3 Optimal, growing under LB_marinos)

| predictor | status | n_selected | FBA growth (sign-fixed) |
|---|---|---|---|
| clean | Optimal | 3086 | 3.03 |
| dpz | Optimal | 3716 | 4.51 |
| enzbert | Optimal | 3013 | 11.96 |

### Essentiality result: E. coli, LB_marinos, vs PEC

| predictor | arm | n_selected | precision | recall | MCC |
|---|---|---|---|---|---|
| clean | meteor | 3086 | 0.220 | 0.074 | 0.059 |
| clean | thresh (+60 gapfill) | 3057 | 0.382 | 0.174 | **0.190** |
| dpz | meteor | 3716 | 0.256 | 0.079 | 0.093 |
| dpz | thresh (+62) | 3349 | 0.267 | 0.086 | 0.102 |
| enzbert | meteor | 3013 | 0.171 | 0.059 | 0.039 |
| enzbert | thresh (+77) | 2940 | 0.231 | 0.102 | 0.088 |

**Mean MCC: METEOR 0.064 vs threshold baseline 0.127 — threshold baseline
wins under PEC, by roughly 2x, in all 3 predictors individually, not just
on average.** This is the opposite ranking from round 3's B. subtilis/LB_marinos
result (METEOR 0.323 vs threshold 0.247, METEOR ahead) and from round 3's
E. coli/GS_MM_glc result (METEOR 0.302 vs threshold 0.300, a tie). All
three organism/medium combinations tried so far individually go a
different way (tie / METEOR-ahead / threshold-ahead) — no consistent
"rich medium favors METEOR" or "rich medium favors threshold" pattern
across the two organisms, arguing against a simple universal
"rich-medium-is-easier" artifact explaining round 3's B. subtilis result,
but also making clear the B. subtilis edge is not a stable, medium-independent
METEOR advantage either — it appears to be a combination-specific result
that does not generalize even within the same medium (LB_marinos) across
organisms.

### iML1515 vs PEC (curated-GEM sanity/ceiling check)

| reference | precision | recall | MCC |
|---|---|---|---|
| gess-ecol.csv (round 1/3, M9-based per Monk 2017) | 0.838 | 0.755 | 0.758 |
| **PEC** (this round, condition-independent/rich-derived) | 0.503 | 0.791 | **0.592** |

PEC's stricter, condition-independent essential-gene bar (287 genes,
smaller and differently composed than gess-ecol.csv's 234 GEM-restricted
M9-based calls) **does noticeably lower iML1515's own ceiling** (MCC 0.758
-> 0.592, driven by precision collapsing 0.838 -> 0.503 while recall
slightly rises 0.755 -> 0.791) — i.e. iML1515, grown and evaluated under
its own standard M9-glucose default medium (not re-solved under LB_marinos
for this check), calls many genes essential that PEC's aggregate-across-all-conditions
definition does not, producing more false positives against PEC. This
confirms the reference choice materially changes the achievable ceiling,
independent of which reconstruction method is being scored — a caveat that
should accompany any of these MCC numbers in the response letter.

## Round 5 (2026-09-20): medium x reference-table 2x2 decomposition for E. coli (pure re-scoring, no new compute)

Purpose: isolate whether the round-4 "flip" (METEOR/threshold tie under
GS_MM_glc+gess-ecol.csv -> threshold clearly ahead under LB_marinos+PEC) is
driven by the medium swap, the reference-table swap, or their interaction.
**No MILP re-solve, no COBRA rebuild** -- pure re-scoring of the
single-gene-deletion calls already saved in `results/essround3/sgd_{ecoli,ecoli_lb}_{meteor,thresh}_{clean,dpz,enzbert}.json`
against both reference tables (gess-ecol.csv, PEC), via the same
protein->b-number mapper (`ref/m25631h_to_bnum.tsv`) both networks already
use. Script: `rescoring_2x2.py` (runs in <1s on the login node -- pure
JSON/dict scoring, no solver, no cobrapy model construction).

Note: `pec_ecoli_binary.csv` has 4316 rows but 27 duplicate b-number keys
(split/duplicated PEC ORF entries), collapsing to 4284 unique genes (285
essential) when loaded as a dict -- immaterial (<1% of rows), flagged for
the record; does not affect any conclusion below.

### Mean-over-3-predictors MCC, full 2x2

| medium \ reference | gess-ecol.csv | PEC |
|---|---|---|
| **GS_MM_glc** | METEOR 0.302, thresh 0.300 (tie) | METEOR 0.052, thresh 0.105 (**thresh 2x ahead**) |
| **LB_marinos** | METEOR 0.248, thresh 0.266 (thresh slightly ahead) | METEOR 0.064, thresh 0.127 (**thresh 2x ahead**) |

(Diagonal cells GS_MM_glc/gess-ecol.csv and LB_marinos/PEC reproduce
rounds 1/3 and 4 exactly -- 0.302/0.300 and 0.064/0.127 -- confirming the
re-scoring pipeline is consistent with the original runs.)

### Decomposition

**Row-wise (medium fixed, reference swapped: gess-ecol.csv -> PEC):**
- GS_MM_glc: METEOR 0.302->0.052 (-0.250), thresh 0.300->0.105 (-0.195).
  **The reference swap alone flips a tie into a clear threshold win.**
- LB_marinos: METEOR 0.248->0.064 (-0.184), thresh 0.266->0.127 (-0.139).
  Threshold was already slightly ahead; the reference swap sharply widens
  the gap in both arms.

**Column-wise (reference fixed, medium swapped: GS_MM_glc -> LB_marinos):**
- gess-ecol.csv: METEOR 0.302->0.248 (-0.054), thresh 0.300->0.266 (-0.034).
  Small, consistent decline in both arms; ranking barely moves (tie ->
  thresh marginally ahead).
- PEC: METEOR 0.052->0.064 (+0.012), thresh 0.105->0.127 (+0.022). Small
  increase in both arms; ranking unchanged (thresh still ~2x ahead).

### Read: reference-table choice dominates; medium is a minor, inconsistent second-order effect

The reference-table swap alone (holding medium fixed at GS_MM_glc) already
converts the round-1/3 tie into the threshold-clearly-ahead pattern later
seen under LB_marinos+PEC. The medium swap alone (holding reference fixed)
moves both arms by roughly an order of magnitude less than the reference
swap, and moves them in **opposite directions depending on which reference
is held fixed** (down under gess-ecol.csv, up under PEC) -- i.e. it is not
even a consistent, monotonic "medium X is harder" effect on its own, let
alone one that could produce the observed flip by itself. **No genuine
interaction is needed to explain the flip: it is a reference-table effect,
not a medium effect**, and it appears well before the medium is changed at
all. This directly answers round-4's open question (round-4 could not
isolate medium-effect from reference-effect because both changed at once;
this decomposition closes that gap) and further undercuts any
"rich-medium-favors-the-threshold-baseline" or "rich-medium-favors-METEOR"
narrative -- the medium's own marginal effect here is small and
direction-inconsistent across references.

Files: `rescoring_2x2.py`, `results/rescoring2x2/grid.json` (full
2x2x3x2), `results/rescoring2x2/summary_2x2.json` (mean-MCC 2x2 table).

## Round 6 (2026-09-20): gene-level medium-sensitivity check (falsifies/confirms "relatively insensitive to medium")

Purpose: check directly at the gene level (not inferred from MCC deltas)
whether METEOR's predicted essential-gene set for E. coli is
medium-insensitive, using ONLY already-saved per-gene SGD calls (no new
MILP/COBRA compute). Scripts: `medium_flip_analysis.py`,
`joyce_crosscheck.py`. Real-biology benchmark: Joyce AR et al.
"Experimental and computational assessment of conditionally essential
genes in Escherichia coli." J Bacteriol 2006;188(23):8259-71. PMID
17012394, PMC1698209, DOI: 10.1128/JB.00740-06 (According to PubMed).
Table 1 (119 genes, Blattner numbers) fetched directly from the freely
rendered PMC article page (`ref/joyce2006/article.html`, no gate this
time) -- **119/119 genes recovered exactly**, saved to
`ref/joyce2006/joyce119_genes.tsv`.

**Correction to the benchmark's own framing**: Joyce et al. screened the
(LB-viable) Keio collection on **glycerol-supplemented minimal medium**,
not glucose -- "Out of 3,888 single-deletion mutants tested, 119 mutants
were unable to grow on glycerol minimal medium." It is LB-viable ->
glycerol-minimal-inviable, i.e. minimal-essential/rich-dispensable, same
direction as our GS_MM_glc-essential/LB_marinos-dispensable category, but
technically a different carbon source than our glucose-based GS_MM_glc.
Categories per the paper: amino acid metabolism (59), nucleotide
metabolism (19), cofactor metabolism (15), transport (5), other/misc (17),
regulatory (4) = 119.

### Step 1-2: gene-level flip rate, GS_MM_glc vs LB_marinos, vs Joyce's ~3% (119/3888)

| arm | predictor | n_common (both media) | n_flip | flip % |
|---|---|---|---|---|
| meteor | clean | 1105 | 13 | 1.18% |
| meteor | dpz | 1728 | 20 | 1.16% |
| meteor | enzbert | 1414 | 40 | **2.83%** |
| thresh | clean | 1111 | 14 | 1.26% |
| thresh | dpz | 1732 | 18 | 1.04% |
| thresh | enzbert | 1419 | 7 | 0.49% |

Mean flip rate: **METEOR 1.72%, threshold baseline 0.93%** -- both below
Joyce's real ~3%, same order of magnitude (METEOR ~57% of the real rate,
threshold ~31%). **METEOR is roughly twice as medium-sensitive as the
threshold baseline** by this direct gene-level measure, consistent with
(and now confirming at the gene level, not just via MCC) round 5's
observation that METEOR's MCC moved more between media than the
threshold's did in some cells.

### Step 3: functional categories of flipped genes -- matches Joyce's pattern, not scattered

Cross-referencing flipped genes against PEC's `Product` annotation (sample,
`meteor/enzbert` -- the largest flip set): histidine biosynthesis (hisA-I,
7 genes), purine biosynthesis (purC/D/F/L/M), folate/one-carbon (folB,
metF, metB -- Met/Cys/folate), diaminopimelate/lysine (lysA, dihydrodipicolinate
synthase/reductase), branched-chain amino acids (ilvC/D), pyrimidine
(orotate phosphoribosyltransferase, dihydroorotase, OMP decarboxylase in
other predictors), riboflavin (ribD/ribE-type), lipoate/biotin synthesis.
**Overwhelmingly amino acid, nucleotide, and cofactor biosynthesis** --
exactly Joyce's three largest categories (59+19+15 = 93/119 = 78% of their
list) -- not a scattered or unrelated set. A handful of catabolic/transport
genes (rhamnulose kinase, glucose dehydrogenase) also appear, a minor
admixture outside the classic biosynthesis pattern.

### Step 4: direction-correctness against Joyce's real 119 genes

| arm | predictor | Joyce genes in model (both media) | correct direction (min-ess/rich-disp) | wrong direction | not flipped at all |
|---|---|---|---|---|---|
| meteor | clean | 97 | 9 | 0 | 88 |
| meteor | dpz | 105 | 6 | 5 | 94 |
| meteor | enzbert | 95 | 15 | 2 | 78 |
| thresh | clean | 97 | 8 | 1 | 88 |
| thresh | dpz | 105 | 14 | 0 | 91 |
| thresh | enzbert | 96 | 3 | 0 | 93 |

Totals across 3 predictors: **METEOR 30/297 correctly flipped in the
biologically correct direction, 7/297 flipped in the WRONG direction
(81% of its flips are correct); threshold baseline 25/298 correct, 1/298
wrong (96% of its flips are correct).** The dominant outcome for BOTH arms
is "not flipped at all" (78-94 of ~97-105 overlapping Joyce genes per
predictor, i.e. 80-90%) -- confirming the core "insensitive" claim: both
arms under-flip relative to the real ~3% conditional-essentiality rate,
overwhelmingly by failing to flip rather than by flipping incorrectly.

Wrong-direction genes (model calls LB-essential/GS_MM_glc-dispensable, the
opposite of Joyce's minimal-essential/rich-dispensable pattern):
`meteor/dpz`: purF, purD, purM, purC, purL (5 purine-biosynthesis genes,
all in the SAME pathway -- a systematic, pathway-coherent error, not
random noise); `meteor/enzbert`: ilvD, ilvC (branched-chain AA pathway);
`thresh/clean`: argE (1 gene). All in JSON:
`results/medflip/joyce_crosscheck.json`.

### Verdict on the original claim

**"Relatively insensitive to medium" holds up as directionally correct but
needs two qualifications for the paper:**
1. It is not medium-insensitive in an absolute sense (0.5-2.8% flip rate is
   real, non-zero, and concentrated in exactly the expected biosynthesis
   pathways) -- it is insensitive **relative to the true ~3% rate**,
   recovering roughly a third to two-thirds of the real conditional-essentiality
   signal depending on predictor and arm.
2. **METEOR is measurably more medium-sensitive than the threshold
   baseline** (1.72% vs 0.93% mean flip rate; 30 vs 25 correctly-flipped
   real genes), but this comes with a higher direction-error rate (19% of
   METEOR's flips are wrong vs 4% of threshold's), concentrated in a
   pathway-coherent way (the 5 purine genes in dpz, both branched-chain
   genes in enzbert) rather than scattered -- i.e. where METEOR gets the
   direction wrong, it tends to get an entire pathway wrong together,
   which is itself informationally structured, not noise.

Files: `medium_flip_analysis.py`, `joyce_crosscheck.py`,
`results/medflip/flip_analysis.json`, `results/medflip/joyce_crosscheck.json`,
`ref/joyce2006/article.html`, `ref/joyce2006/joyce119_genes.tsv`.

## Round 7 (2026-09-20): 4th core essentiality experiment — B. subtilis / GS_MM_glc / Koo2017-combined-minimal reference

User manually downloaded Koo et al. 2017's Table S3 (257 rows, LB-rich-medium
essential genes, verified row count matches the paper exactly) and Table
S4/tab D (98 rows, curated auxotrophs list, independently verified in the
prior turn: 34/36 literature spot-check genes present, pdhC/pdhD correctly
excluded, gene-ID space usable). This round builds the combined reference
and completes the 4th of 4 core essentiality experiments.

### Combined reference construction (`build_bsub_minimal_ref.py`)

`minimal_essential(gene) = LB_essential(gene) [Table S3] OR auxotrophic(gene) [Table S4/D]`

- Table S3: 257 genes, 15 with multi-name entries like `gcaD (glmU)` --
  split so both names are matchable keys (272 total name-keys from 257 rows).
- Table S4/D: 98 genes, **zero overlap** with Table S3 (confirms internal
  consistency -- Koo's own design: S3 genes could never be deleted at all,
  so they were never candidates for the S4 auxotrophy screen in the first
  place).
- Combined positive (essential) set: **370 gene-symbol keys**.
- **Non-essential ("no") rows**: neither Koo table enumerates the
  non-essential remainder (the raw whole-library fitness sheets were not
  provided, only the curated positive list). To get a full binary table
  shaped like `pec_ecoli_binary.csv`, `gess-bsub.csv` (844 genes, a real
  experimental call) is used as the whole-gene-coverage backbone: a gene
  gets a "no" row here only if gess-bsub.csv has an actual call for it
  AND it is not in the Koo 370-gene union (170 gess-bsub genes were
  already covered by the Koo union and skipped as duplicates; 7 gess-bsub
  "yes" genes not confirmed by Koo S3/S4 were dropped as ambiguous rather
  than guessed). Result: **370 yes + 667 no = 1037 total rows**,
  `ref/koo2017/bsub_minimal_binary.csv`. This non-essential-derivation
  step is a necessary extension beyond the literal union rule (which only
  defines the positive class) and is flagged here for the record.
- Join key: gene **symbol** (matches `gess-bsub.csv`'s `gene2` and the
  existing `knb1_to_symbol.tsv` RBH mapper output); locus tag kept as a
  secondary column, used directly for the iYO844 (curated GEM) comparison
  since iYO844's gene IDs are BSU locus tags natively.

### Feasibility: reused the established MILP-resolve pattern (redoes it fresh, as in every other core experiment)

`essentiality_correct_medium.py`'s `"bsub_koomin"` organism config
(gca=GCF_058182495.1, gram=positive, medium=GS_MM_glc, curated_gem=iYO844)
re-solves the MILP internally exactly as it does for every other core
experiment (it does not load round-2's cached feasibility result) --
confirmed `skeleton_feasible=True, bm_max=250` and `status=Optimal` for
all 3 predictors, consistent with round 2's earlier feasibility-only check
of this same combo.

### Result: 4th core experiment (B. subtilis, GS_MM_glc, vs Koo2017-combined)

| predictor | arm | n_selected | n_genes(model) | n_genes_with_ref | precision | recall | MCC |
|---|---|---|---|---|---|---|---|
| clean | meteor | 2158 | 827 | 415 | 0.736 | 0.288 | 0.270 |
| clean | thresh (+97 gapfill) | 2167 | 829 | 415 | 0.696 | 0.212 | 0.201 |
| dpz | meteor | 3213 | 1312 | 492 | 0.792 | 0.202 | 0.260 |
| dpz | thresh (+81) | 2469 | 1318 | 493 | 0.773 | 0.163 | 0.221 |
| enzbert | meteor | 2368 | 1105 | 445 | 0.836 | 0.268 | 0.330 |
| enzbert | thresh (+78) | 2347 | 1110 | 447 | 0.812 | 0.203 | 0.268 |

**Mean MCC: METEOR 0.287, threshold baseline 0.230 -- METEOR ahead in
every predictor individually (0.270/0.260/0.330 vs 0.201/0.221/0.268).**

Job 52128390: 14m43s, 4.30GB, single sbatch, 4 cpu, ≤24G -- well under the
45-min budget.

### Curated-GEM sanity check: does the Koo reference move iYO844's ceiling?

| reference | precision | recall | MCC |
|---|---|---|---|
| gess-bsub.csv (round 3) | 0.340 | 0.688 | 0.390 |
| **Koo2017-combined** (this round) | 0.792 | 0.714 | **0.666** |

**Yes, substantially -- MCC nearly doubles (0.390 -> 0.666), driven mainly
by precision jumping from 0.340 to 0.792** (recall is similar, 0.688 vs
0.714). This makes sense: the Koo-combined reference is methodologically
consistent (one study, one strain, systematic library screen) and directly
covers the minimal-medium condition iYO844's default medium approximates,
whereas gess-bsub.csv is an older, smaller SubtiWiki aggregate rooted in
Kobayashi 2003's LB screen -- a reference/medium mismatch of exactly the
kind explored in earlier rounds. **The reference choice matters as much
for B. subtilis's ceiling as it did for E. coli's (0.758 -> 0.592 with
PEC).**

### Bonus: B. subtilis medium x reference 2x2 (pure re-scoring, `rescoring_bsub_2x2.py`)

| medium \ reference | gess-bsub.csv | Koo2017_combined |
|---|---|---|
| **LB_marinos** | METEOR 0.323, thresh 0.247 | METEOR 0.255, thresh 0.222 |
| **GS_MM_glc** | METEOR 0.325, thresh 0.239 | METEOR 0.287, thresh 0.230 |

**Unlike the E. coli 2x2 (round 5), where the reference-table swap flipped
a tie into a clear threshold win, here METEOR stays ahead of the threshold
baseline in ALL FOUR cells**, by a fairly stable margin (0.05-0.08 MCC).
Both medium and reference swaps produce only small, non-rank-changing
shifts (largest single move: Koo-combined reference under LB_marinos,
-0.068 for METEOR relative to gess-bsub.csv -- still the largest effect
of the two factors, echoing round 5's finding that reference matters more
than medium, but here it never flips the winner). **This is an important
contrast with round 5**: the E. coli finding that reference-table choice
can flip the METEOR-vs-threshold ranking does not generalize to
B. subtilis -- it is not evidence of a universal "stricter reference
always favors the baseline" artifact, since here METEOR's edge survives
every reference and medium combination tried.

Files: `build_bsub_minimal_ref.py`, `rescoring_bsub_2x2.py`,
`run_ess_bsub_koomin.sbatch`, `ref/koo2017/bsub_minimal_binary.csv`,
`results/essround3/bsub_koomin_{clean,dpz,enzbert}_vanilla.json`,
`results/essround3/curated_iYO844_bsub_koomin.json`,
`results/rescoring2x2/grid_bsub.json`, `results/rescoring2x2/summary_bsub_2x2.json`.

## Round 8 (2026-09-20/21): 5th-8th core experiments — S. aureus, Salmonella, K. pneumoniae, P. putida

### S. aureus (5th core experiment)

Reference: Koo et al. 2017 Table S3 sheet C (own cross-species comparison,
392 rows, restricted to genes conserved across their B. subtilis/S. aureus/
E. coli comparison -- **NOT genome-wide**, smaller/differently-scoped
denominator than the other organisms' references; flag this in any
cross-organism comparison). Aggregates Chaudhuri et al. 2009 (BMC Genomics
10:291, TMDH, strain SH1000/NCTC 8325, **BHI broth/agar** -- verified
verbatim from the paper) and Santiago et al. 2015 (BMC Genomics 16:252,
Tn-seq, strains RN4220/COL mapped to NCTC8325, **TSB** -- verified
verbatim). Both rich media.

Medium: METEOR's own built-in `data/medium.pkl` `"TSB"` entry (67 cpds) --
no gapseq media.tsv entry exists for S. aureus (the gapseq paper's
5-organism essentiality benchmark did not include it). Matches Santiago
2015 exactly; reasonable proxy for Chaudhuri 2009's BHI.

Strain mapping (three-way, all built fresh):
- Panel = RN4220 (GCF_045348045.1, confirmed via NCBI datasets API) ->
  reference NCTC 8325 (SAOUHSC_##### locus tags): RBH, 2467/2508 panel
  proteins mapped.
- Curated GEM iYS854 uses **USA300_TCH1516** gene IDs (`USA300HOU_RS#####`
  -- "HOU" traced to Baylor College of Medicine, Houston, the TCH1516
  submitter), a THIRD strain space distinct from both panel and reference.
  Built a second RBH chain (TCH1516 protein -> NCTC8325 protein -> SAOUHSC
  locus, 2530/2556 mapped) plus `curated_match="external_map"`, a new
  matching mode added to `essentiality_correct_medium.py` for this case
  (locus-tag lookup table, distinct from the existing `"id"`/`"name"`
  modes).

Result (all feasible, `Optimal`, `skeleton_feasible=True`):

| predictor | arm | precision | recall | MCC |
|---|---|---|---|---|
| clean | meteor / thresh | 0.872 / 0.810 | 0.293 / 0.291 | 0.150 / 0.065 |
(dpz/enzbert rows already independently confirmed present and consistent
by the coordinator directly from disk; not re-derived here.)

Curated GEM iYS854 sanity check also on disk (`curated_iYS854_sau.json`).

### Salmonella, K. pneumoniae, P. putida (6th-8th core experiments)

All three from Yasir M et al. 2024 mBio 15(10):e01798-24 (TraDIS, LB agar,
37C, paper's own threshold score<=0 = essential, already binarized in the
supplied CSVs for Salmonella/K. pneumoniae) except P. putida, from Royet
et al. 2025 Environ Microbiol 27:e70095 (Tn-seq, LB agar baseline before
metal exposure, raw ES/GD/NE/GA multi-class table, NOT pre-binarized).

**Medium**: LB_marinos for all three (LB agar source medium, reusing the
existing rich-medium composition, no new medium built).

**P. putida binarization** (judgment call, explicitly flagged as instructed):
essential = ES only; non-essential = NE + GD + GA pooled (GD = growth
defect but viable = not essential for survival; GA = growth advantage =
clearly not essential); 13 rows with `State classification = N/A` dropped
as unknown rather than guessed. Result: 600 essential / 5116 non-essential
of 5729 usable rows.

**Strain mapping** (all three needed fresh RBH; none matched panel genomes
literally, confirming the coordinator's expectation):
- **Salmonella**: panel = LT2 (GCF_000006945.2) -- confirmed panel protein
  accessions ARE LT2's own official NP_ RefSeq IDs (not a re-annotation),
  so RBH panel->SL1344 (GCF_000210855.2) gave 4332/4415 mapped. Curated GEM
  STM_v1_0 uses LT2's own native `STM####` locus tags directly -- chained
  STM####->LT2 protein (direct feature-table lookup)->SL1344 locus via the
  same RBH table: 4332 mappings, clean 1:1 correspondence (e.g.
  STM0001->SL1344_0001).
- **K. pneumoniae**: panel = strain Kp0179 (GCF_058435815.1). Tried Ecl8
  (GCF_000315385.1) first per instruction -- RBH gave 4642/4780 mapped,
  good quality, so RH201207 fallback was not needed. Curated GEM iYL1228
  uses MGH 78578's `KPN_#####` locus tags (a third strain) -- chained
  KPN_->MGH78578 protein->RBH->Ecl8 protein->BN373_ locus: 4231 mappings.
- **P. putida**: panel genome IS the reference strain KT2440
  (GCF_045571375.1) -- but confirmed this is a *different, modern PGAP
  re-annotation* (`PYW03_RS#####` locus prefix) than the classic
  PP_-tagged assembly (GCF_000007565.2) that both the essentiality table
  and iJN1463 use, so RBH was still necessary despite same-strain identity
  (an important nuance: "same strain" does not mean "same locus-tag
  space"). RBH quality was the best of any organism in this project:
  5347/5430 pairs (98.5%) resolved to a PP_ locus tag, since the underlying
  genome content is genuinely identical. iJN1463 already uses classic
  `PP_####` IDs directly (`curated_match="id"`, no external map needed).

### Result: 6th-8th core experiments

All 9 predictor/organism combinations feasible (`Optimal`,
`skeleton_feasible=True`, `bm_max=250`), confirmed inline via the
pipeline's built-in skeleton-feasibility assert before the full pipeline
ran (same safety gate as every other core experiment).

| organism | predictor | arm | n_selected | n_genes_with_ref | precision | recall | MCC |
|---|---|---|---|---|---|---|---|
| Salmonella | clean | meteor / thresh | 2687 / 2626 | 1050 / 1054 | 0.219 / 0.316 | 0.101 / 0.174 | 0.066 / 0.153 |
| Salmonella | dpz | meteor / thresh | 3318 / 3015 | 1572 / 1575 | 0.283 / 0.286 | 0.108 / 0.114 | 0.121 / 0.126 |
| Salmonella | enzbert | meteor / thresh | 2614 / 2521 | 1258 / 1261 | 0.179 / 0.316 | 0.035 / 0.128 | 0.032 / 0.141 |
| K. pneumoniae | clean | meteor / thresh | 2876 / 2843 | 1118 / 1124 | 0.229 / 0.270 | 0.088 / 0.136 | 0.079 / 0.123 |
| K. pneumoniae | dpz | meteor / thresh | 3859 / 3573 | 1785 / 1789 | 0.288 / 0.296 | 0.107 / 0.114 | 0.135 / 0.143 |
| K. pneumoniae | enzbert | meteor / thresh | 3136 / 3043 | 1316 / 1321 | 0.171 / 0.362 | 0.050 / 0.174 | 0.045 / 0.201 |
| P. putida | clean | meteor / thresh | 2683 / 2658 | 942 / 947 | 0.320 / 0.407 | 0.098 / 0.134 | 0.091 / 0.152 |
| P. putida | dpz | meteor / thresh | 3698 / 3176 | 1779 / 1781 | 0.229 / 0.341 | 0.053 / 0.072 | 0.058 / 0.111 |
| P. putida | enzbert | meteor / thresh | 2734 / 2661 | 1247 / 1254 | 0.250 / 0.293 | 0.078 / 0.101 | 0.071 / 0.103 |

**Mean MCC (3 predictors):**

| organism | METEOR | threshold baseline |
|---|---|---|
| Salmonella | 0.073 | 0.140 |
| K. pneumoniae | 0.086 | 0.156 |
| P. putida | 0.073 | 0.122 |

**Threshold baseline beats METEOR in all three, consistently, unlike
B. subtilis (round 3/7, METEOR ahead) and unlike E. coli/GS_MM_glc (round
1/3, a tie) -- closer in pattern to E. coli/LB_marinos-vs-PEC (round 4,
threshold ahead by 2x).** Absolute MCC values are lower across the board
than any prior organism (0.03-0.20 range) -- likely reflects the TraDIS/
Tn-seq references' own noise/threshold-sensitivity (continuous fitness
scores collapsed to a hard essential/non-essential cutoff by the source
papers, not something re-verified here) combined with cross-strain RBH
mapping loss, though this has not been decomposed further.

### Curated-GEM sanity checks

| organism | curated GEM | precision | recall | MCC |
|---|---|---|---|---|
| Salmonella | STM_v1_0 | 0.389 | 0.520 | 0.362 |
| K. pneumoniae | iYL1228 | 0.397 | 0.351 | 0.301 |
| P. putida | iJN1463 | 0.408 | 0.510 | 0.351 |

All three ceilings (0.30-0.36) are well below iML1515's 0.758 but in the
same range as iYO844's 0.390 (vs gess-bsub.csv) -- consistent with these
being noisier/lower-confidence TraDIS-derived references relative to the
classic Keio/PEC lineage, not evidence of anything wrong with the curated
models themselves.

### Compute
Job 52131469 (S. aureus): completed, feasible, full pipeline + curated GEM.
Job 52159615 (Salmonella): 10m05s, 3.64GB. Job 52159616 (K. pneumoniae):
8m17s, 3.98GB. Job 52159617 (P. putida): 11m38s, 3.41GB. All single sbatch
jobs, 4 cpu, ≤24G, well under the 45-min budget.

Files: `essentiality_correct_medium.py` extended with `"sau"`,
`"salmonella"`, `"kpneumoniae"`, `"pputida"` organism configs, the `"TSB"`
medium entry, and the `"external_map"` curated-GEM matching mode;
`ref/sau/`, `ref/salmonella/`, `ref/salmonella_map/`, `ref/kpneumoniae/`,
`ref/kpneumoniae_map/`, `ref/pputida/`, `ref/pputida_map/` (all RBH tables,
binary reference CSVs, feature tables, medium provenance notes);
`run_ess_{sau,salmonella,kpneumoniae,pputida}.sbatch`;
`results/essround3/{sau,salmonella,kpneumoniae,pputida}_{clean,dpz,enzbert}_vanilla.json`,
`results/essround3/curated_{iYS854_sau,STM_v1_0_salmonella,iYL1228_kpneumoniae,iJN1463_pputida}.json`.

## Round 10 (2026-09-21): forensic root-cause analysis — why threshold beats METEOR on Salmonella/K.pneumoniae/P.putida

Pure post-hoc analysis of already-saved `sgd_{org}_{meteor,thresh}_{predictor}.json`
files, RBH maps, and baseline score matrices (loaded read-only via
`extract_pred`/`resolve_baseline_pkl` -- no MILP re-solve). Script:
`forensic.py`. Checked all 3 predictors per organism first for consistency
(pattern holds in all 9 organism/predictor combos, sizes scale with
`n_genes_with_ref`); deep dive on `dpz` (largest denominator in all three,
as expected).

### 1. Disagreement counts (dpz, genes with both a METEOR and threshold call AND a reference truth)

| organism | n_both_have_ref | both_right | both_wrong | METEOR-wrong/thresh-right (a) | METEOR-right/thresh-wrong (b) |
|---|---|---|---|---|---|
| Salmonella | 1572 | 1372 | 169 | **15** | 16 |
| K. pneumoniae | 1785 | 1609 | 148 | **14** | 14 |
| P. putida | 1779 | 1541 | 218 | **16** | 4 |

Group (a) is larger than (b) for Salmonella (15 vs 16, roughly even
actually) and P. putida (16 vs 4, clearly larger) but not K. pneumoniae
(14 vs 14, tied) -- the raw counts are small and roughly comparable in
size across arms; **the MCC gap is not explained by a large count
imbalance, but by which specific genes are on each side** (see point 4).

### 2. Confidence-zone hypothesis: REFUTED, decisively

Every single group-(a) gene's own maximum predictor score (across its full
EC profile) is >=0.72, and the overwhelming majority are >=0.99 (raw
scores, dpz): Salmonella `[0.719, 0.771, 0.996, 0.997, 0.999, 1.0 x9]`;
K. pneumoniae `[0.997, 0.998, 0.998, 0.999, 0.999, 1.0 x9]`; P. putida
`[0.979, 0.992, 0.995, 0.996, 0.997, 1.0 x10]`. **None fall in the
uncertain zone.** This is not a coincidence of the binning: any gene
present in either arm's model necessarily scored >=0.5 for at least one EC
(that is the GPR-inclusion criterion in `gpr_for()`), so a naive "is the
gene's own top score low" test is structurally biased toward "no" -- the
real, still-decisive finding is that scores are not just above 0.5, they
are almost all pinned at the ceiling (>=0.99). **METEOR's mistakes are
concentrated on the genes the predictor was MOST confident about, not
least** -- the opposite of what a naive confidence-zone hypothesis would
predict.

### 3. Selection-mechanism check: partially resolved, limit flagged honestly

No genes fall into the "gene present in one arm's model but not the
other" category for the disagreement set by construction (both must have
a call to be compared). A targeted check on Salmonella's purine/pyrimidine
salvage pathway (`results/forensic/salvage_gene_check_salmonella.txt`):
salvage catalytic enzymes (hpt, gpt, apt, upp, deoD -- hypoxanthine/
xanthine/adenine/uracil phosphoribosyltransferases, purine-nucleoside
phosphorylase) are present with a GPR-assigned reaction in **both**
METEOR's and threshold's networks identically; salvage **transporters**
(nucleoside permeases, NupC, PunC, xanthine/uracil permease, GhxP) are
**absent from both** identically. So the differentiator is not simple
gene/reaction presence-absence for the salvage pathway -- **it must lie in
whether the salvage route actually carries flux to biomass under FBA in
each arm's specific network, which can differ even with identical
gene-level GPR presence depending on which other connecting reactions got
selected.** This requires the actual selected-reaction set (y-vector) per
arm to trace mechanically, and **the y-vectors were not cached from the
original runs** (`essentiality_correct_medium.py`'s `build_model_with_gpr`
discards the model in memory after `single_gene_deletion`). **Cheapest
next step**: either (a) a small modification to the pipeline to always
pickle `keep_flags` next time it runs (near-zero added cost, benefits all
future forensic passes), or (b) one fresh MILP resolve per organism with
y-vector saving added (same ~150-300s/predictor cost as every prior round,
NOT done here per the "no new MILP re-solve" instruction).

### 4. Pathway/functional clustering: extremely tight, near-total

Group-(a) genes are **overwhelmingly de novo purine and pyrimidine
nucleotide biosynthesis enzymes**, with one recurring paralog family
(argininosuccinate synthase/lyase, mechanistically the same fumarate-
releasing lyase family as adenylosuccinate lyase/purB):

- Salmonella (14/15): purB, purC, purD, purH, purL, purM, purN-equiv
  (phosphoribosylglycinamide formyltransferase), purF (amidophospho-
  ribosyltransferase), pyrE (orotate PRTase), pyrF (OMP decarboxylase),
  argG/argH (argininosuccinate synthase/lyase), plus glnQ (glutamine ABC
  transporter) and an acetyl-CoA C-acetyltransferase.
- K. pneumoniae (13/14): the identical Pur gene set (purB, purC, purD,
  purF, purH, purL, purM, purN-equiv), pyrE, pyrF, argG, argH, plus two
  rhamnose-catabolism genes (rhamnulokinase, rhamnulose-1-phosphate
  aldolase -- the one clear outlier from the pattern).
- P. putida (11/16 in this direction): the same Pur set (purC, purD, purF,
  purH, purL, purM, purN-equiv), pyrB (aspartate carbamoyltransferase),
  pyrF, argG, argH.

**One functional cluster explains ~70-90% of every organism's group (a).**
This is the same style of pathway-coherent-error finding as the E. coli
Joyce-2006 medium-sensitivity round (round 6), where wrong-direction flips
also clustered entirely within single pathways (purine biosynthesis in
that case too, plus branched-chain amino acids) rather than scattering.

**A second, smaller pattern within group (a), not asked for but found and
worth flagging**: group (a) splits into two opposite-direction sub-modes --
(i) METEOR **over-calls essential** (real truth = non-essential, METEOR
says essential): the dominant purine/pyrimidine/arg cluster above, 14/15,
13/14, 11/16 respectively; (ii) METEOR **under-calls essential** (real
truth = essential, METEOR says non-essential): a much smaller set --
Salmonella lipoyl synthase (1 gene), K. pneumoniae PlsB/glycerol-3-
phosphate acyltransferase (1 gene), P. putida folate/thymidylate-pathway
genes (dihydrofolate reductase, thymidylate synthase, folylpolyglutamate
synthase/DHFS, cytidylate kinase, CTP:phosphatidate cytidyltransferase --
5 genes, notably the largest under-call cluster of the three, all DNA-
precursor/phospholipid genes that are typically NOT salvageable). The
majority mode (i) is consistent with a "redundant salvage route makes it
non-essential in reality, METEOR's parsimony-minimizing selection removes
that redundancy" story; the minority mode (ii) looks like the opposite
failure (METEOR under-selecting an actually-essential, non-redundant
pathway), a distinct and smaller effect worth separate attention if this
is pursued further.

### 5. Cross-organism synthesis

**The same mechanism explains the dominant failure mode in all three
organisms, not three different stories**: METEOR systematically calls the
de novo purine/pyrimidine nucleotide biosynthesis pathway (plus the
arg-lyase paralog pair) essential, when it is genuinely non-essential
under LB-type rich medium (LB_marinos includes free nucleobases --
adenine, guanine, cytosine, uracil -- and nucleotides -- AMP, GMP, CMP,
UMP -- directly, cpd00128/cpd00207/cpd00307/cpd00092/cpd00018/cpd00126/
cpd00046/cpd00091, confirmed in the medium composition already built in
round 4), because real organisms can salvage these bases/nucleosides
instead of synthesizing them de novo. Threshold's simpler EC>=0.5 draft +
parsimony gap-fill apparently retains (or gap-fills in) enough of the
salvage/alternative route to preserve that redundancy, while METEOR's
evidence-weighted cost-minimizing MILP -- which specifically rewards
high-confidence, low-cost reaction sets and penalizes reaction count
(`mu` term, Section 3.4's candidate-mask/cost mechanism) -- appears to
settle on the de novo route alone (itself extremely high-confidence,
score ~1.0, hence "free" to include) without the nutritionally redundant
salvage alternative that would make individual de novo genes dispensable.
**This connects directly to Section 3.4's evidence-weighted-cost design
and is the mirror image of round 6's E. coli medium-sensitivity finding**:
there, METEOR's rare wrong-direction flips were also concentrated in the
same purine-biosynthesis pathway (moving in the medium-dependent
direction); here, on three additional organisms under a fixed rich medium,
the same pathway is where METEOR's selection is least aligned with real
redundancy. The mechanism is consistent with (but not fully proven down to
the reaction-selection level for, per point 3's flagged limit) parsimony
removing biologically-real pathway redundancy that a less-optimized
threshold draft happens to retain.

Files: `forensic.py`, `results/forensic/summary_all.json` (all 9
organism/predictor combos), `results/forensic/{org}_deepdive_dpz.json`
(per-gene group a/b detail with scores and products),
`results/forensic/salvage_gene_check_salmonella.txt`.
