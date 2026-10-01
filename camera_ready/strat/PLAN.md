# Plan: R1 recall stratification, integrated with R2 skeleton ablation
Status: approved 2026-09-16 (do it). No code yet. Owner: Kexin. Target: numbers by 09-18, text by 09-20.

## 0. Question and hypotheses
R1: "predictor vocabulary / recall caps what METEOR can recover; unannotated proteins contribute no evidence."
R2: "how much comes from the shared skeleton rather than evidence weighting?"
Both are the same question at EC level. Design: stratify curated-EC recall by (i) predictor vocabulary,
(ii) baseline confidence bin, and (iii) skeleton reachability, for four arms including skeleton-only.

H1  Out-of-vocab curated ECs: recall(METEOR) ~= recall(skeleton-only) >> recall(baseline)=0.
    -> that recovery is 100% scaffold. Quantifies R2 at EC level, answers R1 directly.
H2  In-vocab curated ECs: recall(METEOR) - recall(skeleton-only) = evidence contribution,
    concentrated in bins (0,0.1) and [0.1,0.5). [0.5,1] saturated for both METEOR and baseline.
H3  Curated ECs mapping only to non-skeleton universal reactions with score < 0.01 are unreachable by
    construction (not candidates); expect a small count -> hard ceiling, report it.
Whatever the outcome, Sec 4.1 sentence "METEOR cannot reach ECs outside the predictor's vocabulary" must
be reworded: the predictor cannot; the scaffold can and does (demo: 68-73% of them).

## 1. Inputs (all on disk, read-only)
| item | path (ibex) |
|---|---|
| curated EC sets, GEM->GCF map, loaders | psb_revision/eval/diag_recall_denominators.py, feasibility_r1_strat/strat_demo.py (GEM2G) |
| baseline score matrices, 3 predictors vanilla | paperA_2026/baseline_preds/{clean,dpz,enzbert}_vanilla_genome_collection/{gcf}_*.pkl |
| METEOR outputs, 3 predictors vanilla | meteor_v8_evw_p2mu3_run/meteor_out/{pred}_vanilla/meteor_{sol,preds}_{gcf}.pkl |
| skeleton-only y*, 6 GCF (T4, DPZ flags) | psb_revision/results/skeleton_abl/{gcf}_skelonly_y.pkl (+ _full_y.pkl as control) |
| Gram-group skeletons (15,580 rxns) | recomputed by skeleton_ablation.py; reuse its function, or dump once to results/ |
| universal, seedr2ec, all_ancestors | psb_revision/code_snapshot/data |
Organisms: iML1515/E.coli GCF_058436375.1 (-), STM_v1_0/Salmonella GCF_000006945.2 (-),
iYL1228/K.pneumoniae GCF_058435815.1 (-), iJN1463/P.putida GCF_045571375.1 (-),
iYS854/S.aureus GCF_045348045.1 (+), iYO844/B.subtilis GCF_058182495.1 (+).

## 2. Definitions (freeze before coding; write into Supp S7.6 verbatim)
Reference R  = 4-digit ECs of the curated GEM (gem_ec; E.coli 865). Same set as Table 5 / S7.4.
Vocab V_p    = 4-digit ECs in predictor p's raw score-matrix columns (CLEAN 5235, DPZ 2829, EnzBERT 4746).
Conf bin     = max-over-proteome baseline score s(e) of curated EC e: {0 (incl. e not in V_p), (0,0.1), [0.1,0.5), [0.5,1]}.
Reachability = e maps via seedr2ec to >=1 reaction in {skeleton of its Gram group} -> "skeleton";
               else to >=1 universal reaction -> "universal-only"; else "not in SEED".
Arms (output EC set O):
  B      baseline threshold: {e : s(e) >= 0.5}                                 (S7.4 draft convention)
  M-act  METEOR active_ecs (preds pkl)                       = Table 5 convention (0.884)
  M-rxn  seedr2ec over METEOR selected reactions (sol pkl)   = S7.4 convention (0.784), strict subset of M-act
  S      seedr2ec over skeleton-only selected reactions (T4 skelonly y*)
  (F     seedr2ec over T4 full re-solve y*, control that F ~= M-rxn)
Recall in stratum X = |O ∩ R ∩ X| / |R ∩ X|. Report n = |R ∩ X| next to every recall.
Dark proteome: fraction of proteins with max_e score < t, t in {0.5 (baseline cutoff), 0.01 (mask cutoff:
  contributes nothing to any w_j)}. Per predictor, threshold stated; never compared across predictors.
Caveat to write: M-act ⊇ B by construction, so in bin [0.5,1] M-act >= B trivially; the honest
  evidence-vs-baseline comparison in-vocab is M-rxn vs B and M-rxn vs S.

## 3. Outputs
results/strat/{gem}_{pred}.json  per organism x predictor: for each arm x stratum: n, hit, recall;
                                  reachability counts; dark fractions; vocab size; sanity fields.
results/strat_summary.json        mean ± sd over 6 organisms, per predictor, per arm, per stratum;
                                  Wilcoxon (6 pairs): M-rxn vs B in-vocab; M-rxn vs S out-of-vocab; M-rxn vs S in-vocab.
Sanity gates (fail = stop):  M-act recall_all(dpz) per organism == recall_denominators.json table3_R (E.coli 0.905);
  M-rxn recall_all(dpz) == samepred_R (0.816); |in|+|out| == gem_ec; sum of bins == gem_ec;
  S n_selected ≈ 1,985 ± 30; F recall_all within 0.01 of M-rxn.

## 4. Execution
P1  Script strat_all.py from strat_demo.py: add arms S/F/M-rxn, reachability stratum, t=0.01 dark,
    per-org JSON + aggregate + Wilcoxon. Reuse skeleton_ablation.py's skeleton builder. ~2 h.
    Test on iML1515 on login node (~4 min) -> gates pass.
P2  sbatch: 1 job, 1 cpu, 8 GB, 90 min, nice 5000, loops 6 GEM x 3 pred (~18 x 3.5 min ≈ 65 min). Poll.
P2b OPTIONAL (decide after P2): S arm was solved with DPZ cost flags. If S recall differs between
    predictors' in-vocab strata in a way that matters, re-solve skelonly for clean/enzbert on 6 GCF:
    12 jobs x ~11 min via skeleton_ablation.py --arm skelonly (needs a --baseline flag; check first). Else footnote.
P3  Copy JSONs to laptop plan/ibex/results/strat/; add change-log lines C37..; write numbers into tex.
P4  Text (below). P5 length pass, tectonic compile, lint, push to Overleaf, update niu_response.tex.

## 5. Paper placement (12-page budget: body ends p.13, +~6 lines main text -> cut ~6 lines elsewhere)
Main text
  3.5 after the same-predictor paragraph (L~725): 3-4 sentences. "Stratified by vocabulary, X% of
      curated ECs lie outside each predictor's vocabulary; the threshold baseline recovers none, METEOR
      recovers a-b% of them, and so does the skeleton-only solve, so this recovery is the scaffold's
      (Section 3.4). Within vocabulary METEOR exceeds the baseline by ... , concentrated in ECs scored
      below 0.1 (Supplementary Table S8)."
  3.4 ablation paragraph (L642-647): one clause "…and supplies the curated ECs outside the predictor's
      vocabulary (Section 3.5)".
  4.1 (L752-756): replace "Stratifying ... we have not done so" with the measured ceiling: predictor
      recall bounds evidence-driven recovery; scaffold supplies out-of-vocab ECs; H3 count is the hard
      ceiling; dark-proteome fraction with threshold.
  Cuts to fund it: Table 5 caption s.d. sentence; last paragraph of 3.5 ("provides no evidence...")
      merge into preceding; 4.1 differentiable-selection sentence shorten.
Supplement
  New S7.6 "Recall stratified by predictor vocabulary, confidence and skeleton reachability":
  Table S8: rows = predictor x arm (B, M-rxn, M-act, S); cols = all / in-vocab / out-of-vocab /
            bins 0, (0,0.1), [0.1,0.5), [0.5,1]; cells = recall mean ± sd (n in header row per predictor).
  Table S9: per-organism recall_all and out-of-vocab recall for M-rxn and S (6 rows x 3 pred).
  Table S10 (small): dark-proteome fraction at t=0.5 and 0.01, 6 organisms x 3 predictors, mean ± sd.
  Cross-ref from S7.4 and the candidate-mask ablation paragraph.
Response letter
  Move "stratification not measured" out of "Not addressed"; add to §1 (skeleton, R2) one sentence with
  the out-of-vocab numbers and to §6 (R1) the in-vocab/bin numbers; keep dark proteome as reported proxy.

## 6. Risks
- E. coli is best case; S. aureus (Table 5 R=0.836) may show out-of-vocab recall well below 0.7. Report as is.
- If H1 fails (M-rxn out-of-vocab recall >> S), then evidence-weighted cost pulls in skeleton reactions
  the skeleton-only solve drops; that is a real METEOR effect and a stronger story, but the 3.4 text
  ("scaffold supplies…") would need softening. Decide wording after numbers.
- Do not present dark-proteome numbers as predictor comparison; calibration differs (DPZ 3% <0.1 vs CLEAN 69%).
- Not touching Table 5 or the 0.884/0.784 numbers.
