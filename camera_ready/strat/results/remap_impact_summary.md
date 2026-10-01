# Remap impact on published curated-model comparisons (dpz vanilla, 6 organisms)

Protocol: none = published pipeline (extract_pred remaps only predictor columns absent from all_ancestors.txt); pkl = every EC set (curated GEM ECs, CarveMe ECs, active_ecs, rxn-mapped ECs, gap-fill baseline ECs, predictor scores, reference vocabulary) canonicalised old->new with the shipped code_snapshot/data/enzymeobsolete.pkl (1,309 transfers), partial ECs dropped. This is the canonical, and only, EC-version dictionary used anywhere in this project. Flag: |delta vs none| > 0.01 (ratios) or > 10 (counts).

## Table 5 (tab:carveme) -- mean over 6

| row | metric | published | none | pkl |
|---|---|---|---|---|
| meteor_full | n | 1521 | 1521 | 1458 |
| meteor_full | P | 0.438 | 0.438 | 0.453 |
| meteor_full | R | 0.884 | 0.884 | 0.890 |
| meteor_full | F1 | 0.585 | 0.585 | 0.600 |
| meteor_full | J | 0.415 | 0.415 | 0.430 |
| meteor_refvocab | n | 876 | 876 | 867 |
| meteor_refvocab | P | 0.761 | 0.761 | 0.762 |
| meteor_refvocab | R | 0.884 | 0.884 | 0.890 |
| meteor_refvocab | F1 | 0.817 | 0.817 | 0.820 |
| meteor_refvocab | J | 0.694 | 0.694 | 0.699 |
| carveme | n | 818 | 818 | 804 |
| carveme | P | 0.822 | 0.822 | 0.823 |
| carveme | R | 0.893 | 0.893 | 0.893 |
| carveme | F1 | 0.854 | 0.854 | 0.855 |
| carveme | J | 0.746 | 0.746 | 0.748 |
| reference vocabulary size | | 1,234 | 1234 | 1209 |
| METEOR FP outside ref vocab (frac) | | 0.76 | 0.760 | 0.747 |

## S7.4 same-predictor: METEOR (rxn-mapped) vs threshold + gap-fill

| metric | published | none | pkl |
|---|---|---|---|
| meteor_R | 0.784 | 0.784 | 0.819 |
| baseline_R | 0.767 | 0.767 | 0.796 |
| meteor_P | | 0.422 | 0.435 |
| baseline_P | | 0.455 | 0.465 |
| diff R, direction, p | +0.017, 6/6, p=.031 | +0.017, 6/6, p=.0312 | +0.023, 6/6, p=.0312 |

### Per-organism S7.4 (none -> pkl)

| organism | meteor R none | meteor R pkl | baseline R none | baseline R pkl |
|---|---|---|---|---|
| E.coli | 0.816 | 0.851 | 0.800 | 0.832 |
| Salmonella | 0.812 | 0.844 | 0.800 | 0.830 |
| K.pneumoniae | 0.823 | 0.855 | 0.807 | 0.836 |
| P.putida | 0.772 | 0.811 | 0.752 | 0.782 |
| S.aureus | 0.706 | 0.744 | 0.683 | 0.711 |
| B.subtilis | 0.775 | 0.811 | 0.758 | 0.785 |

## S7.5 sub-threshold recovery (size-matched top-K) and marginal precision

| metric | published | none | pkl |
|---|---|---|---|
| recall_W_meteor | 0.606 | 0.606 | 0.603 |
| recall_W_topK | 0.483 | 0.483 | 0.495 |
| precision_meteor | 0.422 | 0.422 | 0.435 |
| precision_topK | 0.385 | 0.385 | 0.398 |

## S7.5 marginal precision

extra = sum over the six models of (|METEOR rxn-mapped EC set| - |threshold+gap-fill baseline EC set|) (dpz vanilla, 4-digit ECs, all_ancestors-restricted seedr2ec mapping, no vocabulary restriction); correct = sum of (|M & G| - |B & G|); curated sub-threshold = sum of (|M & W| - |B & W|), W = curated ECs with 0 < max DPZ score < 0.5; marginal precision = correct/extra. The 76% is NOT a property of the extra ECs: it is Table 5's mean fraction of METEOR (active_ecs, full vocabulary) false positives lying outside the 1,234-EC reference vocabulary, |(A-G)-V_ref|/|A-G| averaged over six organisms (toolcompare_ec_vocab.py meteor_fp.frac_outside = 0.760).

| metric | published | none | pkl |
|---|---|---|---|
| extra | 773 | 773 | 773 |
| correct | 77 | 77 | 100 |
| correct_subthreshold | 75 | 75 | 92 |
| marginal_precision | 0.100 | 0.100 | 0.129 |

## Table 2 (tab:subthreshold) -- seed_restricted convention, mean +/- sd over 6

| metric | published | none | pkl |
|---|---|---|---|
| pooled \|W\| | 612 | 610 | 639 |
| recallW_meteor | 0.604 +/- 0.053 | 0.606 +/- 0.052 | 0.604 +/- 0.049 |
| recallW_baseline | 0.480 +/- 0.039 | 0.481 +/- 0.040 | 0.460 +/- 0.033 |
| precision_meteor | 0.422 +/- 0.057 | 0.422 +/- 0.057 | 0.434 +/- 0.057 |
| precision_baseline | 0.455 +/- 0.061 | 0.455 +/- 0.061 | 0.465 +/- 0.062 |

## recall_denominators (per organism: table3_R = M-act, samepred_R = M-rxn)

| organism | table3_R none | table3_R pkl | samepred_R none | samepred_R pkl |
|---|---|---|---|---|
| E.coli | 0.905 | 0.912 | 0.816 | 0.851 |
| Salmonella | 0.898 | 0.905 | 0.812 | 0.844 |
| K.pneumoniae | 0.909 | 0.915 | 0.823 | 0.855 |
| P.putida | 0.869 | 0.874 | 0.772 | 0.811 |
| S.aureus | 0.836 | 0.841 | 0.706 | 0.744 |
| B.subtilis | 0.888 | 0.894 | 0.775 | 0.811 |
