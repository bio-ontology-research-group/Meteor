# Plan: fix the mass-imbalance metric (R3 #4) and add CarveMe/curated reference
Status: PROPOSED 2026-09-16, awaiting go. Must-do in my view: current text concedes a stoichiometry problem that does not exist.

## 0. Finding (feasibility_massimbal_baseline/, Salmonella GCF_000006945.2)
gen_table1.py::profile counts `find_mass_unbalanced_reactions(model.reactions)` over ALL reactions and divides by
all reactions. build_submodel adds one EX_ per extracellular metabolite (5,923 rxns for 3,294 selected -> 2,669 boundary).
Boundary reactions are one-sided, so ~all of them are "imbalanced" -> mi_frac ≈ boundary share ≈ 0.45-0.5, static.
| model | internal rxns | internal imbalanced | internal mi_frac | published mi_frac |
| METEOR dpz | 3,254 | 5 (5 missing formula, 0 genuine) | 0.0015 | 0.4515 |
| CarveMe 1.6.6 | 2,567 | 5 | 0.0019 | 0.143 |
| curated STM_v1_0 | 2,197 | 13 | 0.0059 | 0.142 |
| SEED universal | 34,161 | 12 | 0.0004 | 0.287 |
CarveMe MEMOTE test_reaction_mass_balance across all 108 panel models: mean 0.0015, max 0.0059; on disk already.

## 1. New definition (freeze)
Internal reaction = not EX_/DM_/SK_, not biomass. MI_int = # internal reactions failing memote elemental balance;
mi_frac_int = MI_int / # internal reactions. Split MI_int into "missing formula" vs "genuine imbalance" (memote gives both).
Dead-end definition unchanged. Keep the old boundary-inclusive number nowhere in the paper (it is a measurement artifact).
Report in Table 1 as COUNT (e.g. "5.1 ± 2.0") not fraction: 0.0015 vs 0.0016 is unreadable; supp gives fraction.

## 2. What has to be recomputed (only counts were stored; per-reaction lists were not)
| artifact | arms | source of selected sets | jobs |
| Table 1 MI row, 6 columns; Supp Fig S1 panel (c) | baseline x3, METEOR x3, 108 genomes | gen_table1.py rebuilds both from baseline pkl + meteor_sol pkl | array 108 (one genome = 6 profiles, ~5-8 min incl. universal load), 1 cpu 8G 30 min, nice 5000 |
| Supp Table S7 (candidate-mask ablation) MI column | full, skelonly | psb_revision/results/skeleton_abl/{gca}_{arm}_y.pkl | inside the same array job (+2 profiles/genome) |
| Supp Table S-ablation-structural MI column | B3 (= published), uniform p=0 | B3 = Table 1 meteor_dpz; uniform y* location UNKNOWN (holdout_cfg/ABL_uniform_mu3e0.json exists; find its meteor_out) | if y* not found: drop MI column from that table with a footnote; not load-bearing |
| Supp S7.1 reference values | CarveMe 108 panel, curated 6 GEMs | carveme_structural_metrics.py already does it (4 s/model) | 1 job, 10 min |
Implementation: copy gen_table1.py -> psb_revision/eval/gen_table1_v2.py (code_snapshot is read-only): profile() returns
additionally n_internal, mi_int, mi_int_missing_formula, mi_int_genuine, mi_frac_int; add arms skelonly/full from
skeleton_abl y pkl; write to psb_revision/results/table1_v2/. Aggregate with a copy of agg_table1_sd.py -> table1_v2_meansd.json.
Sanity gates: n_selected, deadends, fba_growth, old mi_frac must reproduce table1_{gca}.json exactly (same code path);
only the new fields are new. Any drift = stop.

## 3. Text changes
Methods L425-427: redefine "mass-imbalanced reaction" as internal reaction failing elemental balance; one clause on why
  boundary reactions are excluded (one-sided by construction).
Table 1 row: "Mass-imbalanced internal reactions ↓" with counts mean ± sd; caption note.
Sec 3.1 L473-478: replace the "about half of reactions fail elemental balance... METEOR does not repair stoichiometry"
  paragraph with: internal SEED stoichiometry is balanced in >99.8% of selected reactions for both arms, the same as
  CarveMe (99.8%) and the curated GEMs (99.4-99.96%); the residual is formula-less metabolites; selection cannot and
  need not repair stoichiometry. Two sentences.
Supp Fig S1(c): regenerate with the new metric (gen_fig_s1_panels.py, panel label).
Supp Table S7, S-ablation-structural: new MI column values or footnote.
Supp S7.1: add CarveMe + curated internal MI (and CarveMe MEMOTE dead-ends = 0 with the shipped-open-bounds caveat).
Response letter §2: replace the mass-imbalance sentence: R3 was right that the fraction was static; the cause was the
  metric counting boundary reactions; corrected metric reported, with CarveMe and curated GEMs as reference.
Length: net ≈ 0 lines (paragraph shrinks, one Methods clause grows).

## 4. Do NOT
- Add a CarveMe column under the old definition (0.14 vs 0.45 looks bad for a spurious reason).
- Compare dead-ends vs CarveMe on shipped bounds (CarveMe 0/108 because all exchanges open; curated GEMs 57-122). Caveat only.
- Run gapseq/ModelSEED (nothing on disk for the panel; 108 x >1 h; no gain).

## 5. Effort: script 1.5 h, array job ~1 h wall, aggregate + Fig S1 + text 3 h. One working day.

## 6. Adjacent issue found (not in scope, decide separately)
Table 1 "growth-feasible under the defined medium" is measured after a post-hoc uptake inference in profile()
(all EX_ open -> optimize -> keep uptake set at -10). growth_verify/*/summary.json shows 0/108 grow under the literal
stated medium in cobra sign convention (EX sign flip in extract_fba_matrices(reversed_trans=True)). Numbers are
consistent with what the pipeline does; the caption wording "under the defined medium" is the exposure. Raise with Robert.
