import json, os
os.chdir("/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_r1_strat")
s75 = json.load(open("results/remap_impact/s75.json")); t5 = json.load(open("results/remap_impact/table5.json"))
DEF = ("extra = sum over the six models of (|METEOR rxn-mapped EC set| - |threshold+gap-fill baseline EC set|) "
       "(dpz vanilla, 4-digit ECs, all_ancestors-restricted seedr2ec mapping, no vocabulary restriction); "
       "correct = sum of (|M & G| - |B & G|); curated sub-threshold = sum of (|M & W| - |B & W|), W = curated ECs with 0 < max DPZ score < 0.5; "
       "marginal precision = correct/extra. The 76% is NOT a property of the extra ECs: it is Table 5's mean fraction of METEOR "
       "(active_ecs, full vocabulary) false positives lying outside the 1,234-EC reference vocabulary, |(A-G)-V_ref|/|A-G| averaged over "
       "six organisms (toolcompare_ec_vocab.py meteor_fp.frac_outside = 0.760).")
out = {"definition": DEF, "published": {"extra": 773, "correct": 77, "correct_subthreshold": 75, "marginal_precision": 0.10, "frac_outside_refvocab": 0.76}}
    m = s75[pn]["marginal"]; rows = s75[pn]["rows"]
    out[pn] = {"extra": m["extra"], "correct": m["correct"], "correct_subthreshold": m["correct_subthreshold"], "marginal_precision": m["marginal_precision"],
               "fp_outside_refvocab_frac_table5": t5[pn]["fp_outside_frac"],
               "alt_set_difference_M_minus_B": {"n": m["extra_set"], "correct": m["extra_set_correct"], "precision": m["marginal_precision_set"],
                                                "outside_refvocab": m["outside_refvocab"], "frac_outside": m["frac_outside_refvocab"]},
               "per_organism": [{"organism": r["organism"], "extra": r["extra"], "correct": r["extra_correct"], "correct_subthreshold": r["extra_correct_subthreshold"]} for r in rows]}
json.dump(out, open("results/remap_impact/marginal_precision.json", "w"), indent=1)
for k, pub in (("extra", 773), ("correct", 77), ("correct_subthreshold", 75), ("marginal_precision", 0.10), ("fp_outside_refvocab_frac_table5", 0.76)):
po = lambda pn: "; ".join(f"{r['organism']} {r['extra']}/{r['correct']}/{r['correct_subthreshold']}" for r in out[pn]["per_organism"])
L.append(f"| per-organism extra/correct/sub | | {po('none')} | {po('pkl')} | |")
L.append(f"| alt: set difference M minus B (n / correct / precision / outside ref vocab / frac) | | {out['none']['alt_set_difference_M_minus_B']} | {out['pkl']['alt_set_difference_M_minus_B']} | |")
d = lambda k: out["pkl"][k] - out["none"][k]
L.append(f"Flags vs none: correct {d('correct'):+}, sub-threshold {d('correct_subthreshold'):+}, marginal precision {d('marginal_precision'):+.3f}, FP-outside-vocab {d('fp_outside_refvocab_frac_table5'):+.3f}; extra unchanged (773). Direction unchanged.")
open("results/remap_impact_summary.md", "a").write("\n".join(L) + "\n"); print("\n".join(L))
