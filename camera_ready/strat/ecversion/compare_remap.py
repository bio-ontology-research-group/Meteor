"""Side-by-side: frozen protocol (results/strat_summary.json) vs remap-consistent (results/strat_remap_summary.json)."""
import json, os, sys
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(f"{D}/results/strat_summary.json")); B = json.load(open(f"{D}/results/strat_remap_summary.json"))
ARMS = ["B", "M-rxn", "M-act", "S", "S-allSEED"]; STR = ["all", "in_vocab", "out_vocab", "bin0", "bin(0,0.1)", "bin[0.1,0.5)", "bin[0.5,1]"]
L = ["# Frozen vs remap-consistent protocol (mean over 6 organisms; delta = remap - frozen)", ""]
flag = []
for p in A["predictors"]:
    a, b = A["predictors"][p], B["predictors"][p]
    L += [f"## {p}: vocab {a['vocab_size']} -> {b['vocab_size']}; out-of-vocab frac {a['frac_out_vocab']['mean']:.3f} -> {b['frac_out_vocab']['mean']:.3f}; "
          f"n_out_vocab {a['n_out_vocab']['mean']:.1f} -> {b['n_out_vocab']['mean']:.1f}; gem_ec {sum(a['gem_ec'])/6:.1f} -> {sum(b['gem_ec'])/6:.1f}; "
          f"ceiling {a['no_candidate_rxn']['mean']:.1f} -> {b['no_candidate_rxn']['mean']:.1f}", "",
          "| arm | " + " | ".join(STR) + " |", "|---|" + "---|" * len(STR)]
    for arm in ARMS:
        cells = []
        for x in STR:
            ma, mb = a["recall"][arm][x]["mean"], b["recall"][arm][x]["mean"]
            if ma is None or mb is None: cells.append("-"); continue
            d = mb - ma; cells.append(f"{ma:.3f} -> {mb:.3f} ({d:+.3f})")
            if abs(d) > 0.005: flag.append((p, arm, x, round(d, 4)))
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L += ["", "Wilcoxon p (frozen / remap): " + "; ".join(f"{k}: {a['wilcoxon'][k]['p']} / {b['wilcoxon'].get(k, {}).get('p')}" for k in
          ("M-rxn_vs_B__in_vocab", "M-rxn_vs_S__out_vocab", "M-rxn_vs_S__in_vocab", "M-act_vs_S-allSEED__out_vocab")), ""]
L += [f"cells with |delta| > 0.005: {len(flag)}", json.dumps(flag)]
open(f"{D}/ecversion/compare_remap.md", "w").write("\n".join(L)); print("\n".join(L))
