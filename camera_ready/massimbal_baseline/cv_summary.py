import json, statistics as s
R = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/results/table1_v2"
d = json.load(open(f"{R}/carveme_panel108.json"))
cv = [v["carveme"] for v in d.values()]
def ms(x, p=5): return f"{s.mean(x):.{p}f} ± {s.stdev(x):.{p}f}"
out = dict(n=len(cv),
    n_internal=ms([c["n_internal"] for c in cv], 1),
    mi_int=ms([c["mass_imbal_internal"] for c in cv], 2),
    mi_int_genuine=ms([c["mi_internal_genuine"] for c in cv], 2),
    mi_int_missing_formula=ms([c["mi_internal_due_to_missing_formula"] for c in cv], 2),
    mi_frac_int=ms([c["mi_frac_internal"] for c in cv]),
    mi_frac_old=ms([c["mi_frac"] for c in cv], 4),
    deadends=ms([c["deadends"] for c in cv], 2),
    n_rxn=ms([c["n_rxn"] for c in cv], 1),
    met_formula_cov_min=min(c["met_formula_cov"] for c in cv),
    n_genomes_genuine_gt0=sum(1 for c in cv if c["mi_internal_genuine"] > 0))
print(json.dumps(out, indent=1))
cur = {}
for g, v in d.items():
    if "curated_bigg" in v:
        c = v["curated_bigg"]
        cur[c["label"]] = {k: c[k] for k in ("n_rxn", "n_internal", "mass_imbal_internal", "mi_internal_due_to_missing_formula",
                                            "mi_internal_genuine", "mi_frac_internal", "mi_frac", "deadends", "n_met_noformula")}
        print(c["label"], json.dumps(cur[c["label"]]))
json.dump({"carveme_panel108": out, "curated": cur}, open(f"{R}/carveme_reference_summary.json", "w"), indent=1)
