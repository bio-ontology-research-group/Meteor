#!/usr/bin/env python3
"""Aggregate the flux-variability records into the numbers reported in
Section 3.1 and Limitations: how many selected reactions can carry flux,
for METEOR and for the thresholded baseline.

Reads:  results/fva_selected/fva_*.json     (METEOR arm, 108 genomes)
        results/fva_baseline/fva_baseline_*.json
Expected: METEOR 54.5% flux-consistent vs baseline 40.5%, METEOR ahead in
        108/108 genomes, paired Wilcoxon p=1.9e-19.
"""
import json, glob, os, sys
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

def load(sub, pat):
    out = {}
    for f in glob.glob(os.path.join(ROOT, "results", sub, pat)):
        r = json.load(open(f)); out[r["gca"]] = r
    return out

M = load("fva_selected", "fva_*.json")
B = load("fva_baseline", "fva_baseline_*.json")
M = {k: v for k, v in M.items() if v.get("arm", "meteor") == "meteor"}
common = sorted(set(M) & set(B))
if not common: sys.exit("no paired records found")

print(f"paired genomes: {len(common)}\n")
print(f"{'arm':10s} {'selected':>10s} {'flux-consistent':>16s} {'fraction':>10s}")
for lab, D in (("baseline", B), ("METEOR", M)):
    print(f"{lab:10s} {mean(D[g]['n_selected'] for g in common):10.0f} "
          f"{mean(D[g]['n_flux_consistent'] for g in common):16.0f} "
          f"{100*mean(D[g]['frac_consistent'] for g in common):9.1f}%")

mf = [M[g]["frac_consistent"] for g in common]
bf = [B[g]["frac_consistent"] for g in common]
print(f"\nMETEOR higher in {sum(1 for a, b in zip(mf, bf) if a > b)}/{len(common)} genomes")
print(f"carrying flux in the FBA optimum (METEOR): "
      f"{100*mean(M[g]['frac_carrying'] for g in common):.1f}%")
try:
    from scipy.stats import wilcoxon
    print(f"paired Wilcoxon on the fraction: p={wilcoxon(mf, bf).pvalue:.3g}")
except ImportError:
    print("(install scipy for the paired test)")
