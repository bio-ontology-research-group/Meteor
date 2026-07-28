#!/usr/bin/env python3
"""Aggregate the per-genome structural records into main-text Table 1."""
import json, glob, os, sys
from statistics import mean
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "table1")
PANEL = os.path.join(ROOT, "cohorts", "panel108_gram.tsv")
panel = {l.split("\t")[0].strip() for l in open(PANEL) if l.strip()} if os.path.exists(PANEL) else None
CFG = ["baseline_clean","baseline_dpz","baseline_enzbert",
       "meteor_clean","meteor_dpz","meteor_enzbert"]
acc = {c: {} for c in CFG}; used = []
for f in sorted(glob.glob(os.path.join(RES, "table1_*.json"))):
    o = json.load(open(f)); g = o["gca"]
    if panel and g not in panel: continue
    used.append(g)
    for c in CFG:
        v = o.get(c)
        if not isinstance(v, dict) or "err" in v: continue
        for k in ("n_selected","n_rxn","deadends","mass_imbal","mi_frac","fba_growth"):
            acc[c].setdefault(k, []).append(v[k])
if not used: sys.exit(f"no records in {RES}")
print(f"genomes: {len(used)}\n")
hdr = f"{'config':18s} {'selected':>9s} {'submodel':>9s} {'deadend':>8s} {'massimb':>8s} {'mi_frac':>8s} {'growing':>9s}"
print(hdr); print("-" * len(hdr))
for c in CFG:
    a = acc[c]
    if not a.get("n_rxn"): print(f"{c:18s}  no data"); continue
    n = len(a["n_rxn"]); grow = sum(1 for x in a["fba_growth"] if x and x > 1e-6)
    print(f"{c:18s} {mean(a['n_selected']):9.1f} {mean(a['n_rxn']):9.1f} "
          f"{mean(a['deadends']):8.2f} {mean(a['mass_imbal']):8.1f} "
          f"{mean(a['mi_frac']):8.4f} {grow:5d}/{n:<3d}")
print("\ndead-end reduction (baseline -> METEOR):")
for b in ("clean","dpz","enzbert"):
    x, y = mean(acc[f"baseline_{b}"]["deadends"]), mean(acc[f"meteor_{b}"]["deadends"])
    print(f"  {b:8s} {x:5.2f} -> {y:4.2f}   {100*(x-y)/x:.1f}% fewer")
