#!/usr/bin/env python3
"""Aggregate the deletion-recovery (decoy) records into the Section 3.3 numbers.

Reads results/recovery_abl/*.json (one per genome) and reports the spurious
reaction counts, ratios, precision and paired statistics quoted in the paper.
"""
import json, glob, os, sys
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "recovery_abl")
PANEL = os.path.join(ROOT, "cohorts", "panel108_gram.tsv")

gram = {}
if os.path.exists(PANEL):
    for line in open(PANEL):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0]:
            gram[parts[0]] = parts[1]

rows = []
for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
    o = json.load(open(f))
    if all(k in o for k in ("true", "uniform", "shuffled")):
        rows.append((os.path.basename(f)[:-5], o))

if not rows:
    sys.exit(f"no records found in {RES}")

def report(rs, label):
    t = [o["true"]["decoy"] for _, o in rs]
    u = [o["uniform"]["decoy"] for _, o in rs]
    s = [o["shuffled"]["decoy"] for _, o in rs]
    pt = [o["true"]["precision"] for _, o in rs]
    pu = [o["uniform"]["precision"] for _, o in rs]
    print(f"{label:10s} n={len(rs):3d}  spurious: true={mean(t):6.1f} "
          f"uniform={mean(u):6.1f} shuffled={mean(s):7.1f}  "
          f"| uniform/true={mean(u)/mean(t):.2f}x shuffled/true={mean(s)/mean(t):.1f}x "
          f"| precision true={mean(pt):.3f} uniform={mean(pu):.3f}")
    worse = sum(1 for a, b in zip(u, t) if a > b)
    print(f"{'':10s} uniform adds more than true in {worse}/{len(rs)} genomes")
    try:
        from scipy.stats import wilcoxon
        print(f"{'':10s} Wilcoxon uniform vs true p={wilcoxon(u, t).pvalue:.3g}; "
              f"shuffled vs true p={wilcoxon(s, t).pvalue:.3g}")
    except ImportError:
        print(f"{'':10s} (install scipy for the paired tests)")

report(rows, "ALL")
for g in ("negative", "positive"):
    sub = [(k, o) for k, o in rows if gram.get(k) == g]
    if sub:
        report(sub, "gram-" + g[:3].upper())
