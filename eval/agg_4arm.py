#!/usr/bin/env python3
"""Aggregate the four-arm deletion-recovery experiment.

The two-stage arm answers a question the original three cannot: all of them
select over the whole universal database in one optimisation, so they cannot
separate the contribution of evidence weighting from that of single-stage
selection. The two-stage arm applies the identical evw cost, but only to
gap-fill candidates on top of a thresholded draft -- the shape ProbAnno,
GLOBUS and probabilistic gap-filling generally take.
"""
import json, glob, os, sys
from statistics import mean, median

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "recovery_4arm")
if not os.path.isdir(RES):
    RES = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/results/recovery_4arm"
PANEL = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results/panel108_gram.tsv"

gram = {}
if os.path.exists(PANEL):
    for l in open(PANEL):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0]: gram[p[0]] = p[1]

ARMS = ["true", "twostage", "uniform", "shuffled"]
rows = []
for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
    o = json.load(open(f))
    if all(a in o and "decoy" in o.get(a, {}) for a in ARMS):
        rows.append(o)
if not rows: sys.exit(f"no complete records in {RES}")

n_infeas = sum(1 for f in glob.glob(os.path.join(RES, "*.json"))
               if "err" in json.load(open(f)).get("twostage", {}))
print(f"genomes with all four arms: {len(rows)}"
      + (f"   (two-stage infeasible in {n_infeas})" if n_infeas else ""))

base = mean(r["true"]["decoy"] for r in rows)
print(f"\n{'arm':12s} {'spurious':>10s} {'median':>8s} {'vs true':>9s} {'precision':>10s}")
for a in ARMS:
    d = [r[a]["decoy"] for r in rows]
    p = [r[a]["precision"] for r in rows]
    print(f"{a:12s} {mean(d):10.1f} {median(d):8.1f} {mean(d)/base:8.2f}x {mean(p):10.3f}")

print("\npaired against METEOR (true), per genome:")
for a in ARMS[1:]:
    t = [r["true"]["decoy"] for r in rows]; x = [r[a]["decoy"] for r in rows]
    worse = sum(1 for u, v in zip(x, t) if u > v)
    print(f"  {a:10s} adds more spurious in {worse}/{len(rows)} genomes", end="")
    try:
        from scipy.stats import wilcoxon
        print(f"   p={wilcoxon(x, t).pvalue:.3g}")
    except Exception:
        print()

if gram:
    print("\nby Gram group:")
    for g in ("negative", "positive"):
        sub = [r for r in rows if gram.get(r["gca"]) == g]
        if not sub: continue
        b = mean(r["true"]["decoy"] for r in sub)
        vals = "  ".join(f"{a}={mean(r[a]['decoy'] for r in sub)/b:.2f}x" for a in ARMS[1:])
        print(f"  {g:9s} n={len(sub):3d}  {vals}")
