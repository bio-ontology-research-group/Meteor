#!/usr/bin/env python3
"""Scan candidate negative tests: check DPZ coverage + METEOR status."""
import os,pickle,glob,sys,collections
sys.path.insert(0,"/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/eval")
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX

OUTROOT = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out_bacdive/dpz_vanilla"
BDFILE = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results/bacdive_meta/bacdive_met_util.tsv"
TOP50 = [l.split()[0] for l in open("/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/bacdive_top50_list.tsv") if not l.startswith("genome")]

# Hand-curated EC -> test mapping (specific, narrow ECs)
CANDIDATES = [
    ("agar hydrolysis", "agar", "EC:3.2.1.81", 3.2, 1, 81),
    ("alginate hydrolysis", "alginate", "EC:4.2.2.3", 4.2, 2, 3),
    ("xylan hydrolysis", "xylan", "EC:3.2.1.8", 3.2, 1, 8),
    ("pectin hydrolysis", "pectin", "EC:3.2.1.15", 3.2, 1, 15),
    ("lecithin hydrolysis", "lecithin", "EC:3.1.1.4", 3.1, 1, 4),
    ("dna hydrolysis", "dna", "EC:3.1.21.1", 3.1, 21, 1),
]

# Count BacDive in top50
neg_tests = collections.Counter()
pos_tests = collections.Counter()
for l in open(BDFILE):
    if l.startswith("genome"): continue
    p = l.strip().split("\t")
    if len(p)<4: continue
    g,c,ab,k = p[0],p[1],p[2],p[3]
    if g not in TOP50: continue
    if k not in ("reduction","hydrolysis"): continue
    tname = c + " " + k
    if ab == "-": neg_tests[tname] += 1
    elif ab == "+": pos_tests[tname] += 1

print("=== Candidate negative tests (top-50) ===")
for tname, ec_full in [(c[0], c[2]) for c in CANDIDATES]:
    n_neg = neg_tests.get(tname, 0)
    n_pos = pos_tests.get(tname, 0)
    keyword = tname.split()[0]
    print("  %-40s %s  +%d -%d" % (tname, ec_full, n_pos, n_neg))

# DPZ coverage check across ALL completed genomes
print("\n=== DPZ coverage + METEOR status across completed genomes ===")
n_total = 0
results = {}
for ec_full in [c[2] for c in CANDIDATES]:
    results[ec_full] = {"n":0,"dpz_gt0":0,"dpz_gt05":0,"dpz_ge09":0,"dpz_eq0":0,"scores":[],
                        "met_active":0,"met_muted":0,"met_absent":0}

for f in sorted(glob.glob(OUTROOT + "/meteor_preds_*.pkl")):
    bn = os.path.basename(f)
    gcf = bn.replace("meteor_preds_", "").replace(".pkl", "")
    n_total += 1
    d = pickle.load(open(f, "rb"))
    active = {str(e).split(":")[-1] for e in d.get("active_ecs", set())}
    muted = {str(e).split(":")[-1] for e in d.get("muted_ecs", set())}

    for ec_full in [c[2] for c in CANDIDATES]:
        ec_short = ec_full.split(":")[-1]
        r = results[ec_full]
        r["n"] += 1
        if ec_short in active: r["met_active"] += 1
        elif ec_short in muted: r["met_muted"] += 1
        else: r["met_absent"] += 1

        dpz_path = resolve_baseline_pkl("dpz", "vanilla", gcf, BASELINE_SUFFIX["dpz"])
        try:
            df = pickle.load(open(dpz_path, "rb"))
            if ec_full in df.columns:
                s = float(df[ec_full].max())
                r["scores"].append(s)
                if s > 0: r["dpz_gt0"] += 1
                if s > 0.5: r["dpz_gt05"] += 1
                if s >= 0.9: r["dpz_ge09"] += 1
                if s == 0: r["dpz_eq0"] += 1
        except:
            pass

for c in CANDIDATES:
    tname, keyword, ec_full = c[0], c[1], c[2]
    r = results[ec_full]
    n_neg = neg_tests.get(tname, 0)
    n_pos = pos_tests.get(tname, 0)
    frac = r["dpz_gt0"] / r["n"] * 100 if r["n"] > 0 else 0
    scores = [s for s in r["scores"] if s > 0]

    print("\n  %-20s %s" % (keyword, ec_full))
    print("    Top-50 BacDive: -%d +%d" % (n_neg, n_pos))
    print("    DPZ coverage: %d/%d = %.0f%% of genomes" % (r["dpz_gt0"], r["n"], frac))
    if scores:
        print("    DPZ scores: mean=%.4f median=%.4f" % (sum(scores)/len(scores), sorted(scores)[len(scores)//2]))
        print("    DPZ >0.5: %d  >=0.9: %d" % (r["dpz_gt05"], r["dpz_ge09"]))
    print("    METEOR: active=%d  muted=%d  absent=%d" % (r["met_active"], r["met_muted"], r["met_absent"]))
    exp_fp = r["met_active"] * n_neg / r["n"] if r["n"] > 0 else 0
    print("    Expected FP in top-50: ~%.1f/%d (spec=%.0f%%)" % (
        exp_fp, n_neg, 100*(1-exp_fp/n_neg) if n_neg > 0 else 0))

print("\nDone")
