#!/usr/bin/env python3
"""Scan candidate negative tests: add min DPZ per genome."""
import os,pickle,glob,sys,collections

from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX

OUTROOT = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out_bacdive/dpz_vanilla"
BDFILE = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results/bacdive_meta/bacdive_met_util.tsv"
TOP50 = [l.split()[0] for l in open("/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/bacdive_top50_list.tsv") if not l.startswith("genome")]

CANDIDATES = [
    ("agar hydrolysis", "agar", "EC:3.2.1.81"),
    ("alginate hydrolysis", "alginate", "EC:4.2.2.3"),
    ("xylan hydrolysis", "xylan", "EC:3.2.1.8"),
    ("pectin hydrolysis", "pectin", "EC:3.2.1.15"),
    ("lecithin hydrolysis", "lecithin", "EC:3.1.1.4"),
    ("dna hydrolysis", "dna", "EC:3.1.21.1"),
    # also add back the original 3 + urease for comparison
    ("urea hydrolysis", "urea", "EC:3.5.1.5"),
    ("chitin hydrolysis", "chitin", "EC:3.2.1.14"),
    ("cellulose hydrolysis", "cellulose", "EC:3.2.1.4"),
    ("xanthine hydrolysis", "xanthine", "EC:1.17.3.2"),
]

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

results = {}
for c in CANDIDATES:
    results[c[2]] = {"n":0, "dpz_gt0":0, "dpz_gt05":0, "max_scores":[], "min_scores":[],
                     "met_active":0, "met_muted":0, "met_absent":0}

n_total = 0
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
                mx = float(df[ec_full].max())
                mn = float(df[ec_full].min())
                r["max_scores"].append(mx)
                r["min_scores"].append(mn)
                if mx > 0: r["dpz_gt0"] += 1
                if mx > 0.5: r["dpz_gt05"] += 1
            else:
                r["max_scores"].append(0.0)
                r["min_scores"].append(0.0)
        except:
            r["max_scores"].append(0.0)
            r["min_scores"].append(0.0)

print("=== Candidate negative tests (top-50 BacDive counts + DPZ across %d genomes) ===" % n_total)
print("")
hdr = "%-20s %-16s %5s %5s  %7s %7s %7s  %7s %7s  %8s %8s  %5s"
print(hdr % ("Test", "EC", "-cnt", "+cnt", "DPZ>0%", "DPZ>50%", "cov%", "max_mean", "min_mean", "met_active", "met_muted", "spec?"))
print("-" * 115)

for c in CANDIDATES:
    tname, keyword, ec_full = c[0], c[1], c[2]
    r = results[ec_full]
    n_neg = neg_tests.get(tname, 0)
    n_pos = pos_tests.get(tname, 0)
    frac_gt0 = r["dpz_gt0"] / r["n"] * 100
    frac_gt05 = r["dpz_gt05"] / r["n"] * 100
    max_mean = sum(r["max_scores"]) / len(r["max_scores"]) if r["max_scores"] else 0
    min_mean = sum(r["min_scores"]) / len(r["min_scores"]) if r["min_scores"] else 0

    # expected specificity in top-50 negative genomes
    fp_rate = r["met_active"] / r["n"] if r["n"] > 0 else 0
    exp_fp = fp_rate * n_neg
    spec_str = "%.0f%%" % (100 * (1 - fp_rate)) if n_neg > 0 else "N/A"

    print("%-20s %-16s %5d %5d  %7.0f %7.0f %7.0f  %7.4f %7.4f  %8d %8d  %5s" % (
        keyword, ec_full, n_neg, n_pos,
        frac_gt0, frac_gt05, frac_gt0,
        max_mean, min_mean,
        r["met_active"], r["met_muted"], spec_str))

print("\nInterpretation:")
print("  cov% = fraction of genomes where DPZ > 0 for this EC")
print("  max_mean = mean of per-genome MAX DPZ score (across proteins)")
print("  min_mean = mean of per-genome MIN DPZ score (across proteins)")
print("  spec% = expected specificity on negative test (met_active/n)")
print("  Best negative controls: high -cnt, low met_active, moderate DPZ cov")
print("Done")
