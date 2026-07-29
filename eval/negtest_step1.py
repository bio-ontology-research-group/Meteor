#!/usr/bin/env python3
import os,pickle,glob,sys,collections

from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX

OUTROOT = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out_bacdive/dpz_vanilla"
BDFILE = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results/bacdive_meta/bacdive_met_util.tsv"
TOP50 = {l.split()[0] for l in open("/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/bacdive_top50_list.tsv") if not l.startswith("genome")}

EC_CAND = {"EC:3.2.1.14": "chitin", "EC:1.17.3.2": "xanthine", "EC:3.2.1.4": "cellulose"}

# BacDive top50 counts
bd_counts = {ec: {"+":0,"-":0} for ec in EC_CAND}
for l in open(BDFILE):
    if l.startswith("genome"): continue
    p = l.strip().split("\t")
    if len(p) < 3: continue
    g, c, ab = p[0], p[1], p[2]
    if g not in TOP50: continue
    cl = c.lower()
    for ec_full, keyword in EC_CAND.items():
        if keyword in cl:
            bd_counts[ec_full][ab] = bd_counts[ec_full].get(ab, 0) + 1

print("=== BacDive phenotype counts (top-50) ===")
for ec_full, keyword in EC_CAND.items():
    c = bd_counts[ec_full]
    print("  %s (%s): +%d  -%d" % (keyword, ec_full, c.get("+", 0), c.get("-", 0)))

# DPZ screening
results = {}
for ec in EC_CAND:
    results[ec] = {"n":0,"dpz_gt0":0,"dpz_gt05":0,"dpz_ge09":0,"dpz_eq0":0,"scores":[],
                   "met_active":0,"met_muted":0,"met_absent":0}

n_total = 0
for f in sorted(glob.glob(OUTROOT + "/meteor_preds_*.pkl")):
    bn = os.path.basename(f)
    gcf = bn.replace("meteor_preds_", "").replace(".pkl", "")
    n_total += 1
    d = pickle.load(open(f, "rb"))
    active = {str(e).split(":")[-1] for e in d.get("active_ecs", set())}
    muted = {str(e).split(":")[-1] for e in d.get("muted_ecs", set())}

    for ec_full in EC_CAND:
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

print("\n=== DPZ baseline screening across %d completed genomes ===" % n_total)
for ec_full, keyword in EC_CAND.items():
    r = results[ec_full]
    print("\n  %s (%s)" % (keyword, ec_full))
    print("    DPZ: >0=%d  >0.5=%d  >=0.9=%d  =0=%d  (n=%d)" % (
        r["dpz_gt0"], r["dpz_gt05"], r["dpz_ge09"], r["dpz_eq0"], r["n"]))
    scores = [s for s in r["scores"] if s > 0]
    if scores:
        s_sorted = sorted(scores)
        print("    Positive scores: min=%.4f  max=%.4f  mean=%.4f  median=%.4f" % (
            min(scores), max(scores), sum(scores)/len(scores), s_sorted[len(s_sorted)//2]))
    frac = r["dpz_gt0"] / r["n"] * 100 if r["n"] > 0 else 0
    tag = "HIGH" if frac > 50 else ("MODERATE" if frac > 20 else "GOOD")
    print("    Ubiquity: DPZ>0 in %.0f%% of genomes  --> %s" % (frac, tag))
    print("    METEOR: active=%d  muted=%d  absent=%d" % (r["met_active"], r["met_muted"], r["met_absent"]))
    bc = bd_counts[ec_full]
    print("    Top-50 BacDive: -%d  +%d" % (bc.get("-", 0), bc.get("+", 0)))

print("\nDone")
