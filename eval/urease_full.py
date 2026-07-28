#!/usr/bin/env python3
"""Check DPZ baseline for EC 3.5.1.5 across completed genomes + summary."""

import os, pickle, glob, sys

OUTROOT = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out_bacdive/dpz_vanilla"
DPZROOT = "/ibex/scratch/projects/c2014/kexin/funcarve/dpec2_result/result_bacdive"
BDFILE = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results/bacdive_meta/bacdive_met_util.tsv"

EC_TARGET = "EC:3.5.1.5"  # DPZ columns have "EC:" prefix

# load urease bacdive
urease_genomes = set()
for l in open(BDFILE):
    if l.startswith("genome"): continue
    p = l.strip().split(chr(9))
    if len(p) < 4: continue
    g, c, ab, k = p[0], p[1], p[2], p[3]
    if "urea" in c.lower() and k in ("reduction", "hydrolysis"):
        urease_genomes.add(g)

# iterate METEOR results
n_urease_active = 0
n_urease_no51 = 0
n_nourease_active = 0
dpz_scores = []
missing_dpz = []

for f in sorted(glob.glob(OUTROOT + "/meteor_preds_*.pkl")):
    bn = os.path.basename(f)
    gcf = bn.replace("meteor_preds_", "").replace(".pkl", "")
    d = pickle.load(open(f, "rb"))
    active = {str(e).split(":")[-1] for e in d.get("active_ecs", set())}
    has_active = "3.5.1.5" in active
    has_urease = gcf in urease_genomes

    if has_urease and has_active:
        n_urease_active += 1
        # DPZ score
        dpz_path = DPZROOT + "/" + gcf + "_negative_DeepECv2_t5.pkl"
        alt = DPZROOT + "/" + gcf + "_positive_DeepECv2_t5.pkl"
        if os.path.exists(dpz_path):
            pass
        elif os.path.exists(alt):
            dpz_path = alt
        else:
            # try glob
            hits = glob.glob(DPZROOT + "/" + gcf.split(".")[0] + "_*_DeepECv2_t5.pkl")
            if hits:
                dpz_path = hits[0]
            else:
                missing_dpz.append(gcf)
                dpz_scores.append((gcf, None))
                continue

        try:
            df = pickle.load(open(dpz_path, "rb"))
            if EC_TARGET in df.columns:
                score = float(df[EC_TARGET].max())
            else:
                score = 0.0
            dpz_scores.append((gcf, score))
        except Exception as e:
            dpz_scores.append((gcf, "ERR:" + str(e)[:40]))

    elif has_urease and not has_active:
        n_urease_no51 += 1
    elif not has_urease and has_active:
        n_nourease_active += 1

print("=== EC 3.5.1.5 urease analysis ===")
print("Total METEOR results:", len(glob.glob(OUTROOT + "/meteor_preds_*.pkl")))
print("Has urease test + EC active:", n_urease_active)
print("Has urease test + EC absent:", n_urease_no51)
print("No urease test + EC active:", n_nourease_active)

valid = [s for _, s in dpz_scores if isinstance(s, float)]
print()
print("=== DPZ baseline scores for EC 3.5.1.5 (urease+active genomes) ===")
print("Valid DPZ scores:", len(valid))
if valid:
    print("  min: {:.4f}".format(min(valid)))
    print("  max: {:.4f}".format(max(valid)))
    print("  mean: {:.4f}".format(sum(valid) / len(valid)))
    print("  weak (<0.5): {} / {}".format(sum(1 for s in valid if s < 0.5), len(valid)))
    print("  very weak (<0.1): {} / {}".format(sum(1 for s in valid if s < 0.1), len(valid)))
    print("  strong (>=0.9): {} / {}".format(sum(1 for s in valid if s >= 0.9), len(valid)))
    print("  zero (0.0): {} / {}".format(sum(1 for s in valid if s == 0.0), len(valid)))
    print()
    print("First 10:")
    for g, s in dpz_scores[:10]:
        print("  {}: {:.4f}".format(g, s) if isinstance(s, float) else "  {}: {}".format(g, s))

if missing_dpz:
    print()
    print("Missing DPZ files ({}):".format(len(missing_dpz)))
    for g in missing_dpz[:5]:
        print("  " + g)
