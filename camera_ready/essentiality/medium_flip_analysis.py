"""Round 6: gene-level medium-sensitivity check for E. coli, pure re-analysis
of already-saved SGD per-gene calls (GS_MM_glc run vs LB_marinos run), no
new compute. For each predictor x arm, intersect the model-gene sets
between the two media and count essentiality flips.
Writes only under psb_revision/feasibility_essentiality/results/medflip/.
"""
import json, csv, os
HERE = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality"
OUT = f"{HERE}/results/medflip"; os.makedirs(OUT, exist_ok=True)

p2ref = {}
for l in open(f"{HERE}/ref/m25631h_to_bnum.tsv"):
    parts = l.rstrip("\n").split("\t")
    if len(parts) >= 3: p2ref[parts[0]] = parts[2]

pecdat = {}
for r in csv.DictReader(open(f"{HERE}/ref/pec/PECData.dat"), delimiter="\t"):
    alt = r["Alternative name"].split(",")[0].strip()
    pecdat[alt] = r["Product"]

PREDICTORS = ["clean", "dpz", "enzbert"]
ARMS = ["meteor", "thresh"]

results = {}
for arm in ARMS:
    for pred in PREDICTORS:
        gsmm = json.load(open(f"{HERE}/results/essround3/sgd_ecoli_{arm}_{pred}.json"))
        lbm = json.load(open(f"{HERE}/results/essround3/sgd_ecoli_lb_{arm}_{pred}.json"))
        common = set(gsmm) & set(lbm)
        n_common = len(common)
        ess_gsmm_only = []  # essential in GS_MM_glc, non-essential in LB_marinos
        ess_lbm_only = []   # essential in LB_marinos, non-essential in GS_MM_glc
        same = 0
        for g in common:
            a, b = bool(gsmm[g]), bool(lbm[g])
            if a == b:
                same += 1
            elif a and not b:
                ess_gsmm_only.append(g)
            else:
                ess_lbm_only.append(g)
        n_flip = len(ess_gsmm_only) + len(ess_lbm_only)
        key = f"{arm}/{pred}"
        results[key] = dict(
            n_gsmm_model=len(gsmm), n_lbm_model=len(lbm), n_common=n_common,
            n_same=same, n_flip=n_flip, flip_frac=round(n_flip / max(1, n_common), 4),
            n_ess_gsmm_only=len(ess_gsmm_only), n_ess_lbm_only=len(ess_lbm_only),
            flipped_genes_gsmm_essential_lb_dispensable=sorted(ess_gsmm_only),
            flipped_genes_lb_essential_gsmm_dispensable=sorted(ess_lbm_only),
        )
        print(f"{key:16s} common={n_common:5d} same={same:5d} flip={n_flip:4d} ({100*n_flip/max(1,n_common):.2f}%)"
              f"  [GSMM-ess-only={len(ess_gsmm_only)}  LBM-ess-only={len(ess_lbm_only)}]")

json.dump(results, open(f"{OUT}/flip_analysis.json", "w"), indent=1)

print("\n=== Functional annotation of flipped genes (PEC Product, via b-number RBH map) ===")
for key, r in results.items():
    print(f"\n-- {key} --")
    for g in r["flipped_genes_gsmm_essential_lb_dispensable"][:30]:
        b = p2ref.get(g)
        print(f"  [GSMM-ess,LB-disp] {g} (b={b}): {pecdat.get(b, '?')}")
    for g in r["flipped_genes_lb_essential_gsmm_dispensable"][:30]:
        b = p2ref.get(g)
        print(f"  [LB-ess,GSMM-disp] {g} (b={b}): {pecdat.get(b, '?')}")

print("\n=== SUMMARY TABLE ===")
print(f"{'arm':10s} {'predictor':10s} {'n_common':9s} {'n_flip':7s} {'flip_%':7s}")
for key, r in results.items():
    arm, pred = key.split("/")
    print(f"{arm:10s} {pred:10s} {r['n_common']:9d} {r['n_flip']:7d} {100*r['flip_frac']:6.2f}%")
