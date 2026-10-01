"""Round 6 step 4: cross-check our model's medium-flip genes against Joyce
et al. 2006's real 119-gene glycerol-minimal-vs-rich-viable benchmark
(J Bacteriol 188:8259, PMID 17012394, PMC1698209 -- Table 1, fetched
directly from the freely-rendered PMC article page, 119/119 genes
recovered with Blattner numbers). Pure re-analysis of already-saved data
(flip_analysis.json from medium_flip_analysis.py + the per-gene SGD calls),
no new compute.
"""
import json, csv

HERE = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality"

p2ref = {}
for l in open(f"{HERE}/ref/m25631h_to_bnum.tsv"):
    parts = l.rstrip("\n").split("\t")
    if len(parts) >= 3: p2ref[parts[0]] = parts[2]

joyce119 = set()
joyce_name = {}
for l in open(f"{HERE}/ref/joyce2006/joyce119_genes.tsv"):
    name, b = l.rstrip("\n").split("\t")
    joyce119.add(b); joyce_name[b] = name
print(f"Joyce 2006 Table 1: {len(joyce119)} genes loaded")

flip = json.load(open(f"{HERE}/results/medflip/flip_analysis.json"))

print("\n=== Step 4: direction-correctness against Joyce's real 119-gene list ===")
print(f"{'arm':10s} {'predictor':10s} {'n_common_model_genes':10s} {'n_common_&_in_joyce119':10s} "
      f"{'n_correct_direction':10s} {'n_flip_but_wrong_dir':10s} {'n_joyce_not_flipped':10s}")
summary = {}
for key, r in flip.items():
    arm, pred = key.split("/")
    # reload the raw per-gene calls to know, for genes IN joyce119 AND in the model
    # intersection, whether our model calls them ess-in-GSMM/non-ess-in-LB (correct direction)
    gsmm = json.load(open(f"{HERE}/results/essround3/sgd_ecoli_{arm}_{pred}.json"))
    lbm = json.load(open(f"{HERE}/results/essround3/sgd_ecoli_lb_{arm}_{pred}.json"))
    common = set(gsmm) & set(lbm)
    # map common model genes (protein ids) to b-numbers, keep only those also in joyce119
    common_b = {}
    for g in common:
        b = p2ref.get(g)
        if b and b in joyce119:
            common_b[b] = g
    n_common_joyce = len(common_b)
    correct_dir = 0; wrong_dir = 0; not_flipped = 0
    correct_genes = []; wrong_genes = []
    for b, g in common_b.items():
        a_ess, l_ess = bool(gsmm[g]), bool(lbm[g])
        if a_ess and not l_ess:
            correct_dir += 1; correct_genes.append((b, joyce_name[b]))
        elif (not a_ess) and l_ess:
            wrong_dir += 1; wrong_genes.append((b, joyce_name[b]))
        else:
            not_flipped += 1
    summary[key] = dict(n_common_model=len(common), n_common_and_joyce119=n_common_joyce,
                         n_correct_direction=correct_dir, n_flip_wrong_direction=wrong_dir,
                         n_joyce_gene_not_flipped=not_flipped,
                         correct_genes=correct_genes, wrong_direction_genes=wrong_genes)
    print(f"{arm:10s} {pred:10s} {len(common):20d} {n_common_joyce:22d} {correct_dir:20d} {wrong_dir:20d} {not_flipped:20d}")

json.dump(summary, open(f"{HERE}/results/medflip/joyce_crosscheck.json", "w"), indent=1)

print("\n=== Correctly-flipped genes detail (sample) ===")
for key, s in summary.items():
    if s["correct_genes"]:
        print(f"{key}: {s['correct_genes']}")

print("\n=== Wrong-direction genes detail (i.e. model calls LB-essential/GSMM-dispensable, but Joyce says minimal-essential) ===")
for key, s in summary.items():
    if s["wrong_direction_genes"]:
        print(f"{key}: {s['wrong_direction_genes']}")
