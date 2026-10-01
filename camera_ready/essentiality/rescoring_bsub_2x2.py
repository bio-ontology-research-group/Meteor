"""Round 7 step 6: B. subtilis medium x reference 2x2, pure re-scoring of
already-saved SGD calls (no new MILP/COBRA compute), mirroring rescoring_2x2.py
for E. coli. Networks: bsub (LB_marinos, round 3) and bsub_koomin (GS_MM_glc,
round 7). References: gess-bsub.csv (gene2 col) and bsub_minimal_binary.csv
(gene col, Koo2017-derived). Join key: symbol via knb1_to_symbol.tsv (same
mapper used to build both organism configs).
"""
import json, csv, os
import numpy as np

HERE = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality"
OUT = f"{HERE}/results/rescoring2x2"; os.makedirs(OUT, exist_ok=True)

p2sym = {}
for l in open(f"{HERE}/ref/knb1_to_symbol.tsv"):
    parts = l.rstrip("\n").split("\t")
    if len(parts) >= 3: p2sym[parts[0]] = parts[2]

def load_ref(csv_path, gene_col):
    ref = {}
    for row in csv.DictReader(open(csv_path)):
        g = str(row[gene_col]).strip()
        if g: ref[g] = (row["ess.experimental"] == "yes")
    return ref

REF = {
    "gess-bsub.csv": load_ref("/ibex/scratch/projects/c2014/kexin/funcarve/gapseq_eval/gapseqEval/GeneEssentiality/essentiality.data/gess-bsub.csv", "gene2"),
    "Koo2017_combined": load_ref(f"{HERE}/ref/koo2017/bsub_minimal_binary.csv", "gene"),
}
for k, v in REF.items():
    print(k, sum(v.values()), "essential /", len(v), "total")

def metrics(pred_ess, ref_ess, mapper=p2sym.get):
    tp = fp = fn = tn = 0; n_mapped = 0; n_ref = 0
    for g, e in pred_ess.items():
        b = mapper(g)
        if b is None: continue
        n_mapped += 1
        if b not in ref_ess: continue
        n_ref += 1; t = ref_ess[b]
        if e and t: tp += 1
        elif e and not t: fp += 1
        elif (not e) and t: fn += 1
        else: tn += 1
    P = tp / max(1, tp + fp); Rc = tp / max(1, tp + fn)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / den if den > 0 else 0.0
    return dict(n_genes_model=len(pred_ess), n_genes_mapped=n_mapped, n_genes_with_ref=n_ref,
                TP=tp, FP=fp, FN=fn, TN=tn, precision=round(P, 3), recall=round(Rc, 3), MCC=round(mcc, 3))

NETWORKS = {"LB_marinos": "bsub", "GS_MM_glc": "bsub_koomin"}
PREDICTORS = ["clean", "dpz", "enzbert"]
ARMS = ["meteor", "thresh"]

grid = {}
for medium, org in NETWORKS.items():
    for refname, ref_ess in REF.items():
        cell = {}
        for pred in PREDICTORS:
            cell[pred] = {}
            for arm in ARMS:
                calls = json.load(open(f"{HERE}/results/essround3/sgd_{org}_{arm}_{pred}.json"))
                calls = {g: bool(e) for g, e in calls.items()}
                cell[pred][arm] = metrics(calls, ref_ess)
        grid[f"{medium}__{refname}"] = cell

json.dump(grid, open(f"{OUT}/grid_bsub.json", "w"), indent=1)

print("\n=== B. subtilis FULL 2x2x3x2 GRID ===")
print(f"{'medium':12s} {'reference':16s} {'predictor':8s} {'arm':8s} {'P':6s} {'R':6s} {'MCC':6s}")
for key, cell in grid.items():
    medium, refname = key.split("__")
    for pred, arms in cell.items():
        for arm, m in arms.items():
            print(f"{medium:12s} {refname:16s} {pred:8s} {arm:8s} {m['precision']:.3f}  {m['recall']:.3f}  {m['MCC']:.3f}")

print("\n=== MEAN-OVER-3-PREDICTORS MCC, B. subtilis 2x2 SUMMARY ===")
summary = {}
print(f"{'medium':12s} {'reference':16s} {'meteor_meanMCC':16s} {'thresh_meanMCC':16s}")
for key, cell in grid.items():
    medium, refname = key.split("__")
    mm = np.mean([cell[p]["meteor"]["MCC"] for p in PREDICTORS])
    tm = np.mean([cell[p]["thresh"]["MCC"] for p in PREDICTORS])
    summary[key] = dict(meteor_mean_mcc=round(float(mm), 3), thresh_mean_mcc=round(float(tm), 3))
    print(f"{medium:12s} {refname:16s} {mm:16.3f} {tm:16.3f}")
json.dump(summary, open(f"{OUT}/summary_bsub_2x2.json", "w"), indent=1)
