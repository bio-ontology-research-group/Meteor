"""Round 5: pure re-scoring, NO MILP re-solve, NO COBRA rebuild. Reuses the
single_gene_deletion per-gene calls already saved in results/essround3/sgd_*.json
from round 3 (GS_MM_glc network, organism="ecoli") and round 4 (LB_marinos
network, organism="ecoli_lb"), and re-scores each against BOTH reference
tables (gess-ecol.csv and PEC) to fill in a medium x reference 2x2 grid.
Writes only under psb_revision/feasibility_essentiality/results/rescoring2x2/.
"""
import json, csv, os
import numpy as np

HERE = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/feasibility_essentiality"
OUT = f"{HERE}/results/rescoring2x2"; os.makedirs(OUT, exist_ok=True)

# protein -> b-number mapper (same RBH mapping used for both networks; MG1655 b-number space)
p2ref = {}
for l in open(f"{HERE}/ref/m25631h_to_bnum.tsv"):
    parts = l.rstrip("\n").split("\t")
    if len(parts) >= 3: p2ref[parts[0]] = parts[2]

def load_ref(csv_path, gene_col):
    ref = {}
    for row in csv.DictReader(open(csv_path)):
        ref[str(row[gene_col])] = (row["ess.experimental"] == "yes")
    return ref

REF = {
    "gess-ecol.csv": load_ref("/ibex/scratch/projects/c2014/kexin/funcarve/gapseq_eval/gapseqEval/GeneEssentiality/essentiality.data/gess-ecol.csv", "gene"),
    "PEC": load_ref(f"{HERE}/ref/pec/pec_ecoli_binary.csv", "gene"),
}
print("gess-ecol.csv:", sum(REF["gess-ecol.csv"].values()), "essential /", len(REF["gess-ecol.csv"]), "total")
print("PEC:", sum(REF["PEC"].values()), "essential /", len(REF["PEC"]), "total")

def metrics(pred_ess: dict, ref_ess: dict, mapper=p2ref.get):
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

NETWORKS = {"GS_MM_glc": "ecoli", "LB_marinos": "ecoli_lb"}
PREDICTORS = ["clean", "dpz", "enzbert"]
ARMS = ["meteor", "thresh"]

grid = {}  # (medium, refname) -> {predictor: {arm: metrics}}
for medium, org in NETWORKS.items():
    for refname, ref_ess in REF.items():
        cell = {}
        for pred in PREDICTORS:
            cell[pred] = {}
            for arm in ARMS:
                fpath = f"{HERE}/results/essround3/sgd_{org}_{arm}_{pred}.json"
                calls = json.load(open(fpath))
                calls = {g: bool(e) for g, e in calls.items()}
                m = metrics(calls, ref_ess)
                cell[pred][arm] = m
        grid[f"{medium}__{refname}"] = cell

json.dump(grid, open(f"{OUT}/grid.json", "w"), indent=1)

print("\n=== FULL 2x2x3x2 GRID ===")
print(f"{'medium':12s} {'reference':14s} {'predictor':8s} {'arm':8s} {'P':6s} {'R':6s} {'MCC':6s}")
for key, cell in grid.items():
    medium, refname = key.split("__")
    for pred, arms in cell.items():
        for arm, m in arms.items():
            print(f"{medium:12s} {refname:14s} {pred:8s} {arm:8s} {m['precision']:.3f}  {m['recall']:.3f}  {m['MCC']:.3f}")

print("\n=== MEAN-OVER-3-PREDICTORS MCC, 2x2 SUMMARY ===")
summary = {}
print(f"{'medium':12s} {'reference':14s} {'meteor_meanMCC':16s} {'thresh_meanMCC':16s}")
for key, cell in grid.items():
    medium, refname = key.split("__")
    meteor_mccs = [cell[p]["meteor"]["MCC"] for p in PREDICTORS]
    thresh_mccs = [cell[p]["thresh"]["MCC"] for p in PREDICTORS]
    mm, tm = np.mean(meteor_mccs), np.mean(thresh_mccs)
    summary[key] = dict(meteor_mean_mcc=round(float(mm), 3), thresh_mean_mcc=round(float(tm), 3))
    print(f"{medium:12s} {refname:14s} {mm:16.3f} {tm:16.3f}")
json.dump(summary, open(f"{OUT}/summary_2x2.json", "w"), indent=1)
print("\n->", f"{OUT}/grid.json", f"{OUT}/summary_2x2.json")
