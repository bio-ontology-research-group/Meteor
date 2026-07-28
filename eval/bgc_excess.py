#!/usr/bin/env python3
"""Per-baseline BGC excess-recall (producer minus non-producer indicator-EC recall),
baseline vs METEOR (v8 active_ecs). Resolves the main-text sign claim."""
import argparse, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
SUFFIX = {"clean": "CLEAN_confidence", "dpz": "DPZ", "enzbert": "enzbert"}

def mean_recall(gcfs, ind, cache):
    n = len(ind)
    return np.mean([len(ind & cache.get(g, set())) / n for g in gcfs]) if gcfs else 0.0

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True); ap.add_argument("--baseline_dir", required=True)
ap.add_argument("--meteor_dir", required=True); ap.add_argument("--class_ec", required=True)
ap.add_argument("--labels", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--tau", type=float, default=0.5)
a = ap.parse_args(); suf = SUFFIX[a.baseline]
class_ec = json.loads(Path(a.class_ec).read_text())["class_ec"]
labels = pd.read_csv(a.labels, sep="\t", dtype={"gcf": str}).set_index("gcf")
bdir, mdir = Path(a.baseline_dir), Path(a.meteor_dir)
gcf_set = sorted({p.name[:-(len(suf)+5)] for p in bdir.glob(f"*_{suf}.pkl")}
                 & {p.name[len("meteor_preds_"):-4] for p in mdir.glob("meteor_preds_*.pkl")}
                 & set(labels.index))
print(f"[{a.baseline}] genomes: {len(gcf_set)}")
bc, mc = {}, {}
for g in gcf_set:
    df = pickle.load(open(bdir / f"{g}_{suf}.pkl", "rb"))
    df.columns = [str(c).replace("EC:", "") for c in df.columns]
    bc[g] = set(df.columns[df.max(axis=0) > a.tau])
    mc[g] = set(pickle.load(open(mdir / f"meteor_preds_{g}.pkl", "rb"))["active_ecs"])
rows = []
for cls in class_ec:
    ind = set(class_ec[cls])
    if not ind or cls not in labels.columns: continue
    prod = [g for g in gcf_set if labels.loc[g, cls] == 1]
    nonp = [g for g in gcf_set if labels.loc[g, cls] == 0]
    if len(prod) < 2 or not nonp: continue
    exB = mean_recall(prod, ind, bc) - mean_recall(nonp, ind, bc)
    exM = mean_recall(prod, ind, mc) - mean_recall(nonp, ind, mc)
    rows.append(dict(bgc_class=cls, n_prod=len(prod), n_nonp=len(nonp),
                     excess_B=round(exB, 4), excess_M=round(exM, 4), excess_delta=round(exM - exB, 4)))
    print(f"  {cls:11s} excess {exB:+.4f} -> {exM:+.4f} (delta {exM-exB:+.4f})")
out = pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(a.out, sep="\t", index=False); print("saved", a.out)
