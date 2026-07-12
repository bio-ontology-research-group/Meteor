#!/usr/bin/env python3
"""BGC indicator-EC producer/non-producer specificity: baseline vs METEOR (v7),
with antiSMASH-called BGC regions on the SAME assemblies as genome-identical
ground truth.

Per BGC class it computes producer recall, non-producer recall, non-producer
any-hit FPR, specificity (1-FPR) and excess recall (producer - non-producer)
for both the raw baseline (max EC conf > tau) and METEOR's active-EC set.

Publication-ready, parameterized. Required public data:
  --class_ec     KEGG-derived {class: [indicator ECs]} JSON (key 'class_ec')
  --labels       antiSMASH producer labels TSV: gcf + one 0/1 column per class
  --baseline_dir {gcf}_{suffix}.pkl   ([P x EC] DataFrame)
  --meteor_dir   meteor_preds_{gcf}.pkl   (dict with active_ecs)
Classes with 0 producers or 0 non-producers in the panel are skipped (undefined).
"""
import argparse, json, pickle
from pathlib import Path
import pandas as pd

SUFFIX = {"clean": "CLEAN_confidence", "dpz": "DPZ", "enzbert": "enzbert"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", choices=list(SUFFIX), required=True)
    ap.add_argument("--variant", default="vanilla")
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--meteor_dir", required=True)
    ap.add_argument("--class_ec", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args()
    suf = SUFFIX[a.baseline]

    class_ec = json.loads(Path(a.class_ec).read_text())["class_ec"]
    labels = pd.read_csv(a.labels, sep="\t", dtype={"gcf": str}).set_index("gcf")

    bdir, mdir = Path(a.baseline_dir), Path(a.meteor_dir)
    gcf_set = sorted(
        {p.name[:-(len(suf) + 5)] for p in bdir.glob(f"*_{suf}.pkl")}
        & {p.name[len("meteor_preds_"):-4] for p in mdir.glob("meteor_preds_*.pkl")}
        & set(labels.index))
    print(f"genomes (baseline+METEOR+antiSMASH): {len(gcf_set)}")

    base_cache, meteor_cache = {}, {}
    for gcf in gcf_set:
        df = pickle.load(open(bdir / f"{gcf}_{suf}.pkl", "rb"))
        df.columns = [str(c).replace("EC:", "") for c in df.columns]
        base_cache[gcf] = set(df.columns[df.max(axis=0) > a.tau])
        p = pickle.load(open(mdir / f"meteor_preds_{gcf}.pkl", "rb"))
        meteor_cache[gcf] = set(p["active_ecs"])

    rows = []
    for cls, ind in ((c, set(class_ec.get(c, []))) for c in class_ec):
        if not ind or cls not in labels.columns:
            continue
        prod = [g for g in gcf_set if labels.loc[g, cls] == 1]
        nonp = [g for g in gcf_set if labels.loc[g, cls] == 0]
        if not prod or not nonp:
            print(f"  skip {cls}: producers={len(prod)} nonproducers={len(nonp)}")
            continue
        mrec = lambda gs, c: sum(len(ind & c[g]) / len(ind) for g in gs) / len(gs)
        fpr = lambda gs, c: sum(1 for g in gs if ind & c[g]) / len(gs)
        b_pr, m_pr = mrec(prod, base_cache), mrec(prod, meteor_cache)
        b_nr, m_nr = mrec(nonp, base_cache), mrec(nonp, meteor_cache)
        b_fpr, m_fpr = fpr(nonp, base_cache), fpr(nonp, meteor_cache)
        rows.append(dict(bgc_class=cls, n_indicator_ecs=len(ind),
                         n_producer=len(prod), n_nonproducer=len(nonp),
                         producer_recall_baseline=round(b_pr, 4), producer_recall_meteor=round(m_pr, 4),
                         producer_recall_delta=round(m_pr - b_pr, 4),
                         nonproducer_fpr_baseline=round(b_fpr, 4), nonproducer_fpr_meteor=round(m_fpr, 4),
                         specificity_baseline=round(1 - b_fpr, 4), specificity_meteor=round(1 - m_fpr, 4),
                         excess_recall_baseline=round(b_pr - b_nr, 4),
                         excess_recall_meteor=round(m_pr - m_nr, 4)))
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, sep="\t", index=False)
    print(out.to_string(index=False))
    print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    main()
