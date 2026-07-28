#!/usr/bin/env python3
"""Price-149 per-protein eval (v7): baseline vs METEOR top-1 / Fmax for all 6
predictors, on the shared (both-scored) protein set. Reads v7 METEOR outputs
from meteor_out_price/{baseline}/meteor_df_{gca}.pkl."""
import os, glob
from pathlib import Path
import pandas as pd

F = "/ibex/scratch/projects/c2014/kexin/funcarve"
ECONTO = "/ibex/user/niuk0a/funcarve/econto"
MOP = f"{F}/meteor_v8_evw_p2mu3_run/meteor_out_price"


def base_pred_path(baseline, gca):
    gns = gca.rsplit(".", 1)[0]
    num = gca.split("_")[1].split(".")[0]
    m = {
        "clean":   f"{F}/paperA_2026/baseline_preds/price22_CLEAN_clean/{gca}/{gca}_CLEAN_confidence.pkl",
        "enzbert": f"{F}/tfpc/resultprice_newg/{gca}_enzbert_predictions.pkl",
        "graphec": f"{F}/graphec_price_new/{gca}_GraphEC.pkl",
        "mapred":  f"{F}/mapred_price_new/{gca}_MAPred.pkl",
        "topec":   f"{F}/topec_price_new/{gca}_TopEC.pkl",
    }
    if baseline == "dpz":
        exact = f"{F}/dpec2_result/result_price/{gns}_DeepECv2_t5.pkl"
        if os.path.exists(exact):
            return exact
        h = glob.glob(f"{F}/dpec2_result/result_price/GC?_{num}_DeepECv2_t5.pkl")
        return h[0] if h else ""
    p = m[baseline]
    if os.path.exists(p):
        return p
    for c in (p, p.replace(gca, gns)):
        if os.path.exists(c):
            return c
    return p


def top1(gt, sc):
    if not gt or not sc: return None
    return 1 if max(sc, key=sc.get) in gt else 0


def topk(gt, sc, k):
    if not gt or not sc: return None
    return 1 if any(e in gt for e in sorted(sc, key=sc.get, reverse=True)[:k]) else 0


def fmax(gt, sc):
    if not gt or not sc: return None
    best = 0.0
    for t in sorted(set(sc.values()), reverse=True):
        pred = {e for e, v in sc.items() if v >= t}
        if not pred: continue
        tp = len(pred & gt); pr = tp / len(pred); rc = tp / len(gt)
        if pr + rc > 0: best = max(best, 2 * pr * rc / (pr + rc))
    return best


def lookup(df, pid, nid=None):
    if df is None: return None
    for x in [pid] + ([nid] if nid else []):
        if x in df.index: return {str(k).replace("EC:", "").strip(): v for k, v in df.loc[x].to_dict().items()}
        b = str(x).rsplit(".", 1)[0]
        mt = [i for i in df.index if i == b or str(i).startswith(b + ".")]
        if mt: return {str(k).replace("EC:", "").strip(): v for k, v in df.loc[mt[0]].to_dict().items()}
    return None


price = pd.read_csv(f"{ECONTO}/data/test/price.csv", sep="\t")
gt_ecs = {r["Entry"]: {e.strip() for e in str(r["EC number"]).split(";")
                       if e.strip().count(".") == 3 and "-" not in e.strip()} for _, r in price.iterrows()}
gt_ecs = {k: v for k, v in gt_ecs.items() if v}
meta = pd.read_csv(f"{ECONTO}/data/processed/geno/price_proteomes/Price_genome_metainfo.csv")
meta_map = {r["input_id"]: (r["protein_id"], r.get("nucleotide_id"), r["assembly"]) for _, r in meta.iterrows()}
print(f"GT proteins (4-digit): {len(gt_ecs)}")

rows = []
for b in ["clean", "dpz", "enzbert", "graphec", "mapred", "topec"]:
    bcache, mcache = {}, {}
    t1b = t1m = t5b = t5m = 0; fmb = fmm = 0.0; n = 0
    for ent, gt in gt_ecs.items():
        if ent not in meta_map: continue
        pid, nid, asm = meta_map[ent]
        if asm not in bcache:
            bp = base_pred_path(b, asm)
            bcache[asm] = pd.read_pickle(bp) if bp and os.path.exists(bp) else None
        if asm not in mcache:
            mp = f"{MOP}/{b}/meteor_df_{asm}.pkl"
            mcache[asm] = pd.read_pickle(mp) if os.path.exists(mp) else None
        sb = lookup(bcache[asm], pid, nid)
        sm = lookup(mcache[asm], pid, nid)
        if not (sb and sm): continue          # fair: shared set only
        n += 1
        t1b += top1(gt, sb); t1m += top1(gt, sm)
        t5b += topk(gt, sb, 5); t5m += topk(gt, sm, 5)
        fmb += fmax(gt, sb); fmm += fmax(gt, sm)
    row = dict(baseline=b, n=n, top1_B=t1b, top1_M=t1m, top5_B=t5b, top5_M=t5m,
               Fmax_B=round(fmb / n, 3) if n else 0, Fmax_M=round(fmm / n, 3) if n else 0)
    rows.append(row)
    print(f"  {b:8s}: n={n}  top1 {t1b}->{t1m}  top5 {t5b}->{t5m}  Fmax {row['Fmax_B']}->{row['Fmax_M']}", flush=True)

out = pd.DataFrame(rows)
o = f"{F}/meteor_v8_evw_p2mu3_run/downstream_results/price149_v8.tsv"
out.to_csv(o, sep="\t", index=False)
print("saved", o)
