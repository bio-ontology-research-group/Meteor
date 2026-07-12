#!/usr/bin/env python3
"""ECE + Brier on the SwissProt 2025+ holdout, v7 hard-biomass MILP, 109-GCF panel.

Per-protein top-1 confidence/correctness for baseline vs METEOR, across the
full 12 baseline x variant grid (CLEAN/DPZ/EnzBERT x vanilla/filt30/filt50/filt70).
Reuses the same seq-hash genome-resolution approach as compute_igrank_v7.py.
"""
import os, glob, json, hashlib
import numpy as np
import pandas as pd
from baseline_io import resolve_baseline_pkl

PA = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026"
MO = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/meteor_out"
RES = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results"

BASELINE_SUF = {"clean": "CLEAN_confidence", "dpz": "DPZ", "enzbert": "enzbert"}
VARIANTS = ["vanilla", "filt30", "filt50", "filt70"]
BASELINES = ["clean", "dpz", "enzbert"]

def seq_hash(s):
    return hashlib.md5(s.strip().upper().encode()).hexdigest()

def resolve_row(df, pid):
    """Exact match first; fall back to prefix match. Some CLEAN baseline
    pkls index by the full FASTA header (e.g. 'WP_000747555.1 MULTISPECIES:
    ATP-grasp domain-containing protein [Bacillus]') instead of the bare
    accession, causing false misses on an exact-match lookup."""
    if pid in df.index:
        return pid
    for i in df.index:
        if str(i).startswith(pid):
            return i
    return None

def ece(confs, correct, n_bins=10):
    if len(confs) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confs, bins[1:-1]), 0, n_bins - 1)
    out, n = 0.0, len(confs)
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        out += (mask.sum() / n) * abs(correct[mask].mean() - confs[mask].mean())
    return float(out)

def brier(confs, correct):
    if len(confs) == 0:
        return float("nan")
    return float(np.mean((confs - correct) ** 2))

def main():
    tsv = pd.read_csv(f"{PA}/benchmark/swissprot_holdout/swissprot_holdout_sample.tsv",
                       sep="\t", dtype=str).fillna("")
    holdout = {}
    for _, r in tsv.iterrows():
        uid = r["Entry"].strip()
        ecs = [e.strip() for e in r.get("EC number", "").split(";")
               if e.strip().count(".") == 3 and "-" not in e.strip()]
        if ecs:
            holdout[uid] = ecs[0]
    uid_to_seq = {r["Entry"].strip(): r["Sequence"].strip().upper()
                  for _, r in tsv.iterrows() if r["Entry"].strip() in holdout}

    GC_DIR = f"{PA}/benchmark/proteomes/genome_collection"
    print("Building seq_hash index...", flush=True)
    hash_to_loc = {}
    for fa in glob.glob(f"{GC_DIR}/*.fasta"):
        gcf = os.path.basename(fa)[:-6]
        pid, seq = None, []
        with open(fa) as f:
            for line in f:
                if line.startswith(">"):
                    if pid and seq:
                        hash_to_loc.setdefault(seq_hash("".join(seq)), (gcf, pid))
                    pid = line[1:].split()[0]
                    seq = []
                else:
                    seq.append(line.strip())
            if pid and seq:
                hash_to_loc.setdefault(seq_hash("".join(seq)), (gcf, pid))
    print(f"  indexed {len(hash_to_loc)} proteins", flush=True)

    uid_loc = {}
    for uid, seq in uid_to_seq.items():
        h = seq_hash(seq)
        if h in hash_to_loc:
            uid_loc[uid] = hash_to_loc[h]
    print(f"resolved {len(uid_loc)}/{len(holdout)} holdout proteins", flush=True)

    rows = []
    os.makedirs(RES, exist_ok=True)
    for baseline in BASELINES:
        suf = BASELINE_SUF[baseline]
        for variant in VARIANTS:
            bdir = f"{PA}/baseline_preds/{baseline}_{variant}_genome_collection"
            mdir = f"{MO}/{baseline}_{variant}"
            if not os.path.isdir(bdir) or not os.path.isdir(mdir):
                print(f"SKIP {baseline}-{variant}: missing dir", flush=True)
                continue
            bcache, mcache = {}, {}
            cb, kb, cm, km = [], [], [], []
            for uid, gt in holdout.items():
                if uid not in uid_loc:
                    continue
                gcf, pid = uid_loc[uid]
                if gcf not in bcache:
                    bp = resolve_baseline_pkl(baseline, variant, gcf, suf)
                    bcache[gcf] = pd.read_pickle(bp) if bp else None
                if gcf not in mcache:
                    mp = f"{mdir}/meteor_df_{gcf}.pkl"
                    mcache[gcf] = pd.read_pickle(mp) if os.path.exists(mp) else None
                bdf, mdf = bcache[gcf], mcache[gcf]
                if bdf is None:
                    continue
                b_key = resolve_row(bdf, pid)
                if b_key is None:
                    continue
                brow = bdf.loc[b_key]
                bcols = {str(c).replace("EC:", "").strip(): c for c in bdf.columns}
                top_ec_b = brow.idxmax()
                top_ec_b_clean = str(top_ec_b).replace("EC:", "").strip()
                cb.append(float(brow.max()))
                kb.append(1 if top_ec_b_clean == gt else 0)
                m_key = resolve_row(mdf, pid) if mdf is not None else None
                if m_key is not None:
                    mrow = mdf.loc[m_key]
                    top_ec_m = mrow.idxmax()
                    top_ec_m_clean = str(top_ec_m).replace("EC:", "").strip()
                    cm.append(float(mrow.max()))
                    km.append(1 if top_ec_m_clean == gt else 0)
            cb, kb, cm, km = map(np.array, (cb, kb, cm, km))
            e_b, e_m = ece(cb, kb), ece(cm, km)
            br_b, br_m = brier(cb, kb), brier(cm, km)
            name = f"{baseline}-{variant}"
            rows.append(dict(pair=name, n_b=len(cb), n_m=len(cm),
                              ece_b=e_b, ece_m=e_m, delta_ece=e_m - e_b,
                              brier_b=br_b, brier_m=br_m, delta_brier=br_m - br_b))
            print(f"  {name:16s} n_b={len(cb):3d} n_m={len(cm):3d}  "
                  f"ECE {e_b:.3f}->{e_m:.3f} ({e_m-e_b:+.3f})  "
                  f"Brier {br_b:.3f}->{br_m:.3f} ({br_m-br_b:+.3f})", flush=True)

    df = pd.DataFrame(rows)
    out = f"{RES}/calibration_v7.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"\nSaved {out}")
    if not df.empty:
        print(f"Mean ECE: {df.ece_b.mean():.3f} -> {df.ece_m.mean():.3f} "
              f"(Delta={df.delta_ece.mean():+.3f}); improved {int((df.delta_ece<0).sum())}/{len(df)}")
        print(f"Mean Brier: {df.brier_b.mean():.3f} -> {df.brier_m.mean():.3f} "
              f"(Delta={df.delta_brier.mean():+.3f}); improved {int((df.delta_brier<0).sum())}/{len(df)}")

if __name__ == "__main__":
    main()
