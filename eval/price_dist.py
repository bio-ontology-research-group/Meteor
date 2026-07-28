#!/usr/bin/env python3
"""Price-149 predictor output-distribution characterization (net-only, shared set).
For each of the 6 predictors: score sparsity/peakiness of the BASELINE preds, and
how many ECs METEOR mutes vs boosts per protein -> explains heterogeneous Fmax effect."""
import os, glob, numpy as np, pandas as pd
import sys as _s2; _s2.path.insert(0,"/ibex/user/niuk0a/funcarve/cobra/v6")
from src.v6utils import extract_pred as _XP, load_ec as _LE
_ANC=_LE("/ibex/user/niuk0a/funcarve/cobra/v6/data/all_ancestors.txt")
F="/ibex/scratch/projects/c2014/kexin/funcarve"; ECONTO="/ibex/user/niuk0a/funcarve/econto"
MOP=f"{F}/meteor_v8_evw_p2mu3_run/meteor_out_price"

def base_pred_path(baseline, gca):
    gns=gca.rsplit(".",1)[0]; num=gca.split("_")[1].split(".")[0]
    m={"clean":f"{F}/paperA_2026/baseline_preds/price22_CLEAN_clean/{gca}/{gca}_CLEAN_confidence.pkl",
       "enzbert":f"{F}/tfpc/resultprice_newg/{gca}_enzbert_predictions.pkl",
       "graphec":f"{F}/graphec_price_new/{gca}_GraphEC.pkl",
       "mapred":f"{F}/mapred_price_new/{gca}_MAPred.pkl",
       "topec":f"{F}/topec_price_new/{gca}_TopEC.pkl"}
    if baseline=="dpz":
        exact=f"{F}/dpec2_result/result_price/{gns}_DeepECv2_t5.pkl"
        if os.path.exists(exact): return exact
        h=glob.glob(f"{F}/dpec2_result/result_price/GC?_{num}_DeepECv2_t5.pkl"); return h[0] if h else ""
    p=m[baseline]
    if os.path.exists(p): return p
    for c in (p,p.replace(gca,gns)):
        if os.path.exists(c): return c
    return p

def lookup(df, pid, nid=None):
    if df is None: return None
    for x in [pid]+([nid] if nid else []):
        if x in df.index: return {str(k).replace("EC:","").strip():float(v) for k,v in df.loc[x].to_dict().items()}
        b=str(x).rsplit(".",1)[0]; mt=[i for i in df.index if i==b or str(i).startswith(b+".")]
        if mt: return {str(k).replace("EC:","").strip():float(v) for k,v in df.loc[mt[0]].to_dict().items()}
    return None

price=pd.read_csv(f"{ECONTO}/data/test/price.csv",sep="\t")
gt_ecs={r["Entry"]:{e.strip() for e in str(r["EC number"]).split(";") if e.strip().count(".")==3 and "-" not in e.strip()} for _,r in price.iterrows()}
gt_ecs={k:v for k,v in gt_ecs.items() if v}
meta=pd.read_csv(f"{ECONTO}/data/processed/geno/price_proteomes/Price_genome_metainfo.csv")
meta_map={r["input_id"]:(r["protein_id"],r.get("nucleotide_id"),r["assembly"]) for _,r in meta.iterrows()}

rows=[]
for b in ["clean","dpz","enzbert","graphec","mapred","topec"]:
    bcache,mcache={},{}
    support=[]; npos=[]; maxsc=[]; gap=[]; muted=[]; boosted=[]; n=0
    for ent,gt in gt_ecs.items():
        if ent not in meta_map: continue
        pid,nid,asm=meta_map[ent]
        if asm not in bcache:
            bp=base_pred_path(b,asm); bcache[asm]=_XP(bp,_ANC) if bp and os.path.exists(bp) else None
        if asm not in mcache:
            mp=f"{MOP}/{b}/meteor_df_{asm}.pkl"; mcache[asm]=pd.read_pickle(mp) if os.path.exists(mp) else None
        sb=lookup(bcache[asm],pid,nid); sm=lookup(mcache[asm],pid,nid)
        if not (sb and sm): continue
        n+=1
        vals=sorted(sb.values(),reverse=True)
        support.append(sum(1 for v in sb.values() if v>0))
        npos.append(sum(1 for v in sb.values() if v>=0.5))
        maxsc.append(vals[0] if vals else 0.0)
        gap.append((vals[0]-vals[1]) if len(vals)>1 else vals[0] if vals else 0.0)
        # METEOR mute/boost: ECs whose score dropped / rose vs baseline
        mu=bo=0
        for ec,vb in sb.items():
            vm=sm.get(ec,vb)
            if vm<vb-1e-9: mu+=1
            elif vm>vb+1e-9: bo+=1
        muted.append(mu); boosted.append(bo)
    r=dict(baseline=b,n=n,
           mean_support=round(float(np.mean(support)),1),
           mean_npos05=round(float(np.mean(npos)),2),
           mean_maxscore=round(float(np.mean(maxsc)),3),
           mean_top1_top2_gap=round(float(np.mean(gap)),3),
           frac_confident=round(float(np.mean([1 if m>=0.5 else 0 for m in maxsc])),3),
           mean_muted_per_prot=round(float(np.mean(muted)),2),
           mean_boosted_per_prot=round(float(np.mean(boosted)),2))
    rows.append(r)
    print(f"  {b:8s} n={n} support={r['mean_support']} npos.5={r['mean_npos05']} "
          f"maxsc={r['mean_maxscore']} gap={r['mean_top1_top2_gap']} conf={r['frac_confident']} "
          f"muted/prot={r['mean_muted_per_prot']} boost/prot={r['mean_boosted_per_prot']}",flush=True)
out=pd.DataFrame(rows)
o=f"{F}/meteor_v8/results/toolcompare/price149_distribution.tsv"; os.makedirs(os.path.dirname(o),exist_ok=True)
out.to_csv(o,sep="\t",index=False); print("saved",o)
