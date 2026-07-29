#!/usr/bin/env python3
"""NET-ONLY seqident (fair T15): per-protein top-1 correction/regression binned by max identity to the DPZ
training set. Base loaded via extract_pred (net-only, same vocab as meteor_df). Answers: under the fair
comparison, is the <30%-identity bin still net-positive (corrections>regressions)?"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import os,json,pickle,sys,numpy as np,pandas as pd
W="/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026"
V7="/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run"
CACHE=f"{V7}/downstream_results/ncbi_ec_cache"
MDIR="/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla"
OUT="/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/results/toolcompare"
sys.path.insert(0,"/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/eval"); from baseline_io import resolve_baseline_pkl
from meteor_v8.utils import extract_pred as XP, load_ec as LE
ANC=LE(data_path('all_ancestors.txt'))
# per-protein max identity to training
pid2id={}
for ln in open(f"{V7}/downstream_results/seq_identity_dpz/diamond_hits.tsv"):
    p=ln.split("\t")
    if len(p)<3: continue
    q=p[0];
    try: pi=float(p[2])
    except: continue
    if pi>pid2id.get(q,-1): pid2id[q]=pi
def binof(pid):
    if pid not in pid2id: return "no hit"
    v=pid2id[pid]
    return "<30%" if v<30 else "30-50%" if v<50 else "50-70%" if v<70 else "70-90%" if v<90 else ">90%"
BINS=["<30%","30-50%","50-70%","70-90%",">90%","no hit"]
agg={b:dict(n=0,bc=0,mc=0,corr=0,regr=0) for b in BINS}
manifest=json.load(open(f"{W}/scripts/genome_collection_manifest.json"))
gcfs=sorted(g for g in manifest["assemblies"] if g.startswith("GCF"))
tot=0
for gi,gcf in enumerate(gcfs):
    cf=f"{CACHE}/{gcf}_ec.json"; mp=f"{MDIR.replace('meteor_out/dpz_vanilla','meteor_out/dpz_vanilla')}/meteor_df_{gcf}.pkl"
    mp=f"/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla/meteor_df_{gcf}.pkl"
    if not (os.path.exists(cf) and os.path.exists(mp)): continue
    bp=resolve_baseline_pkl("dpz","vanilla",gcf)
    if not bp: continue
    ecm={k:set(v) for k,v in json.load(open(cf)).items()}
    if not ecm: continue
    try: bdf=XP(bp,ANC); mdf=pickle.load(open(mp,"rb"))
    except: continue
    if any(str(c).startswith("EC:") for c in bdf.columns[:5]): bdf.columns=[str(c).replace("EC:","") for c in bdf.columns]
    bcols=list(bdf.columns); mcols=list(mdf.columns)
    for pid,gt in ecm.items():
        if pid not in bdf.index or pid not in mdf.index: continue
        bh=bcols[int(np.argmax(bdf.loc[pid].values))] in gt
        mh=mcols[int(np.argmax(mdf.loc[pid].values))] in gt
        b=binof(pid); a=agg[b]; a['n']+=1; a['bc']+=int(bh); a['mc']+=int(mh)
        if (not bh) and mh: a['corr']+=1
        if bh and (not mh): a['regr']+=1
        tot+=1
    if gi%20==0: print(f"[{gi+1}/{len(gcfs)}] tot={tot}",flush=True)
json.dump(agg,open(f"{OUT}/net_seqident.json","w"),indent=1)
print(f"\n=== NET-ONLY seqident (T15), n={tot} ===")
print(f"{'bin':8s} {'n':>7s} {'baseTop1':>8s} {'metTop1':>8s} {'corr':>5s} {'regr':>5s} {'net':>5s} {'corr/1k':>8s}")
for b in BINS:
    a=agg[b]
    if a['n']==0: continue
    print(f"{b:8s} {a['n']:7d} {100*a['bc']/a['n']:7.1f}% {100*a['mc']/a['n']:7.1f}% {a['corr']:5d} {a['regr']:5d} {a['corr']-a['regr']:+5d} {1000*a['corr']/a['n']:7.1f}")
print('-> results/toolcompare/net_seqident.json')
