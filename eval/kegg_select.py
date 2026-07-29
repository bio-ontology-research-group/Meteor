#!/usr/bin/env python3
"""Selectivity of METEOR's weak-signal recovery on panel108. Shows METEOR does NOT adopt every EC with
score>0. Per baseline (mean over 108 genomes):
  weak_ec_total  = # SEED ECs with 0<baseline<0.5 (weak signal)
  weak_ec_adopt  = # of those activated by METEOR;  adopt_rate = adopt/total
  wpath_avail    = # pathways baseline-missed (no EC>=0.5) but with >=1 weak EC (recoverable by weak signal)
  wpath_recov    = # of those METEOR detects (>=1 EC);  recov_rate = recov/avail
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys,os,re,json,pickle,numpy as np
from meteor_v8.utils import extract_pred,load_refmapping,load_ec

from baseline_io import resolve_baseline_pkl,BASELINE_SUFFIX
B='/ibex/scratch/projects/c2014/kexin/funcarve'; FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')
V8ROOT=f'{B}/meteor_v8_evw_p2mu3_run/meteor_out'; OUT=f'{B}/meteor_v8/results/toolcompare'
anc=load_ec(data_path('all_ancestors.txt'))
seedr2ec,_=load_refmapping(data_dir()); seedr2ec={k:v for k,v in seedr2ec.items() if v}
seed_ec=set()
for ecs in seedr2ec.values():
    for e in ecs:
        e=str(e).split(':')[-1]
        if FULL.match(e): seed_ec.add(e)
path2ec=json.load(open(f'{B}/meteor_v7_release/data/kegg_path2ec_metabolic.json'))
pw_denom={}
for p,ecs in path2ec.items():
    d={str(e).split(':')[-1] for e in ecs if FULL.match(str(e).split(':')[-1])} & seed_ec
    if len(d)>=5: pw_denom[p]=d
panel108=[l.split()[0] for l in open(f'{B}/meteor_v7_run/downstream_results/panel108_gram.tsv') if l.strip()]
print(f'pathways={len(pw_denom)} genomes={len(panel108)}',flush=True)

def base_scores(baseline,gcf):
    p=extract_pred(resolve_baseline_pkl(baseline,'vanilla',gcf,BASELINE_SUFFIX[baseline]),anc)
    cols=[str(c).split(':')[-1] for c in p.columns]; mx=p.values.max(axis=0)
    return {cols[j]:float(mx[j]) for j in range(len(cols)) if FULL.match(cols[j]) and cols[j] in seed_ec}
def meteor_ecs(baseline,gcf):
    f=f'{V8ROOT}/{baseline}_vanilla/meteor_preds_{gcf}.pkl'
    if not os.path.exists(f): return None
    ae=pickle.load(open(f,'rb')).get('active_ecs',set())
    return {str(e).split(':')[-1] for e in ae if FULL.match(str(e).split(':')[-1]) and str(e).split(':')[-1] in seed_ec}

res={}
print(f'{"baseline":9s} n  weakEC(tot->adopt=rate) | wPath(avail->recov=rate)')
for baseline in ['clean','dpz','enzbert']:
    WT=[];WA=[];PAV=[];PRC=[]
    for gcf in panel108:
        M=meteor_ecs(baseline,gcf)
        if M is None: continue
        try: sc=base_scores(baseline,gcf)
        except Exception: continue
        weak={e for e,v in sc.items() if 0.0<v<0.5}
        WT.append(len(weak)); WA.append(len(weak&M))
        strong={e for e,v in sc.items() if v>=0.5}
        avail=recov=0
        for p,denom in pw_denom.items():
            if len(strong&denom)==0 and len(weak&denom)>=1:   # baseline-missed but weak-recoverable
                avail+=1
                if len(M&denom)>=1: recov+=1
        PAV.append(avail); PRC.append(recov)
    wt=float(np.mean(WT)); wa=float(np.mean(WA)); pav=float(np.mean(PAV)); prc=float(np.mean(PRC))
    r=dict(n=len(WT),weak_ec_total=round(wt,1),weak_ec_adopt=round(wa,1),
           weak_ec_adopt_rate=round(wa/wt,3) if wt else None,
           wpath_avail=round(pav,1),wpath_recov=round(prc,1),
           wpath_recov_rate=round(prc/pav,3) if pav else None)
    res[baseline]=r
    print(f'{baseline:9s} {r["n"]:3d}  {r["weak_ec_total"]:.0f}->{r["weak_ec_adopt"]:.0f}='
          f'{r["weak_ec_adopt_rate"]:.1%} | {r["wpath_avail"]:.1f}->{r["wpath_recov"]:.1f}='
          f'{r["wpath_recov_rate"]:.1%}',flush=True)
json.dump(dict(panel='panel108',n_pathways=len(pw_denom),results=res),
          open(f'{OUT}/kegg_pathway_108_selectivity.json','w'),indent=1)
print('-> kegg_pathway_108_selectivity.json',flush=True)
