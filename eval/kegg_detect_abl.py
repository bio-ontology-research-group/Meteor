#!/usr/bin/env python3
"""Pathway detection for ONE ablation condition (dpz baseline, given meteor run dir).
Same metric as kegg_detect.py; used for S3 system-level ablation (B2 nobio, uniform mu3e0, ...)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys,os,re,json,pickle,argparse,numpy as np
from meteor_v8.utils import extract_pred,load_refmapping,load_ec

from baseline_io import resolve_baseline_pkl,BASELINE_SUFFIX
ap=argparse.ArgumentParser(); ap.add_argument('--mo',required=True); ap.add_argument('--tag',required=True)
ap.add_argument('--baseline',default='dpz'); a=ap.parse_args()
B='/ibex/scratch/projects/c2014/kexin/funcarve'; FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')
OUT=f'{B}/meteor_v8/results/toolcompare'
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
NPW=len(pw_denom)
panel108=[l.split()[0] for l in open(f'{B}/meteor_v7_run/downstream_results/panel108_gram.tsv') if l.strip()]
print(f'{a.tag}: pathways={NPW} genomes={len(panel108)} mo={a.mo}',flush=True)
def base_scores(gcf):
    p=extract_pred(resolve_baseline_pkl(a.baseline,'vanilla',gcf,BASELINE_SUFFIX[a.baseline]),anc)
    cols=[str(c).split(':')[-1] for c in p.columns]; mx=p.values.max(axis=0)
    return {cols[j]:float(mx[j]) for j in range(len(cols)) if FULL.match(cols[j]) and cols[j] in seed_ec}
def meteor_ecs(gcf):
    f=f'{a.mo}/meteor_preds_{gcf}.pkl'
    if not os.path.exists(f): return None
    ae=pickle.load(open(f,'rb')).get('active_ecs',set())
    return {str(e).split(':')[-1] for e in ae if FULL.match(str(e).split(':')[-1]) and str(e).split(':')[-1] in seed_ec}
D1b=[];D1m=[];D50b=[];D50m=[];MO=[];WS=[];PG=[];WT=[];WA=[]
for gcf in panel108:
    M=meteor_ecs(gcf)
    if M is None: continue
    try: sc=base_scores(gcf)
    except Exception: continue
    Bset={e for e,v in sc.items() if v>=0.5}; weak={e for e,v in sc.items() if 0.0<v<0.5}
    WT.append(len(weak)); WA.append(len(weak&M))
    d1b=d1m=d50b=d50m=0; mo=ws=pg=0
    for p,denom in pw_denom.items():
        n=len(denom); ob=len(Bset&denom); om=len(M&denom)
        d1b+=ob>=1; d1m+=om>=1; d50b+=ob/n>=0.5; d50m+=om/n>=0.5
        if om>=1 and ob==0:
            mo+=1
            if any(sc.get(e,0.0)>0.0 for e in (M&denom)): ws+=1
            else: pg+=1
    D1b.append(d1b);D1m.append(d1m);D50b.append(d50b);D50m.append(d50m);MO.append(mo);WS.append(ws);PG.append(pg)
def m(x): return round(float(np.mean(x)),1)
r=dict(tag=a.tag,n=len(D1b),det1_B=m(D1b),det1_M=m(D1m),det50_B=m(D50b),det50_M=m(D50m),
       Monly=m(MO),weak=m(WS),gap=m(PG),weak_ec_total=m(WT),weak_ec_adopt=m(WA),
       weak_ec_rate=round(float(np.mean(WA))/float(np.mean(WT)),3) if np.mean(WT) else None)
json.dump(r,open(f'{OUT}/abl_detect_{a.tag}.json','w'),indent=1)
print(f"{a.tag}: det>=1 {r['det1_B']}->{r['det1_M']}  det>=50 {r['det50_B']}->{r['det50_M']}  "
      f"Monly {r['Monly']}(weak {r['weak']}/gap {r['gap']})  weakEC {r['weak_ec_adopt']}/{r['weak_ec_total']}={r['weak_ec_rate']}",flush=True)
