#!/usr/bin/env python3
"""Price-149 pathway matched-completeness (K12) + detection, for the 3 additional predictors
(GraphEC, MAPred, TopEC) on the 22 Price genomes. Mirrors kegg_matched.py + kegg_detect.py but
on the Price data layout (baseline via base_pred_path, METEOR via meteor_out_price active_ecs)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys,os,re,glob,json,pickle,numpy as np
from meteor_v8.utils import extract_pred,load_refmapping,load_ec
anc=load_ec(data_path('all_ancestors.txt'))
F='/ibex/scratch/projects/c2014/kexin/funcarve'; FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')
MOP=f'{F}/meteor_v8_evw_p2mu3_run/meteor_out_price'; OUT=f'{F}/meteor_v8/results/toolcompare'
seedr2ec,_=load_refmapping(data_dir()); seedr2ec={k:v for k,v in seedr2ec.items() if v}
seed_ec=set()
for ecs in seedr2ec.values():
    for e in ecs:
        e=str(e).split(':')[-1]
        if FULL.match(e): seed_ec.add(e)
path2ec=json.load(open(f'{F}/meteor_v7_release/data/kegg_path2ec_metabolic.json'))
pw_denom={}
for p,ecs in path2ec.items():
    d={str(e).split(':')[-1] for e in ecs if FULL.match(str(e).split(':')[-1])} & seed_ec
    if len(d)>=5: pw_denom[p]=d
NPW=len(pw_denom); print(f'pathways={NPW}',flush=True)

def base_pred_path(baseline, gca):
    gns=gca.rsplit(".",1)[0]; num=gca.split("_")[1].split(".")[0]
    m={"graphec":f"{F}/graphec_price_new/{gca}_GraphEC.pkl",
       "mapred": f"{F}/mapred_price_new/{gca}_MAPred.pkl",
       "topec":  f"{F}/topec_price_new/{gca}_TopEC.pkl"}
    p=m[baseline]
    if os.path.exists(p): return p
    for c in (p, p.replace(gca,gns)):
        if os.path.exists(c): return c
    return p
def base_scores(baseline,gca):
    p=extract_pred(base_pred_path(baseline,gca),anc)
    cols=[str(c).split(':')[-1] for c in p.columns]; mx=p.values.max(axis=0)
    return {cols[j]:float(mx[j]) for j in range(len(cols)) if FULL.match(cols[j]) and cols[j] in seed_ec}
def meteor_ecs(baseline,gca):
    f=f'{MOP}/{baseline}/meteor_preds_{gca}.pkl'
    if not os.path.exists(f): return None
    ae=pickle.load(open(f,'rb')).get('active_ecs',set())
    return {str(e).split(':')[-1] for e in ae if FULL.match(str(e).split(':')[-1]) and str(e).split(':')[-1] in seed_ec}

res={}
print(f'{"base":8s} n  cThr  ctopK  cM   dThr  dMatch | det>=1(B/M) det>=50(B/M) Monly(weak/gap)')
for b in ['graphec','mapred','topec']:
    gcas=[os.path.basename(f)[len('meteor_preds_'):-4] for f in glob.glob(f'{MOP}/{b}/meteor_preds_*.pkl')]
    cT=[];cK=[];cM=[];dT=[];dM=[];D1b=[];D1m=[];D50b=[];D50m=[];WS=[];PG=[];MO=[]
    for gca in gcas:
        M=meteor_ecs(b,gca)
        if M is None: continue
        try: sc=base_scores(b,gca)
        except Exception as e: print(f'  skip {gca}: {e}'); continue
        K=len(M); Bthr={e for e,v in sc.items() if v>=0.5}
        BtopK=set(sorted(sc,key=sc.get,reverse=True)[:K])
        ct=[];ck=[];cm=[];d1b=d1m=d50b=d50m=0;mo=ws=pg=0
        for p,denom in pw_denom.items():
            n=len(denom); ob=len(Bthr&denom); ok=len(BtopK&denom); om=len(M&denom)
            ct.append(ob/n);ck.append(ok/n);cm.append(om/n)
            d1b+=ob>=1;d1m+=om>=1;d50b+=ob/n>=0.5;d50m+=om/n>=0.5
            if om>=1 and ob==0:
                mo+=1; det=M&denom
                if any(sc.get(e,0.0)>0.0 for e in det): ws+=1
                else: pg+=1
        cT.append(np.mean(ct));cK.append(np.mean(ck));cM.append(np.mean(cm))
        dT.append(np.mean(cm)-np.mean(ct));dM.append(np.mean(cm)-np.mean(ck))
        D1b.append(d1b);D1m.append(d1m);D50b.append(d50b);D50m.append(d50m);WS.append(ws);PG.append(pg);MO.append(mo)
    def m(x): return round(float(np.mean(x)),4) if x else None
    def m1(x): return round(float(np.mean(x)),1) if x else None
    r=dict(n=len(cM),cThr=m(cT),ctopK=m(cK),cM=m(cM),dThr_pp=round(100*float(np.mean(dT)),2),
           dMatched_pp=round(100*float(np.mean(dM)),2),n_matched_neg=int(sum(1 for x in dM if x<0)),
           det1_B=m1(D1b),det1_M=m1(D1m),det50_B=m1(D50b),det50_M=m1(D50m),
           Monly=m1(MO),weak=m1(WS),gap=m1(PG))
    res[b]=r
    print(f'{b:8s} {r["n"]:2d}  {r["cThr"]:.3f} {r["ctopK"]:.3f} {r["cM"]:.3f}  {r["dThr_pp"]:+.1f}  '
          f'{r["dMatched_pp"]:+.1f}({r["n_matched_neg"]}neg) | {r["det1_B"]:.1f}/{r["det1_M"]:.1f}  '
          f'{r["det50_B"]:.1f}/{r["det50_M"]:.1f}  {r["Monly"]:.1f}({r["weak"]:.1f}/{r["gap"]:.1f})',flush=True)
json.dump(dict(panel='price22',n_pathways=NPW,results=res),open(f'{OUT}/kegg_price_matched_detect.json','w'),indent=1)
print('-> kegg_price_matched_detect.json',flush=True)
