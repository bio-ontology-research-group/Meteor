#!/usr/bin/env python3
"""Panel B, namespace-NEUTRAL: EC-level P/R of METEOR vs CarveMe vs curated GEM.
Avoids SEED<->BiGG reaction-id granularity confound by comparing EC number sets.
Only complete 4-level ECs (X.X.X.X) counted."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import external_path
import sys,os,json,pickle,cobra,re,numpy as np
import warnings,logging; warnings.filterwarnings('ignore'); logging.getLogger('cobra').setLevel(logging.ERROR)
B='/ibex/scratch/projects/c2014/kexin/funcarve'
MO=f'{B}/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla'
CV=f'{B}/paperA_2026/results/carveme_gc'; GEMDIR=external_path('curated_gems')
OUT=f'{B}/meteor_v8/results/toolcompare'
FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')
def norm(s):
    s=str(s).strip()
    return s if FULL.match(s) else None
def model_ecs(p):
    m=cobra.io.read_sbml_model(p); ex=set()
    for r in m.reactions:
        e=r.annotation.get('ec-code') if hasattr(r,'annotation') else None
        if not e: continue
        for x in (e if isinstance(e,list) else [e]):
            n=norm(x)
            if n: ex.add(n)
    return ex
def meteor_ecs(gcf):
    d=pickle.load(open(f'{MO}/meteor_preds_{gcf}.pkl','rb'))
    return {n for e in d.get('active_ecs',[]) if (n:=norm(e))}
cvfiles=os.listdir(CV)
def find_cv(acc):
    num=acc.split('_')[1]
    for f in cvfiles:
        if num in f and f.endswith('.xml'): return os.path.join(CV,f)
def pr(P,G):
    i=len(P&G); return round(i/max(1,len(P)),3), round(i/max(1,len(G)),3)
GEM2G={'iML1515':('GCF_058436375.1','E.coli'),'STM_v1_0':('GCF_000006945.2','Salmonella'),
 'iYL1228':('GCF_058435815.1','K.pneumoniae'),'iJN1463':('GCF_045571375.1','P.putida'),
 'iYS854':('GCF_045348045.1','S.aureus'),'iYO844':('GCF_058182495.1','B.subtilis')}
print('=== PANEL B (EC-level, namespace-neutral): METEOR vs CarveMe vs GEM ===')
print(f'{"organism":14s} {"GEM_EC":6s} {"MET_EC":6s} {"CAR_EC":6s} | {"METEOR P/R":15s} | {"CarveMe P/R":15s}')
rows=[]
for gem,(gcf,name) in GEM2G.items():
    G=model_ecs(f'{GEMDIR}/{gem}.xml'); M=meteor_ecs(gcf); C=model_ecs(find_cv(gcf))
    mP,mR=pr(M,G); cP,cR=pr(C,G)
    print(f'{name:14s} {len(G):6d} {len(M):6d} {len(C):6d} | P={mP:.3f} R={mR:.3f}   | P={cP:.3f} R={cR:.3f}')
    rows.append(dict(organism=name,gem=gem,gcf=gcf,gem_ec=len(G),
        meteor=dict(n_ec=len(M),P=mP,R=mR),carveme=dict(n_ec=len(C),P=cP,R=cR)))
json.dump(rows,open(f'{OUT}/panelB_ec.json','w'),indent=1)
print('\n--- means ---')
for t in ['meteor','carveme']:
    print(f'  {t:8s}: P={np.mean([r[t]["P"] for r in rows]):.3f}  R={np.mean([r[t]["R"] for r in rows]):.3f}  n_ec={np.mean([r[t]["n_ec"] for r in rows]):.0f}')
print('-> results/toolcompare/panelB_ec.json')
