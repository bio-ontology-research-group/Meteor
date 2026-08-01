#!/usr/bin/env python3
"""DECISIVE B3 closure: METEOR vs baseline FUNCTIONAL coverage of weak-real ECs across predictors.
functional coverage = fraction of weak-real W (0<dpz<0.5, in GEM) that the model recovers EITHER directly
OR via a same-EC-subclass (x.y.z) alternative (isozyme-level; ~87% of 'misses' are this, per KEGG analysis).
If METEOR > baseline functionally even on clean/enzbert -> the evw weak-real benefit is GENERAL (B3 null was
an exact-EC metric artifact). If baseline ties METEOR -> B3 null stands."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, external_path, data_dir
import sys,os,re,json,pickle,numpy as np,pandas as pd,cobra
import warnings,logging; warnings.filterwarnings('ignore'); logging.getLogger('cobra').setLevel(logging.ERROR)
B='/ibex/scratch/projects/c2014/kexin/funcarve'; sys.path.insert(0,f'{B}/meteor_v8/src')
from meteor_v8.utils import (load_universal,extract_fba_matrices,load_tight_bounds,apply_media,
    find_excluded_reactions,build_rxn_ec_mask,extract_pred,load_refmapping,load_ec,_detect_solver)
from meteor_v8.repair import grow_support
sys.path.insert(0,f'{B}/meteor_v8/eval'); from baseline_io import resolve_baseline_pkl,BASELINE_SUFFIX
GEMDIR=external_path('curated_gems'); OUT=f'{B}/meteor_v8/results/toolcompare'
V8ROOT=f'{B}/meteor_v8_evw_p2mu3_run/meteor_out'; FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')
universal,allrxns,allmet=load_universal()
seedr2ec,_=load_refmapping('data'); seedr2ec={k:v for k,v in seedr2ec.items() if v}
anc=load_ec('data/all_ancestors.txt'); mask=build_rxn_ec_mask(allrxns,seedr2ec,anc)
S,lb0,ub0=extract_fba_matrices(universal,allrxns,reversed_trans=True); solver,_=_detect_solver(threads=4,time_limit=600); NR=len(allrxns)
_pc=None
def scores(pred,acc):
    global _pc
    p=extract_pred(resolve_baseline_pkl(pred,'vanilla',acc,BASELINE_SUFFIX[pred]),anc)
    _pc=[str(c).split(':')[-1] for c in p.columns]; mx=p.values.max(axis=0)
    return {_pc[j]:float(mx[j]) for j in range(len(mx)) if FULL.match(_pc[j])}
def rxn_ecs(j): return {_pc[e] for e in np.where(mask[j]==1)[0] if e<len(_pc) and FULL.match(_pc[e])}
def meteor_ecs(pred,acc):
    p=f'{V8ROOT}/{pred}_vanilla/meteor_sol_{acc}.pkl'
    if not os.path.exists(p): return None
    act=np.array(pickle.load(open(p,'rb'))['y_vals'])>0.5; E=set()
    for j in np.where(act)[0]: E|=rxn_ecs(j)
    return E
_gc={}
def setup(gram):
    if gram in _gc: return _gc[gram]
    lt,ut=load_tight_bounds(data_path(f'tight_bounds_v6_{gram}.pkl')); lb=np.maximum(lb0,lt); ub=np.minimum(ub0,ut)
    oi=allrxns.index('biomass_GmPos' if gram=='pos' else 'biomass_GmNeg')
    exc=find_excluded_reactions(S,lb,ub,allrxns,allrxns[oi]); lb,ub,_,_=apply_media(['default'],allrxns,lb,ub)
    _gc[gram]=(lb,ub,oi,exc); return _gc[gram]
def baseline_ecs(gram,sc):
    lb,ub,oi,exc=setup(gram); hot={_pc.index(e) for e,s in sc.items() if s>=0.5 and e in _pc}
    draft=np.zeros(NR,bool)
    for j in range(NR):
        ei=np.where(mask[j]==1)[0]
        if len(ei) and any(e in hot for e in ei): draft[j]=True
    avail=np.ones(NR,bool); avail[list(exc)]=False; core=draft&avail
    sup=grow_support(S,lb,ub,oi,avail,0.1,core=core); model=core|(sup if sup is not None else np.zeros(NR,bool))
    E=set(); [E.update(rxn_ecs(j)) for j in np.where(model)[0]]; return E
def gem_ecs(gem):
    m=cobra.io.read_sbml_model(f'{GEMDIR}/{gem}.xml'); ex=set()
    for r in m.reactions:
        a=r.annotation.get('ec-code') if hasattr(r,'annotation') else None
        if not a: continue
        for x in (a if isinstance(a,list) else [a]):
            if FULL.match(str(x).strip()): ex.add(str(x).strip())
    return ex
def exact(M,W): return len(M&W)/max(1,len(W))
def func(M,W):
    subM={'.'.join(e.split('.')[:3]) for e in M}
    return sum(1 for e in W if e in M or '.'.join(e.split('.')[:3]) in subM)/max(1,len(W))
GEM2G={'iML1515':('GCF_058436375.1','neg'),'STM_v1_0':('GCF_000006945.2','neg'),'iYL1228':('GCF_058435815.1','neg'),
 'iJN1463':('GCF_045571375.1','neg'),'iYS854':('GCF_045348045.1','pos'),'iYO844':('GCF_058182495.1','pos')}
print('=== METEOR vs baseline: EXACT vs FUNCTIONAL(+isozyme) weak-real coverage, per predictor ===')
print(f'{"pred":8s}| {"EXACT  MET/base (Δ)":22s}| {"FUNCTIONAL MET/base (Δ)":24s}')
res={}
for pred in ['dpz','clean','enzbert']:
    ex_m=[];ex_b=[];fn_m=[];fn_b=[]
    for gem,(acc,gram) in GEM2G.items():
        sc=scores(pred,acc); G=gem_ecs(gem); W={e for e,s in sc.items() if 0<s<0.5}&G
        M=meteor_ecs(pred,acc); Ba=baseline_ecs(gram,sc)
        if M is None: continue
        ex_m.append(exact(M,W)); ex_b.append(exact(Ba,W)); fn_m.append(func(M,W)); fn_b.append(func(Ba,W))
    EM,EB,FM,FB=np.mean(ex_m),np.mean(ex_b),np.mean(fn_m),np.mean(fn_b)
    print(f'{pred:8s}| {EM:.2f} / {EB:.2f}  (Δ{EM-EB:+.2f}) | {FM:.2f} / {FB:.2f}  (Δ{FM-FB:+.2f})',flush=True)
    res[pred]=dict(exact_met=round(EM,3),exact_base=round(EB,3),func_met=round(FM,3),func_base=round(FB,3),
                   exact_gap=round(EM-EB,3),func_gap=round(FM-FB,3))
json.dump(res,open(f'{OUT}/func_cover.json','w'),indent=1)
print('\nB3 closure: if func Δ (MET-base) > 0 for clean/enzbert -> evw benefit GENERAL (exact-EC artifact); if ~0 -> null stands')
print('-> results/toolcompare/func_cover.json')
