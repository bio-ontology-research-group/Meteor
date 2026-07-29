#!/usr/bin/env python3
"""§10 extended: weak-real recovery (Q2) + false-positive QUALITY decomposition (Q1 defended).
For each method's FP set (model ECs not in GEM), classify:
  - subclass-adjacent: shares EC x.y.z with some GEM EC (likely same enzyme family -> plausibly real, GEM omitted)
  - predictor-strong: dpz score >=0.5 (predictor confident -> likely real, GEM incompleteness not METEOR error)
If METEOR's extra FPs are disproportionately adjacent/strong, the precision cost is benign."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys,os,re,json,pickle,numpy as np,pandas as pd,cobra
import warnings,logging; warnings.filterwarnings('ignore'); logging.getLogger('cobra').setLevel(logging.ERROR)
B='/ibex/scratch/projects/c2014/kexin/funcarve'; sys.path.insert(0,f'{B}/meteor_v8/src')
from meteor_v8.utils import (load_universal,extract_fba_matrices,load_tight_bounds,apply_media,
    find_excluded_reactions,build_rxn_ec_mask,extract_pred,load_refmapping,load_ec,_detect_solver)
from meteor_v8.repair import grow_support
sys.path.insert(0,f'{B}/meteor_v8/eval'); from baseline_io import resolve_baseline_pkl,BASELINE_SUFFIX
MO=f'{B}/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla'; GEMDIR=f'{B}/meteor_diag/curated_gems'
RECON=f'{B}/meteor_v8/results/toolcompare/recon_models'
R2ECF='/ibex/user/niuk0a/funcarve/reconstructor/reconstructor/Unique_ModelSEED_Reaction_ECs.txt'
OUT=f'{B}/meteor_v8/results/toolcompare'; FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')
universal,allrxns,allmet=load_universal()
seedr2ec,_=load_refmapping('data'); seedr2ec={k:v for k,v in seedr2ec.items() if v}
anc=load_ec('data/all_ancestors.txt'); mask=build_rxn_ec_mask(allrxns,seedr2ec,anc)
S,lb0,ub0=extract_fba_matrices(universal,allrxns,reversed_trans=True); solver,_=_detect_solver(threads=4,time_limit=600); NR=len(allrxns)
seed2ec={}; _e=pd.read_csv(R2ECF,sep='\t')
for _,r in _e.iterrows():
    ec=str(r['External ID']).strip()
    if FULL.match(ec): seed2ec.setdefault(str(r['ModelSEED ID']).strip(),set()).add(ec)
_pc=None
def dpz_scores(acc):
    global _pc
    p=extract_pred(resolve_baseline_pkl('dpz','vanilla',acc,BASELINE_SUFFIX['dpz']),anc)
    if _pc is None: _pc=[str(c) for c in p.columns]
    sc=p.values.max(axis=0)
    return {str(_pc[j]).split(':')[-1]:float(sc[j]) for j in range(len(sc)) if FULL.match(str(_pc[j]).split(':')[-1])},p
def rxn_ecs(j): return {str(_pc[e]).split(':')[-1] for e in np.where(mask[j]==1)[0] if e<len(_pc) and FULL.match(str(_pc[e]).split(':')[-1])}
_gc={}
def setup(gram):
    if gram in _gc: return _gc[gram]
    lt,ut=load_tight_bounds(data_path(f'tight_bounds_v6_{gram}.pkl')); lb=np.maximum(lb0,lt); ub=np.minimum(ub0,ut)
    oi=allrxns.index('biomass_GmPos' if gram=='pos' else 'biomass_GmNeg')
    exc=find_excluded_reactions(S,lb,ub,allrxns,allrxns[oi]); lb,ub,_,_=apply_media(['default'],allrxns,lb,ub)
    _gc[gram]=(lb,ub,oi,exc); return _gc[gram]
def meteor_ecs(acc):
    sol=pickle.load(open(f'{MO}/meteor_sol_{acc}.pkl','rb')); act=np.array(sol['y_vals'])>0.5
    E=set(); [E.update(rxn_ecs(j)) for j in np.where(act)[0]]; return E
def baseline_ecs(acc,gram,pred):
    lb,ub,oi,exc=setup(gram); ecmax=pred.values.max(axis=0); hot=set(np.where(ecmax>=0.5)[0]); draft=np.zeros(NR,bool)
    for j in range(NR):
        ei=np.where(mask[j]==1)[0]
        if len(ei) and any(e in hot for e in ei): draft[j]=True
    avail=np.ones(NR,bool); avail[list(exc)]=False; core=draft&avail
    sup=grow_support(S,lb,ub,oi,avail,0.1,core=core); model=core|(sup if sup is not None else np.zeros(NR,bool))
    E=set(); [E.update(rxn_ecs(j)) for j in np.where(model)[0]]; return E
def recon_ecs(acc):
    p=f'{RECON}/{acc}.sbml'
    if not os.path.exists(p): return None
    m=cobra.io.read_sbml_model(p); E=set()
    for r in m.reactions:
        base=r.id.split('_')[0]
        if base in seed2ec: E|=seed2ec[base]
    return E
def gem_ecs(gem):
    m=cobra.io.read_sbml_model(f'{GEMDIR}/{gem}.xml'); ex=set()
    for r in m.reactions:
        a=r.annotation.get('ec-code') if hasattr(r,'annotation') else None
        if not a: continue
        for x in (a if isinstance(a,list) else [a]):
            if FULL.match(str(x).strip()): ex.add(str(x).strip())
    return ex
GEM2G={'iML1515':('GCF_058436375.1','neg','E.coli'),'STM_v1_0':('GCF_000006945.2','neg','Salmonella'),
 'iYL1228':('GCF_058435815.1','neg','K.pneumoniae'),'iJN1463':('GCF_045571375.1','neg','P.putida'),
 'iYS854':('GCF_045348045.1','pos','S.aureus'),'iYO844':('GCF_058182495.1','pos','B.subtilis')}
def frac(a,b): return round(len(a&b)/max(1,len(b)),3)
rows=[]
print('=== FP QUALITY DECOMPOSITION (are METEOR extra FPs benign?) ===')
print(f'{"organism":13s}| {"method":8s} FP  adj%(subclass) strong%(dpz>=.5)')
for gem,(gcf,gram,name) in GEM2G.items():
    scores,pred=dpz_scores(gcf); G=gem_ecs(gem); Gsub={'.'.join(e.split('.')[:3]) for e in G}
    M=meteor_ecs(gcf); Ba=baseline_ecs(gcf,gram,pred); Rc=recon_ecs(gcf) if gram=='neg' else None
    def decomp(Mset):
        if Mset is None: return None
        fp=Mset-G
        if not fp: return dict(n_fp=0,adj=None,strong=None)
        adj=sum(1 for e in fp if '.'.join(e.split('.')[:3]) in Gsub)/len(fp)
        strong=sum(1 for e in fp if scores.get(e,0)>=0.5)/len(fp)
        return dict(n_fp=len(fp),adj=round(adj,3),strong=round(strong,3))
    d={'meteor':decomp(M),'baseline':decomp(Ba),'recon':decomp(Rc)}
    for k in ['meteor','baseline','recon']:
        dd=d[k]
        if dd is None: print(f'{name if k=="meteor" else "":13s}| {k:8s} NA'); continue
        print(f'{name if k=="meteor" else "":13s}| {k:8s} {dd["n_fp"]:4d}  adj={100*dd["adj"]:4.1f}%     strong={100*dd["strong"]:4.1f}%')
    rows.append(dict(organism=name,gram=gram,fp=d,W_recovery=dict(meteor=frac(M,gem_ecs(gem)&{e for e,s in scores.items() if 0<s<0.5}))))
json.dump(rows,open(f'{OUT}/fp_decomp.json','w'),indent=1)
def mn(k,sub):
    v=[r['fp'][k][sub] for r in rows if r['fp'][k] and r['fp'][k][sub] is not None]; return np.mean(v) if v else float('nan')
print(f'\n--- MEANS ---')
for k in ['meteor','baseline','recon']:
    print(f'  {k:8s}: FP={mn(k,"n_fp"):.0f}  subclass-adjacent={100*mn(k,"adj"):.1f}%  predictor-strong={100*mn(k,"strong"):.1f}%')
print('-> results/toolcompare/fp_decomp.json')
