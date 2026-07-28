#!/usr/bin/env python3
"""Test user's hypothesis: is evw's weak-real advantage confined to BIOMASS-RELEVANT (flux-capable) reactions,
and do clean/enzbert have their weak signal on PERIPHERAL (blocked) pathways (explaining the null in B3)?
Stratify weak-real ECs W by whether any of the EC's reactions is flux-capable (network-integrated) vs all
blocked (peripheral). Per predictor (dpz/clean/enzbert): capable-fraction of W + METEOR vs baseline recovery
WITHIN each stratum. Prediction: (1) dpz W more capable than clean/enzbert; (2) in capable stratum METEOR>baseline
for all predictors (=> benefit general, just differently distributed)."""
import sys,os,re,json,pickle,numpy as np,pandas as pd,cobra
import warnings,logging; warnings.filterwarnings('ignore'); logging.getLogger('cobra').setLevel(logging.ERROR)
V6='/ibex/user/niuk0a/funcarve/cobra/v6'; sys.path.insert(0,V6); os.chdir(V6)
B='/ibex/scratch/projects/c2014/kexin/funcarve'; sys.path.insert(0,f'{B}/meteor_v8/src')
from src.v6utils import (load_universal,extract_fba_matrices,load_tight_bounds,apply_media,
    find_excluded_reactions,build_rxn_ec_mask,extract_pred,load_refmapping,load_ec,_detect_solver)
from meteor_v8.repair import grow_support
sys.path.insert(0,f'{B}/meteor_v8/eval'); from baseline_io import resolve_baseline_pkl,BASELINE_SUFFIX
GEMDIR=f'{B}/meteor_diag/curated_gems'; OUT=f'{B}/meteor_v8/results/toolcompare'; FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')
universal,allrxns,allmet=load_universal()
seedr2ec,_=load_refmapping('data'); seedr2ec={k:v for k,v in seedr2ec.items() if v}
anc=load_ec('data/all_ancestors.txt'); mask=build_rxn_ec_mask(allrxns,seedr2ec,anc)
S,lb0,ub0=extract_fba_matrices(universal,allrxns,reversed_trans=True); solver,_=_detect_solver(threads=4,time_limit=600); NR=len(allrxns)
CAP={g:pickle.load(open(f'{OUT}/fluxcapable_{g}.pkl','rb'))['capable'] for g in ['neg','pos']}
_pc=None
def dpz_scores(pred_name,acc):
    global _pc
    p=extract_pred(resolve_baseline_pkl(pred_name,'vanilla',acc,BASELINE_SUFFIX[pred_name]),anc)
    if _pc is None: _pc=[str(c) for c in p.columns]
    return p.values.max(axis=0)
def ec_col(e):  # EC name -> mask column index (via _pc)
    return _pc.index(e) if e in _pc else None
def rxn_ecs(j): return {str(_pc[e]).split(':')[-1] for e in np.where(mask[j]==1)[0] if e<len(_pc) and FULL.match(str(_pc[e]).split(':')[-1])}
# EC name (bare) -> reaction indices
ec2rxn={}
def build_ec2rxn():
    for j in range(NR):
        for e in np.where(mask[j]==1)[0]:
            if e<len(_pc):
                nm=str(_pc[e]).split(':')[-1]
                if FULL.match(nm): ec2rxn.setdefault(nm,[]).append(j)
_gc={}
def setup(gram):
    if gram in _gc: return _gc[gram]
    lt,ut=load_tight_bounds(f'{V6}/data/tight_bounds_v6_{gram}.pkl'); lb=np.maximum(lb0,lt); ub=np.minimum(ub0,ut)
    oi=allrxns.index('biomass_GmPos' if gram=='pos' else 'biomass_GmNeg')
    exc=find_excluded_reactions(S,lb,ub,allrxns,allrxns[oi]); lb,ub,_,_=apply_media(['default'],allrxns,lb,ub)
    _gc[gram]=(lb,ub,oi,exc); return _gc[gram]
def sol_ecs(path):
    if not os.path.exists(path): return None
    act=np.array(pickle.load(open(path,'rb'))['y_vals'])>0.5
    E=set(); [E.update(rxn_ecs(j)) for j in np.where(act)[0]]; return E
def baseline_ecs(gram,ecmax):
    lb,ub,oi,exc=setup(gram); hot=set(np.where(ecmax>=0.5)[0]); draft=np.zeros(NR,bool)
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
GEM2G={'iML1515':('GCF_058436375.1','neg','E.coli'),'STM_v1_0':('GCF_000006945.2','neg','Salmonella'),
 'iYL1228':('GCF_058435815.1','neg','K.pneumoniae'),'iJN1463':('GCF_045571375.1','neg','P.putida'),
 'iYS854':('GCF_045348045.1','pos','S.aureus'),'iYO844':('GCF_058182495.1','pos','B.subtilis')}
V8ROOT=f'{B}/meteor_v8_evw_p2mu3_run/meteor_out'
def ec_capable(e,gram):  # EC is biomass-relevant if any of its reactions is flux-capable under this gram
    return any(CAP[gram][j] for j in ec2rxn.get(e,[]))
def rec(M,Wset): return round(len(M&Wset)/max(1,len(Wset)),3)
res={}
for pred in ['dpz','clean','enzbert']:
    globals()['_pc']=None
    per=[]
    for gem,(gcf,gram,name) in GEM2G.items():
        ecmax=dpz_scores(pred,gcf)
        if not ec2rxn: build_ec2rxn()
        G=gem_ecs(gem); W=({str(_pc[j]).split(':')[-1] for j in np.where((ecmax>0)&(ecmax<0.5))[0] if FULL.match(str(_pc[j]).split(':')[-1])}) & G
        Wcap={e for e in W if ec_capable(e,gram)}; Wper=W-Wcap
        M=sol_ecs(f'{V8ROOT}/{pred}_vanilla/meteor_sol_{gcf}.pkl'); Ba=baseline_ecs(gram,ecmax)
        if M is None: continue
        per.append(dict(nW=len(W),capfrac=len(Wcap)/max(1,len(W)),
            cap=dict(n=len(Wcap),met=rec(M,Wcap),base=rec(Ba,Wcap)),
            per=dict(n=len(Wper),met=rec(M,Wper),base=rec(Ba,Wper))))
    res[pred]=per
    cf=np.mean([p['capfrac'] for p in per]);
    mc=np.mean([p['cap']['met'] for p in per]); bc=np.mean([p['cap']['base'] for p in per])
    mp=np.mean([p['per']['met'] for p in per]); bp=np.mean([p['per']['base'] for p in per])
    print(f'{pred:8s}: |W|={np.mean([p["nW"] for p in per]):.0f}  capable-frac={100*cf:.0f}%  | CAPABLE: METEOR={mc:.2f} base={bc:.2f} (Δ{mc-bc:+.2f}) | PERIPH: METEOR={mp:.2f} base={bp:.2f}',flush=True)
json.dump(res,open(f'{OUT}/weakreal_strat.json','w'),indent=1)
print('-> results/toolcompare/weakreal_strat.json')
