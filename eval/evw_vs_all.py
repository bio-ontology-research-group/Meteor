#!/usr/bin/env python3
"""FINAL comprehensive same-predictor comparison (all dpz-fed, current mask -> namespace/EC-vocab fair):
arms = v7 (prior METEOR, loose) / v8=METEOR (evw) / baseline (threshold@0.5+gapfill) / Reconstructor(dpz, neg).
Metrics per curated-GEM genome (ground truth = GEM EC set G):
  - Q2 weak-real recovery |W∩M|/|W|, W = {EC:0<dpz<0.5} ∩ G
  - precision |M∩G|/|M|,  false-positives |M\\G|
  - FP quality: subclass-adjacent% (FP shares EC x.y.z with G) and predictor-strong% (FP dpz>=0.5)
v7 sol reaction set re-mapped to EC via CURRENT mask -> fair vs v8 (bypasses v7's old extract_pred)."""
import sys,os,re,json,pickle,numpy as np,pandas as pd,cobra
import warnings,logging; warnings.filterwarnings('ignore'); logging.getLogger('cobra').setLevel(logging.ERROR)
V6='/ibex/user/niuk0a/funcarve/cobra/v6'; sys.path.insert(0,V6); os.chdir(V6)
B='/ibex/scratch/projects/c2014/kexin/funcarve'; sys.path.insert(0,f'{B}/meteor_v8/src')
from src.v6utils import (load_universal,extract_fba_matrices,load_tight_bounds,apply_media,
    find_excluded_reactions,build_rxn_ec_mask,extract_pred,load_refmapping,load_ec,_detect_solver)
from meteor_v8.repair import grow_support
sys.path.insert(0,f'{B}/meteor_v8/eval'); from baseline_io import resolve_baseline_pkl,BASELINE_SUFFIX
V8=f'{B}/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla'; V7=f'{B}/meteor_v7_run/meteor_out/dpz_vanilla'
GEMDIR=f'{B}/meteor_diag/curated_gems'; RECON=f'{B}/meteor_v8/results/toolcompare/recon_models'
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
def sol_ecs(path):
    if not os.path.exists(path): return None
    d=pickle.load(open(path,'rb')); act=np.array(d['y_vals'])>0.5
    E=set(); [E.update(rxn_ecs(j)) for j in np.where(act)[0]]; return E,int(act.sum())
_gc={}
def setup(gram):
    if gram in _gc: return _gc[gram]
    lt,ut=load_tight_bounds(f'{V6}/data/tight_bounds_v6_{gram}.pkl'); lb=np.maximum(lb0,lt); ub=np.minimum(ub0,ut)
    oi=allrxns.index('biomass_GmPos' if gram=='pos' else 'biomass_GmNeg')
    exc=find_excluded_reactions(S,lb,ub,allrxns,allrxns[oi]); lb,ub,_,_=apply_media(['default'],allrxns,lb,ub)
    _gc[gram]=(lb,ub,oi,exc); return _gc[gram]
def baseline_ecs(acc,gram,pred):
    lb,ub,oi,exc=setup(gram); ecmax=pred.values.max(axis=0); hot=set(np.where(ecmax>=0.5)[0]); draft=np.zeros(NR,bool)
    for j in range(NR):
        ei=np.where(mask[j]==1)[0]
        if len(ei) and any(e in hot for e in ei): draft[j]=True
    avail=np.ones(NR,bool); avail[list(exc)]=False; core=draft&avail
    sup=grow_support(S,lb,ub,oi,avail,0.1,core=core); model=core|(sup if sup is not None else np.zeros(NR,bool))
    E=set(); [E.update(rxn_ecs(j)) for j in np.where(model)[0]]; return E,int(model.sum())
def recon_ecs(acc):
    p=f'{RECON}/{acc}.sbml'
    if not os.path.exists(p): return None
    m=cobra.io.read_sbml_model(p); E=set()
    for r in m.reactions:
        base=r.id.split('_')[0]
        if base in seed2ec: E|=seed2ec[base]
    return E,len([r for r in m.reactions if not r.id.startswith(('EX_','DM_','SK_'))])
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
rows=[]
ARMS=['v7','v8(METEOR)','baseline','recon']
print('=== 4-arm same-predictor comparison (v7 / v8-evw / threshold+gapfill / Reconstructor-dpz) ===')
print(f'{"organism":13s}| arm         n_ec  Wrec  prec   FP  FPadj% FPstrong%')
for gem,(gcf,gram,name) in GEM2G.items():
    scores,pred=dpz_scores(gcf); G=gem_ecs(gem); Gsub={'.'.join(e.split('.')[:3]) for e in G}
    W=({e for e,s in scores.items() if 0<s<0.5}) & G
    res={}
    res['v7']=sol_ecs(f'{V7}/meteor_sol_{gcf}.pkl')
    res['v8(METEOR)']=sol_ecs(f'{V8}/meteor_sol_{gcf}.pkl')
    res['baseline']=baseline_ecs(gcf,gram,pred)
    res['recon']=recon_ecs(gcf) if gram=='neg' else None
    grow={}
    for a in ARMS:
        r=res[a]
        if r is None: print(f'{name if a==ARMS[0] else "":13s}| {a:11s} NA'); grow[a]=None; continue
        M,nrx=r; fp=M-G
        wrec=round(len(M&W)/max(1,len(W)),3); prec=round(len(M&G)/max(1,len(M)),3)
        adj=round(sum(1 for e in fp if '.'.join(e.split('.')[:3]) in Gsub)/max(1,len(fp)),3)
        strong=round(sum(1 for e in fp if scores.get(e,0)>=0.5)/max(1,len(fp)),3)
        print(f'{name if a==ARMS[0] else "":13s}| {a:11s} {len(M):4d} {wrec:.2f}  {prec:.3f} {len(fp):4d}  {100*adj:4.1f}  {100*strong:4.1f}')
        grow[a]=dict(n_ec=len(M),n_rxn=nrx,W_recovery=wrec,precision=prec,FP=len(fp),FP_adj=adj,FP_strong=strong)
    rows.append(dict(organism=name,gram=gram,nW=len(W),arms=grow))
json.dump(rows,open(f'{OUT}/evw_vs_all.json','w'),indent=1)
def mn(a,k):
    v=[r['arms'][a][k] for r in rows if r['arms'][a] is not None]; return np.mean(v) if v else float('nan')
print(f'\n--- MEANS (n=6; recon over 4 neg) ---   |W| avg={np.mean([r["nW"] for r in rows]):.0f}')
print(f'{"arm":12s} n_ec Wrec prec  FP  FPadj% FPstrong%')
for a in ARMS:
    print(f'{a:12s} {mn(a,"n_ec"):.0f} {mn(a,"W_recovery"):.2f} {mn(a,"precision"):.3f} {mn(a,"FP"):.0f}  {100*mn(a,"FP_adj"):.1f}  {100*mn(a,"FP_strong"):.1f}')
print('-> results/toolcompare/evw_vs_all.json')
