#!/usr/bin/env python3
"""The RIGHT question (user 2026-07-26): cost of hard thresholding vs METEOR soft evidence.
Q1 false positives: reactions/ECs the threshold INTRODUCES that are NOT in ground-truth GEM.
Q2 weak-but-real: ECs with 0<dpz<0.5 that ARE in the GEM (dropped from a hard-threshold draft) --
   how many does each FINAL model contain? METEOR(soft) should keep them; baseline/Reconstructor
   only recover the biomass-essential ones via gap-fill.
EC-level, namespace-neutral, ground truth = curated GEM EC annotation."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys,os,re,json,pickle,numpy as np,pandas as pd,cobra
import warnings,logging; warnings.filterwarnings('ignore'); logging.getLogger('cobra').setLevel(logging.ERROR)
B='/ibex/scratch/projects/c2014/kexin/funcarve'
sys.path.insert(0,f'{B}/meteor_v8/src')
from meteor_v8.utils import (load_universal,extract_fba_matrices,load_tight_bounds,apply_media,
    find_excluded_reactions,build_rxn_ec_mask,extract_pred,load_refmapping,load_ec,_detect_solver)
from meteor_v8.repair import grow_support
sys.path.insert(0,f'{B}/meteor_v8/eval')
from baseline_io import resolve_baseline_pkl,BASELINE_SUFFIX
MO=f'{B}/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla'
GEMDIR=f'{B}/meteor_diag/curated_gems'
RECON=f'{B}/meteor_v8/results/toolcompare/recon_models'
R2ECF='/ibex/user/niuk0a/funcarve/reconstructor/reconstructor/Unique_ModelSEED_Reaction_ECs.txt'
OUT=f'{B}/meteor_v8/results/toolcompare'
FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')

universal,allrxns,allmet=load_universal()
seedr2ec,_=load_refmapping('data'); seedr2ec={k:v for k,v in seedr2ec.items() if v}
anc=load_ec('data/all_ancestors.txt'); mask=build_rxn_ec_mask(allrxns,seedr2ec,anc)
S,lb0,ub0=extract_fba_matrices(universal,allrxns,reversed_trans=True)
solver,_=_detect_solver(threads=4,time_limit=600)
NR=len(allrxns)
# seed(ModelSEED rxn)->EC for reconstructor models
seed2ec={}
_e=pd.read_csv(R2ECF,sep='\t')
for _,r in _e.iterrows():
    ec=str(r['External ID']).strip()
    if FULL.match(ec): seed2ec.setdefault(str(r['ModelSEED ID']).strip(),set()).add(ec)
_pred_cols=None
def dpz_ec_scores(acc):
    global _pred_cols
    p=extract_pred(resolve_baseline_pkl('dpz','vanilla',acc,BASELINE_SUFFIX['dpz']),anc)
    if _pred_cols is None: _pred_cols=[str(c) for c in p.columns]
    sc=p.values.max(axis=0)
    return {str(_pred_cols[j]).split(':')[-1]:float(sc[j]) for j in range(len(sc)) if FULL.match(str(_pred_cols[j]).split(':')[-1])}, p
def rxn_ecs(j):
    return {str(_pred_cols[e]).split(':')[-1] for e in np.where(mask[j]==1)[0] if e<len(_pred_cols) and FULL.match(str(_pred_cols[e]).split(':')[-1])}
_gc={}
def setup(gram):
    if gram in _gc: return _gc[gram]
    lt,ut=load_tight_bounds(data_path(f'tight_bounds_v6_{gram}.pkl'))
    lb=np.maximum(lb0,lt); ub=np.minimum(ub0,ut)
    oi=allrxns.index('biomass_GmPos' if gram=='pos' else 'biomass_GmNeg')
    exc=find_excluded_reactions(S,lb,ub,allrxns,allrxns[oi]); lb,ub,_,_=apply_media(['default'],allrxns,lb,ub)
    _gc[gram]=(lb,ub,oi,exc); return _gc[gram]
def meteor_ecs(acc):
    sol=pickle.load(open(f'{MO}/meteor_sol_{acc}.pkl','rb')); act=np.array(sol['y_vals'])>0.5
    E=set();  [E.update(rxn_ecs(j)) for j in np.where(act)[0]]; return E
def baseline_ecs(acc,gram,pred):
    lb,ub,oi,exc=setup(gram); ecmax=pred.values.max(axis=0); hot=set(np.where(ecmax>=0.5)[0])
    draft=np.zeros(NR,bool)
    for j in range(NR):
        ei=np.where(mask[j]==1)[0]
        if len(ei) and any(e in hot for e in ei): draft[j]=True
    avail=np.ones(NR,bool); avail[list(exc)]=False; core=draft&avail
    sup=grow_support(S,lb,ub,oi,avail,0.1,core=core)
    model=core|(sup if sup is not None else np.zeros(NR,bool))
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
print('=== Q2 weak-but-real recovery + Q1 false positives (EC-level, GEM ground truth) ===')
print(f'{"organism":13s} {"|W|":4s}| {"W-recovery METEOR/base/recon":30s} | {"FP(not in GEM) M/b/r":20s}')
for gem,(gcf,gram,name) in GEM2G.items():
    scores,pred=dpz_ec_scores(gcf); G=gem_ecs(gem)
    strong={e for e,s in scores.items() if s>=0.5}; weak={e for e,s in scores.items() if 0<s<0.5}
    W=weak&G                                   # weak-but-real (dropped from hard-threshold draft)
    M=meteor_ecs(gcf); Ba=baseline_ecs(gcf,gram,pred); Rc=recon_ecs(gcf) if gram=='neg' else None
    rec_m,rec_b=frac(M,W),frac(Ba,W); rec_r=frac(Rc,W) if Rc is not None else None
    fp_m=len(M-G); fp_b=len(Ba-G); fp_r=(len(Rc-G) if Rc is not None else None)
    rr = f'{rec_r}' if rec_r is not None else 'NA'
    fr = f'{fp_r}' if fp_r is not None else 'NA'
    print(f'{name:13s} {len(W):4d}| METEOR={rec_m:.2f} base={rec_b:.2f} recon={rr:>4s}      | {fp_m:5d}/{fp_b:5d}/{fr:>5s}')
    rows.append(dict(organism=name,gcf=gcf,gram=gram,n_strong=len(strong),n_weak=len(weak),n_weak_real=len(W),
        strong_spurious=len(strong-G),
        W_recovery=dict(meteor=rec_m,baseline=rec_b,recon=rec_r),
        false_positive=dict(meteor=fp_m,baseline=fp_b,recon=fp_r),
        n_ec=dict(meteor=len(M),baseline=len(Ba),recon=(len(Rc) if Rc is not None else None)),
        precision=dict(meteor=round(len(M&G)/max(1,len(M)),3),baseline=round(len(Ba&G)/max(1,len(Ba)),3),
                       recon=(round(len(Rc&G)/max(1,len(Rc)),3) if Rc is not None else None))))
json.dump(rows,open(f'{OUT}/weakreal.json','w'),indent=1)
def m(key,sub):
    v=[r[key][sub] for r in rows if r[key][sub] is not None]; return np.mean(v) if v else float('nan')
print(f'\n--- MEANS (n={len(rows)}; recon over available) ---')
print(f'  weak-real |W| avg = {np.mean([r["n_weak_real"] for r in rows]):.0f}')
print(f'  W-recovery:  METEOR={m("W_recovery","meteor"):.2f}  baseline={m("W_recovery","baseline"):.2f}  recon={m("W_recovery","recon"):.2f}')
print(f'  false-pos:   METEOR={m("false_positive","meteor"):.0f}  baseline={m("false_positive","baseline"):.0f}  recon={m("false_positive","recon"):.0f}')
print(f'  precision:   METEOR={m("precision","meteor"):.3f}  baseline={m("precision","baseline"):.3f}  recon={m("precision","recon"):.3f}')
print('-> results/toolcompare/weakreal.json')
