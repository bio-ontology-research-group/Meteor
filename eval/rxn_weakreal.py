#!/usr/bin/env python3
"""REACTION-level weak-real (user: more direct than EC). SEED reaction space, ground truth = GEM via MNXR.
A SEED reaction j is 'weak' if its EC's max dpz score in (0,0.5); 'real' if its MNXR is in the GEM's MNXR set.
weak-real reactions W_rxn = weak & real. Recovery per method = fraction of W_rxn active in that model.
Also reaction-level precision = |active∩GEM_MNXR| / |active with MNXR|. Namespace loss (SEED->MNXR) hits all
dpz arms equally (v7/v8/baseline). Arms: v7 / v8=METEOR / baseline."""
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
V8=f'{B}/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla'; V7=f'{B}/meteor_v7_run/meteor_out/dpz_vanilla'
GEMDIR=f'{B}/meteor_diag/curated_gems'; OUT=f'{B}/meteor_v8/results/toolcompare'; FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')
bigg2mnxr={}; seed2mnxr={}
for ln in open(f'{B}/meteor_diag/reac_xref.tsv'):
    if ln.startswith('#'): continue
    p=ln.rstrip().split('\t')
    if len(p)<2 or not p[1].startswith('MNXR'): continue
    if p[0].startswith('bigg.reaction:'): bigg2mnxr.setdefault(p[0].split(':',1)[1],p[1])
    elif p[0].startswith('seed.reaction:'): seed2mnxr.setdefault(p[0].split(':',1)[1],p[1])
universal,allrxns,allmet=load_universal()
seedidx_mnxr={j:seed2mnxr.get(allrxns[j].split('_')[0]) for j in range(len(allrxns))}
seedr2ec,_=load_refmapping('data'); seedr2ec={k:v for k,v in seedr2ec.items() if v}
anc=load_ec('data/all_ancestors.txt'); mask=build_rxn_ec_mask(allrxns,seedr2ec,anc)
S,lb0,ub0=extract_fba_matrices(universal,allrxns,reversed_trans=True); solver,_=_detect_solver(threads=4,time_limit=600); NR=len(allrxns)
_pc=None
def dpz(acc):
    global _pc
    p=extract_pred(resolve_baseline_pkl('dpz','vanilla',acc,BASELINE_SUFFIX['dpz']),anc)
    if _pc is None: _pc=[str(c) for c in p.columns]
    return p.values.max(axis=0),p       # per-EC-column max score (index aligns with mask cols)
def rxn_score(ecmax):
    # per reaction: max dpz over its ECs (0 if none)
    sc=np.zeros(NR)
    for j in range(NR):
        ei=np.where(mask[j]==1)[0]
        if len(ei): sc[j]=ecmax[ei].max()
    return sc
_gc={}
def setup(gram):
    if gram in _gc: return _gc[gram]
    lt,ut=load_tight_bounds(data_path(f'tight_bounds_v6_{gram}.pkl')); lb=np.maximum(lb0,lt); ub=np.minimum(ub0,ut)
    oi=allrxns.index('biomass_GmPos' if gram=='pos' else 'biomass_GmNeg')
    exc=find_excluded_reactions(S,lb,ub,allrxns,allrxns[oi]); lb,ub,_,_=apply_media(['default'],allrxns,lb,ub)
    _gc[gram]=(lb,ub,oi,exc); return _gc[gram]
def sol_mask(path):
    if not os.path.exists(path): return None
    return np.array(pickle.load(open(path,'rb'))['y_vals'])>0.5
def baseline_mask(gram,ecmax):
    lb,ub,oi,exc=setup(gram); hot=set(np.where(ecmax>=0.5)[0]); draft=np.zeros(NR,bool)
    for j in range(NR):
        ei=np.where(mask[j]==1)[0]
        if len(ei) and any(e in hot for e in ei): draft[j]=True
    avail=np.ones(NR,bool); avail[list(exc)]=False; core=draft&avail
    sup=grow_support(S,lb,ub,oi,avail,0.1,core=core); return core|(sup if sup is not None else np.zeros(NR,bool))
def gem_mnxr(gem):
    m=cobra.io.read_sbml_model(f'{GEMDIR}/{gem}.xml'); A=set()
    for r in m.reactions:
        b=r.id[2:] if r.id.startswith('R_') else r.id
        if b in bigg2mnxr: A.add(bigg2mnxr[b])
    return A
GEM2G={'iML1515':('GCF_058436375.1','neg','E.coli'),'STM_v1_0':('GCF_000006945.2','neg','Salmonella'),
 'iYL1228':('GCF_058435815.1','neg','K.pneumoniae'),'iJN1463':('GCF_045571375.1','neg','P.putida'),
 'iYS854':('GCF_045348045.1','pos','S.aureus'),'iYO844':('GCF_058182495.1','pos','B.subtilis')}
rows=[]
print('=== REACTION-level weak-real (SEED rxns, GEM via MNXR) ===')
print(f'{"organism":13s}| {"Wrxn":5s}| {"v7":5s} {"v8":5s} {"base":5s}  (recovery)   | prec v7/v8/base')
for gem,(gcf,gram,name) in GEM2G.items():
    ecmax,pred=dpz(gcf); rs=rxn_score(ecmax); A=gem_mnxr(gem)
    inGEM=np.array([ (seedidx_mnxr[j] in A) if seedidx_mnxr[j] else False for j in range(NR)])
    weak=(rs>0)&(rs<0.5)
    Wrxn=weak&inGEM                                # weak-signal reactions that ARE in the GEM (by MNXR)
    masks={'v7':sol_mask(f'{V7}/meteor_sol_{gcf}.pkl'),'v8':sol_mask(f'{V8}/meteor_sol_{gcf}.pkl'),
           'base':baseline_mask(gram,ecmax)}
    rec={}; prec={}
    for k,mk in masks.items():
        if mk is None: rec[k]=None; prec[k]=None; continue
        rec[k]=round(int((Wrxn&mk).sum())/max(1,int(Wrxn.sum())),3)
        act_map=mk & np.array([seedidx_mnxr[j] is not None for j in range(NR)])
        act_mnxr={seedidx_mnxr[j] for j in np.where(mk)[0] if seedidx_mnxr[j]}
        prec[k]=round(len(act_mnxr&A)/max(1,len(act_mnxr)),3)
    print(f'{name:13s}| {int(Wrxn.sum()):5d}| {rec["v7"]:.2f}  {rec["v8"]:.2f}  {rec["base"]:.2f}          | {prec["v7"]:.2f}/{prec["v8"]:.2f}/{prec["base"]:.2f}')
    rows.append(dict(organism=name,gram=gram,nWrxn=int(Wrxn.sum()),recovery=rec,precision_mnxr=prec))
json.dump(rows,open(f'{OUT}/rxn_weakreal.json','w'),indent=1)
def mn(k,d):
    v=[r[d][k] for r in rows if r[d][k] is not None]; return np.mean(v) if v else float('nan')
print(f'\n--- MEANS (n=6) --- mean |Wrxn|={np.mean([r["nWrxn"] for r in rows]):.0f}')
print(f'  recovery:      v7={mn("v7","recovery"):.2f}  v8={mn("v8","recovery"):.2f}  base={mn("base","recovery"):.2f}')
print(f'  precision(MNXR): v7={mn("v7","precision_mnxr"):.3f}  v8={mn("v8","precision_mnxr"):.3f}  base={mn("base","precision_mnxr"):.3f}')
print('-> results/toolcompare/rxn_weakreal.json')
