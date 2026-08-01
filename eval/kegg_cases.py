#!/usr/bin/env python3
"""KEGG-based classification of METEOR's MISSED weak-real ECs (user hypotheses).
For each missed weak-real EC e (0<dpz<0.5, in curated GEM, not in METEOR model), classify via KEGG pathways:
  peripheral   : e in NO metabolic KEGG pathway (secondary/isolated -> MILP never includes)
  case1_altrec : a same-pathway alternative EC is recovered by METEOR (function covered by stronger-evidence route)
  case2_deadpath: e's richest pathway has <3 confidently-predicted (dpz>=0.5) ECs (signal-dead -> unreconstructable)
  genuine_miss : has pathway + signal + no alternative -> real evw failure
Compare the miss breakdown across predictors (dpz/clean/enzbert). Hypothesis: clean/enzbert have MORE
case2_deadpath (their weak signal sits on signal-poor pathways), explaining the null weak-real advantage."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, external_path, data_dir
import sys,os,re,json,pickle,numpy as np

from meteor_v8.utils import load_universal,build_rxn_ec_mask,extract_pred,load_refmapping,load_ec

from baseline_io import resolve_baseline_pkl,BASELINE_SUFFIX
import cobra,warnings,logging; warnings.filterwarnings('ignore'); logging.getLogger('cobra').setLevel(logging.ERROR)
B='/ibex/scratch/projects/c2014/kexin/funcarve'; FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')
OUT=f'{B}/meteor_v8/results/toolcompare'; GEMDIR=external_path('curated_gems'); V8ROOT=f'{B}/meteor_v8_evw_p2mu3_run/meteor_out'
path2ec=json.load(open(f'{B}/meteor_v7_release/data/kegg_path2ec_metabolic.json'))
ec2path={}
for p,ecs in path2ec.items():
    for e in ecs: ec2path.setdefault(e,set()).add(p)
universal,allrxns,allmet=load_universal()
seedr2ec,_=load_refmapping('data'); seedr2ec={k:v for k,v in seedr2ec.items() if v}
anc=load_ec('data/all_ancestors.txt'); mask=build_rxn_ec_mask(allrxns,seedr2ec,anc)
_pc=None
def scores_and_pc(pred,acc):
    global _pc
    p=extract_pred(resolve_baseline_pkl(pred,'vanilla',acc,BASELINE_SUFFIX[pred]),anc)
    _pc=[str(c).split(':')[-1] for c in p.columns]
    mx=p.values.max(axis=0)
    return {_pc[j]:float(mx[j]) for j in range(len(mx)) if FULL.match(_pc[j])}
def meteor_ecs(pred,acc):
    p=f'{V8ROOT}/{pred}_vanilla/meteor_sol_{acc}.pkl'
    if not os.path.exists(p): return None
    act=np.array(pickle.load(open(p,'rb'))['y_vals'])>0.5; E=set()
    for j in np.where(act)[0]:
        for e in np.where(mask[j]==1)[0]:
            if e<len(_pc) and FULL.match(_pc[e]): E.add(_pc[e])
    return E
def gem_ecs(gem):
    m=cobra.io.read_sbml_model(f'{GEMDIR}/{gem}.xml'); ex=set()
    for r in m.reactions:
        a=r.annotation.get('ec-code') if hasattr(r,'annotation') else None
        if not a: continue
        for x in (a if isinstance(a,list) else [a]):
            if FULL.match(str(x).strip()): ex.add(str(x).strip())
    return ex
GEM2G={'iML1515':'GCF_058436375.1','STM_v1_0':'GCF_000006945.2','iYL1228':'GCF_058435815.1',
 'iJN1463':'GCF_045571375.1','iYS854':'GCF_045348045.1','iYO844':'GCF_058182495.1'}
CATS=['peripheral','case1_altrec','case2_deadpath','genuine_miss']
res={}
print('=== KEGG case classification of MISSED weak-real ECs (per predictor) ===')
print(f'{"pred":8s} {"|W|":4s} {"rec%":5s} | miss breakdown: peripheral / case1_alt / case2_deadpath / genuine')
for pred in ['dpz','clean','enzbert']:
    agg={c:0 for c in CATS}; totW=0; totrec=0; totmiss=0
    for gem,acc in GEM2G.items():
        sc=scores_and_pc(pred,acc); M=meteor_ecs(pred,acc); G=gem_ecs(gem)
        if M is None: continue
        W={e for e,s in sc.items() if 0<s<0.5} & G
        missed=W-M; totW+=len(W); totrec+=len(W&M); totmiss+=len(missed)
        for e in missed:
            paths=ec2path.get(e,set())
            if not paths: agg['peripheral']+=1; continue
            sub='.'.join(e.split('.')[:3]); subalt={e2 for e2 in M if e2!=e and '.'.join(e2.split('.')[:3])==sub}
            if subalt: agg['case1_altrec']+=1; continue
            richest=max(sum(1 for e2 in path2ec[p] if sc.get(e2,0)>=0.5) for p in paths)
            if richest<5: agg['case2_deadpath']+=1
            else: agg['genuine_miss']+=1
    def pct(c): return 100*agg[c]/max(1,totmiss)
    print(f'{pred:8s} {totW:4d} {100*totrec/max(1,totW):4.0f}% | {pct("peripheral"):4.0f}% / {pct("case1_altrec"):4.0f}% / {pct("case2_deadpath"):4.0f}% / {pct("genuine_miss"):4.0f}%  (miss={totmiss})')
    res[pred]=dict(nW=totW,recovered=totrec,missed=totmiss,breakdown=agg)
json.dump(res,open(f'{OUT}/kegg_cases.json','w'),indent=1)
print('\nkey: if clean/enzbert have higher case2_deadpath -> their weak signal is on signal-poor pathways (explains B3 null)')
print('-> results/toolcompare/kegg_cases.json')
