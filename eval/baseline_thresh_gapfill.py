#!/usr/bin/env python3
"""DECISIVE same-input ablation: threshold-1/0 draft + gapfill  vs  METEOR (evw).
Both use the SAME predictor (dpz) and SAME namespace (SEED) -> isolates the value of
evw soft-evidence cost over naive hard-threshold + blind parsimony gap-fill.
baseline: reactions with any EC pred>=0.5 -> draft (hard 1/0); grow_support() adds the
minimal-flux reaction set to make biomass feasible (parsimony gap-fill, NO evidence weighting).
METEOR: the evw MILP solution (already computed).
"""
import sys,os,json,pickle,re,numpy as np,cobra
import warnings,logging; warnings.filterwarnings('ignore'); logging.getLogger('cobra').setLevel(logging.ERROR)
V6='/ibex/user/niuk0a/funcarve/cobra/v6'; sys.path.insert(0,V6); os.chdir(V6)
B='/ibex/scratch/projects/c2014/kexin/funcarve'
sys.path.insert(0,f'{B}/meteor_v8/src')
from src.v6utils import (load_universal,extract_fba_matrices,load_tight_bounds,apply_media,
    find_excluded_reactions,build_rxn_ec_mask,extract_pred,load_refmapping,load_ec,_detect_solver)
from meteor_v8.repair import grow_support, maxbio_active
sys.path.insert(0,f'{B}/meteor_v8/eval')
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
MO=f'{B}/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla'
GEMDIR=f'{B}/meteor_diag/curated_gems'
OUT=f'{B}/meteor_v8/results/toolcompare'; os.makedirs(OUT,exist_ok=True)
FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')

universal,allrxns,allmet=load_universal()
seedr2ec,_=load_refmapping('data'); seedr2ec={k:v for k,v in seedr2ec.items() if v}
anc=load_ec('data/all_ancestors.txt'); mask=build_rxn_ec_mask(allrxns,seedr2ec,anc)
# rxn -> set of complete ECs (for EC-level accuracy)
rxn_ecs=[{c for c in [ (anc_list[e] if False else None) ] } for _ in range(0)]  # placeholder
ec_names=None
S,lb0,ub0=extract_fba_matrices(universal,allrxns,reversed_trans=True)
solver,_=_detect_solver(threads=4,time_limit=600)
NR=len(allrxns)

# EC column names aligned to mask columns: extract_pred columns == anc order. Grab from one pred.
_pred_cols=None
def get_pred(acc):
    global _pred_cols
    p=extract_pred(resolve_baseline_pkl('dpz','vanilla',acc,BASELINE_SUFFIX['dpz']),anc)
    if _pred_cols is None: _pred_cols=list(p.columns)
    return p
def rxn_ec_set(j):
    ei=np.where(mask[j]==1)[0]
    out=set()
    for e in ei:
        if e<len(_pred_cols):
            c=str(_pred_cols[e])
            if FULL.match(c): out.add(c)
    return out

_gram_cache={}
def setup_gram(gram):
    if gram in _gram_cache: return _gram_cache[gram]
    lt,ut=load_tight_bounds(f'{V6}/data/tight_bounds_v6_{gram}.pkl')
    lb=np.maximum(lb0,lt); ub=np.minimum(ub0,ut)
    bid='biomass_GmPos' if gram=='pos' else 'biomass_GmNeg'
    oi=allrxns.index(bid); exc=find_excluded_reactions(S,lb,ub,allrxns,bid)
    lb,ub,_,media=apply_media(['default'],allrxns,lb,ub)
    _gram_cache[gram]=(lb,ub,oi,exc,media); return _gram_cache[gram]

def baseline_model(acc,gram):
    lb,ub,oi,exc,media=setup_gram(gram)
    pred=get_pred(acc); ecmax=pred.values.max(axis=0)
    hot=set(np.where(ecmax>=0.5)[0])
    draft=np.zeros(NR,bool); ev=np.zeros(NR,bool)
    for j in range(NR):
        ei=np.where(mask[j]==1)[0]
        if len(ei)==0: continue
        if any(e in hot for e in ei): draft[j]=True
        if ecmax[ei].max()>0: ev[j]=True
    avail=np.ones(NR,bool); avail[list(exc)]=False   # feasible universe for gap-fill
    core=draft & avail
    sup=grow_support(S,lb,ub,oi,avail,0.1,core=core)
    if sup is None:                 # fallback: allow everything
        sup=grow_support(S,lb,ub,oi,np.ones(NR,bool),0.1,core=core)
    model = core | (sup if sup is not None else np.zeros(NR,bool))
    added = model & ~core           # gap-filled (added for feasibility)
    unsupported = model & ~ev       # reactions with NO sequence evidence
    bf=maxbio_active(S,lb,ub,oi,model)
    return dict(n_rxn=int(model.sum()), n_draft=int(core.sum()), n_gapfill=int(added.sum()),
                unsupported=int(unsupported.sum()),
                unsupported_frac=round(int(unsupported.sum())/max(1,int(model.sum())),3),
                growth=round(float(bf),4), grows=bool(bf>1e-6), _model=model, _ev=ev)

def meteor_model(acc,gram):
    lb,ub,oi,exc,media=setup_gram(gram)
    sol=pickle.load(open(f'{MO}/meteor_sol_{acc}.pkl','rb')); act=np.array(sol['y_vals'])>0.5
    get_pred(acc); ecmax=extract_pred(resolve_baseline_pkl('dpz','vanilla',acc,BASELINE_SUFFIX['dpz']),anc).values.max(axis=0)
    ev=np.zeros(NR,bool)
    for j in np.where(act)[0]:
        ei=np.where(mask[j]==1)[0]
        if len(ei) and ecmax[ei].max()>0: ev[j]=True
    unsupported=act & ~ev
    return dict(n_rxn=int(act.sum()), unsupported=int(unsupported.sum()),
                unsupported_frac=round(int(unsupported.sum())/max(1,int(act.sum())),3),
                growth=round(float(sol.get('biomass_flux',0) or 0),4),
                grows=bool((sol.get('biomass_flux',0) or 0)>1e-6), _model=act)

def model_ecs(mb):
    out=set()
    for j in np.where(mb)[0]: out|=rxn_ec_set(j)
    return out
def gem_ecs(gem):
    m=cobra.io.read_sbml_model(f'{GEMDIR}/{gem}.xml'); ex=set()
    for r in m.reactions:
        e=r.annotation.get('ec-code') if hasattr(r,'annotation') else None
        if not e: continue
        for x in (e if isinstance(e,list) else [e]):
            if FULL.match(str(x).strip()): ex.add(str(x).strip())
    return ex
def pr(P,G): i=len(P&G); return round(i/max(1,len(P)),3), round(i/max(1,len(G)),3)

decoy=[(l.split()[0],l.split()[1][:3]) for l in open(f'{B}/meteor_v8/tools/recabl20.manifest') if l.strip()]
GEM2G={'iML1515':('GCF_058436375.1','neg','E.coli'),'STM_v1_0':('GCF_000006945.2','neg','Salmonella'),
 'iYL1228':('GCF_058435815.1','neg','K.pneumoniae'),'iJN1463':('GCF_045571375.1','neg','P.putida'),
 'iYS854':('GCF_045348045.1','pos','S.aureus'),'iYO844':('GCF_058182495.1','pos','B.subtilis')}
gmap={acc:gram for acc,gram in decoy}
for gem,(gcf,gram,name) in GEM2G.items(): gmap.setdefault(gcf,gram)

print('=== DECISIVE ABLATION: threshold-1/0 draft + gap-fill  vs  METEOR (evw), same predictor ===')
print(f'{"genome":18s}{"gram":4s}| {"BASELINE n_rxn(gapfill) unsup% grow":38s}| {"METEOR n_rxn unsup% grow":26s}')
rows=[]
for acc,gram in sorted(gmap.items()):
    try:
        bm=baseline_model(acc,gram); mm=meteor_model(acc,gram)
    except Exception as e:
        print(f'{acc:18s}{gram:4s}| ERR {str(e)[:50]}'); continue
    print(f'{acc:18s}{gram:4s}| {bm["n_rxn"]:5d} (+{bm["n_gapfill"]:4d}) {100*bm["unsupported_frac"]:4.1f}% {"Y" if bm["grows"] else "n"}'
          f'    | {mm["n_rxn"]:5d} {100*mm["unsupported_frac"]:4.1f}% {"Y" if mm["grows"] else "n"}')
    rows.append(dict(acc=acc,gram=gram,
        baseline={k:v for k,v in bm.items() if not k.startswith('_')},
        meteor={k:v for k,v in mm.items() if not k.startswith('_')}))
json.dump(rows,open(f'{OUT}/ablation_thresh_vs_evw.json','w'),indent=1)
print(f'\n--- MEANS (n={len(rows)}) ---')
for t in ['baseline','meteor']:
    print(f'  {t:9s}: n_rxn={np.mean([r[t]["n_rxn"] for r in rows]):.0f}  '
          f'unsupported%={100*np.mean([r[t]["unsupported_frac"] for r in rows]):.1f}  '
          f'grows={sum(r[t]["grows"] for r in rows)}/{len(rows)}')
print(f'  baseline mean gap-filled added: {np.mean([r["baseline"]["n_gapfill"] for r in rows]):.0f}')

# EC-level accuracy vs GEM (recompute models to get masks)
print('\n=== EC-level accuracy vs curated GEM ===')
print(f'{"organism":14s}{"GEM_EC":7s}| {"BASELINE P/R":15s}| {"METEOR P/R":15s}')
brows=[]
for gem,(gcf,gram,name) in GEM2G.items():
    G=gem_ecs(gem); bm=baseline_model(gcf,gram); mm=meteor_model(gcf,gram)
    bE=model_ecs(bm['_model']); mE=model_ecs(mm['_model'])
    bP,bR=pr(bE,G); mP,mR=pr(mE,G)
    print(f'{name:14s}{len(G):7d}| P={bP:.3f} R={bR:.3f}   | P={mP:.3f} R={mR:.3f}')
    brows.append(dict(organism=name,gem_ec=len(G),baseline=dict(n_ec=len(bE),P=bP,R=bR),meteor=dict(n_ec=len(mE),P=mP,R=mR)))
json.dump(brows,open(f'{OUT}/ablation_thresh_vs_evw_ec.json','w'),indent=1)
print('\n--- EC means ---')
for t in ['baseline','meteor']:
    print(f'  {t:9s}: P={np.mean([r[t]["P"] for r in brows]):.3f} R={np.mean([r[t]["R"] for r in brows]):.3f} n_ec={np.mean([r[t]["n_ec"] for r in brows]):.0f}')
print('-> results/toolcompare/ablation_thresh_vs_evw{,_ec}.json')
