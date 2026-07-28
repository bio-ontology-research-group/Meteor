#!/usr/bin/env python3
"""METEOR vs CarveMe reconstruction comparison.
Panel A (20 decoy genomes): n_rxn / gap-fill(unsupported) fraction / growth.
Panel B (6 curated-GEM genomes): + MNXR-space P/R vs curated GEM.
All models mapped to MNXR for fair cross-namespace size/overlap (METEOR=SEED, CarveMe=BiGG).
gap-fill = reactions with NO sequence evidence (METEOR: active rxn no predicted EC; CarveMe: empty GPR).
"""
import sys,os,json,pickle,glob,numpy as np,cobra
import warnings; warnings.filterwarnings('ignore')
import logging; logging.getLogger('cobra').setLevel(logging.ERROR)
V6='/ibex/user/niuk0a/funcarve/cobra/v6'; sys.path.insert(0,V6); os.chdir(V6)
BASE='/ibex/scratch/projects/c2014/kexin/funcarve'
sys.path.insert(0,f'{BASE}/meteor_v8/src')
from src.v6utils import (load_universal,load_refmapping,load_ec,build_rxn_ec_mask,extract_pred,
    extract_fba_matrices,load_tight_bounds,apply_media)
sys.path.insert(0,f'{BASE}/meteor_v8/eval')
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX

MO=f'{BASE}/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla'
CV=f'{BASE}/paperA_2026/results/carveme_gc'
GEMDIR=f'{BASE}/meteor_diag/curated_gems'
OUT=f'{BASE}/meteor_v8/results/toolcompare'; os.makedirs(OUT,exist_ok=True)

# --- MNXR xref ---
bigg2mnxr={}; seed2mnxr={}
for ln in open(f'{BASE}/meteor_diag/reac_xref.tsv'):
    if ln.startswith('#'): continue
    p=ln.rstrip().split('\t')
    if len(p)<2 or not p[1].startswith('MNXR'): continue
    if p[0].startswith('bigg.reaction:'): bigg2mnxr.setdefault(p[0].split(':',1)[1],p[1])
    elif p[0].startswith('seed.reaction:'): seed2mnxr.setdefault(p[0].split(':',1)[1],p[1])

universal,allrxns,allmet=load_universal()
seedidx_mnxr={j:seed2mnxr.get(allrxns[j].split('_')[0]) for j in range(len(allrxns))}
seedr2ec,_=load_refmapping('data'); seedr2ec={k:v for k,v in seedr2ec.items() if v}
anc=load_ec('data/all_ancestors.txt'); mask=build_rxn_ec_mask(allrxns,seedr2ec,anc)

# FBA matrices + per-gram tight bounds (for METEOR growth double-check; sol already has biomass_flux)
S,lb0,ub0=extract_fba_matrices(universal,allrxns,reversed_trans=True)

cvfiles=os.listdir(CV)
def find_cv(acc):
    num=acc.split('_')[1]
    for f in cvfiles:
        if num in f and f.endswith('.xml'): return os.path.join(CV,f)
    return None

EX_PREF=('EX_','DM_','SK_','sink_')
def is_exch(rid):
    return rid.startswith(EX_PREF) or 'biomass' in rid.lower() or rid.lower().startswith('bio')

def carveme_metrics(acc):
    f=find_cv(acc)
    if not f: return None
    m=cobra.io.read_sbml_model(f)
    metab=[r for r in m.reactions if not is_exch(r.id)]
    empty=sum(1 for r in metab if not r.gene_reaction_rule.strip())
    mnxr=set()
    for r in metab:
        b=r.id[2:] if r.id.startswith('R_') else r.id
        if b in bigg2mnxr: mnxr.add(bigg2mnxr[b])
    try: gr=float(m.slim_optimize()); gr=0.0 if (gr is None or np.isnan(gr)) else gr
    except Exception: gr=float('nan')
    return dict(n_rxn=len(metab), n_mnxr=len(mnxr), gapfill=empty,
                gapfill_frac=round(empty/max(1,len(metab)),3), growth=round(gr,4),
                grows=bool(gr>1e-6), mnxr=mnxr)

def meteor_metrics(acc):
    p=f'{MO}/meteor_sol_{acc}.pkl'
    if not os.path.exists(p): return None
    sol=pickle.load(open(p,'rb')); y=np.array(sol['y_vals']); act=y>0.5
    aidx=np.where(act)[0]
    # evidence support: any EC of the reaction predicted (>0) by baseline dpz
    try:
        pred=extract_pred(resolve_baseline_pkl('dpz','vanilla',acc,BASELINE_SUFFIX['dpz']),anc)
        ecmax=pred.values.max(axis=0); ncol=len(ecmax)
    except Exception:
        ecmax=None
    unsup=0
    for j in aidx:
        ei=np.where(mask[j]==1)[0]
        if len(ei)==0: unsup+=1; continue
        if ecmax is None: continue
        if not any(e<ncol and ecmax[e]>0 for e in ei): unsup+=1
    mnxr={seedidx_mnxr[j] for j in aidx if seedidx_mnxr[j]}
    bf=float(sol.get('biomass_flux',0) or 0)
    return dict(n_rxn=int(act.sum()), n_mnxr=len(mnxr), gapfill=unsup,
                gapfill_frac=round(unsup/max(1,int(act.sum())),3), growth=round(bf,4),
                grows=bool(bf>1e-6), mnxr=mnxr)

def gem_mnxr(gem):
    m=cobra.io.read_sbml_model(f'{GEMDIR}/{gem}.xml'); A=set()
    for r in m.reactions:
        b=r.id[2:] if r.id.startswith('R_') else r.id
        if b in bigg2mnxr: A.add(bigg2mnxr[b])
    return A,len(m.reactions)

def pr(pred_set,gem_set):
    i=len(pred_set&gem_set)
    return round(i/max(1,len(pred_set)),3), round(i/max(1,len(gem_set)),3)

# --- genome sets ---
decoy=[(l.split()[0],l.split()[1][:3]) for l in open(f'{BASE}/meteor_v8/tools/recabl20.manifest') if l.strip()]
GEM2G={'iML1515':('GCF_058436375.1','E.coli'),'STM_v1_0':('GCF_000006945.2','Salmonella'),
 'iYL1228':('GCF_058435815.1','K.pneumoniae'),'iJN1463':('GCF_045571375.1','P.putida'),
 'iYS854':('GCF_045348045.1','S.aureus'),'iYO844':('GCF_058182495.1','B.subtilis')}

# ============ PANEL A: 20 decoy ============
print('=== PANEL A: METEOR vs CarveMe on 20 decoy genomes ===')
print(f'{"genome":18s} {"gram":4s} | {"MET n_rxn/mnxr gapfill% grow":30s} | {"CAR n_rxn/mnxr gapfill% grow":30s}')
rowsA=[]
for acc,gram in decoy:
    mm=meteor_metrics(acc); cm=carveme_metrics(acc)
    if not mm or not cm: print(f'{acc:18s} {gram:4s} | MISSING met={bool(mm)} car={bool(cm)}'); continue
    print(f'{acc:18s} {gram:4s} | {mm["n_rxn"]:5d}/{mm["n_mnxr"]:4d} {100*mm["gapfill_frac"]:4.1f}% {"Y" if mm["grows"] else "n"} '
          f'| {cm["n_rxn"]:5d}/{cm["n_mnxr"]:4d} {100*cm["gapfill_frac"]:4.1f}% {"Y" if cm["grows"] else "n"}')
    rowsA.append(dict(acc=acc,gram=gram,meteor={k:v for k,v in mm.items() if k!='mnxr'},
                      carveme={k:v for k,v in cm.items() if k!='mnxr'}))
json.dump(rowsA,open(f'{OUT}/panelA_decoy20.json','w'),indent=1)

def agg(rows,tool,key):
    v=[r[tool][key] for r in rows]; return np.mean(v)
print('\n--- PANEL A means (n=%d) ---'%len(rowsA))
for tool in ['meteor','carveme']:
    print(f'  {tool:8s}: n_rxn={agg(rowsA,tool,"n_rxn"):.0f}  n_mnxr={agg(rowsA,tool,"n_mnxr"):.0f}  '
          f'gapfill%={100*agg(rowsA,tool,"gapfill_frac"):.1f}  grows={sum(r[tool]["grows"] for r in rowsA)}/{len(rowsA)}')

# ============ PANEL B: 6 curated GEMs ============
print('\n=== PANEL B: METEOR vs CarveMe vs curated GEM (MNXR P/R) ===')
print(f'{"organism":14s} {"GEM":9s} {"GEMmnxr":7s} | {"METEOR P/R":14s} | {"CarveMe P/R":14s}')
rowsB=[]
for gem,(gcf,name) in GEM2G.items():
    A,ngem=gem_mnxr(gem)
    mm=meteor_metrics(gcf); cm=carveme_metrics(gcf)
    if not mm or not cm: print(f'{name:14s} {gem:9s} MISSING'); continue
    mP,mR=pr(mm['mnxr'],A); cP,cR=pr(cm['mnxr'],A)
    print(f'{name:14s} {gem:9s} {len(A):7d} | P={mP:.3f} R={mR:.3f} | P={cP:.3f} R={cR:.3f}')
    rowsB.append(dict(organism=name,gem=gem,gcf=gcf,gem_mnxr=len(A),
        meteor=dict(n_mnxr=mm['n_mnxr'],gapfill_frac=mm['gapfill_frac'],P=mP,R=mR,grows=mm['grows']),
        carveme=dict(n_mnxr=cm['n_mnxr'],gapfill_frac=cm['gapfill_frac'],P=cP,R=cR,grows=cm['grows'])))
json.dump(rowsB,open(f'{OUT}/panelB_curated6.json','w'),indent=1)
if rowsB:
    print('\n--- PANEL B means ---')
    for tool in ['meteor','carveme']:
        print(f'  {tool:8s}: P={np.mean([r[tool]["P"] for r in rowsB]):.3f}  R={np.mean([r[tool]["R"] for r in rowsB]):.3f}  '
              f'gapfill%={100*np.mean([r[tool]["gapfill_frac"] for r in rowsB]):.1f}')
print('\nDONE -> results/toolcompare/{panelA_decoy20,panelB_curated6}.json')
