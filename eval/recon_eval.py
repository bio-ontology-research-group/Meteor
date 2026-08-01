#!/usr/bin/env python3
"""Evaluate Reconstructor-with-dpz models: n_rxn, empty-GPR(unsupported) frac, EC-level P/R vs curated GEM.
recon models = ModelSEED namespace; rxn->EC via the SAME Unique_ModelSEED_Reaction_ECs.txt (r2ecf).
GEM ECs from BiGG annotation. EC-level = namespace-neutral."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import external_path
import sys,os,re,json,numpy as np,pandas as pd,cobra
import warnings,logging; warnings.filterwarnings('ignore'); logging.getLogger('cobra').setLevel(logging.ERROR)
B='/ibex/scratch/projects/c2014/kexin/funcarve'
MODELS=f'{B}/meteor_v8/results/toolcompare/recon_models'
GEMDIR=external_path('curated_gems')
R2ECF=external_path('Unique_ModelSEED_Reaction_ECs.txt')
OUT=f'{B}/meteor_v8/results/toolcompare'
FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')
# rxn(ModelSEED) -> set(EC)
seed2ec={}
e=pd.read_csv(R2ECF,sep='\t')
for _,r in e.iterrows():
    ec=str(r['External ID']).strip()
    if FULL.match(ec): seed2ec.setdefault(str(r['ModelSEED ID']).strip(),set()).add(ec)
def recon_ecs_and_stats(p):
    m=cobra.io.read_sbml_model(p)
    metab=[r for r in m.reactions if not (r.id.startswith(('EX_','DM_','SK_','sink_')) or 'biomass' in r.id.lower() or 'cellwall' in r.id.lower())]
    empty=sum(1 for r in metab if not r.gene_reaction_rule.strip())
    ecs=set()
    for r in metab:
        base=r.id.split('_')[0]              # rxn00001_c -> rxn00001
        if base in seed2ec: ecs|=seed2ec[base]
    try: gr=float(m.slim_optimize()); gr=0.0 if (gr is None or np.isnan(gr)) else gr
    except Exception: gr=float('nan')
    return dict(n_rxn=len(metab),empty_gpr=empty,unsupported_frac=round(empty/max(1,len(metab)),3),
                growth=round(gr,4),grows=bool(gr>1e-6)),ecs
def gem_ecs(gem):
    m=cobra.io.read_sbml_model(f'{GEMDIR}/{gem}.xml'); ex=set()
    for r in m.reactions:
        a=r.annotation.get('ec-code') if hasattr(r,'annotation') else None
        if not a: continue
        for x in (a if isinstance(a,list) else [a]):
            if FULL.match(str(x).strip()): ex.add(str(x).strip())
    return ex
def pr(P,G): i=len(P&G); return round(i/max(1,len(P)),3),round(i/max(1,len(G)),3)
GEM2G={'iML1515':('GCF_058436375.1','E.coli'),'STM_v1_0':('GCF_000006945.2','Salmonella'),
 'iYL1228':('GCF_058435815.1','K.pneumoniae'),'iJN1463':('GCF_045571375.1','P.putida'),
 'iYS854':('GCF_045348045.1','S.aureus'),'iYO844':('GCF_058182495.1','B.subtilis')}
print('=== Reconstructor(dpz) vs curated GEM (EC-level) ===')
print(f'{"organism":14s}{"GEM_EC":7s}{"REC n_rxn unsup% grow":24s}| {"REC EC P/R":14s}')
rows=[]
for gem,(gcf,name) in GEM2G.items():
    p=f'{MODELS}/{gcf}.sbml'
    if not os.path.exists(p): print(f'{name:14s} MISSING {gcf}'); continue
    st,E=recon_ecs_and_stats(p); G=gem_ecs(gem); P,Rc=pr(E,G)
    print(f'{name:14s}{len(G):7d}{st["n_rxn"]:6d} {100*st["unsupported_frac"]:4.1f}% {"Y" if st["grows"] else "n"}     | P={P:.3f} R={Rc:.3f} (n_ec {len(E)})')
    rows.append(dict(organism=name,gem=gem,gcf=gcf,gem_ec=len(G),n_rxn=st['n_rxn'],
        unsupported_frac=st['unsupported_frac'],grows=st['grows'],n_ec=len(E),P=P,R=Rc))
json.dump(rows,open(f'{OUT}/recon_dpz_ec.json','w'),indent=1)
if rows:
    print(f'\n--- means (n={len(rows)}) ---')
    print(f'  recon(dpz): n_rxn={np.mean([r["n_rxn"] for r in rows]):.0f} unsup%={100*np.mean([r["unsupported_frac"] for r in rows]):.1f} '
          f'P={np.mean([r["P"] for r in rows]):.3f} R={np.mean([r["R"] for r in rows]):.3f} grows={sum(r["grows"] for r in rows)}/{len(rows)}')
print('-> results/toolcompare/recon_dpz_ec.json')
