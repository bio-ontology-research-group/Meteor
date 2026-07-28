#!/usr/bin/env python3
"""KEGG pathway completion on panel108 (GCF), v8 evw. Replicates eval_pathway_gc_dpz.py logic but on the
GCF panel108 and v8 meteor_preds. Baseline coverage = fraction of a pathway's SEED-representable ECs scored
>=0.5 in the proteome; METEOR coverage = fraction whose ECs are in v8 active_ecs. delta = meteor - baseline.
Pathways restricted to >=5 SEED-representable 4-digit ECs (matches the paper's 152-pathway denominator)."""
import sys,os,re,json,pickle,numpy as np
V6='/ibex/user/niuk0a/funcarve/cobra/v6'; sys.path.insert(0,V6); os.chdir(V6)
sys.path.insert(0,'/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/src')
from src.v6utils import extract_pred,load_refmapping,load_ec
sys.path.insert(0,'/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/eval')
from baseline_io import resolve_baseline_pkl,BASELINE_SUFFIX
B='/ibex/scratch/projects/c2014/kexin/funcarve'; FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')
V8ROOT=f'{B}/meteor_v8_evw_p2mu3_run/meteor_out'; OUT=f'{B}/meteor_v8/results/toolcompare'
anc=load_ec(f'{V6}/data/all_ancestors.txt')
seedr2ec,_=load_refmapping(f'{V6}/data'); seedr2ec={k:v for k,v in seedr2ec.items() if v}
# SEED EC universe (4-digit ECs represented by >=1 SEED reaction)
seed_ec=set()
for ecs in seedr2ec.values():
    for e in ecs:
        e=str(e).split(':')[-1]
        if FULL.match(e): seed_ec.add(e)
path2ec=json.load(open(f'{B}/meteor_v7_release/data/kegg_path2ec_metabolic.json'))
# restrict pathways to those with >=5 SEED-representable 4-digit ECs; denom = SEED-representable ECs
pw_denom={}
for p,ecs in path2ec.items():
    d={str(e).split(':')[-1] for e in ecs if FULL.match(str(e).split(':')[-1])} & seed_ec
    if len(d)>=5: pw_denom[p]=d
print(f'pathways with >=5 SEED-representable ECs: {len(pw_denom)}',flush=True)
panel108=[l.split()[0] for l in open(f'{B}/meteor_v7_run/downstream_results/panel108_gram.tsv') if l.strip()]
print(f'panel108 genomes: {len(panel108)}',flush=True)
def base_ecs(baseline,gcf):
    p=extract_pred(resolve_baseline_pkl(baseline,'vanilla',gcf,BASELINE_SUFFIX[baseline]),anc)
    cols=[str(c).split(':')[-1] for c in p.columns]; mx=p.values.max(axis=0)
    return {cols[j] for j in range(len(cols)) if mx[j]>=0.5 and FULL.match(cols[j])}
def meteor_ecs(baseline,gcf):
    f=f'{V8ROOT}/{baseline}_vanilla/meteor_preds_{gcf}.pkl'
    if not os.path.exists(f): return None
    d=pickle.load(open(f,'rb')); ae=d.get('active_ecs',set())
    return {str(e).split(':')[-1] for e in ae if FULL.match(str(e).split(':')[-1])}
res={}
print(f'{"baseline":9s} n_gcf  base_cov  meteor_cov  delta(pp)')
for baseline in ['clean','dpz','enzbert']:
    dbl=[]; dml=[]; _pg=[]
    for gcf in panel108:
        M=meteor_ecs(baseline,gcf)
        if M is None: continue
        try: Bset=base_ecs(baseline,gcf)
        except Exception: continue
        bcovs=[]; mcovs=[]
        for p,denom in pw_denom.items():
            n=len(denom)
            bcovs.append(len(Bset&denom)/n); mcovs.append(len(M&denom)/n)
        dbl.append(np.mean(bcovs)); dml.append(np.mean(mcovs)); _pg.append((gcf,round(float(np.mean(bcovs)),4),round(float(np.mean(mcovs)),4)))
    bc=np.mean(dbl); mc=np.mean(dml)
    print(f'{baseline:9s} {len(dbl):4d}  {bc:.4f}   {mc:.4f}   {100*(mc-bc):+.1f}',flush=True)
    open(f'{OUT}/kegg108_pergenome_{baseline}.tsv','w').write('gcf\tbase_cov\tmeteor_cov\n'+''.join(f'{g}\t{b}\t{m}\n' for g,b,m in _pg))
    res[baseline]=dict(n_gcf=len(dbl),base_cov=round(bc,4),meteor_cov=round(mc,4),delta_pp=round(100*(mc-bc),2))
json.dump(dict(n_pathways=len(pw_denom),panel='panel108',results=res),open(f'{OUT}/kegg_pathway_108.json','w'),indent=1)
print('-> results/toolcompare/kegg_pathway_108.json')
