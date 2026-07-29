#!/usr/bin/env python3
"""K12 matched-definition pathway coverage on panel108. Compares METEOR active-EC set (size K)
against BOTH (a) the old baseline threshold set (score>=0.5) and (b) a SIZE-MATCHED baseline set
= the top-K SEED-representable ECs by max score. The matched delta isolates the network
contribution from the threshold-definition artifact flagged in K12."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys,os,re,json,pickle,numpy as np
from meteor_v8.utils import extract_pred,load_refmapping,load_ec

from baseline_io import resolve_baseline_pkl,BASELINE_SUFFIX
B='/ibex/scratch/projects/c2014/kexin/funcarve'; FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')
V8ROOT=f'{B}/meteor_v8_evw_p2mu3_run/meteor_out'; OUT=f'{B}/meteor_v8/results/toolcompare'
anc=load_ec(data_path('all_ancestors.txt'))
seedr2ec,_=load_refmapping(data_dir()); seedr2ec={k:v for k,v in seedr2ec.items() if v}
seed_ec=set()
for ecs in seedr2ec.values():
    for e in ecs:
        e=str(e).split(':')[-1]
        if FULL.match(e): seed_ec.add(e)
path2ec=json.load(open(f'{B}/meteor_v7_release/data/kegg_path2ec_metabolic.json'))
pw_denom={}
for p,ecs in path2ec.items():
    d={str(e).split(':')[-1] for e in ecs if FULL.match(str(e).split(':')[-1])} & seed_ec
    if len(d)>=5: pw_denom[p]=d
panel108=[l.split()[0] for l in open(f'{B}/meteor_v7_run/downstream_results/panel108_gram.tsv') if l.strip()]
print(f'pathways={len(pw_denom)} genomes={len(panel108)}',flush=True)

def base_scores(baseline,gcf):
    """dict ec->max score over proteome, restricted to SEED-representable 4-digit ECs."""
    p=extract_pred(resolve_baseline_pkl(baseline,'vanilla',gcf,BASELINE_SUFFIX[baseline]),anc)
    cols=[str(c).split(':')[-1] for c in p.columns]; mx=p.values.max(axis=0)
    return {cols[j]:float(mx[j]) for j in range(len(cols)) if FULL.match(cols[j]) and cols[j] in seed_ec}
def meteor_ecs(baseline,gcf):
    f=f'{V8ROOT}/{baseline}_vanilla/meteor_preds_{gcf}.pkl'
    if not os.path.exists(f): return None
    d=pickle.load(open(f,'rb')); ae=d.get('active_ecs',set())
    return {str(e).split(':')[-1] for e in ae if FULL.match(str(e).split(':')[-1]) and str(e).split(':')[-1] in seed_ec}

res={}
print(f'{"baseline":9s} n  base_thr  base_topK  meteor  d_thr(pp)  d_matched(pp)')
for baseline in ['clean','dpz','enzbert']:
    d_thr=[]; d_mat=[]; bthr=[]; btopk=[]; mcov=[]; _pg=[]
    for gcf in panel108:
        M=meteor_ecs(baseline,gcf)
        if M is None: continue
        try: sc=base_scores(baseline,gcf)
        except Exception: continue
        K=len(M)
        Bthr={e for e,v in sc.items() if v>=0.5}
        BtopK=set(sorted(sc, key=sc.get, reverse=True)[:K])   # size-matched to METEOR
        bt=[]; bk=[]; mc=[]
        for p,denom in pw_denom.items():
            n=len(denom)
            bt.append(len(Bthr&denom)/n); bk.append(len(BtopK&denom)/n); mc.append(len(M&denom)/n)
        bt=np.mean(bt); bk=np.mean(bk); mc=np.mean(mc)
        bthr.append(bt); btopk.append(bk); mcov.append(mc); d_thr.append(mc-bt); d_mat.append(mc-bk)
        _pg.append((gcf,round(bt,4),round(bk,4),round(mc,4),K))
    r=dict(n=len(mcov),base_thr=round(float(np.mean(bthr)),4),base_topK=round(float(np.mean(btopk)),4),
           meteor=round(float(np.mean(mcov)),4),delta_thr_pp=round(100*float(np.mean(d_thr)),2),
           delta_matched_pp=round(100*float(np.mean(d_mat)),2),
           n_matched_pos=int(sum(1 for x in d_mat if x>0)),n_matched_neg=int(sum(1 for x in d_mat if x<0)))
    res[baseline]=r
    print(f'{baseline:9s} {r["n"]:3d} {r["base_thr"]:.4f}  {r["base_topK"]:.4f}  {r["meteor"]:.4f}  '
          f'{r["delta_thr_pp"]:+.1f}      {r["delta_matched_pp"]:+.1f}  (matched +/-: {r["n_matched_pos"]}/{r["n_matched_neg"]})',flush=True)
    open(f'{OUT}/kegg108_matched_pergenome_{baseline}.tsv','w').write(
        'gcf\tbase_thr\tbase_topK\tmeteor\tK\n'+''.join(f'{g}\t{a}\t{b2}\t{m}\t{k}\n' for g,a,b2,m,k in _pg))
json.dump(dict(panel='panel108',n_pathways=len(pw_denom),results=res),
          open(f'{OUT}/kegg_pathway_108_matched.json','w'),indent=1)
print('-> kegg_pathway_108_matched.json',flush=True)
