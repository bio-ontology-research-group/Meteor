import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, external_path, data_dir
import sys,os,pickle,json,glob,numpy as np,cobra
from meteor_v8.utils import load_universal, load_refmapping, load_ec, build_rxn_ec_mask, extract_pred
bigg2mnxr={}; seed2mnxr={}
for ln in open(external_path('reac_xref.tsv')):
    if ln.startswith('#'): continue
    p=ln.rstrip().split('\t')
    if len(p)<2 or not p[1].startswith('MNXR'): continue
    if p[0].startswith('bigg.reaction:'): bigg2mnxr.setdefault(p[0].split(':',1)[1],p[1])
    elif p[0].startswith('seed.reaction:'): seed2mnxr.setdefault(p[0].split(':',1)[1],p[1])
GEMDIR=external_path('curated_gems')
universal,allrxns,allmet=load_universal()
seedidx_mnxr={j:seed2mnxr.get(allrxns[j].split('_')[0]) for j in range(len(allrxns))}
seedr2ec,_=load_refmapping('data'); seedr2ec={k:v for k,v in seedr2ec.items() if v}
anc=load_ec('data/all_ancestors.txt'); mask=build_rxn_ec_mask(allrxns,seedr2ec,anc)
sys.path.insert(0,'/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/eval'); from baseline_io import resolve_baseline_pkl
def gem_mnxr(g):
    m=cobra.io.read_sbml_model(f'{GEMDIR}/{g}.xml'); A=set()
    for r in m.reactions:
        b=r.id[2:] if r.id.startswith('R_') else r.id
        if b in bigg2mnxr: A.add(bigg2mnxr[b])
    return A,len(m.reactions)
def act_mnxr(mb): return {seedidx_mnxr[j] for j in np.where(mb)[0] if seedidx_mnxr[j]}
def prj(A,B): i=len(A&B); return i/max(1,len(A)),i/max(1,len(B))  # precision(A=pred), recall(B=GEM)
GEM,GCF='iML1515','GCF_058436375.1'  # E.coli
A,ngem=gem_mnxr(GEM)
print(f'=== {GEM} ({GCF}): GEM {ngem} rxns -> {len(A)} MNXR-mapped ===')
sol=pickle.load(open(f'/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla/meteor_sol_{GCF}.pkl','rb'))
yv=np.array(sol['y_vals']); met=yv>0.5
# baseline threshold: reactions whose EC has pred>=0.5
pred=extract_pred(resolve_baseline_pkl('dpz','vanilla',GCF,'DPZ'),anc); Pm=pred.values; hot=set(pred.columns[np.where(Pm.max(axis=0)>=0.5)[0]])
thr=np.zeros(len(allrxns),bool)
for j in range(len(allrxns)):
    ei=np.where(mask[j]==1)[0]
    if len(ei) and any(pred.columns[e] in hot for e in ei): thr[j]=True
for tag,mb in [('baseline@0.5',thr),('METEOR active',met)]:
    n=int(mb.sum()); mapped=sum(1 for j in np.where(mb)[0] if seedidx_mnxr[j]); mn=act_mnxr(mb)
    P,R=prj(mn,A)
    print(f'  {tag:14s}: {n} rxns, {mapped} MNXR-mapped ({100*mapped/max(1,n):.0f}%), unique-MNXR {len(mn)} | P(vs GEM)={P:.3f} R(vs GEM)={R:.3f}')
print(f'  GEM MNXR not recovered by METEOR: {len(A-act_mnxr(met))}/{len(A)}  | METEOR-MNXR not in GEM: {len(act_mnxr(met)-A)}')
# ===== BGC: are indicator ECs muted by METEOR? =====
print('\n=== BGC: indicator ECs active vs muted in METEOR (producer genome) ===')
ce=json.load(open('/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026/results/bgc_class_kegg_ec_dict.json'))
pr=pickle.load(open(f'/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla/meteor_preds_{GCF}.pkl','rb'))
active=set(pr['active_ecs']); muted=set(pr['muted_ecs'])
for cls,ecs in ce.items():
    ecs=set(ecs); na=len(ecs&active); nm=len(ecs&muted); nb=sum(1 for e in ecs if e in pred.columns and pred[e].max()>=0.5)
    print(f'  {cls:12s}: {len(ecs)} indicator ECs | baseline@0.5 predicted {nb} | METEOR active {na} muted {nm}')
