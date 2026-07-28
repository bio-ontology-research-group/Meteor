#!/usr/bin/env python3
"""Pathway DETECTION comparison on panel108 (answers 'how many pathways does each method cover').
A pathway is 'detected' if >=1 of its SEED-representable ECs is present:
  baseline -> EC scored >=0.5;  METEOR -> EC in active set.
Reports, per baseline (mean over 108 genomes): pathways detected by baseline vs METEOR at
detection thresholds {>=1 EC, >=25%, >=50%}, plus the METEOR-only breakdown: of pathways METEOR
detects but baseline misses (at >=1 EC), how many are driven by a WEAK-real-signal EC (baseline
score in (0,0.5)) vs a PURE-GAPFILL EC (baseline score == 0)."""
import sys,os,re,json,pickle,numpy as np
V6='/ibex/user/niuk0a/funcarve/cobra/v6'; sys.path.insert(0,V6); os.chdir(V6)
from src.v6utils import extract_pred,load_refmapping,load_ec
sys.path.insert(0,'/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/eval')
from baseline_io import resolve_baseline_pkl,BASELINE_SUFFIX
B='/ibex/scratch/projects/c2014/kexin/funcarve'; FULL=re.compile(r'^\d+\.\d+\.\d+\.\d+$')
V8ROOT=f'{B}/meteor_v8_evw_p2mu3_run/meteor_out'; OUT=f'{B}/meteor_v8/results/toolcompare'
anc=load_ec(f'{V6}/data/all_ancestors.txt')
seedr2ec,_=load_refmapping(f'{V6}/data'); seedr2ec={k:v for k,v in seedr2ec.items() if v}
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
NPW=len(pw_denom)
panel108=[l.split()[0] for l in open(f'{B}/meteor_v7_run/downstream_results/panel108_gram.tsv') if l.strip()]
print(f'pathways={NPW} genomes={len(panel108)}',flush=True)

def base_scores(baseline,gcf):
    p=extract_pred(resolve_baseline_pkl(baseline,'vanilla',gcf,BASELINE_SUFFIX[baseline]),anc)
    cols=[str(c).split(':')[-1] for c in p.columns]; mx=p.values.max(axis=0)
    return {cols[j]:float(mx[j]) for j in range(len(cols)) if FULL.match(cols[j]) and cols[j] in seed_ec}
def meteor_ecs(baseline,gcf):
    f=f'{V8ROOT}/{baseline}_vanilla/meteor_preds_{gcf}.pkl'
    if not os.path.exists(f): return None
    d=pickle.load(open(f,'rb')); ae=d.get('active_ecs',set())
    return {str(e).split(':')[-1] for e in ae if FULL.match(str(e).split(':')[-1]) and str(e).split(':')[-1] in seed_ec}

res={}
print(f'{"baseline":9s} n  detect>=1 (B/M)   >=25% (B/M)   >=50% (B/M)   Monly  weakSig  pureGap')
for baseline in ['clean','dpz','enzbert']:
    D1b=[];D1m=[];D25b=[];D25m=[];D50b=[];D50m=[];MO=[];WS=[];PG=[]
    for gcf in panel108:
        M=meteor_ecs(baseline,gcf)
        if M is None: continue
        try: sc=base_scores(baseline,gcf)
        except Exception: continue
        Bset={e for e,v in sc.items() if v>=0.5}
        d1b=d1m=d25b=d25m=d50b=d50m=0; mo=ws=pg=0
        for p,denom in pw_denom.items():
            n=len(denom); ob=len(Bset&denom); om=len(M&denom)
            d1b+=ob>=1; d1m+=om>=1; d25b+=ob/n>=0.25; d25m+=om/n>=0.25; d50b+=ob/n>=0.5; d50m+=om/n>=0.5
            if om>=1 and ob==0:  # METEOR-only detected pathway
                mo+=1
                det=M&denom  # ECs that detect it
                if any(sc.get(e,0.0)>0.0 for e in det): ws+=1   # >=1 detecting EC had weak baseline signal
                else: pg+=1                                      # all detecting ECs had zero baseline score
        D1b.append(d1b);D1m.append(d1m);D25b.append(d25b);D25m.append(d25m);D50b.append(d50b);D50m.append(d50m)
        MO.append(mo);WS.append(ws);PG.append(pg)
    def m(x): return round(float(np.mean(x)),1)
    r=dict(n=len(D1b),npw=NPW,
           det1_B=m(D1b),det1_M=m(D1m),det25_B=m(D25b),det25_M=m(D25m),det50_B=m(D50b),det50_M=m(D50m),
           meteor_only=m(MO),weaksig=m(WS),puregap=m(PG))
    res[baseline]=r
    print(f'{baseline:9s} {r["n"]:3d}  {r["det1_B"]:.1f}/{r["det1_M"]:.1f}   '
          f'{r["det25_B"]:.1f}/{r["det25_M"]:.1f}   {r["det50_B"]:.1f}/{r["det50_M"]:.1f}   '
          f'{r["meteor_only"]:.1f}   {r["weaksig"]:.1f}    {r["puregap"]:.1f}',flush=True)
json.dump(dict(panel='panel108',n_pathways=NPW,results=res),
          open(f'{OUT}/kegg_pathway_108_detection.json','w'),indent=1)
print('-> kegg_pathway_108_detection.json',flush=True)
