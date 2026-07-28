#!/usr/bin/env python3
"""Statistical confidence for v8 main results: paired Wilcoxon signed-rank + bootstrap 95% CI,
across the 108 panel genomes (paired = same genome under two conditions). Genome-clustered (each genome
one paired observation), addressing K17. Output: results/toolcompare/stats_confidence.json."""
import glob,json,os,numpy as np
from scipy import stats
B='/ibex/scratch/projects/c2014/kexin/funcarve'; OUT=f'{B}/meteor_v8/results/toolcompare'
rng=np.random.RandomState(0)
def boot_ci(x, fn=np.mean, n=10000):
    x=np.asarray(x); bs=[fn(x[rng.randint(0,len(x),len(x))]) for _ in range(n)]
    return round(float(np.percentile(bs,2.5)),4), round(float(np.percentile(bs,97.5)),4)
def paired(a,b,name):
    a=np.asarray(a,float); b=np.asarray(b,float); d=a-b
    try: w=stats.wilcoxon(a,b); p=float(w.pvalue)
    except Exception: p=float('nan')
    lo,hi=boot_ci(d)
    return dict(name=name,n=len(a),mean_a=round(float(a.mean()),4),mean_b=round(float(b.mean()),4),
                mean_diff=round(float(d.mean()),4),ci95_diff=[lo,hi],wilcoxon_p=p,
                n_a_gt_b=int((d>0).sum()),n_b_gt_a=int((d<0).sum()))
res={}
# --- DECOY: per genome true vs uniform vs shuffled (spurious) ---
tv,uv,sv=[],[],[]
for f in glob.glob(f'{B}/meteor_v8/results/recovery_abl/*.json'):
    d=json.load(open(f))
    if all(k in d for k in ['true','uniform','shuffled']):
        tv.append(d['true']['decoy']); uv.append(d['uniform']['decoy']); sv.append(d['shuffled']['decoy'])
res['decoy_true_vs_uniform']=paired(uv,tv,'decoy uniform-true spurious (uniform>true expected)')
res['decoy_true_vs_shuffled']=paired(sv,tv,'decoy shuffled-true spurious')
# ratio CI
r=np.array(uv)/np.clip(np.array(tv),1e-9,None); res['decoy_uniform_over_true_ratio']=dict(mean=round(float(r.mean()),2),ci95=list(boot_ci(r)))
# --- KEGG pathway: per genome mean baseline vs METEOR coverage (dpz) ---
import csv
byg={}
for row in csv.DictReader(open(f'{B}/paperA_2026/results/pathway_completion_dpz_kegg.tsv'),delimiter='\t'):
    g=row['tax']; byg.setdefault(g,[[],[]])
    try: byg[g][0].append(float(row['baseline_coverage'])); byg[g][1].append(float(row['meteor_coverage']))
    except: pass
bb=[np.mean(v[0]) for v in byg.values() if v[0]]; mm=[np.mean(v[1]) for v in byg.values() if v[1]]
res['kegg_dpz_baseline_vs_meteor']=paired(mm,bb,'KEGG dpz METEOR-baseline coverage (per genome mean)')
# --- structural dead-ends: baseline vs METEOR (dpz), per genome ---
base_de={}; met_de={}
for f in glob.glob(f'{B}/meteor_v7_run/downstream_results/structural_gapfill_v2/*_dpz_gapfill.json'):
    d=json.load(open(f)); g=os.path.basename(f).split('_dpz_')[0]; base_de[g]=d.get('deadends_pre')
for f in glob.glob(f'{B}/meteor_diag/memote_evw_*.json'):
    d=json.load(open(f)); g=d.get('gca'); e=d.get('evw_p2mu3')
    if isinstance(e,dict): met_de[g]=e.get('deadends')
common=[g for g in base_de if g in met_de and base_de[g] is not None and met_de[g] is not None]
if common:
    res['deadends_baseline_vs_meteor_dpz']=paired([base_de[g] for g in common],[met_de[g] for g in common],'dead-ends baseline-METEOR dpz')
json.dump(res,open(f'{OUT}/stats_confidence.json','w'),indent=1)
print('=== v8 statistical confidence (paired, 108 genomes) ===')
for k,v in res.items():
    if 'wilcoxon_p' in v:
        print(f'{k}:\n  mean {v["mean_a"]} vs {v["mean_b"]} | diff {v["mean_diff"]} CI95 {v["ci95_diff"]} | Wilcoxon p={v["wilcoxon_p"]:.2e} | {v["n_a_gt_b"]}/{v["n"]} genomes')
    else:
        print(f'{k}: mean {v["mean"]} CI95 {v["ci95"]}')
print('-> results/toolcompare/stats_confidence.json')
