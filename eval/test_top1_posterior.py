import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys,os,json,pickle,glob,numpy as np
from meteor_v8.utils import extract_pred, load_ec

from baseline_io import resolve_baseline_pkl
anc=load_ec(data_path('all_ancestors.txt'))
CACHE='/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7_run/downstream_results/ncbi_ec_cache'
MO='/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla'
A=0.4
def fmax(gtidx, sc):
    n=len(gtidx)
    if n==0: return 0.0
    order=np.argsort(-sc); mask=np.zeros(len(sc),bool)
    for i in gtidx: mask[i]=True
    sm=mask[order]; tp=np.cumsum(sm).astype(float); npred=np.arange(1,len(sc)+1)
    P=tp/npred; R=tp/n; f=np.where(P+R>0,2*P*R/(P+R),0.0); return float(f.max())
def f105(gtidx,sc):
    pos=set(np.where(sc>=0.5)[0]); tp=len(gtidx&pos); r=tp/max(1,len(gtidx)); p=tp/len(pos) if pos else (1.0 if not gtidx else 0.0); return 2*p*r/(p+r) if p+r>0 else 0.0
gcfs=sorted(os.path.basename(p)[len('meteor_preds_'):-4] for p in glob.glob(f'{MO}/meteor_preds_*.pkl'))[:30]
agg={'base':[[],[]],'multi':[[],[]],'top1':[[],[]]}; ng=0
for gcf in gcfs:
    cf=f'{CACHE}/{gcf}_ec.json'
    if not os.path.exists(cf): continue
    ecmap={k:set(v) for k,v in json.load(open(cf)).items()}
    bp=resolve_baseline_pkl('dpz','vanilla',gcf)
    if not bp: continue
    try: pred=extract_pred(bp,anc); pr=pickle.load(open(f'{MO}/meteor_preds_{gcf}.pkl','rb'))
    except: continue
    active=set(pr['active_ecs']); muted=set(pr['muted_ecs'])
    cols=list(pred.columns); c2i={c:i for i,c in enumerate(cols)}; idx={p:i for i,p in enumerate(pred.index)}
    B=pred.values.astype(float); Pm=B.copy(); Pt=B.copy()
    for ec in active:
        if ec in c2i:
            j=c2i[ec]; cm=Pm[:,j]; Pm[:,j]=np.clip(cm+A*cm*(1-cm),0,1)
            ct=Pt[:,j]; jj=int(np.argmax(ct)); Pt[jj,j]=min(1.0,ct[jj]+A*ct[jj]*(1-ct[jj]))
    for ec in muted:
        if ec in c2i:
            j=c2i[ec]
            for Px in (Pm,Pt): c=Px[:,j]; Px[:,j]=np.clip(c-A*c*(1-c),0,1)
    for pid,gt in ecmap.items():
        if pid not in idx: continue
        i=idx[pid]; gi={c2i[e] for e in gt if e in c2i}
        if not gi: continue
        for tag,Px in [('base',B),('multi',Pm),('top1',Pt)]:
            sc=Px[i]; t1=1 if int(np.argmax(sc)) in gi else 0
            agg[tag][0].append(t1); agg[tag][1].append(fmax(gi,sc))
    ng+=1
print(f'genomes={ng}, proteins={len(agg["base"][0])}')
print('%-6s %8s %8s'%('','top1','Fmax'))
for tag in ['base','multi','top1']:
    print('%-6s %8.4f %8.4f'%(tag,np.mean(agg[tag][0]),np.mean(agg[tag][1])))
b=np.mean(agg['base'][1])
print('\nFmax delta vs baseline(extract_pred): multi=%+.4f  top1=%+.4f'%(np.mean(agg['multi'][1])-b,np.mean(agg['top1'][1])-b))
print('top1-acc delta vs baseline: multi=%+.4f  top1only=%+.4f'%(np.mean(agg['multi'][0])-np.mean(agg['base'][0]),np.mean(agg['top1'][0])-np.mean(agg['base'][0])))
