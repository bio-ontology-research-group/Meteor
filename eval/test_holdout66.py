import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys,os,json,pickle,numpy as np,pandas as pd
from meteor_v8.utils import extract_pred, load_ec

from baseline_io import resolve_baseline_pkl
anc=load_ec(data_path('all_ancestors.txt'))
MO='/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla'; A=0.4
hc=json.load(open('/ibex/scratch/projects/c2014/kexin/funcarve/meteor_diag/holdout_clean.json'))['holdout_clean']
def metr(gt,sc):
    if not sc: return None
    items=sorted(sc.items(),key=lambda x:-x[1])
    t1=1 if items[0][0] in gt else 0
    best=0.0;tp=0
    for i,(ec,s) in enumerate(items,1):
        if ec in gt: tp+=1
        P=tp/i;R=tp/len(gt);f=2*P*R/(P+R) if P+R>0 else 0.0
        if f>best: best=f
    pos={ec for ec,s in sc.items() if s>=0.5};tp2=len(gt&pos)
    r=tp2/len(gt);p=tp2/len(pos) if pos else (1.0 if not gt else 0.0);f105=2*p*r/(p+r) if p+r>0 else 0.0
    return t1,best,f105
def lookup(df,pid):
    if pid in df.index: return {str(k).replace('EC:','').strip():float(v) for k,v in df.loc[pid].to_dict().items()}
    b=str(pid).rsplit('.',1)[0];mt=[i for i in df.index if str(i)==b or str(i).startswith(b+'.')]
    if mt: return {str(k).replace('EC:','').strip():float(v) for k,v in df.loc[mt[0]].to_dict().items()}
    return None
cache={}; agg={'base':[[],[],[]],'multi':[[],[],[]],'top1':[[],[],[]]}
for uid,info in hc.items():
    gcf=info['gcf'];pid=info['refseq_id'];gt=set(info['ec'])
    if gcf not in cache:
        cache[gcf]=None
        bp=resolve_baseline_pkl('dpz','vanilla',gcf,'DPZ')
        try:
            pred=extract_pred(bp,anc); pr=pickle.load(open(f'{MO}/meteor_preds_{gcf}.pkl','rb'))
            active=set(pr['active_ecs']);muted=set(pr['muted_ecs']);c2i={c:i for i,c in enumerate(pred.columns)}
            Pm=pred.values.astype(float).copy();Pt=pred.values.astype(float).copy()
            for ec in active:
                if ec in c2i: j=c2i[ec];cm=Pm[:,j];Pm[:,j]=np.clip(cm+A*cm*(1-cm),0,1);ct=Pt[:,j];jj=int(np.argmax(ct));Pt[jj,j]=min(1.0,ct[jj]+A*ct[jj]*(1-ct[jj]))
            for ec in muted:
                if ec in c2i:
                    j=c2i[ec]
                    for Px in (Pm,Pt): c=Px[:,j];Px[:,j]=np.clip(c-A*c*(1-c),0,1)
            cache[gcf]=(pred,pd.DataFrame(Pm,index=pred.index,columns=pred.columns),pd.DataFrame(Pt,index=pred.index,columns=pred.columns))
        except Exception as e: print('skip',gcf,e)
    if cache[gcf] is None: continue
    for tag,df in zip(['base','multi','top1'],cache[gcf]):
        sc=lookup(df,pid); m=metr(gt,sc)
        if m: 
            for k in range(3): agg[tag][k].append(m[k])
print('=== HOLDOUT-66 (net-only base=extract_pred) ===')
print('%-6s %4s %8s %8s %8s'%('','n','top1','Fmax','F1@0.5'))
for tag in ['base','multi','top1']:
    a=agg[tag]; print('%-6s %4d %8.4f %8.4f %8.4f'%(tag,len(a[0]),np.mean(a[0]),np.mean(a[1]),np.mean(a[2])))
b=agg['base']
print('\nvs base:  multi top1=%+.4f Fmax=%+.4f F1@.5=%+.4f | top1only top1=%+.4f Fmax=%+.4f F1@.5=%+.4f'%(
  np.mean(agg['multi'][0])-np.mean(b[0]),np.mean(agg['multi'][1])-np.mean(b[1]),np.mean(agg['multi'][2])-np.mean(b[2]),
  np.mean(agg['top1'][0])-np.mean(b[0]),np.mean(agg['top1'][1])-np.mean(b[1]),np.mean(agg['top1'][2])-np.mean(b[2])))
