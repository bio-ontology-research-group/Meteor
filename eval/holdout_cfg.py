import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys,os,json,pickle,argparse,numpy as np,pandas as pd
from collections import defaultdict
from meteor_v8.utils import extract_pred, load_ec

from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
from scipy.stats import binomtest
ap=argparse.ArgumentParser()
ap.add_argument('--baseline',required=True); ap.add_argument('--variant',required=True)
ap.add_argument('--mo',required=True); ap.add_argument('--tag',required=True)
ap.add_argument('--alpha',type=float,default=0.4)
a=ap.parse_args()
anc=load_ec(data_path('all_ancestors.txt')); A=a.alpha
hc=json.load(open('/ibex/scratch/projects/c2014/kexin/funcarve/meteor_diag/holdout_clean.json'))['holdout_clean']
OUT='/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/results/holdout_cfg'; os.makedirs(OUT,exist_ok=True)

def lookup(df,pid):
    if pid in df.index: row=df.loc[pid]
    else:
        b=str(pid).rsplit('.',1)[0];mt=[i for i in df.index if str(i)==b or str(i).startswith(b+'.')]
        if not mt: return None
        row=df.loc[mt[0]]
    return {str(k).replace('EC:','').strip():float(v) for k,v in row.to_dict().items()}

def prot_metrics(gt,sc):
    """returns dict: top1_hit, top3_hit, Fmax, F1_05, plus (pred_scores,truth) pairs for calib."""
    if not sc: return None
    items=sorted(sc.items(),key=lambda x:-x[1])
    top1=1 if items[0][0] in gt else 0
    top3=1 if any(ec in gt for ec,_ in items[:3]) else 0
    best=0.0;tp=0
    for i,(ec,s) in enumerate(items,1):
        if ec in gt: tp+=1
        P=tp/i;R=tp/len(gt);f=2*P*R/(P+R) if P+R>0 else 0.0
        if f>best: best=f
    pos={ec for ec,s in sc.items() if s>=0.5};tp2=len(gt&pos)
    r=tp2/len(gt);p=tp2/len(pos) if pos else (1.0 if not gt else 0.0);f105=2*p*r/(p+r) if p+r>0 else 0.0
    # calibration pairs: over ECs with score>0 (the "considered" set)
    calib=[(s,1.0 if ec in gt else 0.0) for ec,s in sc.items() if s>0]
    return dict(top1=top1,top3=top3,Fmax=best,F105=f105,calib=calib)

def ece_brier(pairs):
    if not pairs: return None,None
    ps=np.array([p for p,_ in pairs]); ts=np.array([t for _,t in pairs])
    brier=float(np.mean((ps-ts)**2))
    # 15-bin ECE
    bins=np.linspace(0,1,16); ece=0.0;n=len(ps)
    for k in range(15):
        m=(ps>=bins[k])&(ps<bins[k+1]) if k<14 else (ps>=bins[k])&(ps<=bins[k+1])
        if m.sum()==0: continue
        ece+=m.sum()/n*abs(ps[m].mean()-ts[m].mean())
    return float(ece),brier

# group holdout by genome -> memory safe (load/free one genome at a time)
by_gcf=defaultdict(list)
for uid,info in hc.items(): by_gcf[info['gcf']].append((info['refseq_id'],set(info['ec'])))

agg={'base':defaultdict(list),'meteor':defaultdict(list)}
b_calib=[]; m_calib=[]; pairhits=[]  # (base_top1,meteor_top1) for McNemar
n_ok=0
for gcf,prots in by_gcf.items():
    try:
        bp=resolve_baseline_pkl(a.baseline,a.variant,gcf,BASELINE_SUFFIX[a.baseline])
        base=extract_pred(bp,anc)
        pr=pickle.load(open(f'{a.mo}/meteor_preds_{gcf}.pkl','rb'))
        active=set(pr['active_ecs']);muted=set(pr['muted_ecs'])
        c2i={c:i for i,c in enumerate(base.columns)}
        Pm=base.values.astype(float).copy()
        for ec in active:
            if ec in c2i: j=c2i[ec];cm=Pm[:,j];Pm[:,j]=np.clip(cm+A*cm*(1-cm),0,1)
        for ec in muted:
            if ec in c2i: j=c2i[ec];cm=Pm[:,j];Pm[:,j]=np.clip(cm-A*cm*(1-cm),0,1)
        meteor=pd.DataFrame(Pm,index=base.index,columns=base.columns)
    except Exception as e:
        print(f'  skip {gcf}: {e}',flush=True); continue
    for pid,gt in prots:
        mb=prot_metrics(gt,lookup(base,pid)); mm=prot_metrics(gt,lookup(meteor,pid))
        if mb is None or mm is None: continue
        n_ok+=1
        for k in ('top1','top3','Fmax','F105'):
            agg['base'][k].append(mb[k]); agg['meteor'][k].append(mm[k])
        b_calib+=mb['calib']; m_calib+=mm['calib']
        pairhits.append((mb['top1'],mm['top1']))
    del base,meteor,Pm  # free genome

def summ(tag,calib):
    d=dict(n=len(agg[tag]['top1']),
           top1=int(sum(agg[tag]['top1'])), top3=int(sum(agg[tag]['top3'])),
           Fmax=round(float(np.mean(agg[tag]['Fmax'])),4),
           F1_05=round(float(np.mean(agg[tag]['F105'])),4))
    ece,brier=ece_brier(calib)
    d['ECE']=round(ece,4) if ece is not None else None
    d['Brier']=round(brier,4) if brier is not None else None
    return d
b01=sum(1 for bt,mt in pairhits if bt==0 and mt==1)  # corrections
b10=sum(1 for bt,mt in pairhits if bt==1 and mt==0)  # regressions
mcp=float(binomtest(min(b01,b10),b01+b10,0.5).pvalue) if (b01+b10)>0 else 1.0
res={'config':a.tag,'baseline':a.baseline,'variant':a.variant,'alpha':A,'n':n_ok,
     'base':summ('base',b_calib),'meteor':summ('meteor',m_calib),
     'mcnemar':{'corrections_b01':b01,'regressions_b10':b10,'disc':f'{b10}to{b01}','p':round(mcp,4)}}
json.dump(res,open(f'{OUT}/{a.tag}.json','w'),indent=1)
print(f"{a.tag} (a={A}): n={n_ok} top1 {res['base']['top1']}->{res['meteor']['top1']} "
      f"top3 {res['base']['top3']}->{res['meteor']['top3']} Fmax {res['base']['Fmax']}->{res['meteor']['Fmax']} "
      f"McNemar {b10}->{b01} p={round(mcp,4)} | ECE {res['meteor']['ECE']} Brier {res['meteor']['Brier']}",flush=True)
