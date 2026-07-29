import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys,os,glob,pickle,numpy as np,pandas as pd
from meteor_v8.utils import extract_pred, load_ec
anc=load_ec(data_path('all_ancestors.txt'))
F='/ibex/scratch/projects/c2014/kexin/funcarve'; ECONTO='/ibex/user/niuk0a/funcarve/econto'
MOP=f'{F}/meteor_v8_evw_p2mu3_run/meteor_out_price'; A=0.4; TH=0.5
def base_pred_path(b,gca):
    gns=gca.rsplit('.',1)[0]; num=gca.split('_')[1].split('.')[0]
    m={'clean':f'{F}/paperA_2026/baseline_preds/price22_CLEAN_clean/{gca}/{gca}_CLEAN_confidence.pkl','enzbert':f'{F}/tfpc/resultprice_newg/{gca}_enzbert_predictions.pkl','graphec':f'{F}/graphec_price_new/{gca}_GraphEC.pkl','mapred':f'{F}/mapred_price_new/{gca}_MAPred.pkl','topec':f'{F}/topec_price_new/{gca}_TopEC.pkl'}
    if b=='dpz':
        e=f'{F}/dpec2_result/result_price/{gns}_DeepECv2_t5.pkl'
        if os.path.exists(e): return e
        h=glob.glob(f'{F}/dpec2_result/result_price/GC?_{num}_DeepECv2_t5.pkl'); return h[0] if h else ''
    p=m[b]
    for c in (p,p.replace(gca,gns)):
        if os.path.exists(c): return c
    return ''
def lookup(df,pid,nid=None):
    if df is None: return None
    for x in [pid]+([nid] if nid else []):
        if x in df.index: return {str(k).replace('EC:','').strip():float(v) for k,v in df.loc[x].to_dict().items()}
        b=str(x).rsplit('.',1)[0]; mt=[i for i in df.index if i==b or str(i).startswith(b+'.')]
        if mt: return {str(k).replace('EC:','').strip():float(v) for k,v in df.loc[mt[0]].to_dict().items()}
    return None
def metrics(gt,sc):
    if not sc: return None
    items=sorted(sc.items(),key=lambda x:-x[1])
    pos={ec for ec,s in sc.items() if s>=TH}; tp=len(gt&pos)
    rec=tp/len(gt); prec=tp/len(pos) if pos else (1.0 if not gt else 0.0); f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    t1=1 if items[0][0] in gt else 0; t3=1 if any(e in gt for e,_ in items[:3]) else 0; t5=1 if any(e in gt for e,_ in items[:5]) else 0
    return rec,prec,f1,t1,t3,t5
price=pd.read_csv(f'{ECONTO}/data/test/price.csv',sep='\t')
gt_ecs={r['Entry']:{e.strip() for e in str(r['EC number']).split(';') if e.strip().count('.')==3 and '-' not in e.strip()} for _,r in price.iterrows()}
gt_ecs={k:v for k,v in gt_ecs.items() if v}
meta=pd.read_csv(f'{ECONTO}/data/processed/geno/price_proteomes/Price_genome_metainfo.csv')
meta_map={r['input_id']:(r['protein_id'],r.get('nucleotide_id'),r['assembly']) for _,r in meta.iterrows()}
def build_top1(pred,asm,b):
    pr=f'{MOP}/{b}/meteor_preds_{asm}.pkl'
    if not os.path.exists(pr): return None
    d=pickle.load(open(pr,'rb')); active=set(d['active_ecs']); muted=set(d['muted_ecs'])
    c2i={c:i for i,c in enumerate(pred.columns)}; Pt=pred.values.astype(float).copy()
    for ec in active:
        if ec in c2i:
            j=c2i[ec]; ct=Pt[:,j]; jj=int(np.argmax(ct)); Pt[jj,j]=min(1.0,ct[jj]+A*ct[jj]*(1-ct[jj]))
    for ec in muted:
        if ec in c2i:
            j=c2i[ec]; c=Pt[:,j]; Pt[:,j]=np.clip(c-A*c*(1-c),0,1)
    return pd.DataFrame(Pt,index=pred.index,columns=pred.columns)
print('%-8s %5s | %-32s | %-32s'%('baseline','n','base(extract_pred) R/P/F1@.5 t3 t5','top1-only R/P/F1@.5 t3 t5'))
for b in ['clean','dpz','enzbert','graphec','mapred','topec']:
    bc={}; tc={}; n=0; SB=np.zeros(6); ST=np.zeros(6)
    for ent,gt in gt_ecs.items():
        if ent not in meta_map: continue
        pid,nid,asm=meta_map[ent]
        if asm not in bc:
            bp=base_pred_path(b,asm)
            try: pred=extract_pred(bp,anc) if bp else None
            except: pred=None
            bc[asm]=pred; tc[asm]=build_top1(pred,asm,b) if pred is not None else None
        sb=lookup(bc[asm],pid,nid); st=lookup(tc[asm],pid,nid)
        if not (sb and st): continue
        mb=metrics(gt,sb); mt=metrics(gt,st)
        if mb is None or mt is None: continue
        n+=1; SB+=np.array(mb); ST+=np.array(mt)
    if n:
        b_=SB/n; t_=ST/n
        print('%-8s %5d | R%.3f P%.3f F1%.3f t3=%.2f t5=%.2f | R%.3f P%.3f F1%.3f t3=%.2f t5=%.2f'%(b,n,b_[0],b_[1],b_[2],b_[4],b_[5],t_[0],t_[1],t_[2],t_[4],t_[5]))
