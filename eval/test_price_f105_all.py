import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys,os,glob,pickle,numpy as np,pandas as pd
from meteor_v8.utils import extract_pred, load_ec
anc=load_ec(data_path('all_ancestors.txt'))
F='/ibex/scratch/projects/c2014/kexin/funcarve'; ECONTO='/ibex/user/niuk0a/funcarve/econto'
MOP=f'{F}/meteor_v8_evw_p2mu3_run/meteor_out_price'; A=0.4
def bpp(b,gca):
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
def f105(gt,sc):
    pos={ec for ec,s in sc.items() if s>=0.5}; tp=len(gt&pos); r=tp/len(gt); p=tp/len(pos) if pos else (1.0 if not gt else 0.0)
    return 2*p*r/(p+r) if p+r>0 else 0.0
price=pd.read_csv(f'{ECONTO}/data/test/price.csv',sep='\t')
gt_ecs={r['Entry']:{e.strip() for e in str(r['EC number']).split(';') if e.strip().count('.')==3 and '-' not in e.strip()} for _,r in price.iterrows()}
gt_ecs={k:v for k,v in gt_ecs.items() if v}
meta=pd.read_csv(f'{ECONTO}/data/processed/geno/price_proteomes/Price_genome_metainfo.csv')
mm={r['input_id']:(r['protein_id'],r.get('nucleotide_id'),r['assembly']) for _,r in meta.iterrows()}
def build(pred,asm,b):
    pr=f'{MOP}/{b}/meteor_preds_{asm}.pkl'
    if not os.path.exists(pr): return None,None
    d=pickle.load(open(pr,'rb')); active=set(d['active_ecs']); muted=set(d['muted_ecs'])
    c2i={c:i for i,c in enumerate(pred.columns)}; Pm=pred.values.astype(float).copy(); Pt=pred.values.astype(float).copy()
    for ec in active:
        if ec in c2i: j=c2i[ec];cm=Pm[:,j];Pm[:,j]=np.clip(cm+A*cm*(1-cm),0,1);ct=Pt[:,j];jj=int(np.argmax(ct));Pt[jj,j]=min(1.0,ct[jj]+A*ct[jj]*(1-ct[jj]))
    for ec in muted:
        if ec in c2i:
            j=c2i[ec]
            for Px in (Pm,Pt): c=Px[:,j];Px[:,j]=np.clip(c-A*c*(1-c),0,1)
    return pd.DataFrame(Pm,index=pred.index,columns=pred.columns),pd.DataFrame(Pt,index=pred.index,columns=pred.columns)
print('%-8s %5s | base  multi  top1  | Δmulti  Δtop1'%('baseline','n'))
for b in ['clean','dpz','enzbert','graphec','mapred','topec']:
    bc={};mc={};tc={};n=0;Sb=Sm=St=0.0
    for ent,gt in gt_ecs.items():
        if ent not in mm: continue
        pid,nid,asm=mm[ent]
        if asm not in bc:
            bp=bpp(b,asm)
            try: pred=extract_pred(bp,anc) if bp else None
            except: pred=None
            bc[asm]=pred
            if pred is not None: mc[asm],tc[asm]=build(pred,asm,b)
            else: mc[asm]=tc[asm]=None
        sb=lookup(bc[asm],pid,nid);sm=lookup(mc[asm],pid,nid);st=lookup(tc[asm],pid,nid)
        if not (sb and sm and st): continue
        n+=1; Sb+=f105(gt,sb); Sm+=f105(gt,sm); St+=f105(gt,st)
    if n: print('%-8s %5d | %.3f %.3f %.3f | %+.4f %+.4f'%(b,n,Sb/n,Sm/n,St/n,(Sm-Sb)/n,(St-Sb)/n))
