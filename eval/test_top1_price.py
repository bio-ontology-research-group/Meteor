import sys,os,glob,pickle,numpy as np,pandas as pd
sys.path.insert(0,'/ibex/user/niuk0a/funcarve/cobra/v6'); os.chdir('/ibex/user/niuk0a/funcarve/cobra/v6')
from src.v6utils import extract_pred, load_ec
sys.path.insert(0,'/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/eval')
anc=load_ec('/ibex/user/niuk0a/funcarve/cobra/v6/data/all_ancestors.txt')
F='/ibex/scratch/projects/c2014/kexin/funcarve'; ECONTO='/ibex/user/niuk0a/funcarve/econto'
MOP=f'{F}/meteor_v8_evw_p2mu3_run/meteor_out_price'; A=0.4
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
        if x in df.index: return {str(k).replace('EC:','').strip():v for k,v in df.loc[x].to_dict().items()}
        b=str(x).rsplit('.',1)[0]; mt=[i for i in df.index if i==b or str(i).startswith(b+'.')]
        if mt: return {str(k).replace('EC:','').strip():v for k,v in df.loc[mt[0]].to_dict().items()}
    return None
def top1(gt,sc):
    if not sc: return 0
    return 1 if max(sc,key=sc.get) in gt else 0
def fmax(gt,sc):
    if not gt or not sc: return 0.0
    items=sorted(sc.items(),key=lambda x:-x[1]); best=0.0; tp=0
    for i,(ec,s) in enumerate(items,1):
        if ec in gt: tp+=1
        P=tp/i; R=tp/len(gt); f=2*P*R/(P+R) if P+R>0 else 0.0
        if f>best: best=f
    return best
price=pd.read_csv(f'{ECONTO}/data/test/price.csv',sep='\t')
gt_ecs={r['Entry']:{e.strip() for e in str(r['EC number']).split(';') if e.strip().count('.')==3 and '-' not in e.strip()} for _,r in price.iterrows()}
gt_ecs={k:v for k,v in gt_ecs.items() if v}
meta=pd.read_csv(f'{ECONTO}/data/processed/geno/price_proteomes/Price_genome_metainfo.csv')
meta_map={r['input_id']:(r['protein_id'],r.get('nucleotide_id'),r['assembly']) for _,r in meta.iterrows()}
def build(pred,asm,b):
    pr=f'{MOP}/{b}/meteor_preds_{asm}.pkl'
    if not os.path.exists(pr): return None,None
    d=pickle.load(open(pr,'rb')); active=set(d['active_ecs']); muted=set(d['muted_ecs'])
    c2i={c:i for i,c in enumerate(pred.columns)}; Pm=pred.values.astype(float).copy(); Pt=pred.values.astype(float).copy()
    for ec in active:
        if ec in c2i:
            j=c2i[ec]; cm=Pm[:,j]; Pm[:,j]=np.clip(cm+A*cm*(1-cm),0,1); ct=Pt[:,j]; jj=int(np.argmax(ct)); Pt[jj,j]=min(1.0,ct[jj]+A*ct[jj]*(1-ct[jj]))
    for ec in muted:
        if ec in c2i:
            j=c2i[ec]
            for Px in (Pm,Pt): c=Px[:,j]; Px[:,j]=np.clip(c-A*c*(1-c),0,1)
    return pd.DataFrame(Pm,index=pred.index,columns=pred.columns),pd.DataFrame(Pt,index=pred.index,columns=pred.columns)
print('%-8s %5s | %-18s | %-18s | %-18s'%('baseline','n','base(extract_pred)','multi(current)','top1-only'))
for b in ['clean','dpz','enzbert','graphec','mapred','topec']:
    bc={}; mc={}; tc={}; n=0; fb=fm=ft=0.0; t1b=t1m=t1t=0
    for ent,gt in gt_ecs.items():
        if ent not in meta_map: continue
        pid,nid,asm=meta_map[ent]
        if asm not in bc:
            bp=base_pred_path(b,asm)
            try: pred=extract_pred(bp,anc) if bp else None
            except: pred=None
            bc[asm]=pred
            if pred is not None: mc[asm],tc[asm]=build(pred,asm,b)
            else: mc[asm]=tc[asm]=None
        sb=lookup(bc[asm],pid,nid); sm=lookup(mc[asm],pid,nid); st=lookup(tc[asm],pid,nid)
        if not (sb and sm and st): continue
        n+=1; t1b+=top1(gt,sb); t1m+=top1(gt,sm); t1t+=top1(gt,st)
        fb+=fmax(gt,sb); fm+=fmax(gt,sm); ft+=fmax(gt,st)
    if n:
        print('%-8s %5d | top1 %3d Fmax %.3f | top1 %3d Fmax %.3f | top1 %3d Fmax %.3f'%(b,n,t1b,fb/n,t1m,fm/n,t1t,ft/n))
