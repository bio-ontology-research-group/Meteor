import sys,os,pickle,glob,numpy as np,cobra
sys.path.insert(0,'/ibex/user/niuk0a/funcarve/cobra/v6'); os.chdir('/ibex/user/niuk0a/funcarve/cobra/v6')
from src.v6utils import load_universal, load_refmapping, load_ec, build_rxn_ec_mask, extract_pred
bigg2mnxr={}; seed2mnxr={}
for ln in open('/ibex/scratch/projects/c2014/kexin/funcarve/meteor_diag/reac_xref.tsv'):
    if ln.startswith('#'): continue
    p=ln.rstrip().split('\t')
    if len(p)<2 or not p[1].startswith('MNXR'): continue
    if p[0].startswith('bigg.reaction:'): bigg2mnxr.setdefault(p[0].split(':',1)[1],p[1])
    elif p[0].startswith('seed.reaction:'): seed2mnxr.setdefault(p[0].split(':',1)[1],p[1])
GEMDIR='/ibex/scratch/projects/c2014/kexin/funcarve/meteor_diag/curated_gems'
MO='/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla'
GEM2G={'iML1515':('GCF_058436375.1','E.coli(most-studied)'),'STM_v1_0':('GCF_000006945.2','Salmonella'),'iYL1228':('GCF_058435815.1','K.pneumoniae'),'iJN1463':('GCF_045571375.1','P.putida'),'iYS854':('GCF_045348045.1','S.aureus'),'iYO844':('GCF_058182495.1','B.subtilis(less-studied)')}
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
    return A
def act_mnxr(mb): return {seedidx_mnxr[j] for j in np.where(mb)[0] if seedidx_mnxr[j]}
print('%-22s %6s | %-16s | %-16s | Δrecall'%('organism','GEMmnxr','baseline@0.5 P/R','METEOR P/R'))
for gem,(gcf,name) in GEM2G.items():
    A=gem_mnxr(gem)
    try:
        sol=pickle.load(open(f'{MO}/meteor_sol_{gcf}.pkl','rb')); met=np.array(sol['y_vals'])>0.5
        pred=extract_pred(resolve_baseline_pkl('dpz','vanilla',gcf,'DPZ'),anc); Pm=pred.values; hot=set(pred.columns[np.where(Pm.max(axis=0)>=0.5)[0]])
    except Exception as e: print(name,'skip',e); continue
    thr=np.zeros(len(allrxns),bool)
    for j in range(len(allrxns)):
        ei=np.where(mask[j]==1)[0]
        if len(ei) and any(pred.columns[e] in hot for e in ei): thr[j]=True
    bm=act_mnxr(thr); mm=act_mnxr(met)
    Pb=len(bm&A)/max(1,len(bm)); Rb=len(bm&A)/max(1,len(A)); Pm2=len(mm&A)/max(1,len(mm)); Rm=len(mm&A)/max(1,len(A))
    print('%-22s %6d | P%.3f R%.3f    | P%.3f R%.3f    | %+.3f'%(name,len(A),Pb,Rb,Pm2,Rm,Rm-Rb))
