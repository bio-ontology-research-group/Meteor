import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys,os,json,pickle,argparse,numpy as np

from meteor_v8.utils import (load_universal,extract_fba_matrices,load_tight_bounds,apply_media,
    find_excluded_reactions,compute_costs,build_candidate_mask,build_rxn_ec_mask,extract_pred,
    load_refmapping,load_ec,_detect_solver,posterior_calibrated)
from meteor_v8.milp_hard import biomass_feasible_skeleton
from meteor_v8.milp_v8 import build_milp_v8
from meteor_v8.repair import verify_and_repair

from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
ap=argparse.ArgumentParser(); ap.add_argument('--gca',required=True); ap.add_argument('--gram',required=True); ap.add_argument('--kdel',type=int,default=40)
a=ap.parse_args()
MO='/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla'
OUT='/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/results/recovery'; os.makedirs(OUT,exist_ok=True)
bid='biomass_GmPos' if a.gram=='positive' else 'biomass_GmNeg'
seedr2ec,_=load_refmapping(data_dir()); seedr2ec={k:v for k,v in seedr2ec.items() if v}
ec_with_rxn=set().union(*seedr2ec.values())
universal,allrxns,allmet=load_universal(); anc=load_ec(data_path('all_ancestors.txt'))
pred=extract_pred(resolve_baseline_pkl('dpz','vanilla',a.gca,BASELINE_SUFFIX['dpz']),anc)
orig=pickle.load(open(f'{MO}/meteor_preds_{a.gca}.pkl','rb')); active0=set(orig['active_ecs'])
# predicted ECs (>=0.5) that map to reactions
predicted={c for c in pred.columns if pred[c].max()>=0.5 and c in ec_with_rxn}
net=sorted(predicted & active0); periph=sorted(predicted - active0)
rng=np.random.RandomState(0)
del_net=list(rng.choice(net,min(a.kdel,len(net)),replace=False)) if net else []
del_periph=list(rng.choice(periph,min(a.kdel,len(periph)),replace=False)) if periph else []
delset=set(del_net)|set(del_periph)
print(f'{a.gca}: predicted-with-rxn={len(predicted)} net={len(net)} periph={len(periph)} | deleting net={len(del_net)} periph={len(del_periph)}',flush=True)
# ablate: zero the deleted EC columns
for e in delset: pred[e]=0.0
# ---- standard emit_v8 pipeline on ablated pred ----
mask=build_rxn_ec_mask(allrxns,seedr2ec,anc)
S,lb,ub=extract_fba_matrices(universal,allrxns,reversed_trans=True)
lt,ut=load_tight_bounds(data_path(f'tight_bounds_v6_{a.gram[:3]}.pkl')); lb=np.maximum(lb,lt); ub=np.minimum(ub,ut)
oi=allrxns.index(bid); exc=find_excluded_reactions(S,lb,ub,allrxns,bid)
lb,ub,_,media=apply_media(['default'],allrxns,lb,ub); solver,_=_detect_solver(threads=4,time_limit=600)
feas,skel,_=biomass_feasible_skeleton(S,lb,ub,oi,0.1,solver=solver)
_P=pred.values.astype(np.float32).copy(); _ne=_P.shape[1]
if 5<_ne: _dr=np.argpartition(_P,_ne-5,axis=1)[:,:_ne-5]; np.put_along_axis(_P,_dr,0.0,axis=1)
_l=np.log(np.clip(1.0-_P,1e-9,1.0)); w=np.zeros(len(allrxns))
for j in range(len(allrxns)):
    ei=np.where(mask[j]==1)[0]
    if len(ei): w[j]=1.0-np.exp(float(_l[:,ei].sum()))
w=np.clip(np.nan_to_num(w,nan=0,posinf=1,neginf=0),1e-6,1-1e-6)
c=compute_costs(w,mode='logodds')['c']; c=np.asarray(c,float)+3.0*np.power(np.clip(1.0-w,0,1),2)
cand=build_candidate_mask(w,allrxns,exc,media,essential_skeleton=skel,w_min=0.01)
m,y,vp,vn,_=build_milp_v8(S,lb,ub,c,oi,exc,media,cand,0.1,2.5,1e-4,mu=0.0,eps=0.0); m.solve(solver)
yv=np.array([y[j].value() or 0 for j in range(len(y))]); vv=np.array([(vp[j].value() or 0)-(vn[j].value() or 0) for j in range(len(y))])
yv,na,nr,mbc=verify_and_repair(S,lb,ub,oi,yv,vv,cand)
_,active_new,_=posterior_calibrated(pred,yv,seedr2ec,allrxns,anc,beta=1.0)
active_new=set(active_new)
rn=sum(1 for e in del_net if e in active_new); rp=sum(1 for e in del_periph if e in active_new)
res=dict(gca=a.gca,gram=a.gram,n_del_net=len(del_net),n_del_periph=len(del_periph),
         rec_net=rn,rec_periph=rp,rate_net=round(rn/max(1,len(del_net)),3),rate_periph=round(rp/max(1,len(del_periph)),3))
json.dump(res,open(f'{OUT}/{a.gca}.json','w'))
print('RECOVERY',res,flush=True)
