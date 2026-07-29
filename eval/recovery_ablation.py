import sys,os,json,pickle,argparse,numpy as np
V6='/ibex/user/niuk0a/funcarve/cobra/v6'; sys.path.insert(0,V6); os.chdir(V6)
sys.path.insert(0,'/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/src')
from src.v6utils import (load_universal,extract_fba_matrices,load_tight_bounds,apply_media,
    find_excluded_reactions,compute_costs,build_candidate_mask,build_rxn_ec_mask,extract_pred,
    load_refmapping,load_ec,_detect_solver)
from meteor_v8.milp_hard import biomass_feasible_skeleton
from meteor_v8.milp_v8 import build_milp_v8
from meteor_v8.repair import verify_and_repair
sys.path.insert(0,'/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/eval')
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
ap=argparse.ArgumentParser(); ap.add_argument('--gca',required=True); ap.add_argument('--gram',required=True); ap.add_argument('--kdel',type=int,default=60); ap.add_argument('--outdir',default=None)
a=ap.parse_args()
MO='/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla'
OUT=a.outdir or '/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/results/recovery_abl'; os.makedirs(OUT,exist_ok=True)
bid='biomass_GmPos' if a.gram=='positive' else 'biomass_GmNeg'
seedr2ec,_=load_refmapping(f'{V6}/data'); seedr2ec={k:v for k,v in seedr2ec.items() if v}
universal,allrxns,allmet=load_universal(); anc=load_ec(f'{V6}/data/all_ancestors.txt')
mask=build_rxn_ec_mask(allrxns,seedr2ec,anc)
pred0=extract_pred(resolve_baseline_pkl('dpz','vanilla',a.gca,BASELINE_SUFFIX['dpz']),anc)
sol=pickle.load(open(f'{MO}/meteor_sol_{a.gca}.pkl','rb')); active0=set(np.where(np.array(sol['y_vals'])>0.5)[0])  # reaction indices = synthetic truth
S,lb,ub=extract_fba_matrices(universal,allrxns,reversed_trans=True)
lt,ut=load_tight_bounds(f'{V6}/data/tight_bounds_v6_{a.gram[:3]}.pkl'); lb=np.maximum(lb,lt); ub=np.minimum(ub,ut)
oi=allrxns.index(bid); exc=find_excluded_reactions(S,lb,ub,allrxns,bid)
lb,ub,_,media=apply_media(['default'],allrxns,lb,ub); solver,_=_detect_solver(threads=4,time_limit=600)
feas,skel,_=biomass_feasible_skeleton(S,lb,ub,oi,0.1,solver=solver)
# D = random active reactions that have deletable EC evidence and are NOT skeleton (non-trivial)
ec_rxns=[j for j in active0 if mask[j].sum()>0 and j not in set(skel)]
rng=np.random.RandomState(0); D=set(rng.choice(ec_rxns,min(a.kdel,len(ec_rxns)),replace=False)) if ec_rxns else set()
# ablate: zero the ECs of D reactions in pred
delec=set()
for j in D: delec |= set(np.where(mask[j]==1)[0])
cols=list(pred0.columns); pred=pred0.copy()
for ei in delec:
    if ei<len(cols): pred.iloc[:,ei]=0.0
# recompute w, true cost on ablated pred
_P=pred.values.astype(np.float32).copy(); _ne=_P.shape[1]
if 5<_ne: _dr=np.argpartition(_P,_ne-5,axis=1)[:,:_ne-5]; np.put_along_axis(_P,_dr,0.0,axis=1)
_l=np.log(np.clip(1.0-_P,1e-9,1.0)); w=np.zeros(len(allrxns))
for j in range(len(allrxns)):
    ei=np.where(mask[j]==1)[0]
    if len(ei): w[j]=1.0-np.exp(float(_l[:,ei].sum()))
w=np.clip(np.nan_to_num(w,nan=0,posinf=1,neginf=0),1e-6,1-1e-6)
c_true=np.asarray(compute_costs(w,mode='logodds')['c'],float)+3.0*np.power(np.clip(1.0-w,0,1),2)
sys.path.insert(0,'/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/eval')
from twostage import two_stage
# candidate set = ablated candidates UNION D UNION original active (feasible repair set incl. originals + alternatives)
cand=build_candidate_mask(w,allrxns,exc,media,essential_skeleton=skel,w_min=0.01)
for j in list(D)+list(active0): cand[j]=True
cidx=np.where(cand)[0]
# thresholded draft on the same ablated scores (stage 1 of the two-stage arm)
_hit = (pred.values >= 0.5).any(axis=0)
tau_mask = np.array([bool((mask[j]==1).any() and _hit[mask[j]==1].any())
                     for j in range(len(allrxns))])
def run(cvec):
    m,y,vp,vn,_=build_milp_v8(S,lb,ub,cvec,oi,exc,media,cand,0.1,2.5,1e-4,mu=0.0,eps=0.0); m.solve(solver)
    yv=np.array([y[j].value() or 0 for j in range(len(y))]); vv=np.array([(vp[j].value() or 0)-(vn[j].value() or 0) for j in range(len(y))])
    yv,_,_,_=verify_and_repair(S,lb,ub,oi,yv,vv,cand); return set(np.where(yv>0.5)[0])
# three cost conditions
c_unif=np.where(cand,1.0,0.0)
cs=c_true.copy(); perm=rng.permutation(cidx); cs[cidx]=c_true[perm]  # shuffled over candidates
res={'gca':a.gca,'gram':a.gram,'nD':len(D),'n_active0':len(active0)}
kept=active0-D
for tag,cv in [('true',c_true),('uniform',c_unif),('shuffled',cs)]:
    na=run(cv); added=na-kept
    rec=len(added & D); prec_den=len(added); dec=len(added-active0)
    res[tag]=dict(recall=round(rec/max(1,len(D)),3), precision=round(rec/max(1,prec_den),3),
                  n_added=prec_den, recovered=rec, decoy=dec, decoy_frac=round(dec/max(1,prec_den),3))
    print(f'{a.gca} {tag:9s}: recall={res[tag]["recall"]} precision={res[tag]["precision"]} added={prec_den} recovered={rec} decoy={dec}',flush=True)
# fourth arm: threshold-then-weighted-gapfill, same candidate set and accounting
na = two_stage(S, lb, ub, oi, cand, w, c_true, tau_mask, 0.1)
if na is None:
    res['twostage'] = {'err': 'infeasible'}
    print(f'{a.gca} twostage : infeasible', flush=True)
else:
    added = na - kept
    rec = len(added & D); prec_den = len(added); dec = len(added - active0)
    res['twostage'] = dict(recall=round(rec/max(1,len(D)),3),
                           precision=round(rec/max(1,prec_den),3),
                           n_added=prec_den, recovered=rec, decoy=dec,
                           decoy_frac=round(dec/max(1,prec_den),3))
    print(f'{a.gca} twostage : recall={res["twostage"]["recall"]} '
          f'precision={res["twostage"]["precision"]} added={prec_den} '
          f'recovered={rec} decoy={dec}', flush=True)
json.dump(res,open(f'{OUT}/{a.gca}.json','w'))
