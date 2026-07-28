#!/usr/bin/env python3
"""Compute flux-capable (unblocked) reaction set per gram under METEOR's media+tight bounds.
Unblocked = network-integrated (can be in a biomass-supporting flux solution); blocked = peripheral/dead-end
(evw can NEVER include -> tests user's hypothesis that clean/enzbert weak signal is on peripheral pathways)."""
import sys,os,pickle,numpy as np
V6='/ibex/user/niuk0a/funcarve/cobra/v6'; sys.path.insert(0,V6); os.chdir(V6)
sys.path.insert(0,'/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/src')
from src.v6utils import load_universal,extract_fba_matrices,load_tight_bounds,apply_media,find_excluded_reactions
import cobra
from cobra.flux_analysis import find_blocked_reactions
u,allrxns,allmet=load_universal(); NR=len(allrxns)
S,lb0,ub0=extract_fba_matrices(u,allrxns,reversed_trans=True)
OUT='/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/results/toolcompare'
gram=sys.argv[sys.argv.index('--gram')+1]
lt,ut=load_tight_bounds(f'{V6}/data/tight_bounds_v6_{gram}.pkl'); lb=np.maximum(lb0,lt); ub=np.minimum(ub0,ut)
oi=allrxns.index('biomass_GmPos' if gram=='pos' else 'biomass_GmNeg')
exc=find_excluded_reactions(S,lb,ub,allrxns,allrxns[oi]); lb,ub,_,_=apply_media(['default'],allrxns,lb,ub)
# apply bounds to cobra model, then find blocked
for j,r in enumerate(u.reactions): r.bounds=(float(lb[j]),float(ub[j]))
print(f'[{gram}] finding blocked reactions over {NR}...',flush=True)
blocked=set(find_blocked_reactions(u, open_exchanges=False))
capable=np.array([allrxns[j] not in blocked for j in range(NR)])
print(f'[{gram}] flux-capable {int(capable.sum())} / {NR}  (blocked {len(blocked)})',flush=True)
pickle.dump({'capable':capable,'allrxns':allrxns}, open(f'{OUT}/fluxcapable_{gram}.pkl','wb'))
print('saved',f'{OUT}/fluxcapable_{gram}.pkl')
