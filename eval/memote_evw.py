#!/usr/bin/env python3
"""TABLE-GENERATING SCRIPT --- produces the MEMOTE table in supplement S8.

Builds the COBRA submodel for one genome under each of three cost
configurations (evw p2mu3, uniform mu3, evw p2mu8), infers the uptake medium
post hoc, and records reaction counts, dead-ends, mass-imbalanced reactions
and FBA growth via MEMOTE's consistency tests.

Run once per genome; the per-genome JSONs are then aggregated for the table.
Paths below point at our cluster layout --- see README.md.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys, os, json, pickle, argparse, numpy as np
sys.path.insert(0,"/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7/src")
ap=argparse.ArgumentParser(); ap.add_argument("--gca",required=True); ap.add_argument("--gram",required=True)
a=ap.parse_args()
OUT="/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/downstream_results/memote/memote_evw_%s.json"%a.gca
if os.path.exists(OUT): print("done"); sys.exit(0)
from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                         apply_media, build_submodel)
import memote.support.consistency as cons
CONFIGS={"evw_p2mu3":"/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out/dpz_vanilla",
         "uniform_mu3":"/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_mu3e0_run/meteor_out/dpz_vanilla",
         "evw_p2mu8":"/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu8_run/meteor_out/dpz_vanilla",
         }
universal, allrxns, allmet = load_universal()
for x in list(universal.reactions)+list(universal.metabolites)+list(universal.genes):
    if not hasattr(x,"_annotation"): x._annotation={}
biomass_id="biomass_GmPos" if a.gram=="positive" else "biomass_GmNeg"
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lt,ut = load_tight_bounds(data_path('tight_bounds_v6_%s.pkl' % a.gram[:3]))
if lt is not None: lb=np.maximum(lb,lt); ub=np.minimum(ub,ut)
lb,ub,_,_=apply_media(["default"],allrxns,lb,ub)
ix={rid:i for i,rid in enumerate(allrxns)}
BND=100.0; UPTAKE_LB=-10.0
res={"gca":a.gca,"gram":a.gram}
for cfg,d in CONFIGS.items():
    f="%s/meteor_sol_%s.pkl"%(d,a.gca)
    if not os.path.exists(f): res[cfg]={"err":"no_sol"}; continue
    yv=np.array(pickle.load(open(f,"rb")).get("y_vals",[]))
    keep=[r for r,k in zip(universal.reactions, yv>0.5) if k]
    model=build_submodel(universal, keep, a.gca, biomass_id=biomass_id); model.objective=biomass_id
    for r in model.reactions:
        if r.id.startswith(("EX_","DM_","SK_")) or "biomass" in r.id.lower(): continue
        i=ix.get(r.id); lo=max(-BND,float(lb[i])) if i is not None else -BND; hi=min(BND,float(ub[i])) if i is not None else BND
        r.lower_bound=lo; r.upper_bound=hi
    exs=[r for r in model.reactions if r.id.startswith("EX_") and r.id!="EX_biomass"]
    for r in exs: r.lower_bound=-1000.0; r.upper_bound=1000.0
    so=model.optimize(); uptake={r.id for r in exs if so.fluxes.get(r.id,0.0)<-1e-6}
    for r in exs: r.lower_bound=UPTAKE_LB if r.id in uptake else 0.0; r.upper_bound=1000.0
    bm=model.slim_optimize()
    de=len(cons.find_deadends(model)); mi=len(cons.find_mass_unbalanced_reactions(model.reactions))
    nr=len(model.reactions)
    res[cfg]=dict(n_active=int((yv>0.5).sum()), n_rxn=nr, fba_growth=round(float(bm),4),
                  deadends=de, mass_imbal=mi, mi_frac=round(mi/max(1,nr),4), n_uptake=len(uptake))
    print("[%s] %s active=%d rxn=%d growth=%.3f deadend=%d mifrac=%.3f"%(cfg,a.gca,res[cfg]["n_active"],nr,bm,de,res[cfg]["mi_frac"]),flush=True)
json.dump(res,open(OUT,"w"))
