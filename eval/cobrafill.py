#!/usr/bin/env python3
"""COBRApy gap-fill on the thresholded draft --- comparison against a released tool.

The earlier attempt at this (Supplementary S4) reported zero growth in 321 runs,
which could not be right. The cause was environmental: the medium was applied to
the universal model, but the thresholded draft contains no exchange reactions --
they carry no EC annotation and so never clear tau -- and build_submodel does not
inherit those bounds. The submodel therefore had no uptake, and no set of added
internal reactions can make a model grow without nutrients.

This version sets the medium on the submodel itself, using the same procedure as
eval/gen_table1.py so the numbers line up with Table 1, then gap-fills.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src"))
from meteor_v8.utils import data_path, data_dir
import sys, os, json, argparse, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True); ap.add_argument("--gram", required=True)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--lb", type=float, default=0.05)
ap.add_argument("--outdir", default="/ibex/scratch/projects/c2014/kexin/funcarve/"
                                    "meteor_v8/results/cobra_gapfill")
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
OUT = os.path.join(a.outdir, f"cgf_{a.gca}.json")
if os.path.exists(OUT): print("done"); sys.exit(0)

from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                         apply_media, build_submodel, load_refmapping, load_ec,
                         extract_pred, build_rxn_ec_mask)
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
from cobra.flux_analysis.gapfilling import gapfill

universal, allrxns, allmet = load_universal()
for x in ([universal] + list(universal.reactions) + list(universal.metabolites)
          + list(universal.genes)):
    if not hasattr(x, "_annotation"): x._annotation = {}
biomass_id = "biomass_GmPos" if a.gram == "positive" else "biomass_GmNeg"
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lt, ut = load_tight_bounds(data_path('tight_bounds_v6_%s.pkl' % a.gram[:3]))
if lt is not None: lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
lb, ub, _, _ = apply_media(["default"], allrxns, lb, ub)
ix = {rid: i for i, rid in enumerate(allrxns)}
BND, UPTAKE_LB = 100.0, -10.0

seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path('all_ancestors.txt'))
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
p = resolve_baseline_pkl("dpz", "vanilla", a.gca, BASELINE_SUFFIX["dpz"])
if not p: print("no pred"); sys.exit(3)
pred = extract_pred(p, anc)
hit = (pred.values >= a.tau).any(axis=0)
keep_flags = np.array([bool((mask[j] == 1).any() and hit[mask[j] == 1].any())
                       for j in range(len(allrxns))])
media_ids = {allrxns[j] for j in range(len(allrxns))
             if allrxns[j].startswith("EX_") and lb[j] < 0}
keep_flags_m = keep_flags.copy()
for j, rid in enumerate(allrxns):
    if rid in media_ids: keep_flags_m[j] = True
keep = [r for r, k in zip(universal.reactions, keep_flags_m) if k]
n_draft_only = int(keep_flags.sum())

def set_medium(model):
    """Internal bounds capped as in eval/gen_table1.py; the exchanges carried in
    with the draft take the defined medium's bounds directly, since a
    thresholded draft contains no exchange reactions of its own to infer from."""
    for r in model.reactions:
        if r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower(): continue
        i = ix.get(r.id)
        r.lower_bound = max(-BND, float(lb[i])) if i is not None else -BND
        r.upper_bound = min(BND, float(ub[i])) if i is not None else BND
    exs = [r for r in model.reactions if r.id.startswith("EX_") and r.id != "EX_biomass"]
    n_up = 0
    for r in exs:
        i = ix.get(r.id)
        if i is not None and lb[i] < 0:
            r.lower_bound = max(UPTAKE_LB, float(lb[i])); n_up += 1
        else:
            r.lower_bound = 0.0
        r.upper_bound = 1000.0
    return n_up

model = build_submodel(universal, keep, a.gca, biomass_id=biomass_id)
model.objective = biomass_id
n_uptake = set_medium(model)
pre = model.slim_optimize()
pre = 0.0 if (pre is None or np.isnan(pre)) else pre
n_pre = len(model.reactions)
print(f"{a.gca}: draft {n_draft_only} evidence rxns -> submodel {n_pre}, "
      f"{n_uptake} uptakes, growth={pre:.4f}", flush=True)

t0 = time.time(); n_added = 0; err = None
try:
    res = gapfill(model, universal, lower_bound=a.lb,
                  demand_reactions=False, exchange_reactions=False)
    if res and len(res) > 0:
        have = {r.id for r in model.reactions}
        new = [r for r in res[0] if r.id not in have]
        if new: model.add_reactions(new)
        n_added = len(res[0])
        # the added reactions arrive with the universal's bounds; re-apply ours
        set_medium(model)
except Exception as e:
    err = f"{type(e).__name__}: {e}"
    print(f"  gapfill error: {err}", flush=True)
gf_t = time.time() - t0

post = model.slim_optimize()
post = 0.0 if (post is None or np.isnan(post)) else post
rec = dict(gca=a.gca, gram=a.gram, n_draft=n_draft_only, n_rxn_pre=n_pre, n_rxn_post=len(model.reactions),
           n_added=int(n_added), n_uptake=n_uptake,
           growth_pre=round(float(pre), 6), growth_post=round(float(post), 6),
           grows=bool(post > 1e-6), gapfill_s=round(gf_t, 1), err=err)
json.dump(rec, open(OUT, "w"))
print(f"  post: {len(model.reactions)} rxns (+{n_added}), growth={post:.4f}, "
      f"grows={rec['grows']}, {gf_t:.0f}s", flush=True)
