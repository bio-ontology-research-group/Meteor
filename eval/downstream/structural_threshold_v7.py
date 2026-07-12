#!/usr/bin/env python3
"""Left-3-columns of tab:structural: raw baseline threshold network (tau=0.5,
NO MILP) structural metrics on the 109-GCF panel. Reuses build_submodel from
v6utils and calls memote's internal find_deadends/find_mass_unbalanced_reactions
directly (skips the full memote test suite -- much faster on a ~31k-reaction
union network).
"""
import sys, os, json, argparse, time
import numpy as np
import pandas as pd

V6 = "/ibex/user/niuk0a/funcarve/cobra/v6"
PA = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026"
DOWNSTREAM = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7/eval/downstream"
sys.path.insert(0, DOWNSTREAM)
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
sys.path.insert(0, V6); os.chdir(V6)
from src.v6utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                          build_rxn_ec_mask, extract_pred, load_refmapping, load_ec,
                          build_submodel)

import memote.support.consistency as cons

ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True)
ap.add_argument("--gram", choices=["negative", "positive"], required=True)
ap.add_argument("--baseline", choices=["clean", "dpz", "enzbert"], required=True)
ap.add_argument("--outdir", required=True)
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
t0 = time.time()
biomass_id = "biomass_GmPos" if a.gram == "positive" else "biomass_GmNeg"

seedr2ec, _ = load_refmapping(f"{V6}/data"); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, allmet = load_universal()
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"):
        x._annotation = {}
anc = load_ec(f"{V6}/data/all_ancestors.txt")

_suf = BASELINE_SUFFIX[a.baseline]
_pred_path = resolve_baseline_pkl(a.baseline, "vanilla", a.gca, _suf)
if not _pred_path:
    print(f"[thresh] ABORT {a.gca}: no pred file", flush=True)
    sys.exit(3)
pred = extract_pred(_pred_path, anc)
_ep = resolve_baseline_pkl("enzbert", "vanilla", a.gca, "enzbert")
_ref_n = pd.read_pickle(_ep).shape[0] if _ep else None
if _ref_n and pred.shape[0] < 0.9 * _ref_n:
    print(f"[thresh] ABORT {a.gca} {a.baseline}: pred rows {pred.shape[0]} < 0.9*{_ref_n} (incomplete)", flush=True)
    sys.exit(3)
print(f"[thresh] pred src={_pred_path} shape={pred.shape}", flush=True)

mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)  # (n_rxns, n_ecs), columns aligned to `anc`
# pred.columns == anc (extract_pred reindexes onto ancestorsec) -> positional alignment holds.
ec_above = (pred.values >= 0.5).any(axis=0)  # per-EC (aligned to anc): any protein >= 0.5
keep_flags = (mask[:, ec_above] > 0).any(axis=1) if ec_above.any() else np.zeros(len(allrxns), dtype=bool)

keep = [r for r, k in zip(universal.reactions, keep_flags) if k]
print(f"[thresh] {a.gca} {a.baseline}: {len(keep)}/{len(allrxns)} reactions >= tau=0.5", flush=True)

model = build_submodel(universal, keep, a.gca, biomass_id=biomass_id)

t1 = time.time()
deadends = cons.find_deadends(model)
massimbal = cons.find_mass_unbalanced_reactions(model.reactions)
t2 = time.time()

row = dict(gca=a.gca, baseline=a.baseline, n_rxn=len(model.reactions),
           n_deadends=len(deadends), n_massimbal=len(massimbal),
           build_s=round(t1 - t0, 1), check_s=round(t2 - t1, 1))
json.dump(row, open(f"{a.outdir}/{a.gca}_{a.baseline}_thresh.json", "w"))
print("SUMMARY\t" + json.dumps(row), flush=True)
