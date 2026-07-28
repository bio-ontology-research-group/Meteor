"""Table 1 (main text, tab:structural) — per-genome structural record.

Computes, for one assembly, the six columns of Table 1:
  baseline_{clean,dpz,enzbert}  = network of every reaction carrying an EC with
                                  at least one protein score >= TAU (no top-k
                                  truncation: this is the hard-threshold draft)
  meteor_{clean,dpz,enzbert}    = METEOR's selected set (y > 0.5), evw p2mu3 run

Submodel construction, bound handling, medium inference and the MEMOTE calls are
copied verbatim from diag/memote_evw.py so the new columns are comparable with
the ones already published.
"""
import sys, os, json, pickle, argparse, numpy as np
V6 = "/ibex/user/niuk0a/funcarve/cobra/v6"
sys.path.insert(0, V6); os.chdir(V6)
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v7/src")
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/eval")

ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True)
ap.add_argument("--gram", required=True, choices=["negative", "positive"])
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--outdir", default="/ibex/scratch/projects/c2014/kexin/funcarve/"
                                    "meteor_v8/results/table1")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
OUT = os.path.join(a.outdir, "table1_%s.json" % a.gca)
if os.path.exists(OUT):
    print("done", flush=True); sys.exit(0)

from src.v6utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                         apply_media, build_submodel, load_refmapping, load_ec,
                         extract_pred, build_rxn_ec_mask)
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
import memote.support.consistency as cons

MET = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out"
BASELINES = ["clean", "dpz", "enzbert"]

universal, allrxns, allmet = load_universal()
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
biomass_id = "biomass_GmPos" if a.gram == "positive" else "biomass_GmNeg"
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lt, ut = load_tight_bounds(V6 + "/data/tight_bounds_v6_%s.pkl" % a.gram[:3])
if lt is not None: lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
lb, ub, _, _ = apply_media(["default"], allrxns, lb, ub)
ix = {rid: i for i, rid in enumerate(allrxns)}
BND, UPTAKE_LB = 100.0, -10.0

seedr2ec, _ = load_refmapping(V6 + "/data"); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(V6 + "/data/all_ancestors.txt")
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)


def profile(keep_flags, label):
    """keep_flags: bool array over allrxns. Returns the structural record."""
    keep = [r for r, k in zip(universal.reactions, keep_flags) if k]
    if not keep:
        return {"err": "empty"}
    model = build_submodel(universal, keep, a.gca, biomass_id=biomass_id)
    model.objective = biomass_id
    for r in model.reactions:
        if r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower():
            continue
        i = ix.get(r.id)
        r.lower_bound = max(-BND, float(lb[i])) if i is not None else -BND
        r.upper_bound = min(BND, float(ub[i])) if i is not None else BND
    exs = [r for r in model.reactions if r.id.startswith("EX_") and r.id != "EX_biomass"]
    for r in exs: r.lower_bound, r.upper_bound = -1000.0, 1000.0
    so = model.optimize()
    uptake = {r.id for r in exs if so.fluxes.get(r.id, 0.0) < -1e-6}
    for r in exs:
        r.lower_bound = UPTAKE_LB if r.id in uptake else 0.0
        r.upper_bound = 1000.0
    bm = model.slim_optimize()
    de = len(cons.find_deadends(model))
    mi = len(cons.find_mass_unbalanced_reactions(model.reactions))
    nr = len(model.reactions)
    rec = dict(n_selected=int(keep_flags.sum()), n_rxn=nr,
               fba_growth=round(float(bm), 4), deadends=de, mass_imbal=mi,
               mi_frac=round(mi / max(1, nr), 4), n_uptake=len(uptake))
    print("[%s] %s sel=%d rxn=%d growth=%.3f deadend=%d mi=%d"
          % (label, a.gca, rec["n_selected"], nr, bm, de, mi), flush=True)
    return rec


res = {"gca": a.gca, "gram": a.gram, "tau": a.tau}

for b in BASELINES:
    # --- baseline network: reaction kept if any EC of it scores >= tau ---
    p = resolve_baseline_pkl(b, "vanilla", a.gca, BASELINE_SUFFIX[b])
    if not p:
        res["baseline_" + b] = {"err": "no_pred"}
    else:
        pred = extract_pred(p, anc)
        hit = (pred.values >= a.tau).any(axis=0)          # EC columns above tau
        flags = np.array([bool((mask[j] == 1).any() and hit[mask[j] == 1].any())
                          for j in range(len(allrxns))])
        res["baseline_" + b] = profile(flags, "base_" + b)

    # --- METEOR selected set ---
    f = "%s/%s_vanilla/meteor_sol_%s.pkl" % (MET, b, a.gca)
    if not os.path.exists(f):
        res["meteor_" + b] = {"err": "no_sol"}
    else:
        yv = np.array(pickle.load(open(f, "rb")).get("y_vals", []))
        res["meteor_" + b] = profile(yv > 0.5, "meteor_" + b)

json.dump(res, open(OUT, "w"))
print("written", OUT, flush=True)
