"""Dead ends of a GROWING threshold baseline (tau=0.5 draft + grow_support repair), DeepProZyme-vanilla.

Draft and repair are built exactly as the Section 3.2 comparator (meteor_v8/eval/weakreal.py
baseline_ecs, = baseline_thresh_gapfill.py): draft = reactions with any EC scoring >= tau;
core = draft minus structurally excluded reactions; repaired = core | grow_support(avail=~excluded,
gamma_min=0.1, core=core). Structural metrics use profile() copied verbatim from gen_table1_v2.py
(MEMOTE find_deadends / find_mass_unbalanced_reactions on the COBRA submodel). Controls in the same
run: the raw draft (Table 1 baseline_dpz) and the deposited METEOR selected set (Table 1 meteor_dpz).
Writes psb_revision/results/thresh_gapfill_deadends_met/tgd_{gca}.json only.
Copy of thresh_gapfill_deadends.py that additionally records metabolite counts of the exact
submodel passed to find_deadends: n_metabolites = len(model.metabolites); n_metabolites_internal =
metabolites whose compartment does not start with "e" (extracellular); per-compartment counts.
"""
import os as _os, sys as _sys
_sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval")
import _env  # noqa: E402
import sys, os, json, pickle, time, argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--gca", required=True)
ap.add_argument("--gram", required=True, choices=["negative", "positive"])
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--outdir", default=f"{_env.RESULTS}/thresh_gapfill_deadends_met")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
OUT = os.path.join(a.outdir, "tgd_%s.json" % a.gca)
if os.path.exists(OUT):
    print("done", flush=True); sys.exit(0)
t_start = time.time()

from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                             apply_media, build_submodel, load_refmapping, load_ec,
                             extract_pred, build_rxn_ec_mask, data_path, data_dir,
                             find_excluded_reactions)
from meteor_v8.repair import grow_support, maxbio_active
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
import memote.support.consistency as cons

MET = _env.RUNS
universal, allrxns, allmet = load_universal()
for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
    if not hasattr(x, "_annotation"): x._annotation = {}
biomass_id = "biomass_GmPos" if a.gram == "positive" else "biomass_GmNeg"
S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lt, ut = load_tight_bounds(data_path('tight_bounds_v6_%s.pkl' % a.gram[:3]))
if lt is not None: lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
oi = allrxns.index(biomass_id)
exc = find_excluded_reactions(S, lb, ub, allrxns, biomass_id)   # as weakreal.setup: before media
lb, ub, _, _ = apply_media(["default"], allrxns, lb, ub)
ix = {rid: i for i, rid in enumerate(allrxns)}
BND, UPTAKE_LB = 100.0, -10.0
NR = len(allrxns)

seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path('all_ancestors.txt'))
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)


def profile(keep_flags, label):
    """Verbatim structural block of gen_table1_v2.profile (v1 fields)."""
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
    de_list = cons.find_deadends(model)
    de = len(de_list)
    comp = {}
    for m in model.metabolites: comp[m.compartment] = comp.get(m.compartment, 0) + 1
    n_int = sum(1 for m in model.metabolites if not str(m.compartment).startswith("e"))
    de_int = sum(1 for m in de_list if not str(m.compartment).startswith("e"))
    mi = len(cons.find_mass_unbalanced_reactions(model.reactions))
    nr = len(model.reactions)
    rec = dict(n_selected=int(keep_flags.sum()), n_rxn=nr, fba_growth=round(float(bm), 4),
               deadends=de, mass_imbal=mi, mi_frac=round(mi / max(1, nr), 4),
               n_metabolites=len(model.metabolites), n_metabolites_internal=n_int,
               deadends_internal=de_int, compartments=comp)
    print("[%s] %s sel=%d rxn=%d growth=%.3f deadend=%d mi=%d" % (label, a.gca, rec["n_selected"], nr, bm, de, mi), flush=True)
    return rec


p = resolve_baseline_pkl("dpz", "vanilla", a.gca, BASELINE_SUFFIX["dpz"])
pred = extract_pred(p, anc)
hit = (pred.values >= a.tau).any(axis=0)
draft = np.array([bool((mask[j] == 1).any() and hit[mask[j] == 1].any()) for j in range(NR)])
avail = np.ones(NR, bool); avail[list(exc)] = False
core = draft & avail
draft_growth = maxbio_active(S, lb, ub, oi, core)
t0 = time.time()
sup = grow_support(S, lb, ub, oi, avail, 0.1, core=core)
t_repair = time.time() - t0
repaired = core | (sup if sup is not None else np.zeros(NR, bool))
added = repaired & ~core
rep_growth = maxbio_active(S, lb, ub, oi, repaired)

res = dict(gca=a.gca, gram=a.gram, tau=a.tau, n_draft=int(draft.sum()), n_core=int(core.sum()),
           draft_growth_milp_medium=round(float(draft_growth), 4), draft_grows=bool(draft_growth >= 0.05),
           repair_found=sup is not None, n_added_by_repair=int(added.sum()),
           repaired_growth_milp_medium=round(float(rep_growth), 4), repaired_grows=bool(rep_growth >= 0.05),
           repair_sec=round(t_repair, 2))
res["repaired"] = profile(repaired, "repaired")
res["raw_threshold"] = profile(draft, "raw_threshold")          # control: Table 1 baseline_dpz
f = "%s/dpz_vanilla/meteor_sol_%s.pkl" % (MET, a.gca)
yv = np.array(pickle.load(open(f, "rb")).get("y_vals", [])) if os.path.exists(f) else None
res["meteor"] = {"err": "no_sol"} if yv is None else profile(yv > 0.5, "meteor")   # control: Table 1 meteor_dpz
res["runtime_sec"] = round(time.time() - t_start, 1)
json.dump(res, open(OUT, "w"))
print("written", OUT, flush=True)
