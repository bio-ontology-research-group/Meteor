"""Paired gap-fill comparison on a shared, evidence-free candidate set.

Both arms start from the same thresholded draft and may draw from the same
reduced reaction set R.  R is built without consulting METEOR's confidences,
so neither arm inherits the other's machinery:

    R = EC-annotated SEED reactions
        u medium exchanges u biomass
        - reactions excluded as structurally unusable

Arm A  cobrapy  gapfill(), a MILP minimising the NUMBER of added reactions.
Arm B  meteor   grow_support(), an LP minimising flux through non-core
                reactions.  This is what the paper's baseline already uses.

Restricting to R is what makes arm A tractable at all; applying the identical
restriction to arm B is what keeps the comparison honest.  Runtime is recorded
for both, since arm A's cost on the full universal is the reason R exists.
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
ap.add_argument("--gmin", type=float, default=0.1)
ap.add_argument("--timeout", type=int, default=3000)
ap.add_argument("--int_thresh", type=float, default=1e-9)
ap.add_argument("--outdir", default="/ibex/scratch/projects/c2014/kexin/funcarve/"
                                    "meteor_v8/results/gapfill_paired")
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
OUT = os.path.join(a.outdir, f"gfp_{a.gca}.json")
if os.path.exists(OUT): print("done"); sys.exit(0)

from meteor_v8.utils import (load_universal, extract_fba_matrices, load_tight_bounds,
                         apply_media, build_submodel, load_refmapping, load_ec,
                         extract_pred, build_rxn_ec_mask, find_excluded_reactions,
                                _detect_solver)
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
from meteor_v8.repair import grow_support
from cobra.flux_analysis.gapfilling import GapFiller

universal, allrxns, allmet = load_universal()
for x in ([universal] + list(universal.reactions) + list(universal.metabolites)
          + list(universal.genes)):
    if not hasattr(x, "_annotation"): x._annotation = {}
NR = len(allrxns)
biomass_id = "biomass_GmPos" if a.gram == "positive" else "biomass_GmNeg"
oi = allrxns.index(biomass_id)

S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
lt, ut = load_tight_bounds(data_path('tight_bounds_v6_%s.pkl' % a.gram[:3]))
if lt is not None: lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
exc = find_excluded_reactions(S, lb, ub, allrxns, biomass_id)
lb, ub, _, _ = apply_media(["default"], allrxns, lb, ub)
ix = {rid: i for i, rid in enumerate(allrxns)}
BND, UPTAKE_LB = 100.0, -10.0

seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
anc = load_ec(data_path('all_ancestors.txt'))
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
p = resolve_baseline_pkl("dpz", "vanilla", a.gca, BASELINE_SUFFIX["dpz"])
if not p: print("no pred"); sys.exit(3)
pred = extract_pred(p, anc)

# ---- draft: threshold tau on the predictor, identical for both arms ----
hit = (pred.values >= a.tau).any(axis=0)
draft = np.array([bool((mask[j] == 1).any() and hit[mask[j] == 1].any()) for j in range(NR)])
media_ids = {allrxns[j] for j in range(NR) if allrxns[j].startswith("EX_") and lb[j] < 0}
n_draft = int(draft.sum())

# ---- R: shared candidate set, built without any METEOR confidence ----
# EC-annotated reactions alone cannot carry biomass: transporters and
# spontaneous reactions have no EC and the growing solution needs them.  The
# biomass-feasible skeleton supplies exactly those, and is derived from the
# universal and the medium per Gram group -- it sees no predictor output, so
# it favours neither arm.
from meteor_v8.milp_hard import biomass_feasible_skeleton
_solver, _sname = _detect_solver(threads=4, time_limit=600)
_feas, skel, _bm = biomass_feasible_skeleton(S, lb, ub, oi, a.gmin, solver=_solver)
# skeleton comes back as a set of indices, not a mask
skel_mask = np.zeros(NR, bool)
skel_mask[list(skel)] = True
ec_annotated = np.array([(mask[j] == 1).any() for j in range(NR)])
R = ec_annotated | skel_mask | draft
for j, rid in enumerate(allrxns):
    if rid in media_ids or j == oi: R[j] = True
R[list(exc)] = False
print(f"[{a.gca}] draft {n_draft} | shared candidate set R {int(R.sum())} / {NR} "
      f"(EC-annotated {int(ec_annotated.sum())}, skeleton {int(skel_mask.sum())}, excluded {len(exc)})", flush=True)

rec = dict(gca=a.gca, gram=a.gram, tau=a.tau, n_draft=n_draft,
           n_candidates=int(R.sum()), n_universal=NR)

# ================= Arm B: METEOR's LP repair, restricted to R =================
core = draft & R
t0 = time.time()
sup = grow_support(S, lb, ub, oi, R, a.gmin, core=core)
rec["meteor_s"] = round(time.time() - t0, 1)
if sup is None:
    rec.update(meteor_added=None, meteor_n=None, meteor_grows=False, meteor_err="infeasible")
    print(f"  [meteor] infeasible ({rec['meteor_s']}s)", flush=True)
else:
    added = int((sup & ~core).sum())
    rec.update(meteor_added=added, meteor_n=int((sup | core).sum()), meteor_grows=True,
               meteor_err=None)
    print(f"  [meteor] +{added} rxns, support {rec['meteor_n']}, {rec['meteor_s']}s", flush=True)

# ================= Arm A: COBRApy gapfill, restricted to the same R ==========
keep = [r for r, k in zip(universal.reactions, (draft | (R & (lb < 0) & np.array(
        [rid.startswith("EX_") for rid in allrxns])))) if k]
keep_ids = {r.id for r in keep}
for rid in media_ids: keep_ids.add(rid)
keep = [r for r in universal.reactions if r.id in keep_ids]
model = build_submodel(universal, keep, a.gca, biomass_id=biomass_id)
model.objective = biomass_id

def set_medium(m):
    n_up = 0
    for r in m.reactions:
        if r.id.startswith(("EX_", "DM_", "SK_")) or "biomass" in r.id.lower(): continue
        i = ix.get(r.id)
        r.lower_bound = max(-BND, float(lb[i])) if i is not None else -BND
        r.upper_bound = min(BND, float(ub[i])) if i is not None else BND
    for r in [x for x in m.reactions if x.id.startswith("EX_") and x.id != "EX_biomass"]:
        i = ix.get(r.id)
        if i is not None and lb[i] < 0:
            r.lower_bound = max(UPTAKE_LB, float(lb[i])); n_up += 1
        else:
            r.lower_bound = 0.0
        r.upper_bound = 1000.0
    return n_up

n_uptake = set_medium(model)
pre = model.slim_optimize(); pre = 0.0 if (pre is None or np.isnan(pre)) else pre
rec.update(n_rxn_pre=len(model.reactions), n_uptake=n_uptake, growth_pre=round(float(pre), 6))
print(f"  [cobrapy] submodel {len(model.reactions)} rxns, {n_uptake} uptakes, "
      f"pre-growth {pre:.4f}", flush=True)

# universal restricted to R, with our bounds -- the same space arm B searched
uni_R = universal.copy()
drop = [uni_R.reactions.get_by_id(allrxns[j]) for j in range(NR)
        if not R[j] and allrxns[j] in uni_R.reactions]
uni_R.remove_reactions(drop, remove_orphans=False)
for r in uni_R.reactions:
    i = ix.get(r.id)
    if i is None: continue
    if r.id.startswith("EX_") and r.id != "EX_biomass":
        r.lower_bound = max(UPTAKE_LB, float(lb[i])) if lb[i] < 0 else 0.0
        r.upper_bound = 1000.0
    elif "biomass" not in r.id.lower():
        r.lower_bound = max(-BND, float(lb[i])); r.upper_bound = min(BND, float(ub[i]))
print(f"  [cobrapy] restricted universal {len(uni_R.reactions)} rxns", flush=True)

try:
    model.solver.configuration.timeout = a.timeout
    uni_R.solver.configuration.timeout = a.timeout
except Exception:
    pass

# Even with the timeout set, CBC through optlang does not always honour it.
# Without a hard stop the job hits its SLURM wall limit and writes nothing at
# all, which is the worst outcome: no result and no evidence of why.
import signal as _signal
def _bell(signum, frame):
    raise TimeoutError(f"gap-fill exceeded {a.timeout + 120}s of wall clock")
_signal.signal(_signal.SIGALRM, _bell)
_signal.alarm(a.timeout + 120)

t0 = time.time(); n_added = 0; err = None
try:
    # CBC's integrality tolerance is looser than cobra's default validation
    # threshold of 1e-6, so a solution it does find gets rejected by the
    # validation step with "try lowering the integer threshold".  Reporting
    # that as a gap-fill failure would be wrong -- it is a tolerance mismatch,
    # not an inability to solve.
    # Use GapFiller directly rather than the gapfill() wrapper: the wrapper
    # takes a fixed argument list and cannot pass integer_threshold through.
    # CBC's integrality tolerance is looser than the 1e-6 default, so a
    # solution it does find gets rejected by the validation step. Reporting
    # that as a gap-fill failure would be wrong -- it is a tolerance
    # mismatch, not an inability to solve.
    _gf = GapFiller(model, uni_R, lower_bound=a.lb,
                    demand_reactions=False, exchange_reactions=False,
                    integer_threshold=a.int_thresh)
    # GapFiller.__init__ starts with self.model = model.copy(), and the copy
    # rebuilds the solver with default configuration -- a timeout set on the
    # model we passed in never reaches the problem that actually gets solved.
    # Set it on the GapFiller's own model instead.
    try:
        _gf.model.solver.configuration.timeout = a.timeout
    except Exception as _e:
        print(f"  [cobrapy] could not set solver timeout: {_e}", flush=True)
    res = _gf.fill(iterations=1)
    if res and len(res) > 0:
        have = {r.id for r in model.reactions}
        new = [r for r in res[0] if r.id not in have]
        if new: model.add_reactions(new)
        n_added = len(res[0]); set_medium(model)
except (Exception, TimeoutError) as e:
    err = f"{type(e).__name__}: {e}"
    print(f"  [cobrapy] error: {err}", flush=True)
finally:
    _signal.alarm(0)
rec["cobrapy_s"] = round(time.time() - t0, 1)
post = model.slim_optimize(); post = 0.0 if (post is None or np.isnan(post)) else post
rec.update(cobrapy_added=int(n_added), cobrapy_n=len(model.reactions),
           growth_post=round(float(post), 6), cobrapy_grows=bool(post > 1e-6),
           cobrapy_err=err)
print(f"  [cobrapy] +{n_added} rxns, growth {post:.4f}, grows={rec['cobrapy_grows']}, "
      f"{rec['cobrapy_s']}s", flush=True)

json.dump(rec, open(OUT, "w"))
print(f"[{a.gca}] meteor {rec['meteor_s']}s (+{rec.get('meteor_added')}) | "
      f"cobrapy {rec['cobrapy_s']}s (+{rec['cobrapy_added']})", flush=True)
