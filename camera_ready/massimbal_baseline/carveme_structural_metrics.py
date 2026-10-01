"""Feasibility demo (R3 #4): apply Table 1's structural metrics to CarveMe / curated
BiGG models and (optionally) to METEOR selected sets, splitting mass-imbalance into
"metabolite lacks a formula" vs "genuinely unbalanced".

Metric calls are IDENTICAL to psb_revision/code_snapshot/eval/gen_table1.py::profile:
    deadends   = len(memote.support.consistency.find_deadends(model))
    mass_imbal = len(memote.support.consistency.find_mass_unbalanced_reactions(model.reactions))
    mi_frac    = mass_imbal / n_rxn
Extra diagnostic columns are labelled explicitly. Read-only on every input.
"""
import argparse
import json
import logging
import os
import pickle
import sys
import time
import warnings

sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval")
import _env  # noqa: E402  sets METEOR_DATA + sys.path to code_snapshot

import cobra  # noqa: E402
import numpy as np  # noqa: E402
import memote.support.consistency as cons  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("cobra").setLevel(logging.ERROR)
logger = logging.getLogger("fm")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

BOUNDARY_PREF = ("EX_", "DM_", "SK_", "R_EX_", "R_DM_", "R_SK_")
CARVEME_DIR = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026/results/carveme_gc"
CURATED_DIR = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_diag/curated_gems"
MEMOTE_CV = "/ibex/scratch/projects/c2014/kexin/funcarve/paperA_2026/results/memote_carveme_gc"
TABLE1 = f"{_env.ORIG_RESULTS}/table1"
CURATED6 = f"{_env.ORIG_RESULTS}/toolcompare/panelB_curated6.json"


def is_boundary(r, obj_ids=frozenset()):
    """Exclude exchange/demand/sink reactions and the biomass/objective reaction.
    Name-matching ("biomass"/"bio*") catches METEOR/curated-GEM biomass reactions;
    it does NOT catch CarveMe, whose biomass reaction is named "Growth" (BUG found
    2026-09-16: this let CarveMe's "Growth" reaction be counted as an internal
    genuinely-imbalanced reaction in every genome). obj_ids is the set of reaction
    ids carrying a nonzero linear objective coefficient in the model as loaded from
    its SBML file -- this is namespace-agnostic and catches "Growth" too."""
    return (r.id.startswith(BOUNDARY_PREF) or r.boundary
            or "biomass" in r.id.lower() or r.id.lower().startswith("bio")
            or r.id in obj_ids)


def profile(model, label):
    obj_ids = {rxn.id for rxn, coef in cobra.util.solver.linear_reaction_coefficients(model).items() if coef}
    t0 = time.time()
    rx = list(model.reactions)
    mi_all = cons.find_mass_unbalanced_reactions(rx)          # == gen_table1
    de = len(cons.find_deadends(model))                        # == gen_table1
    internal = [r for r in rx if not is_boundary(r, obj_ids)]
    mi_int = cons.find_mass_unbalanced_reactions(internal)
    mi_int_noformula = [r for r in mi_int if any(not m.formula for m in r.metabolites)]
    mi_int_genuine = [r for r in mi_int if all(m.formula for m in r.metabolites)]
    n_met = len(model.metabolites)
    n_nof = sum(1 for m in model.metabolites if not m.formula)
    try:
        g = float(model.slim_optimize())
        g = 0.0 if g is None or np.isnan(g) else g
    except Exception:
        g = float("nan")
    rec = dict(
        label=label, n_rxn=len(rx), n_internal=len(internal), n_met=n_met,
        objective_reaction_ids=sorted(obj_ids),
        n_met_noformula=n_nof, met_formula_cov=round(1 - n_nof / max(1, n_met), 4),
        deadends=de, mass_imbal=len(mi_all), mi_frac=round(len(mi_all) / max(1, len(rx)), 4),
        mass_imbal_internal=len(mi_int),
        mi_frac_internal=round(len(mi_int) / max(1, len(internal)), 4),
        mi_internal_due_to_missing_formula=len(mi_int_noformula),
        mi_internal_genuine=len(mi_int_genuine),
        mi_frac_internal_genuine=round(len(mi_int_genuine) / max(1, len(internal)), 4),
        fba_growth_as_shipped=round(g, 4), seconds=round(time.time() - t0, 1),
    )
    logger.info("[%s] rxn=%d de=%d mi=%d mi_frac=%.4f mi_int_frac=%.4f genuine=%.4f nof_met=%d (%.0fs)",
                label, rec["n_rxn"], de, rec["mass_imbal"], rec["mi_frac"],
                rec["mi_frac_internal"], rec["mi_frac_internal_genuine"], n_nof, rec["seconds"])
    return rec


def load_sbml(path):
    t0 = time.time()
    m = cobra.io.read_sbml_model(path)
    logger.info("read %s in %.0fs", os.path.basename(path), time.time() - t0)
    return m


def meteor_profile(gca, gram, baseline="dpz"):
    """METEOR selected set (y>0.5), submodel built as in gen_table1 (no medium step)."""
    from meteor_v8.utils import load_universal, build_submodel
    universal, allrxns, allmet = load_universal()
    for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
        if not hasattr(x, "_annotation"):
            x._annotation = {}
    out = {}
    out["universal"] = profile(universal, "SEED_universal")
    f = f"{_env.RUNS}/{baseline}_vanilla/meteor_sol_{gca}.pkl"
    yv = np.array(pickle.load(open(f, "rb")).get("y_vals", []))
    keep = [r for r, k in zip(universal.reactions, yv > 0.5) if k]
    bid = "biomass_GmPos" if gram == "positive" else "biomass_GmNeg"
    model = build_submodel(universal, keep, gca, biomass_id=bid)
    out[f"meteor_{baseline}"] = profile(model, f"METEOR_{baseline}_{gca}_nomedium")
    return out


def memote_cv(gca):
    p = f"{MEMOTE_CV}/{gca}.json"
    if not os.path.exists(p):
        return None
    t = json.load(open(p))["tests"]
    return dict(memote_mass_balance_metric=t["test_reaction_mass_balance"]["metric"],
                memote_deadends_metric=t["test_find_deadends"]["metric"],
                memote_unbounded_metric=t["test_find_reactions_unbounded_flux_default_condition"]["metric"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcas", nargs="*", default=None, help="default: 6 curated organisms")
    ap.add_argument("--curated", action="store_true", help="also profile curated BiGG GEM")
    ap.add_argument("--meteor", action="store_true", help="also rebuild METEOR dpz submodel (needs universal, ~GBs)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cur6 = {e["gcf"]: e for e in json.load(open(CURATED6))}
    gcas = a.gcas or list(cur6)
    res = {}
    for gca in gcas:
        rec = {"organism": cur6.get(gca, {}).get("organism"), "gem": cur6.get(gca, {}).get("gem")}
        t1 = f"{TABLE1}/table1_{gca}.json"
        if os.path.exists(t1):
            d = json.load(open(t1))
            rec["table1"] = {k: {kk: d[k][kk] for kk in ("n_rxn", "deadends", "mass_imbal", "mi_frac")}
                             for k in d if k.startswith(("baseline_", "meteor_")) and "err" not in d[k]}
            gram = d["gram"]
        else:
            gram = None
        rec["carveme"] = profile(load_sbml(f"{CARVEME_DIR}/{gca}.xml"), f"CarveMe_{gca}")
        rec["carveme_memote_existing"] = memote_cv(gca)
        if a.curated and rec["gem"]:
            rec["curated_bigg"] = profile(load_sbml(f"{CURATED_DIR}/{rec['gem']}.xml"), rec["gem"])
        if a.meteor and gram:
            rec.update(meteor_profile(gca, gram))
        res[gca] = rec
        json.dump(res, open(a.out, "w"), indent=1)
    logger.info("written %s", a.out)


if __name__ == "__main__":
    main()
