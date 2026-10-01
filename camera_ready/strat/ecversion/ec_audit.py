"""EC-version audit: where enzymeobsolete.pkl is (not) applied, and what it would change.
Read-only; writes ec_audit.json next to this file. Usage: python ec_audit.py [--preds dpz clean enzbert] [GEM ...]"""
import sys, os, re, json, pickle, argparse, collections, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval")
from _env import *
from meteor_v8.utils import _load_enzyme_obsolete, _resolve_current_ec, extract_pred, load_refmapping, data_dir
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
import strat_all as SA
FULL = SA.FULL; norm = SA.norm
ap = argparse.ArgumentParser(); ap.add_argument("gems", nargs="*", default=list(SA.GEM2G)); ap.add_argument("--preds", nargs="+", default=["dpz", "clean", "enzbert"])
a = ap.parse_args()
obs = _load_enzyme_obsolete(); canon = lambda e: _resolve_current_ec(e, obs)
keys, vals = set(obs), set(obs.values())
out = dict(table=dict(n_entries=len(obs), n_keys_full=sum(bool(FULL.match(k)) for k in obs), n_vals_full=sum(bool(FULL.match(v)) for v in vals),
    n_keys_that_are_also_values_chain=len(keys & vals), n_self_map=sum(k == v for k, v in obs.items()),
    n_targets_distinct=len(vals), n_chains_gt1=sum(canon(k) != obs[k] for k in obs),
    n_deleted_entries="0 (dict old->new only; EC 'deleted' entries cannot be represented and are absent)"))
ctx = SA.build_ctx(f"{os.path.dirname(HERE)}/results/strat")
seedr2ec, _ = load_refmapping(data_dir()); anc_set = ctx["anc_set"]
seed_raw = collections.Counter(str(x) for v in seedr2ec.values() if v for x in v)
seed_full = {e for e in seed_raw if FULL.match(e)}; seed_partial = {e for e in seed_raw if not FULL.match(e)}
miss = seed_full - anc_set
def breakdown(E):
    b = dict(n=len(E), obsolete_key=0, obsolete_key_target_in_anc=0, obsolete_key_target_in_seed=0, is_current_target_of_transfer=0, not_in_table=0, partial=0, examples_obsolete=[])
    for e in sorted(E):
        if not FULL.match(e): b["partial"] += 1; continue
        if e in keys:
            b["obsolete_key"] += 1; t = canon(e); b["obsolete_key_target_in_anc"] += t in anc_set; b["obsolete_key_target_in_seed"] += t in seed_full
            if len(b["examples_obsolete"]) < 8: b["examples_obsolete"].append(f"{e}->{t}")
        elif e in vals: b["is_current_target_of_transfer"] += 1
        else: b["not_in_table"] += 1
    return b
out["seedr2ec"] = dict(n_raw_distinct=len(seed_raw), n_full=len(seed_full), n_partial=len(seed_partial), n_full_missing_from_anc=len(miss),
                       missing_breakdown=breakdown(miss), n_seed_full_obsolete_keys_total=len(seed_full & keys),
                       n_anc_entries_obsolete_keys=len(anc_set & keys), n_anc_full=sum(bool(FULL.match(e)) for e in anc_set))
out["organisms"] = {}
for gem in a.gems:
    gcf, org, gram = SA.GEM2G[gem]; R = SA.gem_ecs(gem); Rc = {canon(e) for e in R}
    o = dict(gem=gem, gcf=gcf, gem_ec=len(R), gem_ec_obsolete_keys=len(R & keys), gem_ec_targets=len(R & vals), gem_ec_after_canon=len(Rc),
             gem_obsolete_examples=[f"{e}->{canon(e)}" for e in sorted(R & keys)][:10], gem_ec_not_in_anc=len(R - anc_set), preds={})
    for pred in a.preds:
        pkl = resolve_baseline_pkl(pred, "vanilla", gcf, BASELINE_SUFFIX[pred]); raw = pd.read_pickle(pkl)
        if raw.index[0].count(".") == 3: raw = raw.T
        cols_raw = [norm(c) for c in raw.columns]; V = {c for c in cols_raw if c}; Vc = {canon(c) for c in V}
        P = SA.extract_pred(pkl, ctx["anc"]); mx = P.values.max(axis=0); cols = list(P.columns)
        score = {cols[j]: float(mx[j]) for j in range(len(cols)) if FULL.match(cols[j])}
        score_c = {}
        for e, s in score.items(): score_c[canon(e)] = max(score_c.get(canon(e), 0.0), s)
        prd = pickle.load(open(f"{RUNS}/{pred}_vanilla/meteor_preds_{gcf}.pkl", "rb")); A = {n for e in prd["active_ecs"] if (n := norm(e))}
        sol = pickle.load(open(f"{RUNS}/{pred}_vanilla/meteor_sol_{gcf}.pkl", "rb")); Mrxn = SA.ecs_of_y(sol["y_vals"], ctx, True)
        yS = pickle.load(open(f"{SA.T4}/{gcf}_skelonly_y.pkl", "rb"))["y_vals"]; S = SA.ecs_of_y(yS, ctx, True); Sall = SA.ecs_of_y(yS, ctx, False)
        B = {e for e, s in score.items() if s >= 0.5}; Bc = {e for e, s in score_c.items() if s >= 0.5}
        arms = {"B": (B, Bc), "M-act": (A, {canon(e) for e in A}), "M-rxn": (Mrxn, {canon(e) for e in Mrxn}), "S": (S, {canon(e) for e in S}), "S-allSEED": (Sall, {canon(e) for e in Sall})}
        Aminus = A - anc_set
        p = dict(vocab=len(V), vocab_obsolete_keys=len(V & keys), vocab_targets=len(V & vals), vocab_after_canon=len(Vc), vocab_not_in_anc=len(V - anc_set),
                 vocab_partial_cols=sum(1 for c in raw.columns if norm(c) is None), n_cols_extract_pred_remapped=len([c for c in V if c not in anc_set and c in keys and canon(c) in anc_set]),
                 active_minus_anc=breakdown(Aminus),
                 active_minus_anc_after_canon_in_R=len({canon(e) for e in Aminus} & Rc), active_minus_anc_in_R_before=len(Aminus & R),
                 active_minus_anc_after_canon_in_vocab=len({canon(e) for e in Aminus} & Vc), active_minus_anc_in_vocab_before=len(Aminus & V),
                 out_vocab_before=len(R - V), out_vocab_after_canon=len(Rc - Vc), out_vocab_rescued_by_canon=len({e for e in R - V if canon(e) in Vc}),
                 arms={})
        for name, (O, Oc) in arms.items():
            p["arms"][name] = dict(recall_before=round(len(O & R) / len(R), 4), recall_after_canon=round(len(Oc & Rc) / len(Rc), 4),
                                   hits_before=len(O & R), hits_after=len(Oc & Rc), n_R_after=len(Rc),
                                   misses_becoming_hits=len({e for e in R - O if canon(e) in Oc}))
        o["preds"][pred] = p
        print(org, pred, {k: v for k, v in p.items() if k not in ("arms", "active_minus_anc")}, {k: (v["recall_before"], v["recall_after_canon"]) for k, v in p["arms"].items()}, flush=True)
    out["organisms"][org] = o; print(org, {k: v for k, v in o.items() if k != "preds"}, flush=True)
print(json.dumps(out["table"], indent=1)); print(json.dumps(out["seedr2ec"], indent=1))
json.dump(out, open(f"{HERE}/ec_audit.json", "w"), indent=1); print("->", f"{HERE}/ec_audit.json")
