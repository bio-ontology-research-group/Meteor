"""P1: curated-EC recall stratified by predictor vocabulary, baseline confidence bin and skeleton
reachability, for arms B / M-act / M-rxn / S / F (definitions frozen in PLAN.md section 2).
Read-only on every input; writes only under --outdir (default results/strat next to this file).
Usage: python strat_all.py GEM [GEM ...] [--preds clean dpz enzbert] [--outdir DIR]
Importable: run_one(gem, pred, ctx) returns the per-(gem,pred) dict.
"""
import sys, os, re, json, time, pickle, argparse, warnings, logging
import numpy as np, pandas as pd, cobra
warnings.filterwarnings("ignore"); logging.getLogger("cobra").setLevel(logging.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision/eval")
from _env import *
from meteor_v8.utils import (load_universal, load_refmapping, load_ec, data_path, data_dir, build_rxn_ec_mask,
                             extract_pred, extract_fba_matrices, load_tight_bounds, apply_media, _detect_solver,
                             find_excluded_reactions, build_candidate_mask)
from meteor_v8.milp_hard import biomass_feasible_skeleton
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
from meteor_v8.utils import _load_enzyme_obsolete, _resolve_current_ec

# --remap: map every EC set (reference, vocab, scores, all arms, reachability) old->new through enzymeobsolete.pkl.
# Default identity = the frozen PLAN section 2 protocol.
CANON = lambda e: e
REMAP = False
def set_remap(on):
    global CANON, REMAP
    REMAP = bool(on)
    if on:
        obs = _load_enzyme_obsolete(); CANON = lambda e: _resolve_current_ec(e, obs)
def cset(X): return {CANON(e) for e in X}

FULL = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
GEM2G = {"iML1515": ("GCF_058436375.1", "E.coli", "negative"), "STM_v1_0": ("GCF_000006945.2", "Salmonella", "negative"),
         "iYL1228": ("GCF_058435815.1", "K.pneumoniae", "negative"), "iJN1463": ("GCF_045571375.1", "P.putida", "negative"),
         "iYS854": ("GCF_045348045.1", "S.aureus", "positive"), "iYO844": ("GCF_058182495.1", "B.subtilis", "positive")}
T4 = f"{RESULTS}/skeleton_abl"                     # T4 outputs (read-only)
THR_B = 0.5
BINS = [("0", lambda s: s <= 0.0), ("(0,0.1)", lambda s: 0.0 < s < 0.1),
        ("[0.1,0.5)", lambda s: 0.1 <= s < 0.5), ("[0.5,1]", lambda s: s >= 0.5)]

def norm(x):
    x = str(x).strip().split("EC:")[-1]
    return x if FULL.match(x) else None

def gem_ecs(gem):
    m = cobra.io.read_sbml_model(f"{GEMDIR}/{gem}.xml"); ex = set()
    for r in m.reactions:
        a = r.annotation.get("ec-code") if hasattr(r, "annotation") else None
        if not a: continue
        for x in (a if isinstance(a, list) else [a]):
            n = norm(x)
            if n: ex.add(n)
    return ex

def rec(O, X):
    return dict(n=len(X), hits=len(O & X), recall=(round(len(O & X) / len(X), 4) if X else None))

# ---------------------------------------------------------------- shared context
def build_ctx(cache_dir):
    """Universal network, EC maps, and the two Gram-group skeletons (built with skeleton_ablation.py's
    exact call sequence; cached as JSON index lists under cache_dir)."""
    t0 = time.time()
    universal, allrxns, _ = load_universal()
    seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
    anc = load_ec(data_path("all_ancestors.txt")); anc_set = set(anc)
    mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)                      # anc-restricted map (S7.4 / diag convention)
    rxn2ec = {j: {n for x in seedr2ec.get(r.split("_")[0], []) if (n := norm(x))} for j, r in enumerate(allrxns)}  # rfinal2ec convention
    ec2rxn = {}
    for j, es in rxn2ec.items():
        for e in es: ec2rxn.setdefault(e, set()).add(j)
    skel, excl, media = {}, {}, {}
    for gram in ("negative", "positive"):
        cp = f"{cache_dir}/skeleton_{gram}.json"
        if os.path.exists(cp) and "excludes" in json.load(open(cp)):
            d = json.load(open(cp)); skel[gram] = set(d["skeleton_idx"]); excl[gram] = list(d["excludes"]); media[gram] = list(d["media_rxns"]); continue
        for x in list(universal.reactions) + list(universal.metabolites) + list(universal.genes):
            if not hasattr(x, "_annotation"): x._annotation = {}
        S, lb, ub = extract_fba_matrices(universal, allrxns, reversed_trans=True)
        lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{gram[:3]}.pkl"))
        if lt is not None: lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
        bid = "biomass_GmPos" if gram == "positive" else "biomass_GmNeg"; obj = allrxns.index(bid)
        ex = find_excluded_reactions(S, lb, ub, allrxns, bid)               # same call as skeleton_ablation.py
        lb, ub, _, media_rxns = apply_media(["default"], allrxns, lb, ub)
        solver, _ = _detect_solver(threads=1, time_limit=600)
        feas, sk, bm = biomass_feasible_skeleton(S, lb, ub, obj, MILP_FLAGS["gmin"], solver=solver)
        skel[gram] = set(int(j) for j in sk); excl[gram] = [int(j) for j in ex]; media[gram] = [int(j) for j in media_rxns]
        json.dump(dict(gram=gram, feasible=bool(feas), n_skeleton=len(sk), bm_max=float(bm), skeleton_idx=sorted(skel[gram]),
                       excludes=excl[gram], media_rxns=media[gram]), open(cp, "w"))
    print(f"[ctx {time.time()-t0:.1f}s] rxns={len(allrxns)} skeleton neg/pos={len(skel['negative'])}/{len(skel['positive'])}", flush=True)
    return dict(allrxns=allrxns, anc=anc, anc_set=anc_set, mask=mask, rxn2ec=rxn2ec, ec2rxn=ec2rxn, skel=skel, excl=excl, media=media)

def candidate_mask(P, ctx, gram):
    """Published candidate mask: w_j = noisy-OR over (protein, EC) with per-protein top-5 truncation, exactly as
    emit_v8.py / skeleton_ablation.py; candidates = {w >= 0.01} U skeleton U media, minus excludes."""
    P = P.astype(np.float32).copy(); ne = P.shape[1]
    if ne > 5:
        dr = np.argpartition(P, ne - 5, axis=1)[:, :ne - 5]; np.put_along_axis(P, dr, 0.0, axis=1)
    colsum = np.log(np.clip(1.0 - P, 1e-9, 1.0)).sum(axis=0).astype(np.float64)       # sum over proteins
    w = 1.0 - np.exp(ctx["mask"].astype(np.float64) @ colsum)                          # sum over ECs of reaction j
    w = np.clip(np.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0), 1e-6, 1 - 1e-6)
    return build_candidate_mask(w, ctx["allrxns"], ctx["excl"][gram], ctx["media"][gram], essential_skeleton=ctx["skel"][gram], w_min=MILP_FLAGS["wmin"]), w

def ecs_of_y(y, ctx, anc_only=False):
    """seedr2ec over selected reactions (y>0.5). anc_only=True reproduces diag_recall_denominators' mask
    convention (ECs restricted to all_ancestors.txt); False is the rfinal2ec convention."""
    act = np.where(np.asarray(y) > 0.5)[0]; out = set()
    if anc_only:
        anc = ctx["anc"]; mask = ctx["mask"]
        for j in act: out.update(anc[e] for e in np.where(mask[j] == 1)[0] if FULL.match(anc[e]))
    else:
        for j in act: out.update(ctx["rxn2ec"][j])
    return out

# ---------------------------------------------------------------- one (gem, pred)
def run_one(gem, pred, ctx):
    gcf, org, gram = GEM2G[gem]; t0 = time.time()
    R = cset(gem_ecs(gem))
    pkl = resolve_baseline_pkl(pred, "vanilla", gcf, BASELINE_SUFFIX[pred])
    raw = pd.read_pickle(pkl)
    if raw.index[0].count(".") == 3: raw = raw.T
    V = cset({n for c in raw.columns if (n := norm(c))})
    raw_pmax = raw.values.max(axis=1).astype(float)
    P = extract_pred(pkl, ctx["anc"])                                        # as METEOR sees it
    cols = list(P.columns); mx = P.values.max(axis=0); seen_pmax = P.values.max(axis=1)
    score = {}
    for j in range(len(cols)):
        if FULL.match(cols[j]): score[CANON(cols[j])] = max(score.get(CANON(cols[j]), 0.0), float(mx[j]))
    s = lambda e: score.get(e, 0.0)
    # arms
    B = {e for e, v in score.items() if v >= THR_B}
    prd = pickle.load(open(f"{RUNS}/{pred}_vanilla/meteor_preds_{gcf}.pkl", "rb"))
    M_act = cset({n for e in prd["active_ecs"] if (n := norm(e))})
    sol = pickle.load(open(f"{RUNS}/{pred}_vanilla/meteor_sol_{gcf}.pkl", "rb"))
    M_rxn = cset(ecs_of_y(sol["y_vals"], ctx, anc_only=True))
    M_rxn_full = cset(ecs_of_y(sol["y_vals"], ctx, anc_only=False))
    yS = pickle.load(open(f"{T4}/{gcf}_skelonly_y.pkl", "rb"))["y_vals"]
    yF = pickle.load(open(f"{T4}/{gcf}_full_y.pkl", "rb"))["y_vals"]
    Sset = cset(ecs_of_y(yS, ctx, anc_only=True)); Fset = cset(ecs_of_y(yF, ctx, anc_only=True))
    arms = {"B": B, "M-act": M_act, "M-rxn": M_rxn, "S": Sset, "F": Fset, "M-rxn-allSEED": M_rxn_full, "S-allSEED": cset(ecs_of_y(yS, ctx, False))}
    # reachability
    sk = ctx["skel"][gram]; ec2rxn = ctx["ec2rxn"]
    def reach(e):
        js = ec2rxn.get(e)
        if not js: return "not_in_seed"
        return "skeleton" if js & sk else "universal_only"
    if not REMAP:
        ec2rxn_c = ec2rxn
    else:                                            # merge reaction sets of old and new numbers
        ec2rxn_c = {}
        for e, js in ec2rxn.items(): ec2rxn_c.setdefault(CANON(e), set()).update(js)
        def reach(e):
            js = ec2rxn_c.get(e)
            if not js: return "not_in_seed"
            return "skeleton" if js & sk else "universal_only"
    RE = {e: reach(e) for e in R}
    cand, w = candidate_mask(P.values, ctx, gram); cand_idx = set(np.where(cand)[0].tolist())
    def has_cand(e): return bool(ec2rxn_c.get(e, set()) & cand_idx)
    # strata
    R_in, R_out = R & V, R - V
    strata = {"all": R, "in_vocab": R_in, "out_vocab": R_out}
    for name, f in BINS: strata[f"bin{name}"] = {e for e in R if f(s(e))}
    strata["bin0_in_vocab"] = strata["bin0"] & R_in
    for rc in ("skeleton", "universal_only", "not_in_seed"):
        X = {e for e in R if RE[e] == rc}
        strata[f"reach_{rc}"] = X; strata[f"out_vocab_reach_{rc}"] = X & R_out; strata[f"in_vocab_reach_{rc}"] = X & R_in
    strata["H3_universal_only_score_lt_0.01"] = {e for e in strata["reach_universal_only"] if s(e) < 0.01}
    strata["H3_universal_only_score_0"] = {e for e in strata["reach_universal_only"] if s(e) <= 0.0}
    strata["no_candidate_rxn"] = {e for e in R if not has_cand(e)}            # true hard ceiling: no mapped reaction is a MILP candidate
    strata["out_vocab_no_candidate_rxn"] = strata["no_candidate_rxn"] & R_out
    strata["in_vocab_no_candidate_rxn"] = strata["no_candidate_rxn"] & R_in
    table = {a: {x: rec(O, X) for x, X in strata.items()} for a, O in arms.items()}
    reach_counts = {rc: len(strata[f"reach_{rc}"]) for rc in ("skeleton", "universal_only", "not_in_seed")}
    reach_counts_out = {rc: len(strata[f"out_vocab_reach_{rc}"]) for rc in ("skeleton", "universal_only", "not_in_seed")}
    dark = {}
    for t in (0.5, 0.01):
        dark[f"raw_max_lt_{t}"] = round(float((raw_pmax < t).mean()), 4)
        dark[f"seen_max_lt_{t}"] = round(float((seen_pmax < t).mean()), 4)
    out = dict(organism=org, gem=gem, gcf=gcf, gram=gram, pred=pred, gem_ec=len(R), vocab_size=len(V),
               n_proteins=int(raw.shape[0]), n_in_vocab=len(R_in), n_out_vocab=len(R_out),
               bin_sizes={f"bin{n}": len(strata[f"bin{n}"]) for n, _ in BINS},
               reach_counts_all=reach_counts, reach_counts_out_vocab=reach_counts_out,
               dark=dark, arm_sizes={a: len(O) for a, O in arms.items()},
               n_candidate=int(cand.sum()), n_no_candidate_rxn=len(strata["no_candidate_rxn"]),
               n_out_vocab_no_candidate_rxn=len(strata["out_vocab_no_candidate_rxn"]),
               sanity=dict(M_act_recall_all=table["M-act"]["all"]["recall"], M_rxn_recall_all=table["M-rxn"]["all"]["recall"],
                           M_rxn_allSEED_recall_all=table["M-rxn-allSEED"]["all"]["recall"],
                           S_n_selected=int((np.asarray(yS) > 0.5).sum()), F_n_selected=int((np.asarray(yF) > 0.5).sum()),
                           M_n_selected=int((np.asarray(sol["y_vals"]) > 0.5).sum()),
                           F_recall_all=table["F"]["all"]["recall"], S_recall_all=table["S"]["all"]["recall"],
                           in_plus_out_eq_gem=(len(R_in) + len(R_out) == len(R)),
                           bins_sum_eq_gem=(sum(len(strata[f"bin{n}"]) for n, _ in BINS) == len(R)),
                           M_rxn_subset_M_act=M_rxn <= M_act, M_act_superset_B=B <= M_act,
                           b_not_in_mact=len(B - M_act), b_curated_not_in_mact=len((B & R) - M_act),
                           M_act_eq_M_rxn_allSEED=(M_act == M_rxn_full),
                           F_minus_Mrxn=round(table["F"]["all"]["recall"] - table["M-rxn"]["all"]["recall"], 4)),
               table=table, sec=round(time.time() - t0, 1))
    return out

def check_gates(o):
    """PLAN section 3 gates for the dpz run, against results/recall_denominators.json."""
    g = {}
    if o["pred"] == "dpz":
        ref = {r["organism"]: r for r in json.load(open(f"{RESULTS}/recall_denominators.json"))}[o["organism"]]
        g["M_act_eq_table3_R"] = (abs(o["sanity"]["M_act_recall_all"] - ref["table3_R"]) < 0.0015, o["sanity"]["M_act_recall_all"], ref["table3_R"])
        g["M_rxn_eq_samepred_R"] = (abs(o["sanity"]["M_rxn_recall_all"] - ref["samepred_R"]) < 0.0015, o["sanity"]["M_rxn_recall_all"], ref["samepred_R"])
    g["in_plus_out"] = (o["sanity"]["in_plus_out_eq_gem"],); g["bins_sum"] = (o["sanity"]["bins_sum_eq_gem"],)
    t4 = json.load(open(f"{T4}/{o['gcf']}_skelonly.json"))
    g["S_n_selected_eq_T4"] = (o["sanity"]["S_n_selected"] == t4["n_selected"], o["sanity"]["S_n_selected"], t4["n_selected"])
    g["F_within_0.01_of_Mrxn"] = (abs(o["sanity"]["F_minus_Mrxn"]) <= 0.01, o["sanity"]["F_minus_Mrxn"])
    if o["pred"] == "dpz":
        t4f = json.load(open(f"{T4}/{o['gcf']}_full.json"))
        g["n_candidate_eq_T4_full"] = (o["n_candidate"] == t4f["n_candidate"], o["n_candidate"], t4f["n_candidate"])
    return {k: dict(ok=bool(v[0]), vals=list(v[1:])) for k, v in g.items()}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("gems", nargs="+")
    ap.add_argument("--preds", nargs="+", default=["clean", "dpz", "enzbert"])
    ap.add_argument("--outdir", default=f"{HERE}/results/strat")
    ap.add_argument("--remap", action="store_true", help="map all EC sets old->new via enzymeobsolete.pkl (remap-consistent protocol)")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True); set_remap(a.remap)
    ctx = build_ctx(a.outdir)
    for gem in a.gems:
        for pred in a.preds:
            o = run_one(gem, pred, ctx); o["remap"] = bool(a.remap); o["gates"] = check_gates(o) if not a.remap else {"skipped": {"ok": True, "vals": ["remap protocol: published-number gates not applicable"]}}
            jp = f"{a.outdir}/{gem}_{pred}.json"; json.dump(o, open(jp, "w"), indent=1)
            T = o["table"]; f = lambda a_, x: T[a_][x]["recall"]
            print(f"{o['organism']:13s} {pred:8s} R={o['gem_ec']} in/out={o['n_in_vocab']}/{o['n_out_vocab']} reach={o['reach_counts_all']} "
                  f"| all B/Mrxn/Mact/S/F={f('B','all')}/{f('M-rxn','all')}/{f('M-act','all')}/{f('S','all')}/{f('F','all')} "
                  f"| in B/Mrxn/S={f('B','in_vocab')}/{f('M-rxn','in_vocab')}/{f('S','in_vocab')} "
                  f"| out Mrxn/Mact/S={f('M-rxn','out_vocab')}/{f('M-act','out_vocab')}/{f('S','out_vocab')} "
                  f"| dark0.5={o['dark']['raw_max_lt_0.5']} dark0.01={o['dark']['raw_max_lt_0.01']} | n_cand={o['n_candidate']} noCand={o['n_no_candidate_rxn']} (out {o['n_out_vocab_no_candidate_rxn']}) [{o['sec']}s]", flush=True)
            print("   gates:", {k: (v["ok"], v["vals"]) for k, v in o["gates"].items()}, flush=True)
