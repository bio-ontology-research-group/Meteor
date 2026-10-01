"""A6: candidate-set, evidence and skeleton sizes per genome (DPZ vanilla). No MILP solve."""
import json, statistics as st, numpy as np
from _env import *
from meteor_v8.utils import (data_path, data_dir, load_universal, extract_fba_matrices, load_tight_bounds, apply_media,
    find_excluded_reactions, build_candidate_mask, build_rxn_ec_mask, extract_pred, load_refmapping, load_ec, _detect_solver)
from meteor_v8.milp_hard import biomass_feasible_skeleton
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, _ = load_universal(); anc = load_ec(data_path("all_ancestors.txt"))
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
S, lb0, ub0 = extract_fba_matrices(universal, allrxns, reversed_trans=True)
solver, _ = _detect_solver(threads=4, time_limit=600)
G = {}
def gram_ctx(gram):
    if gram in G: return G[gram]
    bid = "biomass_GmPos" if gram == "positive" else "biomass_GmNeg"
    lb, ub = lb0.copy(), ub0.copy()
    lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{gram[:3]}.pkl"))
    if lt is not None: lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
    oi = allrxns.index(bid); exc = find_excluded_reactions(S, lb, ub, allrxns, bid)
    lb, ub, _, media = apply_media(["default"], allrxns, lb, ub)
    _, skel, _ = biomass_feasible_skeleton(S, lb, ub, oi, MILP_FLAGS["gmin"], solver=solver)
    G[gram] = dict(exc=exc, media=media, skel=set(skel)); return G[gram]
rows = []
for gca, gram in panel_rows():
    ctx = gram_ctx(gram)
    p = resolve_baseline_pkl("dpz", "vanilla", gca, BASELINE_SUFFIX["dpz"])
    if not p: print("no pred", gca, flush=True); continue
    P = extract_pred(p, anc).values.astype(np.float32).copy(); ne = P.shape[1]
    if ne > 5:
        dr = np.argpartition(P, ne-5, axis=1)[:, :ne-5]; np.put_along_axis(P, dr, 0.0, axis=1)
    l1m = np.log(np.clip(1.0-P, 1e-9, 1.0)); w = np.zeros(len(allrxns))
    for j in range(len(allrxns)):
        ei = np.where(mask[j] == 1)[0]
        if len(ei): w[j] = 1.0 - np.exp(float(l1m[:, ei].sum()))
    w = np.clip(np.nan_to_num(w), 1e-6, 1-1e-6)
    cand = build_candidate_mask(w, allrxns, ctx["exc"], ctx["media"], essential_skeleton=ctx["skel"], w_min=MILP_FLAGS["wmin"])
    rows.append(dict(gca=gca, gram=gram, n_skeleton=len(ctx["skel"]), n_evidence=int((w >= MILP_FLAGS["wmin"]).sum()),
                     n_skel_without_evidence=int(sum(1 for j in ctx["skel"] if w[j] < MILP_FLAGS["wmin"])),
                     n_medium=len(ctx["media"]), n_excluded=len(ctx["exc"]), n_candidate=int(cand.sum()), n_universal=len(allrxns)))
    print(rows[-1], flush=True)
summ = {}
for gram in ("negative", "positive"):
    xs = [r["n_candidate"] for r in rows if r["gram"] == gram]
    if xs: summ[gram] = dict(n=len(xs), cand_mean=st.mean(xs), cand_sd=st.stdev(xs) if len(xs) > 1 else 0.0,
                             n_skeleton=len(G[gram]["skel"]), evidence_mean=st.mean(r["n_evidence"] for r in rows if r["gram"] == gram))
json.dump(dict(rows=rows, summary=summ), open(f"{RESULTS}/candidate_size_dpz_vanilla.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
