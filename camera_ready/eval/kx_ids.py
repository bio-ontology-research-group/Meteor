"""IDs and EC mappings of skeleton-and-excluded reactions (4 per Gram group) and of medium reactions."""
import json, numpy as np
from _env import *
from meteor_v8.utils import (data_path, data_dir, load_universal, extract_fba_matrices, load_tight_bounds, apply_media,
    find_excluded_reactions, build_rxn_ec_mask, load_refmapping, load_ec, _detect_solver)
from meteor_v8.milp_hard import biomass_feasible_skeleton
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, _ = load_universal(); anc = load_ec(data_path("all_ancestors.txt"))
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
S, lb0, ub0 = extract_fba_matrices(universal, allrxns, reversed_trans=True)
solver, _ = _detect_solver(threads=4, time_limit=600)
for gram in ("negative", "positive"):
    bid = "biomass_GmPos" if gram == "positive" else "biomass_GmNeg"
    lb, ub = lb0.copy(), ub0.copy()
    lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{gram[:3]}.pkl"))
    if lt is not None: lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
    oi = allrxns.index(bid); X = set(find_excluded_reactions(S, lb, ub, allrxns, bid))
    lb, ub, _, media = apply_media(["default"], allrxns, lb, ub)
    _, skel, _ = biomass_feasible_skeleton(S, lb, ub, oi, MILP_FLAGS["gmin"], solver=solver)
    kx = sorted(set(skel) & X)
    print(gram, "KX:", [(allrxns[j], int(mask[j].sum())) for j in kx])
    print(gram, "medium with EC:", sum(int(mask[j].sum() > 0) for j in media))
