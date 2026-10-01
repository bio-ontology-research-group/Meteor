"""Set relations between skeleton K, medium M and excluded X per Gram group (for Fig 1 step-3 bar). No MILP beyond the skeleton."""
import json, numpy as np
from _env import *
from meteor_v8.utils import (data_path, load_universal, extract_fba_matrices, load_tight_bounds, apply_media,
    find_excluded_reactions, _detect_solver)
from meteor_v8.milp_hard import biomass_feasible_skeleton
universal, allrxns, _ = load_universal()
S, lb0, ub0 = extract_fba_matrices(universal, allrxns, reversed_trans=True)
solver, _ = _detect_solver(threads=4, time_limit=600)
out = {}
for gram in ("negative", "positive"):
    bid = "biomass_GmPos" if gram == "positive" else "biomass_GmNeg"
    lb, ub = lb0.copy(), ub0.copy()
    lt, ut = load_tight_bounds(data_path(f"tight_bounds_v6_{gram[:3]}.pkl"))
    if lt is not None: lb = np.maximum(lb, lt); ub = np.minimum(ub, ut)
    oi = allrxns.index(bid); X = set(find_excluded_reactions(S, lb, ub, allrxns, bid))
    lb, ub, _, media = apply_media(["default"], allrxns, lb, ub)
    _, skel, _ = biomass_feasible_skeleton(S, lb, ub, oi, MILP_FLAGS["gmin"], solver=solver)
    K, M = set(skel), set(media)
    out[gram] = dict(K=len(K), M=len(M), X=len(X), KX=len(K & X), MK=len(M & K), MX=len(M & X),
                     K_or_M_minus_X=len((K | M) - X))
    print(gram, out[gram], flush=True)
json.dump(out, open(f"{RESULTS}/candidate_partition_sets.json", "w"), indent=1)
