"""Per genome: how many of the EC-mapped skeleton-and-excluded reactions have w >= wmin (to make the Fig 1 bar partition exact)."""
import json, numpy as np
from _env import *
from meteor_v8.utils import data_path, data_dir, load_universal, build_rxn_ec_mask, extract_pred, load_refmapping, load_ec
from baseline_io import resolve_baseline_pkl, BASELINE_SUFFIX
seedr2ec, _ = load_refmapping(data_dir()); seedr2ec = {k: v for k, v in seedr2ec.items() if v}
universal, allrxns, _ = load_universal(); anc = load_ec(data_path("all_ancestors.txt"))
mask = build_rxn_ec_mask(allrxns, seedr2ec, anc)
KX = [allrxns.index(r) for r in ("rxn00724_c", "rxn03234_c")]
out = {}
for gca, gram in panel_rows():
    p = resolve_baseline_pkl("dpz", "vanilla", gca, BASELINE_SUFFIX["dpz"])
    if not p: continue
    P = extract_pred(p, anc).values.astype(np.float32).copy(); ne = P.shape[1]
    if ne > 5:
        dr = np.argpartition(P, ne-5, axis=1)[:, :ne-5]; np.put_along_axis(P, dr, 0.0, axis=1)
    l1m = np.log(np.clip(1.0-P, 1e-9, 1.0))
    n = 0
    for j in KX:
        ei = np.where(mask[j] == 1)[0]
        w = 1.0 - np.exp(float(l1m[:, ei].sum())) if len(ei) else 0.0
        n += int(np.clip(w, 1e-6, 1-1e-6) >= MILP_FLAGS["wmin"])
    out[gca] = n
json.dump(out, open(f"{RESULTS}/candidate_partition_kx_evidence.json", "w"), indent=1)
print(len(out), sum(out.values()), {v: list(out.values()).count(v) for v in set(out.values())})
