#!/usr/bin/env python3
"""Aggregate the paired gap-fill records into the numbers reported in
Supplementary Section S4: what happens when the same thresholded draft is
repaired by METEOR's flux-parsimony LP and by COBRApy's minimum-count MILP,
drawing from an identical candidate set.

Reads:  results/gapfill_paired/gfp_*.json    (108 genomes)

The two arms solve different problems, so this is a statement about
formulations rather than implementations: the LP minimises flux through
non-core reactions, the MILP minimises how many reactions are added and needs
one binary per candidate. Read the timing accordingly -- the MILP was cut off,
so its cost is a lower bound and the ratio between the arms is one too.
"""
import json, glob, os, sys
from statistics import median, mean

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "gapfill_paired")
if len(sys.argv) > 1:
    RES = sys.argv[1]

recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "gfp_*.json")))]
if not recs:
    sys.exit(f"no records under {RES}")
print(f"genomes: {len(recs)}\n")

cand = [r["n_candidates"] for r in recs]
draft = [r["n_draft"] for r in recs]
print(f"shared candidate set : median {median(cand):.0f} reactions "
      f"(range {min(cand)}-{max(cand)}) out of {recs[0]['n_universal']}")
print(f"thresholded draft    : median {median(draft):.0f} reactions "
      f"(range {min(draft)}-{max(draft)})\n")

mt = [r["meteor_s"] for r in recs]
ma = [r["meteor_added"] for r in recs if r["meteor_added"] is not None]
mg = sum(1 for r in recs if r["meteor_grows"])
print("METEOR, flux-parsimony LP over the candidate set")
print(f"  restored growth : {mg}/{len(recs)}")
print(f"  wall time       : median {median(mt):.1f}s  mean {mean(mt):.1f}s  "
      f"range {min(mt):.1f}-{max(mt):.1f}s")
if ma:
    print(f"  reactions added : median {median(ma):.0f}  range {min(ma)}-{max(ma)}")

ct = [r["cobrapy_s"] for r in recs]
cg = sum(1 for r in recs if r["cobrapy_grows"])
print("\nCOBRApy GapFiller, minimum-count MILP over the same candidate set")
print(f"  restored growth : {cg}/{len(recs)}")
print(f"  wall time       : median {median(ct):.1f}s  mean {mean(ct):.1f}s")
outcome = {}
for r in recs:
    k = (r.get("cobrapy_err") or "returned").split(":")[0]
    outcome[k] = outcome.get(k, 0) + 1
for k, v in sorted(outcome.items(), key=lambda x: -x[1]):
    print(f"  {k:<16}: {v}/{len(recs)}")

done = [r for r in recs if not r.get("cobrapy_err")]
if done:
    da = [r["cobrapy_added"] for r in done]
    print(f"\n  of the {len(done)} that returned: added median {median(da):.0f}, "
          f"{sum(1 for r in done if r['cobrapy_grows'])} grow")

print(f"\nratio of medians  : {median(ct)/median(mt):.0f}x")
print("  A lower bound. The MILP did not finish, so its true cost is larger and")
print("  the ratio with it. Nothing here shows the MILP cannot be solved -- only")
print("  that it did not return within the budget it was given.")
