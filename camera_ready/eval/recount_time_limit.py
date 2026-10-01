"""Time-limit proxy per configuration: solve calls with wall time >= 600 s.

PuLP labels a CBC run that hit the time limit with a feasible incumbent as
"Optimal", so the status field under-counts time-limit stops (the
"not_optimal"/"time_limit" fields in repair_by_config.json and
skeleton_ablation_summary.json are status-based and kept for the record).
This script recounts from the recorded wall time, which is what main-text
Section 2.2, Supplementary Section S1.1 and Table S1 report, and deposits the
per-genome wall times for all twelve configurations and the ablation arms.
"""
import glob
import json
import os
import pickle
import statistics as st

from _env import ORIG_RESULTS, R, RESULTS

LIMIT = 600.0
out = {"limit_sec": LIMIT, "rule": "solve_sec >= limit_sec", "configs": {}, "ablation_arms": {}}

for d in sorted(glob.glob(f"{R}/meteor_out/repair_rerun/*/")):
    cfg = os.path.basename(d.rstrip("/"))
    rec = {}
    for f in sorted(glob.glob(f"{d}/meteor_sol_*.pkl")):
        s = pickle.load(open(f, "rb"))
        gca = os.path.basename(f)[len("meteor_sol_"):-4]
        rec[gca] = {"solve_sec": s.get("solve_sec"), "status": s.get("status"),
                    "n_repaired": int(s.get("n_repaired", 0))}
    secs = [r["solve_sec"] for r in rec.values() if r["solve_sec"] is not None]
    out["configs"][cfg] = {
        "n": len(rec), "ge_limit": sum(1 for x in secs if x >= LIMIT),
        "status_not_optimal": sum(1 for r in rec.values() if r["status"] != "Optimal"),
        "solve_sec_median": st.median(secs), "per_genome": rec}

for _c in (f"{ORIG_RESULTS}/solver_status_dpz_vanilla.json", f"{R}/gh_sync_20260925/results/solver_status_dpz_vanilla.json"):
    if os.path.exists(_c):
        d = json.load(open(_c)); break
pg = d.get("per_genome") or d.get("genomes") or d.get("records") or {}
if isinstance(pg, list):
    pg = {r.get("gca", str(i)): r for i, r in enumerate(pg)}
secs = [r["solve_sec"] for r in pg.values() if isinstance(r, dict) and r.get("solve_sec") is not None]
out["configs"]["dpz_vanilla"] = {
    "n": d.get("n_genomes", len(pg)), "ge_limit": sum(1 for x in secs if x >= LIMIT),
    "status_counts": d.get("status_counts"), "solve_sec_median": d["solve_sec"]["median"],
    "source": "results/solver_status_dpz_vanilla.json", "per_genome": pg,
    "top_level_keys": sorted(d.keys())}

for arm in ("full", "skelonly", "noskel"):
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{RESULTS}/skeleton_abl/*_{arm}.json"))]
    ok = [r for r in rows if r.get("solve_sec") is not None]
    out["ablation_arms"][arm] = {
        "n": len(rows), "solved": len(ok), "ge_limit": sum(1 for r in ok if r["solve_sec"] >= LIMIT),
        "status_not_optimal": sum(1 for r in ok if r.get("status") != "Optimal"),
        "per_genome": {r["gca"]: r["solve_sec"] for r in ok}}

ORDER = ["clean_vanilla", "clean_filt30", "clean_filt50", "clean_filt70",
         "dpz_vanilla", "dpz_filt30", "dpz_filt50", "dpz_filt70",
         "enzbert_vanilla", "enzbert_filt30", "enzbert_filt50", "enzbert_filt70"]
out["total_ge_limit"] = sum(out["configs"][c]["ge_limit"] for c in ORDER if c in out["configs"])
out["total_n"] = sum(out["configs"][c]["n"] for c in ORDER if c in out["configs"])
json.dump(out, open(f"{RESULTS}/time_limit_recount.json", "w"), indent=1)
for c in ORDER:
    if c in out["configs"]:
        print(f"{c:16s} n={out['configs'][c]['n']:4d} >= {LIMIT:.0f}s: {out['configs'][c]['ge_limit']}")
print("total", out["total_ge_limit"], "of", out["total_n"])
for arm, v in out["ablation_arms"].items():
    print(arm, v["solved"], "solved,", v["ge_limit"], ">= limit")
