"""B4: repair frequency and added reactions per configuration from the psb_revision reruns
(11 configs) plus the deposited dpz_vanilla rerun status (solver_status_dpz_vanilla.json)."""
import json, glob, pickle, os, statistics as st
from _env import *
out = {}
for d in sorted(glob.glob(f"{R}/meteor_out/repair_rerun/*/")):
    cfg = os.path.basename(d.rstrip("/")); rec = []
    for f in glob.glob(f"{d}/meteor_sol_*.pkl"):
        s = pickle.load(open(f, "rb")); rec.append(dict(n_repaired=int(s.get("n_repaired", 0)), status=s.get("status"), solve_sec=s.get("solve_sec")))
    if rec: out[cfg] = dict(n=len(rec), repaired=sum(1 for r in rec if r["n_repaired"] > 0),
        added=[r["n_repaired"] for r in rec if r["n_repaired"] > 0], not_optimal=sum(1 for r in rec if r["status"] != "Optimal"),
        solve_sec_median=st.median(r["solve_sec"] for r in rec if r["solve_sec"] is not None))
d = json.load(open(f"{ORIG_RESULTS}/solver_status_dpz_vanilla.json"))
out["dpz_vanilla(deposited rerun)"] = dict(n=d["n_genomes"], repaired=d["n_repaired"]["genomes_with_repair"], added=d["n_repaired"]["counts"],
    not_optimal=d["status_counts"].get("Not Solved", 0), solve_sec_median=d["solve_sec"]["median"])
json.dump(out, open(f"{RESULTS}/repair_by_config.json", "w"), indent=1); print(json.dumps(out, indent=1))
