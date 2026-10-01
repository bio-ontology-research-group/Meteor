"""Aggregate skeleton ablation arms against the published full run (results/table1 meteor_dpz)."""
import json, glob, statistics as st
from _env import *
rows = [json.load(open(f)) for f in glob.glob(f"{RESULTS}/skeleton_abl/*.json")]
full = {}
for f in glob.glob(f"{ORIG_RESULTS}/table1/table1_*.json"):
    o = json.load(open(f)); full[o["gca"]] = o["meteor_dpz"]
def summ(R):
    ok = [r for r in R if "err" not in r and r.get("fba_growth") is not None]
    if not ok: return dict(n=len(R), solved=0)
    return dict(n=len(R), solved=len(ok), infeasible=len(R)-len(ok),
        n_selected=round(st.mean(r["n_selected"] for r in ok), 1), deadends=round(st.mean(r["deadends"] for r in ok), 2),
        mi_frac=round(st.mean(r["mi_frac"] for r in ok), 3), growing=sum(1 for r in ok if r["fba_growth"] > 1e-6),
        repaired=sum(1 for r in ok if r.get("n_repaired", 0) > 0), solve_sec_median=st.median(r["solve_sec"] for r in ok),
        time_limit=sum(1 for r in ok if r["status"] != "Optimal"))
out = {arm: summ([r for r in rows if r["arm"] == arm]) for arm in ("noskel", "skelonly", "full")}
out["published_B3"] = dict(n=len(full), n_selected=round(st.mean(v["n_selected"] for v in full.values()), 1),
    deadends=round(st.mean(v["deadends"] for v in full.values()), 2), mi_frac=round(st.mean(v["mi_frac"] for v in full.values()), 3),
    growing=sum(1 for v in full.values() if v["fba_growth"] > 1e-6))
json.dump(out, open(f"{RESULTS}/skeleton_ablation_summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
