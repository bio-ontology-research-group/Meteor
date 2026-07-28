#!/usr/bin/env python3
"""Solver-status audit for all v8 MILP solutions (addresses reproducibility req: solver status/gap for every
optimisation). Reads meteor_sol_*.pkl 'status' + 'biomass_flux' + 'solve_sec' across configs; reports status
distribution, growth, and solver timing. Output: meteor_v8/cohorts/solver_status.{tsv,json}."""
import os,glob,json,pickle,numpy as np
from collections import Counter
B="/ibex/scratch/projects/c2014/kexin/funcarve"
ROOT=f"{B}/meteor_v8_evw_p2mu3_run/meteor_out"
OUT=f"{B}/meteor_v8/cohorts"; os.makedirs(OUT,exist_ok=True)
panel108=set(l.split()[0] for l in open(f"{B}/meteor_v7_run/downstream_results/panel108_gram.tsv") if l.strip())
rows=[]
for d in sorted(glob.glob(f"{ROOT}/*_vanilla")):
    cfg=os.path.basename(d)
    stat=Counter(); grow=0; n=0; nact=[]; bf=[]; sec=[]; n_sec=0; miss_status=0; miss_sec=0
    for p in glob.glob(f"{d}/meteor_sol_*.pkl"):
        g=os.path.basename(p)[len("meteor_sol_"):-4]
        if g not in panel108: continue
        try: s=pickle.load(open(p,"rb"))
        except Exception: continue
        n+=1
        st=s.get("status"); stat[str(st)]+=1
        if st is None: miss_status+=1
        b=float(s.get("biomass_flux",0) or 0); bf.append(b); grow+=int(b>1e-6)
        y=s.get("y_vals"); nact.append(int((np.array(y)>0.5).sum()) if y is not None else s.get("n_active",0))
        sv=s.get("solve_sec")
        if sv is not None: sec.append(float(sv)); n_sec+=1
        else: miss_sec+=1
    rows.append(dict(config=cfg,n=n,status=dict(stat),growing=grow,growing_pct=round(100*grow/max(1,n),1),
                     mean_n_active=round(float(np.mean(nact)),1) if nact else None,
                     mean_biomass=round(float(np.mean(bf)),4) if bf else None,missing_status=miss_status,
                     solve_sec_n=n_sec,solve_sec_missing=miss_sec,
                     solve_sec_mean=round(float(np.mean(sec)),2) if sec else None,
                     solve_sec_max=round(float(np.max(sec)),2) if sec else None))
json.dump(rows,open(f"{OUT}/solver_status.json","w"),indent=1)
with open(f"{OUT}/solver_status.tsv","w") as f:
    hdr="config\tn\tgrowing\tgrowing_pct\tmean_n_active\tmean_biomass\tstatus_distribution\tmissing_status\t"
    hdr+="solve_sec_n\tsolve_sec_missing\tsolve_sec_mean\tsolve_sec_max"
    f.write(hdr+"\n")
    for r in rows:
        f.write(f"{r['config']}\t{r['n']}\t{r['growing']}\t{r['growing_pct']}\t"
                f"{r['mean_n_active']}\t{r['mean_biomass']}\t{r['status']}\t{r['missing_status']}\t"
                f"{r['solve_sec_n']}\t{r['solve_sec_missing']}\t{r['solve_sec_mean']}\t{r['solve_sec_max']}\n")
print("=== v8 solver-status audit (panel108) ===")
for r in rows:
    parts=[f"  {r['config']:16s} n={r['n']:3d} grow={r['growing']}/{r['n']} ({r['growing_pct']}%) n_active={r['mean_n_active']}"]
    if r["solve_sec_n"]:
        parts.append(f"solve_sec: n={r['solve_sec_n']} mean={r['solve_sec_mean']}s max={r['solve_sec_max']}s")
    if r["solve_sec_missing"]:
        parts.append(f"missing={r['solve_sec_missing']}")
    print("  ".join(parts))
print("-> meteor_v8/cohorts/solver_status.{json,tsv}")
