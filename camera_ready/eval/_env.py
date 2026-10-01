"""Shared paths for the PSB revision scripts. Everything under R is ours;
everything else is read-only input from the original runs."""
import os, sys
R = "/ibex/scratch/projects/c2014/kexin/funcarve/psb_revision"
os.environ.setdefault("METEOR_DATA", f"{R}/code_snapshot/data")
for p in (f"{R}/code_snapshot/src", f"{R}/code_snapshot/eval"):
    if p not in sys.path: sys.path.insert(0, p)
RUNS = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8_evw_p2mu3_run/meteor_out"   # read-only
ORIG_RESULTS = "/ibex/scratch/projects/c2014/kexin/funcarve/meteor_v8/results"            # read-only
GEMDIR = f"{R}/code_snapshot/data/external/curated_gems"
PANEL = f"{R}/code_snapshot/panel108_gram.tsv"
RESULTS = f"{R}/results"
MILP_FLAGS = dict(mu=3.0, pexp=2.0, eps=0.0, gmin=0.1, wmin=0.01, gmax=2.5, lam=1e-4)
def panel_rows():
    return [tuple(l.rstrip("\n").split("\t")[:2]) for l in open(PANEL) if l.strip()]
