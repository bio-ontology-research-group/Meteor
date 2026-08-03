"""Single source of truth for resolving a baseline prediction pkl path on the
109-GCF panel. Every downstream script (eval/build) MUST import
`resolve_baseline_pkl` from here instead of re-deriving the reinfer-fallback
logic inline -- that duplication is what caused a real bug (2026-07-05):
several eval scripts read `baseline_preds/{b}_{v}_genome_collection/` directly
and silently used stale/broken pkls for genomes that had since been fixed by
re-inference into `meteor_v7_run/reinfer/{b}_{v}/`.

Canonical fallback order (matches build_grow_memote.py / emit_meteor_out.py,
the scripts that were already correct): reinfer/ (fixed) first, then the
original genome_collection/ location. Neither directory's files are ever
moved, merged, or overwritten by this module -- it only decides which path
to READ.
"""
import os
import glob

try:
    from meteor_v8.utils import run_path
except ImportError:                        # standalone use, outside the package
    def run_path(*parts):
        root = os.environ.get("METEOR_RUNS",
                              "/ibex/scratch/projects/c2014/kexin/funcarve")
        return os.path.join(root, *parts)

PA = run_path("paperA_2026")
REINFER_ROOT = run_path("meteor_v7_run", "reinfer")
BACDIVE_ROOT = run_path("dpec2_result", "result_bacdive")

BASELINE_SUFFIX = {"clean": "CLEAN_confidence", "dpz": "DPZ", "enzbert": "enzbert"}


def resolve_baseline_pkl(baseline: str, variant: str, gca: str, suffix: str = None) -> str:
    """Return the correct pkl path for (baseline, variant, gca): the
    reinfer-fixed version if one exists, else the original genome_collection
    version, else (dpz/vanilla only) the BacDive-scale DeepECv2 inference
    (unversioned GCA, gram tag embedded in filename -- 2026-07-18, full
    6,980-genome BacDive run). Returns "" if none exist.
    """
    suf = suffix or BASELINE_SUFFIX[baseline]
    reinfer_path = f"{REINFER_ROOT}/{baseline}_{variant}/{gca}_{suf}.pkl"
    if os.path.exists(reinfer_path):
        return reinfer_path
    orig_path = f"{PA}/baseline_preds/{baseline}_{variant}_genome_collection/{gca}_{suf}.pkl"
    if os.path.exists(orig_path):
        return orig_path
    if baseline == "dpz" and variant == "vanilla":
        base_gca = gca.split(".")[0]
        hits = glob.glob(f"{BACDIVE_ROOT}/{base_gca}_*_DeepECv2_t5.pkl")
        if hits:
            return hits[0]
    return ""


def list_reinfer_overrides(baseline: str, variant: str) -> set:
    """Set of GCA accessions that have a reinfer-fixed pkl for this (baseline,
    variant) -- i.e. the genomes where the original genome_collection file was
    broken/incomplete and superseded. Empty set if no reinfer dir exists."""
    suf = BASELINE_SUFFIX[baseline]
    d = f"{REINFER_ROOT}/{baseline}_{variant}"
    if not os.path.isdir(d):
        return set()
    return {f[: -len(f"_{suf}.pkl")] for f in os.listdir(d) if f.endswith(f"_{suf}.pkl")}
