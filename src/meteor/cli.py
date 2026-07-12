"""CLI entry point: `python -m meteor` or `meteor` after install.

Mirrors the SLURM-aware research wrapper at _v6main_original.py but with a
cleaner argparse + portable I/O. Reads a baseline's per-protein EC probability
DataFrame (pickle), runs METEOR θ-free, writes the refined DataFrame.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .core import MILPParams, run_meteor


def main():
    p = argparse.ArgumentParser(
        prog="meteor",
        description="METEOR θ-free: model-agnostic MILP post-processing for "
                    "bacterial EC predictions",
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Per-protein EC probability matrix (pkl/parquet)")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Output directory (created if missing)")
    p.add_argument("--name", required=True, type=str,
                   help="Output basename (e.g., organism tax_562)")
    p.add_argument("--data_dir", type=Path, default=Path("data"),
                   help="Directory containing universal model + EC mappings")
    p.add_argument("--gram", choices=["positive", "negative"], default="negative",
                   help="Bacterial Gram type (selects biomass reaction)")
    p.add_argument("--gamma_min", type=float, default=0.1,
                   help="Minimum biomass flux required")
    p.add_argument("--gamma_max", type=float, default=2.5,
                   help="Maximum biomass flux allowed (growth-rate sanity cap)")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Posterior calibration temperature")
    p.add_argument("--time_limit", type=int, default=1800,
                   help="MILP solver time limit (seconds)")
    p.add_argument("--gap", type=float, default=0.05,
                   help="MILP relative gap tolerance")
    p.add_argument("--threads", type=int, default=8,
                   help="CBC solver thread count")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load input baseline prediction ----
    logging.info("Loading input: %s", args.input)
    pred_df = _load_pred(args.input)
    logging.info("  shape: %s, range: [%.4f, %.4f]",
                 pred_df.shape, pred_df.values.min(), pred_df.values.max())

    # ---- Load universal data ----
    data_dir = args.data_dir
    logging.info("Loading universal model from %s", data_dir)
    universal_S, lb, ub, allrxns, allmet, biomass_idx = _load_universal(
        data_dir, gram=args.gram)
    rxn_to_ec = _load_rxn_to_ec(data_dir)
    ancestor_ecs = _load_ec(data_dir / "all_ancestors.txt")

    # Normalize input column names (strip "EC:" prefix)
    pred_df.columns = [c.replace("EC:", "").strip() for c in pred_df.columns]
    pred_df = pred_df.reindex(columns=ancestor_ecs, fill_value=0.0)

    # Build reaction-to-EC mask
    rxn_ec_mask = _build_rxn_ec_mask(allrxns, rxn_to_ec, ancestor_ecs)

    # ---- Run METEOR ----
    params = MILPParams(
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        gap=args.gap,
        time_limit_s=args.time_limit,
        threads=args.threads,
    )

    refined, meta = run_meteor(
        pred_df=pred_df,
        universal_S=universal_S,
        lb=lb, ub=ub,
        rxn_ec_mask=rxn_ec_mask,
        biomass_idx=biomass_idx,
        rxn_to_ec=rxn_to_ec,
        allrxns=allrxns,
        params=params,
        beta=args.beta,
    )

    # ---- Save ----
    out_pkl = args.output_dir / f"meteor_df_{args.name}.pkl"
    refined.to_pickle(out_pkl)
    logging.info("Saved refined predictions: %s", out_pkl)

    out_meta = args.output_dir / f"meteor_meta_{args.name}.json"
    import json
    out_meta.write_text(json.dumps(meta, indent=2))
    logging.info("Saved metadata: %s", out_meta)


# ---------------------------------------------------------------------------
# Internal data loaders (keep close to core.py to avoid cross-module coupling)
# ---------------------------------------------------------------------------

def _load_pred(path: Path) -> pd.DataFrame:
    if path.suffix == ".pkl" or path.suffix == ".pickle":
        return pd.read_pickle(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _load_universal(data_dir: Path, gram: str):
    """Load universal metabolic model and tight bounds."""
    with open(data_dir / "universal.pickle", "rb") as f:
        universal = pickle.load(f)
    allrxns = [r.id for r in universal.reactions]
    allmet = [m.id for m in universal.metabolites]

    # Build S matrix from cobra model
    from cobra.util import create_stoichiometric_matrix
    S = create_stoichiometric_matrix(universal)

    # Load precomputed tight bounds (gram-specific)
    bounds_file = data_dir / f"tight_bounds_v6_{gram[:3]}.pkl"
    with open(bounds_file, "rb") as f:
        bounds = pickle.load(f)
    lb = bounds["lb_tight"]
    ub = bounds["ub_tight"]

    biomass_id = f"biomass_Gm{'Pos' if gram == 'positive' else 'Neg'}"
    biomass_idx = allrxns.index(biomass_id)
    return S, lb, ub, allrxns, allmet, biomass_idx


def _load_rxn_to_ec(data_dir: Path) -> dict[str, list[str]]:
    with open(data_dir / "seedr2ec.pkl", "rb") as f:
        d = pickle.load(f)
    return {k: v for k, v in d.items() if v is not None}


def _load_ec(path: Path) -> list[str]:
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def _build_rxn_ec_mask(allrxns, rxn_to_ec, allecs):
    n_rxns = len(allrxns)
    n_ecs = len(allecs)
    ec_to_idx = {e: i for i, e in enumerate(allecs)}
    mask = np.zeros((n_rxns, n_ecs), dtype=np.float32)
    for i, rxn in enumerate(allrxns):
        rname = rxn[:-2] if rxn.endswith(("_c", "_e", "_p")) else rxn
        for ec in rxn_to_ec.get(rname, []):
            if ec in ec_to_idx:
                mask[i, ec_to_idx[ec]] = 1.0
    return mask


if __name__ == "__main__":
    main()
