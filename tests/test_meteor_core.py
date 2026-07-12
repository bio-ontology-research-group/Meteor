"""Minimal smoke tests for METEOR core algorithm.

Run:
    pytest tests/test_meteor_core.py -v
or just:
    python tests/test_meteor_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from meteor.core import (
    aggregate_confidence,
    logodds_cost,
    parsimonious_gapfill,
    posterior_calibrated,
    solve_milp_hierarchical,
)
from meteor import CapabilitySpec, MILPParams


def test_aggregate_noisy_or_basic():
    """Two proteins predict 0.9 and 0.6 for an EC mapped to one reaction → ~0.96."""
    pred = pd.DataFrame(
        [[0.9, 0.1], [0.6, 0.2]],
        index=["p1", "p2"],
        columns=["1.1.1.1", "2.7.10.2"],
    )
    rxn_ec_mask = np.array([[1, 0], [0, 1]], dtype=np.float32)
    w = aggregate_confidence(pred, rxn_ec_mask, method="noisy_or")
    assert w.shape == (2,)
    assert 0.95 < w[0] < 0.97, f"expected ~0.96, got {w[0]}"
    # Second reaction: 1 - (1-0.1)*(1-0.2) = 1 - 0.72 = 0.28
    assert 0.27 < w[1] < 0.29


def test_aggregate_max():
    pred = pd.DataFrame(
        [[0.9, 0.1], [0.6, 0.2]],
        index=["p1", "p2"],
        columns=["1.1.1.1", "2.7.10.2"],
    )
    rxn_ec_mask = np.array([[1, 0], [0, 1]], dtype=np.float32)
    w = aggregate_confidence(pred, rxn_ec_mask, method="max")
    assert abs(w[0] - 0.9) < 1e-6
    assert abs(w[1] - 0.2) < 1e-6


def test_aggregate_empty_reaction():
    """A reaction with no associated EC stays at w=0."""
    pred = pd.DataFrame(
        [[0.9, 0.1]],
        index=["p1"],
        columns=["1.1.1.1", "2.7.10.2"],
    )
    rxn_ec_mask = np.array([[0, 0]], dtype=np.float32)
    w = aggregate_confidence(pred, rxn_ec_mask, method="noisy_or")
    assert abs(w[0] - 0.0) < 1e-6


def test_logodds_cost():
    w = np.array([0.99, 0.5, 0.01])
    c = logodds_cost(w)
    assert c[0] < -3  # very negative (favored)
    assert abs(c[1]) < 1e-3  # ~0
    assert c[2] > 3  # very positive (penalized)


def test_posterior_active_boost():
    pred = pd.DataFrame(
        [[0.45, 0.99, 0.01]],
        index=["p1"],
        columns=["A", "B", "C"],
    )
    # Pin the historical α=0.4 explicitly to test the boost formula.
    refined = posterior_calibrated(pred, active_ecs={"A", "B", "C"},
                                   muted_ecs=set(), beta=1.0, alpha_scale=0.4)
    # Borderline 0.45 → 0.45 + 0.4*0.45*0.55 = 0.549
    assert 0.548 < refined.iloc[0]["A"] < 0.551
    # Extreme 0.99 → 0.99 + 0.4*0.99*0.01 = 0.9940
    assert 0.993 < refined.iloc[0]["B"] < 0.995
    # Extreme 0.01 → 0.01 + 0.4*0.01*0.99 = 0.014
    assert 0.013 < refined.iloc[0]["C"] < 0.015


def test_posterior_muted_dampen():
    pred = pd.DataFrame(
        [[0.45, 0.99]],
        index=["p1"],
        columns=["A", "B"],
    )
    refined = posterior_calibrated(pred, active_ecs=set(),
                                   muted_ecs={"A", "B"}, beta=1.0,
                                   alpha_scale=0.4)
    # 0.45 - 0.4*0.45*0.55 = 0.351
    assert 0.350 < refined.iloc[0]["A"] < 0.352
    # 0.99 - 0.4*0.99*0.01 = 0.9860
    assert 0.985 < refined.iloc[0]["B"] < 0.987


def test_posterior_default_alpha_is_smaller():
    """Default alpha_scale (0.25) moves borderline scores less than the
    legacy 0.4 — required to preserve precision on phenotype eval.
    """
    pred = pd.DataFrame([[0.5]], index=["p"], columns=["A"])
    legacy = posterior_calibrated(pred, active_ecs={"A"}, muted_ecs=set(),
                                  beta=1.0, alpha_scale=0.4)
    default = posterior_calibrated(pred, active_ecs={"A"}, muted_ecs=set(),
                                   beta=1.0)
    legacy_delta = legacy.iloc[0]["A"] - 0.5
    default_delta = default.iloc[0]["A"] - 0.5
    assert default_delta > 0
    assert default_delta < legacy_delta


def test_posterior_zero_scale_is_identity():
    pred = pd.DataFrame([[0.3, 0.7]], index=["p"], columns=["A", "B"])
    refined = posterior_calibrated(pred, active_ecs={"A"}, muted_ecs={"B"},
                                   beta=1.0, alpha_scale=0.0)
    assert abs(refined.iloc[0]["A"] - 0.3) < 1e-9
    assert abs(refined.iloc[0]["B"] - 0.7) < 1e-9


def test_posterior_ranking_preserved_on_extremes():
    """If all ECs of a protein are extreme, ranking is preserved."""
    pred = pd.DataFrame(
        [[0.01, 0.99, 0.5]],
        index=["p1"],
        columns=["X", "Y", "Z"],
    )
    refined = posterior_calibrated(pred, active_ecs={"X", "Y", "Z"},
                                   muted_ecs=set(), beta=1.0, alpha_scale=0.4)
    assert refined.iloc[0]["Y"] > refined.iloc[0]["Z"] > refined.iloc[0]["X"]


def test_parsimonious_gapfill_resolves_orphan():
    """Active set has metabolite M consumed but not produced; gap-fill adds the
    cheapest universal reaction that produces M."""
    # 3 metabolites (A, B, M), 4 reactions:
    #   r0: A → B          (active)
    #   r1: B → M          (active — consumes nothing from outside, produces M)
    #   r2: M → ∅          (active sink — consumes M)
    #   r3: ∅ → A          (universal candidate, cheap)
    # After the MILP "actives" r0,r1,r2: A is orphan (no producer), M is fine,
    # B is fine. Gap-fill should add r3 to give A a producer.
    S = np.array([
        # r0   r1   r2   r3
        [-1.0, 0.0, 0.0, 1.0],  # A
        [1.0, -1.0, 0.0, 0.0],  # B
        [0.0, 1.0, -1.0, 0.0],  # M
    ])
    lb = np.array([0.0, 0.0, 0.0, 0.0])
    ub = np.array([1000.0] * 4)
    cost = np.array([0.0, 0.0, 0.0, -2.0])  # r3 most favoured
    y_star = np.array([1, 1, 1, 0])
    y_filled, info = parsimonious_gapfill(y_star, S, cost, lb, ub, max_additions=5)
    assert y_filled[3] == 1, "r3 should be added to give A a producer"
    assert info["n_added"] >= 1
    assert info["dead_ends_after"] < info["dead_ends_before"]


def test_hierarchical_preserves_biomass():
    """Hierarchical solve guarantees biomass_stage2 >= biomass_fraction * biomass_stage1.

    Toy model: 3 reactions chained A → B → C(biomass), plus a capability rxn
    D that consumes nothing useful. Single-obj selects A,B,C; multi-obj with
    naive weighting could prefer the capability and drop biomass, but
    hierarchical enforces the floor.
    """
    # A → B → biomass (no other inputs/outputs needed for this toy)
    # rxn idx: 0 = produce A, 1 = A→B, 2 = B→biomass, 3 = capability
    S = np.array([
        # r0  r1   r2  r3
        [1.0, -1.0, 0.0, 0.0],   # A
        [0.0, 1.0, -1.0, 0.0],   # B
        [0.0, 0.0, 1.0, 0.0],    # biomass-met
        [0.0, 0.0, 0.0, 1.0],    # capability output (no biomass connection)
    ])
    lb = np.zeros(4)
    ub = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    cost = np.array([0.0, 0.0, 0.0, -10.0])  # capability looks "cheap"
    biomass_idx = 2
    cap = CapabilitySpec(name="cap", indicator_ecs=["1.1.1.1"],
                         production_rxn_idxs=[3], min_flux=0.01, reward=100.0)
    params = MILPParams(gamma_min=0.5, lam=0.0, time_limit_s=30, gap=0.01,
                        solver="cbc", capabilities=[cap])
    hier = solve_milp_hierarchical(S, lb, ub, cost, biomass_idx, params,
                                    biomass_fraction=0.95)
    # Both stages must achieve at least the base gamma_min biomass
    assert hier["biomass_stage1"] >= 0.5 - 1e-6
    assert hier["biomass_stage2"] >= hier["biomass_floor"] - 1e-6
    # Capability reaction must be active in stage 2
    assert hier["y_stage2"][3] == 1


def test_parsimonious_gapfill_respects_budget():
    """If max_additions=0 the active set is returned unchanged."""
    S = np.eye(3)
    lb = np.zeros(3)
    ub = np.ones(3) * 1000
    cost = np.zeros(3)
    y_star = np.array([1, 0, 0])
    y_filled, info = parsimonious_gapfill(y_star, S, cost, lb, ub, max_additions=0)
    assert (y_filled == y_star).all()
    assert info["n_added"] == 0


if __name__ == "__main__":
    test_aggregate_noisy_or_basic()
    test_aggregate_max()
    test_aggregate_empty_reaction()
    test_logodds_cost()
    test_posterior_active_boost()
    test_posterior_muted_dampen()
    test_posterior_ranking_preserved_on_extremes()
    test_parsimonious_gapfill_resolves_orphan()
    test_parsimonious_gapfill_respects_budget()
    test_hierarchical_preserves_biomass()
    print("All tests passed.")
