"""v7 hard-biomass MILP + capability soft-reward (single-pass, no two-stage).

Differences vs v6utils.build_milp_logodds (see review notes):
  * biomass is a HARD constraint:  gamma_min <= v_biomass <= gamma_max.
    v6 used a soft slack + BIG penalty, which silently accepted
    metabolically-infeasible genomes (slack "absorbs unmet biomass"). [P0-1]
  * `biomass_feasible_skeleton` computes the reactions that must be available
    for biomass to be reachable, so variable-reduction does not zero them out
    and make the hard constraint infeasible. v6 never populated this. [P0-2]
  * optional capability SOFT rewards: for each gated-in capability add
    -reward * sum(net flux over its production reactions) to the objective.
    No hard min-flux constraint -> cannot cause infeasibility; we measure the
    achieved flux instead (point-6 design: gate by predictor prob, reward soft).

Because biomass is hard, a single-pass solve already guarantees biomass while
letting the capability reward act -> the v6 two-stage hierarchical solve is
unnecessary.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from pulp import (LpProblem, LpVariable, lpSum, LpMinimize, LpMaximize,
                  LpStatus, PULP_CBC_CMD, value)


def capability_max_flux(S, lb, ub, obj_idx, cap_idxs, gamma_min, solver=None):
    """LP feasibility probe: max sum(net flux over cap_idxs) s.t. Sv=0,
    lb<=v<=ub, v_biomass >= gamma_min.

    Replaces the futile-flux soft reward: instead of pushing capability flux
    inside the shared MILP objective (which just maxed it out), we ask a clean
    binary question -- given media + biomass floor, can this capability carry
    flux at all? Returns (max_flux, feasible_status_optimal).
    """
    if not cap_idxs:
        return 0.0, False
    M, R = S.shape
    S = csr_matrix(S)
    m = LpProblem("cap_probe", LpMaximize)
    v = {j: LpVariable(f"v_{j}", lowBound=float(lb[j]), upBound=float(ub[j]))
         for j in range(R)}
    for i in range(M):
        row = S.getrow(i)
        if len(row.indices) == 0:
            continue
        m += lpSum(float(row.data[k]) * v[int(row.indices[k])]
                   for k in range(len(row.indices))) == 0
    m += v[obj_idx] >= gamma_min
    m += lpSum(v[j] for j in cap_idxs)
    m.solve(solver or PULP_CBC_CMD(msg=False))
    val = value(m.objective)
    return (float(val) if val is not None else 0.0), (LpStatus[m.status] == "Optimal")


def biomass_feasible_skeleton(S, lb, ub, obj_idx, gamma_min, solver=None, tol=1e-6):
    """LP: maximise biomass s.t. Sv=0, lb<=v<=ub.

    Returns (feasible: bool, skeleton: set[int], max_biomass: float).
    `skeleton` = reactions carrying |flux| > tol in the biomass-maximising
    solution; protecting them in candidate_mask keeps the hard biomass
    constraint feasible under variable reduction.
    """
    M, R = S.shape
    S = csr_matrix(S)
    m = LpProblem("skeleton_fba", LpMaximize)
    v = {j: LpVariable(f"v_{j}", lowBound=float(lb[j]), upBound=float(ub[j]))
         for j in range(R)}
    for i in range(M):
        row = S.getrow(i)
        if len(row.indices) == 0:
            continue
        m += lpSum(float(row.data[k]) * v[int(row.indices[k])]
                   for k in range(len(row.indices))) == 0
    m += v[obj_idx]
    m.solve(solver or PULP_CBC_CMD(msg=False))
    bm = v[obj_idx].value() or 0.0
    skel = {j for j in range(R) if abs(v[j].value() or 0.0) > tol}
    return (bm >= gamma_min), skel, float(bm)


def build_milp_hard(S, lb, ub, c, obj_idx, excludes, media_rxns, candidate_mask,
                    gamma_min, gamma_max, lam, caps=None, reward=0.1):
    """Build the v7 single-pass hard-biomass MILP.

    caps : list of dicts with keys {"name","rxn_idxs"} (already gated in).
    Returns (model, y, v_pos, v_neg, cap_rxns) where cap_rxns maps name->idxs.
    """
    M, R = S.shape
    S = csr_matrix(S)
    model = LpProblem("meteor_v7_hard", LpMinimize)

    y = {}
    for j in range(R):
        if j in excludes:
            y[j] = LpVariable(f"y_{j}", lowBound=0, upBound=0)
        elif j in media_rxns:
            y[j] = LpVariable(f"y_{j}", lowBound=1, upBound=1)
        elif candidate_mask[j]:
            y[j] = LpVariable(f"y_{j}", cat="Binary")
        else:
            y[j] = LpVariable(f"y_{j}", lowBound=0, upBound=0)

    v_pos = {j: LpVariable(f"vp_{j}", lowBound=0, upBound=float(max(0.0, ub[j])))
             for j in range(R)}
    v_neg = {j: LpVariable(f"vn_{j}", lowBound=0, upBound=float(max(0.0, -lb[j])))
             for j in range(R)}

    for i in range(M):
        row = S.getrow(i)
        if len(row.indices) == 0:
            continue
        model += lpSum(float(row.data[k]) *
                       (v_pos[int(row.indices[k])] - v_neg[int(row.indices[k])])
                       for k in range(len(row.indices))) == 0, f"mb_{i}"

    for j in range(R):
        model += v_pos[j] <= float(max(0.0, ub[j])) * y[j]
        if lb[j] < 0:
            model += v_neg[j] <= float(-lb[j]) * y[j]
        else:
            model += v_neg[j] <= 0

    # HARD biomass (no slack)
    model += (v_pos[obj_idx] - v_neg[obj_idx]) >= gamma_min, "biomass_hard_lb"
    model += (v_pos[obj_idx] - v_neg[obj_idx]) <= gamma_max, "biomass_hard_ub"

    obj = lpSum(float(c[j]) * y[j]
                for j in range(R) if candidate_mask[j] and j not in excludes)
    obj += lpSum(lam * (v_pos[j] + v_neg[j]) for j in range(R))

    cap_rxns = {}
    for cap in (caps or []):
        idxs = [j for j in cap["rxn_idxs"] if 0 <= j < R]
        cap_rxns[cap["name"]] = idxs
        if idxs:
            obj -= reward * lpSum(v_pos[j] - v_neg[j] for j in idxs)

    model += obj
    return model, y, v_pos, v_neg, cap_rxns
