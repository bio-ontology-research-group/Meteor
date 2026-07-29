r"""METEOR v8 — evidence-weighted parsimony (evw) MILP.

Extends the v7 hard-biomass MILP (``milp_hard.build_milp_hard``) with a
cost vector ``c`` that the caller pre-loads with the evw penalty.

The MILP objective includes:

  * **Evidence-weighted parsimony** (main contribution, eq.~1 of paper).
    The caller (``emit_v8.py``) computes the per-reaction y-cost as
    :math:`c_j = -\log\frac{w_j+\epsilon}{1-w_j+\epsilon} + \mu(1-w_j)^p`, where
    ``w_j`` is the noisy-OR confidence of reaction *j* and
    ``epsilon = 1e-6`` is the log-odds smoothing (``v6utils.EPS_SMOOTH``).
    Reactions with weak evidence
    (low ``w``) incur a large penalty; high-confidence reactions incur
    almost none.  This is the paper's central mechanism.

  * **Uniform reaction-count penalty** ``mu * sum_j y_j`` (optional,
    default 0).  When ``--penalty evw`` this is set to 0 because the evw
    penalty is already folded into ``c``.  When ``--penalty uniform``
    this provides a flat per-reaction parsimony cost.

  * **Forced-flux floor** ``eps`` (optional, default 0).  If > 0 every
    active non-exchange reaction must carry |v| >= eps.  Not used in the
    paper (eps = 0).  NOTE: the manuscript calls this parameter *delta*,
    reserving *epsilon* for the log-odds smoothing inside compute_costs.
    The two are unrelated; do not conflate them.

With ``c`` = log-odds-only and ``mu = 0, eps = 0`` this reduces exactly
to the v7 MILP.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from pulp import LpProblem, LpVariable, lpSum, LpMinimize


def build_milp_v8(S, lb, ub, c, obj_idx, excludes, media_rxns, candidate_mask,
                  gamma_min, gamma_max, lam, mu=0.0, eps=0.0,
                  caps=None, reward=0.1):
    """Build the v8 hard-biomass MILP with reaction-count penalty + forced flux.

    Same interface as ``build_milp_hard`` plus:
        mu   : per-active-reaction penalty added to the objective (>=0).
        eps  : forced-flux floor; if >0, every active non-exchange reaction
               must carry |v| >= eps.
    Returns (model, y, v_pos, v_neg, cap_rxns).
    """
    M, R = S.shape
    S = csr_matrix(S)
    model = LpProblem("meteor_v8_evw", LpMinimize)

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

    # steady-state mass balance
    for i in range(M):
        row = S.getrow(i)
        if len(row.indices) == 0:
            continue
        model += lpSum(float(row.data[k]) *
                       (v_pos[int(row.indices[k])] - v_neg[int(row.indices[k])])
                       for k in range(len(row.indices))) == 0, f"mb_{i}"

    # flux-y gating (+ optional forced-flux floor)
    for j in range(R):
        model += v_pos[j] <= float(max(0.0, ub[j])) * y[j]
        if lb[j] < 0:
            model += v_neg[j] <= float(-lb[j]) * y[j]
        else:
            model += v_neg[j] <= 0
        if eps > 0 and candidate_mask[j] and j not in excludes and j != obj_idx:
            # exchange/demand/sink reactions are exempt (they legitimately
            # sit at zero flux under a defined medium); the biomass reaction
            # is already floored by gamma_min.
            model += v_pos[j] + v_neg[j] >= eps * y[j], f"ff_{j}"

    # HARD biomass (no slack)
    model += (v_pos[obj_idx] - v_neg[obj_idx]) >= gamma_min, "biomass_hard_lb"
    model += (v_pos[obj_idx] - v_neg[obj_idx]) <= gamma_max, "biomass_hard_ub"

    # objective: evidence cost + flux parsimony + reaction-count penalty
    obj = lpSum(float(c[j]) * y[j]
                for j in range(R) if candidate_mask[j] and j not in excludes)
    obj += lpSum(lam * (v_pos[j] + v_neg[j]) for j in range(R))
    if mu > 0:
        obj += mu * lpSum(y[j] for j in range(R)
                          if candidate_mask[j] and j not in excludes)

    cap_rxns = {}
    for cap in (caps or []):
        idxs = [j for j in cap["rxn_idxs"] if 0 <= j < R]
        cap_rxns[cap["name"]] = idxs
        if idxs:
            obj -= reward * lpSum(v_pos[j] - v_neg[j] for j in idxs)

    model += obj
    return model, y, v_pos, v_neg, cap_rxns
