"""Add a two-stage arm to the deletion-recovery experiment.

METEOR selects the whole reaction set in one MILP over the universal database.
The family it is most likely to be confused with -- ProbAnno, GLOBUS and
probabilistic gap-filling generally -- instead thresholds the annotation into a
draft and then adds reactions to that draft, letting confidence weight only the
additions. Whether METEOR's advantage comes from evidence weighting itself or
from making the choice over the whole set at once is not answerable from the
three arms already reported, because all three are single-stage.

This arm is the two-stage design with the same evidence weighting:
  stage 1  draft = reactions whose EC evidence clears tau = 0.5
  stage 2  minimum-cost bridge to biomass over the same candidate set, with
           non-draft reactions charged the evw cost mu(1-w)^p

Everything else -- candidate set, deleted set D, spurious-reaction accounting --
is shared with the existing arms, so the numbers are directly comparable.
"""
import sys, numpy as np, pulp as _pulp
from scipy.sparse import csr_matrix


def weighted_grow(S, lb, ub, obj_idx, avail, gamma_min, core, cost):
    """Minimum-cost growing solution: reuse `core` freely, charge `cost[j]` per
    unit flux through everything else. cost=1 everywhere reproduces the uniform
    parsimony gap-fill; passing the evw cost gives the confidence-weighted one."""
    Sc = csr_matrix(S); Mn, Rn = Sc.shape
    m = _pulp.LpProblem("wgrow", _pulp.LpMinimize)
    vp = {j: _pulp.LpVariable("p%d" % j, 0, float(max(0.0, ub[j])) if avail[j] else 0.0)
          for j in range(Rn)}
    vn = {j: _pulp.LpVariable("q%d" % j, 0, float(max(0.0, -lb[j])) if avail[j] else 0.0)
          for j in range(Rn)}
    for i in range(Mn):
        row = Sc.getrow(i)
        if len(row.indices) == 0: continue
        m += _pulp.lpSum(float(row.data[k]) * (vp[int(row.indices[k])] - vn[int(row.indices[k])])
                         for k in range(len(row.indices))) == 0
    m += (vp[obj_idx] - vn[obj_idx]) >= gamma_min
    # shift costs positive so the LP cannot pay itself to add reactions
    cpos = np.asarray(cost, float); cpos = cpos - cpos.min() + 1e-3
    m += _pulp.lpSum(float(cpos[j]) * (vp[j] + vn[j]) for j in range(Rn) if not core[j])
    m.solve(_pulp.PULP_CBC_CMD(msg=False))
    if _pulp.LpStatus[m.status] != "Optimal":
        return None
    v = np.array([(vp[j].value() or 0.0) - (vn[j].value() or 0.0) for j in range(Rn)])
    return np.abs(v) > 1e-9


def two_stage(S, lb, ub, obj_idx, cand, w, cost, tau_mask, gamma_min=0.1):
    """Stage 1: the thresholded draft. Stage 2: weighted bridge to biomass.
    Returns the selected reaction index set, or None if stage 2 is infeasible."""
    draft = tau_mask & cand
    sup = weighted_grow(S, lb, ub, obj_idx, cand, gamma_min, draft, cost)
    if sup is None:
        return None
    return set(np.where(draft | sup)[0])
