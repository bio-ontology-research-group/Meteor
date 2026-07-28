"""Shared verify-and-repair for the v8 hard-biomass MILP (big-M gap-fill leak).

The evw parsimony penalty on zero-evidence biomass-essential skeleton (gap-fill)
reactions makes the MILP set them y~0 while still routing flux through the big-M
gate (v <= ub*y, ub~1000). The reported y>0.5 model then cannot actually grow
(a "degenerate" solution). `verify_and_repair` detects this and repairs by
activating the support of a minimal-flux growing solution over the candidate
mask, preferring to reuse the existing evidence core (adds only minimal bridging
reactions). Clean solutions (y>0.5 already grows) are returned unchanged.

Used by both emit_v8.py and emit_price_v8.py so the two share one implementation.
"""
import numpy as np
from scipy.sparse import csr_matrix as _csr
import pulp as _pulp


def maxbio_active(S, lb, ub, obj_idx, active):
    """Max biomass achievable using ONLY the `active` reaction set (bool mask),
    on the same S/lb/ub the MILP used. Returns 0.0 if the set cannot grow."""
    Sc = _csr(S); Mn, Rn = Sc.shape
    mm = _pulp.LpProblem("maxbio", _pulp.LpMaximize)
    vpp = {j: _pulp.LpVariable("p%d" % j, 0, float(max(0.0, ub[j])) if active[j] else 0.0) for j in range(Rn)}
    vnn = {j: _pulp.LpVariable("q%d" % j, 0, float(max(0.0, -lb[j])) if active[j] else 0.0) for j in range(Rn)}
    for i in range(Mn):
        row = Sc.getrow(i)
        if len(row.indices) == 0:
            continue
        mm += _pulp.lpSum(float(row.data[k]) * (vpp[int(row.indices[k])] - vnn[int(row.indices[k])])
                          for k in range(len(row.indices))) == 0
    mm += (vpp[obj_idx] - vnn[obj_idx])
    mm.solve(_pulp.PULP_CBC_CMD(msg=False))
    return float((vpp[obj_idx].value() or 0) - (vnn[obj_idx].value() or 0))


def grow_support(S, lb, ub, obj_idx, avail, gamma_min, core=None):
    """Growing solution within the `avail` reaction set (bool mask): S v = 0,
    v_biomass >= gamma_min. Minimises flux through NON-`core` reactions so the LP
    reuses the existing evidence core for free and adds only minimal bridging
    reactions. Returns the boolean support (|v| > 1e-9), or None if infeasible."""
    Sc = _csr(S); Mn, Rn = Sc.shape
    mm = _pulp.LpProblem("grow", _pulp.LpMinimize)
    vpp = {j: _pulp.LpVariable("p%d" % j, 0, float(max(0.0, ub[j])) if avail[j] else 0.0) for j in range(Rn)}
    vnn = {j: _pulp.LpVariable("q%d" % j, 0, float(max(0.0, -lb[j])) if avail[j] else 0.0) for j in range(Rn)}
    for i in range(Mn):
        row = Sc.getrow(i)
        if len(row.indices) == 0:
            continue
        mm += _pulp.lpSum(float(row.data[k]) * (vpp[int(row.indices[k])] - vnn[int(row.indices[k])])
                          for k in range(len(row.indices))) == 0
    mm += (vpp[obj_idx] - vnn[obj_idx]) >= gamma_min
    if core is not None:
        mm += _pulp.lpSum((vpp[j] + vnn[j]) for j in range(Rn) if not core[j])
    else:
        mm += _pulp.lpSum(vpp[j] + vnn[j] for j in range(Rn))
    mm.solve(_pulp.PULP_CBC_CMD(msg=False))
    if _pulp.LpStatus[mm.status] != "Optimal":
        return None
    sup = np.zeros(Rn, dtype=bool)
    for j in range(Rn):
        if abs((vpp[j].value() or 0) - (vnn[j].value() or 0)) > 1e-9:
            sup[j] = True
    return sup


def verify_and_repair(S, lb, ub, obj_idx, yv, v_vals, cand, gamma_min=0.1, thresh=0.05):
    """Detect a degenerate MILP solution (y>0.5 model cannot grow) and repair it.

    Returns (yv, n_active, n_repaired, maxbio_core):
      yv          : y vector with repaired reactions set to 1.0 (unchanged if clean)
      n_active    : |repaired active set|
      n_repaired  : number of reactions added by the repair (0 if clean)
      maxbio_core : max biomass over the original y>0.5 set (>=thresh => was clean)
    """
    active = yv > 0.5
    mb_core = maxbio_active(S, lb, ub, obj_idx, active)
    n_repaired = 0
    if mb_core < thresh:
        sup = grow_support(S, lb, ub, obj_idx, cand, gamma_min, core=active)
        if sup is not None:
            n_repaired = int((sup & ~active).sum())
            active = active | sup
            yv = np.where(active, 1.0, yv)
    return yv, int(active.sum()), n_repaired, float(mb_core)
