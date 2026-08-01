"""Shared helpers for the METEOR v8 pipeline.

Vendored from the research tree (``cobra/v6/src/v6utils.py``) so that the
released package runs standalone: the evaluation scripts previously reached
outside the repository for these functions, which made the deposit
non-executable.  The bodies are copied verbatim -- the published numbers came
from this code and rewriting it would risk changing them.  Only the data-file
lookup differs, and only so that callers no longer have to ``os.chdir`` into
the research tree; see ``data_path``.

Functions the v6 module also carried but nothing here uses -- the v6-era MILP
builders superseded by :mod:`meteor_v8.milp_v8`, the GPR and annotation
helpers, ``posterior_clip``, ``solve_lp_relaxed`` -- are deliberately absent.
"""
from __future__ import annotations

from collections import defaultdict
import os
import re
import pickle
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix

from cobra.util import create_stoichiometric_matrix

import pulp
from pulp import (
    LpProblem, LpVariable, lpSum, LpMinimize,
    LpStatus, LpStatusOptimal, value,
    PULP_CBC_CMD,
)

# ---------- numeric defaults ----------
EPS_SMOOTH = 1e-6          # log-odds smoothing; the manuscript's epsilon
DEFAULT_GAMMA_MIN = 0.1    # biologically modest lower bound (~ doubling time 7h)
DEFAULT_GAMMA_MAX = 2.5    # biologically realistic upper bound (E. coli max ~2 h^-1)
DEFAULT_LAMBDA = 1e-4      # parsimony weight on |v|
DEFAULT_BETA = 1.0         # posterior calibration temperature
WMIN_KEEP = 0.01           # reactions with w < WMIN_KEEP can be fix-zeroed
                           # (overridden if reaction is essential / media)

# The manuscript's *delta*, the forced-flux floor, is a parameter of
# ``milp_v8.build_milp_v8`` (its ``eps`` keyword) and is unrelated to
# EPS_SMOOTH above.  Both are zero-or-tiny and easy to confuse; they are not
# the same quantity.

# ---------- data files ----------
_DATA_ENV = "METEOR_DATA"


def data_path(name):
    """Locate a bundled data file (universal model, EC maps, media, ...).

    Searched in order: ``$METEOR_DATA``, a ``data/`` directory beside the
    repository root, then ``data/`` under the current directory.  The last
    entry reproduces the original behaviour, in which callers ``chdir``-ed
    into the research tree before importing.  Returns the first hit, or the
    ``$METEOR_DATA`` (or repo-relative) candidate when the file does not yet
    exist, so that callers writing a cache still get a sensible location.
    """
    roots = []
    env = os.environ.get(_DATA_ENV)
    if env:
        roots.append(env)
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              os.pardir, os.pardir, "data"))
    roots.append("data")
    for root in roots:
        cand = os.path.join(root, name)
        if os.path.exists(cand):
            return cand
    return os.path.join(roots[0], name)


def external_path(name):
    """Locate a third-party input under ``data/external``.

    MetaNetX cross-references, the Reconstructor EC mapping and the curated
    BiGG models are not ours; they live together under ``data/external`` with
    their provenance recorded in the README there. Resolution follows
    :func:`data_path`, so ``$METEOR_DATA`` overrides the bundled copy.
    """
    return data_path(os.path.join("external", name))


def data_dir():
    """Directory the bundled data files resolve to.

    Several callers pass a directory rather than a file (``load_refmapping``
    among them), so resolve a file that is always present and hand back its
    parent.
    """
    return os.path.dirname(os.path.abspath(data_path("seedr2ec.pkl")))


_ENZYME_OBSOLETE_PATH = data_path("enzymeobsolete.pkl")
_enzyme_obsolete_cache = None
def load_ec(file):
    with open(file, 'r') as f:
        return f.read().splitlines()


def load_refmapping(datap):
    with open(f'{datap}/seedr2ec.pkl', 'rb') as f:
        seedr2ec = pickle.load(f)
    with open(f'{datap}/seedec2r.pkl', 'rb') as f:
        seedec2r = pickle.load(f)
    return seedr2ec, seedec2r


def load_universal(with_transport=False):
    fname = data_path('universal_model_with_transport_rxns.pickle'
                      if with_transport else 'universal.pickle')
    universal = pickle.load(open(fname, 'rb'))
    allrxns = [r.id for r in universal.reactions]
    allmet = [m.id for m in universal.metabolites]
    return universal, allrxns, allmet


def get_media(mediainput, mediaf=None):
    if mediaf is None:
        mediaf = data_path('medium.pkl')
    with open(mediaf, 'rb') as f:
        mediainfo = pickle.load(f)
    media = mediainput[0] if isinstance(mediainput, (list, tuple)) else mediainput
    if media in mediainfo:
        return [x + "_e" for x in mediainfo[media]]
    return [media] if isinstance(media, str) else list(media)


def _load_enzyme_obsolete():
    global _enzyme_obsolete_cache
    if _enzyme_obsolete_cache is None:
        with open(_ENZYME_OBSOLETE_PATH, "rb") as f:
            _enzyme_obsolete_cache = pickle.load(f)
    return _enzyme_obsolete_cache


def _resolve_current_ec(ec, obsolete_map, max_hops=5):
    current = ec
    seen = set()
    hops = 0
    while current in obsolete_map and current not in seen and hops < max_hops:
        seen.add(current)
        current = obsolete_map[current]
        hops += 1
    return current


def extract_pred(pred_fpath, ancestorsec):
    df = pd.read_pickle(pred_fpath)
    if df.index[0].count('.') == 3:
        df = df.T
    if 'EC:' in df.columns[0]:
        df.columns = [c.split('EC:')[-1] if 'EC:' in c else c for c in df.columns]
    # Remap obsolete/renumbered EC codes to their current equivalent before
    # reindexing, so scores for baseline-predicted ECs that were merely
    # renumbered aren't silently dropped just for being absent from
    # ancestorsec under their old code (2026-07-19 fix).
    anc_set = set(ancestorsec)
    obsolete_map = _load_enzyme_obsolete()
    rename = {}
    for c in df.columns:
        if c not in anc_set and c in obsolete_map:
            current = _resolve_current_ec(c, obsolete_map)
            if current in anc_set and current != c:
                rename[c] = current
    for old, new in rename.items():
        if new in df.columns:
            df[new] = df[[old, new]].max(axis=1)
            df = df.drop(columns=[old])
        else:
            df = df.rename(columns={old: new})
    df = df.reindex(columns=ancestorsec, fill_value=0.0).astype(np.float32)
    return df


def build_rxn_ec_mask(rxn_ids, rxn_to_ec, allecs):
    n_rxns, n_ecs = len(rxn_ids), len(allecs)
    ec_to_idx = {ec: i for i, ec in enumerate(allecs)}
    mask = np.zeros((n_rxns, n_ecs), dtype=np.float32)
    for i, rxn in enumerate(rxn_ids):
        rname = rxn[:-2] if rxn.endswith(('_c', '_e', '_p')) else rxn
        ecs = rxn_to_ec.get(rname, [])
        for ec in ecs:
            if ec in ec_to_idx:
                mask[i, ec_to_idx[ec]] = 1.0
    return mask


def extract_fba_matrices(model, rxn_ids, reversed_trans=True):
    """Extract S, lb, ub. Cached on disk."""
    suffix = 'RE' if reversed_trans else 'noRE'
    svpath = data_path(f'fba_matrices_v6{suffix}.pkl')
    if os.path.exists(svpath):
        with open(svpath, 'rb') as f:
            data = pickle.load(f)
        S = data['S']
        if not isinstance(S, scipy.sparse.csr_matrix):
            S = csr_matrix(S)
        return S, data['lb'].astype(np.float64), data['ub'].astype(np.float64)

    S_df = create_stoichiometric_matrix(model, array_type="DataFrame")
    lb = np.array([model.reactions.get_by_id(r).lower_bound for r in rxn_ids], dtype=np.float64)
    ub = np.array([model.reactions.get_by_id(r).upper_bound for r in rxn_ids], dtype=np.float64)
    S = csr_matrix(S_df.to_numpy())

    if reversed_trans:
        for idx, r in enumerate(rxn_ids):
            # EX_biomass is the biomass DRAIN (cpd11416_c -->), not a nutrient
            # exchange. Flipping its sign turns the drain into a source, making
            # biomass mass-balance-infeasible. Exclude it from the flip.
            if r.startswith('EX_') and r != 'EX_biomass':
                S[:, idx] = -S[:, idx]

    with open(svpath, 'wb') as f:
        pickle.dump({'S': S, 'lb': lb, 'ub': ub}, f)
    return S, lb, ub


def load_tight_bounds(tight_path):
    """
    Load precomputed FVA-tightened bounds from v6_precompute_bounds.py.
    File format: dict {'lb_tight': array, 'ub_tight': array}.
    Returns None if not found.
    """
    if not os.path.exists(tight_path):
        warnings.warn(f"Tight bounds {tight_path} not found; using default bounds. "
                      "Run v6_precompute_bounds.py first for ~3-10x speedup.")
        return None, None
    with open(tight_path, 'rb') as f:
        d = pickle.load(f)
    return d['lb_tight'].astype(np.float64), d['ub_tight'].astype(np.float64)


def find_excluded_reactions(S, lb, ub, allrxns, universal_obj):
    """
    Identify reactions that must be excluded from candidate set:
      - The OTHER Gram biomass (when current is GmNeg, exclude GmPos)
      - Reactions on dead-end metabolites (only one carrier, no balance)
      - Reactions on single-direction metabolites with bound inconsistency
    """
    S_sparse = csr_matrix(S)
    excludes = set()

    # other-gram biomass
    if 'GmNeg' in universal_obj and 'biomass_GmPos' in allrxns:
        excludes.add(allrxns.index('biomass_GmPos'))
    elif 'GmPos' in universal_obj and 'biomass_GmNeg' in allrxns:
        excludes.add(allrxns.index('biomass_GmNeg'))

    # dead-end metabolites
    for i in range(S_sparse.shape[0]):
        row = S_sparse.getrow(i)
        nnz = len(row.indices)
        if nnz == 0:
            continue
        if nnz == 1:
            excludes.add(int(row.indices[0]))
        elif nnz == 2:
            for k in row.indices:
                if allrxns[k].startswith('SNK_'):
                    excludes.add(int(row.indices[0]))
                    break

    # single-direction metabolites: all stoich coefs same sign,
    # and no reversible reaction touching them
    for i in range(S_sparse.shape[0]):
        row = S_sparse.getrow(i)
        if len(row.data) == 0:
            continue
        if np.all(row.data > 0) or np.all(row.data < 0):
            has_reversible = any(lb[k] < 0 and ub[k] > 0 for k in row.indices)
            if not has_reversible:
                for k in row.indices:
                    excludes.add(int(k))

    return excludes


def apply_media(media_input, allrxns, lb, ub):
    """
    Set exchange-reaction bounds based on media composition.

    Returns:
        lb, ub : modified in place
        media_mask : np.int8 [n_rxns]  1=in media, -1=blocked, 0=internal
        media_rxns : set of indices of exchange reactions allowed by media

    BUG FIX vs v5: no longer overwrites c_add/c_remove here. Cost is
    decided by the unified cost function in build_objective().
    """
    n = len(allrxns)
    media_mask = np.zeros(n, dtype=np.int8)
    media_rxns = set()

    media = get_media(media_input) if media_input else []
    minimal_uptake = {
        'EX_cpd00035_e','EX_cpd00051_e','EX_cpd00132_e','EX_cpd00041_e',
        'EX_cpd00084_e','EX_cpd00053_e','EX_cpd00023_e','EX_cpd00033_e',
        'EX_cpd00119_e','EX_cpd00322_e','EX_cpd00107_e','EX_cpd00039_e',
        'EX_cpd00060_e','EX_cpd00066_e','EX_cpd00129_e','EX_cpd00054_e',
        'EX_cpd00161_e','EX_cpd00065_e','EX_cpd00069_e','EX_cpd00156_e',
        'EX_cpd00027_e','EX_cpd00149_e','EX_cpd00030_e','EX_cpd00254_e',
        'EX_cpd00971_e','EX_cpd00063_e','EX_cpd10515_e','EX_cpd00205_e','EX_cpd00099_e',
        # standard inorganic nutrients required for biomass synthesis.
        # cpd00009 (phosphate) was MISSING and is the binding constraint:
        # without it no ATP/DNA/RNA/lipid/cofactors can form -> biomass=0.
        'EX_cpd00009_e',  # phosphate
        'EX_cpd00001_e',  # H2O
        'EX_cpd00067_e',  # H+
        'EX_cpd00007_e',  # O2
        'EX_cpd00013_e',  # NH3
        'EX_cpd00048_e',  # sulfate
        'EX_cpd00011_e',  # CO2
    }
    allowed = set('EX_' + c for c in media) | minimal_uptake

    for i, r in enumerate(allrxns):
        if not r.startswith('EX_'):
            continue
        if r in allowed:
            lb[i] = -100.0
            ub[i] = 100.0
            media_mask[i] = 1
            media_rxns.add(i)
        else:
            lb[i] = 0.0       # blocked uptake (but can still secrete if needed)
            ub[i] = 1000.0
            media_mask[i] = -1
    return lb, ub, media_mask, media_rxns


def aggregate_confidence(
    pred_df, rxn_ec_mask, allrxns,
    gpr_map=None,
    method='noisy_or',
):
    """
    Compute w_j (∈ [0,1]) = confidence that reaction j is present.

    method ∈ {'max', 'noisy_or', 'gpr'}
      'max'      : v5 behaviour — max over (protein, EC) pairs mapping to j
      'noisy_or' : 1 - prod_{p,e}(1 - E_{p,e})  for e ∈ M⁻¹(j), p ∈ P
      'gpr'      : uses gpr_map. complex_conf = min over subunits; reaction
                   conf = max over complexes. Falls back to noisy_or for
                   reactions without GPR entry.

    Returns:
        w : np.ndarray[float32], shape (n_rxns,)
    """
    if method not in {'max', 'noisy_or', 'gpr'}:
        raise ValueError(f"Unknown aggregation method: {method}")

    pred = pred_df.values.astype(np.float32)          # (n_proteins, n_ecs)
    n_proteins, n_ecs = pred.shape
    n_rxns = rxn_ec_mask.shape[0]
    w = np.zeros(n_rxns, dtype=np.float32)

    # per-protein per-reaction max over EC mapping → matrix (n_p, n_rxns)
    # but n_p × n_rxns can be huge; we compute per-reaction lazily.

    if method == 'max':
        for j in range(n_rxns):
            ec_idx = np.where(rxn_ec_mask[j] == 1)[0]
            if len(ec_idx) == 0:
                continue
            w[j] = float(pred[:, ec_idx].max())

    elif method == 'noisy_or':
        # numerical: log-space. log(1 - E) summed.
        log1m = np.log(np.clip(1.0 - pred, 1e-9, 1.0))    # (n_p, n_ecs)
        for j in range(n_rxns):
            ec_idx = np.where(rxn_ec_mask[j] == 1)[0]
            if len(ec_idx) == 0:
                continue
            # sum over all (protein, EC) pairs mapping to reaction j
            s = float(log1m[:, ec_idx].sum())
            w[j] = 1.0 - np.exp(s)

    elif method == 'gpr':
        if gpr_map is None:
            warnings.warn("GPR method requested but no gpr_map; falling back to noisy_or.")
            return aggregate_confidence(pred_df, rxn_ec_mask, allrxns,
                                        gpr_map=None, method='noisy_or')

        # noisy_or fallback for reactions without GPR
        fallback_w = aggregate_confidence(pred_df, rxn_ec_mask, allrxns,
                                          gpr_map=None, method='noisy_or')

        protein_index = {p: i for i, p in enumerate(pred_df.index)}
        log1m = np.log(np.clip(1.0 - pred, 1e-9, 1.0))

        for j, rxn in enumerate(allrxns):
            rkey = rxn[:-2] if rxn.endswith(('_c', '_e', '_p')) else rxn
            ec_idx = np.where(rxn_ec_mask[j] == 1)[0]
            if rkey not in gpr_map or len(ec_idx) == 0:
                w[j] = fallback_w[j]
                continue

            complexes = gpr_map[rkey]  # list of lists
            if not complexes:
                w[j] = fallback_w[j]
                continue

            # for each complex (AND of subunits), confidence = min subunit conf;
            # subunit conf = noisy-OR over ECs mapping to j
            complex_confs = []
            for cplx in complexes:
                subunit_confs = []
                for p_id in cplx:
                    if p_id not in protein_index:
                        subunit_confs.append(0.0)
                        continue
                    p_i = protein_index[p_id]
                    s = float(log1m[p_i, ec_idx].sum())
                    subunit_confs.append(1.0 - np.exp(s))
                complex_confs.append(min(subunit_confs) if subunit_confs else 0.0)
            w[j] = max(complex_confs) if complex_confs else fallback_w[j]

    return w


def compute_costs(w, mode='logodds', theta=0.5, eps=EPS_SMOOTH):
    """
    Compute per-reaction cost c_j.

    mode='logodds' (RECOMMENDED): single continuous cost
        c_j = -log( (w_j + eps) / (1 - w_j + eps) )
      high w → negative cost (encouraged)
      low  w → positive cost (penalized if included)

    mode='piecewise' (v5 baseline for ablation):
        c_add_j = theta - w_j + eps     for w_j ≤ theta
        c_rem_j = w_j - theta + eps     for w_j > theta
      returned as separate arrays.

    Returns:
        mode='logodds'  : dict {'c': array}
        mode='piecewise': dict {'c_add': array, 'c_rem': array, 'r0': int8 array}
    """
    w = np.asarray(w, dtype=np.float64)

    if mode == 'logodds':
        c = -np.log((w + eps) / (1.0 - w + eps))
        return {'c': c.astype(np.float64)}

    elif mode == 'piecewise':
        n = len(w)
        c_add = np.zeros(n, dtype=np.float64)
        c_rem = np.zeros(n, dtype=np.float64)
        r0 = np.zeros(n, dtype=np.int8)
        below = w <= theta
        c_add[below] = theta - w[below] + eps
        c_rem[~below] = w[~below] - theta + eps
        r0[~below] = 1
        return {'c_add': c_add, 'c_rem': c_rem, 'r0': r0}

    else:
        raise ValueError(f"Unknown cost mode: {mode}")


def _detect_solver(threads=8, time_limit=None, gap=0.05, msg=False):
    """
    Auto-detect best available solver. Priority: GUROBI > HiGHS > CBC.
    """
    # Try Gurobi
    try:
        from pulp import GUROBI_CMD, GUROBI
        try:
            solver = GUROBI(msg=msg, threads=threads, timeLimit=time_limit,
                            gapRel=gap, MIPFocus=1)
            if solver.available():
                print(f"[solver] Gurobi  threads={threads}  gap={gap}  tl={time_limit}", flush=True)
                return solver, 'gurobi'
        except Exception:
            pass
        solver = GUROBI_CMD(msg=msg, threads=threads, timeLimit=time_limit, gapRel=gap)
        if solver.available():
            print(f"[solver] Gurobi (CMD)", flush=True)
            return solver, 'gurobi_cmd'
    except Exception:
        pass

    # Try HiGHS
    try:
        from pulp import HiGHS_CMD
        solver = HiGHS_CMD(msg=msg, threads=threads, timeLimit=time_limit, gapRel=gap)
        if solver.available():
            print(f"[solver] HiGHS  threads={threads}", flush=True)
            return solver, 'highs'
    except Exception:
        pass

    # Fallback CBC
    opts = [f"threads {threads}", f"ratio {gap}",
            "heuristics on", "cuts on", "presolve on"]
    solver = PULP_CBC_CMD(msg=msg, threads=threads, timeLimit=time_limit,
                          gapRel=gap, options=opts)
    print(f"[solver] CBC fallback  threads={threads}", flush=True)
    return solver, 'cbc'


def rfinal2ec(y_vals, rxn_to_ec, allecs, allrxns):
    """Identify active and muted ECs based on MILP y solution."""
    active = set()
    for j, yv in enumerate(y_vals):
        if yv > 0.5:
            rkey = allrxns[j].split('_')[0]
            ecs = rxn_to_ec.get(rkey, [])
            active.update(ecs)
    muted = set(allecs) - active
    return active, muted


def posterior_calibrated(pred_df, y_vals, rxn_to_ec, allrxns, ancestorsec,
                        beta=DEFAULT_BETA):
    """
    Bounded multiplicative posterior update — preserves ranking and only
    nudges borderline scores. Low-confidence proteins stay low; high-confidence
    proteins stay high. Unlike the (buggy) E + (1-E)*(1-exp(-beta)) form, no
    protein is unconditionally pushed past threshold by EC activation alone.

    Formula (active EC):  E_new = E + alpha * E * (1 - E)        (boost)
    Formula (muted  EC):  E_new = E - alpha * E * (1 - E)        (dampen)

    alpha is derived from beta as min(1, 0.4*beta). At beta=1 this gives
    alpha=0.4 — a borderline score 0.45 becomes 0.549 (helped); a low
    score 0.01 becomes 0.014 (unchanged); a high score 0.99 stays 0.994.

    Returns:
        opt_df : updated DataFrame (same shape)
        active_ecs, muted_ecs : sets
    """
    active_ecs, muted_ecs = rfinal2ec(y_vals, rxn_to_ec, ancestorsec, allrxns)

    opt_df = pred_df.copy().astype(np.float64)
    alpha = min(1.0, max(0.0, 0.4 * float(beta)))

    for ec in active_ecs:
        if ec not in opt_df.columns:
            continue
        col = opt_df[ec].values
        opt_df[ec] = np.clip(col + alpha * col * (1.0 - col), 0.0, 1.0)

    for ec in muted_ecs:
        if ec not in opt_df.columns:
            continue
        col = opt_df[ec].values
        opt_df[ec] = np.clip(col - alpha * col * (1.0 - col), 0.0, 1.0)

    return opt_df, active_ecs, muted_ecs


def build_candidate_mask(w, allrxns, excludes, media_rxns,
                         essential_skeleton=None, w_min=WMIN_KEEP):
    """
    Variable reduction: y_j is treated as a true binary variable only if:
      - w_j ≥ w_min, OR
      - j ∈ essential_skeleton (must-have for biomass), OR
      - j ∈ media_rxns

    Reactions with w_j < w_min AND not skeleton/media are fixed y_j = 0.
    Returns boolean array.
    """
    R = len(w)
    mask = np.zeros(R, dtype=bool)
    mask[w >= w_min] = True
    if essential_skeleton is not None:
        for j in essential_skeleton:
            mask[j] = True
    for j in media_rxns:
        mask[j] = True
    for j in excludes:
        mask[j] = False
    return mask


def build_submodel(universal, keep_rxn_objs, model_id, *,
                    add_boundary_for_kept_mets=True,
                    biomass_id='biomass_GmNeg'):
    """Build a fresh cobra.Model containing only `keep_rxn_objs`, plus
    boundary (EX_/DM_/SK_) reactions for metabolites that ended up in the
    submodel, plus the biomass reaction as objective.

    Matches meteor_release/scripts/memote_prototype.py::_build_submodel:
    boundary reactions are only pulled in for metabolites ALREADY in the
    kept set (not a broader "any overlapping metabolite anywhere in
    universal" search) -- this keeps submodel size MEMOTE-comparable to a
    CarveMe draft instead of ballooning to ~10k reactions.

    Can't `.copy()` the universal model (deep-copy on ~47880 reactions is
    intractable) and can't `remove_reactions` down to the kept set (cobra
    rebuilds gene/met indexes per removal, O(n^2)). Building a new Model
    with `add_reactions(list)` is linear and fast.
    """
    import cobra
    new_model = cobra.Model(model_id)
    new_model.add_reactions(keep_rxn_objs)

    if add_boundary_for_kept_mets:
        kept_mets = {m.id for m in new_model.metabolites}
        existing_ids = {r.id for r in new_model.reactions}
        extra = []
        for r in universal.reactions:
            if not r.boundary or r.id in existing_ids:
                continue
            mid = next(iter(r.metabolites)).id
            if mid in kept_mets:
                extra.append(r)
        if extra:
            new_model.add_reactions(extra)

    bm = universal.reactions.get_by_id(biomass_id)
    if bm.id not in {r.id for r in new_model.reactions}:
        new_model.add_reactions([bm])
    new_model.objective = bm.id
    return new_model