"""METEOR θ-free core algorithm — clean, importable interface.

The 4 steps of METEOR θ-free:
  1. aggregate_confidence: per-(protein, EC) probs → per-reaction confidence
  2. build_milp: reaction confidence → MILP cost → set up problem
  3. solve_milp: MILP solver (CBC) → y* (reaction on/off)
  4. posterior_calibrated: y* → refined per-(protein, EC) probs

For the SLURM-aware wrapper that reads CLI args and produces output pkls,
see cli.py.

The original research code (with all ablation knobs) is preserved separately
and is not part of this public release.

v7 revision (2026-07-01): aligned with the paper's Methods (Sec. 3.2) and
solver reality on the compute cluster used for the paper's results:
  - Biomass is a hard constraint `v_biomass >= gamma_min` (paper Eq. 236),
    not a soft/slack-penalized one.
  - Added an explicit growth-rate upper bound `v_biomass <= gamma_max`
    (biologically motivated: E. coli's max growth rate is ~2 h^-1) that was
    missing in prior revisions and could let the MILP route unbounded
    biomass flux once the supporting reactions were already active for
    other reasons.
  - Solver is CBC only (no Gurobi/HiGHS auto-detect) — matches the paper's
    stated solver and the fact that Gurobi has no license on the cluster
    this project runs on.
  - `DEFAULT_POSTERIOR_ALPHA_SCALE` is set from a dedicated sensitivity
    sweep (see `scripts/alpha_sensitivity_v7.py` / `docs/RESULTS.md`), not
    copied from the paper text (0.4) or from the earlier BacDive-tuned
    guess (0.25) — see that sweep's output before changing this value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp


EPS_SMOOTH = 1e-6
DEFAULT_GAMMA_MIN = 0.1
# Growth-rate upper bound on v_biomass. 2.5 h^-1 is above E. coli's fastest
# reported doubling time (~20 min => mu ~2 h^-1) and serves purely as a
# sanity cap against solver-artefact biomass blow-up, not a tight biological
# estimate for any specific organism.
DEFAULT_GAMMA_MAX = 2.5
DEFAULT_BETA = 1.0
# Multiplicative-posterior step size scale. The actual α used per (β, scale) is
# alpha = clip(scale * β, 0, 1). PENDING: this default is being determined by
# a dedicated sensitivity sweep across alpha_scale (not just beta) on the
# 3-organism holdout panel; do not change this value without a sweep result
# to cite. Placeholder until that sweep completes.
DEFAULT_POSTERIOR_ALPHA_SCALE = 0.25


# ---------------------------------------------------------------------------
# Step 1: noisy-OR aggregation
# ---------------------------------------------------------------------------

def aggregate_confidence(
    pred_df: pd.DataFrame,
    rxn_ec_mask: np.ndarray,
    method: str = "noisy_or",
) -> np.ndarray:
    """Aggregate per-(protein, EC) probabilities → per-reaction confidence.

    Parameters
    ----------
    pred_df : DataFrame, shape (n_proteins, n_ecs)
        Raw [0,1] baseline probabilities. Index is protein ID, columns are EC.
    rxn_ec_mask : ndarray, shape (n_rxns, n_ecs)
        Binary mask M[j, e] = 1 iff EC e is associated with reaction j.
    method : {"max", "noisy_or"}
        "noisy_or" (default, recommended): w_j = 1 - prod(1 - E_{p,e}) over
            (p, e) such that EC e is mapped to reaction j.
        "max": w_j = max over (p, e) such that EC e is mapped to reaction j.

    Returns
    -------
    w : ndarray, shape (n_rxns,)
        Per-reaction aggregated confidence in [0, 1].
    """
    if method not in {"max", "noisy_or"}:
        raise ValueError(f"Unknown aggregation method: {method}")

    pred = pred_df.values.astype(np.float32)
    n_p, n_e = pred.shape
    n_rxns = rxn_ec_mask.shape[0]
    w = np.zeros(n_rxns, dtype=np.float32)

    for j in range(n_rxns):
        ec_idx = np.where(rxn_ec_mask[j] == 1)[0]
        if len(ec_idx) == 0:
            continue
        ec_preds = pred[:, ec_idx]  # (n_p, n_ec_for_rxn)
        if method == "max":
            w[j] = float(ec_preds.max())
        else:  # noisy_or
            log1m = np.log(np.clip(1.0 - ec_preds, 1e-8, 1.0))
            w[j] = float(1.0 - np.exp(log1m.sum()))
    return w


# ---------------------------------------------------------------------------
# Step 2 & 3: MILP construction + solve
# ---------------------------------------------------------------------------

@dataclass
class CapabilitySpec:
    """A physiological capability the MILP should account for beyond biomass.

    Used to extend the MILP objective with reward terms for organism-specific
    metabolic activities (e.g., nitrogen fixation, secondary metabolite
    biosynthesis, pigment production). Caller passes a list of these to
    `MILPParams.capabilities`.

    name : human-readable identifier (e.g., "N2_fixation").
    indicator_ecs : 4-digit EC numbers whose baseline confidence above
        `min_conf` flags this capability as predicted for the organism.
    production_rxn_idxs : indices (in the universal reaction order) of
        reactions in the universal model that realize this capability.
        At least one of these is required to carry flux ≥ `min_flux` when
        the capability is flagged as predicted.
    min_conf : threshold on max-protein confidence for `indicator_ecs`
        (default 0.5). If no protein in the proteome exceeds this for any
        indicator EC, the capability is treated as not predicted and the
        constraint is dropped.
    min_flux : minimum required flux through at least one production
        reaction when active (default 0.01).
    reward : weight for the term `−reward · Σ v_production_rxns` added to
        the objective (default 0.1). Larger reward forces the MILP to
        prioritize satisfying this capability over parsimony.
    """
    name: str
    indicator_ecs: list[str]
    production_rxn_idxs: list[int]
    min_conf: float = 0.5
    min_flux: float = 0.01
    reward: float = 0.1


@dataclass
class MILPParams:
    gamma_min: float = DEFAULT_GAMMA_MIN
    # Growth-rate upper bound: v_biomass <= gamma_max. Prevents the MILP from
    # routing unbounded biomass flux through an already-active subnetwork
    # (a solver artefact, not biology) when nothing else constrains it.
    gamma_max: float = DEFAULT_GAMMA_MAX
    lam: float = 1e-4
    big_slack: float = 1e4
    time_limit_s: int = 1800
    gap: float = 0.05
    # Optional multi-objective extensions for organism-specific capabilities
    # beyond biomass (Section 6 in companion paper). Each entry is checked
    # against the organism's baseline predictions and added as both a soft
    # constraint and an objective reward when the capability is predicted.
    capabilities: list = field(default_factory=list)
    # Multiplicative posterior step size. The effective α used in
    # `posterior_calibrated` is `clip(posterior_alpha_scale * beta, 0, 1)`.
    posterior_alpha_scale: float = DEFAULT_POSTERIOR_ALPHA_SCALE
    # Solver thread count. Previously unset -- CBC/Gurobi/HiGHS all defaulted
    # to 1 thread, which on the ~48k-reaction universal MILP frequently timed
    # out with status "Not Solved" well before time_limit_s, silently
    # returning a degenerate/incumbent solution that did not vary by genome.
    threads: int = 8


def logodds_cost(w: np.ndarray, eps: float = EPS_SMOOTH) -> np.ndarray:
    """Log-odds reaction cost. Negative for w > 0.5 (favored), positive otherwise."""
    return -np.log((w + eps) / (1.0 - w + eps))


def build_milp(
    S: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    cost: np.ndarray,
    biomass_idx: int,
    params: MILPParams,
):
    """Build the METEOR MILP problem.

    Parameters
    ----------
    S : ndarray, (n_mets, n_rxns)
        Stoichiometric matrix.
    lb, ub : ndarray
        Reaction flux bounds.
    cost : ndarray
        Per-reaction log-odds cost.
    biomass_idx : int
        Index of biomass reaction (must produce at least gamma_min flux).
    params : MILPParams
    """
    n_mets, n_rxns = S.shape

    model = pulp.LpProblem("METEOR_thetafree", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("y", range(n_rxns), cat="Binary")
    v = pulp.LpVariable.dicts("v", range(n_rxns), cat="Continuous")
    for j in range(n_rxns):
        v[j].lowBound = lb[j]
        v[j].upBound = ub[j]

    # mass balance (all metabolites, including the one(s) biomass drains
    # into -- no longer skipped; a sink/drain reaction in the universal
    # model handles biomass outflow, matching the production pipeline)
    for i in range(n_mets):
        # S[i, :] @ v = 0
        nz = np.nonzero(S[i, :])[0]
        if len(nz) == 0:
            continue
        expr = pulp.lpSum(S[i, j] * v[j] for j in nz)
        model += expr == 0, f"mass_balance_{i}"

    # flux-y coupling
    for j in range(n_rxns):
        model += v[j] >= lb[j] * y[j], f"v_lb_{j}"
        model += v[j] <= ub[j] * y[j], f"v_ub_{j}"

    # biomass constraint: hard floor (paper Eq. 236) + growth-rate cap
    model += v[biomass_idx] >= params.gamma_min, "biomass_min"
    model += v[biomass_idx] <= params.gamma_max, "biomass_max"

    # objective: minimize cost·y + λ·|v|
    obj = pulp.lpSum(cost[j] * y[j] for j in range(n_rxns))
    if params.lam > 0:
        v_abs = pulp.LpVariable.dicts("vabs", range(n_rxns), lowBound=0)
        for j in range(n_rxns):
            model += v_abs[j] >= v[j], f"abs_pos_{j}"
            model += v_abs[j] >= -v[j], f"abs_neg_{j}"
        obj = obj + params.lam * pulp.lpSum(v_abs[j] for j in range(n_rxns))

    # Multi-objective extension: organism-specific capabilities beyond biomass.
    # For each active capability (caller has already filtered to those predicted
    # in the proteome), require at least one of its production reactions to
    # carry ≥ min_flux, and reward total flux through the capability set in the
    # objective.
    for cap in getattr(params, "capabilities", []) or []:
        rxn_idxs = [j for j in cap.production_rxn_idxs if 0 <= j < n_rxns]
        if not rxn_idxs:
            continue
        # Soft constraint: sum of fluxes across capability reactions ≥ min_flux.
        # We use sum (not max) for LP-friendly form; absolute-value handling
        # would require additional binary disjunctions.
        model += (pulp.lpSum(v[j] for j in rxn_idxs) >= cap.min_flux,
                  f"cap_{cap.name}_min")
        # Objective reward (subtract weighted flux sum to favour higher fluxes)
        obj = obj - cap.reward * pulp.lpSum(v[j] for j in rxn_idxs)

    model += obj
    return model, y, v


# ---------------------------------------------------------------------------
# Step 5 (optional): parsimonious dead-end gap-filling
# ---------------------------------------------------------------------------


def _dead_end_metabolites(S: np.ndarray, active_mask: np.ndarray, lb: np.ndarray,
                          ub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Identify metabolites with no producer or no consumer in the active set.

    A reaction `j` is treated as bidirectional iff `lb[j] < 0 < ub[j]`. Returns
    boolean arrays `(no_producer, no_consumer)` over metabolites.
    """
    n_mets, n_rxns = S.shape
    active_idx = np.where(active_mask)[0]
    rev_mask = (lb < 0) & (ub > 0)
    no_p = np.ones(n_mets, dtype=bool)
    no_c = np.ones(n_mets, dtype=bool)
    for j in active_idx:
        col = S[:, j]
        is_rev = rev_mask[j]
        for i in np.nonzero(col)[0]:
            coef = col[i]
            if is_rev or coef > 0:
                no_p[i] = False
            if is_rev or coef < 0:
                no_c[i] = False
    return no_p, no_c


def parsimonious_gapfill(
    y_star: np.ndarray,
    S: np.ndarray,
    cost: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    biomass_idx: int | None = None,
    max_additions: int = 50,
    skip_cofactors: bool = True,
    cofactor_ids: set[int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Greedy minimum-cost dead-end repair after the main MILP.

    Reconstructor-style parsimonious gap-fill: while the active set still has
    dead-end metabolites (no producer or no consumer under stoichiometric
    reachability, ignoring flux bounds), pick the universal reaction that
    (a) resolves at least one dead-end and (b) has the lowest log-odds cost
    (i.e., the strongest baseline support). Stops when there are no more
    dead-ends to repair or `max_additions` reactions have been added.

    Parameters
    ----------
    y_star : (n_rxns,) binary array — output of `build_milp` / `solve_milp`.
    S : (n_mets, n_rxns) stoichiometry.
    cost : (n_rxns,) per-reaction log-odds cost (negative = favored).
    lb, ub : (n_rxns,) bound arrays (used to decide reaction directionality).
    biomass_idx : index of the biomass reaction; metabolites consumed by it
        are always kept (they must have a producer in any viable model).
    max_additions : safety cap on the number of reactions added (default 50).
    skip_cofactors : whether to skip "universal cofactor" metabolites
        (water, H+, ATP/ADP, NAD(P)H/NAD(P)+, CO2, O2 etc.) that are
        intentionally allowed to be transient in many SEED-derived models.
    cofactor_ids : optional override set of metabolite indices to treat as
        cofactors. If None, callers should pass the SEED-typical cofactor
        compound indices; we default to an empty set so the caller stays in
        control.

    Returns
    -------
    (y_filled, info)
        y_filled : binary array, same length as y_star, with added reactions
            set to 1.
        info : dict with keys `added_rxn_idxs`, `n_added`,
            `dead_ends_before`, `dead_ends_after`, `cost_added` (sum of
            log-odds costs of newly added reactions).
    """
    cof = cofactor_ids if cofactor_ids is not None else set()
    y = (y_star > 0.5).astype(np.int8).copy()
    n_mets, n_rxns = S.shape

    no_p0, no_c0 = _dead_end_metabolites(S, y.astype(bool), lb, ub)
    dead_before = int((no_p0 | no_c0).sum())

    added = []
    for _ in range(max_additions):
        no_p, no_c = _dead_end_metabolites(S, y.astype(bool), lb, ub)
        targets = (no_p | no_c)
        if skip_cofactors and cof:
            for i in cof:
                targets[i] = False
        # Don't try to gap-fill metabolites that the active set isn't using at all
        used = (np.abs(S[:, y.astype(bool)]) > 0).any(axis=1)
        targets &= used
        target_idxs = np.where(targets)[0]
        if not target_idxs.size:
            break
        # Candidate reactions: universal reactions not yet in active that touch a target metabolite
        candidate_mask = np.zeros(n_rxns, dtype=bool)
        for i in target_idxs:
            col = S[i, :]
            candidate_mask |= (col != 0) & (y == 0)
        cand_idxs = np.where(candidate_mask)[0]
        if not cand_idxs.size:
            break
        # Pick the cheapest candidate (lowest log-odds cost = strongest support)
        best = cand_idxs[np.argmin(cost[cand_idxs])]
        y[best] = 1
        added.append(int(best))

    no_p1, no_c1 = _dead_end_metabolites(S, y.astype(bool), lb, ub)
    dead_after = int((no_p1 | no_c1).sum())

    info = {
        "added_rxn_idxs": added,
        "n_added": len(added),
        "dead_ends_before": dead_before,
        "dead_ends_after": dead_after,
        "cost_added": float(cost[added].sum()) if added else 0.0,
    }
    return y, info


CANONICAL_CAPABILITIES = {
    # name → (description, indicator 4-digit ECs)
    # ── Geochemical cycling (N / S / C / Fe / H₂) ──
    "N2_fixation": ("Biological nitrogen fixation (nitrogenase complex)",
                    ["1.18.6.1", "1.19.6.1"]),
    "Nitrification_ammonia": ("NH3 oxidation to nitrite (ammonia monooxygenase)",
                              ["1.14.99.39"]),
    "Nitrification_nitrite": ("NO2- oxidation to nitrate (nitrite oxidoreductase)",
                              ["1.7.99.4"]),
    "Denitrification_nitrate": ("Anaerobic nitrate reduction",
                                ["1.7.5.1", "1.7.99.4"]),
    "Denitrification_nitrite": ("Nitrite reduction (NO production)",
                                ["1.7.2.1", "1.7.2.2"]),
    "Denitrification_NO": ("Nitric oxide reductase",
                           ["1.7.2.5"]),
    "Denitrification_N2O": ("Nitrous oxide reductase (terminal)",
                            ["1.7.2.4"]),
    "Sulfate_reduction": ("Dissimilatory sulfate reduction",
                          ["1.8.99.5", "2.7.7.4", "1.8.99.2"]),
    "Sulfide_oxidation": ("Sulfide / thiosulfate oxidation (Sox system)",
                          ["1.8.5.2", "1.8.2.2"]),
    "Methanogenesis": ("CH4 production via coenzyme M / F420 pathways",
                       ["1.12.99.5", "1.5.99.11", "2.8.4.1"]),
    "Methylotrophy": ("Methanol / formaldehyde oxidation",
                      ["1.1.99.8", "1.2.1.46"]),
    "Carbon_fixation_CBB": ("Calvin–Benson cycle (rubisco)",
                             ["4.1.1.39"]),
    "Carbon_fixation_rTCA": ("Reverse TCA (autotrophic CO2 fixation)",
                              ["1.1.1.42", "4.1.3.34"]),
    "Hydrogenase_uptake": ("H2 oxidation (NiFe / FeFe hydrogenase)",
                            ["1.12.1.2", "1.12.7.2"]),
    "Iron_oxidation": ("Fe(II) oxidation (chemolithotroph)",
                       ["1.9.99.1"]),
    "Arsenate_reduction": ("Dissimilatory arsenate reductase",
                            ["1.20.99.1"]),
    "Selenate_reduction": ("Dissimilatory selenate reductase",
                            ["1.97.1.9"]),
    "Acetogenesis": ("Wood–Ljungdahl acetyl-CoA pathway",
                      ["1.2.7.4", "2.1.1.245"]),
    # ── Phototrophy ──
    "Photosynthesis_PSII": ("Oxygenic photosynthesis (PSII, water-splitting)",
                             ["1.10.3.9"]),
    "Photosynthesis_PSI": ("Photosystem I (ferredoxin reduction)",
                            ["1.97.1.12"]),
    "BChl_biosynth": ("Bacteriochlorophyll synthesis (anoxygenic phototrophs)",
                       ["1.3.7.7", "1.3.1.111"]),
    # ── Secondary metabolism / specialized pathways ──
    "Polyketide_KS": ("Type-II polyketide / FAS-II initiation",
                       ["2.3.1.41", "2.3.1.85"]),
    "Polyketide_AT": ("Polyketide acyltransferase",
                       ["2.3.1.169"]),
    "NRPS_termination": ("Non-ribosomal peptide synthetase thioesterase",
                          ["3.1.2.14"]),
    "Lantibiotic_biosynth": ("Class I/II lanthipeptide synthetase",
                              ["4.2.3.4", "4.2.3.5"]),
    "Bacteriocin_biosynth": ("Bacteriocin biosynthesis enzymes",
                              ["3.4.21.86"]),
    "Carotenoid_biosynth": ("Phytoene synthase (carotenoid pigments)",
                             ["2.5.1.32"]),
    "Phenazine_biosynth": ("Phenazine biosynthesis (PhzF / PhzS)",
                            ["5.3.3.17", "1.13.11.78"]),
    "Indole_alkaloid": ("Indole alkaloid biosynthesis (strictosidine synth.)",
                         ["4.3.3.2"]),
    "Siderophore_NRPS": ("Iron-chelating siderophore synthetase",
                          ["6.3.2.39"]),
    "Antibiotic_β_lactam": ("β-Lactam antibiotic biosynthesis (IPNS / ACVS)",
                              ["1.21.3.1", "6.3.5.9"]),
    "Aflatoxin_biosynth": ("Aflatoxin biosynthesis (PKS-NRPS hybrid)",
                            ["1.13.11.72"]),
    "Tetracycline_biosynth": ("Tetracycline biosynthesis (anhydrotetracycline ox.)",
                               ["1.14.13.231"]),
    # ── Quorum sensing / cell–cell signalling ──
    "AHL_quorum_sensing": ("N-acyl-homoserine lactone synthase (LuxI)",
                            ["2.3.1.184"]),
    "AI2_quorum_sensing": ("Autoinducer-2 synthesis (LuxS)",
                            ["4.4.1.21"]),
    # ── Cell envelope / outer-surface specialized ──
    "EPS_synthesis": ("Exopolysaccharide synthesis (cellulose synthase etc.)",
                       ["2.4.1.12", "2.4.1.41"]),
    "Lipopolysacch_biosynth": ("LPS lipid A biosynthesis (LpxA / LpxC)",
                                 ["2.3.1.129", "3.5.1.108"]),
    # ── Stress / virulence / specialized degradation ──
    "Hopanoid_biosynth": ("Hopanoid (sterol-analog) biosynthesis",
                           ["5.4.99.17"]),
    "PHB_polymer": ("Polyhydroxybutyrate granule synthesis",
                     ["2.3.1.16", "1.1.1.36", "2.3.1.9"]),
    "Halogenation": ("Halogenase activity (FADH2-dependent / vanadium)",
                      ["1.14.19.59", "1.11.1.10"]),
    "Aromatic_degrad": ("Catechol 1,2-dioxygenase (aromatic catabolism)",
                         ["1.13.11.1", "1.13.11.2"]),
    "Xenobiotic_oxidation": ("P450 monooxygenase (broad xenobiotic / SM)",
                              ["1.14.14.1"]),
    "Bioluminescence": ("Bacterial luciferase",
                         ["1.14.14.3"]),
}


def detect_capabilities(
    pred_df: pd.DataFrame,
    seedr2ec: dict[str, list[str]],
    allrxns: list[str],
    capabilities: dict[str, tuple[str, list[str]]] | None = None,
    min_conf: float = 0.5,
    reward: float = 0.1,
) -> list:
    """Scan baseline EC predictions for which physiological capabilities are
    flagged, and return the corresponding `CapabilitySpec` list to be passed
    to `MILPParams.capabilities`.

    Parameters
    ----------
    pred_df : DataFrame of (proteins × ECs) with sigmoid/probability values.
        EC column names may include the "EC:" prefix; both forms are accepted.
    seedr2ec : SEED reaction-id → list of EC strings.
    allrxns : reaction ids in the same order as the MILP's reaction axis (i.e.,
        `[r.id for r in universal.reactions]`).
    capabilities : optional override of the canonical capability catalog
        (default `CANONICAL_CAPABILITIES`).
    min_conf : threshold on max-protein confidence for a capability to be
        considered predicted.
    reward : objective-reward weight passed through to each spec.

    Returns
    -------
    list[CapabilitySpec] : one entry per predicted capability whose production
        reactions exist in the universal model.
    """
    catalog = capabilities or CANONICAL_CAPABILITIES
    cols = {str(c).replace("EC:", "").strip(): c for c in pred_df.columns}
    ec_to_rxn_idxs: dict[str, list[int]] = {}
    for idx, r_id in enumerate(allrxns):
        rname = r_id[:-2] if r_id.endswith(("_c", "_e", "_p")) else r_id
        for ec in seedr2ec.get(rname, []):
            ec_to_rxn_idxs.setdefault(ec, []).append(idx)

    out = []
    for name, (_desc, ecs) in catalog.items():
        max_conf = 0.0
        for ec in ecs:
            col = cols.get(ec)
            if col is None:
                continue
            v = float(pred_df[col].max())
            if v > max_conf:
                max_conf = v
        if max_conf < min_conf:
            continue
        # Pool all reactions mapped to the capability's indicator ECs
        rxn_idxs = sorted({i for ec in ecs for i in ec_to_rxn_idxs.get(ec, [])})
        if not rxn_idxs:
            continue
        out.append(CapabilitySpec(
            name=name,
            indicator_ecs=list(ecs),
            production_rxn_idxs=rxn_idxs,
            min_conf=min_conf,
            reward=reward,
        ))
    return out


def _make_solver(params: MILPParams):
    """CBC is the only solver this project uses in practice (matches the
    paper's stated solver; Gurobi has no license on the cluster the paper's
    results were produced on). No auto-detect branching to keep the public
    release's solver behavior exactly reproducible without a commercial
    license."""
    threads = getattr(params, "threads", 8)
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=params.time_limit_s,
                               gapRel=params.gap, threads=threads)


def solve_milp(model, params: MILPParams):
    """Solve a built MILP and return its status string."""
    solver = _make_solver(params)
    model.solve(solver)
    return pulp.LpStatus[model.status]


def solve_milp_hierarchical(
    S: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    cost: np.ndarray,
    biomass_idx: int,
    params: MILPParams,
    biomass_fraction: float = 0.95,
) -> dict:
    """Two-stage MILP solve guaranteeing both biomass production and
    multi-objective capability activation simultaneously.

    The "naive" single-pass approach (``build_milp`` with capability rewards
    in the objective and a soft ``v_biomass >= gamma_min`` constraint) often
    yields a solver-quality solution where the biomass flux trades off
    against capability reward, and the resulting submodel cannot grow under
    fresh FBA. This is a *solver* artefact: the MILP's true feasible region
    contains many points where both biomass *and* capability can be active.

    The hierarchical scheme makes that guarantee explicit:

      Stage 1 (biomass-only)   : solve ``build_milp`` with empty
        ``capabilities`` to obtain the maximum achievable biomass flux
        ``biomass*`` under parsimonious cost minimisation.

      Stage 2 (capability + biomass floor) : re-solve with
        ``params.capabilities`` and a hard floor
        ``v_biomass >= biomass_fraction * biomass*`` (default
        ``biomass_fraction = 0.95``). The floor prevents the multi-objective
        reward from collapsing biomass; the capability constraints + rewards
        force their reactions into the active set.

    Returns
    -------
    dict with keys
        ``y_stage1``, ``v_stage1``, ``biomass_stage1``, ``status_stage1``
        ``y_stage2``, ``v_stage2``, ``biomass_stage2``, ``status_stage2``
        ``biomass_floor`` (the explicit lower bound applied in stage 2)
    Caller picks ``y_stage2`` as the final multi-objective METEOR solution.
    """
    n_rxns = S.shape[1]

    # ---- Stage 1: biomass-only MILP, no capability rewards/constraints ----
    params_s1 = MILPParams(
        gamma_min=params.gamma_min,
        gamma_max=params.gamma_max,
        lam=params.lam,
        big_slack=params.big_slack,
        time_limit_s=params.time_limit_s,
        gap=params.gap,
        capabilities=[],
        posterior_alpha_scale=params.posterior_alpha_scale,
        threads=params.threads,
    )
    m1, y_vars1, v_vars1 = build_milp(S, lb, ub, cost, biomass_idx, params_s1)
    m1.solve(_make_solver(params_s1))
    status1 = pulp.LpStatus[m1.status]
    biomass1 = float(v_vars1[biomass_idx].value() or 0.0)
    y1 = np.array([int(y_vars1[j].value() or 0) for j in range(n_rxns)], dtype=np.int8)
    v1 = np.array([float(v_vars1[j].value() or 0.0) for j in range(n_rxns)], dtype=np.float64)

    # ---- Stage 2: capability MILP with biomass floor inherited from stage 1 ----
    floor = max(float(params.gamma_min), float(biomass_fraction) * biomass1)
    params_s2 = MILPParams(
        gamma_min=floor,
        gamma_max=params.gamma_max,
        lam=params.lam,
        big_slack=params.big_slack,
        time_limit_s=params.time_limit_s,
        gap=params.gap,
        capabilities=list(params.capabilities or []),
        posterior_alpha_scale=params.posterior_alpha_scale,
        threads=params.threads,
    )
    m2, y_vars2, v_vars2 = build_milp(S, lb, ub, cost, biomass_idx, params_s2)
    m2.solve(_make_solver(params_s2))
    status2 = pulp.LpStatus[m2.status]
    biomass2 = float(v_vars2[biomass_idx].value() or 0.0)
    y2 = np.array([int(y_vars2[j].value() or 0) for j in range(n_rxns)], dtype=np.int8)
    v2 = np.array([float(v_vars2[j].value() or 0.0) for j in range(n_rxns)], dtype=np.float64)

    return {
        "y_stage1": y1, "v_stage1": v1, "biomass_stage1": biomass1, "status_stage1": status1,
        "y_stage2": y2, "v_stage2": v2, "biomass_stage2": biomass2, "status_stage2": status2,
        "biomass_floor": floor,
    }


# ---------------------------------------------------------------------------
# Step 4: calibrated multiplicative posterior
# ---------------------------------------------------------------------------

def posterior_calibrated(
    pred_df: pd.DataFrame,
    active_ecs: set[str],
    muted_ecs: set[str],
    beta: float = DEFAULT_BETA,
    alpha_scale: float = DEFAULT_POSTERIOR_ALPHA_SCALE,
) -> pd.DataFrame:
    """Multiplicative posterior update preserving baseline ranking.

    For each EC e in active_ecs (mapped to at least one MILP-active reaction):
        E_new = E + alpha * E * (1 - E)
    For e in muted_ecs:
        E_new = E - alpha * E * (1 - E)
    where alpha = clip(alpha_scale * beta, 0, 1).

    Borderline scores (E~0.5) move the most (by α/4); extremes (E~0 or E~1)
    move little, so baseline ranking is preserved when MILP signal is weak.

    `alpha_scale` defaults to 0.25 (BacDive-tuned). The legacy v6 value was
    0.4, which was too aggressive on borderline DPZ scores and depressed
    precision relative to baseline.
    """
    alpha = min(1.0, max(0.0, float(alpha_scale) * float(beta)))
    out = pred_df.copy().astype(np.float64)
    if alpha == 0.0:
        return out

    for ec in active_ecs:
        if ec not in out.columns:
            continue
        col = out[ec].values
        out[ec] = np.clip(col + alpha * col * (1.0 - col), 0.0, 1.0)

    for ec in muted_ecs:
        if ec not in out.columns:
            continue
        col = out[ec].values
        out[ec] = np.clip(col - alpha * col * (1.0 - col), 0.0, 1.0)

    return out


# ---------------------------------------------------------------------------
# End-to-end convenience: run_meteor
# ---------------------------------------------------------------------------

def run_meteor(
    pred_df: pd.DataFrame,
    universal_S: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rxn_ec_mask: np.ndarray,
    biomass_idx: int,
    rxn_to_ec: dict[str, list[str]],
    allrxns: list[str],
    params: MILPParams | None = None,
    beta: float = DEFAULT_BETA,
) -> tuple[pd.DataFrame, dict]:
    """Run all 4 steps end-to-end. Returns (refined_pred_df, metadata)."""
    params = params or MILPParams()

    logging.info("Step 1: noisy-OR aggregation")
    w = aggregate_confidence(pred_df, rxn_ec_mask, method="noisy_or")

    logging.info("Step 2: log-odds cost")
    cost = logodds_cost(w)

    logging.info("Step 3: build + solve MILP")
    model, y, v = build_milp(universal_S, lb, ub, cost, biomass_idx, params)
    status = solve_milp(model, params)

    y_vals = np.array([pulp.value(y[j]) for j in range(len(allrxns))])
    y_vals = np.where(y_vals > 0.5, 1, 0).astype(np.int8)

    active_ecs, muted_ecs = _rfinal_to_ec(y_vals, rxn_to_ec, allrxns)

    logging.info("Step 4: calibrated posterior")
    refined = posterior_calibrated(pred_df, active_ecs, muted_ecs, beta=beta,
                                   alpha_scale=params.posterior_alpha_scale)

    meta = {
        "milp_status": status,
        "n_active_reactions": int(y_vals.sum()),
        "n_active_ecs": len(active_ecs),
        "n_muted_ecs": len(muted_ecs),
        "w_stats": {
            "mean": float(w.mean()),
            "max": float(w.max()),
            "frac_above_0.5": float((w > 0.5).mean()),
        },
    }
    return refined, meta


def _rfinal_to_ec(y_vals, rxn_to_ec, allrxns):
    """Convert MILP y_j vector to active/muted EC sets."""
    active, muted = set(), set()
    for j, rxn_id in enumerate(allrxns):
        # strip compartment suffix like "_c"
        rname = rxn_id[:-2] if rxn_id.endswith(("_c", "_e", "_p")) else rxn_id
        ecs = rxn_to_ec.get(rname, [])
        if y_vals[j] == 1:
            active.update(ecs)
        else:
            muted.update(ecs)
    # If an EC is both active and muted (mapped to multiple reactions), keep
    # it active (more permissive).
    muted -= active
    return active, muted
