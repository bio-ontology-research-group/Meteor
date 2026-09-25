"""Fig 2: multi-panel results figure for METEOR (PSB 2027 camera-ready).
Reproducible: python eval/gen_fig_results.py  (run from the repository root)
All data read-only from results/table1_v2/pergenome/*.json (panels a,d)
and per-organism values transcribed from Supplementary Tables S19 and S18
(panel b; no per-organism raw file exists for the baseline arm).
Panel c: results/recovery_4arm/*.json (108 genomes, produced by
eval/recovery_ablation.py). Panel d: results/table1_v2 and
results/skeleton_ablation_summary.json.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# Font family, set once, globally, before any panel is drawn, so every text
# artist in every panel inherits the same family with no per-call overrides.
# The previous version's `except Exception` never actually fired -- assigning
# rcParams["font.family"] never raises even for an unavailable font name, so
# that loop always silently kept "Arial" regardless of availability. Query the
# font manager directly instead. Separately (and this was the actual bug
# `pdffonts` caught): any text wrapped in `$...$` is rendered by matplotlib's
# *mathtext* engine, which uses `rcParams["mathtext.fontset"]` -- a completely
# separate setting from `font.family`, defaulting to DejaVu Sans regardless of
# what font.family is set to. All `$\tau$`/`$\to$`/`$\approx$` in this script
# have been replaced with literal Unicode characters (tau/arrow/approx-equal)
# in plain (non-mathtext) strings so nothing in this figure invokes mathtext at
# all; mathtext.fontset is still pinned below as a defensive fallback in case a
# future edit reintroduces a `$...$` expression.
import matplotlib.font_manager as _fm
_available = {f.name for f in _fm.fontManager.ttflist}
for _fam in ("Arial", "Helvetica", "DejaVu Sans"):
    if _fam in _available:
        matplotlib.rcParams["font.family"] = _fam
        break
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = matplotlib.rcParams["font.family"][0] if isinstance(matplotlib.rcParams["font.family"], list) else matplotlib.rcParams["font.family"]
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PERGEN = os.path.join(ROOT, "results/table1_v2/pergenome")
FIGDIR = os.path.join(ROOT, "figures")
RNG_SEED = 0

COL = dict(
    meteor="#0072B2",
    thr="#D55E00",
    repair="#E69F00",
    topk="#9CA3AF",
    uniform="#9CA3AF",
    shuffled="#6B7280",
    skel="#4B5563",
    pair="#BDBDBD",
)

TEXTWIDTH_IN = 476.98244 / 72.27  # printed by a minimal ws-procs11x85 compile, not guessed
FIGW = TEXTWIDTH_IN
FIGH = FIGW * 0.75

rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------- load per-genome table1_v2
files = sorted(glob.glob(os.path.join(PERGEN, "table1_*.json")))
records = [json.load(open(f)) for f in files]
print(f"[data] loaded {len(records)} per-genome table1_v2 records from {PERGEN}")

def arm_field(records, arm, field):
    return np.array([r[arm][field] for r in records if arm in r and r[arm] is not None])

# ============================================================== Panel A: dead-ends per genome
PRED_A = [("clean", "CLEAN"), ("dpz", "DeepProZyme"), ("enzbert", "EnzBERT")]

def panel_a(ax):
    pos_pairs = [(1, 2), (4, 5), (7, 8)]
    verify = {}
    for (base, mp), (k, label) in zip(pos_pairs, PRED_A):
        b_de = arm_field(records, f"baseline_{k}", "deadends")
        m_de = arm_field(records, f"meteor_{k}", "deadends")
        b_grow = arm_field(records, f"baseline_{k}", "fba_growth")
        m_grow = arm_field(records, f"meteor_{k}", "fba_growth")
        n = len(b_de)
        jit_b = base + (rng.random(n) - 0.5) * 0.5
        jit_m = mp + (rng.random(n) - 0.5) * 0.5
        for xb, xm, yb, ym in zip(jit_b, jit_m, b_de, m_de):
            ax.plot([xb, xm], [yb, ym], color=COL["pair"], lw=0.4, alpha=0.3, zorder=1)
        ax.scatter(jit_b, b_de, s=4, color=COL["thr"], alpha=0.5, edgecolors="none", zorder=2)
        ax.scatter(jit_m, m_de, s=4, color=COL["meteor"], alpha=0.5, edgecolors="none", zorder=2)
        ax.errorbar([base], [b_de.mean()], yerr=[b_de.std()], fmt="o", color="black", ms=4, lw=0.8, capsize=2, zorder=3)
        ax.errorbar([mp], [m_de.mean()], yerr=[m_de.std()], fmt="o", color="black", ms=4, lw=0.8, capsize=2, zorder=3)
        n_grow_b = int((b_grow >= 0.05).sum())
        n_grow_m = int((m_grow >= 0.05).sum())
        n_fewer = int((m_de < b_de).sum())
        n_tie = int((m_de == b_de).sum())
        verify[label] = dict(baseline_mean=round(float(b_de.mean()), 2), baseline_sd=round(float(b_de.std()), 2),
                              meteor_mean=round(float(m_de.mean()), 2), meteor_sd=round(float(m_de.std()), 2),
                              n_fewer=n_fewer, n_tie=n_tie, n=n,
                              baseline_grow=f"{n_grow_b}/{n}", meteor_grow=f"{n_grow_m}/{n}")
    ax.set_ylim(0, None)
    ymax = ax.get_ylim()[1]
    # Growth is identically 0/108 -> 108/108 for all three predictors (verified
    # per-predictor above); stating this once, as a single panel-level note,
    # instead of repeating the same text over each of the three groups.
    all_grow_b = all(v["baseline_grow"] == "0/108" for v in verify.values())
    all_grow_m = all(v["meteor_grow"] == "108/108" for v in verify.values())
    if not (all_grow_b and all_grow_m):
        raise ValueError(f"growth is not 0/108 vs 108/108 for every predictor: {verify}")
    # In-panel note (panel c's small grey italic style) in the empty top-left band,
    # above the CLEAN/DeepProZyme clouds (max 17) and left of the EnzBERT outlier.
    growth_note = ax.text(0.03, 0.99, "Genomes that grow: threshold 0/108,\nMETEOR 108/108 (each predictor)",
                          transform=ax.transAxes, fontsize=5.8, ha="left", va="top",
                          style="italic", color="dimgray")
    ax._panel_a_note = growth_note
    ax.set_ylim(0, ymax * 1.08)
    ax.set_xticks([1, 2, 4, 5, 7, 8])
    ax.set_xticklabels(["τ", "M", "τ", "M", "τ", "M"], fontsize=6.5)
    for base, mp, (k, label) in zip([p[0] for p in pos_pairs], [p[1] for p in pos_pairs], PRED_A):
        ax.annotate(label, xy=((base + mp) / 2, 0), xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -17), textcoords="offset points", ha="center", va="top", fontsize=8)
    ax.set_ylabel("Dead-end metabolites per genome", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    legend_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["thr"], markersize=5, label="threshold (τ=0.5)"),
                      Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["meteor"], markersize=5, label="METEOR")]
    # Stacked, in the empty band under the in-panel note: above the CLEAN (max 15)
    # and DeepProZyme (max 17) clouds, well left of the EnzBERT outlier line to 24.
    ax._panel_a_legend = ax.legend(handles=legend_handles, fontsize=6.5, frameon=False, loc="upper left",
                                   bbox_to_anchor=(0.0, 0.86), ncol=1, labelspacing=0.3,
                                   handletextpad=0.2, borderaxespad=0.3)
    return verify

# ============================================================== Panel B: sub-threshold recall/precision
# Transcribed from Supplementary Table S19 (baseline, METEOR recall) and Table S18
# (METEOR + top-K recall and precision, per organism). Baseline per-organism PRECISION is not available anywhere in
# the deposited materials -- only the pooled mean (0.455) is reported in the supplement -- so the
# baseline arm is drawn using per-organism recall (Supplementary Table S19) at the pooled mean precision,
# and this substitution is flagged explicitly in the plot and in provenance.
ORGS_B = ["E. coli", "Salmonella", "K. pneumoniae", "P. putida", "S. aureus", "B. subtilis"]
MARKERS_B = ["o", "s", "^", "D", "v", "P"]
# recall_W: baseline (Supplementary Table S19), METEOR & topK recall+precision (Supplementary Table S18)
B_baseline_recall = [0.495, 0.452, 0.478, 0.479, 0.431, 0.544]
B_meteor_recall =   [0.656, 0.560, 0.633, 0.613, 0.525, 0.647]
B_meteor_prec =     [0.474, 0.471, 0.462, 0.419, 0.339, 0.369]
B_topk_recall =     [0.538, 0.429, 0.489, 0.555, 0.426, 0.461]
B_topk_prec =       [0.432, 0.433, 0.425, 0.390, 0.305, 0.326]
B_BASELINE_MEAN_PREC = 0.455  # pooled/mean only -- no per-organism value exists in deposited data

def panel_b(ax):
    for i, org in enumerate(ORGS_B):
        pts = [(B_baseline_recall[i], B_BASELINE_MEAN_PREC, COL["thr"]),
               (B_topk_recall[i], B_topk_prec[i], COL["topk"]),
               (B_meteor_recall[i], B_meteor_prec[i], COL["meteor"])]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs + [xs[0]], ys + [ys[0]], color=COL["pair"], lw=0.5, alpha=0.5, zorder=1)
        for x, y, c in pts:
            ax.scatter([x], [y], marker=MARKERS_B[i], s=18, color=c, edgecolors="white", linewidths=0.3, zorder=3)
    for x_arr, y_arr, c, lab in [(B_baseline_recall, [B_BASELINE_MEAN_PREC]*6, COL["thr"], "threshold+repair"),
                                   (B_topk_recall, B_topk_prec, COL["topk"], "size-matched top-K"),
                                   (B_meteor_recall, B_meteor_prec, COL["meteor"], "METEOR")]:
        mx, my = float(np.mean(x_arr)), float(np.mean(y_arr))
        sx, sy = float(np.std(x_arr)), float(np.std(y_arr))
        # Star marker: deliberately NOT one of the 6 organism shapes {o,s,^,D,v,P}
        # (MARKERS_B) so a cross-organism mean can never be mistaken for a single
        # organism's point (P. putida already uses "D", the shape this used to share).
        # Semi-transparent fill (not opaque) so an organism point that happens to sit
        # at or near the mean (e.g. E. coli near the METEOR mean) stays visible
        # through/around the star rather than being hidden under it.
        face_rgba = matplotlib.colors.to_rgba(c, alpha=0.72)
        ax.errorbar([mx], [my], xerr=[sx], yerr=[sy], fmt="*", color=c, ms=15, lw=1.1, capsize=2.5,
                    markerfacecolor=face_rgba, markeredgecolor="black", markeredgewidth=0.6, zorder=4)
    ax.set_xlabel("Recall of curated sub-threshold ECs", fontsize=8)
    ax.set_ylabel("EC precision (full reference set)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    extra_correct = 77; extra_total = 773
    marg_prec = extra_correct / extra_total
    org_handles = [Line2D([0], [0], marker=MARKERS_B[i], color="gray", linestyle="none", markersize=4, label=ORGS_B[i]) for i in range(6)]
    method_handles = [Line2D([0], [0], marker="*", color="none", markerfacecolor=COL["thr"], markeredgecolor="black", markeredgewidth=0.4, markersize=7.5, label="threshold+repair* (mean)"),
                       Line2D([0], [0], marker="*", color="none", markerfacecolor=COL["topk"], markeredgecolor="black", markeredgewidth=0.4, markersize=7.5, label="size-matched top-K (mean)"),
                       Line2D([0], [0], marker="*", color="none", markerfacecolor=COL["meteor"], markeredgecolor="black", markeredgewidth=0.4, markersize=7.5, label="METEOR (mean)")]
    # Structurally non-overlapping placement: organism legend outside on the right,
    # method legend outside below -- different sides of the axes bbox by construction,
    # then verified programmatically (not just by eye) after the figure is drawn.
    leg1 = ax.legend(handles=org_handles, fontsize=6, frameon=False, loc="upper left", ncol=1,
                      handletextpad=0.3, labelspacing=0.3, bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=method_handles, fontsize=6.2, frameon=False, loc="upper left", ncol=1,
                      handletextpad=0.4, handlelength=1.0, labelspacing=0.4, bbox_to_anchor=(1.0, 0.56))
    ax._panel_b_note = ax.text(1.01, 0.08, "*threshold+repair precision:\npooled mean only, no\nper-organism value deposited",
            transform=ax.transAxes, fontsize=5.8, va="top", ha="left", style="italic", color="dimgray")
    ax._legends_to_check = (leg1, leg2)
    pooled_meteor = f"{368}/{612}"; pooled_repair = f"{293}/{612}"
    return dict(pooled_meteor=pooled_meteor, pooled_repair=pooled_repair, marginal_precision=round(marg_prec, 3),
                meteor_recall_mean=round(float(np.mean(B_meteor_recall)), 3),
                meteor_precision_mean=round(float(np.mean(B_meteor_prec)), 3),
                topk_recall_mean=round(float(np.mean(B_topk_recall)), 3),
                topk_precision_mean=round(float(np.mean(B_topk_prec)), 3),
                baseline_recall_mean=round(float(np.mean(B_baseline_recall)), 3),
                note="main text abstract/Sec3.2 give METEOR recall as 0.604 (Supplementary Table S19, Wilson-CI table); "
                     "this per-organism panel (Supplementary Table S18) uses 0.606 -- the two supplementary tables "
                     "report slightly different mean recall (0.604 vs 0.606) for what the text describes as the same "
                     "quantity; both values and the discrepancy are reported here rather than silently reconciled.")

# ============================================================== Panel C: perturbation stability (decoy)
# Source: results/recovery_4arm/*.json (108 genomes; produced by eval/recovery_ablation.py).
# Each file: {"true": {...}, "twostage": {...}, "uniform": {...}, "shuffled": {...}},
# each arm dict has "decoy" (off-reference reaction count) and "recall" (of deleted
# reactions) per genome -- both directly per-genome, no fallback to a pooled/main-text
# number needed.
DECOY_DIR = os.path.join(ROOT, "results/recovery_4arm")
DECOY_ARM_ORDER = [("true", "Evidence\ncost"), ("twostage", "Two-stage"),
                    ("uniform", "Uniform\ncost"), ("shuffled", "Shuffled\ncost")]
DECOY_COLORS = {"true": COL["meteor"], "twostage": COL["repair"], "uniform": COL["uniform"], "shuffled": COL["shuffled"]}

def panel_c(ax):
    decoy_files = sorted(glob.glob(os.path.join(DECOY_DIR, "*.json")))
    decoy_recs = [json.load(open(f)) for f in decoy_files]
    verify = {}
    positions = [1.0 + 1.5 * i for i in range(len(DECOY_ARM_ORDER))]
    box_data = []
    recall_raw = {}
    for pos, (arm, label) in zip(positions, DECOY_ARM_ORDER):
        decoy = np.array([r[arm]["decoy"] for r in decoy_recs], dtype=float)
        recall = np.array([r[arm]["recall"] for r in decoy_recs], dtype=float)
        box_data.append(decoy)
        recall_raw[arm] = float(recall.mean())
        c = DECOY_COLORS[arm]
        jit = pos + (rng.random(len(decoy)) - 0.5) * 0.32
        ax.scatter(jit, decoy, s=4, color=c, alpha=0.35, edgecolors="none", zorder=2)
        ax.scatter([pos], [decoy.mean()], marker="D", s=26, color=c, edgecolors="black", linewidths=0.5, zorder=4)
        verify[label.replace(chr(10), " ")] = dict(decoy_mean=round(float(decoy.mean()), 1), decoy_sd=round(float(decoy.std()), 1),
                                                     recall_mean=round(float(recall.mean()), 3), n=len(decoy))
    bp = ax.boxplot(box_data, positions=positions, widths=0.5, showfliers=True,
                     patch_artist=True, whis=1.5, zorder=3,
                     boxprops=dict(facecolor="none", edgecolor="black", linewidth=0.7),
                     medianprops=dict(color="black", linewidth=0.9),
                     whiskerprops=dict(color="black", linewidth=0.7),
                     capprops=dict(color="black", linewidth=0.7),
                     flierprops=dict(marker="o", markersize=2, markerfacecolor="gray", markeredgecolor="none", alpha=0.4))
    ax.set_yscale("log")
    ax.set_xlim(positions[0] - 0.65, positions[-1] + 0.65)
    ax.set_xticks(positions)
    ax.set_xticklabels([lab for _, lab in DECOY_ARM_ORDER], fontsize=6.5)
    ax.set_ylabel("Off-reference reactions\n(vs. unperturbed METEOR set)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    recall_true = verify["Evidence cost"]["recall_mean"]
    recall_two = verify["Two-stage"]["recall_mean"]
    recall_uni = verify["Uniform cost"]["recall_mean"]
    recall_shuf = verify["Shuffled cost"]["recall_mean"]
    # Footnote inside the empty upper-left region (above the three low boxes, left
    # of the shuffled box), in panel b's small grey italic footnote style; the
    # regime names are defined in the caption and text, so only the numbers stay.
    # round from the unrounded means (0.02479 -> 0.02), not from the 3-dp values
    low = [recall_raw[k] for k in ("true", "twostage", "uniform")]
    assert all(f"{r:.2f}" == "0.02" for r in low), low
    caption_c = ax.text(0.03, 0.98,
                         f"Deleted-reaction recall: 0.02 (evidence,\n"
                         f"two-stage, uniform), {recall_shuf:.3f} (shuffled)",
                         transform=ax.transAxes, fontsize=5.8, ha="left", va="top",
                         style="italic", color="dimgray")
    ax._panel_c_caption = caption_c
    verify["_source"] = "results/recovery_4arm/*.json, per-genome decoy AND recall fields both present"
    return verify

# ============================================================== Panel D: ablation decomposition (DeepProZyme)
ARM_ORDER = [("baseline_dpz", "Threshold τ=0.5"), ("abl_skelonly", "Skeleton + medium only"),
             ("abl_uniform", "Uniform penalty"), (None, "No biomass floor"), ("abl_full", "METEOR (full)")]

# Post-solve repair counts, from the same skeleton_ablation_summary.json already
# cited for "No skeleton: MILP infeasible in 108/108 genomes". Section 3.4 states
# "with the skeleton and medium alone it returns growing sets of 1,985 reactions
# with zero dead ends, all needing post-solve repair" -- that "all" (108/108) was
# not visible anywhere in this panel; added to the affected arms' labels.
_SKEL_ABL_SUMMARY = json.load(open(os.path.join(ROOT, "results/skeleton_ablation_summary.json")))
REPAIR_INFO = {
    "Skeleton + medium only": f"{_SKEL_ABL_SUMMARY['skelonly']['repaired']}/{_SKEL_ABL_SUMMARY['skelonly']['n']} repaired",
    "METEOR (full)": f"{_SKEL_ABL_SUMMARY['full']['repaired']}/{_SKEL_ABL_SUMMARY['full']['n']} repaired",
}

def panel_d(ax):
    verify = {}
    pts = {}
    for arm, label in ARM_ORDER:
        if arm is None:
            x_mean, x_sd = 3077.0, None
            y_mean, y_sd = None, None
            grow_frac = "0/108"
            filled = False
            verify[label] = dict(n_selected_mean="3077 (main text only, not independently verified in a data file)",
                                  deadends_mean="DATA MISSING", growth=grow_frac)
        else:
            nsel = arm_field(records, arm, "n_selected")
            de = arm_field(records, arm, "deadends")
            grow = arm_field(records, arm, "fba_growth")
            x_mean, x_sd = float(nsel.mean()), float(nsel.std())
            y_mean, y_sd = float(de.mean()), float(de.std())
            n_growing = int((grow >= 0.05).sum())
            grow_frac = f"{n_growing}/{len(records)}"
            filled = True
            verify[label] = dict(n_selected_mean=round(x_mean, 1), n_selected_sd=round(x_sd, 1),
                                  deadends_mean=round(y_mean, 2), deadends_sd=round(y_sd, 2), growth=grow_frac, n=len(records))
            if label in REPAIR_INFO:
                verify[label]["repair_note"] = REPAIR_INFO[label]
        arm_color = {"baseline_dpz": COL["thr"], "abl_skelonly": COL["skel"], "abl_uniform": COL["uniform"],
                     "abl_full": COL["meteor"]}.get(arm, "#888888")
        marker_kw = dict(marker="o", ms=6, mfc=arm_color, mec="black", mew=0.5)
        if y_mean is not None:
            # Error-bar whiskers in the arm's own (saturated) color, not a shared gray,
            # so they read as "this arm's uncertainty" rather than generic decoration
            # that blends with the leader lines and the annotation arrow.
            ax.errorbar([x_mean], [y_mean], xerr=[x_sd] if x_sd else None, yerr=[y_sd] if y_sd else None,
                        fmt="none", ecolor=arm_color, elinewidth=0.9, capsize=3, capthick=0.9,
                        alpha=0.75, zorder=2)
            ax.plot([x_mean], [y_mean], linestyle="none", zorder=4, **marker_kw)
        # y_mean is None only for "No biomass floor": its dead-end count was never
        # measured (no per-genome or aggregate data file found anywhere; only its
        # n_selected and growth outcome are stated in the manuscript prose). It is
        # therefore NOT plotted as a point at any y-coordinate at all -- there is no
        # honest y-value to put it at, and a previous version silently defaulted the
        # missing value to y=0.0 and plotted a real (if hollow) marker there,
        # visually indistinguishable from "Skeleton + medium only"'s genuine,
        # independently verified 0.0 +- 0.0 dead-end count. See the vertical dashed
        # line drawn below instead, which shows only the one value that IS real
        # (n_selected=3077, from manuscript text) without implying a y-value.
        pts[label] = (x_mean, y_mean, grow_frac, filled)

    ax.set_ylim(-0.3, 17)
    ax.set_xlim(1500, 4300)

    # "No biomass floor": shown as a full-height vertical dashed line at its one
    # real value (mean selected reactions, from manuscript prose) instead of a
    # point, since no dead-end y-value exists to plot it at.
    x_nofloor, _, nofloor_grow, _ = pts["No biomass floor"]
    ax.axvline(x_nofloor, color="#888888", lw=1.0, linestyle=(0, (4, 2)), zorder=3)

    # Deterministic, non-overlapping label placement: fixed stack of label boxes in
    # axes-fraction space on the right margin, each connected to its data point by a
    # thin leader line (arrowprops). Vertical spacing (0.86 down to 0.06) guarantees
    # no two label boxes can intersect regardless of data position; verified
    # programmatically after render (see check_no_overlap in the assembly step).
    label_order = ["Threshold τ=0.5", "METEOR (full)", "Uniform penalty", "No biomass floor", "Skeleton + medium only"]
    y_stack = [0.95, 0.78, 0.61, 0.44, 0.27]
    ax._panel_d_annotations = []   # TEXT-only artists, checked for overlap
    ax._panel_d_leaders = []       # leader LINES, drawn separately so crossing lines
                                    # (a normal, accepted pattern for stacked labels)
                                    # are never mistaken for a text-overlap collision
    for label, y_ax in zip(label_order, y_stack):
        x_mean, y_plot, grow_frac, filled = pts[label]
        if y_plot is None:
            # No real y-value exists (see the axvline above) -- point the leader
            # line at a point ON that dashed line (its top) rather than inventing
            # a y-coordinate, and say explicitly that the dead-end value is
            # unknown, not just abbreviate it.
            y_plot = ax.get_ylim()[1] * 0.93
            text = f"{label}\n({grow_frac}, dead-ends unknown)"
        else:
            repair_note = f", {REPAIR_INFO[label]}" if label in REPAIR_INFO else ""
            text = f"{label}\n({grow_frac}{'' if filled else ', d.e. N/A'}{repair_note})"
        leader = ax.annotate("", xy=(x_mean, y_plot), xycoords="data",
                              xytext=(1.03, y_ax), textcoords="axes fraction",
                              arrowprops=dict(arrowstyle="-", color="#B0B0B0", lw=0.6,
                                               linestyle=(0, (1, 1.5)),  # fine dotted: a "pointer", not data
                                               shrinkA=2, shrinkB=2,
                                               connectionstyle="arc3,rad=0.0"))
        ax._panel_d_leaders.append(leader)
        txt = ax.text(1.035, y_ax, text, transform=ax.transAxes, fontsize=6.8, ha="left", va="center")
        ax._panel_d_annotations.append(txt)

    x_skel, y_skel, *_ = pts["Skeleton + medium only"]
    x_full, y_full, *_ = pts["METEOR (full)"]
    n_diff = round(x_full - x_skel)
    # "Skeleton + medium only" and "METEOR (full)" are two independent ablation
    # configurations compared on the same axes, not a before/after or causal
    # sequence -- an arrow between them would visually imply a process or
    # transition that does not exist. Using a dimension-line-style bracket
    # instead (two tick marks + a connecting span, labelled with the delta)
    # states the same fact -- the two configurations' mean reaction counts
    # differ by N -- without implying directionality. Drawn in a neutral
    # technical-annotation color (not a data color, not the arrow orange this
    # replaced). Placed in the genuine empty band between the low cluster
    # (METEOR/uniform/skeleton, all <=3.3) and the threshold point's error bar
    # (bottom ~5.15) -- NOT below y=0, which previously forced the axis into
    # negative territory that a dead-end *count* (bounded at zero) can never
    # actually take.
    DIM_COLOR = "#333333"
    dim_y = 3.9
    tick_h = 0.3
    for x_end in (x_skel, x_full):
        ax.plot([x_end, x_end], [dim_y - tick_h, dim_y + tick_h], color=DIM_COLOR, lw=0.8, zorder=6)
    ax.plot([x_skel, x_full], [dim_y, dim_y], color=DIM_COLOR, lw=0.8, zorder=6)
    dim_ann = ax.text((x_skel + x_full) / 2, dim_y + tick_h + 0.15,
                       f"+{n_diff} evidence-selected\n(skeleton-only vs. full)",
                       fontsize=6.2, ha="center", va="bottom", color=DIM_COLOR, zorder=7,
                       bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9))
    ax._panel_d_annotations.append(dim_ann)

    # "No skeleton" has no data point (the MILP never returns a solution), so it is
    # a label-column entry without a leader line.
    noskel = _SKEL_ABL_SUMMARY["noskel"]
    noskel_ann = ax.text(1.035, 0.10,
                          f"No skeleton: MILP infeasible\n({noskel['solved']}/{noskel['n']} solved)",
                          transform=ax.transAxes, fontsize=6.8, ha="left", va="center")
    ax._panel_d_annotations.append(noskel_ann)
    # Subset meant to stay strictly within this panel's own width (unlike the
    # right-margin label stack above, which is deliberately placed outside the axes).
    ax._panel_d_within_bounds = [("dim_bracket_label", dim_ann)]

    ax.set_xlabel("Selected reactions per genome", fontsize=8)
    ax.set_ylabel("Dead-end metabolites per genome", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    return verify

# ============================================================== overlap checker
def boxes_intersect(b1, b2):
    return not (b1.x1 <= b2.x0 or b2.x1 <= b1.x0 or b1.y1 <= b2.y0 or b2.y1 <= b1.y0)

def check_no_overlap(fig, boxable_objects, label):
    """boxable_objects: list of matplotlib artists with get_window_extent(). Draws the
    canvas first so extents are real (not stale/zero), then checks every pair for a
    pixel-space bounding-box intersection. Returns (ok: bool, pairs: list of colliding
    label-pairs)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for name, artist in boxable_objects:
        try:
            bb = artist.get_window_extent(renderer=renderer)
        except Exception as e:
            print(f"[overlap-check:{label}] could not get extent for {name}: {e}")
            continue
        boxes.append((name, bb))
    collisions = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            n1, b1 = boxes[i]; n2, b2 = boxes[j]
            if boxes_intersect(b1, b2):
                collisions.append((n1, n2))
    ok = len(collisions) == 0
    print(f"[overlap-check:{label}] {len(boxes)} boxes checked, {len(collisions)} collisions: {collisions if not ok else 'none'}")
    return ok, collisions

def check_within_axes(fig, ax, boxable_objects, label, x_margin_px=2, y_margin_px=2, check_y=False):
    """Checks that every given artist's rendered bbox stays within its OWN axes'
    bbox (in figure pixel space), not just clear of other artists. This is the
    check that would have caught panel c's title overflowing its own panel width
    in the combined 2x2 figure -- check_no_overlap alone only catches artists
    colliding with each other, not an artist spilling out of its intended panel."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(renderer=renderer)
    violations = []
    for name, artist in boxable_objects:
        try:
            bb = artist.get_window_extent(renderer=renderer)
        except Exception as e:
            print(f"[bounds-check:{label}] could not get extent for {name}: {e}")
            continue
        if bb.x0 < ax_bb.x0 - x_margin_px or bb.x1 > ax_bb.x1 + x_margin_px:
            violations.append((name, "x-overflow", round(bb.x0 - ax_bb.x0, 1), round(bb.x1 - ax_bb.x1, 1)))
        if check_y and (bb.y0 < ax_bb.y0 - y_margin_px or bb.y1 > ax_bb.y1 + y_margin_px):
            violations.append((name, "y-overflow", round(bb.y0 - ax_bb.y0, 1), round(bb.y1 - ax_bb.y1, 1)))
    ok = len(violations) == 0
    print(f"[bounds-check:{label}] {len(boxable_objects)} boxes checked against own-axes bounds, "
          f"{len(violations)} overflow: {violations if not ok else 'none'}")
    return ok, violations

# ============================================================== assemble figure
# Height trimmed after the panel c/d footnotes moved off the bottom band;
# axes heights stay at the previous size (checked in the print below).
FIG2_H = 5.38
fig, axes = plt.subplots(2, 2, figsize=(FIGW, FIG2_H))
panel_fns = [panel_a, panel_b, panel_c, panel_d]
panel_labels = ["a", "b", "c", "d"]
verify_all = {}
axes_by_label = {}
panel_label_artists = []
for ax, fn, lab in zip(axes.flat, panel_fns, panel_labels):
    v = fn(ax)
    verify_all[lab] = v
    lbl_artist = ax.text(-0.12, 1.05, lab, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    panel_label_artists.append((f"panel_label_{lab}", lbl_artist))
    axes_by_label[lab] = ax

fig.tight_layout()

def ticklabel_boxable(ax):
    """matplotlib's own auto-generated tick Text objects -- the exact artist class
    the first overlap-check pass omitted (it only checked custom annotations),
    which is where the panel-c 2-line tick label collision actually lived. Ticks
    the locator placed outside the current axis limits (never actually rendered
    on screen, just present as an off-canvas Text with real window-extent
    coordinates) are excluded, since they are not a visible collision."""
    out = []
    xlo, xhi = sorted(ax.get_xlim())
    ylo, yhi = sorted(ax.get_ylim())
    for i, t in enumerate(ax.get_xticklabels()):
        if not t.get_text().strip() or not t.get_visible():
            continue
        xpos = t.get_position()[0]
        if not (xlo <= xpos <= xhi):
            continue
        out.append((f"xtick_{i}:{t.get_text()!r}", t))
    for i, t in enumerate(ax.get_yticklabels()):
        if not t.get_text().strip() or not t.get_visible():
            continue
        ypos = t.get_position()[1]
        if not (ylo <= ypos <= yhi):
            continue
        out.append((f"ytick_{i}:{t.get_text()!r}", t))
    return out

overlap_report = {}
ax_b = axes_by_label["b"]
leg1, leg2 = ax_b._legends_to_check
ok_b, coll_b = check_no_overlap(fig, [("species_legend", leg1), ("method_legend", leg2), ("footnote", ax_b._panel_b_note)], "panel_b (combined fig)")
overlap_report["panel_b_combined"] = dict(ok=ok_b, collisions=coll_b)

ax_d = axes_by_label["d"]
d_boxable = [(f"annotation_{i}", a) for i, a in enumerate(ax_d._panel_d_annotations)]
ok_d, coll_d = check_no_overlap(fig, d_boxable, "panel_d (combined fig)")
overlap_report["panel_d_combined"] = dict(ok=ok_d, collisions=coll_d)

# Tick-label check, every panel (this is the check the coordinator's report found
# missing -- matplotlib's auto-generated xtick/ytick Text objects, not just the
# hand-placed custom annotations above).
for lab in panel_labels:
    ax_i = axes_by_label[lab]
    tick_boxable = ticklabel_boxable(ax_i)
    ok_t, coll_t = check_no_overlap(fig, tick_boxable, f"panel_{lab} tick labels (combined fig)")
    overlap_report[f"panel_{lab}_ticklabels_combined"] = dict(ok=ok_t, collisions=coll_t)

# Own-axes containment check for the wide supplementary annotations (this is the
# check that would have caught panel c's title overflowing into panel d's space).
ax_c = axes_by_label["c"]
ok_c_bounds, viol_c = check_within_axes(fig, ax_c, [("caption_c", ax_c._panel_c_caption)], "panel_c caption (combined fig)", check_y=True)
overlap_report["panel_c_caption_bounds_combined"] = dict(ok=ok_c_bounds, violations=viol_c)

def _densify(xy, n=25):
    if len(xy) < 2:
        return xy
    t = np.linspace(0, 1, n)[:, None]
    return np.vstack([xy[k] + t * (xy[k + 1] - xy[k]) for k in range(len(xy) - 1)])

def check_text_clear_of_data(fig, ax, text, label, pad_px=2):
    """An in-panel note or legend (text AND handles, via its window extent) must not
    cover plotted data: scatter points, sampled line and error-bar segments, and
    box/whisker/cap/flier vertices, in display space."""
    fig.canvas.draw()
    bb = text.get_window_extent(renderer=fig.canvas.get_renderer()).padded(pad_px)
    pts = []
    for c in ax.collections:
        segs = c.get_segments() if hasattr(c, "get_segments") else []
        if len(segs):
            pts += [_densify(c.get_transform().transform(np.asarray(sg))) for sg in segs if len(sg)]
        else:
            pts.append(c.get_offset_transform().transform(c.get_offsets()))
    # lines are densified so a segment crossing the text is caught even when
    # neither endpoint lies under it (panel a's paired-genome lines)
    for ln in ax.lines:
        xy = ln.get_transform().transform(ln.get_xydata())
        if len(xy) >= 2:
            t = np.linspace(0, 1, 25)[:, None]
            xy = np.vstack([xy[k] + t * (xy[k + 1] - xy[k]) for k in range(len(xy) - 1)])
        pts.append(xy)
    pts += [pa.get_transform().transform(pa.get_path().vertices) for pa in ax.patches]
    pts = np.vstack([q for q in pts if len(q)])
    hits = int(((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum())
    print(f"[data-clear:{label}] {len(pts)} data vertices checked, {hits} inside bbox")
    return hits == 0, hits

ok_a_bounds, viol_a = check_within_axes(fig, axes_by_label["a"], [("growth_note", axes_by_label["a"]._panel_a_note)], "panel_a note (combined fig)", check_y=True)
overlap_report["panel_a_note_bounds_combined"] = dict(ok=ok_a_bounds, violations=viol_a)
ok_a_data, hits_a = check_text_clear_of_data(fig, axes_by_label["a"], axes_by_label["a"]._panel_a_note, "panel_a note (combined fig)")
overlap_report["panel_a_note_clear_of_data_combined"] = dict(ok=ok_a_data, hits=hits_a)
for _lab, _leg in [("a", axes_by_label["a"]._panel_a_legend)] + \
                  [("b", lg) for lg in axes_by_label["b"]._legends_to_check]:
    _ok, _hits = check_text_clear_of_data(fig, axes_by_label[_lab], _leg, f"panel_{_lab} legend vs data (combined fig)")
    overlap_report.setdefault("legend_clear_of_data_combined", []).append(dict(panel=_lab, ok=_ok, hits=_hits))
ok_al, coll_al = check_no_overlap(fig, [("growth_note", axes_by_label["a"]._panel_a_note),
                                        ("colour_legend", axes_by_label["a"]._panel_a_legend)], "panel_a note vs legend (combined fig)")
overlap_report["panel_a_note_vs_legend_combined"] = dict(ok=ok_al, collisions=coll_al)
ok_c_data, hits_c = check_text_clear_of_data(fig, ax_c, ax_c._panel_c_caption, "panel_c footnote (combined fig)")
overlap_report["panel_c_footnote_clear_of_data_combined"] = dict(ok=ok_c_data, hits=hits_c)

ok_d_bounds, viol_d = check_within_axes(fig, ax_d, ax_d._panel_d_within_bounds, "panel_d annotations (combined fig)")
overlap_report["panel_d_annotation_bounds_combined"] = dict(ok=ok_d_bounds, violations=viol_d)

# Cross-panel check: the a/b/c/d panel-label letters against every panel's own
# title/caption text. This is the check that would have caught panel a's title
# running into panel b's "b" label -- a collision between two DIFFERENT panels'
# artists, which none of the per-panel checks above are positioned to see (each
# only checks artists *within* one panel against each other or against that same
# panel's own axes bounds).
ax_a = axes_by_label["a"]
cross_panel_boxable = list(panel_label_artists) + [("panel_a_note", ax_a._panel_a_note), ("panel_b_note", ax_b._panel_b_note),
                                                     ("panel_c_caption", ax_c._panel_c_caption)] + \
                      ax_d._panel_d_within_bounds
ok_cross, coll_cross = check_no_overlap(fig, cross_panel_boxable, "cross-panel titles/captions vs panel labels (combined fig)")
overlap_report["cross_panel_titles_vs_labels_combined"] = dict(ok=ok_cross, collisions=coll_cross)

for _lab in panel_labels:
    _bb = axes_by_label[_lab].get_position()
    print(f"[axes] {_lab}: {_bb.height * FIG2_H:.3f} in tall")
os.makedirs(FIGDIR, exist_ok=True)
fig.savefig(os.path.join(FIGDIR, "fig2_results.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIGDIR, "fig2_results.svg"), bbox_inches="tight")
fig.savefig(os.path.join(FIGDIR, "fig2_results.png"), dpi=300, bbox_inches="tight")

# individual panel exports (checked separately -- different aspect ratio than the
# combined 2x2 grid, so overlap outcome is NOT assumed to carry over)
for ax, fn, lab in zip(axes.flat, panel_fns, panel_labels):
    fig_i, ax_i = plt.subplots(figsize=(FIGW / 2, FIGH / 2))
    fn(ax_i)
    ax_i.text(-0.12, 1.05, lab, transform=ax_i.transAxes, fontsize=10, fontweight="bold", va="top")
    fig_i.tight_layout()
    if lab == "b":
        leg1_i, leg2_i = ax_i._legends_to_check
        ok_bi, coll_bi = check_no_overlap(fig_i, [("species_legend", leg1_i), ("method_legend", leg2_i), ("footnote", ax_i._panel_b_note)], "panel_b (standalone fig2b.pdf)")
        overlap_report["panel_b_standalone"] = dict(ok=ok_bi, collisions=coll_bi)
    if lab == "d":
        d_boxable_i = [(f"annotation_{i}", a) for i, a in enumerate(ax_i._panel_d_annotations)]
        ok_di, coll_di = check_no_overlap(fig_i, d_boxable_i, "panel_d (standalone fig2d.pdf)")
        overlap_report["panel_d_standalone"] = dict(ok=ok_di, collisions=coll_di)
    if lab == "a":
        check_text_clear_of_data(fig_i, ax_i, ax_i._panel_a_legend, "panel_a legend vs data (standalone fig2a.pdf)")
        check_no_overlap(fig_i, [("growth_note", ax_i._panel_a_note), ("colour_legend", ax_i._panel_a_legend)], "panel_a note vs legend (standalone fig2a.pdf)")
    if lab == "b":
        for lg in ax_i._legends_to_check:
            check_text_clear_of_data(fig_i, ax_i, lg, "panel_b legend vs data (standalone fig2b.pdf)")
    tick_boxable_i = ticklabel_boxable(ax_i)
    ok_ti, coll_ti = check_no_overlap(fig_i, tick_boxable_i, f"panel_{lab} tick labels (standalone fig2{lab}.pdf)")
    overlap_report[f"panel_{lab}_ticklabels_standalone"] = dict(ok=ok_ti, collisions=coll_ti)
    fig_i.savefig(os.path.join(FIGDIR, f"fig2{lab}.pdf"), bbox_inches="tight")
    plt.close(fig_i)

verify_all["_overlap_check"] = overlap_report
json.dump(verify_all, open(os.path.join(FIGDIR, "fig2_verify_raw.json"), "w"), indent=1, default=str)
print("[done] wrote figures/fig2_results.{pdf,svg,png}, fig2a-d.pdf, fig2_verify_raw.json")
print(json.dumps(verify_all, indent=1, default=str))
