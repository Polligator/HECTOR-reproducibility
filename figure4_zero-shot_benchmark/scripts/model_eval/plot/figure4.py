"""Figure 4 — composed master figure (5 panels a–e).

  a — zero-shot benchmark construction
  b — native-decoder comparison (HECTOR, OnClass, scCello)
  c — zero-shot accuracy bars (3 metrics)
  d — hop explainer inset on the hop curve
  e — per-case concentric-hop-ring overlays (4 cases, one row)

Reuses each panel module's draw helpers; reads its inputs from result/ and
result/plot_cache/. Call ``compose_figure_4()`` after the benchmark has written
its prediction CSVs.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from .paths import RESULT_DIR, DATA_DIR, PLOT_CACHE_DIR, ONTOLOGY_OBO, PROJECT_ROOT
from . import style
from .panels.accuracy import (
    draw_panel_accuracy,
    BRACKET_GAP_PT, BRACKET_TEXT_PT, BRACKET_TICK_PT,
)
from .panels.hop import draw_curve, draw_explainer
from .panels.per_case import (
    draw_case, modal_pred, load_tables as load_per_case_tables, CASES,
    SERIES as PER_CASE_SERIES, TRUTH_FILL, TRUTH_EDGE,
    HECTOR_KEYS as PER_CASE_HECTOR_KEYS, HECTOR_HEADING,
)

# Cell-Ontology loader lives at the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Hierarchical_metrics.cell_ontology_metrics import load_cell_ontology  # noqa: E402

mpl.rcParams.update(style.RC)
HOP_CACHE = PLOT_CACHE_DIR / "hop_data.pkl"

# How far the b|c row is inset from the left margin the other rows use, as a
# fraction of page width. See the comment where it is applied.
ROW_BC_INSET = 0.026


# --- panel b: native decoders -------------------------------------------
# HECTOR, OnClass and scCello each answering with their own decoder, on the
# strict 31 cell types unseen to all three and in all three native
# vocabularies: does HECTOR still lead with every model given its own head,
# rather than the shared landmark read-out panel c and the accuracy bars use.
#
# Run offline through run_training_audit.py, run_onclass_native.py,
# run_sccello_native.py and run_hector_native_strict.py, summarized by
# run_comparison_summary.py. The 66-type comparison is not drawn: those types
# are not all zero-shot for OnClass and scCello.
NATIVE_MODELS = ["HECTOR", "OnClass", "scCello"]
NATIVE_COLOR = {"HECTOR": style.HECTOR_NATIVE,
                "OnClass": style.MODEL_COLOR["OnClass"],
                "scCello": style.MODEL_COLOR["scCello"]}

# (subset, model, method) — only the strict common zero-shot rows are
# publication-facing.
NATIVE_CLOSED_ROWS = [
    ("strict_common31_identical_candidates", "HECTOR", "native_strict_common31"),
    ("strict_common31_identical_candidates", "OnClass", "native_released_ensemble_strict_common31"),
    ("strict_common31_identical_candidates", "scCello", "native_checkpoint_strict_common31"),
]
NATIVE_OPEN_ROWS = [
    ("strict_truth_stratum_within_native_task", "HECTOR", "native_open_reference"),
    ("strict_truth_stratum_within_native_task", "OnClass", "native_released_ensemble_open"),
    ("strict_truth_stratum_within_native_task", "scCello", "native_checkpoint_open"),
]
# The same three metrics panel c reports; each gets its own scale, like panel c.
NATIVE_METRICS = [("accuracy", "Accuracy", 0.46),
                  ("macro_f1", "Macro F1", 0.42),
                  ("macro_hierarchical_f1", "Hierarchical macro F1", 0.78)]
NATIVE_BLOCKS = [("closed set", "native_zero_shot_comparison_summary.csv", NATIVE_CLOSED_ROWS),
                 ("open vocabulary", "native_open_vocabulary_primary_summary.csv", NATIVE_OPEN_ROWS)]


def _native_rows(csv_name, wanted) -> pd.DataFrame:
    table = pd.read_csv(DATA_DIR / csv_name)
    picked = []
    for subset, model, method in wanted:
        match = table.loc[(table["subset"] == subset) & (table["model"] == model)
                          & (table["method"] == method)]
        picked.append(match.iloc[0])
    return pd.DataFrame(picked)


def _native_table() -> list:
    """One row per (task, model), in the order they are drawn top to bottom."""
    rows = []
    for block, csv_name, wanted in NATIVE_BLOCKS:
        table = _native_rows(csv_name, wanted)
        for model in NATIVE_MODELS:
            rows.append((block, model, table.loc[table["model"] == model].iloc[0]))
    return rows


def _draw_native_metric(ax, rows, metric: str, xmax: float, *, show_ylabels: bool,
                        title: str) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, color="#BBBBBB", alpha=0.6, zorder=0)

    # A gap between the two tasks, so the six rows read as two groups of three.
    positions = [i + (0.7 if block == NATIVE_BLOCKS[1][0] else 0.0)
                 for i, (block, _m, _r) in enumerate(rows)]
    for y, (_block, model, row) in zip(positions, rows):
        value = float(row[metric])
        is_hector = model == "HECTOR"
        ax.barh(y, value, height=0.72, color=NATIVE_COLOR[model],
                edgecolor="black" if is_hector else "none",
                linewidth=0.4 if is_hector else 0.0, zorder=2, clip_on=False)
        ax.text(value + xmax * 0.03, y, f"{value:.3f}", va="center", ha="left",
                fontsize=style.MAIN_SECONDARY, color="#222222", zorder=3,
                clip_on=False)

    ax.set_ylim(max(positions) + 0.6, min(positions) - 0.6)
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0.0, xmax + 0.001, 0.2))
    ax.set_yticks(positions)
    if show_ylabels:
        ax.set_yticklabels([model for _b, model, _r in rows],
                           fontsize=style.MAIN_SECONDARY)
        for tick in ax.get_yticklabels():
            if tick.get_text() == "HECTOR":
                tick.set_color(style.HECTOR_NATIVE)
                tick.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="both", width=0.5, length=2.2, pad=1.8)
    ax.tick_params(axis="x", labelsize=style.MAIN_SECONDARY)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)
    ax.set_title(title, fontsize=style.MAIN_TEXT, fontweight="bold", pad=3)
    return positions


def draw_panel_b(fig, host_spec) -> None:
    """Panel b — native decoders, drawn as panel c is: one chart per metric."""
    rows = _native_table()
    charts = host_spec.subgridspec(1, len(NATIVE_METRICS), wspace=0.22)
    axes = []
    for i, (metric, title, xmax) in enumerate(NATIVE_METRICS):
        ax = fig.add_subplot(charts[0, i])
        positions = _draw_native_metric(ax, rows, metric, xmax,
                                        show_ylabels=(i == 0), title=title)
        axes.append(ax)

    # Bracket line is placed by measuring where the row labels actually end,
    # matching panel c's bracket geometry rather than a guessed offset.
    first = axes[0]
    fig.canvas.draw()
    label_left = [tl.get_window_extent().x0 for tl in first.get_yticklabels()
                  if tl.get_text()]
    x0_px = min(label_left)
    px_per_pt = fig.dpi / 72.0
    to_axes = first.transAxes.inverted()
    trans = mpl.transforms.blended_transform_factory(first.transAxes, first.transData)
    line_px = x0_px - BRACKET_GAP_PT * px_per_pt
    x_line = to_axes.transform((line_px, 0))[0]
    x_tick = to_axes.transform((line_px + BRACKET_TICK_PT * px_per_pt, 0))[0]
    x_text = to_axes.transform((line_px - BRACKET_TEXT_PT * px_per_pt, 0))[0]
    for block, *_rest in NATIVE_BLOCKS:
        ys = [p for p, (b, _m, _r) in zip(positions, rows) if b == block]
        y0, y1 = min(ys) - 0.42, max(ys) + 0.42
        first.plot([x_line, x_line], [y0, y1], transform=trans, color="#666666",
                   lw=0.7, clip_on=False, zorder=5)
        for y in (y0, y1):
            first.plot([x_line, x_tick], [y, y], transform=trans, color="#666666",
                       lw=0.7, clip_on=False, zorder=5)
        first.text(x_text, (y0 + y1) / 2.0, block, transform=trans, rotation=90,
                   ha="right", va="center", fontsize=style.MAIN_SECONDARY,
                   color="#666666")


def draw_panel_a(fig, host_spec) -> None:
    """Panel a — construction of the two zero-shot evaluation settings.

    The former landmark-embedding UMAP row duplicated Supplementary Figure 9.
    This replacement gives the main figure the information needed to interpret
    the benchmark: how the held-out terms were selected, how many cells entered,
    and why the closed and open evaluations answer different questions.
    """
    ax = fig.add_subplot(host_spec)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    candidate_table = pd.read_csv(RESULT_DIR / "data" / "candidate_label_table.csv")
    landmark_table = pd.read_csv(RESULT_DIR / "data" / "bridge_label_table.csv")
    n_query_types = candidate_table["cell_type_id"].nunique()
    n_query_cells = int(candidate_table["n_cells"].sum())
    cells_per_type = candidate_table["n_cells"].drop_duplicates().tolist()
    if len(cells_per_type) != 1:
        raise ValueError("Panel a expects one fixed query-cell count per held-out type")
    n_cells_per_type = int(cells_per_type[0])
    n_landmark_types = landmark_table["cell_type_id"].nunique()

    # From the human HECTOR checkpoint's ontology/classes (650, training) and
    # ontology/full_classes (1,407, inference); kept explicit so redrawing does
    # not need to load the checkpoint.
    n_training_types = 650
    n_predictable_terms = 1407

    def add_box(x, y, width, height, face, edge, linewidth=1.0):
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.006,rounding_size=0.015",
            transform=ax.transAxes, facecolor=face, edgecolor=edge,
            linewidth=linewidth, clip_on=False,
        )
        ax.add_patch(patch)

    # Main path reads left to right; the two evaluation settings stack at the
    # end so the panel stays horizontal while keeping them visually distinct.
    add_box(0.010, 0.24, 0.175, 0.58, "#F1F7FC", "#377EB8", linewidth=1.1)
    ax.text(0.0975, 0.70, "HECTOR prediction space", transform=ax.transAxes,
            ha="center", va="center", fontsize=style.MAIN_TEXT, fontweight="bold")
    ax.text(0.0975, 0.51, f"{n_predictable_terms:,} terms", transform=ax.transAxes,
            ha="center", va="center", fontsize=style.MAIN_EMPHASIS, fontweight="bold")
    ax.text(0.0975, 0.34, "human Cell Ontology", transform=ax.transAxes,
            ha="center", va="center", fontsize=style.MAIN_SECONDARY, color="#666666")

    add_box(0.230, 0.17, 0.275, 0.72, "#F7F7F7", "#AFAFAF")
    ax.text(0.3675, 0.76, "Prespecified selection", transform=ax.transAxes,
            ha="center", va="center", fontsize=style.MAIN_TEXT, fontweight="bold")
    ax.plot([0.250, 0.485], [0.68, 0.68], transform=ax.transAxes,
            color="#A0A0A0", linewidth=0.6, clip_on=False)
    ax.text(0.250, 0.57, f"exclude {n_training_types:,} training types", transform=ax.transAxes,
            ha="left", va="center", fontsize=style.MAIN_SECONDARY)
    ax.text(0.250, 0.44, f"exclude {n_landmark_types:,} Tabula Sapiens landmark types",
            transform=ax.transAxes, ha="left", va="center", fontsize=style.MAIN_SECONDARY)
    ax.text(0.250, 0.31, "require >500 primary Census cells", transform=ax.transAxes,
            ha="left", va="center", fontsize=style.MAIN_SECONDARY)

    add_box(0.550, 0.17, 0.205, 0.72, "#FFF6D8", "#D08A00", linewidth=1.2)
    ax.text(0.6525, 0.76, "Held-out query set", transform=ax.transAxes,
            ha="center", va="center", fontsize=style.MAIN_TEXT, fontweight="bold")
    ax.text(0.6525, 0.57, f"{n_query_types:,} cell types", transform=ax.transAxes,
            ha="center", va="center", fontsize=style.MAIN_EMPHASIS, fontweight="bold")
    ax.text(0.6525, 0.43, f"{n_cells_per_type:,} cells per type", transform=ax.transAxes,
            ha="center", va="center", fontsize=style.MAIN_SECONDARY)
    ax.text(0.6525, 0.29, f"{n_query_cells:,} cells; all retained", transform=ax.transAxes,
            ha="center", va="center", fontsize=style.MAIN_SECONDARY, color="#666666")

    add_box(0.805, 0.55, 0.185, 0.36, "#FFF0ED", style.HECTOR_NATIVE_OPEN,
            linewidth=1.1)
    ax.text(0.8975, 0.79, "Open-vocabulary\nlocalization", transform=ax.transAxes,
            ha="center", va="center", fontsize=style.MAIN_TEXT, fontweight="bold", linespacing=1.2)
    ax.text(0.8975, 0.62, f"select among all {n_predictable_terms:,} terms", transform=ax.transAxes,
            ha="center", va="center", fontsize=style.MAIN_SECONDARY)

    add_box(0.805, 0.07, 0.185, 0.36, "#FFF0F0", style.HECTOR_NATIVE,
            linewidth=1.1)
    ax.text(0.8975, 0.31, "Closed-set zero-shot\nnaming", transform=ax.transAxes,
            ha="center", va="center", fontsize=style.MAIN_TEXT, fontweight="bold", linespacing=1.2)
    ax.text(0.8975, 0.14, f"select among {n_query_types:,} held-out terms", transform=ax.transAxes,
            ha="center", va="center", fontsize=style.MAIN_SECONDARY)

    # Arrows on the main path, then one branch to the open evaluation above and
    # one to the closed evaluation below.
    arrow_style = dict(arrowstyle="-|>", color="#333333", lw=1.0, mutation_scale=9)
    ax.annotate("", xy=(0.225, 0.53), xytext=(0.190, 0.53),
                xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=arrow_style)
    ax.annotate("", xy=(0.545, 0.53), xytext=(0.510, 0.53),
                xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=arrow_style)
    ax.plot([0.760, 0.782], [0.53, 0.53], transform=ax.transAxes,
            color="#666666", linewidth=0.9, clip_on=False)
    ax.plot([0.782, 0.782], [0.25, 0.73], transform=ax.transAxes,
            color="#666666", linewidth=0.9, clip_on=False)
    ax.annotate("", xy=(0.800, 0.73), xytext=(0.782, 0.73),
                xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=arrow_style)
    ax.annotate("", xy=(0.800, 0.25), xytext=(0.782, 0.25),
                xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=arrow_style)


def draw_panel_hop(host_ax) -> None:
    """Panel d — hop-curve with the explainer schematic as inset.

    The inset sits in the top-left, over the shaded near-miss band, which is the
    one part of the axes no line reaches: the highest curve there is HECTOR's
    own head at 0.40 and the inset's floor is above it.
    """
    with open(HOP_CACHE, "rb") as f:
        hop_data = pickle.load(f)
    draw_curve(host_ax, hop_data)
    ax_inset = host_ax.inset_axes([0.022, 0.455, 0.375, 0.435])
    ax_inset.patch.set_alpha(0.0)
    draw_explainer(ax_inset, draw_ellipse=False, draw_key=False, compact=True)


def draw_panel_per_case(fig, host_spec, dag, ud, modal_all) -> None:
    """Panel e — model legend strip on top + four compact cases in one row."""
    outer = host_spec.subgridspec(2, 1, height_ratios=[1.8, 20.8], hspace=0.0)
    legend_ax = fig.add_subplot(outer[0, 0])
    legend_ax.axis("off")

    def dot(color, label):
        return Line2D([0], [0], marker="o", linestyle="", markersize=6.5,
                      markerfacecolor=color, markeredgecolor="white",
                      markeredgewidth=0.8, label=label)

    handles = [Line2D([0], [0], marker="*", linestyle="", markersize=11,
                      markerfacecolor=TRUTH_FILL, markeredgecolor=TRUTH_EDGE,
                      markeredgewidth=0.5, label="true label"),
               Line2D([], [], marker="", linestyle="", label=HECTOR_HEADING)]
    handles += [dot(color, disp) for key, disp, _csv, color in PER_CASE_SERIES
                if key in PER_CASE_HECTOR_KEYS]
    handles += [dot(color, disp) for key, disp, _csv, color in PER_CASE_SERIES
                if key not in PER_CASE_HECTOR_KEYS]

    leg_kw = dict(fontsize=style.MAIN_TEXT, handletextpad=0.25, columnspacing=0.8,
                  frameon=False, borderpad=0.0)
    legend = legend_ax.legend(handles=handles, loc="center", ncol=len(handles),
                              bbox_to_anchor=(0.5, 0.5), **leg_kw)
    legend.get_texts()[1].set_fontweight("bold")
    # Gutter wide enough that the four cases' outward-reaching names don't touch.
    inner = outer[1, 0].subgridspec(1, 4, wspace=0.16)
    axes = []
    for i, case in enumerate(CASES):
        ax = fig.add_subplot(inner[0, i])
        draw_case(ax, case, dag, ud, modal_all[case])
        # Near-top anchor: equal-aspect axes would otherwise centre vertically
        # in this taller grid cell, leaving a false gap below the legend.
        ax.set_anchor((0.5, 0.82))
        axes.append(ax)
    return axes


def _drawn_span(ax, renderer):
    """How far what an axes actually draws reaches left and right, in pixels.

    Not ``get_tightbbox``: that returns the axes' own rectangle, which for
    panel e is the whole cell the drawing is centred in, and the rings and names
    fill only part of it. Aligning on that moved panel e 6 mm and still left it
    off, because the ink is not centred inside its own cell.
    """
    spans = []
    for artist in [*ax.texts, *ax.collections, *ax.patches, *ax.lines]:
        try:
            box = artist.get_window_extent(renderer)
        except Exception:
            continue
        if box.width > 0:
            spans.append((box.x0, box.x1))
    if not spans:
        box = ax.get_tightbbox(renderer)
        return box.x0, box.x1
    return min(s[0] for s in spans), max(s[1] for s in spans)


def align_panels(fig, ax_reference, axes_to_move) -> None:
    """Slide a panel sideways until what it draws is centred on another panel.

    Panel e's four drawings are evenly spaced in its row, but their names are
    not: nothing reaches left of the first ring, while "photoreceptor cell"
    reaches well past the last, so the block read as pushed 8 mm to the right of
    panel d. Both are measured as drawn rather than nudged by a fixed amount, so
    they stay aligned if the answers or the names change. The full page is
    retained on export; panel d's drawn centre remains the alignment reference
    because its axis labels make that different from the page centre.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ref = ax_reference.get_tightbbox(renderer)      # a normal axes: box is its ink
    lefts, rights = zip(*[_drawn_span(a, renderer) for a in axes_to_move])
    here = (min(lefts) + max(rights)) / 2.0
    shift = ((ref.x0 + ref.x1) / 2.0 - here) / fig.bbox.width
    for ax in axes_to_move:
        pos = ax.get_position()
        ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])


def compose_figure_4(out_dir=RESULT_DIR) -> Path:
    """Assemble the 5-panel figure using scores from the full evaluation sets."""
    mpl.rcParams.update(style.RC)  # pin style — deterministic regardless of import order
    dag = load_cell_ontology(str(ONTOLOGY_OBO))
    ud = nx.Graph(dag)

    tables = load_per_case_tables()
    modal_all = {case: {k: (modal_pred(tables[k], case) if k in tables else None)
                        for k, *_ in PER_CASE_SERIES}
                 for case in CASES}

    # Panel heights in millimetres, with gaps set individually since what hangs
    # outside each row's axes differs (panel c's titles, panel d's legend,
    # panel e's ring names on every side).
    fig_w, left, right = style.MAIN_WIDTH_IN, 0.075, 0.960
    panel_mm = [23.0, 60.0, 48.0, 60.0]
    gap_mm = [6.0, 14.0, 14.0]
    band_mm = [v for pair in zip(panel_mm, gap_mm + [0.0]) for v in pair][:-1]
    fig_h = style.MAIN_HEIGHT_IN
    page_height_mm = fig_h * 25.4
    top_margin_mm = 6.0
    plot_top = 1 - top_margin_mm / page_height_mm
    plot_bottom = plot_top - sum(band_mm) / page_height_mm

    fig = plt.figure(figsize=(fig_w, fig_h))
    grid = fig.add_gridspec(
        len(band_mm), 1, height_ratios=band_mm, hspace=0.0,
        left=left, right=right, top=plot_top, bottom=plot_bottom,
    )
    gs = {i: grid[2 * i, 0] for i in range(len(panel_mm))}   # panels sit on the even bands

    # Row 0: panel a — full-width benchmark construction schematic
    draw_panel_a(fig, gs[0])

    # Row 1: panel b (native decoders) + panel c (accuracy). Panel b's row is
    # inset from the page edge (ROW_BC_INSET) since it hangs a model-name column
    # and a bracket column outside its own axes.
    bc_box = gs[1].get_position(fig)
    gs_bc = fig.add_gridspec(1, 2, width_ratios=[1.045, 1.155], wspace=0.40,
                             left=bc_box.x0 + ROW_BC_INSET, right=bc_box.x1,
                             top=bc_box.y1, bottom=bc_box.y0)
    # Matching chart heights keep the two comparisons aligned; the lower
    # margin separates their labels from panel d's legend.
    b_inner = gs_bc[0, 0].subgridspec(2, 1, height_ratios=[18, 2.6], hspace=0.0)
    draw_panel_b(fig, b_inner[0, 0])

    c_inner = gs_bc[0, 1].subgridspec(2, 1, height_ratios=[18, 2.6], hspace=0.0)
    ax_c = fig.add_subplot(c_inner[0, 0])
    draw_panel_accuracy(ax_c,
                        ytick_fontsize=style.MAIN_TEXT, value_fontsize=style.MAIN_TEXT,
                        title_fontsize=style.MAIN_TEXT, bracket_fontsize=style.MAIN_SECONDARY,
                        xtick_fontsize=style.MAIN_SECONDARY)

    # Row 2: panel d — hop curve, centred by splitting the unused width evenly.
    d_inner = gs[2].subgridspec(1, 3, width_ratios=[0.10, 1.0, 0.10], wspace=0.0)
    ax_d = fig.add_subplot(d_inner[0, 1])
    draw_panel_hop(ax_d)

    # Row 3: panel e — per-case hop rings, slid sideways to centre on panel d.
    e_axes = draw_panel_per_case(fig, gs[3], dag, ud, modal_all)
    align_panels(fig, ax_d, e_axes)

    # Panel letters at each panel's top-left. a, b, d and e share one letter
    # column; panel b's letter does not move with its row's inset.
    fig.canvas.draw()
    letter_specs = [
        (gs[0], "a", 0.0), (gs_bc[0, 0], "b", ROW_BC_INSET), (gs_bc[0, 1], "c", 0.0),
        (gs[2], "d", 0.0), (gs[3], "e", 0.0),
    ]
    for spec, letter, back in letter_specs:
        bbox = spec.get_position(fig)
        x_letter = max(0.010, bbox.x0 - back - 0.062)
        fig.text(x_letter, bbox.y1 + 2.8 / page_height_mm, letter,
                 fontsize=style.MAIN_PANEL_LETTER, fontweight="bold", va="top")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "figure_4.pdf"
    out_png = out_dir / "figure_4.png"
    # Export the complete page, including white space. Override the shared
    # tight-crop default locally so supplementary exports remain unaffected.
    # PDF text/lines stay vector; both formats use the same physical canvas.
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(out_pdf, dpi=style.MAIN_EXPORT_DPI, bbox_inches=None,
                    facecolor="white", transparent=False)
        fig.savefig(out_png, dpi=style.MAIN_EXPORT_DPI, bbox_inches=None,
                    facecolor="white", transparent=False)
    plt.close(fig)
    return out_pdf
