"""Figure 4 — Supplementary Figure 10: per-held-out-type prediction breakdown.

Reads result/per_held_out_type_accuracy.csv; writes supplementary_figure_10_per_type.{pdf,png}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..paths import DATA_DIR as RESULT, ONTOLOGY_OBO, PROJECT_ROOT, RESULT_DIR

# Full-page 3:4 template, using the existing ~180 mm working width. Fonts and
# physical marks track that width; exported pixels target the manuscript page.
CANVAS_WIDTH_IN = 180 / 25.4
CANVAS_HEIGHT_IN = CANVAS_WIDTH_IN * 4 / 3
DRAWING_SCALE = CANVAS_WIDTH_IN / (180 / 25.4)
EXPORT_DPI = 600 / DRAWING_SCALE
TEXT_BODY = CANVAS_WIDTH_IN * 72 / 85
TEXT_SECONDARY = 0.9 * TEXT_BODY
TEXT_PANEL = 1.8 * TEXT_BODY

# The shared read-out names a cell with the candidate scoring highest at its
# assigned landmark, over candidates at that landmark (resolve_column_winners)
# once each candidate's PageRank row is normalized (normalize_ppr_rows). A
# candidate beaten at every one of the 144 landmarks can never be returned,
# whatever a model's embedding. Twelve of the 66 types are in that position,
# which is why their five read-out columns are empty; marked below since the
# figure otherwise draws those rows with no explanation.
# 0.4 is the value scripts/run_figure4.py passes as PPR_ALPHA — NOT the class
# default in cell_ontology_ppr.py (0.9); keep both in sync by hand.
PPR_ALPHA = 0.4
UNREACHABLE_MARK = "*"
MARK_FONTSIZE = TEXT_PANEL
# Drawn separately from the tick label, so it can be sized apart from the name;
# this offset puts it back on the label's right-align anchor rather than the tick dash.
MARK_X_OFFSET_PT = -(3.5 + 3.5) * DRAWING_SCALE

# One benchmark row per (cell type, model, method). "HECTOR" under method
# ppr_candidate is its embedding through the shared read-out; its own head
# arrives under native_closed_set. Both are shown, named as in Fig. 4.
MODEL_DISPLAY = [
    ("HECTOR (own head)",  "HECTOR native, closed"),
    ("HECTOR (embedding)", "HECTOR embedding"),
    ("scimilarity",        "SCimilarity"),
    ("scGPT",              "scGPT"),
    ("scCello",            "scCello"),
    ("Geneformer",         "Geneformer"),
]
HECTOR_COLUMNS = {"HECTOR native, closed", "HECTOR embedding"}
HECTOR_RED = "#ED0000"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 1.0,
    "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 350,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

# Plain + and - , not the superscript pair: Arial has no U+207A/U+207B, so both
# ILC3 rows printed an identical missing-glyph box and could not be told apart.
ABBREV = {
    "NKp44-negative group 3 innate lymphoid cell, human": "NKp44- ILC3",
    "NKp44-positive group 3 innate lymphoid cell, human": "NKp44+ ILC3",
    "medullary thymic epithelial cell type 2": "mTEC type 2",
    "medullary thymic epithelial cell type 3": "mTEC type 3",
    "erythroid progenitor cell, mammalian": "erythroid progenitor cell",
}


def shorten(name: str, n: int = 80) -> str:
    name = ABBREV.get(name, name).replace(", human", "")
    return name if len(name) <= n else name[: n - 1] + "…"


def unreachable_row_labels(df: pd.DataFrame) -> set[str]:
    """Cell types the shared read-out cannot return, whatever the embedding.

    Rebuilds the read-out's second step — the candidate holding the most mass at
    each landmark — and keeps the candidates that win nowhere. Returns the names
    as they appear in the accuracy table, so they can be matched to row labels.
    """
    model_eval_dir = str(PROJECT_ROOT / "scripts" / "model_eval")
    if model_eval_dir not in sys.path:
        sys.path.insert(0, model_eval_dir)
    from cell_ontology_ppr import CellOntologyPPR, normalize_ppr_rows, resolve_column_winners

    ontology = CellOntologyPPR(ONTOLOGY_OBO)
    ontology.load_ontology()
    candidates = pd.read_csv(RESULT / "candidate_label_table.csv")
    landmarks = pd.read_csv(RESULT / "bridge_label_table.csv")
    mass = ontology.compute_candidate_to_bridge_similarity(
        candidates["cell_type_id"].tolist(),
        landmarks["cell_type_id"].tolist(),
        alpha=PPR_ALPHA,
    )
    winner_per_landmark, _tie_groups = resolve_column_winners(normalize_ppr_rows(mass))
    winners = set(winner_per_landmark.tolist())
    unreachable_ids = [cell_type_id for position, cell_type_id
                       in enumerate(candidates["cell_type_id"])
                       if position not in winners]
    names = df.drop_duplicates("cell_type_id").set_index("cell_type_id")["cell_type"]
    return {names[i] for i in unreachable_ids if i in names.index}


def build_matrices(df: pd.DataFrame):
    shared = df[df["method"] == "ppr_candidate"].copy()
    shared["model"] = shared["model"].replace({"HECTOR": "HECTOR (embedding)"})
    own = df[(df["method"] == "native_closed_set") & (df["model"] == "HECTOR")].copy()
    own["model"] = "HECTOR (own head)"
    df = pd.concat([own, shared], ignore_index=True)

    keys = [k for k, _ in MODEL_DISPLAY]
    exact = df.pivot(index="cell_type", columns="model", values="exact_frac")
    hop = df.pivot(index="cell_type", columns="model", values="mean_hop")
    exact = exact.reindex(columns=keys)
    hop = hop.reindex(columns=keys)

    # A cell type a model never returned a prediction for is not missing data:
    # it named none of those cells, the convention every other panel uses. Zero
    # exact matches, and a distance left blank because none was measured.
    exact = exact.fillna(0.0)

    order = hop["HECTOR (own head)"].sort_values(ascending=True).index.tolist()
    return exact.loc[order].to_numpy(), hop.loc[order].to_numpy(), order


def draw_heatmap(ax, mat: np.ndarray, row_labels, title: str,
                 cmap_name: str, show_ylabels: bool, cbar_label: str,
                 letter: str, vmin: float, vmax: float,
                 value_fmt: str, dark_when_high: bool, annotate: bool = True,
                 marked_rows: set[str] | None = None):
    n_rows, n_cols = mat.shape
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad("#F2F2F2")   # never answered
    mat = np.ma.masked_invalid(mat)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest", clip_on=False)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([disp for _k, disp in MODEL_DISPLAY],
                       fontsize=TEXT_SECONDARY, rotation=35, ha="right")
    # Tests membership in both HECTOR columns: neither carries the literal
    # label "HECTOR", so an exact-match test would mark nothing.
    for tl in ax.get_xticklabels():
        if tl.get_text() in HECTOR_COLUMNS:
            tl.set_color(HECTOR_RED); tl.set_fontweight("bold")

    if show_ylabels:
        # Every row carries the same trailing pad regardless of mark, so the
        # name column's right edge sits at a fixed distance from the axis.
        # The mark itself is a separate, larger Text at that fixed offset —
        # at name-label size a "*" reads as a barely visible speck.
        marked = marked_rows or set()
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([shorten(r) + "   " for r in row_labels],
                           fontsize=TEXT_BODY)
        ax.tick_params(axis="y", length=3.5 * DRAWING_SCALE, pad=3.5 * DRAWING_SCALE)
        mark_trans = mpl.transforms.offset_copy(
            ax.get_yaxis_transform(), fig=ax.figure, x=MARK_X_OFFSET_PT, y=0,
            units="points")
        for i, r in enumerate(row_labels):
            if r in marked:
                ax.text(0.0, i, UNREACHABLE_MARK, transform=mark_trans,
                        fontsize=MARK_FONTSIZE, ha="right", va="center_baseline",
                        color="#1a1a1a", clip_on=False)
    else:
        ax.set_yticks([]); ax.tick_params(axis="y", length=0)

    threshold = vmin + (vmax - vmin) * 0.55
    for i in range(n_rows):
        for j in range(n_cols):
            v = float(mat[i, j])
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center",
                        fontsize=TEXT_SECONDARY,
                        color="#999999")
                continue
            cell_is_dark = (v > threshold) if dark_when_high else (v < threshold)
            txt_color = "white" if cell_is_dark else "#1a1a1a"
            text = value_fmt.format(v).lstrip("0") if vmax <= 1 and value_fmt.endswith("f}") else value_fmt.format(v)
            if annotate:
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=TEXT_SECONDARY, color=txt_color)

    ax.set_title(title, fontsize=TEXT_BODY, fontweight="bold", pad=4 * DRAWING_SCALE)
    ax.tick_params(axis="x", length=0, pad=2 * DRAWING_SCALE)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.text(-0.45 if show_ylabels else -0.05, 1.02, letter,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=TEXT_PANEL, fontweight="bold")
    return im


def render(out_dir=RESULT_DIR) -> Path:
    df = pd.read_csv(RESULT / "per_held_out_type_accuracy.csv")
    exact, hop, row_labels = build_matrices(df)
    unreachable = unreachable_row_labels(df)
    hop_vmax = float(np.ceil(np.nanmax(hop)))

    # All 66 names fit on one page at the common body size. A dedicated label
    # column contains the longest names; the two six-column heatmaps retain
    # their arrangement and colour scales within the remaining page width.
    n_rows = len(row_labels)
    fig = plt.figure(figsize=(CANVAS_WIDTH_IN, CANVAS_HEIGHT_IN))
    gs = fig.add_gridspec(
        1, 5, width_ratios=[1.0, 0.035, 0.22, 0.85, 0.035], wspace=0.05,
        # Bottom margin holds only the 35-degree column names; methodological
        # detail lives in the manuscript Methods and legend.
        left=0.42, right=0.94, top=0.95, bottom=0.095,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    cbar_a_ax = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 3])
    cbar_b_ax = fig.add_subplot(gs[0, 4])

    im_a = draw_heatmap(
        ax_a, exact, row_labels,
        "Exact-match fraction\n(per-cell accuracy, higher = better)",
        cmap_name="Reds", show_ylabels=True, cbar_label="Fraction", letter="a",
        vmin=0, vmax=1, value_fmt="{:.2f}", dark_when_high=True,
        annotate=n_rows <= 40, marked_rows=unreachable,
    )
    im_b = draw_heatmap(
        ax_b, hop, row_labels,
        "Mean ontology hop from truth\n(when wrong, how close? lower = better)",
        cmap_name="viridis_r", show_ylabels=False, cbar_label="Mean hop", letter="b",
        vmin=0, vmax=hop_vmax, value_fmt="{:.1f}", dark_when_high=False,
        annotate=n_rows <= 40,
    )

    cbar_a = plt.colorbar(im_a, cax=cbar_a_ax)
    if cbar_a.solids is not None:
        cbar_a.solids.set_clip_on(False)
    cbar_a.set_label("Fraction", fontsize=TEXT_BODY, labelpad=2 * DRAWING_SCALE)
    cbar_a.ax.tick_params(labelsize=TEXT_SECONDARY, width=0.8 * DRAWING_SCALE,
                          length=3.5 * DRAWING_SCALE)
    cbar_b = plt.colorbar(im_b, cax=cbar_b_ax)
    if cbar_b.solids is not None:
        cbar_b.solids.set_clip_on(False)
    cbar_b.set_label("Mean hop", fontsize=TEXT_BODY, labelpad=2 * DRAWING_SCALE)
    cbar_b.ax.tick_params(labelsize=TEXT_SECONDARY, width=0.8 * DRAWING_SCALE,
                          length=3.5 * DRAWING_SCALE)

    out_dir = Path(out_dir)
    out_pdf = out_dir / "supplementary_figure_10_per_type.pdf"
    # Keep white margins and the same complete page in both formats. The
    # heatmap pixels render at insertion resolution; labels remain vector.
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(out_pdf, dpi=EXPORT_DPI, bbox_inches=None,
                    facecolor="white", transparent=False)
        fig.savefig(out_dir / "supplementary_figure_10_per_type.png",
                    dpi=EXPORT_DPI, bbox_inches=None,
                    facecolor="white", transparent=False)
    plt.close(fig)
    return out_pdf
