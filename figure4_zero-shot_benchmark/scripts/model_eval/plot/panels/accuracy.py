"""Figure 4 — zero-shot accuracy panel (3 metrics, 7 series in 2 groups).

Per-cell accuracy / macro F1 / hierarchical macro F1, one horizontal bar per
series. Scores are computed directly from the full prediction
CSVs; pass ``rows=`` to feed in-memory tables instead of reading disk.

Two brackets down the left name the groups, and the row names carry the
distinction that separates the seven rows:

  HECTOR             the first three rows — its own prediction restricted to
                     the 66 held-out types, its unrestricted prediction among
                     1,407 terms, and its embedding
  shared read-out    the last five — HECTOR's embedding and all four
                     comparison models, every one of them answering by the
                     same nearest-landmark rule

  native / embedding   is this HECTOR's own prediction, or only its embedding?

The brackets overlap on the embedding row, which is the point: that row is
HECTOR's and it also answers through the read-out the comparison models are
given. Two spans that share a row cannot be drawn as headings, which is why
they are brackets. This replaced a flat list in which only HECTOR's embedding
row was marked "(embedding)" — all four comparison models are embeddings put
through the same read-out, so a qualifier on HECTOR alone read as a caveat
about HECTOR rather than as the shared condition it is.

The figure legend defines both words. The rows only have to be told apart
at a glance.

HECTOR's native, open arm is retained here because its near-zero exact-match
scores and comparatively high hierarchical score are central to the distinction
between exact naming and ontology localization. The figure legend states that
this row chooses among 1,407 terms while every other row chooses among 66, so
the unequal answer space is explicit rather than implied to be comparable.
"""
from __future__ import annotations

import sys

import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  (kept for standalone use / parity)
from matplotlib.gridspec import GridSpecFromSubplotSpec
from sklearn.metrics import f1_score

from ..paths import DATA_DIR as RESULT, PROJECT_ROOT
from .. import style

_MODEL_EVAL_DIR = str(PROJECT_ROOT / "scripts" / "model_eval")
if _MODEL_EVAL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_EVAL_DIR)
from query_set_padding import (  # noqa: E402
    expected_query_counts,
    pad_to_query_set,
)

# (heading, [(key, csv, display, color), ...]) — top -> bottom. No headings:
# the two brackets below already name the groups. Each row name carries the
# one distinction separating the six rows — native against embedding, closed
# (66 candidates) against open (1,407 terms).
GROUPS = [
    (None, [
        ("hector_own_matched", "hector_native_closed_set_predictions.csv",      "native, closed",         style.HECTOR_NATIVE),
        ("hector_own_open",    "hector_native_open_set_predictions.csv",        "native, open",           style.HECTOR_NATIVE_OPEN),
        ("hector_shared",      "hector_ppr_candidate_predictions.csv",          "embedding",              style.HECTOR_EMBEDDING),
    ]),
    (None, [
        ("scimilarity",        "scimilarity_ppr_candidate_predictions.csv",     "SCimilarity",            style.MODEL_COLOR["scimilarity"]),
        ("scgpt",              "scgpt_ppr_candidate_predictions.csv",           "scGPT",                  style.MODEL_COLOR["scGPT"]),
        ("sccello",            "sccello_ppr_candidate_predictions.csv",         "scCello",                style.MODEL_COLOR["scCello"]),
        ("geneformer",         "geneformer_104m_ppr_candidate_predictions.csv", "Geneformer",             style.MODEL_COLOR["geneformer"]),
    ]),
]

SERIES = [row for _heading, rows in GROUPS for row in rows]
HECTOR_KEYS = {key for key, *_rest in GROUPS[0][1]}
HECTOR_RED = "#ED0000"

# The two spans overlap on the embedding row — HECTOR's row that also answers
# through the shared read-out — so they need separate bracket columns rather
# than headings.
BRACKETS = [
    ("HECTOR", [k for k, *_r in GROUPS[0][1]], HECTOR_RED, "bold"),
    ("shared read-out", ["hector_shared", "scimilarity", "scgpt", "sccello",
                         "geneformer"], "#666666", "normal"),
]

# Bracket geometry, in points, so it holds whatever size the panel is given.
BRACKET_GAP_PT = 4.0     # row labels to the first bracket's line
BRACKET_TEXT_PT = 2.0    # a bracket's line to its own rotated name
BRACKET_STEP_PT = 11.0   # one bracket column to the next
BRACKET_TICK_PT = 4.0    # the end ticks, which point back at the rows

# Row-height gap separating HECTOR's rows from the comparison models; the
# shared-read-out bracket correctly runs straight through it, since its span
# crosses that boundary.
GROUP_GAP = 1.4
HEADING_CLEARANCE = 0.5   # how far above its first row a heading is anchored


def _draw_brackets(ax, y_of, fontsize):
    """Draw every bracket in ``BRACKETS``, outside the row labels.

    Placed by measuring where the row labels actually end rather than by a
    guessed offset, so they stay clear of them whatever the labels say. Each
    bracket's rows must be contiguous, which they are: HECTOR's three arms are
    the first three rows, and its embedding sits last of those, immediately
    above the four comparison models.
    """
    fig = ax.figure
    fig.canvas.draw()
    label_left = [tl.get_window_extent().x0 for tl in ax.get_yticklabels()
                  if tl.get_text()]
    if not label_left:
        return
    x0_px = min(label_left)
    px_per_pt = fig.dpi / 72.0
    to_axes = ax.transAxes.inverted()
    trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)

    for depth, (label, keys, color, weight) in enumerate(BRACKETS):
        ys = [y_of[k] for k in keys if k in y_of]
        if len(ys) < 2:
            continue
        line_px = x0_px - (BRACKET_GAP_PT + depth * BRACKET_STEP_PT) * px_per_pt
        x_line = to_axes.transform((line_px, 0))[0]
        x_tick = to_axes.transform((line_px + BRACKET_TICK_PT * px_per_pt, 0))[0]
        x_text = to_axes.transform((line_px - BRACKET_TEXT_PT * px_per_pt, 0))[0]
        y0, y1 = min(ys) - 0.42, max(ys) + 0.42
        ax.plot([x_line, x_line], [y0, y1], transform=trans, color=color,
                lw=0.7, clip_on=False, zorder=5)
        for y in (y0, y1):
            ax.plot([x_line, x_tick], [y, y], transform=trans, color=color,
                    lw=0.7, clip_on=False, zorder=5)
        ax.text(x_text, (y0 + y1) / 2.0, label, transform=trans, rotation=90,
                ha="right", va="center", fontsize=fontsize, color=color,
                fontweight=weight)


def _row_positions() -> tuple[list[float], list[tuple[str, float]]]:
    """Where each bar sits on the y axis, and where each group heading sits.

    Bars would sit at 0, 1, 2 ... in a flat list. Here the second group is
    pushed down to open a band for its heading, so the positions are floats and
    every drawing call takes them from this one place rather than from
    ``enumerate``. Each heading is anchored just above its group's first row and
    drawn upward from there, so a heading of any number of lines clears the row
    above it without a second number to keep in step.
    """
    positions, headings = [], []
    y = 0.0
    for heading, rows in GROUPS:
        if heading is not None:
            headings.append((heading, y - HEADING_CLEARANCE))
        for _row in rows:
            positions.append(y)
            y += 1.0
        y += GROUP_GAP - 1.0
    return positions, headings

# (key, column title, x limit, x ticks). Titles name the measure; the measures
# themselves are defined in the Methods, not restated here as a question.
METRICS = [
    ("per_cell_accuracy",     "Per-cell accuracy",     0.55, [0.0, 0.2, 0.4]),
    ("macro_f1",              "Macro F1",              0.55, [0.0, 0.2, 0.4]),
    ("hierarchical_macro_f1", "Hierarchical macro F1", 0.85, [0.0, 0.2, 0.4, 0.6, 0.8]),
]
VAL_COLOR = "#222222"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 1.0,
    "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 350,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def load_tables() -> list[tuple]:
    expected = expected_query_counts(RESULT)
    rows = []
    for key, csv, disp, color in SERIES:
        p = RESULT / csv
        if not p.exists():
            print(f"[panel_accuracy] missing {p}")
            continue
        df = pd.read_csv(p, low_memory=False)
        df["truth_cell_type_id"] = df["truth_cell_type_id"].astype(str)
        df["predicted_cell_type_id"] = df["predicted_cell_type_id"].astype(str)
        n_before = len(df)
        df = pad_to_query_set(df, expected)
        if len(df) != n_before:
            print(f"[panel_accuracy] {disp}: {len(df) - n_before:,} query cells "
                  f"not returned, counted as errors")
        rows.append((key, disp, color, df))
    return rows


def _metric(df: pd.DataFrame, metric: str, labels) -> float:
    if metric == "per_cell_accuracy":
        return float(df["flat_accuracy"].mean())
    if metric == "macro_f1":
        return float(f1_score(df["truth_cell_type_id"], df["predicted_cell_type_id"],
                              labels=labels, average="macro", zero_division=0))
    if metric == "hierarchical_macro_f1":
        per_class = (df.groupby("truth_cell_type_id")["hierarchical_f1"]
                       .mean().reindex(labels).fillna(0.0))
        return float(per_class.mean())
    raise ValueError(metric)


def draw_panel_accuracy(host_ax, rows=None,
                        ytick_fontsize=8, value_fontsize=6.8,
                        title_fontsize=9, bracket_fontsize=6.5,
                        xtick_fontsize=7):
    if rows is None:
        rows = load_tables()
    labels = sorted(set().union(*[set(df["truth_cell_type_id"]) for *_, df in rows]))

    host_ax.set_xticks([]); host_ax.set_yticks([])
    for sp in host_ax.spines.values():
        sp.set_visible(False)
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=host_ax.get_subplotspec(), wspace=0.32)
    fig = host_ax.figure
    sub = [fig.add_subplot(gs[0, j]) for j in range(3)]

    y_of, headings = _row_positions()
    y_of = {key: y_of[i] for i, (key, *_r) in enumerate(SERIES)}
    ys = [y_of[key] for key, *_r in rows]

    for j, (mkey, title, xlim_max, xticks) in enumerate(METRICS):
        ax = sub[j]
        ax.set_axisbelow(True)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, color="#BBBBBB", alpha=0.6, zorder=0)
        for yi, (key, disp, color, df) in zip(ys, rows):
            point = _metric(df, mkey, labels)
            is_h = key in HECTOR_KEYS
            ax.barh(yi, point, height=0.66 if is_h else 0.6, color=color,
                    edgecolor="black" if is_h else "none",
                    linewidth=0.6 if is_h else 0.0, zorder=2, clip_on=False)
            ax.text(point + 0.010, yi, f"{point:.3f}",
                    va="center", ha="left",
                    fontsize=value_fontsize, color=VAL_COLOR, zorder=4)
        # Top clears the first bracket's end tick, 0.42 rows above the first row.
        ax.set_ylim(max(ys) + 0.6, min(ys) - 0.6)
        ax.set_xlim(0, xlim_max)
        ax.set_xticks(xticks)
        ax.set_yticks(ys)
        if j == 0:
            ax.set_yticklabels([disp for _k, disp, _c, _d in rows], fontsize=ytick_fontsize)
            # Red marks ours; bold is reserved for the headings, so that a
            # heading and the rows under it are told apart rather than reading
            # as three equal lines.
            for tl, (key, *_r) in zip(ax.get_yticklabels(), rows):
                if key in HECTOR_KEYS:
                    tl.set_color(HECTOR_RED)
            # Group headings, right-aligned with the row labels and anchored
            # just above their group's first row so they grow upward — the
            # second one is two lines and must not touch the row above it.
            label_column = mpl.transforms.blended_transform_factory(
                ax.transAxes, ax.transData)
            for text, y_heading in headings:
                ax.annotate(text, xy=(0.0, y_heading), xycoords=label_column,
                            xytext=(-4.0, 0.0), textcoords="offset points",
                            ha="right", va="bottom", linespacing=1.2,
                            fontsize=ytick_fontsize, fontweight="bold",
                            color=HECTOR_RED if text.startswith("HECTOR") else "#333333",
                            annotation_clip=False)
            ax.tick_params(axis="y", length=0)
            _draw_brackets(ax, y_of, bracket_fontsize)
        else:
            ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=xtick_fontsize)
        ax.spines["left"].set_linewidth(0.6); ax.spines["bottom"].set_linewidth(0.6)
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=5)
