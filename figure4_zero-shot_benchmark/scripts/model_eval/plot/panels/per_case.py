"""Figure 4 — panel e: concentric hop-ring per-case panel.

Four held-out cell types in one row. Truth = gold star at the centre;
concentric dashed rings at hop distance 1..5+; each uniquely predicted cell name
gets one angular slot on its hop ring, with one coloured dot per model.

``draw_case`` takes the ontology graph (``dag``/``ud``) and the per-series modal
predictions from the composer; ``load_tables`` / ``modal_pred`` compute those from
the live prediction CSVs in ``result/``.

Every name is written OUTSIDE its own dot, reading away from the centre. The
earlier version anchored the names on the outer two rings inward instead, to
keep each drawing narrow, and a long name then ran back across the middle of its
own diagram and printed over the dots there — "photoreceptor cell" landed on top
of HECTOR's answer. The drawing is drawn smaller inside its axes instead, which
leaves a clear band around the outermost ring for the names to sit in.
"""
from __future__ import annotations

import math
from collections import defaultdict

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd

from ..paths import DATA_DIR as RESULT_DIR
from .. import style

# (key, display, csv filename, color)
# All three HECTOR arms, named and ordered as panel c names them. The closed arm
# answers every one of these cases exactly, so ``draw_case`` places it just off
# the star; open lands one or two hops out; the embedding sits among the
# comparison models. Only the open arm can answer outside the held-out set,
# which makes its near miss a biological statement, not a swap between
# candidates.
SERIES = [
    ("hector_native_closed", "native, closed", "hector_native_closed_set_predictions.csv",      style.HECTOR_NATIVE),
    ("hector_native", "native, open",       "hector_native_open_set_predictions.csv",        style.HECTOR_NATIVE_OPEN),
    ("hector_ppr",    "embedding",          "hector_ppr_candidate_predictions.csv",          style.HECTOR_EMBEDDING),
    ("scimilarity",   "SCimilarity",        "scimilarity_ppr_candidate_predictions.csv",     style.MODEL_COLOR["scimilarity"]),
    ("scgpt",         "scGPT",              "scgpt_ppr_candidate_predictions.csv",           style.MODEL_COLOR["scGPT"]),
    ("sccello",       "scCello",            "sccello_ppr_candidate_predictions.csv",         style.MODEL_COLOR["scCello"]),
    ("geneformer",    "Geneformer",         "geneformer_104m_ppr_candidate_predictions.csv", style.MODEL_COLOR["geneformer"]),
]

# One case per lineage, chosen by a rule fixed before the results were read:
# HECTOR's own head answers 1-2 ontology hops from the truth for at least half
# the cells of the type, while at least three of the four comparison models
# land 3+ hops away. 14 of the 66 held-out types qualify; these are the most
# consistent one per lineage.
CASES = [
    "centrocyte",                                    # immune     -> tonsil germinal center B cell
    "lens fiber cell",                               # epithelial -> secondary lens fiber
    "medium spiny neuron",                           # neural     -> direct pathway medium spiny neuron
    "adipocyte of epicardial fat of left ventricle",  # stromal    -> adipocyte
]
# Every title is the exact Cell Ontology name in full, never shortened or
# reworded: this is the only place a reader checks a case against the ontology
# directly. A name too long for one line is wrapped (see draw_case).
CASE_TITLE: dict[str, str] = {}

# The lineage each case stands for. Not a column in any table — the held-out
# types carry no lineage annotation — so this grouping is the figure's own,
# and the legend says so.
CASE_LINEAGE: dict[str, str] = {
    "centrocyte": "immune",
    "lens fiber cell": "epithelial",
    "medium spiny neuron": "neural",
    "adipocyte of epicardial fat of left ventricle": "stromal",
}

# A long five-line answer in the stromal case reaches into the header at its
# default angle; rotated toward the open side without changing its hop radius.
CASE_ANGLE_OVERRIDES: dict[tuple[str, str], float] = {
    (
        "adipocyte of epicardial fat of left ventricle",
        "L4/5 intratelencephalic projecting glutamatergic neuron",
    ): 15.0,
}

# The two arms drawn under the "HECTOR" heading in this panel's key. Their
# labels say only what tells them apart, as in panels c and d, so they need the
# heading above them and cannot be listed loose among the model names.
HECTOR_KEYS = ("hector_native_closed", "hector_native", "hector_ppr")
HECTOR_HEADING = "HECTOR"

TRUTH_FILL = style.TRUTH_FILL      # in style.py: panel d's inset uses it too
TRUTH_EDGE = style.TRUTH_EDGE
RING_GREY = "#C4C4C4"
MAX_HOP = 5

# How far outside the outermost ring a name is written, and how much room the
# axes keeps beyond that ring for those names to occupy.
LABEL_OFFSET = 0.55
OUTER_MARGIN = 2.4

DOT_AREA = 38.0          # marker area in points squared
DOT_TOUCH = 0.80         # spacing (dot widths) along a shared ring, so models
                         # overlap by a fifth rather than hide one another


def _dot_width_in_units(ax, half_span: float) -> float:
    """One dot's printed width, expressed in the axes' own units.

    Read from the axes box rather than assumed, so it stays right whatever
    height the panel is given. Equal aspect means the drawing occupies the
    largest square the box holds, hence the smaller of the two sides.
    """
    fig = ax.figure
    pos = ax.get_position()
    side_in = min(pos.width * fig.get_figwidth(), pos.height * fig.get_figheight())
    points_per_unit = side_in * 72.0 / (2.0 * half_span)
    return 2.0 * math.sqrt(DOT_AREA / math.pi) / max(points_per_unit, 1e-6)


def deg(a):
    return a * math.pi / 180.0


def _wrap(t, w=16):
    out, line = [], ""
    for word in str(t).split():
        if len(line) + len(word) + 1 > w:
            out.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return "\n".join(out)


def load_tables() -> dict[str, pd.DataFrame]:
    return {k: pd.read_csv(RESULT_DIR / csv, low_memory=False)
            for k, _d, csv, _c in SERIES if (RESULT_DIR / csv).exists()}


def modal_pred(df: pd.DataFrame, truth: str) -> str | None:
    sub = df[df["truth_cell_type_name"] == truth]
    if sub.empty:
        return None
    return str(sub["predicted_cell_type_name"].value_counts().idxmax())


def _wheel_rotation(n: int) -> float:
    """Where to start the ring of answers, so none of them sits due left or right.

    Names are written outward from their own dot, so an answer on the horizontal
    runs sideways out of its own drawing and into the case beside it —
    "endocrine cell" from the lens fibre case reached into the sinusoid case.
    The slots are a fixed 360/n apart, so the only freedom is where the first one
    starts; this takes the start that keeps every slot as far from horizontal as
    the count allows. With four answers that is a 45° turn, putting them in the
    four diagonals; with two or six the wheel is already clear and it turns none.
    """
    if n <= 1:
        return 0.0
    step = 360.0 / n
    best, best_score = 0.0, -1.0
    for k in range(max(1, int(round(step)))):
        score = min(abs(math.sin(deg(90 + k - step * i))) for i in range(n))
        if score > best_score + 1e-9:
            best, best_score = float(k), score
    return best


def _ring_tick_angle(used_angles: list[float]) -> float:
    """Put the 1..5+ ring numbers down the emptiest direction.

    Fixed at 45° they were printed over by whichever answer happened to take
    that slot.
    """
    if not used_angles:
        return 45.0
    ordered = sorted(a % 360 for a in used_angles)
    gaps = [((ordered[(i + 1) % len(ordered)] - a) % 360, a)
            for i, a in enumerate(ordered)]
    if len(ordered) == 1:
        return (ordered[0] + 180.0) % 360
    width, start = max(gaps)
    return (start + width / 2.0) % 360


def draw_case(ax, truth: str, dag, ud, modal_by_series: dict[str, str | None]) -> None:
    if truth not in dag:
        ax.text(0.5, 0.5, f"truth '{truth}' not in DAG", ha="center",
                va="center", color="red", transform=ax.transAxes)
        ax.axis("off"); return

    R = MAX_HOP + OUTER_MARGIN
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal")
    ax.axis("off")

    series_color = {k: c for k, _d, _csv, c in SERIES}

    by_cell: dict[str | None, list[str]] = defaultdict(list)
    for skey, *_ in SERIES:
        by_cell[modal_by_series.get(skey)].append(skey)

    cell_hop: dict[str, int] = {}
    for cell in by_cell:
        if cell is None or cell not in dag:
            cell_hop[cell] = MAX_HOP
        else:
            try:
                cell_hop[cell] = min(nx.shortest_path_length(ud, truth, cell), MAX_HOP)
            except nx.NetworkXNoPath:
                cell_hop[cell] = MAX_HOP

    cells_sorted = sorted(by_cell.keys(),
                          key=lambda c: (cell_hop.get(c, 99), str(c)))

    angular_cells = [c for c in cells_sorted if cell_hop.get(c) and cell_hop[c] > 0]
    n = len(angular_cells)
    angle_by_cell: dict[str, float] = {}
    if n > 0:
        rotation = _wheel_rotation(n)
        for i, cell in enumerate(angular_cells):
            angle_by_cell[cell] = 90 + rotation - (360.0 / n) * i
    for cell in angular_cells:
        override = CASE_ANGLE_OVERRIDES.get((truth, cell))
        if override is not None:
            angle_by_cell[cell] = override

    tick_angle = _ring_tick_angle(list(angle_by_cell.values()))
    tick_cos, tick_sin = math.cos(deg(tick_angle)), math.sin(deg(tick_angle))
    for r in range(1, MAX_HOP + 1):
        ax.add_patch(mpatches.Circle((0, 0), r, fill=False, lw=0.75, ls=":",
                                     ec=RING_GREY, zorder=1, clip_on=False))
        tag = f"{r}+" if r == MAX_HOP else str(r)
        ax.text(r * tick_cos, r * tick_sin, tag, fontsize=style.MAIN_SECONDARY, color="#777777",
                style="italic", ha="center", va="center", zorder=2,
                bbox=dict(fc="white", ec="none", pad=0.6))

    ax.scatter([0], [0], marker="*", s=170, fc=TRUTH_FILL,
               ec=TRUTH_EDGE, lw=0.6, zorder=8, clip_on=False)

    # A model that got the type exactly right sits just off the star rather than
    # under it.
    exact_models = []
    for cell, models in by_cell.items():
        if cell is not None and cell_hop.get(cell) == 0:
            exact_models.extend(models)
    for i, skey in enumerate(exact_models):
        a = deg(90 - (360 / max(1, len(exact_models))) * i)
        x, y = 0.72 * math.cos(a), 0.72 * math.sin(a)
        ax.scatter([x], [y], s=DOT_AREA, fc=series_color[skey],
                   ec="white", lw=0.8, zorder=9, clip_on=False)

    spacing = DOT_TOUCH * _dot_width_in_units(ax, R)
    for cell, angle in angle_by_cell.items():
        if cell is None:
            continue
        r = cell_hop[cell]
        a = deg(angle)
        models = by_cell[cell]
        n_m = len(models)
        # Models that gave the same answer sit at different points of the same
        # ring, a fifth of a dot's overlap apart.
        step = spacing / max(r, 1e-6)
        for i, skey in enumerate(models):
            ai = a + (i - (n_m - 1) / 2.0) * step
            ax.scatter([r * math.cos(ai)], [r * math.sin(ai)], s=DOT_AREA,
                       fc=series_color[skey], ec="white", lw=0.8,
                       zorder=7, clip_on=False)
        # Outside its own dot, reading away from the centre: a label placed
        # toward the centre instead would run a long name back across the
        # drawing.
        lab_r = r + LABEL_OFFSET
        ha = "left" if math.cos(a) > 0.15 else ("right" if math.cos(a) < -0.15 else "center")
        va = "bottom" if math.sin(a) > 0.15 else ("top" if math.sin(a) < -0.15 else "center")
        wrapped_label = _wrap(cell, 14)
        label_x, label_y = lab_r * math.cos(a), lab_r * math.sin(a)
        # This long word reaches the neighbouring lens-fiber case; nudge the
        # label only, not the dot's angle or hop radius.
        if truth == "medium spiny neuron" and cell == "intestinal enteroendocrine cell":
            label_x += 1.0
        ax.text(label_x, label_y, wrapped_label,
                fontsize=style.MAIN_SECONDARY, color="#1A1A1A",
                ha=ha, va=va, linespacing=1.2)

    title = CASE_TITLE.get(truth, truth)
    wrapped = _wrap(title, 28)
    # One title size for all names, including the wrapped stromal one. Top
    # alignment keeps that two-line title from growing upward into the row above.
    ax.text(0.5, 1.010, "truth: " + wrapped,
            transform=ax.transAxes,
            fontsize=style.MAIN_TEXT, fontweight="semibold", color="#1A1A1A",
            ha="center", va="top", linespacing=1.2)

    # Lineage is the shared first level of each case header, centred directly
    # above the ontology name rather than detached in a corner.
    lineage = CASE_LINEAGE.get(truth)
    if lineage:
        ax.text(0.5, 1.045, lineage.upper(),
                transform=ax.transAxes,
                fontsize=style.MAIN_SECONDARY, fontweight="semibold", color="#777777",
                ha="center", va="bottom")
