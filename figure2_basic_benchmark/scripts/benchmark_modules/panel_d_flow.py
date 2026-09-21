"""Panel D: ground-truth -> raw-prediction ontology flow.

Self-contained on purpose. Everything the panel needs -- Cell Ontology parsing, branch
assignment, the lineage rules, the colour palette and the drawer -- lives in this one module
inside benchmark_modules, so figure2 has no dependency on any folder outside
figure2_basic_benchmark/.

Every ribbon is one (ground truth, prediction) pair, over all cells, unfiltered. The ribbon
span is short and the side margins wide because the cell-type names are the panel's content
and are never abbreviated.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import re
import textwrap

from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Kidney-specific, ordered anatomically. The order only breaks equal-distance ontology ties.
BRANCH_NAMES = [
    "nephron tubule epithelial cell",
    "kidney medulla cell",
    "renal principal cell",
    "kidney collecting duct epithelial cell",
    "epithelial cell of glomerular capsule",
    "endothelial cell",
    "kidney interstitial cell",
    "mononuclear phagocyte",
    "T cell",
    "lymphocyte of B lineage",
    "natural killer cell",
    "lymphocyte",
]

# Both terms sit at equal ontology distance from two displayed branches; route them to the
# endothelium, which is the biologically meaningful parent here.
BRANCH_OVERRIDES = {
    "glomerular capillary endothelial cell": "endothelial cell",
    "kidney capillary endothelial cell": "endothelial cell",
}

# Predicted terms rarer than this are pooled into a per-branch bar rather than discarded, and
# the pooled bar keeps its ribbon. Prediction column only; every ground-truth term is shown.
RARE_TERM_THRESHOLD = 100

BRANCH_LINEAGE = {
    "nephron tubule epithelial cell": "Kidney epithelium",
    "kidney medulla cell": "Kidney epithelium",
    "renal principal cell": "Kidney epithelium",
    "kidney collecting duct epithelial cell": "Kidney epithelium",
    "epithelial cell of glomerular capsule": "Kidney epithelium",
    "endothelial cell": "Endothelium",
    "kidney interstitial cell": "Stroma",
    "mononuclear phagocyte": "Myeloid",
    "T cell": "Lymphoid",
    "lymphocyte of B lineage": "Lymphoid",
    "natural killer cell": "Lymphoid",
    "lymphocyte": "Lymphoid",
    # These cells did get a term; it simply sits outside the branches shown. Not "Unassigned",
    # which would read as an abstention the model never made.
    "Outside the groups shown": "Outside the groups shown",
}

# Okabe & Ito (2008) for dichromat discriminability; Endothelium darkened from
# the published vermillion, which pairs poorly with orange under deuteranopia.
# Colour encodes lineage, not branch (13 categories can't be separated by hue;
# each bar's full name carries branch identity). Out-of-scope group is grey,
# since it is not a lineage.
LINEAGE_HUES = {
    "Kidney epithelium": "#0072B2",
    "Endothelium": "#A83214",
    "Stroma": "#CC79A7",
    "Myeloid": "#F5B942",
    "Lymphoid": "#009E73",
    "Outside the groups shown": "#8C9196",
}
BRANCH_COLORS = {branch: LINEAGE_HUES[group] for branch, group in BRANCH_LINEAGE.items()}

# An ordered severity scale, not a categorical set, so greyscale: no hue to lose, and it can
# never be misread as a lineage colour from the flow above.
SUMMARY_COLORS = {
    "Exact match": "#1A1A1A",
    "Correct, coarser term": "#4A4A4A",
    "Consistent, finer term": "#787878",
    "Wrong cell type, same lineage": "#A5A5A5",
    "Wrong cell type, other lineage": "#D2D2D2",
}
SUMMARY_ORDER = list(SUMMARY_COLORS)
SUMMARY_LABEL_PT = 4.4

# NUMBER_PT is imported by panel C, so both badge circles match.
BADGE_PAD = 0.14        # matplotlib circle padding, in units of font size
BADGE_EDGE = "#8A9099"  # panel C's disc needs no edge over dense points; here it does
NUMBER_PT = 5.2
HAIRLINE_PT = 0.25      # thinnest line a press reliably holds (0.09 mm); see _ribbon
BADGE_SCALE = 0.90      # a two-digit number is wider than tall; check "52" still fits if lowered


def _text_width_pt(text, fontsize, weight="normal"):
    """Rendered width of `text` in points, from the font's own metrics.

    `weight` matters: bold is the wider face, so measuring a bold label as regular
    puts anything placed after it on top of the label's last few characters.
    """
    from matplotlib import rcParams
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    font = FontProperties(family=rcParams["font.family"], size=fontsize, weight=weight)
    return TextPath((0, 0), text, prop=font).get_extents().width


def badge_diameter_pt(highest_number, fontsize):
    """Diameter in points of one circle big enough for the widest number at this size.

    Every circle uses this diameter, so a number is the same mark wherever it appears.
    Measured from font metrics, so it is as small as it can be without clipping.
    """
    from matplotlib import rcParams
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    font = FontProperties(family=rcParams["font.family"], size=fontsize)
    ext = TextPath((0, 0), str(highest_number), prop=font).get_extents()
    half_diagonal = ((ext.width / 2) ** 2 + (ext.height / 2) ** 2) ** 0.5
    return 2 * (half_diagonal + BADGE_PAD * fontsize) * BADGE_SCALE

# Major lineages, resolved by walking is_a to the first matching anchor. ORDER
# MATTERS: "endothelial cell" descends from "epithelial cell" in CL, so testing
# epithelium first would swallow it. "hematopoietic cell" is a trailing
# catch-all for "myeloid cell"/"erythrocyte", which sit outside "leukocyte".
LINEAGE_ANCHORS = [
    ("Immune", "leukocyte"),
    ("Endothelium", "endothelial cell"),
    ("Epithelium", "epithelial cell"),
    ("Stroma", "connective tissue cell"),
    ("Stroma", "kidney interstitial cell"),
    ("Immune", "hematopoietic cell"),
]


# -----------------------------------------------------------------------------
# Cell Ontology
# -----------------------------------------------------------------------------

def parse_cell_ontology(obo_path):
    parents = defaultdict(list)
    names = {}
    obsolete = set()
    current = None
    with open(obo_path) as handle:
        for raw in handle:
            line = raw.strip()
            if line == "[Term]" or line.startswith("[Typedef]"):
                current = None
            elif line.startswith("id:") and current is None:
                current = line.split("id:", 1)[1].strip()
            elif line.startswith("name:") and current:
                names[current] = line.split("name:", 1)[1].strip()
            elif line.startswith("is_a:") and current:
                match = re.match(r"is_a:\s*(\S+)", line)
                if match:
                    parents[current].append(match.group(1))
            elif line.startswith("is_obsolete: true") and current:
                obsolete.add(current)
    names = {k: v for k, v in names.items() if k not in obsolete}
    parents = {
        child: [p for p in ps if p in names]
        for child, ps in parents.items()
        if child in names
    }
    return defaultdict(list, parents), names


def _name_lookup(names):
    lookup = {}
    for term_id, name in names.items():
        lookup.setdefault(name, term_id)
    return lookup


def _ancestors(term_id, parents, cache):
    if term_id in cache:
        return cache[term_id]
    seen = set()
    queue = deque(parents.get(term_id, []))
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        queue.extend(parents.get(current, []))
    cache[term_id] = seen
    return seen


def _assign_branch(term_id, parents, anchor_ids, priority):
    """Nearest displayed ancestor; ties broken by the anatomical order of BRANCH_NAMES."""
    if term_id in anchor_ids:
        return term_id, True, False
    depth = {term_id: 0}
    queue = deque([term_id])
    best_depth, found = None, []
    while queue:
        current = queue.popleft()
        current_depth = depth[current]
        if best_depth is not None and current_depth >= best_depth:
            continue
        for parent in parents.get(current, []):
            next_depth = current_depth + 1
            if parent in anchor_ids:
                best_depth = next_depth
                found.append(parent)
            elif parent not in depth or next_depth < depth[parent]:
                depth[parent] = next_depth
                queue.append(parent)
    if not found:
        return "OTHER_BRANCH", False, False
    unique = sorted(set(found), key=lambda x: priority[x])
    return unique[0], False, len(unique) > 1


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def prepare_flow(predictions, parents, names, threshold=RARE_TERM_THRESHOLD):
    lookup = _name_lookup(names)
    branch_ids = [lookup[b] for b in BRANCH_NAMES]
    priority = {b: i for i, b in enumerate(branch_ids)}

    table = predictions.copy()
    table["true_id"] = table["true_label_raw"].map(lookup)
    table["prediction_id"] = table["pred_label_raw"].map(lookup)
    if table[["true_id", "prediction_id"]].isna().any().any():
        raise ValueError("Every ground-truth and prediction term must resolve to Cell Ontology.")

    assignment, is_direct, is_ambiguous = {}, {}, {}
    for term_id in table["prediction_id"].unique():
        branch, direct, ambiguous = _assign_branch(term_id, parents, set(branch_ids), priority)
        assignment[term_id], is_direct[term_id], is_ambiguous[term_id] = branch, direct, ambiguous
    for raw_name, branch_name in BRANCH_OVERRIDES.items():
        term_id, branch_id = lookup[raw_name], lookup[branch_name]
        assignment[term_id] = branch_id
        is_direct[term_id] = term_id == branch_id
        is_ambiguous[term_id] = False

    table["branch_id"] = table["prediction_id"].map(assignment)
    table["direct_at_branch"] = table["prediction_id"].map(is_direct)
    table["ambiguous_branch"] = table["prediction_id"].map(is_ambiguous)
    table["branch_label"] = table["branch_id"].map(names).fillna("Outside the groups shown")

    counts = table["pred_label_raw"].value_counts()
    retained = set(counts[counts >= threshold].index)
    rare = ~table["pred_label_raw"].isin(retained)
    other_branch = table["branch_id"] == "OTHER_BRANCH"
    table["right_label"] = table["pred_label_raw"]
    table.loc[rare, "right_label"] = f"Other terms (<{threshold} each)"
    table.loc[rare & other_branch, "right_label"] = f"Other ontology terms (<{threshold} each)"

    cache = {}
    anchor_ids = [(group, lookup[anchor]) for group, anchor in LINEAGE_ANCHORS]

    def lineage_of(term_id):
        members = _ancestors(term_id, parents, cache) | {term_id}
        for group, anchor_id in anchor_ids:
            if anchor_id in members:
                return group
        return "Other"

    lineages = {t: lineage_of(t) for t in set(table["true_id"]) | set(table["prediction_id"])}
    table["true_lineage"] = table["true_id"].map(lineages)
    table["pred_lineage"] = table["prediction_id"].map(lineages)

    # These categories come from the ontology alone, never from the coarse
    # scoring vocabulary.
    def categorize(true_id, pred_id, true_lineage, pred_lineage):
        if true_id == pred_id:
            return "Exact match"
        if pred_id in _ancestors(true_id, parents, cache):
            return "Correct, coarser term"
        if true_id in _ancestors(pred_id, parents, cache):
            return "Consistent, finer term"
        # Everything past this point is wrong; the only distinction kept is
        # whether it stayed inside the cell's own lineage. "Other lineage" also
        # holds pairs with no lineage anchor at all — wrong either way.
        if true_lineage != "Other" and true_lineage == pred_lineage:
            return "Wrong cell type, same lineage"
        return "Wrong cell type, other lineage"

    table["category"] = [
        categorize(*row) for row in zip(
            table["true_id"], table["prediction_id"], table["true_lineage"], table["pred_lineage"]
        )
    ]

    branch_order = BRANCH_NAMES + ["Outside the groups shown"]
    left_to_branch = table.groupby(["true_label_raw", "branch_label"]).size()
    branch_to_right = table.groupby(["branch_label", "right_label"]).size()
    return table, left_to_branch, branch_to_right, branch_order


def load_panel_data(prediction_path, ontology_path, threshold=RARE_TERM_THRESHOLD):
    predictions = pd.read_csv(prediction_path)
    parents, names = parse_cell_ontology(ontology_path)
    return prepare_flow(predictions, parents, names, threshold)


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------

def _stack(order, counts, small_gap, groups, large_gap):
    positions, cursor, previous = {}, 0.0, None
    for label in order:
        group = groups.get(label) if groups else None
        if positions:
            cursor += large_gap if (groups and group != previous) else small_gap
        height = float(counts[label])
        positions[label] = [cursor, cursor + height]
        cursor += height
        previous = group
    return positions, cursor


def _stack_to_span(order, counts, small_gap, groups, large_gap, target):
    """Stack a column, then widen its blank space until the column is `target` tall.

    The two columns hold the same cells but a different number of bars, so with one gap size
    the shorter one would be centred with dead space above and below it while its own labels
    crowd. Only the blank space is scaled: every bar keeps the height its cell count gives it,
    and both columns keep the same cells-per-inch, which is what lets a ribbon arrive at the
    thickness it left at.
    """
    positions, span = _stack(order, counts, small_gap, groups, large_gap)
    bars = float(sum(float(counts[k]) for k in order))
    blank = span - bars
    if target is None or span >= target or blank <= 0:
        return positions, span
    scale = (target - bars) / blank
    return _stack(order, counts, small_gap * scale, groups, large_gap * scale)


def _ribbon(axis, x0, x1, left, right, color, alpha, hairline_units=0.0, hairline_pt=0.0):
    """Draw one (ground truth, prediction) pair.

    A non-zero `hairline_units` strokes the ribbon at `hairline_pt` when it is thinner than
    that, so a flow below the output resolution renders as a hairline rather than vanishing --
    an invisible ribbon reads as "no cells went here". The caller applies it to at most one
    ribbon per ground-truth term; see draw_panel_d. The filled shape keeps its exact size and
    no ribbon that already renders is touched.
    """
    lt, lb = left
    rt, rb = right
    thin = min(lb - lt, rb - rt) < hairline_units
    xm = (x0 + x1) / 2
    verts = [(x0, -lt), (xm, -lt), (xm, -rt), (x1, -rt),
             (x1, -rb), (xm, -rb), (xm, -lb), (x0, -lb), (x0, -lt)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    axis.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                             edgecolor=color if thin else "none",
                             linewidth=hairline_pt if thin else 0.0,
                             alpha=alpha, zorder=2, clip_on=False))


def draw_panel_d(fig, spec, table, left_to_branch, branch_to_right, branch_order, numbering,
                 label_fontsize=NUMBER_PT, header_fontsize=7.0, gutter=1.62, wrap=58,
                 threshold=RARE_TERM_THRESHOLD, truth_colors=None, pred_colors=None):
    """Draw the flow and the relationship bar into `spec`.

    `gutter` is the label margin on each side, in units of the ribbon span. Values above 1
    make the ribbons shorter than the label gutters, which is what lets the full Cell Ontology
    names be printed without wrapping or abbreviation.

    `truth_colors` and `pred_colors` map a cell type to its panel-C colour, carried by the
    small square beside each label. The colours only have to support matching a cluster to a
    label, not to be mutually discriminable. Types absent from the mapping fall back to grey.

    `numbering` maps a cell type to the number panel C prints on that cluster; the same number
    is drawn here, so a cell type can be carried between the panels. The pooled "Other terms"
    labels are unnumbered, as they are in panel C.
    """
    # Tight legend and summary bands; retain spare page space below, not between them.
    inner = spec.subgridspec(4, 1, height_ratios=[214, 9, 12, 13], hspace=0.01)
    flow = fig.add_subplot(inner[0, 0])
    legend_axis = fig.add_subplot(inner[1, 0])
    summary = fig.add_subplot(inner[2, 0])

    total = len(table)
    truth_counts = table["true_label_raw"].value_counts()
    branch_counts = table["branch_label"].value_counts().reindex(branch_order).fillna(0)
    rank = {b: i for i, b in enumerate(branch_order)}
    dominant = {t: left_to_branch.loc[t].idxmax() for t in truth_counts.index}
    truth_order = sorted(truth_counts.index, key=lambda t: (rank[dominant[t]], -truth_counts[t]))
    edges = table.groupby(["true_label_raw", "branch_label", "right_label"]).size()

    right_branch, right_order = {}, []
    for branch in branch_order:
        if branch not in branch_to_right.index.get_level_values(0):
            continue
        counts = branch_to_right.loc[branch].sort_values(ascending=False)
        ordinary = [x for x in counts.index if not x.startswith("Other ")]
        pooled = [x for x in counts.index if x.startswith("Other ")]
        for label in ordinary + pooled:
            key = f"{branch}|||{label}"
            right_order.append(key)
            right_branch[key] = branch
    right_counts = {k: int(branch_to_right[(right_branch[k], k.split("|||", 1)[1])])
                    for k in right_order}

    truth_groups = {t: dominant[t] for t in truth_order}
    right_groups = {r: right_branch[r] for r in right_order}

    # Reserve a measured row for each label, then centre its bar in that row.
    # A single cells-per-point scale applies to both columns and every ribbon.
    # This prevents label collisions without detaching names from their bars.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    flow_pt = fig.get_size_inches()[1] * flow.get_position().height * 72
    axis_pt = fig.get_size_inches()[0] * flow.get_position().width * 72
    badge_pt = badge_diameter_pt(max(numbering.values(), default=9), label_fontsize)
    available_pt = gutter * axis_pt / (1 + 2 * gutter) - badge_pt * 1.22 - 12
    wrapped_labels = {}
    row_heights = {}
    for side, order, counts in (("left", truth_order, truth_counts),
                                ("right", right_order, right_counts)):
        for key in order:
            term = key if side == "left" else key.split("|||", 1)[1]
            weight = "bold" if side == "right" and term == right_branch[key] else "normal"
            text = f"{term} ({counts[key]:,})"
            line_width = wrap
            wrapped = textwrap.fill(text, line_width, break_long_words=False, break_on_hyphens=False)
            while max(_text_width_pt(line, label_fontsize, weight) for line in wrapped.splitlines()) > available_pt:
                line_width -= 1
                if line_width < 12:
                    raise ValueError("Panel D label gutter is too narrow.")
                wrapped = textwrap.fill(text, line_width, break_long_words=False, break_on_hyphens=False)
            probe = flow.text(0, 0, wrapped, fontsize=label_fontsize, fontweight=weight, linespacing=1.05)
            height = probe.get_window_extent(renderer).height * 72 / fig.dpi
            probe.remove()
            wrapped_labels[(side, term)] = wrapped
            # Unnamed pooled rows retain enough room for their gray square marker.
            row_heights[(side, key)] = (3.5 if side == "right" and term.startswith("Other ")
                                       and not term.startswith("Other ontology terms")
                                       else max(height, badge_pt if term in numbering else 0))

    available_height = flow_pt / 1.09
    low, high = 0.0, available_height / total
    for _ in range(60):
        scale = (low + high) / 2
        needed = []
        for side, order, counts, groups in (("left", truth_order, truth_counts, truth_groups),
                                           ("right", right_order, right_counts, right_groups)):
            gaps = [0 if i == 0 else (2.0 if groups[key] != groups[order[i - 1]] else 0.8)
                    for i, key in enumerate(order)]
            needed.append(sum(max(float(counts[key]) * scale, row_heights[(side, key)]) + gap
                              for key, gap in zip(order, gaps)))
        if max(needed) > available_height:
            high = scale
        else:
            low = scale
    if low < 1e-8:
        raise ValueError("Panel D needs more height for aligned labels at this font size.")
    span = available_height / low
    column_positions = []
    for side, order, counts, groups in (("left", truth_order, truth_counts, truth_groups),
                                       ("right", right_order, right_counts, right_groups)):
        gaps = [0 if i == 0 else (2.0 if groups[key] != groups[order[i - 1]] else 0.8)
                for i, key in enumerate(order)]
        heights = [max(float(counts[key]) * low, row_heights[(side, key)]) for key in order]
        extra_gap = max(0, available_height - sum(heights) - sum(gaps)) / max(1, len(order) - 1)
        cursor, positions = 0.0, {}
        for i, (key, height, gap) in enumerate(zip(order, heights, gaps)):
            cursor += gap + (extra_gap if i else 0)
            top = (cursor + (height - float(counts[key]) * low) / 2) / low
            positions[key] = [top, top + float(counts[key])]
            cursor += height
        column_positions.append(positions)
    truth_pos, right_pos = column_positions

    x_left, x_left_end = 0.0, 0.030
    x_right, x_right_end = 0.970, 1.0

    # Cell-units per point vertically; the 1.09 factor matches set_ylim, which
    # runs after the ribbons.
    flow_pt = fig.get_size_inches()[1] * flow.get_position().height * 72
    hairline_units = HAIRLINE_PT * (span * 1.09) / flow_pt

    # A fixed-size square beside each label, not the bar: bar height is cell
    # count alone, so the smallest bars are sub-pixel and could never show a
    # colour otherwise.
    CHIP_S = 12.0     # marker area in pt^2 -> ~3.5 pt square, the same for every bar
    chip_dx = 0.024   # square offset from the bar, in ribbon-span units
    label_dx = 0.058  # label offset, clear of the square
    # The triangle marking a branch-level row is DRAWN, not typed: Arial has no glyph
    # for U+25C0, so appending that character to the label would render as nothing.
    PARENT_S = 5.0    # marker area in pt^2 -> ~2.5 pt triangle, a little under the square
    PARENT_PAD_PT = 3.6   # measured from the name's last inked pixel, so it reads as two
                          # spaces separating the marker from the name

    truth_cursor = {t: truth_pos[t][0] for t in truth_order}
    right_cursor = {r: right_pos[r][0] for r in right_order}
    index = {r: i for i, r in enumerate(right_order)}
    for truth_label in truth_order:
        destinations = sorted(
            ((f"{b}|||{r}", int(n)) for (t, b, r), n in edges.items() if t == truth_label),
            key=lambda item: index[item[0]],
        )
        # If even the widest ribbon leaving this term is sub-hairline, all of them are
        # invisible and the term reads as connected to nothing. Rescue that one only.
        widest = max(destinations, key=lambda kv: kv[1])[0] if destinations else None
        rescue = widest if destinations and max(n for _, n in destinations) < hairline_units \
            else None
        for key, n in destinations:
            branch = right_branch[key]
            pooled = key.split("|||", 1)[1].startswith("Other ")
            # A rescued ribbon is drawn at full strength: pooled ribbons are faint by
            # design, and a hairline at that opacity would still not render.
            rescued = key == rescue
            _ribbon(flow, x_left_end, x_right,
                    (truth_cursor[truth_label], truth_cursor[truth_label] + n),
                    (right_cursor[key], right_cursor[key] + n),
                    BRANCH_COLORS[branch], 0.26 if (pooled and not rescued) else 0.52,
                    hairline_units if rescued else 0.0, HAIRLINE_PT)
            truth_cursor[truth_label] += n
            right_cursor[key] += n

    truth_colors = truth_colors or {}
    pred_colors = pred_colors or {}

    # The number is its own mark in a fixed-width column, not glued to the name:
    # "6" and "45" have different widths, so gluing them would ragged-align the names.
    axis_pt = fig.get_size_inches()[0] * flow.get_position().width * 72
    units_per_pt = (1.0 + 2 * gutter) / axis_pt
    badge_pt = badge_diameter_pt(max(numbering.values(), default=9), label_fontsize)
    number_width = badge_pt * units_per_pt
    number_pad = number_width * 0.22
    badge_s = badge_pt ** 2          # marker area in pt^2

    def draw_label(x_bar, mid, term, text, side, parent_marker=False, **style):
        """Number in its own right-aligned column; name aligned against that column."""
        n = numbering.get(term)
        if side == "left":                      # name ... (n) | bar
            # Right-aligned: names run 14-83 characters, so no fixed left x
            # would keep every label against its own bar.
            centre = x_bar - label_dx - number_width / 2
            name_x, name_ha = centre - number_width / 2 - number_pad, "right"
        else:                                   # bar | (n) ... name
            centre = x_bar + label_dx + number_width / 2
            name_x, name_ha = centre + number_width / 2 + number_pad, "left"
        if n is not None:
            flow.scatter([centre], [mid], marker="o", s=badge_s, facecolors="white",
                         edgecolors=BADGE_EDGE, linewidths=0.35, zorder=6, clip_on=False)
            flow.text(centre, mid, str(n), ha="center", va="center",
                      fontsize=label_fontsize, fontweight="bold", color="#111111", zorder=7)
        # Fit each complete name to its actual gutter; never abbreviate ontology terms.
        wrapped = wrapped_labels[(side, term)]
        flow.text(name_x, mid, wrapped, ha=name_ha, va="center",
                  fontsize=label_fontsize, linespacing=1.05, **style)
        if parent_marker:
            # Only the right-hand column is ever marked, where the name runs
            # left to right from name_x. Branch names never wrap (limit 58
            # characters, longest is 43), so the triangle stays on one line.
            width = _text_width_pt(wrapped.splitlines()[-1], label_fontsize,
                                   style.get("fontweight", "normal"))
            end = name_x + width * units_per_pt
            flow.scatter([end + PARENT_PAD_PT * units_per_pt], [mid], marker="<",
                         s=PARENT_S, c="#222222", linewidths=0, zorder=7, clip_on=False)

    for truth_label in truth_order:
        top, bottom = truth_pos[truth_label]
        color = truth_colors.get(truth_label, "#B8BEC7")
        mid = -(top + bottom) / 2
        flow.add_patch(Rectangle((x_left, -bottom), x_left_end - x_left, bottom - top,
                                 facecolor=color, edgecolor="none", zorder=5, clip_on=False))
        flow.scatter([x_left - chip_dx], [mid], marker="s", s=CHIP_S, c=[color],
                     linewidths=0, zorder=6, clip_on=False)
        draw_label(x_left, mid, truth_label,
                   f"{truth_label} ({truth_counts[truth_label]:,})", "left", color="#222222")

    for key in right_order:
        top, bottom = right_pos[key]
        branch = right_branch[key]
        label = key.split("|||", 1)[1]
        pooled = label.startswith("Other ")
        broad = label == branch
        # No outline on branch-level bars: at 1.5-2 pt tall a border would hide the colour.
        color = "#B5B5B5" if pooled else (pred_colors or {}).get(label, BRANCH_COLORS[branch])
        alpha = 0.55 if pooled else 1.0
        mid = -(top + bottom) / 2
        flow.add_patch(Rectangle(
            (x_right, -bottom), x_right_end - x_right, bottom - top,
            facecolor=color, edgecolor="none", linewidth=0, alpha=alpha, zorder=5, clip_on=False))
        # Keep pooled-row gray markers even when their repetitive labels are hidden.
        flow.scatter([x_right_end + chip_dx], [mid], marker="s", s=CHIP_S, c=[color],
                     alpha=alpha, linewidths=0, zorder=6, clip_on=False)
        if not pooled or label.startswith("Other ontology terms"):
            text = f"{label} ({right_counts[key]:,})"
            draw_label(x_right_end, mid, label, text, "right", parent_marker=broad,
                       color="#555555" if pooled else "#222222",
                       style="italic" if pooled else "normal",
                       fontweight="bold" if broad else "normal")

    # Offset the parallel headings toward their label gutters so that they
    # remain distinct even though the ribbons themselves occupy a narrow central span.
    for x, name in ((x_left_end / 2 - 0.7, "Ground truth"),
                    ((x_right + x_right_end) / 2 + 0.7, "HECTOR prediction")):
        flow.text(x, span * 0.008, name, ha="center", va="center",
                  fontsize=header_fontsize, fontweight="bold", linespacing=1.25)
    # Short ribbons, wide gutters: the cell-type names are the content of this panel.
    flow.set_xlim(-gutter, 1.0 + gutter)
    flow.set_ylim(-span * 1.015, span * 0.075)
    flow.axis("off")

    lineage_totals = {}
    for branch in branch_order:
        group = BRANCH_LINEAGE[branch]
        lineage_totals[group] = lineage_totals.get(group, 0) + int(branch_counts[branch])
    legend_axis.axis("off")
    legend_axis.legend(
        # The triangle is the same drawn marker the rows carry, so the key cannot show a
        # symbol the rows do not. Arial has the square, so that one stays a character.
        handles=[Patch(facecolor=LINEAGE_HUES[g], label=f"{g} ({n:,})")
                 for g, n in lineage_totals.items() if n > 0]
        + [Line2D([], [], linestyle="none", marker="<", markersize=2.4, color="#222222",
                  label="Parent-term prediction"),
           Patch(facecolor="none", edgecolor="none",
                 label="■ cell type (panel c); bar height: cells")],
        # Four compact columns keep both rows within panel d.
        loc="center right", bbox_to_anchor=(1, 0.5), ncol=4, frameon=False,
        fontsize=label_fontsize, handlelength=1.0,
        columnspacing=0.8, handletextpad=0.4, borderaxespad=0,
    )

    category_counts = table["category"].value_counts().reindex(SUMMARY_ORDER, fill_value=0)
    summary.set_xlim(0, total)
    summary.set_ylim(-0.55, 0.95)
    summary.axis("off")

    # A segment carries its own figure only when it fits; the narrowest category
    # (under 7 pt, figure needs 10) is read from the key instead.
    cell_pt = total / (fig.get_size_inches()[0] * summary.get_position().width * 72)
    cursor = 0
    for label in SUMMARY_ORDER:
        n = int(category_counts[label])
        summary.barh(0.45, n, left=cursor, height=0.55, color=SUMMARY_COLORS[label],
                     edgecolor="white", linewidth=0.5, clip_on=False)
        text = f"{n / total:.1%}"
        if n >= _text_width_pt(text, label_fontsize) * cell_pt * 1.15:
            summary.text(cursor + n / 2, 0.45, text, ha="center", va="center",
                         fontsize=label_fontsize, fontweight="bold",
                         color="white" if label in ("Exact match", "Correct, coarser term")
                         else "#222222")
        elif n / total >= 0.03:
            # The narrow segment fits regular-weight type without the bold-label padding.
            summary.text(cursor + n / 2, 0.45, text, ha="center", va="center",
                         fontsize=label_fontsize, color="#222222")
        cursor += n

    # Two rows below the bar preserve readable category names at the dense text size.
    summary.legend(
        handles=[Patch(facecolor=SUMMARY_COLORS[x], edgecolor="#767676", linewidth=0.4,
                       label=f"{x} ({category_counts[x] / total:.1%})") for x in SUMMARY_ORDER],
        loc="upper center", bbox_to_anchor=(total / 2, 0.02),
        bbox_transform=summary.transData, ncol=3, frameon=False,
        fontsize=label_fontsize, handlelength=1.0, columnspacing=1.0, handletextpad=0.4,
    )
    return flow, legend_axis, summary
