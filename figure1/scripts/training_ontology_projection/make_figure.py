"""Draw the training composition of a HECTOR model as one Voronoi treemap per species.

Each cell of a drawing is one Cell Ontology term the model was trained on, and its
area is the number of training cells of that type. Cells are grouped and coloured
by a broad Cell Ontology ancestor, one per term.

    python make_figure.py                 build everything
    python make_figure.py 3.5             set the cell type name size, in points
    python make_figure.py --swatches      also write the colour swatch sheet

One run writes the three tables the figure is drawn from, an interactive drawing
per species, and the printed figure, stopping at the first check that fails.

`voronoi_treemap.r` lays out each species for the screen, where every cell can be
hovered. The printed figure reads the geometry back out of those saved drawings
rather than laying it out again, so both versions are the same drawing. A drawing
is rebuilt only when the table it was drawn from changes, or the renderer does.

Reads  ../../input/training_ontology_projection/
Writes ../../result/training_ontology_projection/, ending with its figure.{pdf,png}
"""

import re
import subprocess
import sys
from itertools import combinations
from pathlib import Path


import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

import ontology

HERE = Path(__file__).resolve().parent
FIGURE = HERE.parents[1]          # figure1/, holding input/, scripts/ and result/
OUT_DIR = FIGURE / "result" / HERE.name
RENDERER = HERE / "voronoi_treemap.r"

DRAWINGS = {
    "human": OUT_DIR / "human" / "drawing.svg",
    "mouse": OUT_DIR / "mouse" / "drawing.svg",
}

# Smallest cell type name in the interactive drawing (940 units across); low,
# since a reader can zoom.
LABEL_MIN_FONT = 4.5

# Smallest cell the layout will draw, as a fraction of the largest cell beside
# it. Cells below it are silently enlarged. Cannot go much below 2e-3: the
# layout then fails to place the smallest term.
SMALLEST_CELL_RATIO = "3e-3"

PAGE_WIDTH_MM = 180.0   # widest a journal prints
EDGE_MM = 1.5
MIDDLE_GAP_MM = 5.0

# Distances below the circles: top of the species-name text, top of the counts
# line, and the middle of the key's top row.
SPECIES_NAME_MM = 1.86
SPECIES_COUNTS_MM = 5.85
KEY_TOP_MM = 11.16

LEGEND_COLUMNS = 5
KEY_ROW_MM = 2.28             # between one row of the key and the next
KEY_SWATCH_MM = 1.9

OUTSIDE_LABEL_BAND_MM = 6.8   # band above each circle for broad-term names

# A broad term is named in the colour key only above this percent of either
# species' cells; the rest are named around the circles and counted together
# on the key's last line.
KEY_PERCENT_ABOVE = 1.0

# Cell type name size in points, overridable on the command line. Below the
# journal print floor on purpose: the panel is read as a picture, and the
# complete cell-type list is a supplementary table.
CELL_NAME_PT = 3.0

# A cell type with at least this many training cells is named even if its name
# overruns its cell (see OVERRUN_ALLOWED). Set per species: the same absolute
# count is a different share of each training set.
FORCE_NAME_AT_OR_ABOVE = {"human": 50_000, "mouse": 10_000}

# How far a forced name may overrun its own cell, as a multiple of the room it
# has. Per species: the mean cell is larger in the smaller training set.
OVERRUN_ALLOWED = {"human": 1.45, "mouse": 1.80}

# Extra clearance (mm) tested around a name for overlap with another; negative
# allows some real overlap in exchange for naming more cell types.
NAME_GAP_MM = 0.0

# Whether to leave a cell type unnamed rather than overlap an already-placed
# name (largest population placed first).
KEEP_NAMES_APART = {"human": False, "mouse": True}

# Cell type names set by hand where the automatic break ran one name into its
# neighbour. Keyed by species too: two of these three types appear in both
# panels, and only the human one needs the override.
HAND_SET_CELL_NAMES = {
    ("human", "regular ventricular cardiac myocyte"): None,   # ran into `fast muscle cell`
    ("human", "glial cell"): ("glial", "cell"),               # ran into `radial glial cell`
    ("human", "cerebellar granule cell"): ("cerebellar", "granule cell"),
}

KEY_PT = 5.67
CAPTION_PT = 10.21            # the species name; the counts line beneath it is KEY_PT
SMALL_TERM_PT = 5.67
LINE_SPACING = 1.1

SMALL_TERM_MAX_LINES = 3
FONT = "Arial"                # the typeface all five figures are set in

SPECIES_TITLE = {"human": "Human", "mouse": "Mouse"}

# Reserves one of the solver's own positions for a single term. The key is the species,
# the vertical half of the circle, the horizontal side, and the height as a fraction of
# the radius. Empty, because every broad term named outside the circles is placed by
# the table below instead.
MANUAL_SMALL_TERM_PLACES = {}

# Where every broad term name outside the circles stands, and the two names
# left off. A position is a fraction of the radius from its own circle's
# centre (x right, y up), so it survives a change of page width or radius.
# `align` sets which edge of the text block the point is; `lines` overrides
# the automatic line break.
HAND_PLACED_SMALL_TERMS = {
    "pigment cell":      {"species": "mouse", "at": (+0.398, +1.009), "align": "center"},
    "kidney cell":       {"species": "mouse", "at": (-0.887, +0.521), "align": "right"},
    "notochordal cell":  {"species": "mouse", "at": (+0.751, +0.850), "align": "left"},
    "ectodermal cell":   {"species": "mouse", "at": (+0.767, +0.677), "align": "left",
                          "lines": ("ectodermal cell",)},
    "perivascular cell": {"species": "mouse", "at": (-0.490, -0.898), "align": "right"},
    "interstitial cell": {"species": "mouse", "at": (+0.616, -0.971), "align": "center"},
    "follicular cell of ovary": {"species": "mouse", "at": (-0.794, +0.706),
                                 "align": "right",
                                 "lines": ("follicular", "cell", "of ovary")},
    "extraembryonic cell": {"species": "mouse", "at": (-0.878, -0.545), "align": "right",
                            "lines": ("extraembryonic", "cell")},
    "phagocyte":         {"species": "mouse", "at": (-0.925, -0.397), "align": "right"},
    "transit amplifying cell": {"species": "human", "at": (+0.795, +0.706),
                                "align": "left",
                                "lines": ("transit", "amplifying", "cell")},
    "secretory cell":    {"species": "human", "at": (+0.335, -1.012), "align": "center"},
    "trophoblast cell":  {"species": "human", "at": (+0.639, -0.964), "align": "center"},
    "neuroplacodal cell": {"species": "human", "at": (-0.604, +0.846), "align": "right"},
    "supporting cell":   {"species": "human", "at": (+0.750, -0.835), "align": "center"},
    "germ line cell":    {"species": "human", "at": (-0.015, +1.041), "align": "center"},
    "cell":              {"species": "human", "at": (-0.519, +1.043), "align": "center"},
}

SMALL_TERMS_LEFT_UNNAMED = ("ciliated cell", "mesenchymal cell")

# How far a line to a hand-placed name runs into the letters, so a name
# standing hard against its own region still shows a visible connector.
HAND_PLACED_LINE_REACH_MM = 0.6

OFF_THE_PAGE_ALLOWED_MM = 2.0        # `notochordal cell`/`ectodermal cell` need this
INTO_A_CIRCLE_ALLOWED_MM = 0.4       # `phagocyte`'s empty corner cuts into the circle

# Broad term names whose line runs into the page margin and then vertically,
# for regions where the space beside them is too narrow for a straight line.
EDGE_VERTICAL_SMALL_TERM_LABELS = {
    "cell of skeletal muscle": {
        "species": "human",
        "sideways": -1,
        "height": -0.644,
        "lines": ("cell of", "skeletal", "muscle"),
    },
    "bladder cell": {
        "species": "mouse",
        "sideways": 1,
        "height": -0.55,
        "lines": ("bladder", "cell"),
    },
}


# --------------------------------------------------------- making the drawing

def cells_table(assignments, colours):
    """Build the table the renderer draws from: one row per training cell type.

    Three levels: a single root, then the broad Cell Ontology terms, then the cell
    types. Each cell type carries its own training cell count and no other's, which
    is what keeps a cell from being counted under more than one term. Ordered
    largest broad term first, and largest cell type first within each.
    """
    missing = set(assignments["high_level_cl_id"]).difference(colours)
    if missing:
        raise ValueError(f"no colour for {', '.join(sorted(missing))}")

    table = assignments.assign(
        h1="All_Cells",
        h2=assignments["high_level_cl_name"],
        h3=assignments["cell_type"],
        color=assignments["high_level_cl_id"].map(colours),
        weight=assignments["cells_used_in_training"],
        codes=assignments["cells_used_in_training"].astype(str),
    )
    order = table.groupby("h2")["weight"].transform("sum")
    table = table.assign(_group=order).sort_values(
        ["_group", "h2", "weight"], ascending=[False, True, False])
    return table[["h1", "h2", "h3", "color", "weight", "codes"]]


def make_drawing(species, assignments, colours):
    """Render one species for the screen, reusing the saved drawing if it is current.

    A drawing is current when the table it would be built from is identical to the
    one saved beside it and the renderer has not changed since. Returns True if it
    was rendered, False if the saved one was kept.
    """
    where = OUT_DIR / species
    where.mkdir(parents=True, exist_ok=True)
    table_path, drawing = where / "cells.csv", where / "drawing.svg"

    wanted = cells_table(assignments, colours).to_csv(index=False)
    print(f"{species}: {wanted.count(chr(10)) - 1} cell types, each once, "
          f"{assignments['cells_used_in_training'].sum():,} cells, "
          f"{assignments['high_level_cl_name'].nunique()} broad terms")

    current = (drawing.exists() and table_path.exists()
               and table_path.read_text() == wanted
               and drawing.stat().st_mtime >= RENDERER.stat().st_mtime)
    if current:
        print("  the drawing on disk was made from this table; keeping it")
        return False

    table_path.write_text(wanted)
    result = subprocess.run(
        ["Rscript", str(RENDERER), str(table_path), str(where), "drawing",
         str(LABEL_MIN_FONT), SMALLEST_CELL_RATIO],
        capture_output=True, text=True, timeout=1800)
    for line in result.stdout.splitlines():
        if line.startswith("✓") or line.startswith("ERROR"):
            print(" ", line)
    if result.returncode != 0:
        raise SystemExit(f"the renderer failed:\n{result.stderr}")
    return True


# ------------------------------------------------------------------ the drawing

def read_drawing(path):
    """Read the cells back out of a saved drawing.

    Each cell carries its cell type, its training cell count and its broad term,
    written onto the SVG by the renderer.
    """
    svg = path.read_text()
    cells = []
    for tag in re.findall(r"<path class=\"cell\"[^>]*>", svg):
        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?",
                             re.search(r' d="([^"]*)"', tag).group(1))
        red, green, blue = re.search(r"fill: rgb\((\d+), (\d+), (\d+)\)", tag).groups()
        cells.append({
            "points": np.array(numbers, dtype=float).reshape(-1, 2),
            "colour": (int(red) / 255, int(green) / 255, int(blue) / 255),
            "cell_type": re.search(r'data-cell-type="([^"]*)"', tag).group(1),
            "count": int(re.search(r'data-cells="([^"]*)"', tag).group(1)),
            "broad_term": re.search(r'data-broad-term="([^"]*)"', tag).group(1),
        })
    if not cells:
        raise ValueError(f"{path} holds no cells. Rerun the R pipeline; the drawing has to "
                         "carry data-cell-type and data-broad-term on every cell.")
    return cells


def place_on_page(cells, centre_x, centre_y, radius_mm):
    """Move a drawing onto the page, keeping it round and centred."""
    everything = np.vstack([cell["points"] for cell in cells])
    middle = (everything.min(axis=0) + everything.max(axis=0)) / 2
    reach = np.abs(everything - middle).max()
    for cell in cells:
        moved = (cell["points"] - middle) / reach * radius_mm
        # The drawing counts downwards, the page counts upwards.
        cell["page"] = np.column_stack([moved[:, 0] + centre_x, centre_y - moved[:, 1]])


def outline_of(cells):
    """The outer edges of a set of cells.

    The cells tile their region exactly, so an edge shared by two of them is
    interior and every edge appearing once is on the outside.
    """
    seen = {}
    for cell in cells:
        points = cell["page"]
        for i in range(len(points)):
            a, b = points[i], points[(i + 1) % len(points)]
            key = tuple(sorted((tuple(np.round(a, 3)), tuple(np.round(b, 3)))))
            seen[key] = seen.get(key, 0) + 1
    return [np.array(edge) for edge, times in seen.items() if times == 1]


# ------------------------------------------------------------ where a name goes

def roomiest_point(points, step_mm):
    """The point inside a polygon furthest from its edge, and that distance.

    Found on a grid of `step_mm`. This is where a name can be set largest, which in
    a long thin cell is not its centroid.
    """
    from matplotlib.path import Path as MplPath

    low, high = points.min(axis=0) - step_mm, points.max(axis=0) + step_mm
    width = max(int((high[0] - low[0]) / step_mm), 3)
    height = max(int((high[1] - low[1]) / step_mm), 3)
    xs = np.linspace(low[0], high[0], width)
    ys = np.linspace(low[1], high[1], height)
    grid_x, grid_y = np.meshgrid(xs, ys)

    inside = MplPath(points).contains_points(
        np.column_stack([grid_x.ravel(), grid_y.ravel()])).reshape(height, width)
    if not inside.any():
        return None, 0.0

    room = distance_transform_edt(inside, sampling=[ys[1] - ys[0], xs[1] - xs[0]])
    best = np.unravel_index(np.argmax(room), room.shape)
    return (xs[best[1]], ys[best[0]]), float(room[best])


def measure(figure, text, size_pt, bold=False):
    """How wide a piece of text is, in millimetres."""
    holder = figure.text(0, 0, text, fontsize=size_pt, fontfamily=FONT,
                         fontweight="bold" if bold else "normal")
    width = holder.get_window_extent(figure.canvas.get_renderer()).width
    holder.remove()
    return width / figure.dpi * 25.4


def break_into(words, how_many_lines, width_of=len):
    """Split a name over a fixed number of lines, keeping the widest line narrowest.

    `width_of` judges the width of a line: letter count by default, which is enough
    when a name is only being compared against itself, or a function measuring
    rendered width when the name has to fit a known space.
    """
    if how_many_lines > len(words):
        return None
    best, best_width = None, np.inf

    for cuts in combinations(range(1, len(words)), how_many_lines - 1):
        edges = (0, *cuts, len(words))
        lines = [" ".join(words[edges[i]:edges[i + 1]]) for i in range(how_many_lines)]
        width = max(width_of(line) for line in lines)
        if width < best_width:
            best, best_width = lines, width
    return best


def name_shapes(figure, name, room_mm, size_pt, allowed_overrun, may_overrun):
    """Every way of breaking a name that its cell will accept, best first.

    Returns two lists of (lines, half_wide, half_tall). The first holds the breaks
    that sit inside the cell, fewest lines first; the second those that extend past
    it by no more than `allowed_overrun`, squarest first, and is empty unless
    `may_overrun`.

    What has to fit is the half diagonal of the text block, since a cell's room is
    measured as a radius. Line height comes from the type size rather than from
    measuring the glyphs, which differ line to line.
    """
    words = name.split()
    line_mm = size_pt / 72 * 25.4
    fitting, overrunning = [], []
    for line_count in range(1, min(6, len(words)) + 1):
        lines = break_into(words, line_count)
        if lines is None:
            break
        half_wide = max(measure(figure, line, size_pt) for line in lines) / 2
        half_tall = line_count * LINE_SPACING * line_mm / 2
        reach = float(np.hypot(half_wide, half_tall))
        if line_count <= 4 and reach <= room_mm * 0.92:
            fitting.append((lines, half_wide, half_tall))
        elif may_overrun and reach <= room_mm * allowed_overrun:
            overrunning.append((reach, lines, half_wide, half_tall))

    # Squarest first: that is the one spilling out of its cell by the least.
    overrunning.sort()
    return fitting, [shape[1:] for shape in overrunning]


# -------------------------------------------------------------------- the panel

def draw_species(axes, figure, species, cells, centre_x, centre_y, radius_mm, name_pt):
    """Draw one species' circle, name its cell types and caption it.

    A cell type is named if its name fits its cell, or if the model saw at least
    FORCE_NAME_AT_OR_ABOVE cells of it and the name stays within OVERRUN_ALLOWED.
    Returns a summary of how many were named and why the rest were not.
    """
    axes.add_collection(PolyCollection(
        [cell["page"] for cell in cells],
        facecolors=[cell["colour"] for cell in cells],
        edgecolors="white", linewidths=0.12, clip_on=False))

    groups = {}
    for cell in cells:
        groups.setdefault(cell["broad_term"], []).append(cell)
    for members in groups.values():
        edges = outline_of(members)
        if edges:
            axes.add_collection(LineCollection(edges, colors="white", linewidths=0.7,
                                               clip_on=False))

    line_mm = name_pt / 72 * 25.4
    named, named_cells, forced, crowded_out = 0, 0, 0, 0
    written = []
    for cell in sorted(cells, key=lambda c: -c["count"]):
        set_by_hand = (species, cell["cell_type"]) in HAND_SET_CELL_NAMES
        if set_by_hand and HAND_SET_CELL_NAMES[species, cell["cell_type"]] is None:
            continue

        name_it_anyway = cell["count"] >= FORCE_NAME_AT_OR_ABOVE[species]
        where, room = roomiest_point(cell["page"], step_mm=0.08)
        if where is None or (room < line_mm * 0.55 and not name_it_anyway):
            continue

        if set_by_hand:
            # The name still sits in the roomiest part of its own cell; only where it
            # breaks was settled by hand.
            lines = list(HAND_SET_CELL_NAMES[species, cell["cell_type"]])
            shapes = [(lines,
                       max(measure(figure, line, name_pt) for line in lines) / 2,
                       len(lines) * LINE_SPACING * line_mm / 2)]
            fitting = shapes
        else:
            fitting, overrunning = name_shapes(figure, cell["cell_type"], room, name_pt,
                                               OVERRUN_ALLOWED[species], name_it_anyway)
            shapes = fitting + overrunning
        if not shapes:
            continue

        # Cells are worked through largest population first, so when two names want
        # the same space the larger population keeps it. The other is tried in its
        # remaining shapes before it is given up.
        chosen = None
        for index, (lines, half_wide, half_tall) in enumerate(shapes):
            box = (where[0] - half_wide - NAME_GAP_MM, where[1] - half_tall - NAME_GAP_MM,
                   where[0] + half_wide + NAME_GAP_MM, where[1] + half_tall + NAME_GAP_MM)
            clear = not any(box[0] < other[2] and other[0] < box[2]
                            and box[1] < other[3] and other[1] < box[3]
                            for other in written)
            if clear or not KEEP_NAMES_APART[species]:
                chosen = (lines, box, index >= len(fitting))
                break
        if chosen is None:
            crowded_out += 1
            continue
        lines, box, overran = chosen
        written.append(box)
        forced += overran

        # Dark type on the pale cells, light type on the deep ones.
        red, green, blue = cell["colour"]
        ink = "#111111" if 0.299 * red + 0.587 * green + 0.114 * blue > 0.62 else "white"
        top = where[1] + (len(lines) - 1) / 2 * line_mm * LINE_SPACING
        for i, line in enumerate(lines):
            axes.text(where[0], top - i * line_mm * LINE_SPACING, line,
                      ha="center", va="center", fontsize=name_pt, fontfamily=FONT,
                      color=ink, zorder=5)
        named += 1
        named_cells += cell["count"]

    total = sum(cell["count"] for cell in cells)
    axes.text(centre_x, centre_y - radius_mm - SPECIES_NAME_MM, SPECIES_TITLE[species],
              ha="center", va="top", fontsize=CAPTION_PT, fontfamily=FONT, color="black")
    axes.text(centre_x, centre_y - radius_mm - SPECIES_COUNTS_MM,
              f"{len(cells)} cell types  ·  {total:,} cells",
              ha="center", va="top", fontsize=KEY_PT, fontfamily=FONT, color="#333333")

    return {"named": named, "named_cells": named_cells, "forced": forced,
            "crowded_out": crowded_out, "total": total, "cell_types": len(cells)}


# ------------------------------------------------- naming the rare broad terms

def stands_clear(place_y, centre_y, radius_mm, half_tall_mm, gap_mm):
    """How far from the circle's centre line a block of text must start to clear it.

    A circle is widest at its middle, so what has to clear it is whichever line of
    the block sits nearest that middle, not the block's centre line. Every point of
    the block is then at least `gap_mm` outside the circle.
    """
    nearest_line = max(abs(place_y - centre_y) - half_tall_mm, 0.0)
    return np.sqrt(max((radius_mm + gap_mm) ** 2 - nearest_line ** 2, 0.0))


def region_outline(cells, term):
    """The outer edge of every cell belonging to one broad term, and their centroid.

    A line to the term's name starts on this edge rather than at the centroid, so it
    does not cross the region on its way out.
    """
    polygons = [cell["page"] for cell in cells if cell["broad_term"] == term]
    if not polygons:
        return None

    seen, keep = {}, {}
    for polygon in polygons:
        for index in range(len(polygon)):
            a, b = polygon[index], polygon[(index + 1) % len(polygon)]
            key = tuple(sorted((tuple(np.round(a, 3)), tuple(np.round(b, 3)))))
            seen[key] = seen.get(key, 0) + 1
            keep[key] = np.array([a, b])

    return {"edges": [keep[key] for key, times in seen.items() if times == 1],
            "centre": np.vstack(polygons).mean(axis=0)}


def nearest_outline_point(edges, target):
    """The closest point on a region's outside edge to where the name will sit."""
    closest, closest_distance = None, np.inf
    for a, b in edges:
        along = b - a
        length_squared = np.dot(along, along)
        fraction = 0.0 if length_squared == 0 else np.dot(target - a, along) / length_squared
        candidate = a + np.clip(fraction, 0.0, 1.0) * along
        distance = np.linalg.norm(target - candidate)
        if distance < closest_distance:
            closest, closest_distance = candidate, distance
    return closest, closest_distance


def candidate_places(centres, centre_y, radius_mm):
    """Every position a broad term's name is allowed to take.

    Three heights up each side of each circle and three across the top and bottom of
    each, plus the positions fixed by MANUAL_SMALL_TERM_PLACES and
    EDGE_VERTICAL_SMALL_TERM_LABELS. A position carrying `only_for` is reserved for
    that one term.
    """
    places = []
    heights = (0.84, 0.70, 0.57)
    for species in ("human", "mouse"):
        for upright in (1, -1):
            for sideways in (1, -1):
                for height in heights:
                    place = {
                        "kind": "beside",
                        "where": (species, upright, sideways),
                        "y": centre_y + upright * radius_mm * height,
                    }
                    placement_key = (species, upright, sideways, round(float(height), 2))
                    if placement_key in MANUAL_SMALL_TERM_PLACES:
                        place["only_for"] = MANUAL_SMALL_TERM_PLACES[placement_key]
                    places.append(place)
            for across in (-0.52, 0.0, 0.52):
                places.append({
                    "kind": "above or below",
                    "where": (species, upright, 0),
                    "x": centres[species] + across * radius_mm,
                    "y": centre_y + upright * (radius_mm + 0.7),
                    "available_mm": radius_mm * 0.44,
                })

    # Names in the page margin, whose line leaves the circle sideways and then runs
    # vertically to them, rather than straight across the drawing.
    for term, specification in EDGE_VERTICAL_SMALL_TERM_LABELS.items():
        sideways = specification["sideways"]
        upright = 1 if specification["height"] > 0 else -1
        label_x = 0.3 if sideways < 0 else PAGE_WIDTH_MM - 0.3
        places.append({
            "kind": "edge vertical",
            "where": (specification["species"], upright, sideways),
            "x": label_x,
            "y": centre_y + radius_mm * specification["height"],
            "lines": specification["lines"],
            "only_for": term,
        })

    # The space between the two circles widens quickly below their middle, so these
    # names can sit nearer their regions than the positions above allow.
    places.extend([
        {"kind": "between the circles", "where": ("mouse", -1, -1),
         "y": centre_y - radius_mm * 0.55, "only_for": "extraembryonic cell"},
        {"kind": "between the circles", "where": ("mouse", -1, -1),
         "y": centre_y - radius_mm * 0.42, "only_for": "phagocyte"},
    ])
    return places


def try_place(figure, term, place, region, centres, centre_y, radius_mm,
              size_pt, max_lines, gap_mm):
    """Where a name and its line would fall in one position, and what that costs.

    Returns None if the name will not fit there. How many lines the name takes and
    how far out it has to stand are settled together, since each depends on the
    other: more lines make the block taller, a taller block reaches nearer the
    circle's widest point, and standing further out to clear it leaves less width.
    """
    species, upright, sideways = place["where"]

    if place["kind"] == "edge vertical":
        lines = list(place["lines"])
        outline_points = np.vstack(region["edges"])
        if sideways < 0:
            start = outline_points[np.argmin(outline_points[:, 0])]
            horizontal_alignment = "left"
        else:
            start = outline_points[np.argmax(outline_points[:, 0])]
            horizontal_alignment = "right"

        turn = np.array([place["x"], start[1]])
        if upright > 0:
            end = np.array([place["x"], place["y"] - 0.8])
            vertical_alignment = "bottom"
        else:
            end = np.array([place["x"], place["y"] + 0.8])
            vertical_alignment = "top"
        line_points = np.vstack([start, turn, end])
        line_length = np.linalg.norm(turn - start) + np.linalg.norm(end - turn)

        return {
            "start": start,
            "end": end,
            "line_points": line_points,
            "line_length": line_length,
            "text": {"lines": lines, "line_count": len(lines), "reach": 0.0},
            "draw": {
                "x": place["x"],
                "y": place["y"],
                "ha": horizontal_alignment,
                "va": vertical_alignment,
            },
            "cost": line_length,
        }

    beside = place["kind"] != "above or below"
    words = term.split()
    line_mm = size_pt / 72 * 25.4 * 1.05
    width_of = lambda line: measure(figure, line, size_pt)

    fitted = None
    for line_count in range(1, min(max_lines, len(words)) + 1):
        lines = break_into(words, line_count, width_of=width_of)
        width = max(width_of(line) for line in lines)

        if beside:
            reach = stands_clear(place["y"], centre_y, radius_mm,
                                 line_count * line_mm / 2, gap_mm)
            if place["kind"] == "between the circles":
                available_mm = centres["mouse"] - centres["human"] - 2 * reach
            else:
                available_mm = radius_mm - reach
        else:
            reach, available_mm = 0.0, place["available_mm"]

        if width <= available_mm:
            fitted = {"lines": lines, "line_count": line_count, "reach": reach}
            break
    if fitted is None:
        return None
    text = fitted

    if beside:
        # The line stops at the edge of the circle, not out where the name stands.
        edge_x = np.sqrt(max(radius_mm ** 2 - (place["y"] - centre_y) ** 2, 0.0))
        end = np.array([centres[species] + sideways * edge_x, place["y"]])
        draw = {"x": centres[species] + sideways * text["reach"], "y": place["y"],
                "ha": "left" if sideways > 0 else "right", "va": "center"}
    else:
        edge_y = centre_y + upright * radius_mm
        end = np.array([place["x"], edge_y + upright * gap_mm / 2])
        draw = {"x": place["x"], "y": edge_y + upright * gap_mm,
                "ha": "center", "va": "bottom" if upright > 0 else "top"}

    start, line_length = nearest_outline_point(region["edges"], end)

    # A name on the opposite side of the circle from its own region is charged the
    # whole radius, so it goes there only when nothing else is left.
    offset = region["centre"] - np.array([centres[species], centre_y])
    right_half = upright * offset[1] > 0 and (sideways == 0 or sideways * offset[0] > 0)
    cost = line_length + (0.0 if right_half else radius_mm) + 0.15 * (text["line_count"] - 1)
    return {
        "start": start,
        "end": end,
        "line_points": np.vstack([start, end]),
        "line_length": line_length,
        "text": text,
        "draw": draw,
        "cost": cost,
    }


def place_by_hand(figure, term, specification, region, centres, centre_y, radius_mm):
    """Where a hand placed name and its line fall.

    The name is set at the point recorded for it, and its line runs from the nearest
    point on its region's outline to the side of the block of text facing that region,
    ending a little way in under the letters.
    """
    lines = list(specification.get("lines", (term,)))
    line_mm = SMALL_TERM_PT / 72 * 25.4 * 1.05
    wide = max(measure(figure, line, SMALL_TERM_PT) for line in lines)
    tall = len(lines) * line_mm

    sideways, height = specification["at"]
    x = centres[specification["species"]] + sideways * radius_mm
    y = centre_y + height * radius_mm
    left = {"left": x, "right": x - wide, "center": x - wide / 2}[specification["align"]]

    reach = HAND_PLACED_LINE_REACH_MM
    box = np.array([[left + reach, y - tall / 2 + reach],
                    [left + wide - reach, y + tall / 2 - reach]])
    end = np.clip(region["centre"], box[0], box[1])
    start, line_length = nearest_outline_point(region["edges"], end)

    return {
        "line_points": np.vstack([start, end]),
        "line_length": float(line_length),
        "text": {"lines": lines, "line_count": len(lines)},
        "draw": {"x": x, "y": y, "ha": specification["align"], "va": "center"},
    }


def name_small_terms(axes, figure, drawn, centres, centre_y, radius_mm, terms):
    """Name the broad terms too small for the key, in the space around the circles.

    The names in HAND_PLACED_SMALL_TERMS go where they were put by hand and the two in
    SMALL_TERMS_LEFT_UNNAMED are not drawn. The rest are assigned all at once,
    minimising total line length, rather than each taking its own best position in
    turn; taking them in turn crowds several into one corner and leaves others with
    nowhere to go. Returns the text artists, any terms that could not be placed, and
    each line's length, longest first.
    """
    gap_mm = 0.7
    places = candidate_places(centres, centre_y, radius_mm)
    regions = {(term, species): region_outline(drawn[species], term)
               for term, *_ in terms for species in ("human", "mouse")}

    by_hand = [term for term in terms if term[0] in HAND_PLACED_SMALL_TERMS]
    terms = [term for term in terms
             if term[0] not in HAND_PLACED_SMALL_TERMS
             and term[0] not in SMALL_TERMS_LEFT_UNNAMED]

    impossible = 1e6
    costs = np.full((len(terms), len(places)), impossible)
    worked_out = {}
    manually_placed_terms = set(MANUAL_SMALL_TERM_PLACES.values())
    manually_placed_terms.update(EDGE_VERTICAL_SMALL_TERM_LABELS)
    for i, (term, *_) in enumerate(terms):
        for j, place in enumerate(places):
            if term in manually_placed_terms and place.get("only_for") != term:
                continue
            if place.get("only_for", term) != term:
                continue
            region = regions[(term, place["where"][0])]
            if region is None:
                continue
            found = try_place(figure, term, place, region, centres, centre_y,
                              radius_mm, SMALL_TERM_PT, SMALL_TERM_MAX_LINES, gap_mm)
            if found is not None:
                costs[i, j] = found["cost"]
                worked_out[(i, j)] = found

    settled = []
    for i, j in zip(*linear_sum_assignment(costs)):
        settled.append((terms[i], worked_out.get((i, j))
                        if costs[i, j] < impossible else None))
    for term in by_hand:
        specification = HAND_PLACED_SMALL_TERMS[term[0]]
        settled.append((term, place_by_hand(
            figure, term[0], specification, regions[(term[0], specification["species"])],
            centres, centre_y, radius_mm)))

    unplaced, lengths, written = [], [], []
    for (term, colour, *_), found in settled:
        if found is None:
            unplaced.append(term)
            continue
        line_points = found["line_points"]
        axes.plot(line_points[:, 0], line_points[:, 1],
                  color=colour, linewidth=0.4, solid_capstyle="round", zorder=4,
                  clip_on=False)
        written.append(axes.text(
            found["draw"]["x"], found["draw"]["y"], "\n".join(found["text"]["lines"]),
            ha=found["draw"]["ha"], va=found["draw"]["va"], fontsize=SMALL_TERM_PT,
            fontfamily=FONT, color=colour, linespacing=1.05, zorder=5))
        lengths.append((term, float(found["line_length"])))

    return written, unplaced, sorted(lengths, key=lambda pair: -pair[1])


# ---------------------------------------------------------------- the page

def draw_key(axes, figure, terms, top_y, rarer):
    """The colour key: each broad term, and how many cell types it holds per species.

    Counts rather than shares of the cells, since area in the drawing already is the
    cell count, and counts reconcile with the totals printed under each panel.

    The two numbers are in the order the panels stand above them, first species on
    the left, and are not headed. Column headings would be nearly twice the width of
    the numbers they head and cost a whole row.
    """
    swatch, after_swatch, before_numbers, between_numbers = KEY_SWATCH_MM, 0.8, 2.5, 2.0
    row_mm = KEY_ROW_MM
    number_mm = measure(figure, "888", KEY_PT)

    entries = [(name, colour, human, mouse, "#111111")
               for name, colour, human, mouse in terms]
    entries.append((f"{rarer[0]} rarer terms", None, rarer[1], rarer[2], "#555555"))
    rows = int(np.ceil(len(entries) / LEGEND_COLUMNS))
    in_column = lambda column: entries[column::LEGEND_COLUMNS]

    # Each column is only as wide as its own longest name, so a short name
    # doesn't sit further from its own numbers than the columns stand apart.
    widths = [swatch + after_swatch + max(measure(figure, entry[0], KEY_PT)
                                          for entry in in_column(column))
              + before_numbers + 2 * number_mm + between_numbers
              for column in range(LEGEND_COLUMNS)]
    between_columns = (PAGE_WIDTH_MM - 2 * EDGE_MM - sum(widths)) / (LEGEND_COLUMNS - 1)
    lefts = np.concatenate([[EDGE_MM],
                            EDGE_MM + np.cumsum(np.array(widths) + between_columns)])

    # A name must sit closer to its own numbers than to the next column.
    worst = max(width - swatch - after_swatch - before_numbers - 2 * number_mm
                - between_numbers - measure(figure, entry[0], KEY_PT)
                for width, column in zip(widths, range(LEGEND_COLUMNS))
                for entry in in_column(column))
    if worst + before_numbers >= between_columns:
        raise ValueError(f"the key reads as columns of numbers, not as terms: a name sits "
                         f"{worst + before_numbers:.1f} mm from its own numbers but only "
                         f"{between_columns:.1f} mm from the next column")

    for slot, (name, colour, human, mouse, ink) in enumerate(entries):
        row, column = divmod(slot, LEGEND_COLUMNS)
        x, y = lefts[column], top_y - row * row_mm
        if colour is not None:
            axes.add_patch(plt.Rectangle((x, y - swatch / 2), swatch, swatch,
                                         facecolor=colour, edgecolor="#9a9a9a",
                                         linewidth=0.2, clip_on=False))
        axes.text(x + swatch + after_swatch, y, name, ha="left", va="center",
                  fontsize=KEY_PT, fontfamily=FONT, color=ink)
        right = x + widths[column]
        for count, at in zip((human, mouse),
                             (right - number_mm - between_numbers, right)):
            axes.text(at, y, f"{count:,}", ha="right", va="center",
                      fontsize=KEY_PT, fontfamily=FONT, color=ink)


def check(figure, axes, drawn, name_pt, outside_names, centres, centre_y, radius_mm):
    """Refuse to save a figure that is wrong or unreadable."""
    trouble = []

    # A broad term named outside the circles has to stay outside them. What is measured
    # is the rectangle the name is set in, whose corners are empty, so a circle's edge
    # cutting a corner off is allowed a little way before it reaches any letter.
    inverse = axes.transData.inverted()
    for text in outside_names:
        box = text.get_window_extent(figure.canvas.get_renderer())
        low = inverse.transform((box.x0, box.y0))
        high = inverse.transform((box.x1, box.y1))
        for centre_x in centres.values():
            near = np.array([min(max(centre_x, low[0]), high[0]),
                             min(max(centre_y, low[1]), high[1])])
            into = radius_mm - np.hypot(*(near - np.array([centre_x, centre_y])))
            if into > INTO_A_CIRCLE_ALLOWED_MM:
                trouble.append(f"the name {text.get_text()!r} runs {into:.2f} mm "
                               "into a circle")

    for species, cells in drawn.items():
        def area(points):
            return abs(np.dot(points[:, 0], np.roll(points[:, 1], -1)) -
                       np.dot(points[:, 1], np.roll(points[:, 0], -1))) / 2
        areas = np.array([area(cell["page"]) for cell in cells])
        counts = np.array([cell["count"] for cell in cells], dtype=float)
        off = np.abs(areas / areas.sum() - counts / counts.sum())
        # The layout settles by repeated passes and stops while cells are still
        # slightly off their target area. This is the most that is accepted.
        if off.max() > 0.003:
            trouble.append(f"{species}: a cell's area is off its cell count by "
                           f"{100 * off.max():.2f} percentage points")

    renderer = figure.canvas.get_renderer()
    page = figure.get_window_extent(renderer)
    dots_per_mm = figure.dpi / 25.4
    for text in axes.texts:
        if text.get_fontsize() < name_pt - 0.01:
            trouble.append(f"type at {text.get_fontsize():.2f} pt: {text.get_text()!r}")
        # Two hand placed broad term names stand a little past the side of the page.
        # It is where they were put, and this is the panel's own edge rather than the
        # trimmed page: the panel is placed into the figure by hand and scaled there.
        room = (1 + OFF_THE_PAGE_ALLOWED_MM * dots_per_mm
                if text.get_text() in HAND_PLACED_SMALL_TERMS else 1)
        box = text.get_window_extent(renderer)
        if (box.x0 < page.x0 - room or box.x1 > page.x1 + room
                or box.y0 < page.y0 - 1 or box.y1 > page.y1 + 1):
            over = max(page.x0 - box.x0, box.x1 - page.x1,
                       page.y0 - box.y0, box.y1 - page.y1) / dots_per_mm
            trouble.append(f"text runs {over:.2f} mm off the page: {text.get_text()!r}")

    if trouble:
        raise ValueError("the figure was not saved:\n  " + "\n  ".join(trouble))


def main():
    sizes = [word for word in sys.argv[1:] if not word.startswith("-")]
    name_pt = float(sizes[0]) if sizes else CELL_NAME_PT
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tables first: they carry the checks that no cell type is counted twice and
    # that the totals match what the model trained on. Nothing is drawn until those
    # pass.
    assignments, coloured = ontology.write_tables()
    if "--swatches" in sys.argv:
        ontology.write_swatch_sheet(coloured)

    colours = {broad_id: colour for broad_id, _, colour, _ in coloured}
    print()
    for species in ontology.SPECIES:
        make_drawing(species, assignments[assignments["species"] == species], colours)
    print()

    # How many training cell types each broad term holds, which is what the key
    # gives. Which terms the key names is settled by their share of the cells,
    # since that is what decides how much of the drawing they occupy.
    held = (assignments.groupby(["high_level_cl_name", "species"]).size()
            .unstack(fill_value=0).reindex(columns=list(ontology.SPECIES), fill_value=0))
    in_the_key = {name for _, name, _, split in coloured
                  if max(split.values()) >= KEY_PERCENT_ABOVE}
    entry = lambda name, colour: (name, colour, int(held.at[name, "human"]),
                                  int(held.at[name, "mouse"]))

    # Ordered by the first species' cell type count, so that column reads in order.
    # The second cannot also be in order, and the difference is itself informative.
    key_terms = sorted((entry(name, colour) for _, name, colour, _ in coloured
                        if name in in_the_key), key=lambda term: (-term[2], -term[3]))
    rest = [entry(name, colour) for _, name, colour, _ in coloured
            if name not in in_the_key]
    rarer = (len(rest), sum(term[2] for term in rest), sum(term[3] for term in rest))
    key_rows = int(np.ceil((len(key_terms) + 1) / LEGEND_COLUMNS))

    radius_mm = (PAGE_WIDTH_MM - 2 * EDGE_MM - MIDDLE_GAP_MM) / 4
    # From the bottom of a circle down to the bottom of the page: the species name and
    # the counts line beneath it stand within the first part of this, then the key,
    # whose last row holds the line counting the terms it does not name.
    below_circles = (KEY_TOP_MM + (key_rows - 1) * KEY_ROW_MM
                     + KEY_SWATCH_MM / 2 + EDGE_MM)
    page_height = (EDGE_MM + OUTSIDE_LABEL_BAND_MM + 2 * radius_mm + below_circles)

    figure = plt.figure(figsize=(PAGE_WIDTH_MM / 25.4, page_height / 25.4))
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_xlim(0, PAGE_WIDTH_MM)
    axes.set_ylim(0, page_height)
    axes.set_aspect("equal")
    axes.axis("off")
    figure.canvas.draw()

    centre_y = page_height - EDGE_MM - OUTSIDE_LABEL_BAND_MM - radius_mm
    centres = {
        "human": EDGE_MM + radius_mm,
        "mouse": PAGE_WIDTH_MM - EDGE_MM - radius_mm,
    }

    drawn, panels = {}, {}
    for species, path in DRAWINGS.items():
        cells = read_drawing(path)
        place_on_page(cells, centres[species], centre_y, radius_mm)
        drawn[species] = cells
        panels[species] = draw_species(axes, figure, species, cells,
                                       centres[species], centre_y, radius_mm, name_pt)

    outside_names, unplaced, lengths = name_small_terms(
        axes, figure, drawn, centres, centre_y, radius_mm, rest)
    draw_key(axes, figure, key_terms, centre_y - radius_mm - KEY_TOP_MM, rarer)

    wanted = len(rest) - len(SMALL_TERMS_LEFT_UNNAMED)
    print(f"  {len(outside_names)} of {wanted} small broad terms named around the circles"
          f" ({len(HAND_PLACED_SMALL_TERMS)} of them where they were put by hand;"
          f" {', '.join(SMALL_TERMS_LEFT_UNNAMED)} left unnamed)"
          + (f"; not placed: {', '.join(unplaced)}" if unplaced else ""))
    print(f"  lines to those names: {sum(l for _, l in lengths):.1f} mm total, "
          f"{lengths[0][1]:.1f} mm longest")
    print("  longest: " + ", ".join(f"{term} ({length:.1f} mm)"
                                    for term, length in lengths[:5]))

    figure.canvas.draw()
    check(figure, axes, drawn, name_pt, outside_names, centres, centre_y, radius_mm)


    figure.savefig(OUT_DIR / "figure.pdf")
    figure.savefig(OUT_DIR / "figure.png", dpi=600)
    plt.close(figure)

    print(f"page {PAGE_WIDTH_MM:.0f} x {page_height:.1f} mm, each circle "
          f"{2 * radius_mm:.1f} mm across, cell names at {name_pt} pt")
    for species, panel in panels.items():
        print(f"  {species}: {panel['named']} of {panel['cell_types']} cell types named, "
              f"holding {100 * panel['named_cells'] / panel['total']:.1f}% of the cells "
              f"({panel['forced']} of them overrunning their cell a little; "
              f"{panel['crowded_out']} left unnamed to keep off another name)")
    print(f"wrote {(OUT_DIR / 'figure.pdf').relative_to(FIGURE)}")


if __name__ == "__main__":
    main()
