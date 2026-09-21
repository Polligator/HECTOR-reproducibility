"""Assign every training cell type one broad Cell Ontology term, and a colour.

Imported by `make_figure.py`, not run directly. `palette()` returns the broad terms
with their colours and their share of each species; `write_tables()` writes the
three tables the drawings are built from and returns what they need.

The Cell Ontology is a directed acyclic graph, not a tree: a cell type can descend
from several broad terms at once, so `neuron` is beneath `neural cell` and also
beneath `electrically responsive cell`. Counting a cell type under each of them
inflates the totals several fold. The rule below gives each cell type exactly one
broad term before anything is counted, keeping the alternatives on record in
`result/term_assignments.csv`.

Colour is decided once for both species together, so a term is the same colour in
every panel and in both the interactive and the printed drawing. Which colour a
term gets depends on how large it turned out to be, so it cannot be settled before
the counting. Colours are handed to the drawings in memory and never written to a
file, so there is no copy that can go stale; `result/broad_term_counts.csv` records
what was used.
"""

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIGURE = HERE.parents[1]          # figure1/, holding input/, scripts/ and result/
INPUT = FIGURE / "input" / HERE.name
OUTPUT = FIGURE / "result" / HERE.name

# What each model was trained on: the first so many cell types of the ordered
# table, and at most this many cells of each. The totals are checked every run.
CELL_CAP = 50_000
SPECIES = {
    "human": {"counts": "human_cell_counts.csv", "cell_types": 650, "cells": 14_252_055},
    "mouse": {"counts": "mouse_cell_counts.csv", "cell_types": 340, "cells": 5_952_542},
}

# The broad terms a cell type can be shown under. Every one is a real Cell
# Ontology term used under its own name; none is an invented grouping.
BROAD_TERMS = {
    "CL:0000988": "hematopoietic cell",
    "CL:0002319": "neural cell",
    "CL:0000115": "endothelial cell",
    "CL:0000066": "epithelial cell",
    "CL:0000183": "contractile cell",
    "CL:0002320": "connective tissue cell",
    "CL:0000039": "germ line cell",
    "CL:0000034": "stem cell",
    "CL:0011026": "progenitor cell",
}

# Stem cell and progenitor cell sit above a great many mature cell types. They
# are used only when nothing else is reachable, so a cell type is never shown as
# a stem or progenitor cell merely because that path exists in the ontology.
FALLBACK_TERMS = {"CL:0000034", "CL:0011026"}

REVIEWED_COUNT = 76


# ============================================================== the broad terms

def parse_cell_ontology(obo_path=None):
    """Read the non-obsolete CL names, their is-a parents, and the release line."""
    obo_path = Path(obo_path or INPUT / "cl.obo")
    names, parents, obsolete = {}, {}, set()
    release = "not recorded"
    current = None

    with obo_path.open(encoding="utf-8") as ontology:
        for raw in ontology:
            line = raw.strip()
            if line.startswith("data-version:"):
                release = line.split("data-version:", 1)[1].strip()
            elif line == "[Term]":
                current = None
            elif line.startswith("id:") and current is None:
                current = line.split("id:", 1)[1].strip()
            elif line.startswith("name:") and current:
                names[current] = line.split("name:", 1)[1].strip()
            elif line.startswith("is_a:") and current:
                parents.setdefault(current, []).append(line.split("is_a:", 1)[1].split()[0])
            elif line == "is_obsolete: true" and current:
                obsolete.add(current)
            elif line == "[Typedef]":
                current = None

    names = {term: name for term, name in names.items()
             if term.startswith("CL:") and term not in obsolete}
    parents = {term: [p for p in up if p in names]
               for term, up in parents.items() if term in names}

    for broad_id, expected in BROAD_TERMS.items():
        if names.get(broad_id) != expected:
            raise ValueError(f"{broad_id} is named {names.get(broad_id)!r} in release "
                             f"{release}, not {expected!r}")
    return names, parents, release


def load_training_cell_types(species):
    """The cell types this model trained on, in order, with the cell cap applied.

    Both rules the training code used: keep the first so many rows of the ordered
    table, and expose at most `CELL_CAP` cells of each type.
    """
    settings = SPECIES[species]
    wanted = settings["cell_types"]
    table = pd.read_csv(INPUT / settings["counts"]).head(wanted).copy()

    missing = {"cell_type_ontology_term_id", "cell_type", "count"}.difference(table.columns)
    if missing:
        raise ValueError(f"{settings['counts']} is missing: {', '.join(sorted(missing))}")
    if len(table) != wanted:
        raise ValueError(f"{settings['counts']} holds {len(table)} rows, expected {wanted}")
    if table["cell_type_ontology_term_id"].duplicated().any():
        repeated = table.loc[table["cell_type_ontology_term_id"].duplicated(keep=False),
                             "cell_type_ontology_term_id"].tolist()
        raise ValueError(f"these cell types appear more than once: {repeated}")

    table = table.rename(columns={"count": "cells_available"})
    table["cells_used_in_training"] = table["cells_available"].clip(upper=CELL_CAP)
    table["training_rank"] = range(1, len(table) + 1)
    return table


def ancestors_with_distance(term_id, parents):
    """Every term reachable upwards by is-a, and how many steps away it is."""
    distance = {term_id: 0}
    queue = deque([term_id])
    while queue:
        child = queue.popleft()
        for parent in parents.get(child, []):
            step = distance[child] + 1
            if parent not in distance or step < distance[parent]:
                distance[parent] = step
                queue.append(parent)
    return distance


def reachable_broad_terms(term_id, parents):
    """Which broad terms sit above this cell type, and how far away each one is."""
    distance = ancestors_with_distance(term_id, parents)
    reachable = {broad: distance[broad] for broad in BROAD_TERMS if broad in distance}
    if set(reachable).difference(FALLBACK_TERMS):
        reachable = {broad: step for broad, step in reachable.items()
                     if broad not in FALLBACK_TERMS}
    return reachable


def load_reviewed_choices(names):
    """The hand-made decisions for cell types the ontology cannot settle alone."""
    path = INPUT / "reviewed_assignments.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} holds the choice and the reason for every cell type that reaches "
            "more than one broad term, or none. Without it those cannot be placed.")

    reviewed = pd.read_csv(path)
    required = {"original_cl_id", "original_cl_name", "high_level_cl_id",
                "high_level_cl_name", "selection_rationale"}
    if not required.issubset(reviewed.columns):
        raise ValueError(f"{path.name} is missing: {', '.join(sorted(required - set(reviewed.columns)))}")
    if len(reviewed) != REVIEWED_COUNT:
        raise ValueError(f"{path.name} holds {len(reviewed)} rows, expected {REVIEWED_COUNT}")
    if reviewed["original_cl_id"].duplicated().any():
        raise ValueError(f"{path.name} reviews the same cell type twice")
    if not reviewed["selection_rationale"].str.strip().ne("").all():
        raise ValueError(f"{path.name} has a decision with no reason given")
    if not reviewed["original_cl_name"].equals(reviewed["original_cl_id"].map(names)):
        raise ValueError(f"{path.name} names a cell type differently from the ontology")
    if not reviewed["high_level_cl_name"].equals(reviewed["high_level_cl_id"].map(names)):
        raise ValueError(f"{path.name} names a broad term differently from the ontology")

    indexed = reviewed.set_index("original_cl_id")
    return indexed["high_level_cl_id"].to_dict(), indexed["selection_rationale"].to_dict()


def assign(species, names=None, parents=None):
    """One row per training cell type, each under the one broad term it belongs to.

    A cell type with exactly one reachable broad term is assigned automatically.
    Anything else must appear in the reviewed table, and every reviewed choice is
    checked to be a genuine is-a ancestor before it is accepted.
    """
    if names is None or parents is None:
        names, parents, _ = parse_cell_ontology()
    reviewed_choice, reviewed_reason = load_reviewed_choices(names)

    unknown = set(reviewed_choice.values()).difference(names)
    if unknown:
        raise ValueError(f"reviewed choices use unknown CL IDs: {', '.join(sorted(unknown))}")

    rows = []
    for cell_type in load_training_cell_types(species).itertuples(index=False):
        term_id = cell_type.cell_type_ontology_term_id
        if term_id not in names:
            raise ValueError(f"{term_id} ({cell_type.cell_type}) is not in the ontology")

        reachable = reachable_broad_terms(term_id, parents)
        alternatives = "; ".join(
            f"{names.get(broad, broad)} ({broad}, {step} step{'' if step == 1 else 's'})"
            for broad, step in sorted(reachable.items(), key=lambda item: item[1]))

        if term_id in reviewed_choice:
            broad_id = reviewed_choice[term_id]
            if broad_id not in ancestors_with_distance(term_id, parents):
                raise ValueError(f"reviewed choice {broad_id} ({names[broad_id]}) is not "
                                 f"above {term_id} ({names[term_id]}) in the ontology")
            rule, reason = "reviewed", reviewed_reason[term_id]
        elif len(reachable) == 1:
            broad_id = next(iter(reachable))
            rule, reason = "automatic", "Only one high-level CL term is reachable"
        else:
            trouble = ("reaches no broad term" if not reachable else
                       "reaches " + ", ".join(sorted(BROAD_TERMS[b] for b in reachable)))
            raise ValueError(f"{term_id} ({names[term_id]}) {trouble}, so it needs a line in "
                             f"{(INPUT / 'reviewed_assignments.csv').name}")

        rows.append({
            "species": species,
            "training_rank": cell_type.training_rank,
            "cell_type_ontology_term_id": term_id,
            "cell_type": cell_type.cell_type,
            "cells_available": int(cell_type.cells_available),
            "cells_used_in_training": int(cell_type.cells_used_in_training),
            "high_level_cl_id": broad_id,
            "high_level_cl_name": names[broad_id],
            "assignment_rule": rule,
            "selection_rationale": reason,
            "all_reachable_high_level_terms": alternatives,
        })

    assignments = pd.DataFrame(rows)
    check(assignments, species)
    return assignments


def check(assignments, species):
    """Refuse to go on if a cell type is counted twice or a total is wrong.

    This is what keeps a cell type descending from several broad terms from being
    counted under more than one of them.
    """
    settings = SPECIES[species]
    if len(assignments) != settings["cell_types"]:
        raise ValueError(f"built {len(assignments)} rows, expected {settings['cell_types']}")
    if assignments["cell_type_ontology_term_id"].duplicated().any():
        raise ValueError("a cell type appears more than once")
    if assignments["cell_type"].duplicated().any():
        repeated = sorted(assignments.loc[assignments["cell_type"].duplicated(), "cell_type"])
        raise ValueError(f"these cell type names appear more than once: {repeated}")
    if not assignments["high_level_cl_id"].ne("").all():
        raise ValueError("a cell type was left without a broad term")

    total = int(assignments["cells_used_in_training"].sum())
    if total != settings["cells"]:
        raise ValueError(f"cells total {total:,}, expected {settings['cells']:,}")


def broad_term_shares(names=None, parents=None):
    """The broad terms of both species together, largest first.

    A term is ranked by the larger of its two shares, so a term that is small in
    one species but not the other is not treated as small.
    """
    if names is None or parents is None:
        names, parents, _ = parse_cell_ontology()
    shares = {}
    for species in SPECIES:
        assignments = assign(species, names, parents)
        total = assignments["cells_used_in_training"].sum()
        for (broad_id, broad_name), rows in assignments.groupby(
                ["high_level_cl_id", "high_level_cl_name"]):
            entry = shares.setdefault((broad_id, broad_name), {s: 0.0 for s in SPECIES})
            entry[species] = round(100 * rows["cells_used_in_training"].sum() / total, 3)
    ordered = sorted(shares.items(), key=lambda item: -max(item[1].values()))
    return [(broad_id, name, split) for (broad_id, name), split in ordered]


# ====================================================================== colour

# Two colours less than this far apart in a perceptually even colour space read as
# the same colour at a glance. It is a rule of thumb, not a sharp edge, so pairs a
# little above it are reported too.
TOO_CLOSE = 12.0
VIEWS = ["as it is", "red-blind (protanopia)", "green-blind (deuteranopia)"]

# Base palette. The terms above MAJOR_PERCENT take colours from here; the rest are
# built between them.
BASE_PALETTE = [
    "#2E86AB",  # deep azure blue
    "#A23B72",  # rich magenta
    "#F18F01",  # vibrant amber
    "#C73E1D",  # deep coral red
    "#6A994E",  # sage green
    "#BC4B51",  # muted rose
    "#8E7DBE",  # soft lavender
    "#F4A261",  # warm peach
    "#2A9D8F",  # teal
    "#E76F51",  # terracotta
    "#457B9D",  # steel blue
    "#B5838D",  # dusty mauve
]

# A term takes a colour straight from BASE_PALETTE if it holds at least this percent
# of either species' cells. Rarer terms get colours built around those.
MAJOR_PERCENT = 1.0


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])
_FROM_XYZ = np.linalg.inv(_TO_XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def hex_to_lab(hex_color):
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    xyz = _TO_XYZ @ _srgb_to_linear(rgb) / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def lab_to_rgb(lab):
    fy = (lab[0] + 16) / 116
    f = np.array([fy + lab[1] / 500, fy, fy - lab[2] / 200])
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
    return _linear_to_srgb(_FROM_XYZ @ xyz)


def lab_to_hex(lab):
    rgb = np.clip(lab_to_rgb(lab), 0, 1)
    return "#" + "".join(f"{round(v * 255):02X}" for v in rgb)


def lab_to_lch(lab):
    return np.array([lab[0], np.hypot(lab[1], lab[2]), np.degrees(np.arctan2(lab[2], lab[1])) % 360])


def lch_to_lab(lch):
    a = np.radians(lch[2])
    return np.array([lch[0], lch[1] * np.cos(a), lch[1] * np.sin(a)])


def fit_into_gamut(lch):
    """Take as much colourfulness as the screen can actually show at this hue.

    Clipping out-of-gamut colours instead pushes them towards neon; reducing
    chroma until they fit keeps them in the same family as the base palette.
    """
    low, high = 0.0, lch[1]
    for _ in range(20):
        mid = (low + high) / 2
        rgb = lab_to_rgb(lch_to_lab(np.array([lch[0], mid, lch[2]])))
        if np.all(rgb >= -1e-4) and np.all(rgb <= 1 + 1e-4):
            low = mid
        else:
            high = mid
    return np.array([lch[0], low, lch[2]])


_TO_LMS = np.array([[17.8824, 43.5161, 4.11935],
                    [3.45565, 27.1554, 3.86714],
                    [0.0299566, 0.184309, 1.46709]])
_NO_LONG_CONE = np.array([[0.0, 2.02344, -2.52581], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_NO_MIDDLE_CONE = np.array([[1.0, 0.0, 0.0], [0.494207, 0.0, 1.24827], [0.0, 0.0, 1.0]])


def simulate_colour_blindness(hex_color, kind):
    """What the colour looks like to someone missing one kind of cone.

    Kind 0 is ordinary sight, 1 is red-blindness and 2 is green-blindness. The
    method is Vienot, Brettel and Mollon's: separate the three kinds of cone,
    replace the missing one by what the other two imply, and turn the result
    back into a colour. Between them the two kinds affect about one man in
    twelve, so a figure that only works in ordinary sight does not work.
    """
    if kind == 0:
        return hex_color
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    lms = _TO_LMS @ _srgb_to_linear(rgb)
    drop = _NO_LONG_CONE if kind == 1 else _NO_MIDDLE_CONE
    seen = np.clip(_linear_to_srgb(np.linalg.inv(_TO_LMS) @ (drop @ lms)), 0, 1)
    return "#" + "".join(f"{round(v * 255):02X}" for v in seen)


def order_by_separation(palette_colours):
    """Reorder a palette so the first few colours are the ones furthest apart.

    The largest terms are then given the colours that are hardest to confuse.
    """
    labs = [hex_to_lab(c) for c in palette_colours]
    centre = np.mean(labs, axis=0)
    picked = [int(np.argmax([np.linalg.norm(lab - centre) for lab in labs]))]
    while len(picked) < len(palette_colours):
        best, best_gap = None, -1.0
        for i in range(len(palette_colours)):
            if i in picked:
                continue
            gap = min(np.linalg.norm(labs[i] - labs[j]) for j in picked)
            if gap > best_gap:
                best, best_gap = i, gap
        picked.append(best)
    return [palette_colours[i] for i in picked]


def build_colours(terms):
    """One colour per broad term, keyed by CL id.

    Terms holding at least MAJOR_PERCENT of a species take the base palette colours
    that stand furthest apart. The rest are built in the wide gaps those leave on the
    colour wheel, taking lightness and chroma from the two base colours their hue
    sits between so they stay in the same family.

    Two choices are deliberate. The rarer terms are held back in chroma, so the eye
    goes first to the terms holding most of the cells rather than to one worth a
    hundredth of a percent. And within each gap they step up and down in lightness,
    which keeps neighbours apart where their hues have to sit close together.

    Choosing the base colours by colour-blind separation rather than ordinary
    separation is not done: it fills the largest terms with several shades of red,
    since warm colours are what stay apart when a cone type is missing.

    Raises if two terms end up with the same colour.
    """
    major = [t for t in terms if max(t[2].values()) >= MAJOR_PERCENT]
    minor = [t for t in terms if max(t[2].values()) < MAJOR_PERCENT]

    anchors = order_by_separation(BASE_PALETTE)[:len(major)]
    out = {t[0]: c for t, c in zip(major, anchors)}

    # At each hue, interpolate lightness/chroma between the two bounding
    # anchors, damping chroma so the small terms recede visually.
    anchor_lch = sorted((lab_to_lch(hex_to_lab(c)) for c in anchors), key=lambda l: l[2])

    def between(hue):
        for i, lch in enumerate(anchor_lch):
            nxt = anchor_lch[(i + 1) % len(anchor_lch)]
            width = (nxt[2] - lch[2]) % 360
            along = (hue - lch[2]) % 360
            if along <= width:
                return lch, nxt, along / width if width else 0.0
        return anchor_lch[0], anchor_lch[0], 0.0

    candidates = []
    for hue in np.arange(0, 360, 2.0):
        start, end, t = between(hue)
        for light_shift in (-16, -8, 0, 8, 16):
            for damping in (0.55, 0.68, 0.80):
                light = start[0] + t * (end[0] - start[0]) + light_shift
                chroma = (start[1] + t * (end[1] - start[1])) * damping
                candidates.append(fit_into_gamut(np.array([light, chroma, hue])))

    # Take them one at a time, each the furthest from every colour already in use.
    # This separates them better than spacing them evenly around the wheel.
    used = [hex_to_lab(c) for c in anchors]
    built = []
    for _ in minor:
        labs = [lch_to_lab(lch) for lch in candidates]
        pick = int(np.argmax([min(np.linalg.norm(lab - u) for u in used) for lab in labs]))
        used.append(labs[pick])
        built.append(candidates.pop(pick))

    # Give the largest of the small terms the colours that stand apart best.
    anchor_labs = [hex_to_lab(c) for c in anchors]
    built.sort(key=lambda lch: -min(np.linalg.norm(lch_to_lab(lch) - a) for a in anchor_labs))
    for term, lch in zip(minor, built):
        out[term[0]] = lab_to_hex(lch_to_lab(lch))

    if len(set(out.values())) != len(out):
        raise ValueError("two terms were given the same colour")
    return out


def palette(names=None, parents=None):
    """The broad terms, largest first, each with its colour and its share of each species.

    This is what the two drawings ask for. It is worked out on the spot rather
    than read from a file, so there is no way for a drawing to be coloured from a
    stale table.

    Each entry is the CL id, the term name, the colour, and a share per species.
    """
    shares = broad_term_shares(names, parents)
    colours = build_colours(shares)
    return [(broad_id, name, colours[broad_id], split) for broad_id, name, split in shares]


def close_pairs(named, kind):
    """Which colours a reader of this kind of sight would have trouble telling apart."""
    labs = {name: hex_to_lab(simulate_colour_blindness(colour, kind))
            for name, colour in named}
    found = []
    for i, (a, _) in enumerate(named):
        for b, _ in named[i + 1:]:
            gap = float(np.linalg.norm(labs[a] - labs[b]))
            if gap < TOO_CLOSE * 1.4:
                found.append((gap, a, b))
    return sorted(found)


def draw_swatch_sheet(named, shares, out_path):
    """The colours on their own, and as a colour-blind reader sees them.

    Drawn only when asked for, with `--swatches`. It is for looking at while
    deciding whether the colours work, not something the paper needs.
    """
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    height = 0.34 * len(named) + 1.2
    figure, axes = plt.subplots(1, 3, figsize=(13, height))

    for ax, kind in zip(axes, range(3)):
        ax.set_title(VIEWS[kind], fontsize=10, pad=10)
        for row, (name, colour) in enumerate(named):
            y = len(named) - row - 1
            ax.add_patch(Rectangle((0, y), 1.6, 0.8,
                                   facecolor=simulate_colour_blindness(colour, kind),
                                   edgecolor="#cccccc", linewidth=0.5, clip_on=False))
            ax.text(1.8, y + 0.4, f"{name}  {shares[name]}", fontsize=8,
                    va="center", ha="left")
        ax.set_xlim(0, 9)
        ax.set_ylim(-0.4, len(named))
        ax.axis("off")

    figure.tight_layout()
    figure.savefig(out_path, dpi=200)
    plt.close(figure)


# ====================================================================== tables

PUBLISHED_COLUMNS = [
    "species", "training_rank", "cell_type_ontology_term_id", "cell_type",
    "cells_available", "cells_used_in_training", "high_level_cl_id",
    "high_level_cl_name", "assignment_rule", "selection_rationale",
    "all_reachable_high_level_terms",
]


def write_tables():
    """Write the three tables the drawings are built from, and hand back what they need.

    Returns every cell type with the broad term it was given, and the broad terms
    with their colours and their share of each species. Both are worked out here
    anyway, and passing them on saves the drawings reading the ontology again.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names, parents, release = parse_cell_ontology()
    print("Cell Ontology release:", release)

    assignments = pd.concat(
        [assign(species, names, parents) for species in SPECIES], ignore_index=True)
    coloured = palette(names, parents)
    colour_of = {broad_id: colour for broad_id, _, colour, _ in coloured}

    # One row per broad term per species, which is what the drawing is made of.
    # The colour is carried alongside so the table records what was drawn; it is
    # not read back, and the drawings work it out for themselves.
    counts = assignments.groupby(
        ["species", "high_level_cl_id", "high_level_cl_name"], as_index=False).agg(
        training_term_count=("cell_type_ontology_term_id", "size"),
        cells_used_in_training=("cells_used_in_training", "sum"))
    totals = counts.groupby("species")["cells_used_in_training"].transform("sum")
    counts["percent_of_species_cells"] = (
        100 * counts["cells_used_in_training"] / totals).round(3)
    counts["color"] = counts["high_level_cl_id"].map(colour_of)
    counts = counts.sort_values(["species", "cells_used_in_training"],
                                ascending=[True, False]).reset_index(drop=True)

    # Adding the broad terms back up has to give the same total as counting the
    # cell types one by one. If it does not, something is being counted twice.
    for species in SPECIES:
        gathered = counts.loc[counts["species"] == species, "cells_used_in_training"].sum()
        one_by_one = assignments.loc[assignments["species"] == species,
                                     "cells_used_in_training"].sum()
        if gathered != one_by_one:
            raise ValueError(f"{species}: broad terms total {gathered:,} but the cell types "
                             f"total {one_by_one:,}")
    if counts["color"].isna().any():
        raise ValueError("a broad term was left without a colour")

    summary = pd.DataFrame([{
        "species": species,
        "ontology_release": release,
        "cell_cap_per_term": CELL_CAP,
        "selected_terms": len(rows),
        "cells_used_in_training": int(rows["cells_used_in_training"].sum()),
        "automatically_assigned_terms": int((rows["assignment_rule"] == "automatic").sum()),
        "reviewed_terms": int((rows["assignment_rule"] == "reviewed").sum()),
        "terms_needing_review": 0,
        "high_level_terms_used": int(rows["high_level_cl_id"].nunique()),
    } for species, rows in assignments.groupby("species", sort=False)])

    assignments[PUBLISHED_COLUMNS].to_csv(OUTPUT / "term_assignments.csv", index=False)
    counts.to_csv(OUTPUT / "broad_term_counts.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)

    print()
    print(summary.to_string(index=False))
    print(f"\nCells capped at {CELL_CAP:,}: "
          f"{int((assignments['cells_available'] > CELL_CAP).sum())} of {len(assignments)} cell types")
    print(f"Broad terms used across both species: {counts['high_level_cl_id'].nunique()}")
    print(f"Wrote three tables to {OUTPUT}")

    named = [(name, colour) for _, name, colour, _ in coloured]
    print(f"\n{len(named)} broad terms, each in a colour of its own")
    for kind, view in enumerate(VIEWS):
        found = close_pairs(named, kind)
        hard = [pair for pair in found if pair[0] < TOO_CLOSE]
        print(f"  {view}: {len(hard)} of {len(named) * (len(named) - 1) // 2} pairs too close"
              + (f"; closest {found[0][0]:.1f}, {found[0][1]} and {found[0][2]}" if found else ""))

    return assignments, coloured


def write_swatch_sheet(coloured):
    """Draw the colours out to look at. Asked for with --swatches; never automatic."""
    sheet = OUTPUT / "broad_term_colors.png"
    draw_swatch_sheet([(name, colour) for _, name, colour, _ in coloured],
                      {name: f"{max(split.values()):.2f}%" for _, name, _, split in coloured},
                      sheet)
    print(f"Wrote {sheet.name}")
