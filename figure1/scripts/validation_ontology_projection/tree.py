"""Where each cell type sits in the drawing, and what score it is drawn in.

The cell types, and which broad term each belongs to, come from the training
composition figure next door: this module imports its rule rather than repeating
it, so the two figures cannot disagree about where a cell type belongs.

Imported by make_figure.py, not run directly.
"""

from collections import Counter
from importlib import util as import_util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FIGURE = HERE.parents[1]          # figure1/, holding input/, scripts/ and result/
INPUT = FIGURE / "input" / HERE.name
OUTPUT = FIGURE / "result" / HERE.name

# The training composition figure: its ontology.py holds the one copy of the
# rule that gives each cell type its broad term, and its input/cl.obo is the
# ontology release both figures are built from. Loaded by path rather than by
# adding that folder to the import path, since both folders hold a
# make_figure.py. Two folder names are checked, since this figure's mirrored
# copy uses a different one.
TRAINING_FIGURE_NAMES = ["training_ontology_projection", "ontology_data_visual"]


def _load_training_figure_rule():
    beside = [HERE.parent / name for name in TRAINING_FIGURE_NAMES]
    where = next((folder / "ontology.py" for folder in beside
                  if (folder / "ontology.py").exists()), None)
    if where is None:
        raise FileNotFoundError(
            "the training composition figure holds the rule that gives each cell type its "
            "broad term, and this figure nests the ontology the way that one does, so it "
            "cannot be built without it. Looked beside this folder for: "
            + ", ".join(folder.name for folder in beside))
    spec = import_util.spec_from_file_location("training_figure_ontology", where)
    module = import_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ontology = _load_training_figure_rule()

# Per-class scores, one file per species. Cell_Type holds the ontology
# identifier; the score columns are read by name.
SCORE_FILES = {"human": "human_class_scores.csv", "mouse": "mouse_class_scores.csv"}

# Which score fills each cell type's tile.
TILE_SCORE = "F1"

# Recall is omitted (identical to accuracy for every class here); ROC AUC is
# omitted (never falls below 0.957, so it draws as a flat line at the top).
DISTRIBUTION_SCORES = ["Accuracy", "Precision", "F1"]

# What the outermost box is called. It stands for nothing and is never named on
# the page; it exists only so the drawing package does not invent one.
EVERYTHING = "the whole drawing"

# A tie between two equidistant ancestors (the ontology doesn't say which box
# a cell type belongs in) goes to whichever holds more training cell types, to
# keep the boxes few and large. Every alternative is written to the output
# table; add a row here to overrule a specific tie.
PARENT_OVERRIDES = "reviewed_parents.csv"


def load_scores(species):
    """The per-class scores, checked against the cell types the model trained on."""
    path = INPUT / SCORE_FILES[species]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} holds one row per training cell type, keyed by ontology "
            "identifier in a Cell_Type column, with the score columns beside it.")

    scores = pd.read_csv(path)
    wanted = {"Cell_Type", TILE_SCORE, *DISTRIBUTION_SCORES}
    missing = wanted.difference(scores.columns)
    if missing:
        raise ValueError(f"{path.name} is missing: {', '.join(sorted(missing))}")
    if scores["Cell_Type"].duplicated().any():
        repeated = scores.loc[scores["Cell_Type"].duplicated(keep=False), "Cell_Type"].tolist()
        raise ValueError(f"{path.name} scores the same cell type twice: {repeated}")

    return scores.rename(columns={"Cell_Type": "cell_type_ontology_term_id"})


def load_parent_overrides(names):
    """Hand-made choices for cell types the ontology leaves in two places at once."""
    path = INPUT / PARENT_OVERRIDES
    if not path.exists():
        return {}

    reviewed = pd.read_csv(path)
    required = {"cell_type_ontology_term_id", "parent_cl_id", "reason"}
    if not required.issubset(reviewed.columns):
        raise ValueError(f"{path.name} is missing: {', '.join(sorted(required - set(reviewed.columns)))}")
    if reviewed["cell_type_ontology_term_id"].duplicated().any():
        raise ValueError(f"{path.name} overrules the same cell type twice")
    if not reviewed["reason"].astype(str).str.strip().ne("").all():
        raise ValueError(f"{path.name} has a choice with no reason given")

    unknown = set(reviewed["cell_type_ontology_term_id"]) | set(reviewed["parent_cl_id"])
    unknown -= set(names)
    if unknown:
        raise ValueError(f"{path.name} names terms that are not in the ontology: {sorted(unknown)}")

    return dict(zip(reviewed["cell_type_ontology_term_id"], reviewed["parent_cl_id"]))


def place(species):
    """One row per training cell type, each under the cell type that contains it.

    A cell type goes under the nearest is-a ancestor that the model also trained
    on and that shares its broad term; failing that, straight under the broad
    term itself. Restricting the search to the same broad term is what keeps this
    drawing and the training composition figure telling the same story.
    """
    names, parents, release = ontology.parse_cell_ontology()
    trained = ontology.assign(species, names, parents)
    scores = load_scores(species)
    overrides = load_parent_overrides(names)

    trained_ids = set(trained["cell_type_ontology_term_id"])
    unscored = trained_ids.difference(scores["cell_type_ontology_term_id"])
    unwanted = set(scores["cell_type_ontology_term_id"]).difference(trained_ids)
    if unscored or unwanted:
        raise ValueError(
            f"the {species} scores do not cover the same cell types the model trained on: "
            f"{len(unscored)} trained but unscored, {len(unwanted)} scored but not trained")

    broad_term = dict(zip(trained["cell_type_ontology_term_id"], trained["high_level_cl_id"]))
    above = {term: ontology.ancestors_with_distance(term, parents) for term in trained_ids}
    # How many training cell types sit beneath each ontology term. Counted from
    # the ontology alone, so it does not depend on the drawing being built.
    beneath = Counter(reachable for term in trained_ids
                      for reachable in above[term] if reachable != term)

    rows = []
    for term in trained["cell_type_ontology_term_id"]:
        candidates = [(step, other) for other, step in above[term].items()
                      if other != term
                      and other in trained_ids
                      and broad_term[other] == broad_term[term]]

        chosen, passed_over, rule = None, [], ""
        if candidates:
            nearest = min(step for step, _ in candidates)
            equally_near = sorted(other for step, other in candidates if step == nearest)
            if len(equally_near) == 1:
                chosen, rule = equally_near[0], "one nearest ancestor"
            else:
                # One of them may sit beneath all the others, and then it is the
                # one that actually contains this cell type.
                most_specific = [one for one in equally_near
                                 if all(other == one or other in above[one] for other in equally_near)]
                if len(most_specific) == 1:
                    chosen, rule = most_specific[0], "one nearest ancestor"
                else:
                    ranked = sorted(equally_near, key=lambda one: (-beneath[one], one))
                    chosen, rule = ranked[0], "largest of the equally near"
                    passed_over = ranked[1:]

        if term in overrides:
            wanted = overrides[term]
            if wanted not in above[term] or wanted == term:
                raise ValueError(f"{PARENT_OVERRIDES} puts {names[term]!r} under "
                                 f"{names.get(wanted, wanted)!r}, which is not above it")
            if wanted in trained_ids and broad_term[wanted] != broad_term[term]:
                raise ValueError(f"{PARENT_OVERRIDES} puts {names[term]!r} under "
                                 f"{names[wanted]!r}, which the training composition "
                                 "figure shows under a different broad term")
            passed_over = [one for one in ([chosen] if chosen else []) + passed_over if one != wanted]
            chosen, rule = wanted, "reviewed by hand"

        if chosen is None:
            rule = ("a broad term, drawn at the top" if broad_term[term] == term
                    else "no nearer ancestor was trained on")
            chosen = None if broad_term[term] == term else broad_term[term]

        rows.append({
            "cell_type_ontology_term_id": term,
            "cell_type": names[term],
            "broad_cl_id": broad_term[term],
            "broad_cl_name": names[broad_term[term]],
            "drawn_inside_cl_id": chosen or "",
            "drawn_inside": names[chosen] if chosen else "",
            "placement_rule": rule,
            "equally_near_alternatives": "; ".join(
                f"{names[one]} ({one}, holds {beneath[one]})" for one in passed_over),
        })

    placed = pd.DataFrame(rows).merge(
        trained[["cell_type_ontology_term_id", "cell_type", "training_rank",
                 "cells_used_in_training"]],
        on=["cell_type_ontology_term_id", "cell_type"], validate="one_to_one")
    placed = placed.merge(scores, on="cell_type_ontology_term_id", validate="one_to_one")

    check(placed, species, names)
    return placed, names, release


def check(placed, species, names):
    """Refuse to go on if the drawing would not be a tree, or would lose a cell type."""
    expected = ontology.SPECIES[species]["cell_types"]
    if len(placed) != expected:
        raise ValueError(f"{len(placed)} {species} cell types placed, expected {expected}")

    inside = dict(zip(placed["cell_type_ontology_term_id"], placed["drawn_inside_cl_id"]))
    for term in inside:
        seen, walk = set(), term
        while walk:
            if walk in seen:
                raise ValueError(f"{names[term]!r} ends up inside itself")
            seen.add(walk)
            walk = inside.get(walk, "")

    broad = dict(zip(placed["cell_type_ontology_term_id"], placed["broad_cl_id"]))
    for term, container in inside.items():
        if container and container in broad and broad[container] != broad[term]:
            raise ValueError(f"{names[term]!r} is drawn inside {names[container]!r}, "
                             "which the training composition figure puts elsewhere")


def as_tiles(placed, names):
    """The boxes to draw: one per cell type, plus the broad terms that hold them.

    Every cell type gets a tile of the same size, so a cell type the model does
    badly on is as visible as any other however rare it is. A cell type that also
    contains other cell types is drawn as a box around them with a tile of its
    own inside, which is what keeps the sizes equal.
    """
    trained = set(placed["cell_type_ontology_term_id"])
    holds_others = set(placed.loc[placed["drawn_inside_cl_id"] != "", "drawn_inside_cl_id"])
    tiles = []

    for _, row in placed.iterrows():
        term = row["cell_type_ontology_term_id"]
        # Empty only for a broad term the model trained on, which is a box at the
        # top of the drawing and a cell type in its own right at the same time.
        container = row["drawn_inside_cl_id"]
        if container and container not in trained:
            container = f"broad:{container}"

        if term in holds_others:
            tiles.append(dict(node=term, inside=container, name=row["cell_type"],
                              score=None, own_size=0, kind="holds others"))
            tiles.append(dict(node=f"{term}:itself", inside=term, name=row["cell_type"],
                              score=row[TILE_SCORE], own_size=1, kind="cell type"))
        else:
            tiles.append(dict(node=term, inside=container, name=row["cell_type"],
                              score=row[TILE_SCORE], own_size=1, kind="cell type"))

    for broad_id in sorted(set(placed["broad_cl_id"]).difference(trained)):
        tiles.append(dict(node=f"broad:{broad_id}", inside="", name=names[broad_id],
                          score=None, own_size=0, kind="broad term"))

    # One unnamed box holds the broad terms; without it, Plotly invents its own
    # outermost box filled dark grey with a title strip, and no setting recolours it.
    tiles = [dict(tile, inside=tile["inside"] or EVERYTHING) for tile in tiles]
    tiles.append(dict(node=EVERYTHING, inside="", name="", score=None,
                      own_size=0, kind="the whole drawing"))

    tiles = pd.DataFrame(tiles)

    # Plotly is given each box's whole size, so a box has to be as large as
    # everything inside it.
    size = dict(zip(tiles["node"], tiles["own_size"]))
    children = {}
    for _, tile in tiles.iterrows():
        children.setdefault(tile["inside"], []).append(tile["node"])

    def total(node):
        return size[node] + sum(total(child) for child in children.get(node, []))

    tiles["size"] = [total(node) for node in tiles["node"]]
    drawn = tiles.loc[tiles["inside"] == "", "size"].sum()
    if drawn != len(placed):
        raise ValueError(f"the boxes add up to {drawn} cell types, not {len(placed)}")
    return tiles
