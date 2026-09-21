"""Read-out schematic (SCHEMATIC EXPLAINER) — Supplementary Figure 9, panel a.

How the one shared head names a cell of a type no model was trained on, in two
steps:

  left   the cell lands in the embedding among cells of known types. Of those
         known ("landmark") types it is most similar to one.
  right  from that landmark, the head walks the cell ontology and answers with
         the nearby term carrying the most personalized PageRank mass.

An unseen cell type has no honest location in the embedding, so its answer
comes from its relatives on the ontology instead.

Values are ILLUSTRATIVE — the measured numbers live in the accuracy bars and
the per-case panel. Pure drawing: no data or cache I/O. Laid out about 2.5x
wider than tall; ASPECT below keeps every round mark circular at that ratio.
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

COMP_COLORS = {"Immune / blood": "#0072B2", "Epithelial": "#E69F00", "Endothelial": "#009E73",
               "Stromal / fibroblast": "#CC79A7", "Muscle": "#D55E00", "Neural / glial": "#56B4E9",
               "Germ": "#8C564B", "Other": "#9E9E9E"}
BRIDGE_GREY = "#C2C2C2"; EMBED_BG = "#F5F6F9"; BG_EDGE = "#CFCFCF"
ONTO_BG = "#FBF7FF"; ONTO_EDGE = "#DCCFEA"
GOLD = "#F5A800"; PURPLE = "#7E57C2"; ORANGE = "#E8820C"; INK = "#1A1A1A"

# The two halves, each a rounded box: (x, y, width, height) in axes fraction.
# They sit side by side across the full panel and leave a lane between them for
# the arrow that carries the landmark's name from one to the other.
LEFT_BOX = (0.008, 0.100, 0.552, 0.740)
RIGHT_BOX = (0.585, 0.100, 0.407, 0.740)
HEADING_Y = 0.865          # "embedding" / "cell ontology", above their boxes
CAPTION_Y = 0.165          # the two numbered step captions, inside the boxes

# The panel is ~2.5x wider than tall, so a circle needs a half height 2.5x its
# half width; every round mark below is sized from this.
ASPECT = 2.42

# ---- left half: the embedding -------------------------------------------
# Lineage clusters. ``w`` is the half width; the half height follows from
# ASPECT. ``node_y`` moves the ring off centre for the one cluster that also
# holds the star (the landmark), so the two don't overlap.
#
# Each carries a ringed node (the "known cell type" mark), its cell type below
# the ring, and its lineage above the outline. Six is what the panel's width
# fits at two words each; muscle is left out (too close in colour to
# epithelial).
CLUSTERS = [
    dict(name="neuron",           lineage="neural",      comp="Neural / glial",
         cx=0.058, cy=0.655, w=0.038, n=14, seed=1),
    dict(name="endothelial cell", lineage="endothelial", comp="Endothelial",
         cx=0.228, cy=0.725, w=0.026, n=7,  seed=11),
    dict(name="fibroblast",       lineage="stromal",     comp="Stromal / fibroblast",
         cx=0.090, cy=0.360, w=0.045, n=19, seed=7),
    dict(name="T cell",           lineage="immune",      comp="Immune / blood",
         cx=0.243, cy=0.470, w=0.032, n=10, seed=4),
    dict(name="germ cell",        lineage="germ",        comp="Germ",
         cx=0.310, cy=0.290, w=0.022, n=5,  seed=17),
    dict(name="gland cell",       lineage="epithelial",  comp="Epithelial",
         cx=0.440, cy=0.545, w=0.082, n=30, seed=3, node_y=0.390),
]
NAME_OFFSET = (0, -6)   # cell-type tag, offset from the ring
LINEAGE_GAP = 0.008     # lineage word sits above the outline, not below (tag is below)
NAME_PAD = 0.5
# The landmark the query cell is nearest to, held separately since the arrows
# need its position.
NEAREST = dict(name="gland cell", x=0.440, y=0.390, comp="Epithelial")
QUERY = (0.440, 0.615)

# ---- right half: the ontology neighbourhood ------------------------------
# The same landmark type as an ontology node, with the terms it sends mass to
# fanning right: one above, one level, one below.
# (name, x, y, mass, answer, label_x) — mass is the illustrative PageRank weight.
ONTO_ROOT = dict(name="gland cell", x=0.655, y=0.430)
ONTO_TERMS = [
    dict(name="duct cell",      x=0.772, y=0.700, mass=0.18, answer=False,
         label_x=0.752, label_ha="right"),
    dict(name="secretory cell", x=0.885, y=0.500, mass=0.62, answer=True,
         label_x=0.885, label_ha="center"),
    dict(name="basal cell",     x=0.772, y=0.240, mass=0.11, answer=False,
         label_x=0.752, label_ha="right"),
]
# "the name it answers with" and the answer's own name stack above the diamond,
# which is the one term with clear height above it. The two unchosen terms are
# named on their inner side, so neither name reaches into that stack.
ANSWER_NAME_Y = 0.560
ANSWER_CAPTION_Y = 0.632


def aw(v):
    # Visible minimum preserves weak edges without letting the strongest one
    # dominate the schematic.
    return 1.1 + 3.4 * max(v, 0.0)


def scatter_cluster(ax, cx, cy, w, h, n, seed, avoid):
    rng = np.random.default_rng(seed)
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 40:
        tries += 1
        x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
        if x * x + y * y > 1:
            continue
        px, py = cx + x * w * 0.93, cy + y * h * 0.93
        if any((px - ax_) ** 2 + (py - ay_) ** 2 < 0.0011 for ax_, ay_ in avoid):
            continue
        pts.append((px, py))
    if pts:
        arr = np.array(pts)
        ax.scatter(arr[:, 0], arr[:, 1], s=15, c=BRIDGE_GREY, edgecolors="white",
                   linewidths=0.3, zorder=2, clip_on=False)


def legend_handles():
    return [
        Line2D([0], [0], marker="*", ls="", ms=13, mfc=GOLD, mec=INK,
               label="cell of an unseen type"),
        # One is a cell and the other is the type itself, which is what the
        # read-out compares against, so each label says which.
        Line2D([0], [0], marker="o", ls="", ms=5.5, mfc=BRIDGE_GREY, mec="white",
               label="one cell of a known type"),
        Line2D([0], [0], marker="o", ls="", ms=6.5, mfc=BRIDGE_GREY, mec="#777", mew=1.8,
               label="the known type itself (ring = lineage)"),
        Line2D([0], [0], color=ORANGE, lw=2.4, label="most similar known type"),
        Line2D([0], [0], marker="D", ls="", ms=7, mfc="white", mec=PURPLE, mew=1.8,
               label="the name it answers with"),
        Line2D([0], [0], color=PURPLE, lw=2.2, ls="--",
               label="PageRank mass from that type"),
    ]


def draw_panel_mechanism(ax, *, show_subtitle=True, show_legend=True,
                         show_candidate_label=True, show_query_label=True,
                         show_obs_exp=True, show_landmark_names=True,
                         compact_labels=False,
                         legend_anchor=(0.5, -0.115), legend_ncol=3,
                         legend_fontsize=6.4, scale=1.0,
                         text_scale=None):
    """Draw the mechanism schematic on the given axes (see module docstring).

    ``show_obs_exp`` controls the step numbers and mass values.
    ``scale`` sizes markers, arrows and line widths; ``text_scale`` sizes the
    words and defaults to ``scale``, so the marker geometry and the labels can
    be tuned independently for the schematic's size within the full figure.
    """
    s = scale
    ts = scale if text_scale is None else text_scale
    label_size = 6.4 * ts
    heading_size = label_size / 0.9
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # the two halves
    for (bx, by, bw, bh), edge, face in ((LEFT_BOX, BG_EDGE, EMBED_BG),
                                         (RIGHT_BOX, ONTO_EDGE, ONTO_BG)):
        ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                     boxstyle="round,pad=0.006,rounding_size=0.012",
                     lw=0.8, edgecolor=edge, facecolor=face, zorder=0, clip_on=False))
    ax.text(LEFT_BOX[0] + LEFT_BOX[2] / 2, HEADING_Y, "embedding",
            ha="center", va="bottom",
            fontsize=heading_size, fontweight="bold", color="#5A5A5A", zorder=8)
    ax.text(RIGHT_BOX[0] + RIGHT_BOX[2] / 2, HEADING_Y, "cell ontology",
            ha="center", va="bottom",
            fontsize=heading_size, fontweight="bold", color=PURPLE, zorder=8)

    rings = [dict(name=cl["name"], x=cl["cx"], y=cl.get("node_y", cl["cy"]),
                  comp=cl["comp"])
             for cl in CLUSTERS]
    named_pos = [(r["x"], r["y"]) for r in rings] + [QUERY]

    for cl in CLUSTERS:
        col = COMP_COLORS[cl["comp"]]
        cx, cy, w = cl["cx"], cl["cy"], cl["w"]
        h = w * ASPECT
        ax.add_patch(Ellipse((cx, cy), 2 * w, 2 * h, lw=0.0, facecolor=col, alpha=0.10,
                             zorder=1, clip_on=False))
        ax.add_patch(Ellipse((cx, cy), 2 * w, 2 * h, lw=0.8, edgecolor=col, facecolor="none",
                             linestyle=":", alpha=0.55, zorder=1, clip_on=False))
        scatter_cluster(ax, cx, cy, w, h, cl["n"], cl["seed"], named_pos)
        ax.text(cx, cy + h + LINEAGE_GAP, cl["lineage"], ha="center", va="bottom",
                fontsize=label_size, style="italic", color=col, zorder=7,
                fontweight="bold", clip_on=False)

    # step 1 — the single landmark that is read
    ax.add_patch(FancyArrowPatch(QUERY, (NEAREST["x"], NEAREST["y"]), arrowstyle="-|>",
                 mutation_scale=10 * s, color=ORANGE, lw=aw(0.85) * s,
                 shrinkA=13 * s, shrinkB=10 * s, alpha=0.95, zorder=3, clip_on=False))

    for lm in rings:
        ax.scatter([lm["x"]], [lm["y"]], s=92 * s * s, facecolors=BRIDGE_GREY,
                   edgecolors=COMP_COLORS[lm["comp"]], linewidths=2.2 * s, zorder=5, clip_on=False)
        if show_landmark_names and lm["name"]:
            ax.annotate(lm["name"], (lm["x"], lm["y"]), textcoords="offset points",
                        xytext=(NAME_OFFSET[0] * ts, NAME_OFFSET[1] * ts),
                        ha="center", va="top",
                        fontsize=label_size, fontweight="bold", color=INK, zorder=7,
                        bbox=dict(fc="none", ec="#CCCCCC", lw=0.5, pad=NAME_PAD * ts),
                        clip_on=False)

    ax.scatter([QUERY[0]], [QUERY[1]], marker="*", s=470 * s * s, c=GOLD,
               edgecolors=INK, linewidths=1.1 * s, zorder=8, clip_on=False)
    if show_query_label:
        ax.annotate("cell of an\nunseen type", QUERY, textcoords="offset points",
                    xytext=(0, 9), ha="center", va="bottom",
                    fontsize=label_size, fontweight="bold", color="#8a6200", zorder=8, clip_on=False)

    # the hand-off: the same cell type, now read as an ontology node
    ax.add_patch(FancyArrowPatch((LEFT_BOX[0] + LEFT_BOX[2] - 0.035, NEAREST["y"]),
                                 (ONTO_ROOT["x"] - 0.030, ONTO_ROOT["y"]),
                 arrowstyle="-|>", mutation_scale=10 * s, color="#8A8A8A",
                 lw=1.3 * s, connectionstyle="arc3,rad=-0.22",
                 alpha=0.9, zorder=4, clip_on=False))

    # step 2 — ontology mass out of that node
    for t in ONTO_TERMS:
        ax.add_patch(FancyArrowPatch((ONTO_ROOT["x"], ONTO_ROOT["y"]), (t["x"], t["y"]),
                     arrowstyle="-|>", mutation_scale=8 * s, color=PURPLE,
                     lw=aw(t["mass"]) * s, ls=(0, (4, 2)),
                     shrinkA=10 * s, shrinkB=11 * s, alpha=0.9, zorder=3, clip_on=False))
        if show_obs_exp:
            # Offset perpendicular to the edge, always outward.
            dx, dy = t["x"] - ONTO_ROOT["x"], t["y"] - ONTO_ROOT["y"]
            norm = (dx * dx + dy * dy) ** 0.5 or 1.0
            px, py = -dy / norm, dx / norm
            if px < 0:
                px, py = -px, -py
            ax.annotate(f"{t['mass']:.2f}",
                        (ONTO_ROOT["x"] + 0.55 * dx, ONTO_ROOT["y"] + 0.55 * dy),
                        textcoords="offset points", xytext=(px * 9 * ts, py * 9 * ts),
                        ha="center", va="center", fontsize=label_size, fontweight="bold",
                        color=PURPLE, zorder=7, clip_on=False)

    ax.scatter([ONTO_ROOT["x"]], [ONTO_ROOT["y"]], s=92 * s * s, facecolors=BRIDGE_GREY,
               edgecolors=COMP_COLORS["Epithelial"], linewidths=2.2 * s, zorder=6, clip_on=False)
    ax.annotate(ONTO_ROOT["name"], (ONTO_ROOT["x"], ONTO_ROOT["y"]),
                textcoords="offset points", xytext=(0, -7), ha="center", va="top",
                fontsize=label_size, fontweight="bold", color=INK, zorder=7,
                bbox=dict(fc="white", ec="#DDD", lw=0.5, pad=1.5 * ts), clip_on=False)

    for t in ONTO_TERMS:
        if t["answer"]:
            ax.scatter([t["x"]], [t["y"]], marker="D", s=104 * s * s, facecolors="white",
                       edgecolors=PURPLE, linewidths=2.0 * s, zorder=8, clip_on=False)
            ax.text(t["label_x"], ANSWER_NAME_Y, t["name"], ha="center", va="bottom",
                    fontsize=label_size, fontweight="bold", color=PURPLE, zorder=7)
        else:
            # An ellipse, not a circle: a circle in axes fraction reads as a
            # lozenge on a panel this wide.
            ax.add_patch(Ellipse((t["x"], t["y"]), 2 * 0.0075, 2 * 0.0075 * ASPECT,
                                 facecolor="white", edgecolor="#B9A6CE",
                                 lw=1.2 * s, zorder=6, clip_on=False))
            ax.text(t["label_x"], t["y"], t["name"], ha=t["label_ha"], va="center",
                    fontsize=label_size, color="#7A7A7A", zorder=7)

    if show_candidate_label:
        answer = next(t for t in ONTO_TERMS if t["answer"])
        ax.text(answer["label_x"], ANSWER_CAPTION_Y, "the name it answers with",
                ha="center", va="bottom", fontsize=label_size, color=PURPLE,
                fontweight="bold", zorder=8)

    if show_obs_exp:
        for num, cx, tx, col in (("1", LEFT_BOX[0] + 0.027, LEFT_BOX[0] + 0.057, ORANGE),
                                 ("2", RIGHT_BOX[0] + 0.028, RIGHT_BOX[0] + 0.062, PURPLE)):
            ax.text(cx, CAPTION_Y, num, ha="center", va="center", fontsize=heading_size,
                    fontweight="bold", color="white", zorder=9,
                    bbox=dict(boxstyle="circle,pad=0.28", fc=col, ec="none"))
        ax.text(LEFT_BOX[0] + 0.057, CAPTION_Y, "most similar known type",
                ha="left", va="center", fontsize=label_size, color="#666", zorder=9)
        ax.text(RIGHT_BOX[0] + 0.062, CAPTION_Y, "PageRank mass → prediction",
                ha="left", va="center", fontsize=label_size, linespacing=1.2,
                color="#666", zorder=9)

    if show_legend:
        ax.legend(handles=legend_handles(), loc="lower center",
                  bbox_to_anchor=legend_anchor, ncol=legend_ncol,
                  fontsize=legend_fontsize, handletextpad=0.5, columnspacing=1.5)

    if show_subtitle:
        ax.text(0.5, 1.01,
                "Foundation models return only embeddings, so all five are given one shared head: "
                "read the known cell type each cell is most similar to, then answer with the term "
                "carrying the most ontology mass at that type.",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7 * ts, style="italic", color="#555",
                clip_on=False, wrap=True)
