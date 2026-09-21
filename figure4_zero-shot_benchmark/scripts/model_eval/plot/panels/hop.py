"""Figure 4 — panel d: ontology hop-tolerant accuracy (explainer + curve).

``draw_curve`` plots the cumulative fraction of predictions within <=k hops for
each model; ``draw_explainer`` is the Cell-Ontology subtree schematic overlaid as
an inset. ``hop_data`` (dict label -> pd.Series of hop distances) is supplied by
the composer from ``result/plot_cache/hop_data.pkl``.
"""
from __future__ import annotations

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.lines import Line2D

from .. import style

# Locked palette + order (top -> bottom in legend; HECTOR first).
SERIES = [
    ("HECTOR (own head)",       style.HECTOR_NATIVE,              "-"),
    ("HECTOR (own head, open)", style.HECTOR_NATIVE_OPEN,         "--"),
    ("HECTOR (embedding)",      style.HECTOR_EMBEDDING,           "-"),
    ("scimilarity",             style.MODEL_COLOR["scimilarity"], "-"),
    ("scGPT",                   style.MODEL_COLOR["scGPT"],       "-"),
    ("scCello",                 style.MODEL_COLOR["scCello"],     "-"),
    ("Geneformer",              style.MODEL_COLOR["geneformer"],  "-"),
    ("Random Baseline",         "#777777",                        ":"),
]

# Legend display names only; the raw label stays the hop_data lookup key.
# Names match panel c: HECTOR's two head arms are told apart by how many terms
# they choose among; the shared-read-out embeddings carry no qualifier, since
# all five models (HECTOR included) go through it equally.
_LEGEND_DISPLAY = {
    "HECTOR (own head)": "native, closed",
    "HECTOR (own head, open)": "native, open",
    "HECTOR (embedding)": "embedding",
    "scimilarity": "SCimilarity",
    "Random Baseline": "random guessing",
}

# HECTOR's three arms share the first row under one heading, as in panel c;
# the four comparison models take the second row, needing no heading.
LEGEND_ROWS = [
    ("HECTOR", ["HECTOR (own head)", "HECTOR (own head, open)",
                "HECTOR (embedding)"]),
    (None, ["scimilarity", "scGPT", "scCello", "Geneformer",
            "Random Baseline"]),
]

MAX_HOP = 5
NEARBY_BAND = (0, 2)

# Marker areas in the explainer inset (points squared) and line width between
# them, cut to about a third of their original diameter so the subtree's shape
# reads rather than a row of discs. Halving the area alone (a diameter cut
# under a third) read as no change — the eye doesn't register it.
NODE_AREA_TRUTH = 55       # the query cell, a star
NODE_AREA_NEAR = 18        # the nodes one and two hops from it
NODE_AREA_FAR = 11         # everything further out
EDGE_WIDTH = 0.7           # the lines between them


def cumulative_within_k(series: pd.Series, ks: list[int]) -> list[float]:
    series = pd.Series(series).reset_index(drop=True)
    n = len(series)
    return [float(((series >= 0) & (series <= k)).sum()) / n if n else 0.0 for k in ks]


def legend_handles(hop_data: dict[str, pd.Series]):
    """The eight series, in locked order, for a legend drawn outside the axes."""
    return [
        Line2D([0], [0], color=color, lw=1.4, linestyle=ls,
               label=_LEGEND_DISPLAY.get(label, label))
        for label, color, ls in SERIES if label in hop_data
    ]


def _draw_grouped_legend(ax, hop_data, *, fontsize=style.MAIN_TEXT):
    """One legend per row of ``LEGEND_ROWS``, stacked above the axes.

    Two legends rather than one with ``ncol``, because matplotlib fills columns
    top to bottom and there is no way to say which entry belongs on which row.
    A row's heading rides in as an entry with an invisible handle, so it sits on
    the same baseline as the names it heads instead of on a line of its own.
    """
    style_of = {label: (color, ls) for label, color, ls in SERIES}
    legends = []
    for heading, labels in LEGEND_ROWS:
        handles = []
        if heading is not None:
            handles.append(Line2D([], [], color="none", lw=0.0, label=heading))
        for label in labels:
            if label not in hop_data:
                continue
            color, ls = style_of[label]
            handles.append(Line2D([0], [0], color=color, lw=1.4, linestyle=ls,
                                  label=_LEGEND_DISPLAY.get(label, label)))
        if len(handles) <= (1 if heading is not None else 0):
            continue
        legends.append((heading, handles))

    row_height = 0.075
    for i, (heading, handles) in enumerate(legends):
        y = 1.02 + row_height * (len(legends) - 1 - i)
        leg = ax.legend(handles=handles, loc="lower center", ncol=len(handles),
                        fontsize=fontsize, handlelength=2.0, handletextpad=0.40,
                        columnspacing=1.0, labelspacing=0.35,
                        bbox_to_anchor=(0.5, y), frameon=False)
        if heading is not None:
            leg.get_texts()[0].set_fontweight("bold")
        # Each ax.legend() call replaces the axes' own legend, so every row but
        # the last must be re-added as a plain artist to survive.
        if i < len(legends) - 1:
            ax.add_artist(leg)


def draw_curve(ax, hop_data: dict[str, pd.Series], *, show_legend: bool = True) -> None:
    ks = list(range(0, MAX_HOP + 1))
    # Runs half a hop past the last tick, or k = MAX_HOP draws as a bare
    # vertical jump on the right spine with no readable final height.
    x_end = MAX_HOP + 0.5

    band_start, band_end = NEARBY_BAND
    ax.axvspan(band_start, band_end, facecolor="#E9F7EF", alpha=0.85,
               edgecolor="none", zorder=0, clip_on=False)
    ax.axvline(band_end, color="#2E8B57", lw=1.2, ls="--", alpha=0.85, zorder=0, clip_on=False)
    ax.axvline(band_start, color="#2E8B57", lw=1.2, ls="--", alpha=0.85, zorder=0, clip_on=False)
    ax.text(
        (band_start + band_end) / 2, 0.975,
        "Near-miss on the Cell Ontology  (≤2 hops)",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=style.MAIN_TEXT, color="#2E8B57",
        fontweight="semibold",
    )

    for label, color, ls in SERIES:
        if label not in hop_data:
            continue
        ys = cumulative_within_k(hop_data[label], ks)
        line_pe = [pe.Stroke(linewidth=2.0, foreground="#404040", alpha=0.30),
                   pe.Normal()]
        ax.step(ks + [x_end], ys + [ys[-1]], where="post", color=color, lw=1.4,
                linestyle=ls, label=_LEGEND_DISPLAY.get(label, label),
                zorder=3, path_effects=line_pe, clip_on=False)

    ax.set_xlim(0, x_end)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.tick_params(axis="both", labelsize=style.MAIN_TEXT)
    ax.set_xlabel("Hop tolerance, k", fontsize=style.MAIN_EMPHASIS)
    ax.set_ylabel("Cumulative accuracy", fontsize=style.MAIN_EMPHASIS, labelpad=2)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    if show_legend:
        _draw_grouped_legend(ax, hop_data)


def draw_explainer(ax, *, draw_ellipse: bool = True, draw_key: bool = True,
                   compact: bool = False) -> None:
    """Custom Cell-Ontology subtree schematic (0/1/2-hop tolerance).

    ``compact`` is the form that fits beside the curve rather than over it. It
    drops the two branches that only repeat the shape already shown (B cell,
    CD4+ T cell), names every node beside itself instead of above or below, and
    sets its own type sizes, because nine nodes and nine names will not fit in a
    column a third the width of the panel.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    if compact:
        nodes = {
            "Cell":           (5.0, 6.3),
            "Leukocyte":      (3.9, 5.0),
            "Stromal cell":   (6.1, 5.0),
            "T cell":         (3.3, 3.7),
            "CD8+ T cell":    (2.9, 2.4),
            # The two leaves are 2.8 units apart, not 2.3: at 2.3 their names
            # (13.0 mm and 11.1 mm) met in the middle and printed as one word.
            "Effector CD8+":  (1.85, 1.1),
            "Memory CD8+":    (4.65, 1.1),
        }
        edges = [
            ("Cell", "Leukocyte"), ("Cell", "Stromal cell"),
            ("Leukocyte", "T cell"), ("T cell", "CD8+ T cell"),
            ("CD8+ T cell", "Effector CD8+"), ("CD8+ T cell", "Memory CD8+"),
        ]
    else:
        nodes = {
            "Cell":           (6.2, 6.2),
            "Leukocyte":      (4.9, 5.0),
            "Stromal cell":   (7.6, 5.0),
            "T cell":         (4.0, 3.8),
            "B cell":         (5.8, 3.8),
            "CD8+ T cell":    (3.2, 2.4),
            "CD4+ T cell":    (4.8, 2.4),
            "Effector CD8+":  (2.4, 1.2),
            "Memory CD8+":    (3.7, 1.2),
        }
        edges = [
            ("Cell", "Leukocyte"), ("Cell", "Stromal cell"),
            ("Leukocyte", "T cell"), ("Leukocyte", "B cell"),
            ("T cell", "CD8+ T cell"), ("T cell", "CD4+ T cell"),
            ("CD8+ T cell", "Effector CD8+"), ("CD8+ T cell", "Memory CD8+"),
        ]

    if draw_ellipse:
        nearby = mpatches.Ellipse(
            (2.35, 1.95) if compact else (2.0, 2.3),
            width=4.3 if compact else 3.85, height=2.4 if compact else 2.5,
            angle=58 if compact else 70,
            facecolor="#E9F7EF", edgecolor="#2E8B57",
            linestyle="--", linewidth=1.2, zorder=0, clip_on=False,
        )
        ax.add_patch(nearby)

    for u, v in edges:
        x0, y0 = nodes[u]; x1, y1 = nodes[v]
        ax.plot([x0, x1], [y0, y1], color="#9E9E9E", lw=EDGE_WIDTH, zorder=1, clip_on=False)

    for name, (x, y) in nodes.items():
        if name == "Effector CD8+":
            # Gold, as the true label is marked in panel e. The blue ramp on
            # the other nodes is a different scale -- distance from this star --
            # and stays blue, the two hop counts included: each is coloured to
            # match the node it counts, not the star.
            ax.scatter([x], [y], marker="*", s=NODE_AREA_TRUTH,
                       color=style.TRUTH_FILL, edgecolors=style.TRUTH_EDGE,
                       lw=0.4, zorder=3, clip_on=False)
        elif name == "CD8+ T cell":
            ax.scatter([x], [y], s=NODE_AREA_NEAR, color="#5B8FF9",
                       edgecolors="black", lw=0.25, zorder=3, clip_on=False)
        elif name in ("T cell", "Memory CD8+"):
            ax.scatter([x], [y], s=NODE_AREA_NEAR, color="#B8CCE8",
                       edgecolors="black", lw=0.25, zorder=3, clip_on=False)
        else:
            ax.scatter([x], [y], s=NODE_AREA_FAR, color="#D9D9D9",
                       edgecolors="black", lw=0.2, zorder=2, clip_on=False)

    if compact:
        # Every name reads away from the tree, never into it. The three nodes
        # on the spine are named on their left, because to their right is where
        # the tree itself is and the word landed in the middle of it. The two
        # leaves are named underneath, both of them, since they share a height.
        label_specs = [
            ("Cell",          5.00, 6.62, "center", "bottom", style.MAIN_SECONDARY),
            ("Leukocyte",     3.70, 5.05, "right",  "center", style.MAIN_SECONDARY),
            ("Stromal cell",  6.32, 5.05, "left",   "center", style.MAIN_SECONDARY),
            ("T cell",        3.10, 3.75, "right",  "center", style.MAIN_SECONDARY),
            ("CD8+ T cell",   2.70, 2.45, "right",  "center", style.MAIN_SECONDARY),
            ("Effector CD8+", 1.85, 0.62, "center", "top",    style.MAIN_SECONDARY),
            ("Memory CD8+",   4.65, 0.62, "center", "top",    style.MAIN_SECONDARY),
        ]
        # Each count sits on the edge it counts: "1 hop" beside the step from
        # the query cell up to CD8+ T cell, "2 hops" beside the step on from
        # there to Memory CD8+.
        # Each count sits to the right of the node it counts, where the inset is
        # empty. Two nodes are two hops from the star and both say so: T cell up
        # the spine and Memory CD8+ out along the branch. They already carry the
        # two-hop shade, so the labels only name what the colour is saying.
        hop_specs = [
            ("1 hop",  3.15, 2.62, "left", "center", style.MAIN_SECONDARY, "#2F68B1"),
            ("2 hops", 3.55, 3.70, "left", "center", style.MAIN_SECONDARY, "#6F8FB3"),
            ("2 hops", 4.95, 1.15, "left", "center", style.MAIN_SECONDARY, "#6F8FB3"),
        ]
    else:
        label_specs = [
            ("Cell",          6.2, 6.55, "center", "bottom", 6.0),
            ("Leukocyte",     4.7, 5.18, "right",  "bottom", 6.0),
            ("Stromal cell",  7.75, 5.18, "left",  "bottom", 6.0),
            ("T cell",        3.55, 4.00, "right", "bottom", 6.8),
            ("B cell",        5.95, 3.65, "left",  "top",    6.0),
            ("CD8+ T cell",   2.85, 2.18, "right", "top",    6.8),
            ("CD4+ T cell",   4.98, 2.18, "left",  "top",    6.0),
            ("Effector CD8+", 2.30, 0.92, "right", "top",    6.8),
            ("Memory CD8+",   3.85, 0.92, "left",  "top",    6.8),
        ]
        hop_specs = [
            ("1 hop",  3.00, 1.95, "left", "center", 6.4, "#2F68B1"),
            ("2 hops", 3.85, 3.10, "left", "center", 6.4, "#6F8FB3"),
            ("2 hops", 3.85, 1.45, "left", "top",    6.4, "#6F8FB3"),
        ]
    for name, x, y, ha, va, fs in label_specs:
        ax.text(x, y, name, fontsize=fs, ha=ha, va=va)
    for name, x, y, ha, va, fs, color in hop_specs:
        ax.text(x, y, name, fontsize=fs, color=color, ha=ha, va=va,
                fontweight="bold")

    if draw_key:
        ax.text(9.85, 0.75,
                "0 hops = exact match\n"
                "1 hop = ontology neighbor\n"
                "2 hops = still nearby on the tree",
                fontsize=7.0, color="#444444", ha="right", va="bottom",
                linespacing=1.5)
