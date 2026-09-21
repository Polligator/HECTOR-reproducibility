"""Figure 4 — panel a: 5-model UMAP small-multiple grid, colored by lineage.

Coordinates come from ``result/plot_cache/umap_coords.pkl`` (the relocated,
locked layout). Bio-conservation scores are read **live** from the per-model
clustering summaries (not hardcoded), so a fresh run updates them.
"""
from __future__ import annotations

import pickle

import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
from matplotlib.lines import Line2D

from ..paths import DATA_DIR as RESULT_DIR, PLOT_CACHE_DIR

# Compartment (lineage) colors — Okabe-Ito + brown/gray, colorblind-aware.
COMP_COLORS = {
    "Immune / blood":        "#0072B2",
    "Epithelial":            "#E69F00",
    "Endothelial":           "#009E73",
    "Stromal / fibroblast":  "#CC79A7",
    "Muscle":                "#D55E00",
    "Neural / glial":        "#56B4E9",
    "Germ":                  "#8C564B",
    "Other":                 "#C8C8C8",
}
COMP_ORDER = ["Immune / blood", "Epithelial", "Endothelial",
              "Stromal / fibroblast", "Muscle", "Neural / glial", "Germ", "Other"]

_BIO_PREFIX = {"Hector": "hector", "scimilarity": "scimilarity", "scGPT": "scgpt",
               "scCello": "sccello", "geneformer": "geneformer_104m"}
_UMAP_CACHE = None


def _cache():
    """Lazy-load the relocated UMAP coordinate cache."""
    global _UMAP_CACHE
    if _UMAP_CACHE is None:
        with open(PLOT_CACHE_DIR / "umap_coords.pkl", "rb") as f:
            _UMAP_CACHE = pickle.load(f)
    return _UMAP_CACHE


def scib_scores(metric: str = "overall_score") -> dict[str, float]:
    """One scIB column per model, read live from the clustering summary CSVs.

    ``overall_score`` is the conventional scIB composite, 0.6 x the biological
    conservation score plus 0.4 x the batch correction score, and is what panel
    a prints. The two halves are ``avg_bio`` and ``avg_batch``; the full table
    of five measures is Supplementary Fig. 1.

    Panel a printed ``avg_bio`` alone until 2026-08-14. The ranking is the same
    either way, but showing only the half of a composite that flatters the
    model is the kind of thing a reader asks about, so the composite is what
    prints and the halves live in the supplement.
    """
    out = {}
    for key, prefix in _BIO_PREFIX.items():
        df = pd.read_csv(RESULT_DIR / f"{prefix}_clustering_metrics_summary.csv")
        out[key] = float(df[metric].iloc[0])
    return out


# Every layer is drawn solid, not see-through. Measured across 138 UMAP panels in the six
# papers under reference/, a published map is drawn opaque, so each cluster reads as one
# clean shape; these were drawn at 0.55 and came out speckled, and where two colours
# overlapped they blended into a hue belonging to no lineage at all. The "Other" layer,
# 8.6 % of the cells, keeps its place in the background by being a pale grey underneath
# everything else rather than by being see-through.
UMAP_OPAQUE = 1.0


def draw_umap(ax, model, *, point_size=2.2, title=None, show_axis_label=True):
    d = _cache()[model]
    coords, comp = d["coords"], d["compartment"]
    # plot "Other" underneath, in the background grey
    other = comp == "Other"
    ax.scatter(coords[other, 0], coords[other, 1], s=point_size,
               c=COMP_COLORS["Other"], alpha=UMAP_OPAQUE, linewidths=0,
               rasterized=True, clip_on=False)
    for c in COMP_ORDER:
        if c == "Other":
            continue
        m = comp == c
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=point_size, c=COMP_COLORS[c],
                   alpha=UMAP_OPAQUE, linewidths=0, rasterized=True, label=c, clip_on=False)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if show_axis_label:
        ax.set_xlabel("UMAP 1", fontsize=6, labelpad=1)
        ax.set_ylabel("UMAP 2", fontsize=6, labelpad=1)
    if title:
        ax.set_title(title, fontsize=7.5, pad=3)
    ax.set_aspect("equal", adjustable="datalim")


def comp_legend_handles():
    return [Line2D([0], [0], marker="o", linestyle="", markersize=4.5,
                   markerfacecolor=COMP_COLORS[c], markeredgewidth=0, label=c)
            for c in COMP_ORDER]
