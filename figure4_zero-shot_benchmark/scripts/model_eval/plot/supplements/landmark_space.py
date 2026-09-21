"""Figure 4 supplement — the landmark space: how good it is, and what it cost.

Writes ``supplementary_figure_9_landmark_space.{pdf,png}``.

  a  how the shared read-out names a cell of an unseen type: the landmark it is
     most similar to, then the ontology term carrying the most mass there
  b  the landmark cells in each model's embedding, coloured by broad cell class
  c  the same cells coloured by donor
  d  every scIB measure behind the one score main panel a prints, plus the time
     each model took to encode the cells (encoding time)

Panel a was panel b of Figure 4 until 2026-09-14; it is drawn by
``../panels/mechanism.py``, which the main figure no longer calls.

One figure out of three. Until 2026-08-14 this was S1 (the scIB table), S3 (the
two UMAP rows) and S7 (a five-bar timing chart), which are three views of one
subject: the quality of the space the unseen cells are later placed into, and
its cost. The main figure prints only the composite score; the two halves and
all five component measures live here.

Reads ``result/plot_cache/{umap_coords,donor,broad_class}.pkl``,
``result/data/clustering_summary_comparison.csv`` and
``result/data/clustering_embedding_run_status.csv``.
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar

from ..paths import DATA_DIR as RESULT, RESULT_DIR, PLOT_CACHE_DIR
from ..style import MODEL_ORDER, MODEL_DISPLAY
from ..panels.mechanism import draw_panel_mechanism, legend_handles
from ..panels.umap import scib_scores

UMAP_CACHE = PLOT_CACHE_DIR / "umap_coords.pkl"
DONOR_CACHE = PLOT_CACHE_DIR / "donor.pkl"
BROAD_CACHE = PLOT_CACHE_DIR / "broad_class.pkl"
TIMING_CSV = "clustering_embedding_run_status.csv"
# Last real encode timings, carried across runs that reuse embeddings.
TIMING_KEEP = "encode_timings.csv"

N_TOP_BROAD = 20
OTHER_COLOR = "#D9D9D9"

# Full 3:4 page at the ~180 mm working width; fonts and marks scale with it.
CANVAS_WIDTH_IN = 180 / 25.4
CANVAS_HEIGHT_IN = CANVAS_WIDTH_IN * 4 / 3
DRAWING_SCALE = CANVAS_WIDTH_IN / (180 / 25.4)
EXPORT_DPI = 600 / DRAWING_SCALE
TEXT_BODY = 1.2 * CANVAS_WIDTH_IN * 72 / 85   # 20% larger than the other supplements' text
TEXT_SECONDARY = 0.9 * TEXT_BODY
TEXT_PANEL = 1.8 * TEXT_BODY

# Raw summary-CSV model names -> what the figure calls them. Lower case is not
# used anywhere else in figure 4, so scimilarity is written SCimilarity here too.
TABLE_DISPLAY = {
    "Hector": "HECTOR",
    "SCimilarity_v1.1": "SCimilarity",
    "scGPT_CP": "scGPT",
    "scCello-zeroshot": "scCello",
    "Geneformer-V2-104M": "Geneformer",
}

# The scIB summary names models by checkpoint ("Geneformer-V2-104M"); the
# timing file uses the short key ("geneformer"). Without this map, joining on
# the raw strings matches only HECTOR and the rest print "nan".
_TO_KEY = {
    "Hector": "Hector",
    "SCimilarity_v1.1": "scimilarity",
    "scGPT_CP": "scGPT",
    "scCello-zeroshot": "scCello",
    "Geneformer-V2-104M": "geneformer",
}


def _canonical(name: str) -> str:
    """Either naming of a model, reduced to the short key both can be joined on."""
    key = str(name).strip()
    return _TO_KEY.get(key, key)

# (csv column, printed name, group)
COLUMNS = [
    ("NMI",           "NMI",              "Bio conservation"),
    ("ARI",           "ARI",              "Bio conservation"),
    ("ASW",           "ASW (label)",      "Bio conservation"),
    ("ASWb",          "ASW (batch)",      "Batch correction"),
    ("GraphConn",     "Graph conn.",      "Batch correction"),
    ("AvgBio",        "Bio conservation", "scIB score"),
    ("AvgBatch",      "Batch correction", "scIB score"),
    ("overall_score", "Total",            "scIB score"),
    ("encode_s",      "seconds",          "Encoding time"),
]

# Statuses that are not a real, freshly-measured encode: these carry near-zero
# cache-check times, not encode time, and would make the one freshly-timed
# model look fastest by comparison.
NON_ENCODE_STATUSES = {"reused_existing", "failed", "missing_cell_embeddings",
                       "cached_mode"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 1.0,
    "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 350,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def shorten(name: str, n: int = 28) -> str:
    name = name.replace(", human", "")
    return name if len(name) <= n else name[: n - 1] + "…"


def make_categorical_palette(categories, cmap_name: str):
    cm = plt.get_cmap(cmap_name, len(categories) + 1)
    return {c: to_hex(cm(i)) for i, c in enumerate(categories)}


def make_glasbey_palette(categories):
    src = cc.glasbey_dark
    if len(categories) > len(src):
        raise ValueError(f"glasbey palette runs out: {len(categories)} > {len(src)}")
    out = {}
    for i, cat in enumerate(categories):
        v = src[i]
        out[cat] = v if isinstance(v, str) else to_hex(tuple(v))
    return out


def build_broad_palette(broad_cache) -> tuple[dict, list]:
    counter = Counter()
    for m in MODEL_ORDER:
        counter.update(broad_cache[m].tolist())
    top = [t for t, _ in counter.most_common(N_TOP_BROAD)]
    return make_categorical_palette(top, "tab20"), top


def build_donor_palette(donor_cache) -> tuple[dict, list]:
    donors = sorted({d for arr in donor_cache.values() for d in arr.tolist()})
    return make_glasbey_palette(donors), donors


def encode_seconds() -> dict[str, float]:
    """Seconds each model took to encode the benchmark cells, by short key.

    Rows whose status says the embedding was reused or the job failed are
    dropped: their elapsed time is a cache check, not an encode.
    """
    df = pd.read_csv(RESULT / TIMING_CSV)
    df["elapsed_seconds"] = pd.to_numeric(df["elapsed_seconds"], errors="coerce")
    fresh = df.loc[~df["status"].isin(NON_ENCODE_STATUSES)
                   & (df["elapsed_seconds"] > 0)].copy()
    if not fresh.empty:
        fresh["key"] = fresh["model"].map(_canonical)
        fresh = fresh.sort_values("elapsed_seconds").drop_duplicates("key", keep="last")
        out = dict(zip(fresh["key"], fresh["elapsed_seconds"].astype(float)))
        # Kept, because the run-status file is overwritten every run: a run that
        # reuses its embeddings records status "reused_existing" and an elapsed
        # time of 0, which is a cache check and not an encode. Without this the
        # last real measurement would be lost the first time the pipeline is
        # re-run.
        pd.DataFrame({"model": list(out), "elapsed_seconds": list(out.values())}
                     ).to_csv(RESULT / TIMING_KEEP, index=False)
        return out
    if (RESULT / TIMING_KEEP).exists():
        kept = pd.read_csv(RESULT / TIMING_KEEP)
        print(f"[landmark_space] embeddings were reused; encoding times read from {TIMING_KEEP}")
        return dict(zip(kept["model"].astype(str), kept["elapsed_seconds"].astype(float)))
    print("[landmark_space] no encoding times available; that column will be blank")
    return {}


def encode_cell_count() -> int | None:
    df = pd.read_csv(RESULT / TIMING_CSV)
    if "n_cells_requested" not in df.columns:
        return None
    vals = pd.to_numeric(df["n_cells_requested"], errors="coerce").dropna()
    return int(vals.iloc[0]) if not vals.empty else None


def load_table() -> pd.DataFrame:
    df = pd.read_csv(RESULT / "clustering_summary_comparison.csv")
    seconds = encode_seconds()
    df["encode_s"] = df["Model"].map(lambda m: seconds.get(_canonical(m)))
    missing = df.loc[df["encode_s"].isna(), "Model"].tolist()
    if missing:
        print(f"[landmark_space] no encode time for {missing}")
    df["Method"] = df["Model"].map(TABLE_DISPLAY)
    df = df[["Method"] + [c for c, *_ in COLUMNS]].copy()
    df = df.rename(columns={c: disp for c, disp, _ in COLUMNS})
    df = df.sort_values("Total", ascending=False).reset_index(drop=True)
    df = df.set_index("Method")
    df["Method"] = df.index
    return df


def draw_umap_row(fig, gs_row, umap, labels_by_model, palette, other_color,
                  titles=None) -> None:
    for j, m in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs_row[0, j])
        coords, labels = umap[m]["coords"], labels_by_model[m]
        colors = np.array([palette.get(l, other_color) if other_color is not None
                           else palette[l] for l in labels])
        rng = np.random.default_rng(0)
        order = rng.permutation(len(coords))
        ax.scatter(coords[order, 0], coords[order, 1], s=0.45 * DRAWING_SCALE**2, c=colors[order],
                   alpha=0.55, linewidths=0, rasterized=True, clip_on=False)
        # Square box, undistorted data: without forcing it, the frame takes the
        # shape of whatever grid cell it was handed and stretches the map.
        ax.set_box_aspect(1)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.5 * DRAWING_SCALE); sp.set_color("#888888")
        if titles is not None:
            # Separate model names from the score captions so equivalent text
            # has the same role in both UMAP rows.
            if "\n" in titles[m]:
                model_name, score_caption = titles[m].split("\n", 1)
                ax.set_title(model_name, fontsize=TEXT_BODY, pad=12 * DRAWING_SCALE,
                             fontweight="bold" if m == "Hector" else "normal")
                ax.text(0.5, 1.01, score_caption, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=TEXT_SECONDARY)
            else:
                ax.set_title(titles[m], fontsize=TEXT_SECONDARY, pad=2.5 * DRAWING_SCALE)


def draw_legend(ax, palette, ordered_keys, title, ncol, fontsize, has_other):
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=4.2 * DRAWING_SCALE,
               markerfacecolor=palette[k], markeredgecolor="white",
               markeredgewidth=0.3 * DRAWING_SCALE, label=shorten(k))
        for k in ordered_keys
    ]
    if has_other:
        handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=4.2 * DRAWING_SCALE,
                              markerfacecolor=OTHER_COLOR, markeredgecolor="white",
                              markeredgewidth=0.3 * DRAWING_SCALE, label="other"))
    leg = ax.legend(handles=handles, loc="center", ncol=ncol, fontsize=fontsize,
                    handletextpad=0.22, columnspacing=0.6, frameon=False,
                    borderpad=0.0, title=title,
                    title_fontsize=TEXT_BODY)
    leg.get_title().set_fontweight("bold")
    ax.axis("off")


def draw_table(ax, df) -> None:
    group_of = {disp: group for _c, disp, group in COLUMNS}
    per_metric = [disp for _c, disp, g in COLUMNS if g == "Bio conservation"
                  or g == "Batch correction"]
    aggregate = [disp for _c, disp, g in COLUMNS if g == "scIB score"]
    cost = [disp for _c, disp, g in COLUMNS if g == "Encoding time"]

    cmap_fn = lambda col: normed_cmap(col, cmap=mpl.cm.PRGn, num_stds=2.5)
    defs = [ColumnDefinition("Method", width=1.6,
                             textprops={"ha": "left", "weight": "bold"})]
    defs += [
        ColumnDefinition(col, title=col.replace(" ", "\n", 1), width=0.95,
                         textprops={"ha": "center",
                                    "bbox": {"boxstyle": "circle", "pad": 0.30}},
                         cmap=cmap_fn(df[col]), group=group_of[col],
                         formatter="{:.2f}")
        for col in per_metric
    ]
    defs += [
        ColumnDefinition(col, width=1.05, title=col.replace(" ", "\n", 1),
                         plot_fn=bar,
                         plot_kw={"cmap": mpl.cm.YlGnBu, "plot_bg_bar": False,
                                  "annotate": True, "height": 0.9,
                                  "formatter": "{:.2f}",
                                  "textprops": {"fontsize": TEXT_BODY}},
                         group=group_of[col],
                         border="left" if i == 0 else None)
        for i, col in enumerate(aggregate)
    ]
    # Seconds, drawn as a bar so the spread is visible. Flat grey, no colour
    # scale: low is good here but high is good in every column to its left.
    span = (0.0, float(df[cost[0]].max()) * 1.06) if cost else (0.0, 1.0)
    defs += [
        ColumnDefinition(col, width=1.0, title=col.replace(" ", "\n", 1),
                         plot_fn=bar,
                         plot_kw={"xlim": span, "color": "#9E9E9E",
                                  "plot_bg_bar": False, "annotate": True,
                                  "height": 0.9, "formatter": "{:.1f}",
                                  "textprops": {"fontsize": TEXT_BODY}},
                         group=group_of[col], border="left")
        for col in cost
    ]

    with mpl.rc_context({"svg.fonttype": "none"}):
        Table(df, cell_kw={"linewidth": 0, "edgecolor": "k"},
              column_definitions=defs, ax=ax, row_dividers=True,
              footer_divider=True,
              textprops={"fontsize": TEXT_BODY, "ha": "center"},
              row_divider_kw={"linewidth": DRAWING_SCALE, "linestyle": (0, (1, 5))},
              col_label_divider_kw={"linewidth": DRAWING_SCALE, "linestyle": "-"},
              column_border_kw={"linewidth": DRAWING_SCALE, "linestyle": "-"},
              index_col="Method").autoset_fontcolors(colnames=df.columns)


def render(out_dir=RESULT_DIR) -> Path:
    out_dir = Path(out_dir)
    umap = pickle.load(open(UMAP_CACHE, "rb"))
    donor = pickle.load(open(DONOR_CACHE, "rb"))
    broad = pickle.load(open(BROAD_CACHE, "rb"))
    table = load_table()
    bc_palette, bc_top = build_broad_palette(broad)
    dn_palette, dn_keys = build_donor_palette(donor)

    fig = plt.figure(figsize=(CANVAS_WIDTH_IN, CANVAS_HEIGHT_IN))
    page_height_mm = CANVAS_HEIGHT_IN * 25.4
    plot_top = 1 - 5 * DRAWING_SCALE / page_height_mm
    occupied_mm = 7.0 * 25.4 * 0.94 * DRAWING_SCALE
    normal_gap_mm = occupied_mm * 0.16 / (5 + 4 * 0.16)
    row_weights = np.array([17.5, 7.0, 17.5, 4.5, 30.0])
    row_mm = row_weights / row_weights.sum() * (occupied_mm - 4 * normal_gap_mm)
    # The read-out schematic (panel a) is drawn in its own 0-1 axes at ~1.6x
    # wider than tall; a wider box only flattens the ellipses and arrows rather
    # than adding room, so it takes the width its height allows and the key
    # fills the rest of the row.
    schematic_mm = 51.0 * DRAWING_SCALE
    schematic_gap_mm = 8.0 * DRAWING_SCALE
    band_mm = [schematic_mm, schematic_gap_mm,
               row_mm[0], normal_gap_mm, row_mm[1],
               normal_gap_mm + 6 * DRAWING_SCALE, row_mm[2], normal_gap_mm,
               row_mm[3], normal_gap_mm, row_mm[4]]
    plot_bottom = plot_top - sum(band_mm) / page_height_mm
    page_grid = fig.add_gridspec(len(band_mm), 1, height_ratios=band_mm,
                                hspace=0, left=0.030, right=0.985,
                                top=plot_top, bottom=plot_bottom)
    # Panel a sits on band 0. The five remaining rows start at band 2.
    outer = {(0, 0): page_grid[0, 0]}
    outer.update({(i + 1, 0): page_grid[2 + 2 * i, 0] for i in range(5)})

    # Each row carries the score for the thing that row shows: the maps coloured
    # by cell class carry the biological conservation score, the maps coloured by
    # donor carry the batch correction score, so the number over each row always
    # matches what the picture beneath it measures.
    bio = scib_scores("avg_bio")
    batch = scib_scores("avg_batch")

    # Panel a — how a cell of an unseen type is named: the landmark it is most
    # similar to, then the ontology term carrying the most mass at that
    # landmark. It is the read-out that builds everything below it, so it comes
    # first.
    # 123 mm of drawing beside 44 mm of key. 123 by the band's 51 mm is the 2.42
    # the schematic is laid out for — see ``ASPECT`` in ../panels/mechanism.py,
    # which sizes every round mark in it.
    schematic_cols = outer[0, 0].subgridspec(1, 2, width_ratios=[122, 44],
                                             wspace=0.05)
    draw_panel_mechanism(fig.add_subplot(schematic_cols[0, 0]),
                         show_subtitle=False, show_legend=False,
                         scale=0.62, text_scale=0.95)
    ax_key = fig.add_subplot(schematic_cols[0, 1]); ax_key.axis("off")
    ax_key.legend(handles=legend_handles(), loc="center", ncol=1,
                  fontsize=TEXT_SECONDARY, handletextpad=0.5,
                  labelspacing=1.1, frameon=False, borderpad=0.0)

    draw_umap_row(fig, outer[1, 0].subgridspec(1, len(MODEL_ORDER), wspace=0.06),
                  umap, broad, bc_palette, OTHER_COLOR,
                  titles={m: f"{MODEL_DISPLAY[m]}\nbio conservation {bio[m]:.2f}"
                          for m in MODEL_ORDER})
    draw_legend(fig.add_subplot(outer[2, 0]), bc_palette, bc_top,
                title=f"Top {N_TOP_BROAD} broad cell classes "
                      f"(~82% of cells; rest = 'other', light grey)",
                ncol=6, fontsize=TEXT_SECONDARY, has_other=True)

    draw_umap_row(fig, outer[3, 0].subgridspec(1, len(MODEL_ORDER), wspace=0.06),
                  umap, donor, dn_palette, None,
                  titles={m: f"batch correction {batch[m]:.2f}" for m in MODEL_ORDER})
    draw_legend(fig.add_subplot(outer[4, 0]), dn_palette, dn_keys,
                title=f"Donor (n = {len(dn_keys)})", ncol=12,
                fontsize=TEXT_SECONDARY,
                has_other=False)

    # The two rotated row labels the old supplement carried. Panel letters say
    # which row is which but not what its colours mean, and the legend under a
    # row is read after it, not before.
    for row, text in ((1, "by broad cell class"), (3, "by donor")):
        row_box = outer[row, 0].get_position(fig)
        y = (row_box.y0 + row_box.y1) / 2
        fig.text(0.010, y, text, rotation=90, ha="left", va="center",
                 fontsize=TEXT_BODY, fontweight="bold", color="#444444")

    draw_table(fig.add_subplot(outer[5, 0]), table)

    for letter, spec in (("a", outer[0, 0]), ("b", outer[1, 0]),
                         ("c", outer[3, 0]), ("d", outer[5, 0])):
        box = spec.get_position(fig)
        fig.text(0.008, box.y1, letter, fontsize=TEXT_PANEL,
                 fontweight="bold", va="top")

    out_pdf = out_dir / "supplementary_figure_9_landmark_space.pdf"
    # Export the whole page from the same figure. Dense point layers rasterize
    # at the target insertion resolution; PDF labels and table remain vector.
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(out_pdf, dpi=EXPORT_DPI, bbox_inches=None,
                    facecolor="white", transparent=False)
        fig.savefig(out_dir / "supplementary_figure_9_landmark_space.png",
                    dpi=EXPORT_DPI, bbox_inches=None,
                    facecolor="white", transparent=False)
    plt.close(fig)
    return out_pdf
