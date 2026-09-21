from __future__ import annotations

"""Assembler for Figure 2 (human kidney): panels A-D plus the supplements.

compute_deepdive_artifacts does the deep-dive compute (ontology matching,
per-cell prediction table, UMAP coordinates) once and caches it; every drawer
below reads that cache and never recomputes.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_benchmark")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_scanpy")

import matplotlib as mpl
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from benchmark_modules import panel_d_flow

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

mpl.rcParams.update({
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
})

HECTOR_ACCENT = "#FB8072"

# Main figure typography, sized for its 270 mm working page.
MAIN_WIDTH_MM = 270
MAIN_HEIGHT_MM = 360
MAIN_BODY_PT = 9.0
MAIN_SECONDARY_PT = 7.2
MAIN_DENSE_PT = 6.5
MAIN_PANEL_PT = 16.2

SUPPLEMENT_BODY_FONT_SIZE = 9
SUPPLEMENT_PANEL_TITLE_SIZE = 11

# Dot area in points squared. Diameter scales with the square root of the plotted
# cloud's width, capped so no dot in figures 2-4 prints more than ~1.7x another.
PANEL_A_DOT_AREA = 0.95
PANEL_C_DOT_AREA = 2.0

# Drawn opaque: overlapping translucent clusters blend into a hue belonging to no
# cell type.
UMAP_OPAQUE = 1.0

# figure2_basic_benchmark/, holding input/, scripts/ and result/.
_FIG2_ROOT = Path(__file__).resolve().parents[2]
# Deep-dive cache, shared with the prediction cache (.cache/<ds>/). Not shipped.
CACHE_ROOT = _FIG2_ROOT / ".cache"

_CACHE_FILES = ["deepdive_umap.csv", "shared_types.json"]


def cache_dir(dataset_key: str) -> Path:
    d = CACHE_ROOT / dataset_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))


def deepdive_cache_complete(dataset_key: str) -> bool:
    d = CACHE_ROOT / dataset_key
    return all((d / n).exists() for n in _CACHE_FILES)


def ensure_deepdive(config: dict) -> None:
    """Compute the deep-dive artifacts into the cache if they're not already there
    (compute-if-missing). Runs HECTOR once; both figure scripts then read the same
    cached artifacts, so the deep-dive HECTOR pass never runs twice per pipeline."""
    dataset_key = config["dataset_key"]
    if deepdive_cache_complete(dataset_key):
        print(f"  [{dataset_key}] using cached deep-dive artifacts ({CACHE_ROOT / dataset_key})")
    else:
        print(f"  [{dataset_key}] computing deep-dive artifacts (HECTOR)...")
        compute_deepdive_artifacts(config)


def compute_deepdive_artifacts(config: dict) -> dict:
    """Compute the deep-dive artifacts once and cache them.

    Two files: the per-cell truth/prediction table with UMAP coordinates, and the list
    of types shared by both vocabularies. Drawers read the cache and never recompute.
    """
    import matplotlib
    matplotlib.use("Agg")
    import anndata as ad

    from benchmark_modules.benchmark_core import load_or_create_cell_ids
    from benchmark_modules.hector_helpers import (
        run_hector_prediction, prepare_subset, _build_name_to_cl_id, _label_to_cl_id,
        _is_valid_cl_id,
    )
    from benchmark_modules.ontology_compare import parse_obo_file

    data_path = config["data_path"]
    output_dir = cache_dir(config["dataset_key"])
    label_key = config["label_key"]
    truth_id_key = config.get("truth_id_key", "cell_type_ontology_term_id")
    sample_size = config["sample_size"]
    random_state = config["random_state"]
    obo_path = config["obo_path"]
    predict_full_dataset = config.get("predict_full_dataset", True)
    predictor_config = {"hector": dict(config["hector"])}

    adata = ad.read_h5ad(data_path)
    # Scores the same cell sample the benchmark pinned to cell_ids.txt.
    chosen_cells = load_or_create_cell_ids(
        adata,
        sample_size=sample_size,
        random_state=random_state,
        cell_id_file=config["cell_id_file"],
    )
    subset = prepare_subset(adata, chosen_cells)

    prediction_adata = adata if predict_full_dataset else subset
    predictor, predictions, _, _ = run_hector_prediction(predictor_config, prediction_adata, verbose=True)
    predictor.write_predictions(prediction_adata, predictions, rare_rollup=False)

    if predict_full_dataset:
        subset.obs["hector_prediction"] = adata.obs.loc[subset.obs_names, "hector_prediction"].values

    if truth_id_key not in subset.obs.columns:
        raise KeyError(
            f"compute_deepdive_artifacts matches on ontology IDs and needs "
            f"subset.obs[{truth_id_key!r}], which is missing. "
            f"Available obs columns: {list(subset.obs.columns)}"
        )

    y_true_name = subset.obs[label_key].astype(str)
    y_pred_name = subset.obs["hector_prediction"].astype(str)

    y_true_id = (
        subset.obs[truth_id_key]
        .astype(str)
        .str.strip()
        .str.replace(r"^([A-Za-z]+)_(\d+)$", r"\1:\2", regex=True)
    )
    pred_name_to_cl_id = _build_name_to_cl_id(predictor)
    if not pred_name_to_cl_id:
        print(
            "  WARNING: predictor exposed no id_to_name_map; predicted labels "
            "will be matched by name only."
        )
    pred_id_lookup = {
        label: (_label_to_cl_id(label, pred_name_to_cl_id) or "")
        for label in y_pred_name.unique()
    }
    y_pred_id = y_pred_name.map(pred_id_lookup)

    with open(str(obo_path), "r") as f:
        obo_content = f.read()
    relationships, cl_names, _ = parse_obo_file(obo_content)

    gt_name_by_id: dict[str, str] = {}
    for cid, nm in zip(y_true_id, y_true_name):
        if _is_valid_cl_id(cid):
            gt_name_by_id.setdefault(cid, nm)

    def _canonical_by_id(id_series: pd.Series, name_series: pd.Series) -> pd.Series:
        id_to_disp = {
            cid: (cl_names.get(cid) or gt_name_by_id.get(cid))
            for cid in id_series.unique()
            if _is_valid_cl_id(cid)
        }
        disp = id_series.map(id_to_disp)
        return disp.where(disp.notna(), name_series).astype(str)

    y_true_raw = _canonical_by_id(y_true_id, y_true_name)
    y_pred_raw = _canonical_by_id(y_pred_id, y_pred_name)

    valid_types = sorted(set(y_true_raw.unique()) & set(y_pred_raw.unique()))
    name_only_shared = len(set(y_true_name.unique()) & set(y_pred_name.unique()))
    print(
        f"Valid shared types: {len(valid_types)} by ontology ID "
        f"(was {name_only_shared} by exact name)"
    )
    if not valid_types:
        raise ValueError("No overlap between ground truth and predicted labels.")

    subset.obs["cell_type_plot"] = y_true_raw.where(y_true_raw.isin(valid_types), other="other")
    subset.obs["hector_prediction_plot"] = y_pred_raw.where(y_pred_raw.isin(valid_types), other="other")

    y_true = subset.obs["cell_type_plot"]
    y_pred = subset.obs["hector_prediction_plot"]
    top1_accuracy = (y_true == y_pred).mean()
    print(f"EXACT MATCH Top-1: {top1_accuracy:.2%}")

    if "X_umap" not in subset.obsm:
        raise ValueError("compute_deepdive_artifacts needs precomputed subset.obsm['X_umap'].")

    # ----- deep-dive UMAP frame (coords already in subset.obsm['X_umap']) -----
    umap = np.asarray(subset.obsm["X_umap"])[:, :2]
    umap_df = pd.DataFrame({
        "cell_id": subset.obs_names.astype(str),
        "umap1": umap[:, 0],
        "umap2": umap[:, 1],
        "cell_type_plot": subset.obs["cell_type_plot"].astype(str).values,
        # Uncollapsed ground-truth label, so panel C can colour every annotated
        # type, not only the ones in the shared confusion set.
        "cell_type_full": y_true_raw.astype(str).values,
        "hector_prediction_plot": subset.obs["hector_prediction_plot"].astype(str).values,
        # Uncollapsed prediction, symmetric with cell_type_full above.
        "hector_prediction_full": y_pred_raw.astype(str).values,
    })

    # ----- write cache -----
    umap_df.to_csv(output_dir / "deepdive_umap.csv", index=False)
    _save_json(output_dir / "shared_types.json", list(map(str, valid_types)))
    return {"n_cells": int(len(subset)), "n_shared_types": len(valid_types),
            "top1_accuracy": float(top1_accuracy)}


# -----------------------------------------------------------------------------
# Bottom-region drawers (consume the cached arrays; no recomputation)
# -----------------------------------------------------------------------------

def _color_pool(extended=False):
    """Distinct, non-grey categorical colours.

    The head (tab20/tab20b/tab20c, near-greys removed) is the *exact* pool
    `_build_type_palette` has always used, so the shared-type colours never move.
    `extended=True` appends more distinct colours for datasets with many
    ground-truth-only types (panel C needs up to ~50).
    """
    import matplotlib.cm as cm
    from matplotlib.colors import to_hex

    pool = []
    for cmap in (cm.tab20, cm.tab20b, cm.tab20c):  # unchanged head (locks shared colours)
        for i in range(cmap.N):
            r, g, b, _ = cmap(i)
            if max(r, g, b) - min(r, g, b) >= 0.12:  # skip near-greys
                pool.append(to_hex((r, g, b)))
    if extended:
        for cmap in (cm.Set1, cm.Dark2, cm.tab10, cm.Set2, cm.Paired, cm.Accent):
            for i in range(cmap.N):
                r, g, b = cmap(i)[:3]
                if max(r, g, b) - min(r, g, b) >= 0.12:
                    h = to_hex((r, g, b))
                    if h not in pool:
                        pool.append(h)
    return pool


def _build_type_palette(dataset_key):
    """One shared per-type colour map for the SHARED types in panels C and D.

    - Distinct, **non-grey** colours for the valid (shared) cell types, so none of
      them can be confused with the grey "other" bucket (HECTOR's minority calls).
    - Assigned by *sorted type name*, so the same cell type gets the same colour
      in every panel (UMAPs, ontology dots, confusion dots).
    """
    types = sorted(json.loads((CACHE_ROOT / dataset_key / "shared_types.json").read_text()))
    pool = _color_pool()
    palette = {t: pool[i % len(pool)] for i, t in enumerate(types)}
    palette["other"] = "#b5b5b5"  # the only grey — reserved for "other" (predictions)
    return palette


def _full_gt_palette(dataset_key, extra_types):
    """Full ground-truth palette for panel C: the shared types keep their EXACT
    `_build_type_palette` colours (locked to panels C/D), and every ground-truth-only
    type (not in HECTOR's label set) gets its own distinct colour — never grey, since
    grey is reserved for the prediction "other" bucket. So ground-truth cells are
    always shown at full granularity in a real colour.
    """
    base = _build_type_palette(dataset_key)            # shared + "other"; shared colours locked
    used = set(base.values())
    avail = [c for c in _color_pool(extended=True) if c not in used]
    palette = dict(base)
    for i, t in enumerate(sorted(extra_types)):
        if i < len(avail):
            palette[t] = avail[i]
        else:  # pool exhausted (only for pathologically many types) -> deterministic HSV fallback
            import colorsys
            r, g, b = colorsys.hsv_to_rgb((0.137 * (i + 1)) % 1.0, 0.62, 0.92)
            palette[t] = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    return palette


def _deepdive_gt_column(df):
    """Full (uncollapsed) ground-truth labels if cached, else the collapsed column
    (graceful fallback for caches written before cell_type_full was added)."""
    return "cell_type_full" if "cell_type_full" in df.columns else "cell_type_plot"


# Closest two numbers may sit, as a fraction of the map's span; must exceed the
# badge diameter or circles touch.
BADGE_MIN_SEP = 0.034
# Digit opacity floor for 3:1 contrast against the darkest cluster.
BADGE_TEXT_ALPHA = 0.70


def _draw_cluster_badges(ax, df, series, keyed, numbering, fontsize, alpha):
    """Draw each cell type's number on the body of its own cluster.

    Shared by panel C in the main figure and the full-page version in the detail supplement,
    so a cell type sits at the same place with the same number in both.
    """
    present = sorted(set(series) & keyed)
    if not present:
        return
    # Median, not mean: robust to a scattered tail.
    homes = np.array([[float(np.median(df["umap1"][(series == t).values])),
                       float(np.median(df["umap2"][(series == t).values]))]
                      for t in present])
    span = max(float(df["umap1"].max() - df["umap1"].min()),
               float(df["umap2"].max() - df["umap2"].min()))
    ax.figure.canvas.draw()
    badge_pt = panel_d_flow.badge_diameter_pt(max(numbering.values(), default=9), fontsize)
    axis_width_pt = ax.get_position().width * ax.figure.get_size_inches()[0] * 72
    badge_units = badge_pt * (ax.get_xlim()[1] - ax.get_xlim()[0]) / axis_width_pt
    placed = _spread_badges(homes, min_sep=max(span * BADGE_MIN_SEP, badge_units * 1.15),
                            max_shift=span * 0.08, iters=500)
    for home, position in zip(homes, placed):
        if np.linalg.norm(position - home) > badge_units * 0.5:
            ax.plot([home[0], position[0]], [home[1], position[1]],
                    color="#555555", linewidth=0.35, alpha=0.65, zorder=3)
    # A marker, not a text background box, so it is sized in points like panel D's
    # badges rather than fitted to the string. Kept faint so it doesn't punch a
    # hole in the cells it labels.
    badge_s = panel_d_flow.badge_diameter_pt(max(numbering.values(), default=9), fontsize) ** 2
    ax.scatter(placed[:, 0], placed[:, 1], marker="o", s=badge_s, facecolors="white",
               edgecolors="none", alpha=alpha, linewidths=0, zorder=4, clip_on=False)
    for term, (bx, by) in zip(present, placed):
        ax.text(bx, by, str(numbering[term]), fontsize=fontsize, ha="center", va="center",
                fontweight="bold", color="black", alpha=BADGE_TEXT_ALPHA, zorder=5)


def draw_deepdive_umaps(fig, spec, dataset_key, numbering, point_size=PANEL_C_DOT_AREA,
                        badge_fontsize=panel_d_flow.NUMBER_PT, badge_alpha=0.30,
                        text_size=8, row_gap=0.08):
    """Stacked Ground-Truth and HECTOR-predicted UMAPs from the cached frame.

    Both rows share one palette with panel D's node bars (_panel_cd_palette), so a
    cell type keeps one colour across the figure and panel D doubles as this
    panel's key. Terms too rare to show individually fall back to grey — grey
    means "pooled", never "unclassified".
    """
    d = CACHE_ROOT / dataset_key
    df = pd.read_csv(d / "deepdive_umap.csv")
    base = _build_type_palette(dataset_key)
    gt_col = _deepdive_gt_column(df)
    palette = _panel_cd_palette(dataset_key)
    pred_col = ("hector_prediction_full" if "hector_prediction_full" in df.columns
                else "hector_prediction_plot")

    inner = spec.subgridspec(2, 1, hspace=row_gap)
    axes = []
    for row, (col, title) in enumerate([
        (gt_col, "Ground Truth Cell Types"),
        (pred_col, "HECTOR Predicted Cell Types"),
    ]):
        ax = fig.add_subplot(inner[row, 0])
        series = df[col].astype(str)
        colors = series.map(palette).fillna(base["other"])
        ax.scatter(df["umap1"], df["umap2"], c=colors, s=point_size, alpha=UMAP_OPAQUE,
                   linewidths=0, rasterized=True, clip_on=False)
        # Panel D is this panel's key: each number here matches a label there.
        ax.set_title(title, fontsize=text_size)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        map_half_span = 0.55 * max(np.ptp(df["umap1"]), np.ptp(df["umap2"]))
        x_center = (df["umap1"].min() + df["umap1"].max()) / 2
        y_center = (df["umap2"].min() + df["umap2"].max()) / 2
        ax.set_xlim(x_center - map_half_span, x_center + map_half_span)
        ax.set_ylim(y_center - map_half_span, y_center + map_half_span)
        ax.set_autoscale_on(False)
        if row == 1:
            ax.set_xlabel("UMAP1", fontsize=text_size)
        ax.set_ylabel("UMAP2", fontsize=text_size)
        _draw_cluster_badges(ax, df, series, set(numbering), numbering,
                             badge_fontsize, badge_alpha)
        axes.append(ax)
    return axes


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

def kidney_config() -> dict:
    """Deep-dive + benchmark config for human kidney. Benchmark keys (incl. the
    .cache/ results dir) come from dataset_configs; deep-dive keys are added here."""
    from benchmark_modules import benchmark_core as core
    from benchmark_modules import dataset_configs as dsc

    config = core.resolve_dataset_config(dsc.benchmark_config("kidney"))
    config["obo_path"] = str(_FIG2_ROOT / "input" / "cl.obo")
    config["truth_id_key"] = "cell_type_ontology_term_id"
    config["selected_method"] = "ward"
    config["selected_strategy"] = "untangle_step1side"
    config["predict_full_dataset"] = True
    return config


def write_kidney_tables(config: dict, out_dir) -> None:
    """Write human-kidney's interpretable tables to out_dir: per-class F1 (from the
    benchmark cache) and the ontology-vs-Ward topology stats (from the deep-dive cache
    — call this after ensure_deepdive). No confusion matrix; see the note below."""
    from benchmark_modules import benchmark_core as core

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dkey = config["dataset_key"]
    rd = Path(config["results_dir"])
    metrics = core.load_combined_metrics(rd)
    ordered = core.figure1_model_order(metrics)
    per_class = core.load_per_class_artifacts(rd, ordered)
    class_order = core.per_class_column_order(per_class)
    core.build_per_class_heatmap_frame(per_class, ordered, class_order).to_csv(out / f"{dkey}_per_class_f1.csv")

    # No confusion matrix: ground truth and HECTOR use different vocabularies, so
    # an ontology-correct but non-exact call has no diagonal cell to fall on.
    # Panel D reports ontology-aware categories instead. Per-cell table:
    # .cache/<ds>/hector_predictions_HECTOR.csv.


def write_benchmark_metrics_table(dataset_keys, results_root, out_dir) -> None:
    """Write benchmark_metrics.csv: every tool × metric, one block per dataset."""
    from benchmark_modules import benchmark_core as core

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    frames = []
    for key in dataset_keys:
        m = core.load_combined_metrics(Path(results_root) / key).copy()
        m.insert(0, "dataset", key)
        frames.append(m)
    pd.concat(frames, ignore_index=True, sort=False).to_csv(out / "benchmark_metrics.csv", index=False)


# -----------------------------------------------------------------------------
# Benchmark scores by method and metric (circle heatmap). Reimplemented here so the colour
# scale and labelling suit the combined figure; the values are the benchmark metrics on disk.
# -----------------------------------------------------------------------------

def macro_cmap(name):
    from matplotlib.colors import LinearSegmentedColormap as LSC
    scales = {
        "diverging":        (["#C46A58", "#F2EEE7", "#4F7C91"], "diverging"),
        "sequential_teal":  (["#F1F6F7", "#7FA6B4", "#4F7C91", "#2C4A57"], "sequential"),
        "sequential_purple":(["#FAF4FB", "#D2AAD8", "#B060B6", "#98348e"], "sequential"),
        "sequential_slate": (["#F4F7FA", "#9FBACD", "#5F86A3", "#31536B"], "sequential"),
    }
    colors, mode = scales[name]
    return LSC.from_list(name, colors), mode


def macro_norm(frame, mode):
    from matplotlib.colors import Normalize, TwoSlopeNorm
    vals = pd.to_numeric(pd.Series(np.asarray(frame.to_numpy()).ravel()), errors="coerce").dropna()
    vmin, vmax = float(vals.min()), float(vals.max())
    if mode == "diverging":
        return TwoSlopeNorm(vmin=vmin, vcenter=float(vals.mean()), vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def draw_macro_matrix(ax, frame, cmap, norm, value_fmt=".3f", hector_row="HECTOR",
                      grid_color="#E6E2DB", label_fontsize=6.5,
                      value_fontsize=5.2, x_label_rotation=45):
    import matplotlib.patches as patches
    from matplotlib.colors import to_rgb

    n_rows, n_cols = frame.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_xticks(range(n_cols)); ax.set_yticks(range(n_rows))
    ax.set_xticklabels(
        frame.columns,
        rotation=x_label_rotation,
        ha="right",
        rotation_mode="anchor",
        fontsize=label_fontsize,
    )
    ax.set_yticklabels(frame.index, fontsize=label_fontsize)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=grid_color, linewidth=0.8)
    ax.tick_params(which="both", length=0, pad=4)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n_rows):
        for j in range(n_cols):
            v = frame.iat[i, j]
            if pd.isna(v):
                ax.add_patch(patches.Circle((j, i), 0.44, facecolor="none",
                                            edgecolor="#CFCFCF", linewidth=1.2, clip_on=False))
                ax.text(j, i, "NA", ha="center", va="center", fontsize=value_fontsize,
                        color="#9A9A9A", style="italic")
                continue
            fc = cmap(norm(float(v)))
            ax.add_patch(patches.Circle((j, i), 0.44, facecolor=fc,
                                        edgecolor="white", linewidth=0.8, clip_on=False))
            r, g, b = to_rgb(fc)
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(j, i, format(float(v), value_fmt), ha="center", va="center",
                    fontsize=value_fontsize, color="white" if lum < 0.5 else "#1A1A1A")
    if hector_row in list(frame.index):
        hi = list(frame.index).index(hector_row)
        ax.add_patch(patches.Rectangle((-0.5, hi - 0.5), n_cols, 1, fill=False,
                                       edgecolor=HECTOR_ACCENT, linewidth=1.8, clip_on=False))
        ax.get_yticklabels()[hi].set_color(HECTOR_ACCENT)
        ax.get_yticklabels()[hi].set_fontweight("bold")


# -----------------------------------------------------------------------------
# Top benchmark region. Reuses the benchmark_core drawers unchanged; only the arrangement
# differs. Per-class F1 lives in the detail supplement, not here.
# -----------------------------------------------------------------------------

def draw_benchmark_region(fig, spec, config):
    import math
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    from textwrap import fill

    from benchmark_modules import benchmark_core as core

    results_dir = Path(config["results_dir"])
    metrics = core.load_combined_metrics(results_dir)
    ordered_models = core.figure1_model_order(metrics)

    prediction_tables = core._load_prediction_tables(results_dir, metrics["model_name"].tolist())
    ordered_models = [m for m in ordered_models if m in prediction_tables]

    subset, cell_ids = core._load_umap_subset(config)
    available = set(cell_ids)
    for m in ordered_models:
        available &= set(prediction_tables[m].index.astype(str))
    if len(available) < len(cell_ids):
        kept = [c for c in cell_ids if c in available]
        subset = subset[kept].copy()
        cell_ids = kept
    truth_labels = core._resolve_ground_truth_labels(prediction_tables, cell_ids, subset, config)
    ordered_labels, shared_palette = core._build_shared_label_palette(truth_labels)
    target_label_set = set(ordered_labels)
    plot_palette = shared_palette.copy()
    plot_palette[core.OFF_TARGET_PLOT_LABEL] = core.OFF_TARGET_PLOT_COLOR
    if "X_umap" not in subset.obsm:
        raise ValueError("Supplementary Figure 2 requires cached X_umap coordinates.")
    umap_coords = subset.obsm["X_umap"]

    plot_frame = pd.DataFrame({
        "cell_id": cell_ids,
        "umap_1": umap_coords[:, 0],
        "umap_2": umap_coords[:, 1],
        "Ground Truth": truth_labels.to_numpy(),
    })
    for m in ordered_models:
        preds = core._get_series_for_cells(prediction_tables[m], cell_ids, "pred_label", m)
        plot_labels = preds.where(preds.isin(target_label_set), other=core.OFF_TARGET_PLOT_LABEL)
        plot_frame[core.display_name(m)] = plot_labels.to_numpy()

    present = set(ordered_models)
    canonical = [m for m in core.MODEL_DISPLAY_ORDER if m not in ("HECTOR", "POPV")]
    method_models = [m for m in canonical if m in present]
    method_models += [m for m in ordered_models
                      if m not in core.MODEL_DISPLAY_ORDER and m not in ("HECTOR", "POPV")]
    LEGEND = "__legend__"
    right_cells = [core.display_name(h) for h in ("HECTOR", "POPV") if h in present]
    right_cells += [core.display_name(m) for m in method_models]
    ncols_right = 4
    nrows_right = max(1, math.ceil(len(right_cells) / ncols_right))
    right_cells += [None] * (nrows_right * ncols_right - len(right_cells))

    xv = plot_frame["umap_1"].to_numpy(); yv = plot_frame["umap_2"].to_numpy()
    xpad = max(0.5, 0.03 * (float(xv.max()) - float(xv.min()) or 1.0))
    ypad = max(0.5, 0.03 * (float(yv.max()) - float(yv.min()) or 1.0))
    xlim = (float(xv.min()) - xpad, float(xv.max()) + xpad)
    ylim = (float(yv.min()) - ypad, float(yv.max()) + ypad)
    half_span = 0.55 * max(float(xv.max() - xv.min()), float(yv.max() - yv.min()))
    x_center = float(xv.max() + xv.min()) / 2
    y_center = float(yv.max() + yv.min()) / 2
    xlim = (x_center - half_span, x_center + half_span)
    ylim = (y_center - half_span, y_center + half_span)
    psize = PANEL_A_DOT_AREA

    left_ratio = 1.9
    outer = spec.subgridspec(1, 2, width_ratios=[left_ratio, ncols_right], wspace=0.22)
    left_gs = outer[0, 0].subgridspec(
        2, 1, hspace=0.02, height_ratios=[0.72, 1.28])
    right_gs = outer[0, 1].subgridspec(
        nrows_right + 1, ncols_right,
        height_ratios=[1] * nrows_right + [0.65], hspace=0.22, wspace=0.18)

    ax_truth = fig.add_subplot(left_gs[0, 0])
    core._draw_umap_panel(ax_truth, plot_frame, "Ground Truth", plot_palette, psize,
                          xlim, ylim, title="Ground Truth", alpha=UMAP_OPAQUE)

    ax_truth.set_anchor("E")

    ax_macro = fig.add_subplot(left_gs[1, 0])
    heatmap_df = core.build_macro_heatmap_frame(metrics, ordered_models)
    cmap, mode = macro_cmap("sequential_teal")
    ax_truth.title.set_fontsize(SUPPLEMENT_BODY_FONT_SIZE)
    norm = macro_norm(heatmap_df, mode)
    draw_macro_matrix(
        ax_macro,
        heatmap_df,
        cmap,
        norm,
        value_fmt=".3f",
        label_fontsize=SUPPLEMENT_BODY_FONT_SIZE,
        value_fontsize=SUPPLEMENT_BODY_FONT_SIZE,
        x_label_rotation=30,
    )

    ax_macro.set_aspect("equal")

    ax_macro.set_anchor("NE")
    from matplotlib.cm import ScalarMappable
    cax = ax_macro.inset_axes([1.04, 0.08, 0.04, 0.84])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    if cbar.solids is not None:
        cbar.solids.set_clip_on(False)
    cbar.set_label(
        "Benchmark score",
        fontsize=SUPPLEMENT_BODY_FONT_SIZE,
        labelpad=0,
    )
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.set_label_coords(1.45, 0.5)
    cbar.ax.tick_params(axis="y", pad=1)
    cbar.set_ticks([float(norm.vmin), float(norm.vmax)])
    cbar.set_ticklabels([f"{norm.vmin:.2f}", f"{norm.vmax:.2f}"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=SUPPLEMENT_BODY_FONT_SIZE)

    legend_ax = None
    for idx, cell in enumerate(right_cells):
        r, c = divmod(idx, ncols_right)
        ax = fig.add_subplot(right_gs[r, c])
        if cell == LEGEND:
            legend_ax = ax; ax.axis("off"); continue
        if cell is None or cell not in plot_frame.columns:
            ax.axis("off"); continue
        core._draw_umap_panel(ax, plot_frame, cell, plot_palette, psize, xlim, ylim,
                              alpha=UMAP_OPAQUE)

        # Enlarge each square about its own center.
        map_position = ax.get_position()
        map_scale = 1.15
        ax.set_position([
            map_position.x0 - map_position.width * (map_scale - 1) / 2,
            map_position.y0 - map_position.height * (map_scale - 1) / 2,
            map_position.width * map_scale,
            map_position.height * map_scale,
        ])
        ax.title.set_fontsize(SUPPLEMENT_BODY_FONT_SIZE)
    legend_columns = 3
    label_wrap = 100
    handles = [Line2D([0], [0], marker="o", linestyle="", color="w",
                      markerfacecolor=shared_palette[l], markeredgecolor=shared_palette[l],
                      markersize=5, label=fill(l, width=label_wrap)) for l in ordered_labels]
    panel_cols = [c for c in plot_frame.columns if c not in ("cell_id", "umap_1", "umap_2", "Ground Truth")]
    if any((plot_frame[c] == core.OFF_TARGET_PLOT_LABEL).any() for c in panel_cols):
        handles.append(Line2D([0], [0], marker="o", linestyle="", color="w",
                              markerfacecolor=core.OFF_TARGET_PLOT_COLOR,
                              markeredgecolor=core.OFF_TARGET_PLOT_COLOR, markersize=5,
                              label=fill("Unknown / other", width=label_wrap)))
    # Matplotlib fills multi-column legends by column; force macrophage to start
    # the right column so a 15-entry legend splits 7/8.
    if legend_columns == 2:
        macrophage_handle = next(
            (handle for handle in handles if handle.get_label() == "macrophage"),
            None,
        )
        if macrophage_handle is not None:
            handles.remove(macrophage_handle)
            handles.insert(7, Line2D([], [], linestyle="", alpha=0.0, label=""))
            handles.insert(8, macrophage_handle)
    if legend_ax is None:
        legend_ax = fig.add_subplot(right_gs[nrows_right, :]); legend_ax.axis("off")
    leg = legend_ax.legend(handles=handles, loc="center", frameon=False, title="Cell Types",
                           ncol=legend_columns, fontsize=SUPPLEMENT_BODY_FONT_SIZE,
                           title_fontsize=SUPPLEMENT_BODY_FONT_SIZE, handletextpad=0.35,
                           columnspacing=0.7, labelspacing=0.15)
    leg._legend_box.align = "left"
    return ax_truth, ax_macro


# -----------------------------------------------------------------------------
# Main panels A and B. Panel A is one representative comparison; panel B carries
# the six-dataset generalization result. The complete kidney method grid and
# metric matrix are in Supplementary Figure 2.
# -----------------------------------------------------------------------------

def draw_main_panel_a(fig, spec, config):
    """Draw ground truth, HECTOR and PopV on one shared human-kidney UMAP."""
    from matplotlib.lines import Line2D

    from benchmark_modules import benchmark_core as core

    results_dir = Path(config["results_dir"])
    metrics = core.load_combined_metrics(results_dir)
    prediction_tables = core._load_prediction_tables(
        results_dir, metrics["model_name"].tolist())

    subset, cell_ids = core._load_umap_subset(config)
    required_models = [model for model in ("HECTOR", "POPV")
                       if model in prediction_tables]
    available_cells = set(cell_ids)
    for model in required_models:
        available_cells &= set(prediction_tables[model].index.astype(str))
    if len(available_cells) < len(cell_ids):
        cell_ids = [cell_id for cell_id in cell_ids if cell_id in available_cells]
        subset = subset[cell_ids].copy()

    truth_labels = core._resolve_ground_truth_labels(
        prediction_tables, cell_ids, subset, config)
    ordered_labels, shared_palette = core._build_shared_label_palette(truth_labels)
    target_labels = set(ordered_labels)
    plot_palette = shared_palette.copy()
    plot_palette[core.OFF_TARGET_PLOT_LABEL] = core.OFF_TARGET_PLOT_COLOR
    if "X_umap" not in subset.obsm:
        raise ValueError("Main Figure 2 requires the frozen input UMAP; drawing must not recompute it.")
    umap_coordinates = core._get_umap_coordinates(subset)

    plot_frame = pd.DataFrame({
        "cell_id": cell_ids,
        "umap_1": umap_coordinates[:, 0],
        "umap_2": umap_coordinates[:, 1],
        "Ground Truth": truth_labels.to_numpy(),
    })
    for model in required_models:
        predictions = core._get_series_for_cells(
            prediction_tables[model], cell_ids, "pred_label", model)
        plot_labels = predictions.where(
            predictions.isin(target_labels), other=core.OFF_TARGET_PLOT_LABEL)
        plot_frame[core.display_name(model)] = plot_labels.to_numpy()

    x_values = plot_frame["umap_1"].to_numpy()
    y_values = plot_frame["umap_2"].to_numpy()
    x_padding = max(0.5, 0.03 * (float(x_values.max()) - float(x_values.min()) or 1.0))
    y_padding = max(0.5, 0.03 * (float(y_values.max()) - float(y_values.min()) or 1.0))
    x_limits = (float(x_values.min()) - x_padding, float(x_values.max()) + x_padding)
    y_limits = (float(y_values.min()) - y_padding, float(y_values.max()) + y_padding)

    # Identical square windows keep all three views at the same coordinate scale.
    half_span = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0]) / 2
    x_center = sum(x_limits) / 2
    y_center = sum(y_limits) / 2
    x_limits = (x_center - half_span, x_center + half_span)
    y_limits = (y_center - half_span, y_center + half_span)

    columns = ["Ground Truth", "HECTOR", "POPV"]
    panel_grid = spec.subgridspec(
        1, 4, width_ratios=[1.0, 1.0, 1.0, 1.50], wspace=0.10)
    axes = []
    for column_index, column in enumerate(columns):
        ax = fig.add_subplot(panel_grid[0, column_index])
        core._draw_umap_panel(
            ax, plot_frame, column, plot_palette, 1.25,
            x_limits, y_limits, title=column, alpha=UMAP_OPAQUE)
        ax.title.set_fontsize(MAIN_BODY_PT)
        axes.append(ax)

    legend_ax = fig.add_subplot(panel_grid[0, 3])
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="w",
               markerfacecolor=shared_palette[label],
               markeredgecolor=shared_palette[label], markersize=4.7,
               label=label)
        for label in ordered_labels
    ]
    panel_columns = [column for column in columns if column != "Ground Truth"]
    if any((plot_frame[column] == core.OFF_TARGET_PLOT_LABEL).any()
           for column in panel_columns):
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", color="w",
                   markerfacecolor=core.OFF_TARGET_PLOT_COLOR,
                   markeredgecolor=core.OFF_TARGET_PLOT_COLOR, markersize=4.7,
                   label="Unknown / other"))
    legend = legend_ax.legend(
        handles=handles, loc="center left", frameon=False, title="Cell types",
        ncol=2, fontsize=MAIN_SECONDARY_PT, title_fontsize=MAIN_BODY_PT, handletextpad=0.30,
        handlelength=0.8, columnspacing=0.8, labelspacing=0.35, borderaxespad=0.0)
    legend._legend_box.align = "left"
    return axes


def draw_cross_dataset_panel(fig, spec):
    """Compare HECTOR with PopV across the six independent benchmark datasets."""
    from matplotlib.lines import Line2D

    from benchmark_modules import benchmark_core as core
    from benchmark_modules import dataset_configs as dsc

    dataset_order = [
        "kidney", "lung", "skin",
        "mouse_kidney", "mouse_hippocampus", "mouse_aorta",
    ]
    dataset_labels = [
        "human\nkidney", "human BAL", "human\nskin",
        "mouse\nkidney", "mouse\nhippocampus", "mouse\naorta",
    ]
    metric_frames = {
        dataset_key: core.load_combined_metrics(dsc.CACHE_ROOT / dataset_key)
        for dataset_key in dataset_order
    }

    # A classifier keeps its identity even if one dataset has a missing score.
    classifier_styles = {
        "KNN_SCVI": ("kNN (scVI)", "#0072B2"),
        "KNN_SCANORAMA": ("kNN (Scanorama)", "#E69F00"),
        "KNN_BBKNN": ("kNN (BBKNN)", "#009E73"),
        "KNN_HARMONY": ("kNN (Harmony)", "#7B3294"),
        "Support_Vector": ("SVM", "#8C564B"),
        "XGboost": ("XGBoost", "#CC79A7"),
        "ONCLASS": ("OnClass", "#8C9400"),
        "SCANVI_POPV": ("scANVI", "#56B4E9"),
        "CELLTYPIST": ("CellTypist", "#005A55"),
    }
    classifier_offsets = dict(zip(
        classifier_styles, np.linspace(-0.16, 0.16, len(classifier_styles))))

    # An explicit spacer accommodates the two-line dataset labels above the key.
    panel_grid = spec.subgridspec(
        4, 2, height_ratios=[1.0, 0.23, 0.28, 0.12], hspace=0.04, wspace=0.16)
    axes = []
    metric_specs = [("f1_macro", "Macro F1"), ("accuracy", "Accuracy")]
    for panel_index, (metric_name, title) in enumerate(metric_specs):
        ax = fig.add_subplot(panel_grid[0, panel_index])
        for dataset_index, dataset_key in enumerate(dataset_order):
            metrics = metric_frames[dataset_key].drop_duplicates("model_name")
            values = metrics.set_index("model_name")[metric_name]
            for model, (_, classifier_color) in classifier_styles.items():
                if model not in values.index or pd.isna(values[model]):
                    continue
                ax.scatter(
                    [dataset_index + classifier_offsets[model]], [values[model]],
                    s=10, color=classifier_color, edgecolors="white",
                    linewidths=0.3, alpha=0.65, zorder=2, clip_on=False)
            if "POPV" in values.index and pd.notna(values["POPV"]):
                ax.scatter(
                    [dataset_index], [values["POPV"]], s=20,
                    facecolors="none", edgecolors="#3C4043", linewidths=0.8,
                    marker="D", zorder=3, clip_on=False)
            if "HECTOR" in values.index and pd.notna(values["HECTOR"]):
                ax.scatter(
                    [dataset_index], [values["HECTOR"]], s=36, color=HECTOR_ACCENT,
                    marker="o", edgecolors="white", linewidths=0.6,
                    zorder=4, clip_on=False)

        ax.set_xlim(-0.45, len(dataset_order) - 0.55)
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(range(len(dataset_order)))
        ax.set_xticklabels(dataset_labels, fontsize=MAIN_SECONDARY_PT)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.set_title(title, fontsize=MAIN_BODY_PT, pad=17)
        ax.grid(axis="y", color="#E8EAED", linewidth=0.45, zorder=0)
        ax.axvline(2.5, color="#DADCE0", linewidth=0.7, zorder=1)
        ax.tick_params(axis="both", direction="out", length=2.5, width=0.6,
                       labelsize=MAIN_SECONDARY_PT)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.text(1.0, 1.035, "Human", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=MAIN_SECONDARY_PT, fontweight="bold")
        ax.text(4.0, 1.035, "Mouse", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=MAIN_SECONDARY_PT, fontweight="bold")
        axes.append(ax)

    axes[0].set_ylabel("benchmark score", fontsize=MAIN_BODY_PT)
    axes[1].tick_params(axis="y", labelleft=False)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5,
               markerfacecolor=HECTOR_ACCENT, markeredgecolor="white",
               label="HECTOR"),
        Line2D([], [], marker="D", linestyle="", markersize=5,
               markerfacecolor="none", markeredgecolor="#3C4043",
               label="PopV consensus"),
    ]
    for classifier_label, classifier_color in classifier_styles.values():
        legend_handles.append(
            Line2D([], [], marker="o", linestyle="", markersize=5,
                   markerfacecolor=classifier_color, markeredgecolor="white",
                   markeredgewidth=0.3, alpha=0.65, label=classifier_label))
    legend_ax = fig.add_subplot(panel_grid[2, :])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=legend_handles, loc="center", frameon=False, fontsize=MAIN_SECONDARY_PT,
        ncol=len(legend_handles), handlelength=0.8, handletextpad=0.3,
        columnspacing=0.65, borderaxespad=0.0)
    return axes


# -----------------------------------------------------------------------------
# Master assembly
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Panel D: ontology flow. Implemented in benchmark_modules/panel_d_flow.py.
# -----------------------------------------------------------------------------

_PANEL_D_CACHE = {}
_PANEL_CD_PALETTE = {}


def _panel_d_data(dataset_key):
    if dataset_key not in _PANEL_D_CACHE:
        _PANEL_D_CACHE[dataset_key] = panel_d_flow.load_panel_data(
            cache_dir(dataset_key) / "hector_predictions_HECTOR.csv",
            _FIG2_ROOT / "input" / "cl.obo",
        )
    return _PANEL_D_CACHE[dataset_key]


def _panel_cd_palette(dataset_key):
    """One name -> colour map shared by panel C's maps and panel D's bars, so a
    cell type keeps one colour wherever it appears. Rare terms pooled below the
    display threshold fall back to grey ("not shown individually", not
    "unclassified") in both panels.
    """
    if dataset_key in _PANEL_CD_PALETTE:
        return _PANEL_CD_PALETTE[dataset_key]
    d = cache_dir(dataset_key)
    frame = pd.read_csv(d / "deepdive_umap.csv")
    shared = set(json.loads((d / "shared_types.json").read_text()))
    gt_col = _deepdive_gt_column(frame)
    extra_gt = sorted(set(frame[gt_col].astype(str)) - shared - {"other"})
    palette = _full_gt_palette(dataset_key, extra_gt)

    _, _, branch_to_right, _ = _panel_d_data(dataset_key)
    pred_terms = sorted({t for t in branch_to_right.index.get_level_values(1)
                         if not t.startswith("Other ")} - set(palette))
    used = set(palette.values())
    avail = [c for c in _color_pool(extended=True) if c not in used]
    for i, term in enumerate(pred_terms):
        if i < len(avail):
            palette[term] = avail[i]
        else:  # pool exhausted: deterministic HSV fallback
            import colorsys
            r, g, b = colorsys.hsv_to_rgb((0.211 * (i + 1)) % 1.0, 0.58, 0.86)
            palette[term] = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    _PANEL_CD_PALETTE[dataset_key] = palette
    return palette


def _draw_panel_d_flow(fig, spec, config, numbering):
    dataset_key = config["dataset_key"]
    palette = _panel_cd_palette(dataset_key)
    return panel_d_flow.draw_panel_d(fig, spec, *_panel_d_data(dataset_key),
                                     numbering=numbering,
                                     label_fontsize=MAIN_DENSE_PT,
                                     header_fontsize=MAIN_BODY_PT, gutter=3.0,
                                     wrap=70,
                                     truth_colors=palette, pred_colors=palette)


def assemble_figure2(config, out_stem="figure2_main",
                     figsize=(MAIN_WIDTH_MM / 25.4, MAIN_HEIGHT_MM / 25.4),
                     height_ratios=(104, 230), bottom_width_ratios=(1.45, 2.5)):
    import matplotlib.pyplot as plt

    dataset_key = config["dataset_key"]
    fig = plt.figure(figsize=figsize)
    master = fig.add_gridspec(
        2, 1, height_ratios=list(height_ratios), hspace=28 / 334,
        left=0.045, right=0.970, top=1 - 8 / 360, bottom=4 / 360,
    )

    # ----- top: representative kidney result, then six-dataset generalization -----
    top = master[0, 0].subgridspec(
        2, 1, height_ratios=[44, 50], hspace=20 / 94)
    panel_a_axes = draw_main_panel_a(fig, top[0, 0], config)
    b_position = top[1, 0].get_position(fig)
    b_band = fig.add_gridspec(
        1, 3, width_ratios=[0.08, 0.84, 0.08], wspace=0,
        left=b_position.x0, right=b_position.x1,
        top=b_position.y1 - 4 / MAIN_HEIGHT_MM,
        bottom=b_position.y0 - 4 / MAIN_HEIGHT_MM,
    )
    panel_b_axes = draw_cross_dataset_panel(fig, b_band[0, 1])

    # ----- bottom: [numbered UMAPs | ground-truth -> prediction ontology flow] -----
    numbering = panel_c_blocks(dataset_key)["numbering"]
    bottom = master[1, 0].subgridspec(1, 2, width_ratios=list(bottom_width_ratios), wspace=0.10)
    c_position = bottom[0, 0].get_position(fig)
    c_band = fig.add_gridspec(1, 1, left=c_position.x0,
                             right=c_position.x1 + 8 / MAIN_WIDTH_MM,
                             top=c_position.y1, bottom=c_position.y1 - 210 / MAIN_HEIGHT_MM)
    ax_umaps = draw_deepdive_umaps(
        fig, c_band[0, 0], dataset_key, numbering,
        badge_fontsize=MAIN_DENSE_PT, text_size=MAIN_BODY_PT, row_gap=0.14)
    d_position = bottom[0, 1].get_position(fig)
    d_grid = fig.add_gridspec(1, 1, left=d_position.x0, right=d_position.x1,
                             bottom=d_position.y0, top=d_position.y1 + 18 / MAIN_HEIGHT_MM)
    ax_flow, _ax_flow_legend, _ax_flow_summary = _draw_panel_d_flow(
        fig, d_grid[0, 0], config, numbering)

    # Panel letters: A/B/C share one left x (a clean left rail); D sits at its own left edge.
    fig.canvas.draw()
    abc_x = min(panel_a_axes[0].get_position().x0, panel_b_axes[0].get_position().x0,
                ax_umaps[0].get_position().x0) - 0.012
    bottom_letter_y = ax_umaps[0].get_position().y1
    placements = [
        ("a", abc_x, panel_a_axes[0].get_position().y1),
        ("b", abc_x, panel_b_axes[0].get_position().y1),
        ("c", abc_x, bottom_letter_y),
        ("d", ax_flow.get_position().x0 - 0.008, bottom_letter_y),
    ]
    for letter, x, y in placements:
        fig.text(max(0.004, x), min(0.992, y + 0.006), letter,
                 fontsize=MAIN_PANEL_PT, fontweight="bold", va="bottom", ha="right")

    # Full-page exports from the same live figure; no tight crop or PDF resizing.
    export_dpi = 600 * 180 / MAIN_WIDTH_MM
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(f"{out_stem}.pdf", bbox_inches=None, dpi=export_dpi, facecolor="white")
        fig.savefig(f"{out_stem}.png", bbox_inches=None, dpi=export_dpi, facecolor="white",
                    pil_kwargs={"dpi": (600, 600)})
    plt.close(fig)
    return f"{out_stem}.png"


# -----------------------------------------------------------------------------
# Supplements
# -----------------------------------------------------------------------------

def _spread_badges(points, min_sep, max_shift, iters=120):
    """Nudge overlapping badge positions apart, capped so a badge cannot leave its own
    cluster.

    Badges are placed at each type's median, which is correct but collides where several
    types share a region -- unresolved, `32` and `3` render as `323`, which is worse than
    no number at all. Displacement is capped at `max_shift` from the true median: a badge
    that still overlaps after the cap stays overlapping rather than drifting onto a
    neighbouring cluster and asserting something false about where the cells are.
    """
    pts = np.asarray(points, dtype=float).copy()
    if len(pts) < 2:
        return pts
    home = pts.copy()
    for _ in range(iters):
        delta = pts[:, None, :] - pts[None, :, :]
        dist = np.hypot(delta[..., 0], delta[..., 1])
        np.fill_diagonal(dist, np.inf)
        if dist.min() >= min_sep:
            break
        # Clip before multiplying: the diagonal is +inf, and inf * 0 is NaN.
        overlap = np.clip(min_sep - dist, 0.0, None)
        if not overlap.any():
            break
        safe = np.where(dist > 0, dist, 1.0)
        pts = pts + (delta / safe[..., None] * (overlap / 2)[..., None]).sum(axis=1)
        off = pts - home
        mag = np.hypot(off[:, 0], off[:, 1])
        over = mag > max_shift
        if over.any():
            pts[over] = home[over] + off[over] / mag[over, None] * max_shift
    return pts


def panel_c_blocks(dataset_key):
    """The three key blocks of panel C and the number each cell type carries.

    One function decides the numbering, so panel C's badge on a cluster and panel D's
    number beside that cell type's label can never drift apart.
    Numbering runs continuously through the blocks in display order, and a cell type keeps
    one number wherever it appears.
    """
    d = CACHE_ROOT / dataset_key
    df = pd.read_csv(d / "deepdive_umap.csv")
    palette = _panel_cd_palette(dataset_key)
    keyed = set(palette) - {"other"}
    shared = set(json.loads((d / "shared_types.json").read_text()))
    gt_types = set(df[_deepdive_gt_column(df)].astype(str))

    shared_types = sorted(keyed & shared)
    gt_only = sorted((keyed & gt_types) - shared)
    pred_only = sorted(keyed - gt_types - shared)
    numbering = {}
    for terms in (shared_types, gt_only, pred_only):
        for t in terms:
            numbering[t] = len(numbering) + 1
    return {"numbering": numbering, "shared_types": shared_types,
            "gt_only": gt_only, "pred_only": pred_only, "keyed": keyed, "palette": palette}


# Six benchmark datasets, mirroring _DATASETS in dataset_configs.py. Duplicated
# rather than imported to avoid a circular import through run_figure2.py.
_COVERAGE_PANEL_DATASETS = [
    ("kidney",            "human_kidney_normal_cells.csv", "human kidney"),
    ("lung",              "Human_BAL.csv",                 "bronchoalveolar lavage"),
    ("skin",              "HumanSkin_normal_cells.csv",    "human skin"),
    ("mouse_kidney",      "mouse_kidney_normal_cells.csv", "mouse kidney"),
    ("mouse_aorta",       "mouse_VSMC.csv",                "mouse aorta"),
    ("mouse_hippocampus", "mouse_hippocampus.csv",         "mouse hippocampus"),
]


def _coverage_restricted_macro_f1() -> pd.DataFrame:
    """HECTOR's macro F1 advantage over the PopV consensus, on all six benchmark
    datasets, computed twice: once over every cell, once restricted to cells
    whose ground-truth class the PopV consensus can also return (its nine
    methods' reference atlases cannot emit every class -- see the harmonization
    tables' ``sources`` column). Recomputed from the prediction cache on every
    call, per project convention: never hardcode a figure-2 number.
    """

    def macro_f1(frame: pd.DataFrame, ids: pd.Index) -> float:
        y_true = frame.loc[ids, "true_label"]
        y_pred = frame.loc[ids, "pred_label"]
        # Score only real classes: an off-target guess (e.g. "Unknown") should
        # count against its true class, not be averaged in as a class of its own.
        classes = sorted(y_true.unique())
        return f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)

    rows = []
    for key, harmonization_file, label in _COVERAGE_PANEL_DATASETS:
        harmonization = pd.read_csv(_FIG2_ROOT / "input" / "harmonization" / harmonization_file)
        sources = harmonization["sources"].fillna("").str.split("|")
        target = harmonization["target_label"].fillna("")
        popv_reachable = set(target[sources.apply(lambda s: "popv" in s) & ~target.isin(["Unknown", ""])])

        cache_dir = CACHE_ROOT / key
        hector = pd.read_csv(cache_dir / "hector_predictions_HECTOR.csv").set_index("cell_id")
        popv = pd.read_csv(cache_dir / "popv_predictions_POPV.csv").set_index("cell_id")
        shared_ids = hector.index.intersection(popv.index)
        restricted_ids = shared_ids[hector.loc[shared_ids, "true_label"].isin(popv_reachable)]

        rows.append({
            "dataset": label,
            "full": macro_f1(hector, shared_ids) - macro_f1(popv, shared_ids),
            "restricted": macro_f1(hector, restricted_ids) - macro_f1(popv, restricted_ids),
        })
    return pd.DataFrame(rows)


def _draw_coverage_restriction_panel(ax) -> None:
    """Panel e: does HECTOR's macro F1 advantage survive removing every cell
    whose ground-truth class the PopV consensus cannot reach at all? Two dots
    per dataset (all cells; shared-vocabulary cells only) joined by a line,
    against a zero reference line. Font sizes reuse this file's own supplement
    tiers so the new panel matches a-d rather than introducing a fifth size.
    """
    from matplotlib.lines import Line2D

    df = _coverage_restricted_macro_f1().sort_values("restricted").reset_index(drop=True)
    y = np.arange(len(df))
    gray = "#707070"  # darker than the file's usual #9A9A9A: clears protanopia contrast
    small = 6.5

    for yi, full, restricted in zip(y, df["full"], df["restricted"]):
        ax.plot([full, restricted], [yi, yi], color=gray, linewidth=1.0,
                zorder=1, solid_capstyle="round")
    ax.scatter(df["full"], y, s=26, facecolors="white", edgecolors=gray,
               linewidths=1.1, zorder=2)
    ax.scatter(df["restricted"], y, s=26, facecolors=HECTOR_ACCENT,
               edgecolors=HECTOR_ACCENT, linewidths=0, zorder=3)
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle=(0, (4, 2)), zorder=0)

    label_offset = 0.15
    for yi, full, restricted in zip(y, df["full"], df["restricted"]):
        ax.text(full, yi + label_offset, f"{full:+.3f}", color=gray, fontsize=small,
                ha="center", va="bottom")
        ax.text(restricted, yi - label_offset, f"{restricted:+.3f}", color=HECTOR_ACCENT,
                fontsize=small, ha="center", va="top", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(df["dataset"], fontsize=SUPPLEMENT_BODY_FONT_SIZE)
    ax.set_xlabel("Macro F1 advantage, HECTOR minus PopV consensus",
                  fontsize=SUPPLEMENT_BODY_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=SUPPLEMENT_BODY_FONT_SIZE)
    span = pd.concat([df["full"], df["restricted"]])
    pad = 0.12 * (span.max() - span.min())
    ax.set_xlim(span.min() - pad, span.max() + pad)
    ax.set_ylim(-0.7, len(df) - 0.3)
    ax.set_title("Coverage-restricted comparison, six datasets",
                 fontsize=SUPPLEMENT_PANEL_TITLE_SIZE)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#E6E2DB", linewidth=0.6, zorder=-1)
    ax.set_axisbelow(True)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor=gray, markeredgewidth=1.1, markersize=5, label="all cells"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=HECTOR_ACCENT,
               markeredgecolor=HECTOR_ACCENT, markersize=5, label="shared-vocabulary cells only"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=small,
              handletextpad=0.4, borderaxespad=0.3)


def plot_supplement_detail_combined(config, out="supplementary_figure_2_human_kidney_detail.pdf"):
    """Complete human-kidney comparison demoted from the main figure.

    Panels a and b retain the full eleven-method UMAP grid and six-metric score matrix;
    panels c and d retain inference timing and per-class F1. Together they provide the
    complete method audit behind the selective comparison in main Figure 2a.
    """
    import matplotlib.pyplot as plt
    from benchmark_modules import benchmark_core as core

    results_dir = Path(config["results_dir"])
    metrics = core.load_combined_metrics(results_dir)
    ordered = core.figure1_model_order(metrics)
    per_class = core.load_per_class_artifacts(results_dir, ordered)
    class_order = core.per_class_column_order(per_class)
    pc_frame = core.build_per_class_heatmap_frame(per_class, ordered, class_order)

    fig = plt.figure(figsize=(270 / 25.4, 360 / 25.4))
    master = fig.add_gridspec(
        2, 1, height_ratios=[170, 55], hspace=0.20,
        top=0.96, bottom=0.28, left=0.12, right=0.95)

    ax_truth, ax_macro = draw_benchmark_region(fig, master[0, 0], config)

    bottom = master[1, 0].subgridspec(
        1, 3, width_ratios=[0.48, 0.52, 2.0], wspace=0.30)
    ax_time = fig.add_subplot(bottom[0, 1])
    core._draw_timing_axis(ax_time, metrics, ordered)

    truth_position = ax_truth.get_position()
    time_position = ax_time.get_position()
    left_stack_center = truth_position.x0 + truth_position.width / 2
    timing_width = time_position.width * 1.25
    ax_time.set_position([
        left_stack_center - timing_width / 2,
        time_position.y0,
        timing_width,
        time_position.height,
    ])
    ax_time.tick_params(axis="both", labelsize=SUPPLEMENT_BODY_FONT_SIZE)
    ax_time.xaxis.label.set_fontsize(SUPPLEMENT_BODY_FONT_SIZE)
    ax_time.set_title("")
    for annotation in ax_time.texts:
        annotation.set_fontsize(SUPPLEMENT_BODY_FONT_SIZE)

    ax_f1 = fig.add_subplot(bottom[0, 2])
    core.draw_heatmap(ax_f1, pc_frame, core._per_class_heatmap_cmap(), 0.0, 1.0, ".3f", "Per-class F1")
    ax_f1.tick_params(axis="both", labelsize=SUPPLEMENT_BODY_FONT_SIZE)
    ax_f1.set_title("Per-class F1", fontsize=SUPPLEMENT_PANEL_TITLE_SIZE)
    for annotation in ax_f1.texts:
        annotation.set_fontsize(6.5)
    f1_colorbar = ax_f1.images[0].colorbar
    f1_colorbar.set_label("Per-class F1", fontsize=SUPPLEMENT_BODY_FONT_SIZE)
    f1_colorbar.ax.tick_params(labelsize=SUPPLEMENT_BODY_FONT_SIZE)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _tight(ax):
        return ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())

    cd_bottom = min(_tight(ax_f1).y0, _tight(ax_time).y0)

    # Panel e's top is anchored to c/d's real rendered bottom, since their
    # rotated labels extend below the master gridspec's nominal row edge.
    e_left = 0.19
    e_bottom = 0.03
    e_top = min(cd_bottom - 0.018, 0.235)
    ax_coverage = fig.add_axes([e_left, e_bottom, 0.95 - e_left, max(0.10, e_top - e_bottom)])
    _draw_coverage_restriction_panel(ax_coverage)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    tight_truth = _tight(ax_truth)
    tight_macro = _tight(ax_macro)
    tight_time = _tight(ax_time)
    tight_f1 = _tight(ax_f1)
    tight_coverage = _tight(ax_coverage)

    # Panels a-c share one left rail rather than each letter's own tight bounding
    # box, which produced a diagonal staircase. Panel d keeps its own anchor.
    left_abc = max(
        0.002,
        min(tight_truth.x0, tight_macro.x0, tight_time.x0) - 0.024,
    )
    placements = [
        ("a", left_abc, tight_truth.y1 + 0.004),
        ("b", left_abc, tight_macro.y1 + 0.004),
        ("c", left_abc, tight_time.y1 + 0.006),
        ("d", max(0.002, tight_f1.x0 - 0.024), tight_time.y1 + 0.006),
        ("e", max(0.002, tight_coverage.x0 - 0.024), tight_coverage.y1 + 0.006),
    ]
    for letter, x_position, y_position in placements:
        fig.text(x_position, min(0.97, y_position), letter,
                 fontsize=16, fontweight="bold", va="bottom", ha="left")
    fig.savefig(out, bbox_inches=None, dpi=400)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches=None, dpi=400,
                pil_kwargs={"dpi": (600, 600)})
    plt.close(fig)
    return out
