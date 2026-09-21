"""Figure 4 supplement — accuracy by depth of the true term.

Writes ``supplementary_figure_11_robustness.{pdf,png}``.

  a  per-cell exact accuracy by depth of the true term
  b  fraction within two ontology hops by depth of the true term

Both panels restage the seven model arms in Fig. 4c, plus random guessing.

A third panel, the native-decoder comparison, was here until 2026-09-14, when
it moved into the main figure as Figure 4b. Its drawing went with it, to
``../figure4.py``.

The page keeps its full 3:4 sheet with the third panel gone. Canvas sizes in
this project are fixed whether or not every millimetre is used.

Reads ``result/plot_cache/depth_rows.csv``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..paths import RESULT_DIR, PLOT_CACHE_DIR

# Use the 180 mm standard working width on a complete 3:4 page.
# Fonts and strokes follow page width. Raster output targets the same pixel
# dimensions as the other figures at their common manuscript insertion size.
CANVAS_WIDTH_IN = 180 / 25.4
CANVAS_HEIGHT_IN = CANVAS_WIDTH_IN * 4 / 3
DRAWING_SCALE = CANVAS_WIDTH_IN / (180 / 25.4)
EXPORT_DPI = 600 / DRAWING_SCALE
TEXT_BODY = CANVAS_WIDTH_IN * 72 / 85
TEXT_SECONDARY = 0.9 * TEXT_BODY
TEXT_EMPHASIS = 1.2 * TEXT_BODY
TEXT_PANEL = 1.8 * TEXT_BODY

SERIES = [
    # (csv_model_name, display, color, is_hector)
    # Named as Figure 4 names them: native is HECTOR's own prediction and
    # embedding is the embedding alone; closed chooses among the 66 held-out
    # types and open chooses among all 1,407 terms. The four comparison models
    # stay unqualified, as they do in Figure 4.
    ("Hector (Native)",       "HECTOR native, closed", "#ED0000", True),
    ("Hector (Native, open)", "HECTOR native, open",   "#FB8072", True),
    ("Hector (PPR)",          "HECTOR embedding", "#C97B72", True),
    ("scimilarity (PPR)", "SCimilarity",      "#925E9F", False),
    ("scGPT (PPR)",      "scGPT",             "#0099B4", False),
    ("scCello (PPR)",    "scCello",           "#00468B", False),
    ("Geneformer (PPR)", "Geneformer",        "#42B540", False),
    ("Random Baseline",  "random guessing",   "#888888", False),
]
HECTOR_RED = "#ED0000"

BIN_TITLES = {
    0: "Shallow\n(depth 1–4)",
    1: "Intermediate\n(depth 5–6)",
    2: "Deep\n(depth 7–10)",
}

ROWS = [
    ("frac_exact",   "Per-cell exact accuracy", "Per-cell exact accuracy", 0.55),
    ("frac_near_le2", "Within ≤2 ontology hops",
     "Within ≤2 ontology hops (exact + near)", 1.00),
]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 1.0,
    "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 350,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def load_data() -> pd.DataFrame:
    # Every model is read from one cache, binned on the fixed ranges BIN_TITLES
    # states (1-4 / 5-5 / 6-10 here), so all models share the same partition.
    df = pd.read_csv(PLOT_CACHE_DIR / "depth_rows.csv")
    df["depth_bin_label"] = df["depth_bin_label"].str.replace("\n", " ")
    df["frac_near_le2"] = df["frac_exact"] + df["frac_near"]
    return df


def draw_bin(ax, df_bin: pd.DataFrame, metric: str, xlim_max: float,
             show_yticks: bool, title: str | None, n_cells: int | None):
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle=":", linewidth=0.5 * DRAWING_SCALE, color="#BBBBBB", alpha=0.6, zorder=0)

    hector_idx = [i for i, (_k, _d, _c, is_h) in enumerate(SERIES) if is_h]
    if hector_idx:
        ax.axhspan(min(hector_idx) - 0.45, max(hector_idx) + 0.45,
                   color=HECTOR_RED, alpha=0.07, zorder=1, clip_on=False)

    for yi, (key, disp, color, is_h) in enumerate(SERIES):
        row = df_bin[df_bin["model"] == key]
        if row.empty:
            continue
        val = float(row[metric].iloc[0])
        is_random = "Random" in key
        ax.barh(yi, val, height=0.66 if is_h else 0.6, color=color,
                edgecolor="black" if is_h else "none",
                linewidth=0.5 * DRAWING_SCALE if is_h else 0.0,
                hatch="///" if is_random else None, zorder=2, clip_on=False)
        ax.text(val + 0.012, yi, f"{val:.3f}",
                va="center", ha="left", fontsize=TEXT_SECONDARY,
                color="#222222", zorder=3)

    ax.set_ylim(len(SERIES) - 0.5, -0.5)
    ax.set_xlim(0, xlim_max)
    step = 0.1 if xlim_max <= 0.5 else 0.2
    ax.set_xticks(np.arange(0.0, xlim_max + 0.001, step))
    ax.tick_params(axis="both", width=0.8 * DRAWING_SCALE,
                   length=3.5 * DRAWING_SCALE, pad=3.5 * DRAWING_SCALE)
    ax.tick_params(axis="x", labelsize=TEXT_SECONDARY)
    ax.set_yticks(range(len(SERIES)))
    if show_yticks:
        ax.set_yticklabels([disp for _k, disp, _c, _h in SERIES],
                           fontsize=TEXT_BODY)
        for tl in ax.get_yticklabels():
            if tl.get_text().startswith("HECTOR"):
                tl.set_color(HECTOR_RED); tl.set_fontweight("bold")
    else:
        ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_linewidth(0.6 * DRAWING_SCALE)
    ax.spines["bottom"].set_linewidth(0.6 * DRAWING_SCALE)

    if title is not None:
        ax.set_title(title, fontsize=TEXT_BODY, fontweight="bold", pad=11 * DRAWING_SCALE)
    if n_cells is not None:
        ax.text(0.5, 1.005, f"n = {n_cells:,} cells", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=TEXT_SECONDARY,
                style="italic", color="#666666")


def render(out_dir=RESULT_DIR) -> Path:
    df = load_data()
    bins = sorted(df["depth_bin_rank"].unique())

    fig = plt.figure(figsize=(CANVAS_WIDTH_IN, CANVAS_HEIGHT_IN))
    # Explicit heading bands keep the row descriptions clear of model labels and
    # depth-bin headings. The rest of the page is white space, not a reason to
    # stretch the two charts.
    band_mm = np.array([18, 46, 12, 7, 46]) * DRAWING_SCALE
    page_height_mm = CANVAS_HEIGHT_IN * 25.4
    plot_top = 1 - 6 * DRAWING_SCALE / page_height_mm
    outer = fig.add_gridspec(
        len(band_mm), 1, height_ratios=band_mm, hspace=0,
        left=0.19, right=0.98, top=plot_top,
        bottom=plot_top - band_mm.sum() / page_height_mm,
    )

    row_a = outer[1, 0].subgridspec(1, 3, wspace=0.10)
    row_b = outer[4, 0].subgridspec(1, 3, wspace=0.10)
    row_gs = [row_a, row_b]
    for ri, (metric, row_label, xlabel, xlim_max) in enumerate(ROWS):
        for cj, brank in enumerate(bins):
            sub = df[df["depth_bin_rank"] == brank]
            n_cells = int(sub["n_total"].max())
            ax = fig.add_subplot(row_gs[ri][0, cj])
            draw_bin(ax, sub, metric=metric, xlim_max=xlim_max,
                     show_yticks=(cj == 0),
                     title=BIN_TITLES[brank] if ri == 0 else None,
                     n_cells=n_cells if ri == 0 else None)
            if ri == 1:
                ax.set_xlabel(xlabel, fontsize=TEXT_BODY)

    row_letters = {
        "a": "Per-cell exact accuracy",
        "b": "Within ≤2 ontology hops",
    }
    for letter, spec in (("a", outer[0, 0]), ("b", outer[3, 0])):
        box = spec.get_position(fig)
        fig.text(0.013, box.y1, letter, fontsize=TEXT_PANEL,
                 fontweight="bold", va="top", color="#222222")
        fig.text(0.047, box.y1 - 0.003, row_letters[letter], fontsize=TEXT_EMPHASIS,
                 fontweight="bold", va="top", color="#222222")

    out_dir = Path(out_dir)
    out_pdf = out_dir / "supplementary_figure_11_robustness.pdf"
    # Fixed page, including unused white space; no content-dependent cropping.
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(out_pdf, dpi=EXPORT_DPI, bbox_inches=None,
                    facecolor="white", transparent=False)
        fig.savefig(out_dir / "supplementary_figure_11_robustness.png",
                    dpi=EXPORT_DPI, bbox_inches=None,
                    facecolor="white", transparent=False,
                    pil_kwargs={"dpi": (600, 600)})
    plt.close(fig)
    return out_pdf
