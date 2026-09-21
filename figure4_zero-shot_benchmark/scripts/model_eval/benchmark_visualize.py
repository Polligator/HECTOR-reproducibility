#!/usr/bin/env python3
"""One table builder, left over from what was a plotting module.

``build_per_type_accuracy_table`` is the only thing outside this file still
calls: one row per (held-out cell type, model arm), which `scripts/run_figure4.py`
writes to ``result/data/per_held_out_type_accuracy.csv`` and Supplementary
Figure 10 draws.

It was 2,234 lines and 29 functions until 2026-08-14. The other 24 drew the
standalone figures retired in July 2026 — the efficiency plot, the clustering
metric chart, the hierarchical and neighbourhood comparisons, the UMAP grid,
the depth-conditioned outcomes, the cumulative hop chart and its explainer —
every one of them superseded by a panel in `plot/panels/` or by a supplement.
`scripts/run_figure4.py` still imported two of them and called neither.

The cut was made by reachability, and the surviving table was checked to
produce a byte-identical CSV before and after. Seed such an analysis from
module-level statements as well as from the entry point: line 33 calls
``_setup_pub_quality_style()`` at import time, and a first attempt that walked
only function bodies deleted it and broke the import.
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent          # scripts/, which carries Hierarchical_metrics
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from model_eval import clustering_evaluate, evaluate  # noqa: E402

logger = logging.getLogger(__name__)

def _setup_pub_quality_style():
    """Configure Matplotlib/Seaborn for publication-quality figures."""
    try:
        import seaborn as sns
        sns.set_theme(style="ticks", context="paper")
    except ImportError:
        pass
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif'],
        'font.size': 11,               # Base font size
        'axes.labelsize': 12,          # Axis labels slightly larger
        'axes.titlesize': 14,          # Title significantly larger
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': False,       # Remove legend box (Chartjunk)
        'axes.spines.top': False,      # Remove top spine
        'axes.spines.right': False,    # Remove right spine
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.linewidth': 2,    
        'lines.markersize': 6,
    })

_setup_pub_quality_style()

# Canonical PPR model order used by the hierarchical chart.
# Keep the lookup key aligned with benchmark outputs and the label aligned with display text.
_PPR_MODEL_SPECS = [
    ("scGPT", "scGPT (PPR)"),
    ("scCello", "scCello (PPR)"),
    ("geneformer", "Geneformer (PPR)"),
    ("scimilarity", "scimilarity (PPR)"),
    ("Hector", "Hector (PPR)"),
]
_NATIVE_LABEL = "Hector (Native)"
_TERMINAL_CL_ID_PATTERN = re.compile(r"\((CL:\d+)\)\s*$")

# ---------------------------------------------------------------------------
# Canonical model → color mapping
# ---------------------------------------------------------------------------
# Single source of truth so the same model is the same colour in every panel.
# Keys are canonical short names; resolve display labels through
# ``_canonical_model_name`` below.
MODEL_COLOR_BY_NAME: dict[str, str] = {
    "scGPT":       "#0099B4",  # Lancet teal-blue
    "scCello":     "#00468B",  # prussian blue (unchanged)
    "geneformer":  "#42B540",  # apple green (unchanged)
    "scimilarity": "#925E9F",  # purple (formerly HECTOR's color)
    "Hector":      "#FB8072",  # HECTOR (PPR) light salmon-red — accent
}

# Display labels — what should appear in legends, tick labels, annotations.
# Keys are canonical short names. Use ``model_display_name(name)`` for any
# string that might already be a display label or alias.
MODEL_DISPLAY_NAME: dict[str, str] = {
    "Hector":      "HECTOR",      # all-caps for the accent / our model
    "scGPT":       "scGPT",
    "scCello":     "scCello",
    "geneformer":  "Geneformer",  # camel-case in display
    "scimilarity": "scimilarity",
}


def model_display_name(name: str) -> str:
    """Resolve canonical or aliased model name → its display string."""
    canon = _canonical_model_name(name)
    if canon is None:
        return name
    return MODEL_DISPLAY_NAME.get(canon, canon)
# Standardise display-name → canonical-name aliases.
_MODEL_NAME_ALIASES: dict[str, str] = {
    "hector": "Hector",
    "hector (ppr)": "Hector",
    "hector (native)": "Hector",
    "hector_native_open": "Hector",
    "hector_native_closed": "Hector",
    "scgpt": "scGPT",
    "scgpt_cp": "scGPT",
    "scgpt (ppr)": "scGPT",
    "sccello": "scCello",
    "sccello-zeroshot": "scCello",
    "sccello (ppr)": "scCello",
    "geneformer": "geneformer",
    "geneformer-v2-104m": "geneformer",
    "geneformer (ppr)": "geneformer",
    "scimilarity": "scimilarity",
    "scimilarity_v1.1": "scimilarity",
    "scimilarity (ppr)": "scimilarity",
}


def _canonical_model_name(name: str) -> str | None:
    """Normalise a model label to its canonical short name, or ``None``.

    Accepts display labels with " (PPR)", " (Native)", version suffixes, or
    case variants. Returns ``None`` for unrecognised inputs (e.g. ``Random
    Baseline``), so the caller can fall back to a positional palette.
    """
    if not isinstance(name, str):
        return None
    key = name.strip().lower()
    return _MODEL_NAME_ALIASES.get(key)


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Hierarchical bar chart
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bridge scatter: embedding quality vs zero-shot transfer (Figure 4 panel C)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Efficiency plot
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------

# Keep biological conservation metrics first so the data series aligns with the
# chart background bands and section headers.
_CLUSTERING_METRIC_KEYS = [
    "NMI_cluster/label",
    "ARI_cluster/label",
    "ASW_label",
    "avg_bio",
    "ASW_batch",
    "graph_conn",
    "avg_batch",
]

_CLUSTERING_METRIC_LABELS = [
    "NMI",
    "ARI",
    "ASW_label",
    "avg_bio",
    "ASW_batch",
    "graph_conn",
    "avg_batch",
]

# Distinct colors for up to ~10 models.
_MODEL_COLORS = [
    "#ED0000",  # Carmine Red
    "#00468B",  #Prussian Blue
    "#42B540",  # Apple Green
    "#FDAF91",  # Soft Coral
    "#925E9F",  # Muted Purple
    "#0099B4",  # Deep Teal
    "#AD002A",  # Dark Crimson
    "#ADB6B6",  # Cool Grey
]

_CLUSTERING_MODEL_ORDER = ["Hector", "scGPT", "scCello", "geneformer", "scimilarity"]

_CLUSTERING_POINT_LABEL_OFFSETS = {
    "Hector": (0.05, 0.022),
    "scGPT": (0.05, -0.020),
    "scCello": (0.05, 0.016),
    "geneformer": (0.05, -0.030),
    "scimilarity": (0.05, 0.032),
}


# ---------------------------------------------------------------------------
# UMAP grid
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cumulative hop-distance line plot
# ---------------------------------------------------------------------------

_DEPTH_BIN_PREFIXES = ["Shallow", "Intermediate", "Deep"]
_OUTCOME_ORDER = ["Exact", "Near (1-2 hops)", "Far (>2 hops)"]
_OUTCOME_COLORS = {
    "Exact": "#2E8B57",
    "Near (1-2 hops)": "#E69F00",
    "Far (>2 hops)": "#C44E52",
}
_DEFAULT_DEPTH_PANEL_ORDER = [
    "Hector (PPR)",
    "scGPT (PPR)",
    "scCello (PPR)",
    "Geneformer (PPR)",
    "scimilarity (PPR)",
    "Random Baseline",
    None,
    "Hector (Native)",
    None,
]


# ---------------------------------------------------------------------------
# Case-study mosaic: per-held-out-type outcome breakdown (Figure 4 panel)
# ---------------------------------------------------------------------------

# Default held-out cell types to spotlight, chosen to span tiers of "what
# relatives exist in the bridge" (easy/medium/hard, marked below).
_DEFAULT_CASE_STUDY_TYPES = [
    # Easy (sibling/parent in bridge):
    "NKp44-positive group 3 innate lymphoid cell, human",
    "antibody secreting cell",
    "erythroid progenitor cell, mammalian",
    # Medium (informative neighbourhood, no exact relative):
    "glomerular endothelial cell",
    "PP cell",
    # Hard (no close relative in bridge):
    "medium spiny neuron",
]


def _parse_hop_distance(value) -> int | None:
    """Convert a ``fine_ontology_distance`` entry to an unsigned hop count.

    Returns ``None`` for the disconnected sentinel (``1000`` or NaN). The
    sibling-style "N_sib" string is parsed as ``N`` undirected hops; signed
    ancestor/descendant integers collapse to their absolute value.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value)
    if s.endswith("_sib"):
        try:
            return int(s[:-4])
        except ValueError:
            return None
    try:
        n = int(s)
    except ValueError:
        return None
    if n == 1000:
        return None
    return abs(n)


# ---------------------------------------------------------------------------
# Figure 4 orchestrator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Supplement: per-held-out-type prediction breakdown
# ---------------------------------------------------------------------------

def build_per_type_accuracy_table(
    prediction_tables: dict[str, dict],
    *,
    method: str = "ppr_candidate",
) -> pd.DataFrame:
    """One row per (held-out cell type, model) — accuracy + outcome fractions.

    Used as a supplementary CSV that complements the Panel G case-study
    mosaic by reporting metrics for all 17 candidate types instead of the
    6 spotlighted in the main figure.

    Columns: ``cell_type, cell_type_id, model, n_cells, exact, near_le2,
    far_gt2, mean_hop``.
    """
    rows = []
    for model_name, tables in prediction_tables.items():
        df = tables.get(method)
        if df is None or not hasattr(df, "columns"):
            continue
        if "truth_cell_type_name" not in df.columns:
            continue
        for (ct_id, ct_name), sub in df.groupby(
            ["truth_cell_type_id", "truth_cell_type_name"]
        ):
            hops = sub["fine_ontology_distance"].map(_parse_hop_distance)
            n_total = int(len(sub))
            if n_total == 0:
                continue
            # Direct-match accuracy is the quantity reported in Figure 4c.
            # Ontology distance can be zero for an alternative ontology term,
            # which is biologically close but is not an exact predicted ID.
            exact_mask = sub["flat_accuracy"].fillna(0).astype(bool)
            exact = int(exact_mask.sum())
            near = int((~exact_mask & (hops >= 1) & (hops <= 2)).sum())
            far = int(n_total - exact - near)
            valid_hops = hops.loc[~exact_mask].dropna()
            rows.append({
                "cell_type": ct_name,
                "cell_type_id": ct_id,
                "model": model_display_name(model_name),
                "method": method,
                "n_cells": n_total,
                "exact": exact,
                "near_le2": near,
                "far_gt2": far,
                "exact_frac": exact / n_total,
                "near_frac": near / n_total,
                "far_frac": far / n_total,
                "mean_hop": float(valid_hops.mean()) if len(valid_hops) else None,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["cell_type", "model"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Supplement: multi-metric bridge scatter
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Supplement: baseline_bridge vs ppr_candidate comparison
# ---------------------------------------------------------------------------

