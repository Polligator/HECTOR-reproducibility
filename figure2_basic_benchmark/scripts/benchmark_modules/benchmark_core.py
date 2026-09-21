from __future__ import annotations

"""Shared utilities for the benchmark runners.

This module stores the reusable benchmark logic:
- dataset config resolution
- label harmonization and metric calculation
- artifact saving
- publication plotting
- subprocess handoff to internal workers
"""

from copy import deepcopy
import json
import math
import os
import subprocess
from pathlib import Path
from textwrap import fill
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_benchmark")

import matplotlib as mpl
import sys
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.text import Text
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# Use TrueType embedding so Illustrator keeps PDF text editable as text.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


# -----------------------------------------------------------------------------
# Benchmark settings
# -----------------------------------------------------------------------------

REQUESTED_POPV_METHODS = [
    "KNN_SCVI",
    "KNN_SCANORAMA",
    "KNN_BBKNN",
    "KNN_HARMONY",
    "Support_Vector",
    "Random_Forest",
    "XGboost",
    "ONCLASS",
    "SCANVI_POPV",
    "CELLTYPIST",
]

# Methods that build their integration + classifier entirely at inference time,
# so they run even when absent from hub_model.metadata.methods. Any method not
# listed here needs a pretrained artifact (e.g. Random_Forest loads
# rf_classifier.joblib) and so only runs in POPV "retrain" mode.
RUNTIME_TRAINABLE_POPV_METHODS = {"KNN_SCANORAMA"}

MODEL_DISPLAY_ORDER = [
    "HECTOR",
    "POPV",
    "KNN_SCVI",
    "KNN_SCANORAMA",
    "KNN_BBKNN",
    "KNN_HARMONY",
    "Support_Vector",
    "Random_Forest",
    "XGboost",
    "ONCLASS",
    "SCANVI_POPV",
    "CELLTYPIST",
]

DISPLAY_NAME_MAP = {
    "HECTOR": "HECTOR",
    "POPV": "POPV",
    "KNN_SCVI": "KNN_SCVI",
    "KNN_SCANORAMA": "KNN_SCANORAMA",
    "KNN_BBKNN": "KNN_BBKNN",
    "KNN_HARMONY": "KNN_HARMONY",
    "Support_Vector": "SVM",
    "Random_Forest": "RandomForest",
    "XGboost": "XGBoost",
    "ONCLASS": "OnClass",
    "SCANVI_POPV": "scANVI",
    "CELLTYPIST": "CellTypist",
}

POPV_METHOD_TO_OBS_KEY = {
    "KNN_SCVI": "popv_knn_on_scvi_prediction",
    "KNN_SCANORAMA": "popv_knn_scanorama_prediction",
    "KNN_BBKNN": "popv_knn_bbknn_prediction",
    "KNN_HARMONY": "popv_knn_harmony_prediction",
    "Support_Vector": "popv_svm_prediction",
    "Random_Forest": "popv_rf_prediction",
    "XGboost": "popv_xgboost_prediction",
    "ONCLASS": "popv_onclass_prediction",
    "SCANVI_POPV": "popv_scanvi_prediction",
    "CELLTYPIST": "popv_celltypist_prediction",
    "POPV": "popv_prediction",
}


METRIC_COLUMNS = [
    "accuracy",
    "f1_macro",
    "recall_macro",
    "precision_macro",
    "roc_auc_macro",
    "pr_auc_macro",
]

MACRO_HEATMAP_LOW_COLOR = "#C46A58"
MACRO_HEATMAP_MID_COLOR = "#F2EEE7"
MACRO_HEATMAP_HIGH_COLOR = "#4F7C91"
MACRO_HEATMAP_GRID_COLOR = "#DDD7CF"
MACRO_HEATMAP_NA_EDGE_COLOR = "#BCBCBC"
MACRO_HEATMAP_TEXT_COLOR = "#1A1A1A"
PER_CLASS_HEATMAP_BEST_TEXT_COLOR = "#98715F"
PER_CLASS_HEATMAP_COLORS = [
    MACRO_HEATMAP_MID_COLOR,
    "#C9DCE4",
    "#93B2BF",
    "#5F8698",
    "#355F72",
]
OFF_TARGET_PLOT_LABEL = "__off_target_prediction__"
OFF_TARGET_PLOT_COLOR = "#D0D0D0"


# -----------------------------------------------------------------------------
# Shared data helpers
# -----------------------------------------------------------------------------

def display_name(model_name: str) -> str:
    return DISPLAY_NAME_MAP.get(model_name, model_name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def merge_label_map(overrides: dict[str, str] | None = None) -> dict[str, str]:
    label_map = {}
    if overrides:
        label_map.update(overrides)
    return label_map


# figure2_basic_benchmark/, holding input/, scripts/ and result/. Used to resolve
# harmonization-table paths given relative to the project.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_harmonization_table(path: str | Path) -> dict[str, str]:
    """Load a per-dataset label harmonization table.

    The table is a CSV with at least the columns ``raw_label`` and
    ``target_label`` (extra columns such as ``sources``/counts are ignored).
    It is the transparent, human-validatable source of the label map: every raw
    label that the ground truth, HECTOR, or the POPV model can emit is mapped to
    a target. A ``target_label`` of "Unknown" means the label has no defensible
    target -- such ground-truth cells are dropped (see resolve_dataset_config)
    and such predictions are scored as wrong.

    Returns {raw_label: target_label}. Relative paths resolve against REPO_ROOT.
    """
    table_path = Path(path)
    if not table_path.is_absolute():
        table_path = REPO_ROOT / table_path
    frame = pd.read_csv(table_path)
    missing = {"raw_label", "target_label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Harmonization table {table_path} is missing columns: {sorted(missing)}")
    return {
        str(raw).strip(): str(target).strip()
        for raw, target in zip(frame["raw_label"], frame["target_label"])
    }


def harmonize_label(value: object, label_map: dict[str, str] | None = None) -> str:
    if pd.isna(value):
        return "Unknown"
    label_map = merge_label_map(label_map)
    normalized_label_map = {key.casefold(): mapped for key, mapped in label_map.items()}
    text = str(value).strip()
    direct = label_map.get(text)
    if direct is not None:
        return direct
    
    normalized = normalized_label_map.get(text.casefold())
    if normalized is not None:
        return normalized
    
    return text


def harmonize_series(values: Iterable[object], label_map: dict[str, str] | None = None) -> pd.Series:
    return pd.Series([harmonize_label(value, label_map=label_map) for value in values], dtype="object")


def benchmark_target_labels(truth: pd.Series) -> list[str]:
    return sorted(pd.Series(truth, dtype="object").astype(str).unique().tolist())


def choose_cells(obs_names: pd.Index, sample_size: int, random_state: int) -> pd.Index:
    if sample_size <= 0 or sample_size >= len(obs_names):
        return obs_names
    rng = np.random.default_rng(random_state)
    chosen = np.sort(rng.choice(len(obs_names), size=sample_size, replace=False))
    return obs_names[chosen]


def load_or_create_cell_ids(adata, sample_size: int, random_state: int, cell_id_file: str | Path) -> list[str]:
    cell_id_file = Path(cell_id_file)
    expected_size = len(adata.obs_names) if sample_size <= 0 or sample_size >= len(adata.obs_names) else sample_size
    if cell_id_file.exists():
        existing = [line.strip() for line in cell_id_file.read_text().splitlines() if line.strip()]
        if len(existing) == expected_size and set(existing).issubset(set(map(str, adata.obs_names))):
            return existing
    chosen = choose_cells(adata.obs_names, sample_size=sample_size, random_state=random_state)
    ensure_dir(cell_id_file.parent)
    cell_id_file.write_text("\n".join(map(str, chosen)) + "\n")
    return list(map(str, chosen))


def onehot_score_frame(predictions: pd.Series, classes: list[str]) -> pd.DataFrame:
    scores = np.zeros((len(predictions), len(classes)), dtype=float)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    for row_idx, label in enumerate(predictions):
        if label in class_to_idx:
            scores[row_idx, class_to_idx[label]] = 1.0
    return pd.DataFrame(scores, index=predictions.index, columns=classes)


def _json_ready(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, pd.Index, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Series):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [_json_ready(item) for item in value.to_dict(orient="records")]
    if pd.isna(value):
        return None
    return str(value)


def _is_integer_like_label(value: object) -> bool:
    text = str(value).strip()
    return text.isdigit()


def validate_metric_inputs(
    y_true: pd.Series,
    y_pred: pd.Series,
    score_frame: pd.DataFrame,
) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(f"Truth/prediction length mismatch: {len(y_true)} != {len(y_pred)}")
    if len(score_frame) != len(y_true):
        raise ValueError(f"Score row count mismatch: {len(score_frame)} != {len(y_true)}")
    if score_frame.columns.duplicated().any():
        duplicates = sorted(pd.Index(score_frame.columns)[pd.Index(score_frame.columns).duplicated()].unique().tolist())
        raise ValueError(f"Score columns contain duplicates: {duplicates}")
    integer_like = [str(col) for col in score_frame.columns if _is_integer_like_label(col)]
    if integer_like:
        preview = integer_like[:10]
        raise ValueError(f"Score columns look unresolved/unlabeled: {preview}")
    values = score_frame.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Score matrix contains NaN or infinite values.")
    if np.nanmax(values) < np.nanmin(values):
        raise ValueError("Score matrix has invalid numeric range.")


def aggregate_score_frame(
    score_frame: pd.DataFrame,
    label_order: list[str] | None = None,
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    renamed = pd.Index([harmonize_label(col, label_map=label_map) for col in score_frame.columns])
    aggregated = score_frame.copy()
    aggregated.columns = renamed
    aggregated = aggregated.T.groupby(level=0).sum().T
    if label_order is None:
        label_order = sorted(aggregated.columns.tolist())
    for label in label_order:
        if label not in aggregated.columns:
            aggregated[label] = 0.0
    return aggregated.loc[:, label_order]


def popv_label_categories(adata) -> dict[str, Any]:
    all_labels = [str(label) for label in adata.uns.get("label_categories", [])]
    unknown_label = str(adata.uns.get("unknown_celltype_label", "")).strip() or None
    if unknown_label and all_labels and all_labels[-1] == unknown_label:
        probability_labels = all_labels[:-1]
    else:
        probability_labels = all_labels
        unknown_label = None
    return {
        "all_labels": all_labels,
        "probability_labels": probability_labels,
        "unknown_label": unknown_label,
    }


def _named_score_columns(columns: list[object]) -> bool:
    return not all(_is_integer_like_label(col) for col in columns)


def resolve_popv_raw_score_frame(
    model_name: str,
    raw_scores: object,
    query_index: pd.Index,
    adata,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(raw_scores, pd.DataFrame):
        frame = raw_scores.copy()
        raw_type = "dataframe"
    else:
        frame = pd.DataFrame(raw_scores)
        raw_type = type(raw_scores).__name__
    frame.index = pd.Index(query_index.astype(str), dtype="object")
    raw_columns = [str(col) for col in frame.columns]
    
    labels_meta = popv_label_categories(adata)
    probability_labels = labels_meta["probability_labels"]
    all_labels = labels_meta["all_labels"]
    candidate_orders: list[tuple[str, list[str]]] = []
    if model_name == "ONCLASS":
        candidate_orders.append(("label_categories_full", all_labels))
        candidate_orders.append(("label_categories_probability_only", probability_labels))
    else:
        candidate_orders.append(("label_categories_probability_only", probability_labels))
        if all_labels != probability_labels:
            candidate_orders.append(("label_categories_full", all_labels))
    
    resolved_source = "raw_columns"
    if _named_score_columns(raw_columns):
        resolved_columns = raw_columns
    else:
        resolved_columns = []
        for source_name, labels in candidate_orders:
            if len(labels) == frame.shape[1]:
                resolved_columns = labels
                resolved_source = source_name
                break
        if not resolved_columns:
            raise ValueError(
                f"{model_name} score matrix has {frame.shape[1]} unlabeled columns and no matching class order."
            )
    
    frame.columns = [str(col) for col in resolved_columns]
    return frame, {
        "raw_type": raw_type,
        "raw_shape": [int(frame.shape[0]), int(frame.shape[1])],
        "raw_columns_preview": raw_columns[:10],
        "resolved_columns_preview": list(frame.columns[:10]),
        "resolved_column_source": resolved_source,
        "n_resolved_columns": int(frame.shape[1]),
        "label_categories_probability_only": probability_labels,
        "label_categories_full": all_labels,
        "unknown_label": labels_meta["unknown_label"],
    }


def prepare_popv_score_frame(
    model_name: str,
    raw_scores: object,
    query_index: pd.Index,
    strict_pred: pd.Series,
    pred: pd.Series,
    adata,
    label_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    resolved_frame, audit = resolve_popv_raw_score_frame(model_name, raw_scores, query_index, adata)
    if len(resolved_frame) != len(strict_pred):
        raise ValueError(
            f"{model_name} score rows do not match predictions: {len(resolved_frame)} != {len(strict_pred)}"
        )
    if resolved_frame.columns.duplicated().any():
        duplicates = sorted(resolved_frame.columns[resolved_frame.columns.duplicated()].unique().tolist())
        raise ValueError(f"{model_name} raw score columns contain duplicates: {duplicates}")
    
    all_nan_columns = resolved_frame.columns[resolved_frame.isna().all(axis=0)].tolist()
    partial_nan_columns = resolved_frame.columns[resolved_frame.isna().any(axis=0) & ~resolved_frame.isna().all(axis=0)].tolist()
    if partial_nan_columns:
        raise ValueError(f"{model_name} raw score matrix contains partial NaN columns: {partial_nan_columns}")
    if all_nan_columns:
        resolved_frame = resolved_frame.fillna({column: 0.0 for column in all_nan_columns})
    
    values = resolved_frame.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{model_name} raw score matrix contains NaN or infinite values.")
    if (values < 0).any():
        raise ValueError(f"{model_name} raw score matrix contains negative values.")
    row_sum = values.sum(axis=1)
    if (row_sum <= 0).any():
        raise ValueError(f"{model_name} raw score matrix has rows with non-positive total score.")
    
    raw_argmax = resolved_frame.idxmax(axis=1).astype(str).reset_index(drop=True)
    raw_match_count = int((raw_argmax == strict_pred.astype(str).reset_index(drop=True)).sum())
    
    harmonized_columns = [harmonize_label(col, label_map=label_map) for col in resolved_frame.columns]
    duplicate_harmonized = (
        pd.Series(harmonized_columns, dtype="object").value_counts().loc[lambda s: s > 1].index.tolist()
    )
    aggregated = aggregate_score_frame(resolved_frame, label_map=label_map)
    validate_metric_inputs(pred, pred, aggregated)
    aggregated_argmax = aggregated.idxmax(axis=1).astype(str).reset_index(drop=True)
    harmonized_raw_argmax = harmonize_series(raw_argmax, label_map=label_map).reset_index(drop=True)
    harmonized_raw_match_count = int((harmonized_raw_argmax == pred.reset_index(drop=True)).sum())
    aggregated_match_count = int((aggregated_argmax == pred.reset_index(drop=True)).sum())
    aggregated_mismatch_count = int(len(pred) - aggregated_match_count)
    audit.update(
        {
            "raw_argmax_match_count": raw_match_count,
            "raw_argmax_match_fraction": float(raw_match_count / len(strict_pred)) if len(strict_pred) else None,
            "harmonized_raw_argmax_match_count": harmonized_raw_match_count,
            "harmonized_raw_argmax_match_fraction": float(harmonized_raw_match_count / len(pred)) if len(pred) else None,
            "aggregated_argmax_match_count": aggregated_match_count,
            "aggregated_argmax_match_fraction": float(aggregated_match_count / len(pred)) if len(pred) else None,
            "aggregated_argmax_mismatch_count": aggregated_mismatch_count,
            "use_aggregated_argmax_for_metrics": aggregated_mismatch_count > 0,
            "harmonized_column_duplicates": duplicate_harmonized,
            "harmonized_columns_preview": harmonized_columns[:10],
            "harmonized_score_columns": aggregated.columns.tolist(),
            "all_nan_score_columns_filled_with_zero": all_nan_columns,
            "score_row_sum_min": float(row_sum.min()) if len(row_sum) else None,
            "score_row_sum_max": float(row_sum.max()) if len(row_sum) else None,
        }
    )
    
    if raw_match_count != len(strict_pred):
        raise ValueError(
            f"{model_name} score argmax does not match the raw prediction label for {len(strict_pred) - raw_match_count} cells."
        )
    return aggregated, aggregated_argmax, audit


def _binarize_truth(y_true: pd.Series, classes: list[str]) -> np.ndarray:
    y_true_bin = label_binarize(y_true, classes=classes)
    if len(classes) == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    return y_true_bin


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    score_frame: pd.DataFrame,
    strict_y_true: pd.Series | None = None,
    strict_y_pred: pd.Series | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_metric_inputs(y_true, y_pred, score_frame)
    # Per-class outputs and publication diagnostics are defined over the benchmark
    # target classes, not every off-target label a model may emit.
    eval_classes = sorted(set(pd.Series(y_true, dtype="object").astype(str)))
    score_classes = sorted(set(eval_classes) | set(score_frame.columns))
    aligned_scores = aggregate_score_frame(score_frame, label_order=score_classes)
    y_true = pd.Series(y_true, dtype="object").reset_index(drop=True)
    y_pred = pd.Series(y_pred, dtype="object").reset_index(drop=True)
    aligned_scores = aligned_scores.reset_index(drop=True)
    
    y_true_bin = _binarize_truth(y_true, score_classes)
    valid_mask = y_true_bin.sum(axis=0) > 0
    valid_classes = [score_classes[idx] for idx, keep in enumerate(valid_mask) if keep]
    
    metrics = {
        "n_cells": int(len(y_true)),
        "n_classes": int(len(valid_classes)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", labels=eval_classes, zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", labels=eval_classes, zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", labels=eval_classes, zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", labels=eval_classes, zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", labels=eval_classes, zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", labels=eval_classes, zero_division=0)
        ),
    }
    if strict_y_true is not None and strict_y_pred is not None:
        metrics["strict_accuracy"] = float(accuracy_score(strict_y_true, strict_y_pred))
    
    if valid_mask.sum() >= 2:
        valid_true = y_true_bin[:, valid_mask]
        valid_scores = aligned_scores.loc[:, valid_classes].to_numpy()
        
        # Row-normalize so a model with many non-benchmark score columns (e.g.
        # ONCLASS) is on the same probability scale as one that already sums to 1.
        row_sums = valid_scores.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        valid_scores = valid_scores / row_sums
        
        metrics["roc_auc_macro"] = float(
            roc_auc_score(valid_true, valid_scores, average="macro", multi_class="ovr")
        )
        metrics["roc_auc_weighted"] = float(
            roc_auc_score(valid_true, valid_scores, average="weighted", multi_class="ovr")
        )
        metrics["roc_auc_micro"] = float(roc_auc_score(valid_true, valid_scores, average="micro"))
        metrics["pr_auc_macro"] = float(average_precision_score(valid_true, valid_scores, average="macro"))
        metrics["pr_auc_micro"] = float(average_precision_score(valid_true, valid_scores, average="micro"))
    else:
        metrics["roc_auc_macro"] = np.nan
        metrics["roc_auc_weighted"] = np.nan
        metrics["roc_auc_micro"] = np.nan
        metrics["pr_auc_macro"] = np.nan
        metrics["pr_auc_micro"] = np.nan

    per_class = pd.DataFrame(
        {
            "class_label": eval_classes,
            "f1": f1_score(y_true, y_pred, average=None, labels=eval_classes, zero_division=0),
            "precision": precision_score(y_true, y_pred, average=None, labels=eval_classes, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=None, labels=eval_classes, zero_division=0),
            "support": pd.Series(y_true).value_counts().reindex(eval_classes, fill_value=0).values,
        }
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=eval_classes)
    confusion_long = (
        pd.DataFrame(cm, index=eval_classes, columns=eval_classes)
        .rename_axis(index="true_label", columns="pred_label")
        .stack()
        .reset_index(name="count")
    )
    return metrics, per_class, confusion_long


def clear_score_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = metrics.copy()
    metrics["roc_auc_macro"] = np.nan
    metrics["roc_auc_weighted"] = np.nan
    metrics["roc_auc_micro"] = np.nan
    metrics["pr_auc_macro"] = np.nan
    metrics["pr_auc_micro"] = np.nan
    return metrics


# -----------------------------------------------------------------------------
# Metric and artifact helpers
# -----------------------------------------------------------------------------

def measure_device(run_fn, *, util_threshold: int = 12, mem_threshold_mb: int = 50, sample_interval: float = 0.2) -> str:
    """Run ``run_fn()`` and report which backend it actually used: "GPU" or "CPU".

    POPV methods mix torch (scVI/scANVI), TensorFlow (ONCLASS) and CPU-only
    sklearn/scanpy backends with no common device attribute, so the device is
    measured empirically: a method counts as GPU if it allocates new CUDA memory
    via torch, or if nvidia-smi GPU utilization spikes during the call (which
    catches non-torch CUDA backends such as TensorFlow or XGBoost-CUDA).
    """
    import subprocess
    import threading

    samples: list[int] = []
    stop = threading.Event()

    def _sample() -> None:
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                ).stdout.strip().splitlines()
                if out:
                    samples.append(int(out[0].split()[0]))
            except Exception:
                pass
            stop.wait(sample_interval)

    torch_mod = None
    baseline = 0
    try:
        import torch as torch_mod  # type: ignore
        if torch_mod.cuda.is_available():
            torch_mod.cuda.synchronize()
            baseline = torch_mod.cuda.memory_allocated()
            torch_mod.cuda.reset_peak_memory_stats()
        else:
            torch_mod = None
    except Exception:
        torch_mod = None

    thread = threading.Thread(target=_sample, daemon=True)
    thread.start()
    try:
        run_fn()
    finally:
        stop.set()
        thread.join(timeout=2)

    torch_used = False
    if torch_mod is not None:
        try:
            torch_mod.cuda.synchronize()
            torch_used = (torch_mod.cuda.max_memory_allocated() - baseline) > mem_threshold_mb * 1024 * 1024
        except Exception:
            pass
    util_max = max(samples) if samples else 0
    return "GPU" if (torch_used or util_max >= util_threshold) else "CPU"


def metrics_path(results_dir: str | Path, family: str) -> Path:
    return Path(results_dir) / f"{family}_metrics.csv"


def artifact_path(results_dir: str | Path, family: str, artifact: str, model_name: str) -> Path:
    return Path(results_dir) / f"{family}_{artifact}_{model_name}.csv"


def label_audit_path(results_dir: str | Path, family: str, model_name: str) -> Path:
    return Path(results_dir) / f"{family}_label_audit_{model_name}.json"


def model_audit_path(results_dir: str | Path, family: str, model_name: str) -> Path:
    return Path(results_dir) / f"{family}_audit_{model_name}.json"


def save_model_artifacts(
    results_dir: str | Path,
    family: str,
    model_name: str,
    per_class: pd.DataFrame,
    confusion_long: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    results_dir = ensure_dir(results_dir)
    per_class.to_csv(artifact_path(results_dir, family, "per_class", model_name), index=False)
    confusion_long.to_csv(artifact_path(results_dir, family, "confusion", model_name), index=False)
    predictions.to_csv(artifact_path(results_dir, family, "predictions", model_name), index=False)


def save_label_audit(
    results_dir: str | Path,
    family: str,
    model_name: str,
    truth_raw: pd.Series,
    truth_harmonized: pd.Series,
    pred_raw: pd.Series,
    pred_harmonized: pd.Series,
) -> None:
    target_labels = benchmark_target_labels(truth_harmonized)
    off_target_pred_counts = (
        pd.Series(pred_harmonized, dtype="object")
        .astype(str)
        .value_counts()
        .loc[lambda s: ~s.index.isin(target_labels)]
        .sort_index()
        .to_dict()
    )
    unmapped_truth = sorted(
        {
            raw
            for raw, mapped in zip(truth_raw.astype(str), truth_harmonized.astype(str))
            if raw == mapped
        }
    )
    unmapped_pred = sorted(
        {
            raw
            for raw, mapped in zip(pred_raw.astype(str), pred_harmonized.astype(str))
            if raw == mapped
        }
    )
    save_json(
        label_audit_path(results_dir, family, model_name),
        {
            "model_name": model_name,
            "benchmark_target_labels": target_labels,
            "truth_labels_raw": sorted(pd.Series(truth_raw, dtype="object").astype(str).unique().tolist()),
            "truth_labels_harmonized": sorted(pd.Series(truth_harmonized, dtype="object").astype(str).unique().tolist()),
            "pred_labels_raw": sorted(pd.Series(pred_raw, dtype="object").astype(str).unique().tolist()),
            "pred_labels_harmonized": sorted(pd.Series(pred_harmonized, dtype="object").astype(str).unique().tolist()),
            "off_target_pred_labels": sorted(off_target_pred_counts),
            "off_target_pred_counts": off_target_pred_counts,
            "unmapped_truth_labels": unmapped_truth,
            "unmapped_pred_labels": unmapped_pred,
        },
    )


def save_model_audit(results_dir: str | Path, family: str, model_name: str, payload: dict[str, Any]) -> None:
    save_json(model_audit_path(results_dir, family, model_name), _json_ready(payload))


def save_metrics_table(results_dir: str | Path, family: str, metrics_df: pd.DataFrame) -> None:
    metrics_df.to_csv(metrics_path(results_dir, family), index=False)


def load_metrics_table(results_dir: str | Path, family: str) -> pd.DataFrame | None:
    path = metrics_path(results_dir, family)
    if path.exists():
        return pd.read_csv(path)
    
    legacy = Path(results_dir) / family / "metrics.csv"
    if legacy.exists():
        return pd.read_csv(legacy)
    return None


def load_model_artifact(results_dir: str | Path, family: str, artifact: str, model_name: str) -> pd.DataFrame | None:
    path = artifact_path(results_dir, family, artifact, model_name)
    if path.exists():
        return pd.read_csv(path)
    
    legacy = Path(results_dir) / family / artifact / f"{model_name}.csv"
    if legacy.exists():
        return pd.read_csv(legacy)
    return None


def load_combined_metrics(results_dir: str | Path) -> pd.DataFrame:
    frames = []
    for family in ("hector", "popv"):
        frame = load_metrics_table(results_dir, family)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No metrics tables found.")
    
    metrics = pd.concat(frames, ignore_index=True, sort=False)
    metrics = metrics[metrics["available"].fillna(False)]
    order_map = {name: idx for idx, name in enumerate(MODEL_DISPLAY_ORDER)}
    metrics["plot_rank"] = metrics["model_name"].map(order_map).fillna(10_000)
    metrics = metrics.sort_values(["plot_rank", "model_name"]).drop(columns="plot_rank")
    return metrics


def figure1_model_order(metrics: pd.DataFrame) -> list[str]:
    missing_auc = metrics[["roc_auc_macro", "pr_auc_macro"]].isna().any(axis=1)
    ordered = metrics["model_name"].tolist()
    missing = set(metrics.loc[missing_auc, "model_name"])
    non_hector = [model for model in ordered if model != "HECTOR"]
    reordered = [model for model in non_hector if model not in missing]
    reordered.extend(model for model in non_hector if model in missing)
    if "POPV" in reordered:
        reordered = [model for model in reordered if model != "POPV"] + ["POPV"]
    if "HECTOR" in ordered:
        reordered.append("HECTOR")
    return reordered


def build_combined_metrics_wide(metrics: pd.DataFrame, ordered_models: list[str]) -> pd.DataFrame:
    model_order = [model for model in ordered_models if model in set(metrics["model_name"])]
    wide = (
        metrics.drop_duplicates(subset="model_name", keep="first")
        .set_index("model_name")
        .reindex(model_order)[METRIC_COLUMNS]
        .T.rename_axis("metric_name")
        .reset_index()
    )
    return wide


def _load_tagged_model_artifacts(results_dir: str | Path, ordered_models: list[str], artifact: str) -> pd.DataFrame:
    tagged_frames = []
    for model_name in ordered_models:
        family = "hector" if model_name == "HECTOR" else "popv"
        frame = load_model_artifact(results_dir, family, artifact, model_name)
        if frame is None:
            continue
        tagged_frame = frame.copy()
        tagged_frame["model_name"] = model_name
        tagged_frames.append(tagged_frame)
    if not tagged_frames:
        raise FileNotFoundError(f"Missing {artifact} artifacts needed for publication figures.")
    return pd.concat(tagged_frames, ignore_index=True)


def load_per_class_artifacts(results_dir: str | Path, ordered_models: list[str]) -> pd.DataFrame:
    return _load_tagged_model_artifacts(results_dir, ordered_models, "per_class")


def _load_prediction_tables(results_dir: str | Path, ordered_models: list[str]) -> dict[str, pd.DataFrame]:
    prediction_tables: dict[str, pd.DataFrame] = {}
    for model_name in ordered_models:
        family = "hector" if model_name == "HECTOR" else "popv"
        frame = load_model_artifact(results_dir, family, "predictions", model_name)
        if frame is None or frame.empty or "cell_id" not in frame.columns:
            continue
        table = frame.copy()
        table["cell_id"] = table["cell_id"].astype(str)
        table = table.drop_duplicates(subset="cell_id", keep="last").set_index("cell_id")
        prediction_tables[model_name] = table
    return prediction_tables


def _require_anndata_module():
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "Figure 3 UMAP plotting requires `anndata`. Run the plotting step in the `sc` conda environment."
        ) from exc
    return ad


def _require_scanpy_module():
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "Computing a fallback UMAP requires `scanpy`. Run the plotting step in the `sc` conda environment."
        ) from exc
    return sc


def _load_saved_cell_ids(cell_id_file: str | Path) -> list[str]:
    cell_id_file = Path(cell_id_file)
    if not cell_id_file.exists():
        raise FileNotFoundError(f"Missing sampled cell ID file: {cell_id_file}")
    cell_ids = [line.strip() for line in cell_id_file.read_text().splitlines() if line.strip()]
    if not cell_ids:
        raise ValueError(f"No sampled cell IDs were found in {cell_id_file}.")
    return cell_ids


def _load_umap_subset(config: dict[str, Any]):
    ad = _require_anndata_module()
    data_path = Path(config["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {data_path}")

    cell_ids = _load_saved_cell_ids(config["cell_id_file"])
    adata = ad.read_h5ad(data_path)
    obs_names = pd.Index(adata.obs_names.astype(str), dtype="object")
    obs_name_set = set(obs_names)
    missing_ids = [cell_id for cell_id in cell_ids if cell_id not in obs_name_set]
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise ValueError(
            f"{len(missing_ids)} sampled cell IDs from {config['cell_id_file']} were not found in {data_path}. "
            f"Examples: {preview}"
        )

    subset = adata[cell_ids].copy()
    subset.obs_names = pd.Index(subset.obs_names.astype(str), dtype="object")
    return subset, cell_ids


def _get_series_for_cells(table: pd.DataFrame, cell_ids: list[str], column: str, series_name: str) -> pd.Series:
    if column not in table.columns:
        raise ValueError(f"Prediction artifact for {series_name} is missing required column {column!r}.")
    series = table.reindex(cell_ids)[column]
    missing_mask = series.isna()
    if missing_mask.any():
        missing_ids = series.index[missing_mask].tolist()
        preview = ", ".join(missing_ids[:10])
        raise ValueError(
            f"Prediction artifact for {series_name} is missing {missing_mask.sum()} sampled cells. Examples: {preview}"
        )
    return series.astype(str)


def _resolve_ground_truth_labels(
    prediction_tables: dict[str, pd.DataFrame],
    cell_ids: list[str],
    subset,
    config: dict[str, Any],
) -> pd.Series:
    truth_series: pd.Series | None = None
    for model_name, table in prediction_tables.items():
        if "true_label" not in table.columns:
            continue
        candidate = _get_series_for_cells(table, cell_ids, "true_label", model_name)
        if truth_series is None:
            truth_series = candidate
            continue
        if not candidate.equals(truth_series):
            raise ValueError(f"Saved true labels are inconsistent across prediction artifacts. First mismatch: {model_name}")

    if truth_series is not None:
        return truth_series

    label_key = config.get("label_key", LABEL_KEY)
    if label_key not in subset.obs:
        raise ValueError(
            f"Could not recover harmonized truth labels from predictions, and {label_key!r} is not present in the dataset."
        )
    label_map = merge_label_map(config.get("label_map_overrides"))
    truth_raw = subset.obs[label_key].astype(str)
    return harmonize_series(truth_raw, label_map=label_map)


def _build_shared_label_palette(
    truth_labels: pd.Series,
) -> tuple[list[str], dict[str, str]]:
    ordered_labels = benchmark_target_labels(truth_labels)

    color_pool: list[str] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = mpl.colormaps[cmap_name]
        color_pool.extend([mcolors.to_hex(cmap(step / 19)) for step in range(20)])
    if len(ordered_labels) > len(color_pool):
        extra_count = len(ordered_labels) - len(color_pool)
        spectral = mpl.colormaps["nipy_spectral"]
        color_pool.extend([mcolors.to_hex(spectral((step + 1) / (extra_count + 1))) for step in range(extra_count)])

    palette = {label: color_pool[idx] for idx, label in enumerate(ordered_labels)}
    return ordered_labels, palette


def _get_umap_coordinates(subset) -> np.ndarray:
    if "X_umap" in subset.obsm:
        coords = np.asarray(subset.obsm["X_umap"])
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2]

    if subset.n_obs < 3 or subset.n_vars < 2:
        raise ValueError("Fallback UMAP computation needs at least 3 cells and 2 features.")

    sc = _require_scanpy_module()
    if "X_pca" not in subset.obsm:
        max_components = max(1, min(50, subset.n_obs - 1, subset.n_vars - 1))
        sc.pp.pca(subset, n_comps=max_components)
    neighbor_count = min(15, subset.n_obs - 1)
    sc.pp.neighbors(subset, n_neighbors=neighbor_count, use_rep="X_pca")
    sc.tl.umap(subset)
    coords = np.asarray(subset.obsm["X_umap"])
    return coords[:, :2]


def _umap_point_size(n_cells: int) -> float:
    if n_cells >= 50_000:
        return 1.8
    if n_cells >= 20_000:
        return 2.6
    return 4.0


def _macro_heatmap_cmap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "macro_circle_heatmap",
        [MACRO_HEATMAP_LOW_COLOR, MACRO_HEATMAP_MID_COLOR, MACRO_HEATMAP_HIGH_COLOR],
    )


def _per_class_heatmap_cmap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "per_class_heatmap",
        PER_CLASS_HEATMAP_COLORS,
    )


def _macro_heatmap_text_color(fill_color: str | tuple[float, float, float, float]) -> str:
    red, green, blue = mcolors.to_rgb(fill_color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.55 else MACRO_HEATMAP_TEXT_COLOR


def draw_macro_circle_heatmap(
    ax,
    frame: pd.DataFrame,
    value_fmt: str | None = None,
) -> plt.cm.ScalarMappable:
    cmap = _macro_heatmap_cmap()
    values = pd.to_numeric(pd.Series(frame.to_numpy().ravel()), errors="coerce").dropna()
    if values.empty:
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    else:
        min_value = float(values.min())
        max_value = float(values.max())
        mean_value = float(values.mean())
        if np.isclose(min_value, max_value):
            spread = 0.05 if np.isclose(min_value, 0.0) else abs(min_value) * 0.05
            norm = mcolors.Normalize(vmin=min_value - spread, vmax=max_value + spread)
        else:
            norm = mcolors.TwoSlopeNorm(vmin=min_value, vcenter=mean_value, vmax=max_value)
    
    n_rows, n_cols = frame.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(frame.columns, rotation=45, ha="right")
    ax.set_yticklabels(frame.index)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=MACRO_HEATMAP_GRID_COLOR, linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=False, labelsize=10, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    circle_radius = 0.47
    for row_idx in range(n_rows):
        for col_idx, column in enumerate(frame.columns):
            value = frame.iat[row_idx, col_idx]
            center = (col_idx, row_idx)
            if pd.isna(value):
                ax.add_patch(
                    patches.Circle(
                        center,
                        radius=circle_radius,
                        facecolor="none",
                        edgecolor=MACRO_HEATMAP_NA_EDGE_COLOR,
                        linewidth=1.4,
                        zorder=3,
                        clip_on=False,
                    )
                )
                ax.text(
                    col_idx,
                    row_idx,
                    "NA",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#7A7A7A",
                    style="italic",
                    zorder=4,
                )
                continue
            
            fill_color = cmap(norm(float(value)))
            ax.add_patch(
                patches.Circle(
                    center,
                    radius=circle_radius,
                    facecolor=fill_color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                    clip_on=False,
                )
            )
            if value_fmt is not None:
                ax.text(
                    col_idx,
                    row_idx,
                    format(value, value_fmt),
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color=_macro_heatmap_text_color(fill_color),
                    zorder=4,
                )
    
    # The colorbar communicates the shared scale directly; no explanatory sentence
    # is printed beneath the matrix.
    ax.set_xlabel("")
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def draw_heatmap(
    ax,
    frame: pd.DataFrame,
    cmap: str | mcolors.Colormap,
    vmin: float,
    vmax: float,
    value_fmt: str | None = None,
    colorbar_label: str | None = None,
) -> None:
    image = ax.imshow(frame.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, clip_on=False)
    ax.set_xticks(np.arange(frame.shape[1]))
    ax.set_yticks(np.arange(frame.shape[0]))
    ax.set_xticklabels(frame.columns, rotation=45, ha="right")
    ax.set_yticklabels(frame.index)
    ax.set_xticks(np.arange(-0.5, frame.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, frame.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#d9d9d9", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    if value_fmt is not None:
        midpoint = (vmin + vmax) / 2
        for row_idx in range(frame.shape[0]):
            for col_idx in range(frame.shape[1]):
                value = frame.iat[row_idx, col_idx]
                text_color = "white" if value >= midpoint else "#111111"
                ax.text(
                    col_idx,
                    row_idx,
                    format(value, value_fmt),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )
    if colorbar_label is not None:
        colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        if colorbar.solids is not None:
            colorbar.solids.set_clip_on(False)
        colorbar.set_label(colorbar_label)


def build_macro_heatmap_frame(metrics: pd.DataFrame, ordered_models: list[str]) -> pd.DataFrame:
    return metrics.set_index("model_name").reindex(ordered_models)[METRIC_COLUMNS].rename(
        index={name: display_name(name) for name in ordered_models}
    )


def per_class_column_order(per_class: pd.DataFrame) -> list[str]:
    return per_class.groupby("class_label")["f1"].mean().sort_values().index.tolist()


def build_per_class_heatmap_frame(
    per_class: pd.DataFrame,
    ordered_models: list[str],
    class_order: list[str],
) -> pd.DataFrame:
    frame = (
        per_class.pivot(index="model_name", columns="class_label", values="f1")
        .reindex(index=ordered_models, columns=class_order)
        .rename(index={name: display_name(name) for name in ordered_models})
    )
    # Every scoring class is shown, including those no method reaches: dropping
    # all-zero columns hides exactly the classes that are the finding — a
    # reference vocabulary that cannot name a population the query contains —
    # and would make this panel irreconcilable with the macro F1 beside it
    # (mouse aorta: 5 dropped columns average 0.787 against a reported macro of 0.492).
    return frame


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def _plot_combined_figure(
    config: dict[str, Any],
    results_dir: Path,
    metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    ordered_models: list[str],
) -> None:
    # Sized to content, not to a page. No title row: these supplementary pages
    # carry no figure-level title, identified instead by the manuscript legend
    # and numbered file name.
    fig_width = 22.0
    top_h, bottom_h = 12.0, 8.0          # height_ratios: UMAP region taller than heatmaps
    spacer_h = 0.75                      # empty row between top/bottom regions (constrained_layout-proof)
    fig_height = top_h + spacer_h + bottom_h
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    # An explicit empty spacer row guarantees the gap; constrained_layout otherwise
    # overrides hspace when its own label-driven spacing is already larger.
    master = fig.add_gridspec(3, 1, height_ratios=[top_h, spacer_h, bottom_h], hspace=0.0)
    native_supplement = config["figure_file_name"] in (
        "supplementary_figure_3_human_lung.pdf",
        "supplementary_figure_4_human_skin.pdf", "supplementary_figure_5_mouse_kidney.pdf",
        "supplementary_figure_6_mouse_aorta.pdf", "supplementary_figure_7_mouse_hippocampus.pdf")
    if native_supplement:
        fig.set_size_inches(270 / 25.4, 360 / 25.4)
        fig.set_layout_engine(None)
        master = fig.add_gridspec(
            3, 1, height_ratios=[165, 22, 95], hspace=0,
            left=0.12, right=0.94, top=0.96, bottom=0.17)

    ax_truth, ax_timing = _draw_figure3_region(fig, master[0], config, results_dir, metrics, ordered_models)
    ax_macro, ax_perclass = _draw_figure1_region(fig, master[2], metrics, per_class, ordered_models)

    # Panel C's method names set the size for every text artist here (titles,
    # ticks, values, legends, colorbars), so nothing drifts into its own tier.
    panel_c_row_labels = ax_macro.get_yticklabels()
    supplement_font_size = (
        panel_c_row_labels[0].get_fontsize()
        if panel_c_row_labels
        else 10.0
    )
    for text_artist in fig.findobj(match=Text):
        text_artist.set_fontsize(supplement_font_size)
    if native_supplement:
        for text_artist in fig.findobj(match=Text):
            text_artist.set_fontsize(9)
        for axis in (ax_macro, ax_perclass):
            for annotation in axis.texts:
                annotation.set_fontsize(6.5)
            axis.title.set_fontsize(9)
        ax_macro.set_title("Benchmark scores")
        ax_timing.set_title("")
        # Reserve a real gutter for the score scale and the F1 method names.
        macro_position = ax_macro.get_position()
        ax_macro.set_position([macro_position.x0 - 0.025, macro_position.y0,
                               macro_position.width, macro_position.height])
        f1_position = ax_perclass.get_position()
        ax_perclass.set_position([f1_position.x0 + 0.055, f1_position.y0,
                                  f1_position.width - 0.045, f1_position.height])
        ax_perclass.tick_params(axis="y", labelsize=7.2)
        score_bar = ax_macro.child_axes[0]
        score_bar.set_yticklabels([
            label.get_text().removeprefix("Mean ")
            for label in score_bar.get_yticklabels()
        ])
        score_bar.tick_params(labelsize=7.2)
        score_bar.yaxis.label.set_fontsize(7.2)
        # Centre the descriptor and clear the midpoint's numeric tick label.
        score_bar.yaxis.set_label_coords(5.0, 0.5)
        if config["dataset_key"] == "mouse_kidney":
            # Thirteen F1 columns need more width than the skin comparison.
            ax_macro.set_position([macro_position.x0 - 0.025, macro_position.y0,
                                   macro_position.width * 0.90, macro_position.height])
            ax_macro.set_anchor("N")
            aligned_macro = ax_macro.get_position()
            ax_perclass.set_position([f1_position.x0 + 0.015, aligned_macro.y0,
                                      f1_position.width - 0.005, aligned_macro.height])
            ax_perclass.set_xticklabels([
                fill(label.get_text(), width=26)
                for label in ax_perclass.get_xticklabels()
            ], rotation=45, ha="right", fontsize=9)

    # Panel letters share a common left edge (measured after constrained_layout
    # resolves, just left of each panel's own leftmost element).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _tightbbox_left(ax):
        return ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).x0

    pos = {name: ax.get_position() for name, ax in
           (("a", ax_truth), ("b", ax_timing), ("c", ax_macro), ("d", ax_perclass))}
    left_abc = min(_tightbbox_left(ax_truth), _tightbbox_left(ax_timing), _tightbbox_left(ax_macro)) - 0.016
    for name in ("a", "b", "c"):
        fig.text(left_abc, pos[name].y1 + 0.004, name,
                 fontsize=15, fontweight="bold", va="bottom", ha="left")
    fig.text(_tightbbox_left(ax_perclass) - 0.016, pos["d"].y1 + 0.004, "d",
             fontsize=15, fontweight="bold", va="bottom", ha="left")

    figure_pdf_path = results_dir / config["figure_file_name"]
    figure_png_path = figure_pdf_path.with_suffix(".png")

    if native_supplement:
        fig.savefig(figure_pdf_path, bbox_inches=None, dpi=400)
        fig.savefig(figure_png_path, bbox_inches=None, dpi=400,
                    pil_kwargs={"dpi": (600, 600)})
    else:
        fig.savefig(figure_pdf_path, bbox_inches="tight", pad_inches=0.5, dpi=400)
        fig.savefig(figure_png_path, bbox_inches="tight", pad_inches=0.5, dpi=200)
    plt.close(fig)


def plot_publication_figures(config: dict[str, Any]) -> None:
    config = resolve_dataset_config(config)
    results_dir = Path(config["results_dir"])
    metrics = load_combined_metrics(results_dir)
    ordered_models = metrics["model_name"].tolist()
    figure1_models = figure1_model_order(metrics)
    per_class = load_per_class_artifacts(results_dir, ordered_models)
    build_combined_metrics_wide(metrics, figure1_models).to_csv(results_dir / "combined_metrics.csv", index=False)
    # Single combined main figure (Fig 1 + Fig 3 regions); no standalone sub-figures.
    _plot_combined_figure(config, results_dir, metrics, per_class, figure1_models)


def _draw_figure1_region(
    fig,
    spec,
    metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    ordered_models: list[str],
) -> None:
    # Macro matrix (C) and per-class heatmap (D) share the y-axis so model rows
    # line up. The colorbar is pinned to the matrix's right edge with an inset
    # axes rather than a floating gap.
    gs = spec.subgridspec(1, 2, width_ratios=[1.2, 1.5], wspace=0.14)
    ax_heatmap = fig.add_subplot(gs[0, 0])
    heatmap_df = build_macro_heatmap_frame(metrics, ordered_models)
    color_scale = draw_macro_circle_heatmap(
        ax_heatmap,
        heatmap_df,
        value_fmt=".3f",
    )
    cax = ax_heatmap.inset_axes([1.04, 0.0, 0.045, 1.0])  # pinned to the matrix's right edge
    colorbar = fig.colorbar(color_scale, cax=cax)
    if colorbar.solids is not None:
        colorbar.solids.set_clip_on(False)
    scale_norm = color_scale.norm
    scale_min = float(scale_norm.vmin)
    scale_max = float(scale_norm.vmax)
    scale_mean = float(getattr(scale_norm, "vcenter", (scale_min + scale_max) / 2))
    colorbar.set_label("Benchmark score", fontsize=9)
    colorbar.set_ticks([scale_min, scale_mean, scale_max])
    colorbar.set_ticklabels([f"{scale_min:.3f}", f"Mean {scale_mean:.3f}", f"{scale_max:.3f}"])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8)
    ax_heatmap.set_title("Benchmark scores by method and metric")
    ax_heatmap.set_ylabel("")
    # NA cells (PopV, CellTypist, HECTOR return labels, not an aligned per-class
    # probability matrix, so no threshold-based score exists) are explained in
    # the legend, not drawn here.
    if "HECTOR" in ordered_models:
        hector_idx = ordered_models.index("HECTOR")
        ax_heatmap.add_patch(
            patches.Rectangle(
                (-0.5, hector_idx - 0.5),
                len(METRIC_COLUMNS),
                1,
                fill=False,
                edgecolor="black",
                linewidth=2.5,
                clip_on=False,
            )
        )
    
    class_order = per_class_column_order(per_class)
    ax_per_class = fig.add_subplot(gs[0, 1], sharey=ax_heatmap)
    per_class_heatmap_df = build_per_class_heatmap_frame(per_class, ordered_models, class_order)
    draw_heatmap(
        ax_per_class,
        per_class_heatmap_df,
        cmap=_per_class_heatmap_cmap(),
        vmin=0.0,
        vmax=1.0,
        value_fmt=".3f",
        colorbar_label="Per-class F1",
    )
    ax_per_class.set_title("Per-class F1")
    ax_per_class.set_xlabel("")
    ax_per_class.set_ylabel("")
    heatmap_texts = ax_per_class.texts
    n_heatmap_cols = per_class_heatmap_df.shape[1]
    for col_idx, class_label in enumerate(per_class_heatmap_df.columns):
        series = per_class[per_class["class_label"] == class_label].set_index("model_name")["f1"]
        available = series.reindex(ordered_models).dropna()
        if available.empty:
            continue
        best_value = float(available.max())
        if np.isclose(best_value, 0.0):
            continue
        best_models = available[np.isclose(available, best_value)].index.tolist()
        for best_model in best_models:
            row_idx = ordered_models.index(best_model)
            text_idx = row_idx * n_heatmap_cols + col_idx
            if text_idx >= len(heatmap_texts):
                continue
            best_text = heatmap_texts[text_idx]
            best_text.set_fontweight("bold")
            best_text.set_fontsize(9.3)
            best_text.set_color(PER_CLASS_HEATMAP_BEST_TEXT_COLOR)

    return ax_heatmap, ax_per_class


def _draw_timing_axis(
    ax,
    metrics: pd.DataFrame,
    ordered_models: list[str],
) -> None:
    """Draw the inference-time horizontal bar chart onto ``ax``.

    Previously this was the standalone Figure 2; it now lives as the lower-left
    panel of Figure 3.
    """
    timing_rows = metrics[
        metrics["model_name"].isin(ordered_models) & metrics["inference_seconds"].notna()
    ].copy()
    if timing_rows.empty:
        ax.axis("off")
        return

    rank_map = {name: idx for idx, name in enumerate(ordered_models)}
    timing_rows = timing_rows.assign(
        rank=timing_rows["model_name"].map(rank_map)
    ).sort_values("rank")

    labels = [display_name(n) for n in timing_rows["model_name"]]
    seconds = timing_rows["inference_seconds"].values
    devices = (
        timing_rows["inference_device"].tolist()
        if "inference_device" in timing_rows.columns
        else [None] * len(labels)
    )
    # Single quantitative hue, matching Figure 1's heatmap accent rather than
    # the saturated categorical (tab20) colours of the UMAP panels.
    bar_color = MACRO_HEATMAP_HIGH_COLOR

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, seconds, color=bar_color, edgecolor="white", height=0.6, clip_on=False)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Inference Time (seconds)", fontsize=10)
    ax.set_title("Inference Time", fontsize=12)

    max_sec = float(max(seconds))
    for i, (sec, dev) in enumerate(zip(seconds, devices)):
        label = f"{sec / 60:.1f} min" if sec >= 60 else f"{sec:.1f}s"
        if isinstance(dev, str) and dev:
            label = f"{label}  ·  {dev}"
        # Long bars: label sits inside the bar (right-aligned, white) so it never
        # spills past the panel edge; short bars: label just past the bar end.
        if sec > 0.55 * max_sec:
            ax.text(sec - max_sec * 0.012, i, label, va="center", ha="right", fontsize=8, color="white")
        else:
            ax.text(sec + max_sec * 0.015, i, label, va="center", ha="left", fontsize=8, color="#555555")

    ax.set_xlim(0, max_sec * 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_umap_panel(
    ax,
    plot_frame: pd.DataFrame,
    column: str,
    plot_palette: dict,
    point_size: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    title: str | None = None,
    alpha: float = 0.82,
) -> None:
    """Draw one UMAP scatter panel.

    The main figure passes alpha=1.0 and draws its dots solid; the per-dataset
    supplements keep the see-through default they were tuned with.
    """
    labels = plot_frame[column].astype(str)
    colors = labels.map(plot_palette).fillna(OFF_TARGET_PLOT_COLOR)
    ax.scatter(
        plot_frame["umap_1"],
        plot_frame["umap_2"],
        c=colors,
        s=point_size,
        alpha=alpha,
        linewidths=0.0,
        rasterized=True,
        clip_on=False,
    )
    ax.set_title(title or column, fontsize=11, color="#1A1A1A")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_figure3_region(
    fig,
    spec,
    config: dict[str, Any],
    results_dir: Path,
    metrics: pd.DataFrame,
    ordered_models: list[str],
) -> None:
    native_supplement = config["figure_file_name"] in (
        "supplementary_figure_3_human_lung.pdf",
        "supplementary_figure_4_human_skin.pdf", "supplementary_figure_5_mouse_kidney.pdf",
        "supplementary_figure_6_mouse_aorta.pdf", "supplementary_figure_7_mouse_hippocampus.pdf")
    prediction_tables = _load_prediction_tables(results_dir, ordered_models)
    ordered_models = [model_name for model_name in ordered_models if model_name in prediction_tables]
    if not ordered_models:
        raise FileNotFoundError(
            f"No saved prediction artifacts were found in {results_dir}. Figure 3 needs *_predictions_*.csv outputs."
        )

    subset, cell_ids = _load_umap_subset(config)
    # Some methods (notably POPV) drop a few cells during QC, so not every
    # sampled cell has a prediction from every model. Restrict the figure to
    # cells shared across all plotted models so the panels stay aligned.
    available = set(cell_ids)
    for model_name in ordered_models:
        available &= set(prediction_tables[model_name].index.astype(str))
    if len(available) < len(cell_ids):
        kept = [cell_id for cell_id in cell_ids if cell_id in available]
        print(
            f"  Figure 3: {len(cell_ids) - len(kept)} of {len(cell_ids)} cells lack a "
            f"prediction from every model; plotting the {len(kept)} shared cells."
        )
        subset = subset[kept].copy()
        cell_ids = kept
    truth_labels = _resolve_ground_truth_labels(prediction_tables, cell_ids, subset, config)
    ordered_labels, shared_palette = _build_shared_label_palette(truth_labels)
    target_label_set = set(ordered_labels)
    plot_palette = shared_palette.copy()
    plot_palette[OFF_TARGET_PLOT_LABEL] = OFF_TARGET_PLOT_COLOR
    if config["dataset_key"] == "lung":
        # Uses the cached fallback embedding, not the dataset's own X_UMAP; cell
        # IDs enforce alignment after the shared-prediction filter.
        embedding_cache = results_dir / "plot_umap_coordinates.csv"
        if embedding_cache.exists():
            cached_embedding = pd.read_csv(embedding_cache, dtype={"cell_id": str})
            cached_embedding = cached_embedding.set_index("cell_id")
            if not cached_embedding.index.is_unique or set(cached_embedding.index) != set(cell_ids):
                raise ValueError("Lung UMAP cache cell IDs differ from the plotted cells.")
            subset.obsm["X_umap"] = cached_embedding.loc[cell_ids, ["umap_1", "umap_2"]].to_numpy()
        else:
            coordinates = _get_umap_coordinates(subset)
            pd.DataFrame({"cell_id": cell_ids, "umap_1": coordinates[:, 0],
                          "umap_2": coordinates[:, 1]}).to_csv(embedding_cache, index=False)
    if native_supplement and "X_umap" not in subset.obsm:
        raise ValueError("Native supplementary figures require cached X_umap coordinates.")
    umap_coords = subset.obsm["X_umap"] if native_supplement else _get_umap_coordinates(subset)

    plot_frame = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "umap_1": umap_coords[:, 0],
            "umap_2": umap_coords[:, 1],
            "Ground Truth": truth_labels.to_numpy(),
        }
    )
    for model_name in ordered_models:
        prediction_labels = _get_series_for_cells(
            prediction_tables[model_name],
            cell_ids,
            "pred_label",
            model_name,
        )
        plot_labels = prediction_labels.where(prediction_labels.isin(target_label_set), other=OFF_TARGET_PLOT_LABEL)
        plot_frame[display_name(model_name)] = plot_labels.to_numpy()

    # Layout: left column = large Ground Truth (top) over Inference Timing
    # (bottom); right grid = HECTOR | POPV | individual methods | legend.
    present = set(ordered_models)
    canonical_methods = [m for m in MODEL_DISPLAY_ORDER if m not in ("HECTOR", "POPV")]
    method_models = [m for m in canonical_methods if m in present]
    method_models += [
        m for m in ordered_models
        if m not in MODEL_DISPLAY_ORDER and m not in ("HECTOR", "POPV")
    ]

    LEGEND_CELL = "__legend__"
    # HECTOR/POPV first, then individual methods, then the legend; padded with
    # None to fill the final row.
    right_cells: list[str | None] = [
        display_name(headline) for headline in ("HECTOR", "POPV") if headline in present
    ]
    right_cells += [display_name(m) for m in method_models]
    if not native_supplement:
        right_cells.append(LEGEND_CELL)

    ncols_right = 4
    nrows_right = max(1, math.ceil(len(right_cells) / ncols_right))
    right_cells += [None] * (nrows_right * ncols_right - len(right_cells))

    x_values = plot_frame["umap_1"].to_numpy()
    y_values = plot_frame["umap_2"].to_numpy()
    x_pad = max(0.5, 0.03 * (float(x_values.max()) - float(x_values.min()) or 1.0))
    y_pad = max(0.5, 0.03 * (float(y_values.max()) - float(y_values.min()) or 1.0))
    xlim = (float(x_values.min()) - x_pad, float(x_values.max()) + x_pad)
    ylim = (float(y_values.min()) - y_pad, float(y_values.max()) + y_pad)
    point_size = _umap_point_size(len(plot_frame))
    if native_supplement:
        half_span = 0.55 * max(float(np.ptp(x_values)), float(np.ptp(y_values)))
        x_center = float(x_values.max() + x_values.min()) / 2
        y_center = float(y_values.max() + y_values.min()) / 2
        xlim = (x_center - half_span, x_center + half_span)
        ylim = (y_center - half_span, y_center + half_span)
        point_size *= (270 / (22 * 25.4)) ** 2

    left_ratio = 1.9  # wide left column so Ground Truth / timing dominate the small grid
    outer = spec.subgridspec(1, 2, width_ratios=[left_ratio, ncols_right], wspace=0.12)
    # Narrow gutter (timing row labels) + wide plot column; their left edges
    # align in the plot column.
    left_gs = outer[0, 0].subgridspec(2, 2, width_ratios=[1.0, 4.4], hspace=0.12, wspace=0.0)
    right_gs = outer[0, 1].subgridspec(nrows_right, ncols_right, hspace=0.05, wspace=0.05)
    if native_supplement:
        right_gs = outer[0, 1].subgridspec(
            nrows_right + 1, ncols_right,
            height_ratios=[1] * nrows_right + [0.65], hspace=0.16, wspace=0.10)

    # Ground Truth spans the full left column (gutter + plot) so it starts at
    # the left margin, aligned with the timing row labels rather than indented.
    ax_truth = fig.add_subplot(left_gs[0, :])
    _draw_umap_panel(ax_truth, plot_frame, "Ground Truth",
                     plot_palette, point_size, xlim, ylim, title="Ground Truth")
    ax_timing = fig.add_subplot(left_gs[1, 1])
    _draw_timing_axis(ax_timing, metrics, ordered_models)

    # Right grid: HECTOR leads (top-left); the legend goes in its own cell.
    legend_ax = None
    for idx, cell in enumerate(right_cells):
        row, col = divmod(idx, ncols_right)
        ax = fig.add_subplot(right_gs[row, col])
        if cell == LEGEND_CELL:
            legend_ax = ax
            ax.axis("off")
            continue
        if cell is None or cell not in plot_frame.columns:
            ax.axis("off")
            continue
        _draw_umap_panel(ax, plot_frame, cell, plot_palette, point_size, xlim, ylim)
        if native_supplement:
            # Use the spare row spacing while retaining distinct square maps.
            map_position = ax.get_position()
            map_scale = 1.15
            ax.set_position([
                map_position.x0 - map_position.width * (map_scale - 1) / 2,
                map_position.y0 - map_position.height * (map_scale - 1) / 2,
                map_position.width * map_scale,
                map_position.height * map_scale,
            ])

    legend_columns = 1 if len(ordered_labels) <= 12 else 2
    label_wrap = 18 if legend_columns == 1 else 12
    if native_supplement:
        legend_columns = 3
        label_wrap = 100
        if config["dataset_key"] in ("mouse_kidney", "mouse_aorta", "mouse_hippocampus"):
            label_wrap = 26
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="w",
            markerfacecolor=shared_palette[label],
            markeredgecolor=shared_palette[label],
            markersize=8,
            label=fill(label, width=label_wrap),
        )
        for label in ordered_labels
    ]
    # Predictions that harmonize to "Unknown" or to a type absent from the
    # ground truth are drawn in gray; explain that swatch when it appears.
    panel_columns = [c for c in plot_frame.columns if c not in ("cell_id", "umap_1", "umap_2", "Ground Truth")]
    off_target_present = any((plot_frame[col] == OFF_TARGET_PLOT_LABEL).any() for col in panel_columns)
    if off_target_present:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color="w",
                markerfacecolor=OFF_TARGET_PLOT_COLOR,
                markeredgecolor=OFF_TARGET_PLOT_COLOR,
                markersize=8,
                label=fill("Unknown / other", width=label_wrap),
            )
        )
    if legend_ax is None:
        legend_ax = fig.add_subplot(right_gs[nrows_right, :] if native_supplement
                                    else right_gs[nrows_right - 1, ncols_right - 1])
        legend_ax.axis("off")
    legend = legend_ax.legend(
        handles=legend_handles,
        loc="center",
        frameon=False,
        title="Cell Types",
        ncol=legend_columns,
        fontsize=8,
        title_fontsize=10,
        handletextpad=0.6,
        columnspacing=1.1,
        labelspacing=0.2 if native_supplement else 0.9,
    )
    legend._legend_box.align = "left"

    return ax_truth, ax_timing


# -----------------------------------------------------------------------------
# Orchestration helpers
# -----------------------------------------------------------------------------

def resolve_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    dataset_key = str(config["dataset_key"])
    results_dir = str(config.get("results_dir") or (Path(RESULTS_DIR) / dataset_key))
    cell_id_file = str(config.get("cell_id_file") or (Path(results_dir) / "cell_ids.txt"))
    popv_config = POPV_SETTINGS.copy()
    popv_config.update(config.get("popv", {}))
    hector_config = HECTOR_SETTINGS.copy()
    hector_config.update(config.get("hector", {}))

    # Inline overrides, then the per-dataset harmonization table (which wins on
    # conflicts). A label mapping to "Unknown" is dropped — the one mechanism
    # for excluding ground-truth cells with no defensible target.
    label_map = deepcopy(config.get("label_map_overrides", {})) or {}
    harmonization_table = config.get("harmonization_table")
    if harmonization_table:
        label_map.update(load_harmonization_table(harmonization_table))
    drop_labels = {raw for raw, target in label_map.items() if str(target).strip() == "Unknown"}

    # display_name and figure_file_name (the printed title and file name) must
    # be carried through explicitly, or this fixed-key rebuild silently drops
    # them back to the internal dataset key.
    return {
        "dataset_key": dataset_key,
        "display_name": config.get("display_name", dataset_key),
        "figure_file_name": config.get("figure_file_name", f"figure_main_{dataset_key}.pdf"),
        "data_path": config["data_path"],
        "label_key": config.get("label_key", LABEL_KEY),
        "batch_key": config.get("batch_key", BATCH_KEY),
        "sample_size": config.get("sample_size", SAMPLE_SIZE),
        "random_state": config.get("random_state", RANDOM_STATE),
        "results_dir": results_dir,
        "cell_id_file": cell_id_file,
        "harmonization_table": str(harmonization_table) if harmonization_table else None,
        "label_map_overrides": label_map,
        "drop_truth_labels": sorted(drop_labels),
        "popv": popv_config,
        "hector": hector_config,
        "popv_conda_env": config.get("popv_conda_env", POPV_CONDA_ENV),
        "hector_conda_env": config.get("hector_conda_env", HECTOR_CONDA_ENV),
    }


def print_run_settings(config: dict, include_sampling: bool = True) -> None:
    config = resolve_dataset_config(config)
    print("Setup")
    print(f"  dataset: {config['dataset_key']}")
    print(f"  results folder: {config['results_dir']}")
    print(f"  data file: {config['data_path']}")
    print(f"  truth label key: {config['label_key']}")
    print(f"  batch key: {config['batch_key']}")
    print(f"  random state: {config['random_state']}")
    if include_sampling:
        print(f"  sample size: {config['sample_size']}")
    print(f"  POPV env: {config['popv_conda_env']}")
    print(f"  Hector env: {config['hector_conda_env']}")
    print(f"  POPV repo: {config['popv']['repo']}")
    print(f"  Hector model selector: {config['hector']['model_path']}")


def prepare_results_dir(config: dict) -> None:
    config = resolve_dataset_config(config)
    ensure_dir(config["results_dir"])


def write_worker_config(config: dict) -> Path:
    config = resolve_dataset_config(config)
    config_path = Path(config["results_dir"]) / WORKER_CONFIG_NAME
    ensure_dir(config["results_dir"])
    save_json(config_path, config)
    return config_path


def load_worker_config(config_path: str | Path) -> dict:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def run_worker(module_name: str, conda_env: str, config_path: str | Path, worker_name: str = "benchmark") -> None:
    command = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "-m",
        module_name,
        "--worker",
        worker_name,
        "--config",
        str(config_path),
    ]
    print(f"  Running {' '.join(command)}")
    # `python -m benchmark_modules...` finds the package only from scripts/, so the
    # worker is started there rather than wherever this run happened to begin.
    subprocess.run(command, check=True, cwd=REPO_ROOT / "scripts")


def generate_figures(config: dict) -> None:
    config = resolve_dataset_config(config)
    plot_publication_figures(config)
    print(f"  Figures saved under {config['results_dir']}")


def _count_saved_artifacts(results_dir: str | Path, artifact: str) -> int:
    results_dir = Path(results_dir)
    return sum(1 for _ in results_dir.glob(f"*_{artifact}_*.csv"))


def _validate_saved_cell_ids_in_dataset(config: dict[str, Any]) -> None:
    ad = _require_anndata_module()
    data_path = Path(config["data_path"])
    cell_ids = _load_saved_cell_ids(config["cell_id_file"])
    adata = ad.read_h5ad(data_path, backed="r")
    try:
        obs_name_set = set(pd.Index(adata.obs_names.astype(str), dtype="object"))
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    missing_ids = [cell_id for cell_id in cell_ids if cell_id not in obs_name_set]
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise ValueError(
            f"Cannot generate plots for dataset {config['dataset_key']!r}: "
            f"{len(missing_ids)} sampled cell IDs from {config['cell_id_file']} were not found in {data_path}. "
            f"Examples: {preview}"
        )


def validate_plot_inputs(config: dict) -> None:
    config = resolve_dataset_config(config)
    results_dir = Path(config["results_dir"])
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Cannot generate plots for dataset {config['dataset_key']!r}: "
            f"{results_dir} does not exist. Run the inference section or the whole script first."
        )
    
    missing: list[str] = []
    if not any(load_metrics_table(results_dir, family) is not None for family in ("hector", "popv")):
        missing.append("metrics tables")
    if _count_saved_artifacts(results_dir, "per_class") == 0:
        missing.append("per-class artifacts")
    if _count_saved_artifacts(results_dir, "confusion") == 0:
        missing.append("confusion artifacts")
    if _count_saved_artifacts(results_dir, "predictions") == 0:
        missing.append("prediction artifacts")
    if not Path(config["data_path"]).exists():
        missing.append(f"dataset file ({config['data_path']})")
    if not Path(config["cell_id_file"]).exists():
        missing.append(f"cell ID file ({config['cell_id_file']})")
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Cannot generate plots for dataset {config['dataset_key']!r}: missing {missing_text}. "
            "Run the inference section or the whole script first."
        )
    
    metrics = load_combined_metrics(results_dir)
    if metrics.empty:
        raise FileNotFoundError(
            f"Cannot generate plots for dataset {config['dataset_key']!r}: "
            f"no available model rows were found in {results_dir}. "
            "Run the inference section or the whole script first."
        )
    _validate_saved_cell_ids_in_dataset(config)


def run_inference_for_dataset(config: dict) -> None:
    config = resolve_dataset_config(config)
    print_run_settings(config)
    print("Inference")
    prepare_results_dir(config)
    config_path = write_worker_config(config)
    run_worker("benchmark_modules.hector_helpers", config["hector_conda_env"], config_path, worker_name="benchmark")
    run_worker("benchmark_modules.popv_helpers", config["popv_conda_env"], config_path, worker_name="benchmark")
    print(f"  Inference outputs saved under {config['results_dir']}")


def run_plots_for_dataset(config: dict) -> None:
    config = resolve_dataset_config(config)
    print("Plots")
    print(f"  dataset: {config['dataset_key']}")
    print(f"  results folder: {config['results_dir']}")
    validate_plot_inputs(config)
    generate_figures(config)


# -----------------------------------------------------------------------------
# Benchmark defaults
# -----------------------------------------------------------------------------
RESULTS_DIR = "benchmark_results"
LABEL_KEY = "cell_type"
BATCH_KEY = "donor_id"
SAMPLE_SIZE = 60000
RANDOM_STATE = 0
WORKER_CONFIG_NAME = "_worker_config.json"

POPV_CONDA_ENV = "sc"
HECTOR_CONDA_ENV = "sc"

POPV_SETTINGS = {
    "cache_dir": None,
    "force_cpu": True,
    "prediction_mode": "inference",
}

HECTOR_SETTINGS = {
    "model_path": "human",
    "top_k": 3,
    "use_grit": True,
    "use_asymmetric_ppr": False,
    "ppr_alpha": 0.25,
    "forward_weight": 0.6,
}
