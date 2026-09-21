#!/usr/bin/env python3
"""Shared evaluation helpers for the model benchmark."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from cell_ontology_ppr import normalize_ppr_rows, resolve_column_winners
from query_set_padding import pad_to_query_set

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent          # scripts/, which carries Hierarchical_metrics
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import logging  # noqa: E402

import networkx as nx  # noqa: E402

from Hierarchical_metrics.cell_ontology_metrics import (  # noqa: E402
    HIERARCHICAL_METRIC_COLUMNS,
    _canonicalize_labels,
    _coerce_category_input,
    _coerce_pair_inputs,
    _node_depth,
    _resolve_matching_dag,
    _validate_labels,
    classify_ontology_matches,
    compute_fine_ontology_distance,
    compute_hierarchical_metrics,
    summarize_hierarchical_metrics,
)


REQUIRED_EMBEDDING_KEYS = {
    "embeddings",
    "cell_types",
    "model",
    "source",
    "n_cells",
    "embedding_dim",
}
OPTIONAL_SEQUENCE_KEYS = {
    "cell_type_ids",
    "row_ids",
    "sampled_obs_indices",
    "kept_row_indices",
}

PREFERRED_OBS_ROW_ID_COLUMNS = (
    "observation_joinid",
    "soma_joinid",
)


def load_embeddings(path: str) -> dict:
    """Load embeddings pickle file and validate format."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    with open(p, "rb") as handle:
        data = pickle.load(handle)

    missing = REQUIRED_EMBEDDING_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Embeddings pickle missing keys: {missing}")

    embeddings = np.asarray(data["embeddings"])
    cell_types = list(data["cell_types"])
    if embeddings.shape[0] != len(cell_types):
        raise ValueError(
            f"Row count mismatch: embeddings has {embeddings.shape[0]} rows "
            f"but cell_types has {len(cell_types)} entries"
        )

    expected_row_count = len(cell_types)
    original_sampled_count = int(data["n_cells"])
    kept_row_indices = data.get("kept_row_indices")
    if kept_row_indices is not None:
        sampled_obs_indices = data.get("sampled_obs_indices")
        sampled_count = 0 if sampled_obs_indices is None else len(sampled_obs_indices)
        original_sampled_count = max(original_sampled_count, sampled_count)
        original_sampled_count = max(original_sampled_count, max(kept_row_indices, default=-1) + 1)

    for key in OPTIONAL_SEQUENCE_KEYS:
        values = data.get(key)
        if values is None:
            continue
        if key == "sampled_obs_indices" and len(values) not in {expected_row_count, original_sampled_count}:
            raise ValueError(
                f"Optional key {key!r} has length {len(values)}; expected "
                f"{expected_row_count} or original sampled count {original_sampled_count}."
            )
        elif key != "sampled_obs_indices" and len(values) != len(cell_types):
            raise ValueError(
                f"Optional key {key!r} has length {len(values)} but expected {len(cell_types)}."
            )

    return data


def preferred_obs_row_id_column(obs) -> str | None:
    """Return the preferred obs column name for stable row identity."""
    obs_columns = getattr(obs, "columns", [])
    for key in PREFERRED_OBS_ROW_ID_COLUMNS:
        if key in obs_columns:
            return key
    return None


def resolve_query_obs_row_id_column(
    obs,
    query_row_ids,
) -> str | None:
    """Pick the obs identifier source that best matches ``query_row_ids``.

    Returns an obs column name when one of the preferred row-id columns has the
    highest overlap with the provided query identifiers. Returns ``None`` when
    ``obs_names`` is the best match or when no preferred column is available.
    """
    obs_columns = getattr(obs, "columns", [])
    query_row_ids = pd.Index(pd.Series(query_row_ids).astype(str))
    query_row_id_set = set(query_row_ids)

    best_column = None
    best_hits = -1

    for key in PREFERRED_OBS_ROW_ID_COLUMNS:
        if key not in obs_columns:
            continue

        candidate_ids = pd.Series(obs[key]).astype(str)
        hits = int(candidate_ids.isin(query_row_id_set).sum())
        if hits > best_hits:
            best_column = key
            best_hits = hits

    obs_name_hits = int(pd.Index(obs.index.astype(str)).isin(query_row_id_set).sum())
    if obs_name_hits > best_hits:
        return None

    return best_column


def save_results(results: dict, output_path: str) -> None:
    """Save results dictionary to a pickle file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(results, handle)


def build_prototypes(
    embeddings: np.ndarray, cell_types: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute centroid embedding per label."""
    cell_types = np.asarray(cell_types)
    unique_types = np.unique(cell_types)
    prototypes: dict[str, np.ndarray] = {}
    for cell_type in unique_types:
        mask = cell_types == cell_type
        prototypes[str(cell_type)] = embeddings[mask].mean(axis=0)
    return prototypes


def compute_S_cell(
    embeddings: np.ndarray,
    prototypes: dict[str, np.ndarray],
    temperature: float = 0.1,
) -> tuple[np.ndarray, list[str]]:
    """Compute scaled cosine similarity between embeddings and prototypes."""
    type_names = list(prototypes.keys())
    prototype_matrix = np.array([prototypes[cell_type] for cell_type in type_names])

    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    proto_norm = prototype_matrix / (
        np.linalg.norm(prototype_matrix, axis=1, keepdims=True) + 1e-8
    )

    emb_scaled = emb_norm / temperature
    proto_scaled = proto_norm / temperature
    s_cell = emb_scaled @ proto_scaled.T
    return s_cell, type_names


def predict_with_candidate_profiles(
    query_to_bridge: np.ndarray,
    candidate_to_bridge: np.ndarray,
    candidate_ids: list[str],
) -> tuple[list[str], list[float]]:
    """Name each query cell from the bridge cell type it most resembles.

    The one read-out every embedding-only model is given, so that they compete on
    the embedding alone: find the bridge ("landmark") cell type the cell is most
    similar to, then answer with the candidate that scores highest there once
    every candidate's score is put on the same scale.  There is nothing to tune.

    Each candidate's raw personalized-PageRank row is rescaled by its own total
    across the tracked landmarks (`normalize_ppr_rows`) before the per-landmark
    winner is picked (`resolve_column_winners`), because comparing raw PageRank
    mass directly favours candidates sitting near well-connected ontology hubs
    regardless of the landmark being compared. Exact ties between candidates
    (structural mirror images in the ontology graph) are split across their tied
    landmarks by a seeded permutation rather than resolved to whichever
    candidate's ID happens to sort first.

    Rank-correlating the cell's similarity profile against each candidate's
    PageRank profile was tried and rejected: candidates' PageRank profiles are
    ~62% correlated with one another, so the rank transform discarded the
    magnitude that separates them and predictions collapsed onto a handful of
    labels (Geneformer answered one label for half of all cells); ranks are
    also invariant to monotone rescaling, so a temperature parameter on that
    rule changed no prediction at all.
    """
    if query_to_bridge.shape[1] != candidate_to_bridge.shape[1]:
        raise ValueError(
            "Query-to-bridge and candidate-to-bridge matrices must share the bridge axis: "
            f"{query_to_bridge.shape} vs {candidate_to_bridge.shape}"
        )

    nearest_landmark = np.argmax(np.asarray(query_to_bridge), axis=1)
    normalized = normalize_ppr_rows(candidate_to_bridge)
    winners, _tie_groups = resolve_column_winners(normalized)
    best = winners[nearest_landmark]
    scores = normalized[best, nearest_landmark].astype(float)

    predictions = [candidate_ids[int(j)] for j in best]
    return predictions, scores.tolist()


def evaluate_predictions(
    true_types: list[str], predicted_types: list[str], *, labels: list[str] | None = None,
) -> dict:
    """Compute flat classification metrics and collect misclassification examples.

    ``labels`` restricts macro averaging to the given classes. Pass the true
    held-out cell types when ``predicted_types`` can contain a sentinel for a
    cell a model never returned (query_set_padding.UNSCORED) — left to sklearn's
    default of inferring labels from the data, that sentinel would count as an
    extra always-zero class and dilute every real class's share of the average.
    """
    accuracy = accuracy_score(true_types, predicted_types)
    macro_f1 = f1_score(true_types, predicted_types, labels=labels, average="macro",
                         zero_division=0)
    weighted_f1 = f1_score(
        true_types,
        predicted_types,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    errors = [
        (i, truth, pred)
        for i, (truth, pred) in enumerate(zip(true_types, predicted_types))
        if truth != pred
    ]
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "errors": errors,
    }


def get_embedding_metadata_frame(
    data: dict,
    *,
    ontology=None,
    strict_ids: bool = False,
    label_map: dict[str, str] | None = None,
    canonical_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Materialize row-level metadata for an embedding pickle."""
    n_rows = len(data["cell_types"])
    row_ids = list(data.get("row_ids", [str(i) for i in range(n_rows)]))
    if len(row_ids) != n_rows:
        raise ValueError(f"row_ids length mismatch: {len(row_ids)} != {n_rows}")

    cell_type_names = [None if x is None else str(x) for x in data["cell_types"]]
    cell_type_ids = data.get("cell_type_ids")
    if cell_type_ids is None:
        if label_map is not None:
            cell_type_ids = [label_map.get(name) for name in cell_type_names]
        else:
            cell_type_ids = [None] * n_rows

        unresolved_mask = [cell_type_id is None for cell_type_id in cell_type_ids]
        if any(unresolved_mask) and ontology is not None:
            unresolved_names = [
                cell_type_names[idx]
                for idx, is_unresolved in enumerate(unresolved_mask)
                if is_unresolved
            ]
            resolved_unmapped = ontology.resolve_labels_to_ids(
                unresolved_names,
                strict=strict_ids,
            )
            unresolved_iter = iter(resolved_unmapped)
            cell_type_ids = [
                next(unresolved_iter) if is_unresolved else cell_type_id
                for cell_type_id, is_unresolved in zip(cell_type_ids, unresolved_mask)
            ]
    else:
        cell_type_ids = [None if x is None else str(x) for x in cell_type_ids]

    sampled_obs_indices = data.get("sampled_obs_indices")
    kept_row_indices = data.get("kept_row_indices")
    final_sampled_indices = _resolve_final_sampled_indices(
        n_rows=n_rows,
        sampled_obs_indices=sampled_obs_indices,
        kept_row_indices=kept_row_indices,
    )

    dataset_name = data.get("dataset_name", data.get("source", "unknown"))

    frame = pd.DataFrame(
        {
            "row_id": [str(x) for x in row_ids],
            "cell_type": cell_type_names,
            "cell_type_id": cell_type_ids,
            "sampled_obs_index": final_sampled_indices,
            "dataset_name": [dataset_name] * n_rows,
        }
    )
    if ontology is not None:
        frame["canonical_cell_type"] = [
            _resolve_display_name(
                ontology=ontology,
                ontology_id=cell_type_id,
                fallback_name=canonical_name_map.get(cell_type_id) if canonical_name_map else cell_type_name,
            )
            if cell_type_id
            else cell_type_name
            for cell_type_id, cell_type_name in zip(frame["cell_type_id"], frame["cell_type"])
        ]
    return frame


def build_bridge_prototypes_from_embeddings(
    data: dict,
    *,
    ontology,
    label_map: dict[str, str] | None = None,
    canonical_name_map: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Build TS bridge centroids keyed by CL ID."""
    metadata = get_embedding_metadata_frame(
        data,
        ontology=ontology,
        strict_ids=True,
        label_map=label_map,
        canonical_name_map=canonical_name_map,
    )
    embeddings = np.asarray(data["embeddings"])
    label_ids = metadata["cell_type_id"].tolist()
    if any(label_id is None for label_id in label_ids):
        unresolved = sorted(set(metadata.loc[metadata["cell_type_id"].isna(), "cell_type"]))
        raise ValueError(
            "Bridge embeddings still contain unresolved cell_type_id values: "
            f"{unresolved[:10]}"
        )
    prototypes = build_prototypes(embeddings, np.asarray(label_ids))

    label_table = (
        metadata.assign(
            bridge_label_name=[
                _resolve_display_name(
                    ontology=ontology,
                    ontology_id=cell_type_id,
                    fallback_name=canonical_cell_type or cell_type,
                )
                for cell_type_id, canonical_cell_type, cell_type in zip(
                    metadata["cell_type_id"],
                    metadata.get("canonical_cell_type", metadata["cell_type"]),
                    metadata["cell_type"],
                )
            ],
        )
        .groupby(["cell_type_id", "bridge_label_name"], dropna=False)
        .size()
        .reset_index(name="n_bridge_cells")
        .rename(columns={"cell_type_id": "bridge_label_id"})
        .sort_values(["bridge_label_name", "bridge_label_id"])
        .reset_index(drop=True)
    )
    return prototypes, label_table, metadata


def compute_query_to_bridge_similarity(
    query_embeddings: np.ndarray,
    bridge_prototypes: dict[str, np.ndarray],
    *,
    temperature: float = 0.1,
) -> tuple[np.ndarray, list[str]]:
    """Compute query-to-bridge scaled cosine similarity."""
    return compute_S_cell(query_embeddings, bridge_prototypes, temperature=temperature)


def _rank_normalize_profiles(profiles: np.ndarray) -> np.ndarray:
    """Convert profile rows into L2-normalized rank vectors for Spearman scoring."""
    profile_array = np.asarray(profiles, dtype=np.float32)
    if profile_array.ndim != 2:
        raise ValueError(f"Expected a 2D profile matrix, got shape {profile_array.shape}.")
    if profile_array.shape[1] == 0:
        raise ValueError("Profile matrices must contain at least one bridge dimension.")

    ranked = stats.rankdata(profile_array, axis=1, method="average").astype(np.float32, copy=False)
    ranked -= ranked.mean(axis=1, keepdims=True)

    norms = np.linalg.norm(ranked, axis=1, keepdims=True)
    normalized = np.divide(
        ranked,
        norms,
        out=np.zeros_like(ranked, dtype=np.float32),
        where=norms > 0,
    )
    return normalized


def compute_rank_correlation_score_matrix(
    query_profiles: np.ndarray,
    candidate_profiles: np.ndarray,
    *,
    row_ids: list[str] | None = None,
    candidate_ids: list[str] | None = None,
    batch_size: int = 1024,
    return_dataframe: bool = True,
) -> pd.DataFrame | np.ndarray:
    """Compute a full query-by-candidate Spearman score matrix in batches."""
    query_profiles = np.asarray(query_profiles, dtype=np.float32)
    candidate_profiles = np.asarray(candidate_profiles, dtype=np.float32)

    if query_profiles.ndim != 2 or candidate_profiles.ndim != 2:
        raise ValueError(
            "query_profiles and candidate_profiles must both be 2D matrices: "
            f"{query_profiles.shape} vs {candidate_profiles.shape}"
        )
    if query_profiles.shape[1] != candidate_profiles.shape[1]:
        raise ValueError(
            "Query and candidate profile matrices must share the bridge axis: "
            f"{query_profiles.shape} vs {candidate_profiles.shape}"
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    query_ranked = _rank_normalize_profiles(query_profiles)
    candidate_ranked = _rank_normalize_profiles(candidate_profiles)

    score_matrix = np.empty(
        (query_ranked.shape[0], candidate_ranked.shape[0]),
        dtype=np.float32,
    )
    for start in range(0, query_ranked.shape[0], batch_size):
        end = min(start + batch_size, query_ranked.shape[0])
        score_matrix[start:end] = query_ranked[start:end] @ candidate_ranked.T

    if not return_dataframe:
        return score_matrix

    resolved_row_ids = row_ids or [str(i) for i in range(score_matrix.shape[0])]
    resolved_candidate_ids = candidate_ids or [str(i) for i in range(score_matrix.shape[1])]
    return pd.DataFrame(
        score_matrix,
        index=pd.Index([str(row_id) for row_id in resolved_row_ids], name="row_id"),
        columns=[str(candidate_id) for candidate_id in resolved_candidate_ids],
    )


def _prepare_neighborhood_target_frame(
    query_metadata: pd.DataFrame,
    *,
    candidate_ids: list[str],
    ontology,
    include_truth_in_targets: bool = False,
) -> pd.DataFrame:
    """Resolve neighborhood target columns per query row.

    When ``include_truth_in_targets`` is True, the truth label itself is also
    counted as a positive — Hit@k then rewards predicting truth at rank 1, and
    ROC-AUC measures whether truth-or-relatives outrank the rest.
    """
    required = {"row_id", "cell_type_id"}
    missing = required - set(query_metadata.columns)
    if missing:
        raise ValueError(
            "query_metadata is missing required columns for neighborhood metrics: "
            f"{sorted(missing)}"
        )

    candidate_id_to_index = {
        str(candidate_id): idx for idx, candidate_id in enumerate(candidate_ids)
    }
    frame = query_metadata.copy().reset_index(drop=True)
    if "cell_type" not in frame.columns:
        frame["cell_type"] = [None] * len(frame)

    neighborhood_target_ids: list[list[str]] = []
    available_target_ids: list[list[str]] = []
    available_target_indices: list[list[int]] = []
    skip_reasons: list[str] = []

    for truth_id in frame["cell_type_id"].astype(str).tolist():
        parent_ids = []
        if hasattr(ontology, "get_parents"):
            parent_ids = [str(parent_id) for parent_id in ontology.get_parents(truth_id)]

        target_ids = list(parent_ids)
        if hasattr(ontology, "get_children"):
            for parent_id in parent_ids:
                target_ids.extend(str(child_id) for child_id in ontology.get_children(parent_id))
        if include_truth_in_targets and truth_id:
            target_ids.append(truth_id)

        deduplicated_target_ids = [
            target_id
            for target_id in dict.fromkeys(target_ids)
            if target_id and (include_truth_in_targets or target_id != truth_id)
        ]
        filtered_target_ids = [
            target_id
            for target_id in deduplicated_target_ids
            if target_id in candidate_id_to_index
        ]

        neighborhood_target_ids.append(deduplicated_target_ids)
        available_target_ids.append(filtered_target_ids)
        available_target_indices.append(
            [candidate_id_to_index[target_id] for target_id in filtered_target_ids]
        )

        if not deduplicated_target_ids:
            skip_reasons.append("no_neighborhood_targets")
        elif not filtered_target_ids:
            skip_reasons.append("no_target_columns")
        else:
            skip_reasons.append("")

    frame["neighborhood_target_ids"] = neighborhood_target_ids
    frame["neighborhood_target_count"] = [len(ids) for ids in neighborhood_target_ids]
    frame["neighborhood_available_target_ids"] = available_target_ids
    frame["neighborhood_available_target_count"] = [len(ids) for ids in available_target_ids]
    frame["neighborhood_target_indices"] = available_target_indices
    frame["neighborhood_skip_reason"] = skip_reasons
    frame["neighborhood_has_target"] = frame["neighborhood_target_indices"].map(bool)
    return frame


def _compute_neighborhood_metrics_from_scores(
    score_values: np.ndarray,
    target_index_lists: list[list[int]],
    *,
    hit_ks: tuple[int, ...] = (3, 5),
    initial_skip_reasons: list[str] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], list[str]]:
    """Compute per-cell neighborhood ROC-AUC and hit rates from score rows."""
    score_values = np.asarray(score_values, dtype=np.float32)
    if score_values.ndim != 2:
        raise ValueError(f"Expected a 2D score matrix, got shape {score_values.shape}.")

    hit_ks = tuple(sorted({int(k) for k in hit_ks if int(k) > 0}))
    n_rows, n_classes = score_values.shape
    auc_values = np.full(n_rows, np.nan, dtype=np.float32)
    hit_values = {
        k: np.full(n_rows, np.nan, dtype=np.float32)
        for k in hit_ks
    }
    skip_reasons = list(initial_skip_reasons or [""] * n_rows)

    if n_classes == 0:
        return auc_values, hit_values, ["no_score_columns"] * n_rows

    for row_idx, target_indices in enumerate(target_index_lists):
        if not target_indices:
            if not skip_reasons[row_idx]:
                skip_reasons[row_idx] = "no_target_columns"
            continue

        target_mask = np.zeros(n_classes, dtype=bool)
        target_mask[target_indices] = True
        n_positive = int(target_mask.sum())
        n_negative = int(n_classes - n_positive)
        if n_positive == 0:
            skip_reasons[row_idx] = "no_target_columns"
            continue
        if n_negative == 0:
            skip_reasons[row_idx] = "no_negative_columns"
            continue

        row_scores = score_values[row_idx]
        auc_values[row_idx] = float(roc_auc_score(target_mask.astype(np.int8), row_scores))

        ranked_indices = np.argsort(row_scores)[::-1]
        for k in hit_ks:
            top_indices = ranked_indices[: min(k, n_classes)]
            hit_values[k][row_idx] = float(np.any(target_mask[top_indices]))

        skip_reasons[row_idx] = ""

    return auc_values, hit_values, skip_reasons


def _summarize_neighborhood_cell_metrics(
    target_frame: pd.DataFrame,
    auc_values: np.ndarray,
    hit_values: dict[int, np.ndarray],
    skip_reasons: list[str],
) -> dict[str, object]:
    """Aggregate per-cell neighborhood metrics into summary outputs."""
    cell_table = target_frame.copy()
    cell_table["neighborhood_roc_auc"] = auc_values
    for k, values in sorted(hit_values.items()):
        cell_table[f"neighborhood_hit_at_{k}"] = values
    cell_table["neighborhood_scored"] = cell_table["neighborhood_roc_auc"].notna()
    cell_table["neighborhood_skip_reason"] = [
        reason if reason else None
        for reason in skip_reasons
    ]

    scored_mask = cell_table["neighborhood_scored"].to_numpy()
    summary = {
        "neighborhood_roc_auc": None,
        "neighborhood_n_scored_cells": int(scored_mask.sum()),
        "neighborhood_n_skipped_cells": int((~scored_mask).sum()),
        "cell_table": cell_table,
    }

    if scored_mask.any():
        summary["neighborhood_roc_auc"] = float(
            cell_table.loc[scored_mask, "neighborhood_roc_auc"].mean()
        )
        for k in sorted(hit_values):
            summary[f"neighborhood_hit_rate_at_{k}"] = float(
                cell_table.loc[scored_mask, f"neighborhood_hit_at_{k}"].mean()
            )
    else:
        for k in sorted(hit_values):
            summary[f"neighborhood_hit_rate_at_{k}"] = None

    return summary


def compute_neighborhood_metrics(
    score_matrix: pd.DataFrame | np.ndarray,
    query_metadata: pd.DataFrame,
    ontology,
    *,
    candidate_ids: list[str] | None = None,
    hit_ks: tuple[int, ...] = (3, 5),
    include_truth_in_targets: bool = False,
) -> dict[str, object]:
    """Compute neighborhood ROC-AUC and hit rates from a score matrix."""
    query_table = query_metadata.copy().reset_index(drop=True)

    if isinstance(score_matrix, pd.DataFrame):
        score_frame = pd.DataFrame(score_matrix).copy()
        score_frame.index = score_frame.index.map(str)
        score_frame.columns = score_frame.columns.map(str)

        if "row_id" not in query_table.columns:
            raise ValueError(
                "query_metadata must include a 'row_id' column for DataFrame score matrices."
            )

        query_row_ids = query_table["row_id"].astype(str)
        aligned_scores = score_frame.reindex(query_row_ids)
        missing_mask = aligned_scores.isna().all(axis=1).to_numpy()
        missing_row_ids = query_row_ids[missing_mask].unique().tolist()
        if missing_row_ids:
            raise KeyError(
                "score_matrix is missing rows for query row_ids while computing neighborhood metrics: "
                f"{missing_row_ids[:10]}"
            )
        if aligned_scores.isna().any().any():
            raise ValueError("score_matrix contains missing values after neighborhood alignment.")

        candidate_ids = aligned_scores.columns.astype(str).tolist()
        score_values = aligned_scores.to_numpy(dtype=np.float32, copy=False)
    else:
        score_values = np.asarray(score_matrix, dtype=np.float32)
        if score_values.ndim != 2:
            raise ValueError(f"Expected a 2D score matrix, got shape {score_values.shape}.")
        if score_values.shape[0] != len(query_table):
            raise ValueError(
                "score_matrix row count must match query_metadata row count: "
                f"{score_values.shape[0]} != {len(query_table)}"
            )
        if candidate_ids is None:
            raise ValueError("candidate_ids must be provided when score_matrix is an ndarray.")
        candidate_ids = [str(candidate_id) for candidate_id in candidate_ids]

    target_frame = _prepare_neighborhood_target_frame(
        query_table,
        candidate_ids=candidate_ids,
        ontology=ontology,
        include_truth_in_targets=include_truth_in_targets,
    )
    auc_values, hit_values, skip_reasons = _compute_neighborhood_metrics_from_scores(
        score_values,
        target_frame["neighborhood_target_indices"].tolist(),
        hit_ks=hit_ks,
        initial_skip_reasons=target_frame["neighborhood_skip_reason"].tolist(),
    )
    return _summarize_neighborhood_cell_metrics(
        target_frame,
        auc_values,
        hit_values,
        skip_reasons,
    )


def compute_neighborhood_metrics_from_rank_profiles(
    query_profiles: np.ndarray,
    candidate_profiles: np.ndarray,
    candidate_ids: list[str],
    query_metadata: pd.DataFrame,
    ontology,
    *,
    batch_size: int = 1024,
    hit_ks: tuple[int, ...] = (3, 5),
    include_truth_in_targets: bool = False,
) -> dict[str, object]:
    """Compute neighborhood metrics from query and candidate rank profiles."""
    query_profiles = np.asarray(query_profiles, dtype=np.float32)
    candidate_profiles = np.asarray(candidate_profiles, dtype=np.float32)

    if query_profiles.ndim != 2 or candidate_profiles.ndim != 2:
        raise ValueError(
            "query_profiles and candidate_profiles must both be 2D matrices: "
            f"{query_profiles.shape} vs {candidate_profiles.shape}"
        )
    if query_profiles.shape[1] != candidate_profiles.shape[1]:
        raise ValueError(
            "Query and candidate profile matrices must share the bridge axis: "
            f"{query_profiles.shape} vs {candidate_profiles.shape}"
        )
    if query_profiles.shape[0] != len(query_metadata):
        raise ValueError(
            "query_profiles row count must match query_metadata row count: "
            f"{query_profiles.shape[0]} != {len(query_metadata)}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    candidate_ids = [str(candidate_id) for candidate_id in candidate_ids]
    target_frame = _prepare_neighborhood_target_frame(
        query_metadata,
        candidate_ids=candidate_ids,
        ontology=ontology,
        include_truth_in_targets=include_truth_in_targets,
    )

    query_ranked = _rank_normalize_profiles(query_profiles)
    candidate_ranked = _rank_normalize_profiles(candidate_profiles)
    auc_values = np.full(query_ranked.shape[0], np.nan, dtype=np.float32)
    hit_values = {
        k: np.full(query_ranked.shape[0], np.nan, dtype=np.float32)
        for k in tuple(sorted({int(k) for k in hit_ks if int(k) > 0}))
    }
    skip_reasons = target_frame["neighborhood_skip_reason"].tolist()
    target_index_lists = target_frame["neighborhood_target_indices"].tolist()

    for start in range(0, query_ranked.shape[0], batch_size):
        end = min(start + batch_size, query_ranked.shape[0])
        batch_scores = query_ranked[start:end] @ candidate_ranked.T
        batch_auc_values, batch_hit_values, batch_skip_reasons = _compute_neighborhood_metrics_from_scores(
            batch_scores,
            target_index_lists[start:end],
            hit_ks=tuple(hit_values),
            initial_skip_reasons=skip_reasons[start:end],
        )
        auc_values[start:end] = batch_auc_values
        for k in hit_values:
            hit_values[k][start:end] = batch_hit_values[k]
        skip_reasons[start:end] = batch_skip_reasons

    return _summarize_neighborhood_cell_metrics(
        target_frame,
        auc_values,
        hit_values,
        skip_reasons,
    )


def derive_observed_label_table(
    rows: pd.DataFrame,
    *,
    ontology=None,
) -> pd.DataFrame:
    """Build an observed row-level label table from dataset metadata."""
    required = {"cell_type", "cell_type_id"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Row metadata is missing required columns: {sorted(missing)}")

    table = (
        rows.groupby(["cell_type_id", "cell_type"], dropna=False)
        .size()
        .reset_index(name="n_cells")
        .sort_values(["cell_type", "cell_type_id"])
        .reset_index(drop=True)
    )
    if ontology is not None:
        table["canonical_cell_type"] = [
            _resolve_display_name(
                ontology=ontology,
                ontology_id=cell_type_id if pd.notna(cell_type_id) else None,
                fallback_name=cell_type,
            )
            for cell_type_id, cell_type in zip(table["cell_type_id"], table["cell_type"])
        ]
    return table


def select_candidate_label_table(
    query_label_rows: pd.DataFrame,
    bridge_label_ids: list[str],
    *,
    ontology=None,
    strict_unseen: bool = True,
) -> pd.DataFrame:
    """Select query candidate labels, optionally excluding bridge overlaps."""
    table = derive_observed_label_table(query_label_rows, ontology=ontology)
    bridge_label_ids = {str(x) for x in bridge_label_ids if x is not None}
    table["seen_in_bridge"] = table["cell_type_id"].astype(str).isin(bridge_label_ids)
    if strict_unseen:
        table = table.loc[~table["seen_in_bridge"]].copy()
    return table.reset_index(drop=True)


def assemble_prediction_table(
    query_metadata: pd.DataFrame,
    *,
    predicted_ids: list[str],
    scores: list[float],
    method: str,
    model_name: str,
    ontology,
    predicted_name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Assemble a row-level prediction table for display or export."""
    if len(query_metadata) != len(predicted_ids) or len(query_metadata) != len(scores):
        raise ValueError("Prediction table inputs must have matching row counts.")

    table = query_metadata.copy().reset_index(drop=True)
    if "sampled_obs_index" not in table.columns:
        table["sampled_obs_index"] = [None] * len(table)
    table["model"] = model_name
    table["method"] = method
    table["truth_cell_type_name"] = table["cell_type"]
    table["truth_cell_type_id"] = table["cell_type_id"]
    table["truth_canonical_name"] = [
        _resolve_display_name(
            ontology=ontology,
            ontology_id=cell_type_id,
            fallback_name=cell_type_name,
        )
        if cell_type_id
        else cell_type_name
        for cell_type_id, cell_type_name in zip(
            table["truth_cell_type_id"],
            table["truth_cell_type_name"],
        )
    ]
    table["predicted_cell_type_id"] = [str(x) if x is not None else None for x in predicted_ids]
    table["predicted_cell_type_name"] = [
        _resolve_display_name(
            ontology=ontology,
            ontology_id=pred_id,
            fallback_name=predicted_name_map.get(pred_id) if predicted_name_map else pred_id,
        )
        if pred_id
        else None
        for pred_id in predicted_ids
    ]
    table["score"] = scores
    return table[
        [
            "model",
            "method",
            "dataset_name",
            "row_id",
            "sampled_obs_index",
            "truth_cell_type_id",
            "truth_cell_type_name",
            "truth_canonical_name",
            "predicted_cell_type_id",
            "predicted_cell_type_name",
            "score",
        ]
    ]


def assemble_prediction_table_from_score_matrix(
    query_metadata: pd.DataFrame,
    *,
    score_matrix: pd.DataFrame,
    method: str,
    model_name: str,
    ontology,
    predicted_name_map: dict[str, str] | None = None,
    candidate_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble a prediction table from a row-aligned score matrix."""
    if "row_id" not in query_metadata.columns:
        raise ValueError("query_metadata must include a 'row_id' column.")

    score_frame = pd.DataFrame(score_matrix).copy()
    if not score_frame.index.is_unique:
        raise ValueError("score_matrix index must be unique.")

    score_frame.index = score_frame.index.map(str)
    query_table = query_metadata.copy().reset_index(drop=True)
    query_row_ids = query_table["row_id"].astype(str)
    aligned_scores = score_frame.reindex(query_row_ids)

    missing_mask = aligned_scores.isna().all(axis=1).to_numpy()
    missing_row_ids = query_row_ids[missing_mask].unique().tolist()
    if missing_row_ids:
        raise KeyError(
            "score_matrix is missing rows for query row_ids: "
            f"{missing_row_ids[:10]}"
        )

    if candidate_ids is not None:
        candidate_ids = [str(candidate_id) for candidate_id in candidate_ids]
        missing_candidate_ids = [
            candidate_id for candidate_id in candidate_ids
            if candidate_id not in aligned_scores.columns
        ]
        if missing_candidate_ids:
            raise KeyError(
                "score_matrix is missing candidate columns: "
                f"{missing_candidate_ids[:10]}"
            )
        aligned_scores = aligned_scores.loc[:, candidate_ids]

    if aligned_scores.shape[1] == 0:
        raise ValueError("score_matrix must contain at least one prediction column.")
    if aligned_scores.isna().any().any():
        raise ValueError("score_matrix contains missing values after alignment.")

    predicted_ids = aligned_scores.idxmax(axis=1).astype(str).tolist()
    scores = aligned_scores.max(axis=1).astype(float).tolist()
    return assemble_prediction_table(
        query_table,
        predicted_ids=predicted_ids,
        scores=scores,
        method=method,
        model_name=model_name,
        ontology=ontology,
        predicted_name_map=predicted_name_map,
    )


def run_hector_native_predictions(
    query_adata,
    query_metadata: pd.DataFrame,
    *,
    hector_predictor,
    candidate_ids: list[str],
    ontology,
    ontology_path: str | Path,
    expected_query_counts: pd.Series,
    strict_unseen: bool = True,
    model_name: str = "Hector",
    candidate_name_map: dict[str, str] | None = None,
    neighborhood_candidate_ids: list[str] | None = None,
    include_truth_in_neighborhood: bool = False,
) -> dict[str, object]:
    """Run Hector native inference and assemble only the native tables it can support."""
    if "row_id" not in query_metadata.columns:
        raise ValueError("query_metadata must include a 'row_id' column.")

    row_id_column = resolve_query_obs_row_id_column(
        query_adata.obs,
        query_metadata["row_id"],
    )
    predictions_df, score_matrix_df = hector_predictor.predict(
        query_adata,
        label_format="id",
        export_score_matrix=True,
        cell_id_column=row_id_column,
    )

    score_matrix_df = pd.DataFrame(score_matrix_df).copy()
    score_matrix_df.index = score_matrix_df.index.map(str)
    score_matrix_df.columns = score_matrix_df.columns.map(str)

    query_table = query_metadata.copy()
    query_table["row_id"] = query_table["row_id"].astype(str)

    candidate_ids = [str(candidate_id) for candidate_id in candidate_ids]
    if strict_unseen:
        eval_mask = query_table["cell_type_id"].astype(str).isin(candidate_ids)
    else:
        eval_mask = query_table["cell_type_id"].notna()
    query_metadata_eval = query_table.loc[eval_mask].reset_index(drop=True)

    result = {
        "predictions_df": predictions_df,
        "score_matrix_df": score_matrix_df,
        "row_id_column": row_id_column,
        "query_metadata_eval": query_metadata_eval,
        "requested_candidate_ids": candidate_ids,
        "supported_candidate_ids": [],
        "missing_candidate_ids": [],
        "open_table": None,
        "open_summary": None,
        "closed_table": None,
        "closed_summary": None,
        "summary_rows": [],
        "status_rows": [],
    }

    if len(query_metadata_eval) == 0:
        skip_reason = "no query rows after candidate filtering"
        result["status_rows"] = [
            {
                "model": "Hector Native Open-Set",
                "status": "skipped",
                "reason": skip_reason,
                "n_query_rows": 0,
            },
            {
                "model": "Hector Native Closed-Set",
                "status": "skipped",
                "reason": skip_reason,
                "n_query_rows": 0,
            },
        ]
        return result

    open_table = assemble_prediction_table_from_score_matrix(
        query_metadata_eval,
        score_matrix=score_matrix_df,
        method="native_open_set",
        model_name=model_name,
        ontology=ontology,
        predicted_name_map=getattr(hector_predictor, "id_to_name_map", None),
    )
    open_table = annotate_prediction_table(open_table, ontology_path=ontology_path)
    open_summary = summarize_prediction_table(
        open_table, expected_query_counts=expected_query_counts,
    )

    available_columns = set(score_matrix_df.columns.astype(str))
    if neighborhood_candidate_ids is None:
        neighborhood_score_matrix = score_matrix_df
        neighborhood_candidate_ids_used = list(score_matrix_df.columns.astype(str))
    else:
        neighborhood_candidate_ids_used = [
            str(cid) for cid in neighborhood_candidate_ids
            if str(cid) in available_columns
        ]
        neighborhood_score_matrix = score_matrix_df.loc[:, neighborhood_candidate_ids_used]
    native_neighborhood_summary = compute_neighborhood_metrics(
        neighborhood_score_matrix,
        query_metadata_eval,
        ontology,
        candidate_ids=neighborhood_candidate_ids_used,
        include_truth_in_targets=include_truth_in_neighborhood,
    )
    open_summary["neighborhood_summary"] = {
        key: value
        for key, value in native_neighborhood_summary.items()
        if key != "cell_table"
    }

    available_candidate_ids = set(score_matrix_df.columns.astype(str))
    supported_candidate_ids = [
        candidate_id for candidate_id in candidate_ids if candidate_id in available_candidate_ids
    ]
    missing_candidate_ids = [
        candidate_id for candidate_id in candidate_ids if candidate_id not in available_candidate_ids
    ]

    result["supported_candidate_ids"] = supported_candidate_ids
    result["missing_candidate_ids"] = missing_candidate_ids
    result["neighborhood_candidate_ids_used"] = neighborhood_candidate_ids_used
    result["open_table"] = open_table
    result["open_summary"] = open_summary
    result["open_neighborhood_metric_table"] = native_neighborhood_summary["cell_table"]
    result["summary_rows"] = [flatten_summary_row(model_name, "native_open_set", open_summary)]
    result["status_rows"] = [
        {
            "model": "Hector Native Open-Set",
            "status": "ready",
            "reason": "",
            "n_query_rows": len(query_metadata_eval),
        }
    ]

    if missing_candidate_ids:
        missing_preview = ", ".join(missing_candidate_ids[:10])
        closed_reason = (
            f"{len(missing_candidate_ids)} of {len(candidate_ids)} candidate IDs are absent from "
            f"Hector native score_matrix columns: {missing_preview}"
        )
        result["status_rows"].append(
            {
                "model": "Hector Native Closed-Set",
                "status": "skipped",
                "reason": closed_reason,
                "n_query_rows": len(query_metadata_eval),
            }
        )
        return result

    closed_table = assemble_prediction_table_from_score_matrix(
        query_metadata_eval,
        score_matrix=score_matrix_df,
        method="native_closed_set",
        model_name=model_name,
        ontology=ontology,
        predicted_name_map=candidate_name_map,
        candidate_ids=candidate_ids,
    )
    closed_table = annotate_prediction_table(closed_table, ontology_path=ontology_path)
    closed_summary = summarize_prediction_table(
        closed_table, expected_query_counts=expected_query_counts,
    )
    closed_summary["neighborhood_summary"] = {
        key: value
        for key, value in native_neighborhood_summary.items()
        if key != "cell_table"
    }

    result["closed_table"] = closed_table
    result["closed_summary"] = closed_summary
    result["closed_neighborhood_metric_table"] = native_neighborhood_summary["cell_table"].copy()
    result["summary_rows"].append(flatten_summary_row(model_name, "native_closed_set", closed_summary))
    result["status_rows"].append(
        {
            "model": "Hector Native Closed-Set",
            "status": "ready",
            "reason": "",
            "n_query_rows": len(query_metadata_eval),
        }
    )
    return result


def annotate_prediction_table(
    prediction_table: pd.DataFrame,
    *,
    ontology_path: str | Path,
) -> pd.DataFrame:
    """Attach ontology coarse/fine/hierarchical metrics to a prediction table."""
    pred_ids = prediction_table["predicted_cell_type_id"]
    truth_ids = prediction_table["truth_cell_type_id"]

    ontology_match = classify_ontology_matches(pred_ids, truth_ids, obofile=ontology_path)
    fine_distance = compute_fine_ontology_distance(pred_ids, truth_ids, obofile=ontology_path)
    hierarchical = compute_hierarchical_metrics(pred_ids, truth_ids, obofile=ontology_path)

    annotated = prediction_table.copy()
    annotated["ontology_match"] = ontology_match
    annotated["fine_ontology_distance"] = fine_distance
    annotated = pd.concat([annotated, hierarchical], axis=1)
    return annotated


def summarize_prediction_table(
    prediction_table: pd.DataFrame,
    *,
    expected_query_counts: pd.Series,
) -> dict[str, object]:
    """Summarize flat and hierarchical metrics for a prediction table.

    Padded first to the full query set (query_set_padding.pad_to_query_set): a
    query cell a model never returned a prediction for is scored as an error in
    every metric here, not silently averaged out of the cells it chose to
    answer. scCello's tokenizer is the model this matters for today, but any
    model with incomplete coverage gets the same treatment.

    ``prediction_table`` must already carry the per-row hierarchical columns
    ``annotate_prediction_table`` adds — the summary is read from those rather
    than recomputed from ``predicted_cell_type_id``/``truth_cell_type_id`` so
    that padded rows (whose ``predicted_cell_type_id`` is a sentinel, not a
    real ontology term) never have to pass through ontology-label validation.
    """
    padded = pad_to_query_set(prediction_table, expected_query_counts)
    truth_ids = padded["truth_cell_type_id"].astype(str).tolist()
    pred_ids = padded["predicted_cell_type_id"].astype(str).tolist()
    flat = evaluate_predictions(truth_ids, pred_ids, labels=sorted(set(truth_ids)))

    hierarchical_micro_macro = summarize_hierarchical_metrics(
        metrics=padded[list(HIERARCHICAL_METRIC_COLUMNS)],
        group_labels=padded["truth_cell_type_id"].astype(str),
        averaging="both",
    )
    return {
        "accuracy": flat["accuracy"],
        "macro_f1": flat["macro_f1"],
        "weighted_f1": flat["weighted_f1"],
        "errors": flat["errors"],
        "hierarchical_summary": hierarchical_micro_macro,
    }


def _resolve_display_name(
    *,
    ontology,
    ontology_id: str | None,
    fallback_name: str | None,
) -> str | None:
    if ontology_id is None:
        return fallback_name

    if hasattr(ontology, "get_name"):
        display_name = ontology.get_name(str(ontology_id))
        if display_name == ontology_id and fallback_name:
            return fallback_name
        return display_name

    if isinstance(ontology, nx.Graph) and ontology.has_node(str(ontology_id)):
        node_attrs = ontology.nodes[str(ontology_id)]
        display_name = (
            node_attrs.get("name")
            or node_attrs.get("label")
            or node_attrs.get("term")
            or str(ontology_id)
        )
        if display_name == ontology_id and fallback_name:
            return fallback_name
        return display_name

    return fallback_name or str(ontology_id)


def read_h5ad_observation_metadata(
    path: str | Path,
    *,
    conda_env: str = "sc",
) -> pd.DataFrame:
    """Read row-level label metadata from an h5ad file.

    Uses local ``h5py`` if available; otherwise falls back to the existing
    ``sc`` environment through ``conda run``.
    """
    path = Path(path).expanduser().resolve()
    try:
        import h5py  # type: ignore
    except ImportError:
        data = _read_h5ad_obs_metadata_via_subprocess(path, conda_env=conda_env)
    else:
        data = _read_h5ad_obs_metadata_via_h5py(path, h5py_module=h5py)

    row_id = _choose_row_id_column(data)
    frame_data = {
        "row_id": row_id,
        "cell_type": data.get("cell_type", []),
        "cell_type_id": data.get("cell_type_ontology_term_id", []),
        "dataset_name": [path.name] * len(row_id),
    }
    for key in ("organism", "assay", "is_primary_data"):
        values = data.get(key)
        if values is not None:
            frame_data[key] = values

    frame = pd.DataFrame(frame_data)
    return frame


def build_exact_label_map(rows: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """Build exact name<->ID maps from observed row-level pairs."""
    required = {"cell_type", "cell_type_id"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Row metadata is missing required columns: {sorted(missing)}")

    pairs = rows.loc[:, ["cell_type", "cell_type_id"]].dropna().drop_duplicates()
    name_counts = pairs.groupby("cell_type")["cell_type_id"].nunique()
    ambiguous_names = sorted(name_counts[name_counts > 1].index.tolist())
    if ambiguous_names:
        raise ValueError(
            "Observed cell_type names map to multiple CL IDs: "
            f"{ambiguous_names[:10]}"
        )

    id_counts = pairs.groupby("cell_type_id")["cell_type"].nunique()
    ambiguous_ids = sorted(id_counts[id_counts > 1].index.tolist())
    if ambiguous_ids:
        raise ValueError(
            "Observed CL IDs map to multiple cell_type names: "
            f"{ambiguous_ids[:10]}"
        )

    name_to_id = dict(pairs.itertuples(index=False, name=None))
    id_to_name = {cell_type_id: cell_type for cell_type, cell_type_id in name_to_id.items()}
    return name_to_id, id_to_name


def resolve_target_n_cells(dataset_path, requested_n_cells):
    """Return *requested_n_cells* if set, otherwise the full dataset row count."""
    if requested_n_cells is not None:
        return int(requested_n_cells)
    return int(len(read_h5ad_observation_metadata(dataset_path)))


def flatten_summary_row(model_name, method_name, summary_dict):
    """Flatten a summarize_prediction_table result into a single dict row."""
    row = {
        "model": model_name,
        "method": method_name,
        "accuracy": summary_dict["accuracy"],
        "macro_f1": summary_dict["macro_f1"],
        "weighted_f1": summary_dict["weighted_f1"],
        "n_errors": len(summary_dict.get("errors", [])),
    }
    neighborhood_summary = summary_dict.get("neighborhood_summary")
    if isinstance(neighborhood_summary, dict):
        for key in (
            "neighborhood_roc_auc",
            "neighborhood_hit_rate_at_3",
            "neighborhood_hit_rate_at_5",
            "neighborhood_n_scored_cells",
            "neighborhood_n_skipped_cells",
        ):
            if key in neighborhood_summary:
                row[key] = neighborhood_summary.get(key)
    hierarchical_summary = summary_dict.get("hierarchical_summary")
    if hierarchical_summary is None:
        return row
    if isinstance(hierarchical_summary, pd.Series):
        for metric_name, metric_value in hierarchical_summary.to_dict().items():
            row[f"micro_{metric_name}"] = metric_value
        return row
    if isinstance(hierarchical_summary, pd.DataFrame):
        for averaging_name, metric_values in hierarchical_summary.iterrows():
            for metric_name, metric_value in metric_values.items():
                row[f"{averaging_name}_{metric_name}"] = metric_value
    return row


def _resolve_final_sampled_indices(
    *,
    n_rows: int,
    sampled_obs_indices,
    kept_row_indices,
) -> list[int | None]:
    if sampled_obs_indices is None:
        return [None] * n_rows

    sampled = [int(x) for x in sampled_obs_indices]
    if len(sampled) == n_rows:
        return sampled

    if kept_row_indices is None:
        raise ValueError(
            "sampled_obs_indices length differs from output row count but "
            "kept_row_indices is missing."
        )

    kept = [int(x) for x in kept_row_indices]
    if len(kept) != n_rows:
        raise ValueError(
            f"kept_row_indices length mismatch: {len(kept)} != {n_rows}"
        )
    return [sampled[idx] for idx in kept]


def _decode_h5ad_column(obs, name: str) -> list[object] | None:
    if name not in obs:
        return None
    obj = obs[name]
    if hasattr(obj, "keys") and "categories" in obj and "codes" in obj:
        categories = [_decode_h5ad_scalar(x) for x in obj["categories"][:]]
        codes = obj["codes"][:]
        return [categories[int(code)] if int(code) >= 0 else None for code in codes]
    return [_decode_h5ad_scalar(x) for x in obj[:]]


def _decode_h5ad_scalar(value) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def _choose_row_id_column(data: dict[str, list[object] | None]) -> list[str]:
    for key in ("observation_joinid", "soma_joinid", "_index"):
        values = data.get(key)
        if values is not None:
            return [str(value) if value is not None else "" for value in values]

    n_rows = len(data.get("cell_type", []))
    return [str(i) for i in range(n_rows)]


def _read_h5ad_obs_metadata_via_h5py(path: Path, *, h5py_module) -> dict[str, list[object] | None]:
    with h5py_module.File(path, "r") as handle:
        obs = handle["obs"]
        return {
            "cell_type": _decode_h5ad_column(obs, "cell_type"),
            "cell_type_ontology_term_id": _decode_h5ad_column(obs, "cell_type_ontology_term_id"),
            "observation_joinid": _decode_h5ad_column(obs, "observation_joinid"),
            "soma_joinid": _decode_h5ad_column(obs, "soma_joinid"),
            "_index": _decode_h5ad_column(obs, "_index"),
            "organism": _decode_h5ad_column(obs, "organism"),
            "assay": _decode_h5ad_column(obs, "assay"),
            "is_primary_data": _decode_h5ad_column(obs, "is_primary_data"),
        }


def _read_h5ad_obs_metadata_via_subprocess(
    path: Path,
    *,
    conda_env: str,
) -> dict[str, list[object] | None]:
    script = r"""
import json
import sys
import h5py

def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value

def decode_column(obs, name):
    if name not in obs:
        return None
    obj = obs[name]
    if hasattr(obj, "keys") and "categories" in obj and "codes" in obj:
        categories = [decode_scalar(x) for x in obj["categories"][:]]
        codes = obj["codes"][:]
        return [categories[int(code)] if int(code) >= 0 else None for code in codes]
    return [decode_scalar(x) for x in obj[:]]

with h5py.File(sys.argv[1], "r") as handle:
    obs = handle["obs"]
    payload = {
        "cell_type": decode_column(obs, "cell_type"),
        "cell_type_ontology_term_id": decode_column(obs, "cell_type_ontology_term_id"),
        "observation_joinid": decode_column(obs, "observation_joinid"),
        "soma_joinid": decode_column(obs, "soma_joinid"),
        "_index": decode_column(obs, "_index"),
        "organism": decode_column(obs, "organism"),
        "assay": decode_column(obs, "assay"),
        "is_primary_data": decode_column(obs, "is_primary_data"),
    }
print(json.dumps(payload))
"""
    result = subprocess.run(
        ["conda", "run", "-n", conda_env, "python", "-c", script, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = result.stdout.strip().splitlines()
    if not stdout:
        raise RuntimeError(f"No metadata returned when reading {path}")
    return json.loads(stdout[-1])


logger = logging.getLogger(__name__)


def compute_hop_distance(
    pred,
    truth,
    *,
    obofile=None,
    dag=None,
) -> pd.Series:
    """Compute unsigned shortest-path distance in the undirected CL DAG.

    Parameters
    ----------
    pred : Iterable[str] | pd.Series | str
        Predicted cell-type labels.
    truth : Iterable[str] | pd.Series | str
        Ground-truth cell-type labels.
    obofile : str | Path | None
        Path to .obo file (mutually exclusive with *dag*).
    dag : nx.DiGraph | None
        Pre-loaded DAG from ``load_cell_ontology()``.

    Returns
    -------
    pd.Series
        Integer hop distances: 0 = exact match, positive = hop count,
        -1 = disconnected sentinel.
    """
    graph = _resolve_matching_dag(obofile=obofile, dag=dag)
    pred_series, truth_series, _scalar = _coerce_pair_inputs(pred, truth)

    if len(pred_series) == 0:
        return pd.Series([], dtype=int, name="hop_distance")

    pred_series = _canonicalize_labels(pred_series, graph, role="Prediction")
    truth_series = _canonicalize_labels(truth_series, graph, role="Ground-truth")
    _validate_labels(pred_series, graph, role="Prediction")
    _validate_labels(truth_series, graph, role="Ground-truth")

    undirected = nx.Graph(graph)
    pair_cache: dict[tuple[str, str], int] = {}
    distances: list[int] = []

    for p, t in zip(pred_series.tolist(), truth_series.tolist()):
        key = (p, t)
        cached = pair_cache.get(key)
        if cached is not None:
            distances.append(cached)
            continue

        if p == t:
            dist = 0
        else:
            try:
                dist = nx.shortest_path_length(undirected, p, t)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                dist = -1
                logger.warning(
                    "No path between %r and %r in the CL DAG; returning -1.",
                    p,
                    t,
                )

        pair_cache[key] = dist
        distances.append(dist)

    return pd.Series(distances, index=pred_series.index, name="hop_distance")


def compute_label_depths(
    labels,
    *,
    obofile=None,
    dag=None,
) -> pd.Series:
    """Compute the depth of each label in the CL DAG (shortest path from root).

    Parameters
    ----------
    labels : Iterable[str] | pd.Series | str
        Cell-type labels.
    obofile : str | Path | None
        Path to .obo file (mutually exclusive with *dag*).
    dag : nx.DiGraph | None
        Pre-loaded DAG from ``load_cell_ontology()``.

    Returns
    -------
    pd.Series
        Integer depths aligned to the input index, with name ``"label_depth"``.
    """
    graph = _resolve_matching_dag(obofile=obofile, dag=dag)
    label_series = _coerce_category_input(labels)
    # Preserve the original index when the caller passes a pd.Series
    if isinstance(labels, pd.Series):
        label_series.index = labels.index
    label_series = _canonicalize_labels(label_series, graph, role="Label")
    _validate_labels(label_series, graph, role="Label")

    root_name = graph.graph.get("root_name", "cell")
    depths: list[int] = [
        _node_depth(graph, lbl, root=root_name) for lbl in label_series.tolist()
    ]
    return pd.Series(depths, index=label_series.index, name="label_depth")


def generate_random_baseline(
    bridge_labels,
    n_samples: int,
    *,
    seed: int = 42,
) -> pd.Series:
    """Generate dummy predictions by sampling from training-set class frequencies.

    Parameters
    ----------
    bridge_labels : pd.Series | list[str]
        Training-set (bridge) cell-type labels whose class frequencies define
        the sampling distribution.
    n_samples : int
        Number of predictions to generate (= query set size).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    pd.Series
        Predicted labels with length *n_samples* and name
        ``"random_baseline_prediction"``.

    Raises
    ------
    ValueError
        If *n_samples* <= 0 or *bridge_labels* is empty.
    """
    bridge_series = pd.Series(bridge_labels) if not isinstance(bridge_labels, pd.Series) else bridge_labels

    if len(bridge_series) == 0:
        raise ValueError("bridge_labels must not be empty.")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")

    freq = bridge_series.value_counts(normalize=True)
    classes = freq.index.to_numpy()
    probabilities = freq.values

    rng = np.random.default_rng(seed)
    sampled = rng.choice(classes, size=n_samples, p=probabilities)

    return pd.Series(sampled, name="random_baseline_prediction")
