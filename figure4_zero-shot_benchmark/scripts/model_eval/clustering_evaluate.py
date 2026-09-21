#!/usr/bin/env python3
"""Complementary clustering benchmark for single-cell model embeddings.

This module evaluates saved cell embeddings against user-selected cell-type
and batch columns from the source h5ad file. It implements the explicit
Leiden-resolution sweep plus scib-metrics computation used by single-cell
integration benchmarks, while remaining model-agnostic.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluate import load_embeddings


DEFAULT_RESOLUTIONS = [round(step / 10, 1) for step in range(1, 21)]
REQUIRED_EVALUATION_RESULT_KEYS = {
    "config",
    "alignment",
    "summary",
    "resolution_sweep",
    "cluster_assignments",
}


def _decode_h5ad_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def _decode_h5ad_column(obs, name: str) -> list[object] | None:
    if name not in obs:
        return None
    obj = obs[name]
    if hasattr(obj, "keys") and "categories" in obj and "codes" in obj:
        categories = [_decode_h5ad_scalar(x) for x in obj["categories"][:]]
        codes = obj["codes"][:]
        return [categories[int(code)] if int(code) >= 0 else None for code in codes]
    return [_decode_h5ad_scalar(x) for x in obj[:]]


def _choose_row_id_column(data: dict[str, list[object] | None]) -> list[str]:
    for key in ("observation_joinid", "soma_joinid", "_index"):
        values = data.get(key)
        if values is not None:
            return [str(value) if value is not None else "" for value in values]

    first_key = next(iter(data), None)
    n_rows = len(data.get(first_key, [])) if first_key is not None else 0
    return [str(i) for i in range(n_rows)]


def _parse_resolutions(text: str) -> list[float]:
    values = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(float(piece))
    if not values:
        raise ValueError("Resolution list is empty.")
    return values


def _resolve_final_sampled_indices(
    n_rows: int,
    sampled_obs_indices,
    kept_row_indices,
) -> list[int] | None:
    if sampled_obs_indices is None:
        return None

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


def read_h5ad_obs_columns(dataset_path: str | Path, columns: list[str]) -> pd.DataFrame:
    """Read row_id plus requested obs columns directly from an h5ad file."""
    import h5py

    dataset_path = Path(dataset_path).expanduser().resolve()
    required = ["observation_joinid", "soma_joinid", "_index", *columns]
    payload: dict[str, list[object] | None] = {}

    with h5py.File(dataset_path, "r") as handle:
        obs = handle["obs"]
        for column in required:
            payload[column] = _decode_h5ad_column(obs, column)

    missing = [column for column in columns if payload.get(column) is None]
    if missing:
        raise ValueError(
            f"Dataset {dataset_path.name} is missing required obs columns: {sorted(missing)}"
        )

    row_id = _choose_row_id_column(payload)
    source_obs_index = list(range(len(row_id)))
    frame = pd.DataFrame(
        {
            "row_id": row_id,
            "source_obs_index": source_obs_index,
        }
    )
    for column in columns:
        frame[column] = payload[column]

    if frame["row_id"].duplicated().any():
        duplicated = frame.loc[frame["row_id"].duplicated(), "row_id"].astype(str).tolist()
        raise ValueError(f"Source dataset row_id values are not unique: {duplicated[:5]}")

    return frame


def attach_source_metadata(
    embedding_data: dict,
    source_metadata: pd.DataFrame,
    label_col: str,
    batch_col: str,
) -> tuple[pd.DataFrame, str]:
    """Align source metadata to the saved cell-embeddings rows."""
    n_rows = len(embedding_data["embeddings"])
    row_ids = embedding_data.get("row_ids")

    if row_ids is not None:
        if len(row_ids) != n_rows:
            raise ValueError(f"row_ids length mismatch: {len(row_ids)} != {n_rows}")

        source_by_row = source_metadata.copy()
        source_by_row["row_id"] = source_by_row["row_id"].astype(str)
        source_by_row = source_by_row.set_index("row_id", drop=False)
        cell_embeddings_row_ids = [str(value) for value in row_ids]
        missing = [row_id for row_id in cell_embeddings_row_ids if row_id not in source_by_row.index]

        if missing:
            raise ValueError(
                "Saved cell-embeddings row_ids do not align with the requested dataset; "
                f"missing source row_ids include {missing[:5]}"
            )

        joined = source_by_row.loc[cell_embeddings_row_ids].reset_index(drop=True)
        return joined.loc[:, ["row_id", "source_obs_index", label_col, batch_col]], "row_id"

    final_indices = _resolve_final_sampled_indices(
        n_rows=n_rows,
        sampled_obs_indices=embedding_data.get("sampled_obs_indices"),
        kept_row_indices=embedding_data.get("kept_row_indices"),
    )
    if final_indices is None:
        raise ValueError(
            "Could not align source metadata: saved cell embeddings have neither usable row_ids "
            "nor sampled_obs_indices."
        )

    source_by_index = source_metadata.set_index("source_obs_index", drop=False)
    missing = [idx for idx in final_indices if idx not in source_by_index.index]
    if missing:
        raise ValueError(
            "Could not align source metadata by sampled_obs_index; missing source "
            f"indices include {missing[:5]}"
        )

    joined = source_by_index.loc[final_indices].reset_index(drop=True)
    return joined.loc[:, ["row_id", "source_obs_index", label_col, batch_col]], "sampled_obs_index"


def prepare_evaluation_inputs(
    embeddings_path: str | Path,
    dataset_path: str | Path,
    label_col: str,
    batch_col: str,
) -> tuple[dict, np.ndarray, pd.DataFrame, dict]:
    """Load saved cell embeddings and join the requested source metadata columns."""
    embedding_data = load_embeddings(str(embeddings_path))
    embeddings = np.asarray(embedding_data["embeddings"], dtype=np.float32)
    source_metadata = read_h5ad_obs_columns(dataset_path, [label_col, batch_col])
    joined, join_mode = attach_source_metadata(
        embedding_data=embedding_data,
        source_metadata=source_metadata,
        label_col=label_col,
        batch_col=batch_col,
    )

    mask = joined[label_col].notna() & joined[batch_col].notna()
    filtered = joined.loc[mask].copy().reset_index(drop=True)
    filtered[label_col] = filtered[label_col].astype(str)
    filtered[batch_col] = filtered[batch_col].astype(str)
    filtered_embeddings = embeddings[mask.to_numpy()]

    if filtered_embeddings.shape[0] == 0:
        raise ValueError("No rows remain after dropping missing label/batch metadata.")
    if filtered[label_col].nunique() < 2:
        raise ValueError(f"Need at least 2 label groups in {label_col!r} after filtering.")
    if filtered[batch_col].nunique() < 2:
        raise ValueError(f"Need at least 2 batch groups in {batch_col!r} after filtering.")

    alignment = {
        "join_mode": join_mode,
        "n_source_rows": int(len(source_metadata)),
        "n_cell_embeddings_rows": int(embeddings.shape[0]),
        "n_dropped_missing_label_or_batch": int((~mask).sum()),
        "n_eval_rows": int(mask.sum()),
        "n_unique_labels": int(filtered[label_col].nunique()),
        "n_unique_batches": int(filtered[batch_col].nunique()),
    }
    return embedding_data, filtered_embeddings, filtered, alignment


def _build_scib_neighbors(
    embeddings: np.ndarray,
    n_neighbors: int,
    distance_metric: str,
    random_state: int,
):
    """Build a scib-metrics neighbor graph with a Euclidean fast path."""
    from scib_metrics.nearest_neighbors import NeighborsResults, jax_approx_min_k, pynndescent

    if embeddings.shape[0] < 3:
        raise ValueError("Need at least 3 evaluation rows for clustering metrics.")

    effective_n_neighbors = min(int(n_neighbors), int(embeddings.shape[0] - 1))
    effective_n_neighbors = max(2, effective_n_neighbors)
    metric_name = str(distance_metric).lower()
    jax_backend = "not-used"

    if metric_name == "euclidean":
        try:
            import jax

            jax_backend = str(jax.default_backend())
            neighbors = jax_approx_min_k(
                np.asarray(embeddings, dtype=np.float32),
                n_neighbors=effective_n_neighbors,
                chunk_size=min(2048, int(embeddings.shape[0])),
            )
            return neighbors, effective_n_neighbors, "jax_approx_min_k", jax_backend, metric_name
        except Exception:
            jax_backend = "unavailable"
            neighbors = pynndescent(
                np.asarray(embeddings, dtype=np.float32),
                n_neighbors=effective_n_neighbors,
                random_state=int(random_state),
                n_jobs=1,
            )
            return neighbors, effective_n_neighbors, "pynndescent", jax_backend, metric_name

    knn = NearestNeighbors(
        n_neighbors=effective_n_neighbors,
        metric=metric_name,
    )
    knn.fit(np.asarray(embeddings, dtype=np.float32))
    distances, indices = knn.kneighbors(np.asarray(embeddings, dtype=np.float32))
    neighbors = NeighborsResults(indices=indices, distances=distances)
    return neighbors, effective_n_neighbors, "sklearn_nearest_neighbors", jax_backend, metric_name


def _compute_leiden_membership(
    connectivity_graph,
    resolution: float,
    random_state: int,
) -> np.ndarray:
    """Run Leiden directly on the scib-metrics connectivity graph."""
    import igraph

    rng = random.Random(int(random_state))
    igraph.set_random_number_generator(rng)
    graph = igraph.Graph.Weighted_Adjacency(connectivity_graph, mode="directed")
    graph.to_undirected(mode="each")
    clustering = graph.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=float(resolution),
    )
    return np.asarray(clustering.membership, dtype=int)


def run_resolution_sweep(
    neighbors,
    labels: np.ndarray,
    resolutions: list[float],
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sweep Leiden resolutions on a precomputed scib-metrics graph."""
    labels = np.asarray(labels).astype(str)
    connectivity_graph = neighbors.knn_graph_connectivities
    records = []
    clusters_by_key = {}

    for resolution in resolutions:
        cluster_key = f"cluster_res_{str(resolution).replace('.', '_')}"
        clusters = _compute_leiden_membership(
            connectivity_graph,
            resolution=float(resolution),
            random_state=int(random_state),
        )
        clusters_by_key[cluster_key] = clusters
        records.append(
            {
                "resolution": float(resolution),
                "cluster_key": cluster_key,
                "n_clusters": int(pd.Series(clusters).nunique()),
                "NMI_cluster/label": float(
                    normalized_mutual_info_score(
                        labels,
                        clusters,
                        average_method="arithmetic",
                    )
                ),
                "ARI_cluster/label": float(adjusted_rand_score(labels, clusters)),
            }
        )

    resolution_sweep = pd.DataFrame(records).sort_values(
        ["NMI_cluster/label", "ARI_cluster/label", "resolution"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    resolution_sweep["is_best"] = False
    resolution_sweep.loc[0, "is_best"] = True
    best_cluster_key = str(resolution_sweep.loc[0, "cluster_key"])
    return resolution_sweep, clusters_by_key[best_cluster_key]


def compute_benchmark_metrics(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    label_col: str,
    batch_col: str,
    clusters: np.ndarray,
    neighbors,
    distance_metric: str,
) -> dict:
    """Compute final biological and batch-removal metrics."""
    from scib_metrics import graph_connectivity, silhouette_batch, silhouette_label

    labels = metadata[label_col].astype(str).to_numpy()
    batches = metadata[batch_col].astype(str).to_numpy()
    metric_name = str(distance_metric).lower()
    if metric_name not in {"euclidean", "cosine"}:
        raise ValueError(
            f"Unsupported distance metric for scib_metrics silhouettes: {distance_metric!r}. "
            "Use 'euclidean' or 'cosine'."
        )

    chunk_size = min(2048, int(embeddings.shape[0]))
    nmi = float(
        normalized_mutual_info_score(
            labels,
            clusters,
            average_method="arithmetic",
        )
    )
    ari = float(adjusted_rand_score(labels, clusters))
    asw_label = float(
        silhouette_label(
            np.asarray(embeddings, dtype=np.float32),
            labels,
            rescale=True,
            chunk_size=chunk_size,
            metric=metric_name,
        )
    )
    asw_batch = float(
        silhouette_batch(
            np.asarray(embeddings, dtype=np.float32),
            labels,
            batches,
            rescale=True,
            chunk_size=chunk_size,
            metric=metric_name,
        )
    )
    graph_conn = float(graph_connectivity(neighbors, labels))
    avg_bio = float((nmi + ari + asw_label) / 3.0)
    avg_batch = float((asw_batch + graph_conn) / 2.0)
    overall_score = float((0.6 * avg_bio) + (0.4 * avg_batch))
    return {
        "NMI_cluster/label": nmi,
        "ARI_cluster/label": ari,
        "ASW_label": asw_label,
        "ASW_batch": asw_batch,
        "graph_conn": graph_conn,
        "avg_bio": avg_bio,
        "avg_batch": avg_batch,
        "overall_score": overall_score,
    }


def evaluate_cell_embeddings(
    embeddings_path: str | Path,
    dataset_path: str | Path,
    label_col: str,
    batch_col: str,
    n_neighbors: int = 15,
    distance_metric: str = "euclidean",
    resolutions: list[float] | None = None,
    random_state: int = 0,
) -> dict:
    """Run the full clustering benchmark for one saved cell-embeddings file."""
    if resolutions is None:
        resolutions = list(DEFAULT_RESOLUTIONS)

    embedding_data, embeddings, metadata, alignment = prepare_evaluation_inputs(
        embeddings_path=embeddings_path,
        dataset_path=dataset_path,
        label_col=label_col,
        batch_col=batch_col,
    )
    labels = metadata[label_col].astype(str).to_numpy()
    neighbors, effective_n_neighbors, neighbor_backend, jax_backend, neighbor_metric = _build_scib_neighbors(
        embeddings=embeddings,
        n_neighbors=n_neighbors,
        distance_metric=distance_metric,
        random_state=random_state,
    )
    resolution_sweep, best_clusters = run_resolution_sweep(
        neighbors=neighbors,
        labels=labels,
        resolutions=resolutions,
        random_state=random_state,
    )
    metric_summary = compute_benchmark_metrics(
        embeddings=embeddings,
        metadata=metadata,
        label_col=label_col,
        batch_col=batch_col,
        clusters=best_clusters,
        neighbors=neighbors,
        distance_metric=distance_metric,
    )

    best_resolution = float(resolution_sweep.loc[0, "resolution"])
    model_name = str(embedding_data.get("model", "unknown"))
    dataset_name = str(embedding_data.get("dataset_name", embedding_data.get("source", "unknown")))
    summary = {
        "model": model_name,
        "dataset_name": dataset_name,
        "label_col": label_col,
        "batch_col": batch_col,
        "join_mode": alignment["join_mode"],
        "n_cell_embeddings_rows": alignment["n_cell_embeddings_rows"],
        "n_eval_cells": alignment["n_eval_rows"],
        "n_dropped_missing_label_or_batch": alignment["n_dropped_missing_label_or_batch"],
        "n_unique_labels": alignment["n_unique_labels"],
        "n_unique_batches": alignment["n_unique_batches"],
        "n_neighbors": int(n_neighbors),
        "n_neighbors_used": int(effective_n_neighbors),
        "distance_metric": str(distance_metric),
        "neighbor_distance_metric": str(neighbor_metric),
        "metric_backend": "scib_metrics",
        "neighbor_backend": str(neighbor_backend),
        "jax_backend": str(jax_backend),
        "best_resolution": best_resolution,
    }
    summary.update(metric_summary)

    cluster_assignments = metadata.copy()
    cluster_assignments["model"] = model_name
    cluster_assignments["cluster"] = pd.Series(best_clusters).astype(str).tolist()
    cluster_assignments["best_resolution"] = best_resolution
    cluster_assignments = cluster_assignments.loc[
        :, ["model", "row_id", "source_obs_index", label_col, batch_col, "cluster", "best_resolution"]
    ]

    resolution_sweep = resolution_sweep.copy()
    resolution_sweep["model"] = model_name
    resolution_sweep["dataset_name"] = dataset_name
    resolution_sweep["label_col"] = label_col
    resolution_sweep["batch_col"] = batch_col

    return {
        "config": {
            "embeddings_path": str(Path(embeddings_path).resolve()),
            "dataset_path": str(Path(dataset_path).resolve()),
            "label_col": label_col,
            "batch_col": batch_col,
            "n_neighbors": int(n_neighbors),
            "n_neighbors_used": int(effective_n_neighbors),
            "distance_metric": str(distance_metric),
            "neighbor_distance_metric": str(neighbor_metric),
            "metric_backend": "scib_metrics",
            "neighbor_backend": str(neighbor_backend),
            "jax_backend": str(jax_backend),
            "resolutions": [float(value) for value in resolutions],
            "random_state": int(random_state),
        },
        "alignment": alignment,
        "summary": summary,
        "resolution_sweep": resolution_sweep,
        "cluster_assignments": cluster_assignments,
    }


def save_evaluation_result(result: dict, output_path: str | Path) -> list[Path]:
    """Persist the evaluation result bundle plus standard CSV exports."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as handle:
        pickle.dump(result, handle)

    stem = output_path.with_suffix("")
    summary_path = stem.with_name(f"{stem.name}_summary.csv")
    resolution_path = stem.with_name(f"{stem.name}_resolution_sweep.csv")
    clusters_path = stem.with_name(f"{stem.name}_clusters.csv")

    pd.DataFrame([result["summary"]]).to_csv(summary_path, index=False)
    result["resolution_sweep"].to_csv(resolution_path, index=False)
    result["cluster_assignments"].to_csv(clusters_path, index=False)
    return [output_path, summary_path, resolution_path, clusters_path]


def _validate_evaluation_result_bundle(result: dict, source: str) -> dict:
    if not isinstance(result, dict):
        raise ValueError(f"Expected a dict result bundle from {source}, got {type(result).__name__}.")

    missing = REQUIRED_EVALUATION_RESULT_KEYS - set(result.keys())
    if missing:
        raise ValueError(
            f"Evaluation result bundle from {source} is missing keys: {sorted(missing)}"
        )

    if not isinstance(result["summary"], dict):
        raise ValueError(f"Evaluation result bundle from {source} has a non-dict summary.")
    if not isinstance(result["resolution_sweep"], pd.DataFrame):
        raise ValueError(
            f"Evaluation result bundle from {source} has a non-DataFrame resolution_sweep."
        )
    if not isinstance(result["cluster_assignments"], pd.DataFrame):
        raise ValueError(
            f"Evaluation result bundle from {source} has a non-DataFrame cluster_assignments."
        )
    return result


def load_evaluation_results(path: str | Path) -> dict:
    """Load one saved clustering evaluation bundle from disk."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Evaluation results file not found: {path}")

    with open(path, "rb") as handle:
        result = pickle.load(handle)
    return _validate_evaluation_result_bundle(result, source=str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved cell embeddings with clustering metrics."
    )
    parser.add_argument("--embeddings", required=True, help="Path to the cell-embeddings pickle.")
    parser.add_argument("--dataset", required=True, help="Path to the source h5ad dataset.")
    parser.add_argument("--label-col", required=True, help="Obs column used as biological labels.")
    parser.add_argument("--batch-col", required=True, help="Obs column used as batch labels.")
    parser.add_argument("--output", required=True, help="Output pickle path.")
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for the kNN graph (default: 15).",
    )
    parser.add_argument(
        "--distance-metric",
        default="euclidean",
        help="Distance metric for neighbors and silhouettes (default: euclidean).",
    )
    parser.add_argument(
        "--resolutions",
        default=",".join(str(value) for value in DEFAULT_RESOLUTIONS),
        help="Comma-separated Leiden resolutions (default: 0.1,...,2.0).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random state for Leiden (default: 0).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resolutions = _parse_resolutions(args.resolutions)
    result = evaluate_cell_embeddings(
        embeddings_path=args.embeddings,
        dataset_path=args.dataset,
        label_col=args.label_col,
        batch_col=args.batch_col,
        n_neighbors=args.n_neighbors,
        distance_metric=args.distance_metric,
        resolutions=resolutions,
        random_state=args.random_state,
    )
    saved_paths = save_evaluation_result(result, args.output)
    print(f"Saved {len(saved_paths)} evaluation results:")
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
