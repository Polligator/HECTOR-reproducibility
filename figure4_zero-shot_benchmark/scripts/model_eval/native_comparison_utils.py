"""# Native zero-shot comparison utilities

Reusable model-specific logic for auditing training labels and reproducing the
native scCello and OnClass decoders. Analysis scripts keep the biological
choices explicit and linear while this module contains the implementation
details needed by more than one step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx
from scipy import sparse, stats


# ==== scCello representation head ====
# The published cell representation is the transformed CLS token, not a mean
# over gene-token states. Keeping the head here prevents analysis scripts from
# reimplementing trained model logic.
class ScCelloRepresentationHead(torch.nn.Module):
    """Reproduce scCello's trained ``cell_cls`` transformation."""

    def __init__(self, hidden_size: int, layer_norm_eps: float, activation):
        super().__init__()
        self.dense_1 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.activation = activation
        self.dense_2 = torch.nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.dense_2(hidden_states)


def load_sccello_representation_head(
    checkpoint_path: str | Path,
    config,
    activation,
    device: str,
) -> ScCelloRepresentationHead:
    """Load the trained scCello cell-representation head."""
    from safetensors.torch import load_file

    checkpoint = load_file(str(checkpoint_path))
    head = ScCelloRepresentationHead(
        hidden_size=int(config.hidden_size),
        layer_norm_eps=float(config.layer_norm_eps),
        activation=activation,
    )

    state = {
        "dense_1.weight": checkpoint["cell_cls.predictions.dense_1.weight"],
        "dense_1.bias": checkpoint["cell_cls.predictions.dense_1.bias"],
        "layer_norm.weight": checkpoint["cell_cls.predictions.layer_norm.weight"],
        "layer_norm.bias": checkpoint["cell_cls.predictions.layer_norm.bias"],
        "dense_2.weight": checkpoint["cell_cls.predictions.dense_2.weight"],
        "dense_2.bias": checkpoint["cell_cls.predictions.bias"],
    }
    head.load_state_dict(state, strict=True)
    head.eval()
    return head.to(device)


def load_sccello_known_prototypes(
    checkpoint_path: str | Path,
    number_known_types: int,
) -> np.ndarray:
    """Return checkpoint-resident prototypes for known pretraining types."""
    from safetensors.torch import load_file

    checkpoint = load_file(str(checkpoint_path))
    prototypes = checkpoint["learned_centroid.weight"][:number_known_types]
    return prototypes.detach().cpu().numpy().astype(np.float32)


def cosine_similarity_matrix(
    query_matrix: np.ndarray,
    reference_matrix: np.ndarray,
) -> np.ndarray:
    """Compute row-wise cosine similarities between two matrices."""
    query_norm = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(reference_matrix, axis=1, keepdims=True)
    query_scaled = query_matrix / np.maximum(query_norm, 1e-12)
    reference_scaled = reference_matrix / np.maximum(reference_norm, 1e-12)
    return query_scaled @ reference_scaled.T


def rank_normalize_rows(values: np.ndarray) -> np.ndarray:
    """Rank rows and standardize them for vectorized Spearman correlation."""
    ranks = stats.rankdata(values, axis=1, method="average")
    ranks = ranks - ranks.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(ranks, axis=1, keepdims=True)
    return ranks / np.maximum(scale, 1e-12)


def predict_from_spearman_profiles(
    query_to_known: np.ndarray,
    candidate_to_known: np.ndarray,
    candidate_ids: list[str],
    batch_size: int = 2048,
) -> tuple[list[str], list[float]]:
    """Classify query profiles by vectorized Spearman correlation."""
    query_ranks = rank_normalize_rows(query_to_known)
    candidate_ranks = rank_normalize_rows(candidate_to_known)

    predicted_ids: list[str] = []
    predicted_scores: list[float] = []
    for start in range(0, len(query_ranks), batch_size):
        stop = min(start + batch_size, len(query_ranks))
        score_matrix = query_ranks[start:stop] @ candidate_ranks.T
        best_indices = np.argmax(score_matrix, axis=1)
        predicted_ids.extend(candidate_ids[index] for index in best_indices)
        predicted_scores.extend(
            score_matrix[np.arange(len(best_indices)), best_indices].astype(float)
        )
    return predicted_ids, predicted_scores


def compute_candidate_to_bridge_ppr_sparse(
    ontology,
    candidate_ids: list[str],
    bridge_ids: list[str],
    alpha: float = 0.9,
    max_iter: int = 100,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Vectorize NetworkX's single-seed PageRank convergence rule."""
    graph = ontology.G
    if graph is None:
        raise RuntimeError("Load the ontology before computing PPR profiles.")

    node_ids = list(graph.nodes())
    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=node_ids,
        dtype=np.float64,
        format="csr",
    )
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_degrees = np.zeros_like(degrees)
    nonisolated = degrees > 0
    inverse_degrees[nonisolated] = 1.0 / degrees[nonisolated]
    transition = sparse.diags(inverse_degrees) @ adjacency

    valid_candidates = [
        cell_type_id for cell_type_id in candidate_ids if cell_type_id in node_to_index
    ]
    personalization = np.zeros(
        (len(node_ids), len(valid_candidates)), dtype=np.float64
    )
    for column, cell_type_id in enumerate(valid_candidates):
        node_index = node_to_index[cell_type_id]
        personalization[node_index, column] = 1.0

    solutions = np.full_like(
        personalization,
        fill_value=1.0 / len(node_ids),
    )
    active_columns = np.arange(len(valid_candidates))
    dangling_indices = np.flatnonzero(~nonisolated)
    for _ in range(max_iter):
        previous = solutions[:, active_columns]
        updated = alpha * (transition.T @ previous)
        dangling_mass = alpha * previous[dangling_indices].sum(axis=0)
        updated += personalization[:, active_columns] * (
            dangling_mass[np.newaxis, :] + 1.0 - alpha
        )
        solutions[:, active_columns] = updated

        errors = np.abs(updated - previous).sum(axis=0)
        still_active = errors >= len(node_ids) * tolerance
        active_columns = active_columns[still_active]
        if len(active_columns) == 0:
            break
    else:
        raise RuntimeError("Vectorized PageRank did not converge within max_iter.")

    output = np.zeros((len(candidate_ids), len(bridge_ids)), dtype=np.float32)
    candidate_to_output = {
        cell_type_id: index for index, cell_type_id in enumerate(candidate_ids)
    }
    valid_bridge_columns = [
        (column, node_to_index[cell_type_id])
        for column, cell_type_id in enumerate(bridge_ids)
        if cell_type_id in node_to_index
    ]
    for valid_column, cell_type_id in enumerate(valid_candidates):
        output_row = candidate_to_output[cell_type_id]
        for output_column, node_index in valid_bridge_columns:
            output[output_row, output_column] = solutions[
                node_index, valid_column
            ]
    return output


# ==== OnClass inference helpers ====
# Each checkpoint uses a different gene order and a different seen-first term
# order. These helpers make both mappings explicit before ensemble averaging.
def checkpoint_seen_ids(checkpoint_path: str | Path) -> set[str]:
    """Extract the exact supervised label set from an OnClass NPZ."""
    checkpoint = np.load(checkpoint_path, allow_pickle=True)
    index_to_id = checkpoint["i2co"].item()
    number_seen = int(checkpoint["nseen"])
    return {str(index_to_id[index]) for index in range(number_seen)}


def checkpoint_ontology_ids(checkpoint_path: str | Path) -> set[str]:
    """Extract every label represented in an OnClass checkpoint ontology."""
    checkpoint = np.load(checkpoint_path, allow_pickle=True)
    return {str(value) for value in checkpoint["co2i"].item()}


def align_sparse_expression_to_checkpoint(
    query_expression,
    query_gene_names: np.ndarray,
    checkpoint_gene_names: np.ndarray,
) -> sparse.csr_matrix:
    """Align a sparse query matrix to one checkpoint's exact gene order."""
    query_gene_to_index = {
        str(gene).upper(): index for index, gene in enumerate(query_gene_names)
    }

    matched_checkpoint_indices: list[int] = []
    matched_query_indices: list[int] = []
    for checkpoint_index, gene in enumerate(checkpoint_gene_names):
        query_index = query_gene_to_index.get(str(gene).upper())
        if query_index is None:
            continue
        matched_checkpoint_indices.append(checkpoint_index)
        matched_query_indices.append(query_index)

    selected = sparse.csr_matrix(query_expression[:, matched_query_indices])
    row_indices, local_column_indices = selected.nonzero()
    checkpoint_columns = np.asarray(matched_checkpoint_indices)[local_column_indices]
    aligned = sparse.csr_matrix(
        (selected.data, (row_indices, checkpoint_columns)),
        shape=(query_expression.shape[0], len(checkpoint_gene_names)),
    )
    return aligned


def load_onclass_source(source_root: str | Path):
    """Import the official TensorFlow OnClass implementation from a local clone."""
    source_root = str(Path(source_root).resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    from OnClass.OnClassModel import OnClassModel
    from OnClass.OnClass_utils import (
        create_propagate_networks_using_nlp,
        extend_prediction_2unseen,
    )

    return OnClassModel, create_propagate_networks_using_nlp, extend_prediction_2unseen


def predict_onclass_batches(
    model,
    aligned_expression: sparse.csr_matrix,
    propagation_networks: list[np.ndarray],
    extend_prediction,
    batch_size: int = 256,
) -> np.ndarray:
    """Run one frozen OnClass checkpoint without a query-composition gate."""
    number_cells = aligned_expression.shape[0]
    predictions = np.empty((number_cells, int(model.nco)), dtype=np.float32)
    propagation_ratio = (float(model.nco) / float(model.nseen)) ** 2

    for start in range(0, number_cells, batch_size):
        stop = min(start + batch_size, number_cells)
        expression_batch = aligned_expression[start:stop].toarray().astype(np.float32)
        expression_batch = np.log1p(expression_batch)
        seen_scores = model.model.predict(expression_batch)
        all_scores = extend_prediction(
            seen_scores,
            propagation_networks,
            int(model.nseen),
            ratio=propagation_ratio,
            use_normalize=False,
        )
        predictions[start:stop] = all_scores.astype(np.float32)

    return predictions


def predictions_from_candidate_scores(
    score_matrix: np.ndarray,
    reference_ids: list[str],
    candidate_ids: list[str],
) -> tuple[list[str], list[float]]:
    """Restrict aligned scores to candidates and return top-one calls."""
    reference_to_index = {
        cell_type_id: index for index, cell_type_id in enumerate(reference_ids)
    }
    candidate_indices = np.asarray(
        [reference_to_index[cell_type_id] for cell_type_id in candidate_ids]
    )
    candidate_scores = score_matrix[:, candidate_indices]
    best_indices = np.argmax(candidate_scores, axis=1)
    predicted_ids = [candidate_ids[index] for index in best_indices]
    predicted_scores = candidate_scores[
        np.arange(len(best_indices)), best_indices
    ].astype(float)
    return predicted_ids, predicted_scores


def build_query_metadata_from_frozen_predictions(
    prediction_path: str | Path,
) -> pd.DataFrame:
    """Recover the frozen 33,000-row query manifest in its scored row order."""
    predictions = pd.read_csv(prediction_path)
    columns = [
        "dataset_name",
        "row_id",
        "sampled_obs_index",
        "truth_cell_type_id",
        "truth_cell_type_name",
    ]
    metadata = predictions.loc[:, columns].copy()
    metadata = metadata.rename(
        columns={
            "truth_cell_type_id": "cell_type_id",
            "truth_cell_type_name": "cell_type",
        }
    )
    metadata["row_id"] = metadata["row_id"].astype(str)
    return metadata
