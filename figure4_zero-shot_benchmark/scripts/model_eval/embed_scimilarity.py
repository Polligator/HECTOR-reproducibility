#!/usr/bin/env python3
"""Self-contained SCimilarity embedding generator for Tabula Sapiens benchmark.

Loads h5ad data, optionally subsamples cells, sets gene symbols as var index,
applies lognorm + alignment, generates embeddings via CellEmbedding, validates
output, and saves standardized pickle.

Usage:
    python model_eval/embed_scimilarity.py
    python model_eval/embed_scimilarity.py --input path/to/data.h5ad --n-cells 500 --device cuda
"""

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path constants resolved relative to this file
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # model_eval/
_WORKSPACE_ROOT = _SCRIPT_DIR.parents[1]                 # project root
_DEFAULT_INPUT = _WORKSPACE_ROOT / "input" / "TS_ALL_cells_10k.h5ad"
_DEFAULT_CHECKPOINT = _SCRIPT_DIR.parents[1] / "models" / "scimilarity" / "model_v1.1"
_DEFAULT_OUTPUT = _WORKSPACE_ROOT / "result" / "scimilarity_embeddings.pkl"

MAX_CELLS = 100_000
SEED = 42


def load_and_subsample(input_path: str, n_cells: Optional[int], seed: int = SEED):
    """Load h5ad and optionally subsample *n_cells* with fixed seed.

    Args:
        input_path: Path to the h5ad file.
        n_cells: Number of cells to subsample. Use ``None`` for all cells.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (AnnData, sampled_obs_indices).

    Raises:
        ValueError: If n_cells exceeds MAX_CELLS.
        FileNotFoundError: If input_path does not exist.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Expected location: {path.resolve()}"
        )

    adata = sc.read_h5ad(str(path), backed="r")
    total_cells = int(adata.n_obs)

    if n_cells is None:
        target_n_cells = total_cells
    else:
        target_n_cells = int(n_cells)

    if n_cells is not None and target_n_cells > MAX_CELLS:
        raise ValueError(
            f"Max supported cell count is {MAX_CELLS:,}, got {target_n_cells:,}"
        )
    if target_n_cells > total_cells:
        raise ValueError(
            f"Requested {target_n_cells:,} cells but dataset only has {total_cells:,}."
        )

    if target_n_cells == total_cells:
        indices = None
        adata = adata.to_memory()
    else:
        np.random.seed(seed)
        indices = np.random.choice(total_cells, target_n_cells, replace=False)
        adata = adata[indices].to_memory()

    if "decontXcounts" in adata.layers:
        adata.X = adata.layers["decontXcounts"].copy()

    sampled_obs_indices = None if indices is None else indices.astype(int).tolist()
    return adata, sampled_obs_indices


def _get_row_ids(adata) -> list[str]:
    for key in ("observation_joinid", "soma_joinid", "_index"):
        if key in adata.obs:
            return adata.obs[key].astype(str).tolist()
    return adata.obs_names.astype(str).tolist()


def _get_cell_type_ids(adata) -> list[str | None]:
    if "cell_type_ontology_term_id" not in adata.obs:
        return [None] * adata.n_obs
    return adata.obs["cell_type_ontology_term_id"].astype(str).tolist()


def generate_embeddings(adata, model_path: str, use_gpu: bool = False):
    """Generate SCimilarity embeddings via CellEmbedding API.

    Sets gene symbols from var['feature_name'] as var index (handling
    duplicates), applies lognorm_counts and align_dataset, then calls
    CellEmbedding.get_embeddings on the aligned expression matrix.

    Args:
        adata: AnnData with gene symbols in var['feature_name'] and raw counts in X.
        model_path: Path to SCimilarity model_v1.1 checkpoint directory.
        use_gpu: Whether to use GPU for inference.

    Returns:
        np.ndarray of shape (n_cells, embedding_dim).

    Raises:
        FileNotFoundError: If model_path does not exist.
        RuntimeError: If embeddings contain NaN values.
    """
    from scimilarity.cell_embedding import CellEmbedding
    from scimilarity import align_dataset, lognorm_counts

    ckpt = Path(model_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"Expected location: {ckpt.resolve()}"
        )

    # Set gene symbols as var index — SCimilarity requires gene symbols
    print("  Setting gene symbols as var index...")
    adata.var_names_make_unique()
    adata = adata[:, ~adata.var["feature_name"].duplicated()]
    adata.var.index = adata.var["feature_name"]
    print(f"  Genes after dedup: {adata.n_vars}")

    # Load model
    print(f"  Loading model from {ckpt}...")
    ce = CellEmbedding(model_path=str(ckpt), use_gpu=use_gpu)

    # Prepare data: lognorm then align to model gene order
    # lognorm_counts expects raw counts in layers['counts']
    print("  Applying lognorm_counts...")
    adata.layers["counts"] = adata.X.copy()
    adata = lognorm_counts(adata)
    print(f"  Aligning to {len(ce.gene_order)} model genes...")
    adata = align_dataset(adata, ce.gene_order)

    # Generate embeddings
    print("  Running inference...")
    embeddings = ce.get_embeddings(adata.X)

    if isinstance(embeddings, np.matrix):
        embeddings = np.asarray(embeddings)

    # Validate
    nan_count = int(np.isnan(embeddings).sum())
    if nan_count > 0:
        raise RuntimeError(
            f"Embeddings contain {nan_count} NaN values — aborting."
        )

    print(f"  Embedding shape: {embeddings.shape}")
    return embeddings


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SCimilarity embeddings for Tabula Sapiens benchmark"
    )
    parser.add_argument(
        "--input", "-i",
        default=str(_DEFAULT_INPUT),
        help=f"Input h5ad file (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--n-cells", "-n",
        type=int,
        default=None,
        help="Number of cells to subsample; omit to use all cells",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(_DEFAULT_OUTPUT),
        help=f"Output pickle path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device (default: cpu)",
    )
    args = parser.parse_args()

    use_gpu = args.device == "cuda"

    print("=" * 60)
    print("SCimilarity Embedding Generation")
    print("=" * 60)

    # 1. Load the requested AnnData subset
    if args.n_cells is None:
        print("\n[1/4] Loading all cells...")
    else:
        print(f"\n[1/4] Loading and subsampling {args.n_cells} cells...")
    adata, sampled_obs_indices = load_and_subsample(args.input, args.n_cells)
    requested_n_cells = int(args.n_cells) if args.n_cells is not None else int(adata.n_obs)
    input_n_cells = int(adata.n_obs)
    cell_types = adata.obs["cell_type"].values.tolist()
    cell_type_ids = _get_cell_type_ids(adata)
    row_ids = _get_row_ids(adata)
    dataset_name = Path(args.input).name
    print(f"  Shape: {adata.shape}")
    print(f"  Unique cell types: {len(set(cell_types))}")

    # 2. Generate embeddings
    print("\n[2/4] Generating embeddings...")
    embeddings = generate_embeddings(
        adata, str(_DEFAULT_CHECKPOINT), use_gpu=use_gpu
    )

    # 3. Report
    print(f"\n[3/4] Validation passed")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Cell types: {len(set(cell_types))} unique")
    print(f"  Sample (cell 0, first 5 dims): {embeddings[0, :5]}")

    # 4. Save standardized pickle
    print(f"\n[4/4] Saving results...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "embeddings": embeddings,
        "cell_types": cell_types,
        "cell_type_ids": cell_type_ids,
        "row_ids": row_ids,
        "dataset_name": dataset_name,
        "sampled_obs_indices": sampled_obs_indices,
        "requested_n_cells": requested_n_cells,
        "input_n_cells": input_n_cells,
        "output_n_cells": int(embeddings.shape[0]),
        "model": "SCimilarity_v1.1",
        "source": dataset_name,
        "n_cells": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"  Saved to {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
