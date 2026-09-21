#!/usr/bin/env python3
"""Self-contained scGPT embedding generator for Tabula Sapiens benchmark.

Loads h5ad data, optionally subsamples cells, generates 512-dim embeddings via
the scGPT_CP checkpoint, validates output, and saves standardized pickle.

Usage:
    python model_eval/embed_scgpt.py
    python model_eval/embed_scgpt.py --input path/to/data.h5ad --n-cells 500 --device cuda
"""

import argparse
import pickle
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import scanpy as sc
import scgpt as scg

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path constants resolved relative to this file
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # model_eval/
_WORKSPACE_ROOT = _SCRIPT_DIR.parents[1]                 # project root
_DEFAULT_INPUT = _WORKSPACE_ROOT / "input" / "TS_ALL_cells_10k.h5ad"
_DEFAULT_CHECKPOINT = _SCRIPT_DIR.parents[1] / "models" / "scGPT" / "checkpoints" / "scGPT_CP"
_DEFAULT_OUTPUT = _WORKSPACE_ROOT / "result" / "scgpt_embeddings.pkl"

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


def _get_row_ids(adata) -> List[str]:
    for key in ("observation_joinid", "soma_joinid", "_index"):
        if key in adata.obs:
            return adata.obs[key].astype(str).tolist()
    return adata.obs_names.astype(str).tolist()


def _get_cell_type_ids(adata) -> List[Optional[str]]:
    if "cell_type_ontology_term_id" not in adata.obs:
        return [None] * adata.n_obs
    return adata.obs["cell_type_ontology_term_id"].astype(str).tolist()


def generate_embeddings(adata, checkpoint_dir: str, device: str = "cpu"):
    """Generate 512-dim scGPT embeddings using embed_data API.

    Args:
        adata: AnnData with gene symbols in var['feature_name'].
        checkpoint_dir: Path to scGPT_CP checkpoint directory.
        device: 'cpu' or 'cuda'.

    Returns:
        np.ndarray of shape (n_cells, 512).

    Raises:
        FileNotFoundError: If checkpoint directory is missing.
        RuntimeError: If embeddings contain NaN values.
    """
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"Expected location: {ckpt.resolve()}"
        )

    gene_col = "feature_name"
    print(f"  Gene column: {gene_col}")
    print(f"  Sample genes: {list(adata.var[gene_col][:5])}")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Device: {device}")

    # scGPT's helper uses multiprocessing workers by default, which fails in
    # restricted environments where semaphore creation is unavailable.
    try:
        import scgpt.tasks.cell_emb as cell_emb

        original_dataloader = cell_emb.DataLoader

        def single_process_dataloader(*args, **kwargs):
            kwargs["num_workers"] = 0
            kwargs["pin_memory"] = False
            return original_dataloader(*args, **kwargs)

        cell_emb.DataLoader = single_process_dataloader
    except Exception:
        cell_emb = None
        original_dataloader = None

    embed_adata = scg.tasks.embed_data(
        adata,
        ckpt,
        gene_col=gene_col,
        batch_size=64,
        device=device,
        use_fast_transformer=False,
        return_new_adata=True,
    )
    if cell_emb is not None and original_dataloader is not None:
        cell_emb.DataLoader = original_dataloader

    # scGPT stores embeddings in obsm if available, otherwise in X
    if len(embed_adata.obsm) > 0:
        key = list(embed_adata.obsm.keys())[0]
        embeddings = embed_adata.obsm[key]
    else:
        embeddings = embed_adata.X

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
        description="Generate scGPT embeddings for Tabula Sapiens benchmark"
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

    print("=" * 60)
    print("scGPT Embedding Generation")
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
        adata, str(_DEFAULT_CHECKPOINT), args.device
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
        "model": "scGPT_CP",
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
