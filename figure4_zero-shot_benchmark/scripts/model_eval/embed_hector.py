#!/usr/bin/env python3
"""Self-contained Hector embedding generator for the benchmark workflow.

Loads h5ad data, optionally subsamples cells, extracts Hector latent embeddings
via the installed ``hector`` package, validates output, and saves the
standardized pickle schema the other embed_*.py scripts use.

Usage:
    python model_eval/embed_hector.py
    python model_eval/embed_hector.py --input path/to/data.h5ad --n-cells 500 --device cuda
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import re
import warnings
from pathlib import Path
from typing import List, Optional

import anndata as ad
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path constants resolved relative to this file
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _SCRIPT_DIR.parents[1]   # the figure folder
_DEFAULT_INPUT = _WORKSPACE_ROOT / "input" / "TS_ALL_cells_10k.h5ad"
_DEFAULT_MODEL_NAME = "human"  # the packaged checkpoint, same entry the native head uses
_DEFAULT_OUTPUT = _WORKSPACE_ROOT / "result" / "hector_embeddings.pkl"

MAX_CELLS = 100_000
SEED = 42
_GENE_ID_COLUMNS = ("feature_id", "gene_ids", "ensembl_id", "gene_id")
_ENSEMBL_PATTERN = re.compile(r"(ENS(?:MUS)?G\d+)")


def configure_runtime(device: str) -> None:
    """Set writable cache dirs and optionally force CPU execution."""
    cache_dirs = {
        "NUMBA_CACHE_DIR": Path("/tmp") / "numba_cache_hector",
        "MPLCONFIGDIR": Path("/tmp") / "mpl_hector",
        "XDG_CACHE_HOME": Path("/tmp") / "xdg_cache_hector",
    }
    for env_key, cache_dir in cache_dirs.items():
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_key, str(cache_dir))

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_and_subsample(
    input_path: str,
    n_cells: Optional[int],
    seed: int = SEED,
):
    """Load h5ad and optionally subsample *n_cells* with fixed seed."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Expected location: {path.resolve()}"
        )

    adata = ad.read_h5ad(str(path), backed="r")
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


def load_hector_package():
    """Import the installed Hector package after runtime env configuration."""
    return importlib.import_module("hector")


def _get_row_ids(adata) -> List[str]:
    for key in ("observation_joinid", "soma_joinid", "_index"):
        if key in adata.obs:
            return adata.obs[key].astype(str).tolist()
    return adata.obs_names.astype(str).tolist()


def _get_cell_type_ids(adata) -> List[Optional[str]]:
    if "cell_type_ontology_term_id" not in adata.obs:
        return [None] * adata.n_obs
    return adata.obs["cell_type_ontology_term_id"].astype(str).tolist()


def _canonicalize_gene_id(value: object) -> str:
    value = str(value).strip()
    match = _ENSEMBL_PATTERN.search(value)
    if match is not None:
        return match.group(1)
    return value


def resolve_gene_id_column(adata, required_gene_ids):
    """Pick the source with the highest overlap against Hector's reference genes."""
    required_gene_ids = [str(gene_id) for gene_id in required_gene_ids]
    required_gene_id_set = set(required_gene_ids)
    candidates = [("var_names", adata.var_names.astype(str).tolist())]

    for column in _GENE_ID_COLUMNS:
        if column in adata.var.columns:
            candidates.append((f"var[{column!r}]", adata.var[column].astype(str).tolist()))

    best_source = None
    best_values = None
    best_hits = -1
    overlap_stats = []

    for source_name, values in candidates:
        normalized_values = [_canonicalize_gene_id(value) for value in values]
        hits = len(required_gene_id_set.intersection(normalized_values))
        overlap_stats.append((source_name, hits))
        if hits > best_hits:
            best_source = source_name
            best_values = normalized_values
            best_hits = hits

    print("  Hector gene-ID overlap by source:")
    for source_name, hits in overlap_stats:
        print(f"    {source_name}: {hits}/{len(required_gene_ids)} matched")

    if best_values is None or best_hits <= 0:
        summary = ", ".join(
            f"{source_name}={hits}/{len(required_gene_ids)}"
            for source_name, hits in overlap_stats
        )
        raise RuntimeError(
            "Could not find any overlap between the dataset gene identifiers and "
            f"Hector's reference gene set. Candidate overlaps: {summary}"
        )

    # Build a minimal AnnData view for prediction so Hector cannot silently swap
    # to adata.raw and lose the normalized gene IDs we selected here.
    predict_adata = ad.AnnData(X=adata.X, obs=adata.obs.copy(), var=adata.var.copy())
    predict_adata.var["_hector_gene_ids"] = best_values

    print(
        f"  Selected Hector gene source: {best_source} "
        f"({best_hits}/{len(required_gene_ids)} matched)"
    )
    return predict_adata, "_hector_gene_ids"


def generate_embeddings(adata, model_name: str = _DEFAULT_MODEL_NAME):
    """Extract Hector latent embeddings from an AnnData input."""
    hector_package = load_hector_package()
    predictor = hector_package.HECTOR(model_name, verbose=True)
    predict_adata, gene_id_column = resolve_gene_id_column(adata, predictor.gene_ids)

    print(f"  Gene ID column: {gene_id_column}")
    print(f"  Model registry entry: {model_name}")

    predictor.compute_cell_embeddings(predict_adata)

    if "X_hector" not in predict_adata.obsm:
        raise RuntimeError("Hector embedding output was not cached in adata.obsm['X_hector'].")

    embeddings = np.asarray(predict_adata.obsm["X_hector"])
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Expected 2D embedding matrix, got shape {embeddings.shape}"
        )
    if embeddings.shape[0] != adata.n_obs:
        raise RuntimeError(
            f"Embedding row count mismatch: {embeddings.shape[0]} != {adata.n_obs}"
        )
    if embeddings.shape[1] == 0:
        raise RuntimeError("Hector returned zero-dimensional embeddings.")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Embeddings contain NaN or Inf values.")
    if np.allclose(embeddings, embeddings[:1], atol=1e-6, rtol=1e-6):
        raise RuntimeError(
            "Hector produced an identical embedding for every cell. "
            "This usually indicates failed gene matching or zero-filled inputs."
        )

    print(f"  Embedding shape: {embeddings.shape}")
    return embeddings


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Hector embeddings for the benchmark"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(_DEFAULT_INPUT),
        help=f"Input h5ad file (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--n-cells",
        "-n",
        type=int,
        default=None,
        help="Number of cells to subsample; omit to use all cells",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(_DEFAULT_OUTPUT),
        help=f"Output pickle path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device preference (default: cpu)",
    )
    args = parser.parse_args()

    configure_runtime(args.device)

    print("=" * 60)
    print("Hector Embedding Generation")
    print("=" * 60)

    if args.n_cells is None:
        print("\n[1/4] Loading all cells...")
    else:
        print(f"\n[1/4] Loading and subsampling {args.n_cells} cells...")
    adata, sampled_obs_indices = load_and_subsample(args.input, args.n_cells)
    requested_n_cells = int(args.n_cells) if args.n_cells is not None else int(adata.n_obs)
    input_n_cells = int(adata.n_obs)
    cell_types = adata.obs["cell_type"].astype(str).tolist()
    cell_type_ids = _get_cell_type_ids(adata)
    row_ids = _get_row_ids(adata)
    dataset_name = Path(args.input).name
    print(f"  Shape: {adata.shape}")
    print(f"  Unique cell types: {len(set(cell_types))}")

    print("\n[2/4] Generating embeddings...")
    embeddings = generate_embeddings(adata, _DEFAULT_MODEL_NAME)

    print("\n[3/4] Validation passed")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Cell types: {len(set(cell_types))} unique")
    print(f"  Sample (cell 0, first 5 dims): {embeddings[0, :5]}")

    print("\n[4/4] Saving results...")
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
        "model": "Hector",
        "source": dataset_name,
        "n_cells": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
    }

    with open(output_path, "wb") as handle:
        pickle.dump(output_data, handle)

    print(f"  Saved to {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
