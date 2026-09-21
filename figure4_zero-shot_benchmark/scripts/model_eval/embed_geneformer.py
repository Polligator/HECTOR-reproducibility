#!/usr/bin/env python3
"""Self-contained Geneformer embedding generator for Tabula Sapiens benchmark.

Loads h5ad data, optionally subsamples cells, tokenizes via
TranscriptomeTokenizer (V2, h5ad format), extracts cell embeddings via
EmbExtractor from the Geneformer-V2-316M checkpoint, validates output, and
saves standardized pickle.

Usage:
    python model_eval/embed_geneformer.py
    python model_eval/embed_geneformer.py --input path/to/data.h5ad --n-cells 500 --device cuda
"""

import argparse
import os
import pickle
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional

# Remove this script's directory from sys.path so imports resolve from the
# vendored Geneformer package root under model_eval/geneformer/.
_script_dir = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if os.path.realpath(p) != _script_dir]

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path constants resolved relative to this file
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # model_eval/
_WORKSPACE_ROOT = _SCRIPT_DIR.parents[1]                 # project root
_LOCAL_GENEFORMER_ROOT = _SCRIPT_DIR.parents[1] / "models" / "geneformer"
_DEFAULT_INPUT = _WORKSPACE_ROOT / "input" / "TS_ALL_cells_10k.h5ad"
_AVAILABLE_VERSIONS = {
    "104M": {
        "checkpoint": _SCRIPT_DIR.parents[1] / "models" / "geneformer" / "Geneformer-V2-104M",
        "output": _WORKSPACE_ROOT / "result" / "geneformer_104m_embeddings.pkl",
        "model_name": "Geneformer-V2-104M",
    },
    "316M": {
        "checkpoint": _SCRIPT_DIR.parents[1] / "models" / "geneformer" / "Geneformer-V2-316M",
        "output": _WORKSPACE_ROOT / "result" / "geneformer_316m_embeddings.pkl",
        "model_name": "Geneformer-V2-316M",
    },
}
_DEFAULT_VERSION = "316M"
_DEFAULT_CHECKPOINT = _AVAILABLE_VERSIONS[_DEFAULT_VERSION]["checkpoint"]
_DEFAULT_OUTPUT = _AVAILABLE_VERSIONS[_DEFAULT_VERSION]["output"]
_DEFAULT_MODEL_NAME = _AVAILABLE_VERSIONS[_DEFAULT_VERSION]["model_name"]

if str(_LOCAL_GENEFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_GENEFORMER_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MAX_CELLS = 100_000
SEED = 42
_ENSEMBL_PATTERN = re.compile(r"^(ENS[A-Z]*G\d+)(?:\\..*)?$")
_GENEFORMER_INPUT_POSITION_COL = "geneformer_input_position"
_GENEFORMER_ROW_ID_COL = "geneformer_row_id"


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


def _select_ensembl_source(adata):
    """Choose the var source that most plausibly contains Ensembl IDs."""
    candidates = [("var_names", adata.var_names.astype(str))]
    for column in ("ensembl_id", "feature_id", "gene_ids", "gene_id", "index"):
        if column in adata.var.columns:
            candidates.append((f"var[{column!r}]", adata.var[column].astype(str)))

    best_name = None
    best_values = None
    best_score = -1.0

    for name, values in candidates:
        total = len(values)
        if total == 0:
            continue
        match_mask = values.str.match(_ENSEMBL_PATTERN)
        score = float(match_mask.mean())
        if score > best_score:
            best_name = name
            best_values = values
            best_score = score

    if best_values is None or best_score <= 0.0:
        raise ValueError(
            "Could not find an Ensembl ID source in adata.var or var_names."
        )
    return best_name, best_values.astype(str).tolist(), best_score


def prepare_adata(adata):
    """Prepare AnnData for Geneformer tokenization.

    Sets var['ensembl_id'] from an Ensembl-bearing source column and computes
    obs['n_counts'] from the expression matrix sum per cell.

    Args:
        adata: AnnData object with an Ensembl-bearing source in var metadata or var_names.

    Returns:
        AnnData ready for Geneformer tokenization.
    """
    source_name, ensembl_ids, match_rate = _select_ensembl_source(adata)
    adata.var["ensembl_id"] = ensembl_ids
    adata.obs["n_counts"] = np.array(adata.X.sum(axis=1)).flatten()
    adata.uns["_geneformer_ensembl_source"] = source_name
    adata.uns["_geneformer_use_h5ad_index"] = source_name == "var_names"
    print(
        f"  Selected Ensembl source: {source_name} "
        f"({match_rate:.1%} rows matched Ensembl pattern)"
    )
    return adata


def _attach_tracking_columns(adata, row_ids: list[str]) -> None:
    """Persist original sampled-row order through Geneformer's internal sort."""
    if len(row_ids) != adata.n_obs:
        raise ValueError(f"row_ids length mismatch: {len(row_ids)} != {adata.n_obs}")
    adata.obs[_GENEFORMER_INPUT_POSITION_COL] = np.arange(adata.n_obs, dtype=np.int64)
    # AnnData in the pinned Geneformer env cannot write pandas nullable string
    # arrays unless a newer serialization flag is enabled. Store plain Python
    # strings so the temporary h5ad remains portable across env versions.
    adata.obs[_GENEFORMER_ROW_ID_COL] = np.asarray(row_ids, dtype=object)


def tokenize_data(adata, tmpdir: str):
    """Tokenize AnnData into a Geneformer .dataset directory.

    Saves the prepared AnnData as a temporary h5ad, then runs
    TranscriptomeTokenizer to produce a HuggingFace .dataset.

    Args:
        adata: Prepared AnnData with resolved `var['ensembl_id']` and `obs['n_counts']`.
        tmpdir: Temporary directory for intermediate files.

    Returns:
        Path to the tokenized .dataset directory.
    """
    from geneformer import TranscriptomeTokenizer

    # Save prepared AnnData to temp h5ad
    data_dir = os.path.join(tmpdir, "input_data")
    os.makedirs(data_dir, exist_ok=True)
    h5ad_path = os.path.join(data_dir, "cells.h5ad")
    adata.write_h5ad(h5ad_path)
    print(f"  Saved temp h5ad: {h5ad_path}")

    # Tokenize
    token_dir = os.path.join(tmpdir, "tokenized")
    os.makedirs(token_dir, exist_ok=True)

    use_h5ad_index = bool(adata.uns.get("_geneformer_use_h5ad_index", False))
    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={
            "cell_type": "cell_type",
            _GENEFORMER_INPUT_POSITION_COL: _GENEFORMER_INPUT_POSITION_COL,
            _GENEFORMER_ROW_ID_COL: _GENEFORMER_ROW_ID_COL,
        },
        model_version="V2",
        use_h5ad_index=use_h5ad_index,
    )
    tokenizer.nproc = None
    print(f"  Geneformer tokenizer using h5ad index: {use_h5ad_index}")
    tokenizer.tokenize_data(
        data_directory=data_dir,
        output_directory=token_dir,
        output_prefix="geneformer",
        file_format="h5ad",
    )

    dataset_path = os.path.join(token_dir, "geneformer.dataset")
    if not os.path.exists(dataset_path):
        # List what was actually created for debugging
        contents = os.listdir(token_dir)
        raise FileNotFoundError(
            f"Tokenized dataset not found at {dataset_path}\n"
            f"Contents of {token_dir}: {contents}"
        )

    print(f"  Tokenized dataset: {dataset_path}")
    return dataset_path


def _load_extractor_frame(output_dir: str, embs_df) -> pd.DataFrame:
    """Return the extractor output as a DataFrame regardless of backend path."""
    if isinstance(embs_df, pd.DataFrame):
        return embs_df

    csv_path = os.path.join(output_dir, "geneformer.csv")
    parquet_path = os.path.join(output_dir, "geneformer.parquet")

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, index_col=0)
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    contents = os.listdir(output_dir)
    raise FileNotFoundError(
        f"No embedding output found in {output_dir}\n"
        f"Contents: {contents}"
    )


def _embedding_columns(frame: pd.DataFrame) -> list:
    """Identify embedding-dimension columns without consuming metadata labels."""
    numeric_cols = [c for c in frame.columns if isinstance(c, (int, float))]
    if numeric_cols:
        return numeric_cols

    string_numeric_cols = [
        c for c in frame.columns if str(c).replace(".", "").replace("-", "").isdigit()
    ]
    if string_numeric_cols:
        return string_numeric_cols

    meta_cols = {
        "cell_type",
        "n_counts",
        "cell_id",
        _GENEFORMER_INPUT_POSITION_COL,
        _GENEFORMER_ROW_ID_COL,
    }
    return [c for c in frame.columns if c not in meta_cols]


def extract_embeddings(dataset_path: str, model_dir: str, tmpdir: str):
    """Extract cell embeddings from tokenized dataset using EmbExtractor.

    Args:
        dataset_path: Path to the tokenized .dataset directory.
        model_dir: Path to the Geneformer checkpoint directory.
        tmpdir: Temporary directory for extractor output.

    Returns:
        Tuple of (embedding matrix, metadata aligned to original sampled order).
    """
    from geneformer import EmbExtractor

    output_dir = os.path.join(tmpdir, "embeddings")
    os.makedirs(output_dir, exist_ok=True)

    extractor = EmbExtractor(
        model_type="Pretrained",
        emb_mode="cell",
        max_ncells=None,
        forward_batch_size=20,
        emb_label=["cell_type", _GENEFORMER_INPUT_POSITION_COL, _GENEFORMER_ROW_ID_COL],
        model_version="V2",
    )
    extractor.nproc = None

    # extract_embs returns a DataFrame and also saves CSV
    embs_df = extractor.extract_embs(
        model_directory=model_dir,
        input_data_file=dataset_path,
        output_directory=output_dir,
        output_prefix="geneformer",
    )
    frame = _load_extractor_frame(output_dir, embs_df)
    print(f"  DataFrame shape: {frame.shape}")
    print(f"  DataFrame columns (first 5): {list(frame.columns[:5])}")

    if _GENEFORMER_INPUT_POSITION_COL not in frame.columns:
        raise RuntimeError(
            "Geneformer extractor output is missing the saved input-position column; "
            "cannot restore embeddings to the original sampled-cell order."
        )

    emb_cols = _embedding_columns(frame)
    input_positions = pd.to_numeric(
        frame[_GENEFORMER_INPUT_POSITION_COL], errors="raise"
    ).to_numpy(dtype=np.int64)
    restore_order = np.argsort(input_positions, kind="stable")
    restored_positions = input_positions[restore_order]
    expected_positions = np.arange(len(restored_positions), dtype=np.int64)
    if not np.array_equal(restored_positions, expected_positions):
        raise RuntimeError(
            "Geneformer extractor output did not preserve a valid permutation of the "
            "sampled-cell order."
        )

    embeddings = frame.loc[:, emb_cols].to_numpy(dtype=np.float32)[restore_order]
    aligned_metadata = {
        "cell_types": frame["cell_type"].astype(str).to_numpy()[restore_order].tolist()
        if "cell_type" in frame.columns else None,
        "row_ids": frame[_GENEFORMER_ROW_ID_COL].astype(str).to_numpy()[restore_order].tolist()
        if _GENEFORMER_ROW_ID_COL in frame.columns else None,
    }
    return embeddings, aligned_metadata


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Geneformer embeddings for Tabula Sapiens benchmark"
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
    parser.add_argument(
        "--model-version",
        default=_DEFAULT_VERSION,
        choices=list(_AVAILABLE_VERSIONS),
        help=f"Geneformer model version (default: {_DEFAULT_VERSION})",
    )
    args = parser.parse_args()

    # Resolve version-specific defaults when user didn't override output
    version_cfg = _AVAILABLE_VERSIONS[args.model_version]
    if args.output == str(_DEFAULT_OUTPUT):
        args.output = str(version_cfg["output"])

    # Set CUDA visibility before any geneformer imports (which import torch).
    # Geneformer's load_model auto-detects CUDA and moves the model there,
    # so we must hide CUDA entirely when --device cpu is requested.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("=" * 60)
    print("Geneformer Embedding Generation")
    print("=" * 60)

    # 1. Load the requested AnnData subset
    if args.n_cells is None:
        print("\n[1/5] Loading all cells...")
    else:
        print(f"\n[1/5] Loading and subsampling {args.n_cells} cells...")
    adata, sampled_obs_indices = load_and_subsample(args.input, args.n_cells)
    expected_n_cells = int(adata.n_obs)
    requested_n_cells = int(args.n_cells) if args.n_cells is not None else expected_n_cells
    input_n_cells = expected_n_cells
    cell_types = adata.obs["cell_type"].values.tolist()
    cell_type_ids = _get_cell_type_ids(adata)
    row_ids = _get_row_ids(adata)
    _attach_tracking_columns(adata, row_ids)
    dataset_name = Path(args.input).name
    print(f"  Shape: {adata.shape}")
    print(f"  Unique cell types: {len(set(cell_types))}")

    # 2. Prepare for Geneformer
    print("\n[2/5] Preparing data for Geneformer...")
    adata = prepare_adata(adata)
    print(f"  var['ensembl_id'] prepared for tokenization")
    print(f"  obs['n_counts'] computed (mean: {adata.obs['n_counts'].mean():.1f})")

    # 3. Tokenize & extract embeddings
    checkpoint_dir = str(version_cfg["checkpoint"])
    model_name = version_cfg["model_name"]
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"Expected location: {ckpt.resolve()}"
        )
    print(f"  Checkpoint: {ckpt}")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n[3/5] Tokenizing data...")
        dataset_path = tokenize_data(adata, tmpdir)

        print("\n[4/5] Extracting embeddings...")
        embeddings, extracted_metadata = extract_embeddings(dataset_path, checkpoint_dir, tmpdir)

    # Validate
    print(f"\n[5/5] Validation...")
    print(f"  Embedding shape: {embeddings.shape}")

    nan_count = int(np.isnan(embeddings).sum())
    if nan_count > 0:
        raise RuntimeError(
            f"Embeddings contain {nan_count} NaN values — aborting."
        )
    print(f"  No NaN values")

    if embeddings.shape[0] != expected_n_cells:
        raise RuntimeError(
            f"Row count mismatch: expected {expected_n_cells}, "
            f"got {embeddings.shape[0]}"
        )
    print(f"  Row count matches input ({expected_n_cells})")

    if extracted_metadata.get("cell_types") is not None and extracted_metadata["cell_types"] != cell_types:
        raise RuntimeError(
            "Geneformer cell_type metadata does not match the original sampled-cell "
            "order after restoring embeddings."
        )
    if extracted_metadata.get("row_ids") is not None and extracted_metadata["row_ids"] != row_ids:
        raise RuntimeError(
            "Geneformer row_id metadata does not match the original sampled-cell "
            "order after restoring embeddings."
        )
    print("  Restored Geneformer embeddings to original sampled-cell order")
    print(f"  Sample (cell 0, first 5 dims): {embeddings[0, :5]}")

    # Save standardized pickle
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
        "model": model_name,
        "source": dataset_name,
        "n_cells": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"\n  Saved to {output_path}")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Cell types: {len(set(cell_types))} unique")
    print("\nDone.")


if __name__ == "__main__":
    main()
