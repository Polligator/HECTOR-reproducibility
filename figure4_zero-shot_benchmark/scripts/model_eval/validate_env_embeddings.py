#!/usr/bin/env python3
"""Force fresh per-model embedding generation for environment validation.

This helper exercises the same embedding subprocess path the benchmark uses,
but disables cache reuse by writing bridge/query outputs to a fresh temporary
directory for each validation run.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from benchmark_utils import (
    find_benchmark_workspace_dir,
    find_existing_path,
    load_cell_embeddings_result,
    resolve_default_device,
    run_embedding_job,
)


def build_model_configs(workspace_dir: Path, geneformer_version: str) -> dict[str, dict[str, object]]:
    """Build the model config table used for embedding generation."""
    geneformer_key = geneformer_version.upper()
    if geneformer_key not in {"104M", "316M"}:
        raise ValueError(f"Unsupported geneformer version: {geneformer_version}")

    return {
        "sccello": {
            "label": "scCello",
            "script": workspace_dir / "embed_sccello.py",
            "env": "sccello",
            "cell_embeddings_prefix": "sccello",
        },
        "scgpt": {
            "label": "scGPT",
            "script": workspace_dir / "embed_scgpt.py",
            "env": "scgpt",
            "cell_embeddings_prefix": "scgpt",
        },
        "geneformer": {
            "label": "geneformer",
            "script": workspace_dir / "embed_geneformer.py",
            "env": "geneformer",
            "cell_embeddings_prefix": f"geneformer_{geneformer_key.lower()}",
            "extra_args": ["--model-version", geneformer_key],
        },
        "scimilarity": {
            "label": "scimilarity",
            "script": workspace_dir / "embed_scimilarity.py",
            "env": "scimilarity",
            "cell_embeddings_prefix": "scimilarity",
        },
    }


def inspect_embeddings(output_path: Path, dataset_name: str, requested_n_cells: int) -> dict[str, object]:
    """Load one embedding artifact and assert the expected structural fields."""
    data = load_cell_embeddings_result(output_path)
    if data is None:
        raise RuntimeError(f"Could not read embedding output: {output_path}")

    embeddings = np.asarray(data.get("embeddings"))
    if embeddings.ndim != 2 or embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0:
        raise RuntimeError(f"Invalid embedding matrix shape in {output_path}: {embeddings.shape}")

    row_ids = data.get("row_ids")
    if row_ids is None or len(row_ids) != embeddings.shape[0]:
        raise RuntimeError(
            f"Embedding output {output_path} is missing row_ids or has a mismatched row count."
        )

    saved_dataset = Path(str(data.get("dataset_name", data.get("source", "")))).name
    if saved_dataset != dataset_name:
        raise RuntimeError(
            f"Embedding output {output_path} reports dataset {saved_dataset}, expected {dataset_name}."
        )

    sampled_obs_indices = data.get("sampled_obs_indices")
    if sampled_obs_indices is not None and len(sampled_obs_indices) != int(requested_n_cells):
        raise RuntimeError(
            f"Embedding output {output_path} sampled {len(sampled_obs_indices)} rows, "
            f"expected {requested_n_cells}."
        )

    return {
        "output_path": str(output_path),
        "dataset_name": saved_dataset,
        "n_rows": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "n_row_ids": int(len(row_ids)),
        "n_sampled_obs_indices": None if sampled_obs_indices is None else int(len(sampled_obs_indices)),
    }


def run_role_validation(
    model_key: str,
    cfg: dict[str, object],
    dataset_role: str,
    dataset_path: Path,
    output_root: Path,
    requested_n_cells: int,
    device: str,
) -> dict[str, object]:
    """Run one embedding subprocess and validate the saved pickle output."""
    output_path = output_root / f"{cfg['cell_embeddings_prefix']}_{dataset_role}_embeddings.pkl"
    record = run_embedding_job(
        model_name=str(cfg["label"]),
        cfg=cfg,
        dataset_role=dataset_role,
        dataset_path=dataset_path,
        output_path=output_path,
        n_cells=requested_n_cells,
        device=device,
        reported_n_cells=requested_n_cells,
    )

    if record["status"] != "completed":
        raise RuntimeError(
            f"{model_key}/{dataset_role} embedding job failed: {record['stderr_tail'] or record['stdout_tail']}"
        )

    artifact = inspect_embeddings(
        output_path=output_path,
        dataset_name=dataset_path.name,
        requested_n_cells=requested_n_cells,
    )
    return {"record": record, "artifact": artifact}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Force fresh bridge/query embedding generation for one benchmark model."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["sccello", "scgpt", "geneformer", "scimilarity"],
        help="Model key to validate.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. 'auto' uses resolve_default_device().",
    )
    parser.add_argument(
        "--bridge-n-cells",
        type=int,
        default=200,
        help="Bridge embedding subsample size.",
    )
    parser.add_argument(
        "--query-n-cells",
        type=int,
        default=100,
        help="Query embedding subsample size.",
    )
    parser.add_argument(
        "--geneformer-version",
        default="104M",
        choices=["104M", "316M", "104m", "316m"],
        help="Geneformer model version to validate when --model geneformer is selected.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional validation output directory. Defaults to a fresh /tmp directory.",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional JSON summary output path.",
    )
    args = parser.parse_args()

    workspace_dir = find_benchmark_workspace_dir(start=Path(__file__).resolve().parent)
    input_dir = Path(__file__).resolve().parents[2] / "input"
    bridge_data_path = Path(find_existing_path("TS_downsampled_cells.h5ad", start=input_dir))
    query_data_path = Path(find_existing_path("Zero_shot_query_set.h5ad", start=input_dir))

    if args.output_root is None:
        output_root = Path(
            tempfile.mkdtemp(prefix=f"zero_shot_{args.model}_validation_", dir="/tmp")
        )
    else:
        output_root = Path(args.output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    device = resolve_default_device() if args.device == "auto" else args.device
    configs = build_model_configs(workspace_dir, args.geneformer_version)
    cfg = configs[args.model]

    summary = {
        "model": args.model,
        "env": cfg["env"],
        "device": device,
        "workspace_dir": str(workspace_dir),
        "output_root": str(output_root),
        "bridge_data_path": str(bridge_data_path),
        "query_data_path": str(query_data_path),
        "bridge_n_cells_requested": int(args.bridge_n_cells),
        "query_n_cells_requested": int(args.query_n_cells),
        "geneformer_version": args.geneformer_version.upper(),
    }

    bridge_result = run_role_validation(
        model_key=args.model,
        cfg=cfg,
        dataset_role="bridge",
        dataset_path=bridge_data_path,
        output_root=output_root,
        requested_n_cells=int(args.bridge_n_cells),
        device=device,
    )
    query_result = run_role_validation(
        model_key=args.model,
        cfg=cfg,
        dataset_role="query",
        dataset_path=query_data_path,
        output_root=output_root,
        requested_n_cells=int(args.query_n_cells),
        device=device,
    )

    summary["bridge"] = bridge_result
    summary["query"] = query_result

    if args.summary_out is not None:
        summary_out = Path(args.summary_out).resolve()
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
