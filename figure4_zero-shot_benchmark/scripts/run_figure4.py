#!/usr/bin/env python3
"""Run the zero-shot benchmark end-to-end and export every plot as a PDF.

Running it from the project root regenerates every publication PDF under
`result/`.

The Figure-4 deliverables are produced by the ``model_eval/plot`` package
(paths are relative to the project root):

    result/figure_4.pdf                                 # composed 5-panel master figure (a-e)
    result/supplementary_figure_9_landmark_space.pdf    # two UMAP rows + the scIB table with encoding time
    result/supplementary_figure_10_per_type.pdf         # 66 types x 6 arms
    result/supplementary_figure_11_robustness.pdf       # depth, native decoders, open vocabulary

Per-model embeddings and clustering metrics are reused from previous runs when
the cached files exist. Use the command-line flags to disable the heavy
embedding-generation, clustering-evaluation, or Hector native-inference steps
when you only need the plots from cached results.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")  # headless-safe; lets the script run without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Workspace bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODEL_EVAL_DIR = SCRIPTS_DIR / "model_eval"
if not (MODEL_EVAL_DIR / "benchmark_utils.py").exists():
    raise FileNotFoundError(
        f"Could not locate model_eval/benchmark_utils.py under {SCRIPTS_DIR}"
    )
if str(MODEL_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_EVAL_DIR))
# scripts/ carries Hierarchical_metrics, imported as a package by several modules.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_utils  # noqa: E402
import benchmark_visualize  # noqa: E402
import cell_ontology_ppr  # noqa: E402
import clustering_evaluate  # noqa: E402
import evaluate  # noqa: E402
import query_set_padding  # noqa: E402

from benchmark_utils import (  # noqa: E402
    ensure_cell_embeddings,
    find_existing_path,
    get_available_conda_envs,
    resolve_default_device,
    run_clustering_evaluation_job,
    run_native_comparison_job,
)
from cell_ontology_ppr import CellOntologyPPR  # noqa: E402
from clustering_evaluate import load_evaluation_results  # noqa: E402
from query_set_padding import expected_query_counts_from_table  # noqa: E402
from evaluate import (  # noqa: E402
    annotate_prediction_table,
    assemble_prediction_table,
    build_bridge_prototypes_from_embeddings,
    build_exact_label_map,
    compute_hop_distance,
    compute_label_depths,
    compute_neighborhood_metrics_from_rank_profiles,
    compute_query_to_bridge_similarity,
    derive_observed_label_table,
    flatten_summary_row,
    generate_random_baseline,
    get_embedding_metadata_frame,
    load_embeddings,
    predict_with_candidate_profiles,
    read_h5ad_observation_metadata,
    resolve_target_n_cells,
    run_hector_native_predictions,
    save_results,
    select_candidate_label_table,
    summarize_prediction_table,
)


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

FIG_DIR = PROJECT_ROOT / "result"            # figures (figure_4.*, supp_s*.pdf, standalone PDFs)
OUTPUT_DIR = FIG_DIR / "data"                # data (embeddings, predictions, metrics, summaries)
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = PROJECT_ROOT / "input" / "TS_downsampled_cells.h5ad"
QUERY_DATA_PATH = PROJECT_ROOT / "input" / "Zero_shot_query_set.h5ad"
ONTOLOGY_PATH = PROJECT_ROOT / "input" / "cl.obo"

GENEFORMER_VERSION = "104M"  # '104M' or '316M'

MODEL_CONFIGS = {
    "Hector": {
        "script": MODEL_EVAL_DIR / "embed_hector.py",
        "env": "bench",
        "cell_embeddings_prefix": "hector",
    },
    "scGPT": {
        "script": MODEL_EVAL_DIR / "embed_scgpt.py",
        "env": "scgpt",
        "cell_embeddings_prefix": "scgpt",
    },
    "scCello": {
        "script": MODEL_EVAL_DIR / "embed_sccello.py",
        "env": "sccello",
        "cell_embeddings_prefix": "sccello",
    },
    "geneformer": {
        "script": MODEL_EVAL_DIR / "embed_geneformer.py",
        "env": "geneformer",
        "cell_embeddings_prefix": f"geneformer_{GENEFORMER_VERSION.lower()}",
        "extra_args": ["--model-version", GENEFORMER_VERSION],
    },
    "scimilarity": {
        "script": MODEL_EVAL_DIR / "embed_scimilarity.py",
        "env": "scimilarity",
        "cell_embeddings_prefix": "scimilarity",
    },
}
MODELS_TO_RUN = list(MODEL_CONFIGS)

# Native-decoder comparison (Supplementary Fig. 11c): HECTOR, OnClass and
# scCello each scored with their native decoder instead of the shared read-out,
# on the strict common 31-type zero-shot set. Runs after run_prediction_benchmark
# writes hector_native_closed_set_predictions.csv. Each step takes no CLI
# arguments; "check_path" is the file it writes last, used to skip a rerun.
NATIVE_COMPARISON_STEPS = [
    {
        "name": "training_audit",
        "script": MODEL_EVAL_DIR / "run_training_audit.py",
        "env": "bench",
        "check_path": OUTPUT_DIR / "training_and_ontology_audit_summary.csv",
    },
    {
        "name": "sccello_native",
        "script": MODEL_EVAL_DIR / "run_sccello_native.py",
        "env": "sccello",
        "check_path": OUTPUT_DIR / "sccello_native_strict_common31_predictions.csv",
    },
    {
        "name": "onclass_native",
        "script": MODEL_EVAL_DIR / "run_onclass_native.py",
        "env": "bench",
        "check_path": OUTPUT_DIR / "onclass_checkpoint_qc.csv",
        "extra_env": {"CUDA_VISIBLE_DEVICES": ""},
    },
    {
        "name": "hector_native_strict",
        "script": MODEL_EVAL_DIR / "run_hector_native_strict.py",
        "env": "hector",
        "check_path": OUTPUT_DIR / "hector_native_strict_common31_summary.csv",
    },
]
# Aggregates the four steps above into the summary CSVs Supplementary Fig. 11
# reads. Cheap (plain pandas), so it always reruns rather than being cached.
NATIVE_COMPARISON_SUMMARY_STEP = {
    "name": "comparison_summary",
    "script": MODEL_EVAL_DIR / "run_comparison_summary.py",
    "env": "bench",
}

LABEL_COL = "cell_type"
BATCH_COL = "donor_id"

# Clustering benchmark. None means every cell in the landmark file, the same
# convention BRIDGE_N_CELLS and QUERY_N_CELLS below use, so the scIB scores and
# the UMAP beside them describe the same cells.
N_CELLS_CLUSTERING = None
N_NEIGHBORS = 30
DISTANCE_METRIC = "euclidean"
LEIDEN_RESOLUTIONS = np.round(np.arange(0.5, 2.5, 0.1), 1).tolist()
RANDOM_STATE = 0

# Prediction benchmark
STRICT_UNSEEN = True
BRIDGE_N_CELLS = None
QUERY_N_CELLS = None
PPR_ALPHA = 0.4
# Scales the query-to-bridge cosine similarities. The prediction rule reads only
# which landmark is nearest, which no positive scaling changes, so this affects
# the neighbourhood metric in supplement S2 and nothing else.
TEMPERATURE = 0.1

DISPLAY_NAME_BY_MODEL = {
    "Hector": "Hector (PPR)",
    "scGPT": "scGPT (PPR)",
    "scCello": "scCello (PPR)",
    "geneformer": "Geneformer (PPR)",
    "scimilarity": "scimilarity (PPR)",
}

HIERARCHICAL_SHARED_ORDER = [
    "scGPT (PPR)",
    "scCello (PPR)",
    "Geneformer (PPR)",
    "scimilarity (PPR)",
    "Hector (PPR)",
    "Hector (Native Open-Set)",
    "Hector (Native Closed-Set)",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  {title}\n{bar}")


def _show(label: str, table) -> None:
    """Print a small preview of a DataFrame."""
    if table is None:
        print(f"{label}: <None>")
        return
    if hasattr(table, "empty"):
        if table.empty:
            print(f"{label}: <empty>")
            return
        print(f"\n{label}:")
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(table.head(20).to_string(index=False))
        if len(table) > 20:
            print(f"... ({len(table)} rows total)")
        return
    print(f"\n{label}: {table}")


# ---------------------------------------------------------------------------
# Clustering benchmark
# ---------------------------------------------------------------------------

def run_clustering_benchmark(
    *,
    run_embedding: bool,
    reuse_existing_embeddings: bool,
    run_evaluation: bool,
) -> tuple[dict[str, Path], dict[str, object]]:
    """Embeds every model, clusters, and scores against the landmark panel.

    Returns ``(pdf_paths, state)`` where *state* exposes the in-memory dicts
    needed by ``compose_figure_4`` (currently ``clustering_summaries`` and
    ``embedding_run_status``).
    """
    _banner("Clustering benchmark")
    pdf_paths: dict[str, Path] = {}

    device = resolve_default_device()
    available_envs, conda_error = get_available_conda_envs()
    preferred_eval_envs = ["hector", "scgpt", "sccello", "geneformer", "scimilarity"]
    eval_env = next((env for env in preferred_eval_envs if env in available_envs),
                    preferred_eval_envs[0])
    target_n_cells = resolve_target_n_cells(DATASET_PATH, N_CELLS_CLUSTERING)

    with h5py.File(DATASET_PATH, "r") as handle:
        obs_columns = sorted(handle["obs"].keys())

    print(pd.DataFrame({
        "parameter": [
            "dataset", "label_col", "batch_col", "output_dir",
            "run_embedding", "reuse_embeddings", "run_evaluation",
            "device", "n_cells", "n_neighbors", "distance_metric",
            "random_state", "eval_env", "label_col_present", "batch_col_present",
        ],
        "value": [
            str(DATASET_PATH), LABEL_COL, BATCH_COL, str(OUTPUT_DIR),
            run_embedding, reuse_existing_embeddings, run_evaluation,
            device, N_CELLS_CLUSTERING, N_NEIGHBORS, DISTANCE_METRIC,
            RANDOM_STATE, eval_env,
            LABEL_COL in obs_columns, BATCH_COL in obs_columns,
        ],
    }).to_string(index=False))
    if conda_error:
        print(f"Conda discovery issue: {conda_error}")

    # --- Section 4: Embedding generation --------------------------------------
    cell_embeddings_paths: dict[str, Path | None] = {}
    embedding_job_records = []

    for model_name in MODELS_TO_RUN:
        cfg = MODEL_CONFIGS[model_name]
        path, record = ensure_cell_embeddings(
            model_name,
            cfg,
            "clustering",
            DATASET_PATH,
            OUTPUT_DIR,
            N_CELLS_CLUSTERING,
            target_n_cells,
            device,
            available_envs=available_envs,
            run_embedding=run_embedding,
            reuse_existing_embeddings=reuse_existing_embeddings,
        )
        cell_embeddings_paths[model_name] = path
        embedding_job_records.append(record)

    embedding_run_status = pd.DataFrame(embedding_job_records)
    _show("clustering embedding_run_status", embedding_run_status)

    # --- Section 4a: Efficiency timing ----------------------------------------
    # Supplement S7 renders this from the persisted clustering_embedding_run_status.csv.

    # --- Section 5: Cell-embedding inventory ----------------------------------
    inventory_rows = []
    for model_name, path in cell_embeddings_paths.items():
        if path is None or not path.exists():
            inventory_rows.append({
                "model": model_name, "status": "missing_cell_embeddings",
                "cell_embeddings_path": str(path) if path is not None else None,
                "n_rows": None, "embedding_dim": None,
                "has_row_ids": False, "has_sampled_obs_indices": False,
                "has_kept_row_indices": False, "source": None,
            })
            continue
        data = load_embeddings(str(path))
        inventory_rows.append({
            "model": model_name, "status": "ready",
            "cell_embeddings_path": str(path),
            "n_rows": len(data["cell_types"]),
            "embedding_dim": int(data["embedding_dim"]),
            "has_row_ids": "row_ids" in data,
            "has_sampled_obs_indices": "sampled_obs_indices" in data,
            "has_kept_row_indices": "kept_row_indices" in data,
            "source": data.get("source"),
        })
    cell_embeddings_inventory = pd.DataFrame(inventory_rows)
    _show("clustering cell_embeddings_inventory", cell_embeddings_inventory)

    # --- Section 6: Clustering evaluation -------------------------------------
    evaluation_result_paths: dict[str, Path] = {}
    eval_job_records = []
    clustering_script = MODEL_EVAL_DIR / "clustering_evaluate.py"

    for model_name in MODELS_TO_RUN:
        cfg = MODEL_CONFIGS[model_name]
        cell_embeddings_path = cell_embeddings_paths[model_name]
        result_path = OUTPUT_DIR / f"{cfg['cell_embeddings_prefix']}_clustering_metrics.pkl"
        evaluation_result_paths[model_name] = result_path

        common = dict(
            model=model_name, eval_env=eval_env, output_path=str(result_path),
            label_col=LABEL_COL, batch_col=BATCH_COL,
            n_neighbors=N_NEIGHBORS, distance_metric=DISTANCE_METRIC,
            elapsed_seconds=0.0, returncode=None, stdout_tail="",
        )
        if eval_env not in available_envs:
            eval_job_records.append({
                **common, "status": "missing_eval_env",
                "stderr_tail": f"Evaluation env '{eval_env}' is not installed locally.",
            })
            continue
        if cell_embeddings_path is None or not cell_embeddings_path.exists():
            eval_job_records.append({
                **common, "status": "missing_cell_embeddings",
                "stderr_tail": "Cell embeddings not found.",
            })
            continue
        if not run_evaluation:
            eval_job_records.append({
                **common, "status": "cached_mode", "stderr_tail": "",
                "stdout_tail": "Clustering evaluation skipped because run_evaluation=False.",
            })
            continue
        eval_job_records.append(run_clustering_evaluation_job(
            model_name, eval_env, clustering_script, cell_embeddings_path,
            DATASET_PATH, result_path, LABEL_COL, BATCH_COL,
            N_NEIGHBORS, DISTANCE_METRIC, LEIDEN_RESOLUTIONS, RANDOM_STATE,
        ))

    evaluation_run_status = pd.DataFrame(eval_job_records)
    _show("clustering evaluation_run_status", evaluation_run_status)

    # --- Section 7: Results ---------------------------------------------------
    evaluation_results: dict[str, dict] = {}
    summary_rows = []
    resolution_tables: dict[str, pd.DataFrame] = {}
    cluster_tables: dict[str, pd.DataFrame] = {}
    summary_comparison = pd.DataFrame()

    for model_name, result_path in evaluation_result_paths.items():
        if not result_path.exists():
            continue
        result = load_evaluation_results(str(result_path))
        evaluation_results[model_name] = result
        summary_rows.append(result["summary"])
        resolution_tables[model_name] = result["resolution_sweep"]
        cluster_tables[model_name] = result["cluster_assignments"]

    if summary_rows:
        summary_comparison = pd.DataFrame(summary_rows)
        summary_comparison["AvgBio"] = summary_comparison[
            ["NMI_cluster/label", "ARI_cluster/label", "ASW_label"]
        ].mean(axis=1)
        summary_comparison["AvgBatch"] = summary_comparison[
            ["ASW_batch", "graph_conn"]
        ].mean(axis=1)
        summary_comparison = summary_comparison.rename(columns={
            "model": "Model",
            "best_resolution": "Best Resolution",
            "NMI_cluster/label": "NMI",
            "ARI_cluster/label": "ARI",
            "ASW_label": "ASW",
            "ASW_batch": "ASWb",
            "graph_conn": "GraphConn",
        })
        ordered_cols = ["Model", "Best Resolution", "NMI", "ARI", "ASW",
                        "AvgBio", "ASWb", "GraphConn", "AvgBatch", "overall_score"]
        summary_comparison = (
            summary_comparison[ordered_cols]
            .sort_values(["AvgBio", "AvgBatch", "Model"],
                         ascending=[False, False, True])
            .reset_index(drop=True)
        )
        _show("clustering summary_comparison", summary_comparison)
    else:
        print("No clustering evaluation results are available yet.")

    # --- Section 7a: UMAP grid ------------------------------------------------
    # The per-model UMAP embeddings are shown in Figure 4 panel a (coarse
    # lineage) and supplement S3 (broad class + donor), both built from
    # result/data/plot_cache/umap_coords.pkl.

    # --- Section 7b: Clustering summaries -------------------------------------
    # Kept for the state dict and supplement S1 (scIB table).
    clustering_summaries = {
        m: r["summary"] for m, r in evaluation_results.items() if "summary" in r
    }

    # --- Section 8: Resolution sweep ------------------------------------------
    resolution_overview = pd.DataFrame()
    if resolution_tables:
        resolution_overview = (
            pd.concat(resolution_tables.values(), ignore_index=True)
            .sort_values(["model", "resolution"])
            .reset_index(drop=True)
        )
        _show(
            "Best resolution per model",
            resolution_overview.loc[
                resolution_overview["is_best"],
                ["model", "resolution", "n_clusters",
                 "NMI_cluster/label", "ARI_cluster/label"],
            ],
        )
    else:
        print("No resolution sweep tables are available yet.")

    # --- Section 9: Save outputs ---------------------------------------------
    combined_results = {
        "config": {
            "dataset_path": str(DATASET_PATH), "label_col": LABEL_COL,
            "batch_col": BATCH_COL, "eval_env": eval_env,
            "run_embedding": run_embedding,
            "reuse_existing_embeddings": reuse_existing_embeddings,
            "run_evaluation": run_evaluation, "device": device,
            "n_cells": N_CELLS_CLUSTERING, "n_neighbors": N_NEIGHBORS,
            "distance_metric": DISTANCE_METRIC, "random_state": RANDOM_STATE,
            "models_to_run": MODELS_TO_RUN,
        },
        "embedding_run_status": embedding_run_status,
        "cell_embeddings_inventory": cell_embeddings_inventory,
        "evaluation_run_status": evaluation_run_status,
        "summary_comparison": summary_comparison,
        "resolution_overview": resolution_overview,
        "evaluation_results": evaluation_results,
    }
    save_results(combined_results, str(OUTPUT_DIR / "clustering_benchmark_results.pkl"))
    for name, table in [
        ("clustering_embedding_run_status.csv", embedding_run_status),
        ("clustering_cell_embeddings_inventory.csv", cell_embeddings_inventory),
        ("clustering_evaluation_run_status.csv", evaluation_run_status),
        ("clustering_summary_comparison.csv", summary_comparison),
        ("clustering_resolution_overview.csv", resolution_overview),
    ]:
        if table is not None and hasattr(table, "to_csv") and not table.empty:
            table.to_csv(OUTPUT_DIR / name, index=False)

    state: dict[str, object] = {
        "clustering_summaries": clustering_summaries,
        "embedding_run_status": embedding_run_status,
    }
    return pdf_paths, state


# ---------------------------------------------------------------------------
# Prediction benchmark
# ---------------------------------------------------------------------------

def run_prediction_benchmark(
    *,
    run_embedding: bool,
    reuse_existing_ts_embeddings: bool,
    run_hector_native: bool,
) -> tuple[dict[str, Path], dict[str, object]]:
    """Runs the zero-shot prediction benchmark for every model.

    Returns ``(pdf_paths, state)`` where *state* exposes the in-memory dicts
    needed downstream (``prediction_summaries``, ``hop_data``,
    ``prediction_tables``).
    """
    _banner("Prediction benchmark")
    pdf_paths: dict[str, Path] = {}

    bridge_data_path = find_existing_path(
        DATASET_PATH.name, start=DATASET_PATH.parent
    ) or DATASET_PATH
    query_data_path = find_existing_path(
        QUERY_DATA_PATH.name, start=QUERY_DATA_PATH.parent
    ) or QUERY_DATA_PATH

    device = resolve_default_device()
    available_envs, conda_error = get_available_conda_envs()
    bridge_target_n_cells = resolve_target_n_cells(bridge_data_path, BRIDGE_N_CELLS)
    query_target_n_cells = resolve_target_n_cells(query_data_path, QUERY_N_CELLS)

    print(pd.DataFrame({
        "parameter": [
            "bridge_data", "query_data", "ontology", "output_dir",
            "strict_unseen", "run_embedding", "reuse_ts",
            "device", "bridge_n", "query_n", "models",
        ],
        "value": [
            str(bridge_data_path), str(query_data_path), str(ONTOLOGY_PATH), str(OUTPUT_DIR),
            STRICT_UNSEEN, run_embedding, reuse_existing_ts_embeddings,
            device, BRIDGE_N_CELLS, QUERY_N_CELLS, MODELS_TO_RUN,
        ],
    }).to_string(index=False))
    if conda_error:
        print(f"Conda discovery issue: {conda_error}")

    # --- Section 3+4: Embedding generation ------------------------------------
    bridge_cell_embeddings_paths: dict[str, Path | None] = {}
    query_cell_embeddings_paths: dict[str, Path | None] = {}
    embedding_job_records = []

    for model_name in MODELS_TO_RUN:
        cfg = MODEL_CONFIGS[model_name]
        bridge_path, bridge_record = ensure_cell_embeddings(
            model_name, cfg, "bridge", bridge_data_path, OUTPUT_DIR,
            BRIDGE_N_CELLS, bridge_target_n_cells, device,
            available_envs=available_envs,
            run_embedding=run_embedding,
            reuse_existing_embeddings=reuse_existing_ts_embeddings,
        )
        bridge_cell_embeddings_paths[model_name] = bridge_path
        embedding_job_records.append(bridge_record)

        query_path, query_record = ensure_cell_embeddings(
            model_name, cfg, "query", query_data_path, OUTPUT_DIR,
            QUERY_N_CELLS, query_target_n_cells, device,
            available_envs=available_envs,
            run_embedding=run_embedding,
            reuse_existing_embeddings=reuse_existing_ts_embeddings,
        )
        query_cell_embeddings_paths[model_name] = query_path
        embedding_job_records.append(query_record)

    embedding_run_status = pd.DataFrame(embedding_job_records)
    _show("prediction embedding_run_status", embedding_run_status)

    # --- Section 5: Ontology + dataset metadata -------------------------------
    ontology = CellOntologyPPR(ontology_path=ONTOLOGY_PATH)
    ontology.load_ontology()

    bridge_rows = read_h5ad_observation_metadata(bridge_data_path)
    query_rows = read_h5ad_observation_metadata(query_data_path)
    bridge_name_to_id, bridge_id_to_name = build_exact_label_map(bridge_rows)
    query_name_to_id, query_id_to_name = build_exact_label_map(query_rows)

    bridge_label_table = derive_observed_label_table(bridge_rows, ontology=ontology)
    candidate_label_table = select_candidate_label_table(
        query_rows,
        bridge_rows["cell_type_id"].dropna().astype(str).unique().tolist(),
        ontology=ontology,
        strict_unseen=STRICT_UNSEEN,
    )
    candidate_label_table = (
        candidate_label_table.sort_values(["cell_type", "cell_type_id"])
        .reset_index(drop=True)
    )
    candidate_ids = candidate_label_table["cell_type_id"].astype(str).tolist()
    candidate_name_map = dict(
        candidate_label_table[["cell_type_id", "cell_type"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    # Every summarize_prediction_table call pads to this, so a model that drops
    # cells it can't represent (e.g. scCello's tokenizer) is scored wrong on
    # them rather than excluded.
    expected_query_counts = expected_query_counts_from_table(candidate_label_table)
    print(f"\nCandidate labels: {len(candidate_ids)}  |  Strict unseen: {STRICT_UNSEEN}")

    # --- Section 6: Load embedding artifacts ----------------------------------
    cell_embeddings_inventory_rows = []
    model_cell_embeddings: dict[str, dict] = {}

    for model_name in MODELS_TO_RUN:
        bridge_path = bridge_cell_embeddings_paths.get(model_name)
        query_path = query_cell_embeddings_paths.get(model_name)

        entry: dict = {
            "bridge_cell_embeddings_path": bridge_path if bridge_path and bridge_path.exists() else None,
            "query_cell_embeddings_path": query_path if query_path and query_path.exists() else None,
        }
        bridge_row_count = query_row_count = None
        bridge_row_id_source = query_row_id_source = "missing"
        notes = []

        if bridge_path is not None and bridge_path.exists():
            bridge_data = load_embeddings(str(bridge_path))
            bridge_metadata = get_embedding_metadata_frame(
                bridge_data, ontology=ontology, strict_ids=True,
                label_map=bridge_name_to_id, canonical_name_map=bridge_id_to_name,
            )
            entry.update(bridge_data=bridge_data, bridge_metadata=bridge_metadata)
            bridge_row_count = len(bridge_metadata)
            bridge_row_id_source = "cell_embeddings" if "row_ids" in bridge_data else "legacy_synthetic"
            if bridge_metadata["cell_type_id"].isna().any():
                notes.append("bridge: unresolved cell_type_id values")
        else:
            notes.append("bridge cell embeddings not found")

        if query_path is not None and query_path.exists():
            query_data = load_embeddings(str(query_path))
            query_metadata = get_embedding_metadata_frame(
                query_data, ontology=ontology, strict_ids=True,
                label_map=query_name_to_id, canonical_name_map=query_id_to_name,
            )
            entry.update(query_data=query_data, query_metadata=query_metadata)
            query_row_count = len(query_metadata)
            query_row_id_source = "cell_embeddings" if "row_ids" in query_data else "legacy_synthetic"
            if query_metadata["cell_type_id"].isna().any():
                notes.append("query: unresolved cell_type_id values")
            if "row_ids" not in query_data:
                notes.append("query: lacks row_ids")
        else:
            notes.append("query cell embeddings not found")

        model_cell_embeddings[model_name] = entry
        cell_embeddings_inventory_rows.append({
            "model": model_name,
            "bridge_cell_embeddings": str(bridge_path) if bridge_path and bridge_path.exists() else None,
            "bridge_rows": bridge_row_count,
            "bridge_row_id_source": bridge_row_id_source,
            "query_cell_embeddings": str(query_path) if query_path and query_path.exists() else None,
            "query_rows": query_row_count,
            "query_row_id_source": query_row_id_source,
            "notes": "; ".join(notes),
        })

    cell_embeddings_inventory = pd.DataFrame(cell_embeddings_inventory_rows)
    _show("prediction cell_embeddings_inventory", cell_embeddings_inventory)

    # --- Section 7: Bridge construction ---------------------------------------
    expected_bridge_ids = set(
        bridge_rows["cell_type_id"].dropna().astype(str).unique()
    )
    bridge_models: dict[str, dict] = {}
    bridge_status_rows = []

    for model_name, entry in model_cell_embeddings.items():
        if "bridge_data" not in entry:
            bridge_status_rows.append({
                "model": model_name, "status": "missing_bridge_cell_embeddings",
                "n_bridge_rows": 0, "n_bridge_labels": 0,
                "missing_ids": len(expected_bridge_ids), "extra_ids": 0,
            })
            continue
        try:
            prototypes, label_table, bridge_metadata = build_bridge_prototypes_from_embeddings(
                entry["bridge_data"], ontology=ontology,
                label_map=bridge_name_to_id, canonical_name_map=bridge_id_to_name,
            )
            produced = set(label_table["bridge_label_id"].astype(str))
            bridge_models[model_name] = {
                "prototypes": prototypes,
                "label_table": label_table,
                "bridge_metadata": bridge_metadata,
            }
            bridge_status_rows.append({
                "model": model_name, "status": "ready",
                "n_bridge_rows": len(bridge_metadata),
                "n_bridge_labels": len(label_table),
                "missing_ids": len(expected_bridge_ids - produced),
                "extra_ids": len(produced - expected_bridge_ids),
            })
        except Exception as exc:
            bridge_status_rows.append({
                "model": model_name, "status": f"failed: {exc}",
                "n_bridge_rows": 0, "n_bridge_labels": 0,
                "missing_ids": len(expected_bridge_ids), "extra_ids": 0,
            })

    bridge_status = pd.DataFrame(bridge_status_rows)
    _show("bridge_status", bridge_status)

    # --- Section 8: Prediction ------------------------------------------------
    # Neighborhood metric: candidate-scoped (17 unseen labels) and truth-inclusive.
    # Scoring against the full 3,129-label ontology while excluding truth from
    # positives makes the most-accurate model look bad on Hit@k; this scope
    # gives a "did we put truth or an immediate ontology neighbor in the top-k"
    # reading that moves monotonically with AUC.
    NEIGHBORHOOD_INCLUDE_TRUTH = True

    prediction_tables: dict[str, dict] = {}
    prediction_status_rows = []
    summary_rows = []
    neighborhood_metric_tables: dict[str, pd.DataFrame] = {}
    # State per PPR model so we can recompute the neighborhood metric on a
    # shared (Hector-native-supported) candidate subset after native runs.
    ppr_state: dict[str, dict] = {}

    for model_name in MODELS_TO_RUN:
        entry = model_cell_embeddings.get(model_name, {})
        bridge_entry = bridge_models.get(model_name)

        if bridge_entry is None:
            prediction_status_rows.append({
                "model": model_name, "status": "skipped",
                "reason": "bridge centroids not available", "n_query_rows": 0,
            })
            continue
        if "query_data" not in entry:
            prediction_status_rows.append({
                "model": model_name, "status": "skipped",
                "reason": "query cell embeddings not available", "n_query_rows": 0,
            })
            continue
        if "row_ids" not in entry["query_data"]:
            prediction_status_rows.append({
                "model": model_name, "status": "skipped",
                "reason": "query cell embeddings lack row_ids", "n_query_rows": 0,
            })
            continue

        query_metadata = get_embedding_metadata_frame(
            entry["query_data"], ontology=ontology, strict_ids=True,
            label_map=query_name_to_id, canonical_name_map=query_id_to_name,
        )
        query_embeddings = np.asarray(entry["query_data"]["embeddings"])

        eval_mask = (
            query_metadata["cell_type_id"].astype(str).isin(candidate_ids)
            if STRICT_UNSEEN
            else query_metadata["cell_type_id"].notna()
        )
        query_metadata_eval = query_metadata.loc[eval_mask].reset_index(drop=True)
        query_embeddings_eval = query_embeddings[eval_mask.to_numpy()]

        if len(query_metadata_eval) == 0:
            prediction_status_rows.append({
                "model": model_name, "status": "skipped",
                "reason": "no query rows after candidate filtering",
                "n_query_rows": 0,
            })
            continue

        query_to_bridge, bridge_id_order = compute_query_to_bridge_similarity(
            query_embeddings_eval, bridge_entry["prototypes"],
            temperature=TEMPERATURE,
        )

        candidate_to_bridge = ontology.compute_candidate_to_bridge_similarity(
            candidate_ids, bridge_id_order, alpha=PPR_ALPHA,
        )
        ppr_ids, ppr_scores = predict_with_candidate_profiles(
            query_to_bridge, candidate_to_bridge, candidate_ids,
        )

        ppr_neighborhood_summary = compute_neighborhood_metrics_from_rank_profiles(
            query_to_bridge,
            candidate_to_bridge,
            candidate_ids,
            query_metadata_eval,
            ontology,
            batch_size=1024,
            hit_ks=(3, 5),
            include_truth_in_targets=NEIGHBORHOOD_INCLUDE_TRUTH,
        )

        ppr_state[model_name] = {
            "query_to_bridge": query_to_bridge,
            "bridge_id_order": bridge_id_order,
            "query_metadata_eval": query_metadata_eval,
        }

        ppr_table = assemble_prediction_table(
            query_metadata_eval, predicted_ids=ppr_ids,
            scores=ppr_scores, method="ppr_candidate",
            model_name=model_name, ontology=ontology,
            predicted_name_map=candidate_name_map,
        )

        ppr_table = annotate_prediction_table(ppr_table, ontology_path=ONTOLOGY_PATH)

        ppr_summary = summarize_prediction_table(
            ppr_table, expected_query_counts=expected_query_counts,
        )
        ppr_summary["neighborhood_summary"] = {
            k: v for k, v in ppr_neighborhood_summary.items() if k != "cell_table"
        }

        ppr_neighborhood_table = ppr_neighborhood_summary["cell_table"].copy()
        ppr_neighborhood_table.insert(0, "method", "ppr_candidate")
        ppr_neighborhood_table.insert(0, "model", model_name)
        neighborhood_metric_tables[f"{model_name}/ppr_candidate"] = ppr_neighborhood_table

        prediction_tables[model_name] = {
            "ppr_candidate": ppr_table,
            "candidate_to_bridge_shape": candidate_to_bridge.shape,
            "query_to_bridge_shape": query_to_bridge.shape,
            "neighborhood_candidate_count": len(candidate_ids),
        }
        summary_rows.append(flatten_summary_row(model_name, "ppr_candidate", ppr_summary))
        prediction_status_rows.append({
            "model": model_name, "status": "ready",
            "reason": "", "n_query_rows": len(query_metadata_eval),
        })

    prediction_status = pd.DataFrame(prediction_status_rows)
    summary_table = pd.DataFrame(summary_rows)
    _show("prediction_status", prediction_status)

    # --- Section 9b: Hector native predictions --------------------------------
    hector_native_open_table = None
    hector_native_closed_table = None
    hector_native_open_summary = None
    hector_native_closed_summary = None
    missing_native_candidate_ids: list = []

    if run_hector_native:
        try:
            import anndata as ad
            import hector
        except ImportError as exc:
            print(f"Hector native predictions skipped (import error): {exc}")
        else:
            query_adata = ad.read_h5ad(query_data_path)
            hector_predictor = hector.HECTOR("human", verbose=True)
            hector_native_result = run_hector_native_predictions(
                query_adata,
                query_rows,
                hector_predictor=hector_predictor,
                candidate_ids=candidate_ids,
                ontology=ontology,
                ontology_path=ONTOLOGY_PATH,
                expected_query_counts=expected_query_counts,
                strict_unseen=STRICT_UNSEEN,
                candidate_name_map=candidate_name_map,
                neighborhood_candidate_ids=candidate_ids,
                include_truth_in_neighborhood=NEIGHBORHOOD_INCLUDE_TRUTH,
            )

            hector_native_open_table = hector_native_result["open_table"]
            hector_native_closed_table = hector_native_result["closed_table"]
            hector_native_open_summary = hector_native_result["open_summary"]
            hector_native_closed_summary = hector_native_result["closed_summary"]
            missing_native_candidate_ids = hector_native_result["missing_candidate_ids"]
            hector_native_open_neighborhood_metric_table = hector_native_result.get(
                "open_neighborhood_metric_table"
            )
            hector_native_closed_neighborhood_metric_table = hector_native_result.get(
                "closed_neighborhood_metric_table"
            )

            if hector_native_open_neighborhood_metric_table is not None:
                tbl = hector_native_open_neighborhood_metric_table.copy()
                tbl.insert(0, "method", "native_open_set")
                tbl.insert(0, "model", "Hector")
                neighborhood_metric_tables["Hector/native_open_set"] = tbl
            if hector_native_closed_neighborhood_metric_table is not None:
                tbl = hector_native_closed_neighborhood_metric_table.copy()
                tbl.insert(0, "method", "native_closed_set")
                tbl.insert(0, "model", "Hector")
                neighborhood_metric_tables["Hector/native_closed_set"] = tbl

            hector_tables = prediction_tables.setdefault("Hector", {})
            if hector_native_open_table is not None:
                hector_tables["native_open_set"] = hector_native_open_table
            else:
                hector_tables.pop("native_open_set", None)
            if hector_native_closed_table is not None:
                hector_tables["native_closed_set"] = hector_native_closed_table
            else:
                hector_tables.pop("native_closed_set", None)

            native_summary_rows = pd.DataFrame(hector_native_result["summary_rows"])
            if not summary_table.empty:
                base_summary_table = summary_table.loc[
                    ~(
                        (summary_table["model"] == "Hector")
                        & summary_table["method"].isin(["native_open_set", "native_closed_set"])
                    )
                ]
            else:
                base_summary_table = pd.DataFrame()
            summary_table = (
                pd.concat([base_summary_table, native_summary_rows], ignore_index=True)
                if not native_summary_rows.empty
                else base_summary_table.reset_index(drop=True)
            )

            if not prediction_status.empty:
                base_prediction_status = prediction_status.loc[
                    ~prediction_status["model"].isin(
                        ["Hector Native Closed-Set", "Hector Native Open-Set"]
                    )
                ]
            else:
                base_prediction_status = pd.DataFrame()
            prediction_status = pd.concat(
                [base_prediction_status, pd.DataFrame(hector_native_result["status_rows"])],
                ignore_index=True,
            )

            if missing_native_candidate_ids:
                print(f"Native closed-set skipped; missing candidate IDs: {missing_native_candidate_ids}")

            # Shared-vocabulary recompute: if Hector native could not score
            # every strict-unseen candidate, restrict the PPR neighborhood
            # metric to the same intersection so every bar in the figure ranks
            # against the same column set.
            supported_native_ids = list(
                hector_native_result.get("supported_candidate_ids") or []
            )
            shared_candidate_ids = [
                cid for cid in candidate_ids if cid in set(supported_native_ids)
            ] if supported_native_ids else list(candidate_ids)

            if shared_candidate_ids and shared_candidate_ids != list(candidate_ids):
                print(
                    "Recomputing PPR neighborhood metrics on the "
                    f"{len(shared_candidate_ids)}-label subset that Hector "
                    "native also supports."
                )
                shared_candidate_to_bridge_cache: dict = {}
                metric_cols = [
                    "neighborhood_roc_auc",
                    "neighborhood_hit_rate_at_3",
                    "neighborhood_hit_rate_at_5",
                    "neighborhood_n_scored_cells",
                    "neighborhood_n_skipped_cells",
                ]
                for state_model_name, state in ppr_state.items():
                    bridge_id_order = state["bridge_id_order"]
                    cache_key = (tuple(bridge_id_order), tuple(shared_candidate_ids))
                    shared_c2b = shared_candidate_to_bridge_cache.get(cache_key)
                    if shared_c2b is None:
                        shared_c2b = ontology.compute_candidate_to_bridge_similarity(
                            shared_candidate_ids, bridge_id_order, alpha=PPR_ALPHA,
                        )
                        shared_candidate_to_bridge_cache[cache_key] = shared_c2b
                    new_summary = compute_neighborhood_metrics_from_rank_profiles(
                        state["query_to_bridge"],
                        shared_c2b,
                        shared_candidate_ids,
                        state["query_metadata_eval"],
                        ontology,
                        batch_size=1024,
                        hit_ks=(3, 5),
                        include_truth_in_targets=NEIGHBORHOOD_INCLUDE_TRUTH,
                    )
                    row_mask = (
                        (summary_table["model"] == state_model_name)
                        & (summary_table["method"] == "ppr_candidate")
                    )
                    for col in metric_cols:
                        if col in new_summary and col in summary_table.columns:
                            summary_table.loc[row_mask, col] = new_summary[col]
    else:
        print("Hector native predictions skipped (run_hector_native=False).")

    # --- Section 9c: Hierarchical accuracy chart ------------------------------
    prediction_summaries: dict[str, dict] = {}
    for model_name, tables in prediction_tables.items():
        ppr_table = tables.get("ppr_candidate")
        if ppr_table is not None and hasattr(ppr_table, "columns"):
            prediction_summaries[model_name] = summarize_prediction_table(
                ppr_table, expected_query_counts=expected_query_counts,
            )

    # These per-cell acc / macro F1 / hierarchical macro F1 numbers are drawn in
    # Figure 4 panel c. prediction_summaries is also consumed by the hop section
    # below and returned in the prediction state dict.

    # --- Section 9d: Neighborhood metrics ------------------------------------
    # The neighborhood ROC-AUC is persisted in summary_table.csv and plotted in
    # supplement S2 (embedding quality vs zero-shot).

    # --- Section 9d (hop-distance): ontology hop explainer, cumulative hop, ---
    # --- and depth-conditioned outcome panels --------------------------------
    from Hierarchical_metrics.cell_ontology_metrics import load_cell_ontology

    cl_dag = load_cell_ontology(ONTOLOGY_PATH)
    bridge_labels_for_baseline = bridge_rows["cell_type_id"].dropna().astype(str)

    random_baseline_n = None
    random_baseline_truth = None
    for mn in MODELS_TO_RUN:
        tables = prediction_tables.get(mn, {})
        ppr_table = tables.get("ppr_candidate")
        if ppr_table is not None and hasattr(ppr_table, "columns"):
            random_baseline_n = len(ppr_table)
            random_baseline_truth = ppr_table["truth_cell_type_id"].astype(str)
            break

    hop_data: dict[str, pd.Series] = {}
    pred_depths: dict[str, pd.Series] = {}
    gt_depths_dict: dict[str, pd.Series] = {}
    accuracies: dict[str, float] = {}

    # The matched arm chooses among the same held-out types as the other models;
    # the open arm chooses among all ~1,400 terms it can name, so its exact-match
    # rate is not directly comparable. Supplement S5 draws both.
    hector_native_arms = [
        ("Hector (Native)", hector_native_closed_table, hector_native_closed_summary),
        ("Hector (Native, open)", hector_native_open_table, hector_native_open_summary),
    ]
    hector_native_table = (
        hector_native_open_table
        if hector_native_open_table is not None
        else hector_native_closed_table
    )

    for hector_native_label, native_table, native_summary in hector_native_arms:
        if native_table is None:
            continue
        pred_ids = native_table["predicted_cell_type_id"].astype(str)
        truth_ids = native_table["truth_cell_type_id"].astype(str)
        hop_data[hector_native_label] = compute_hop_distance(pred_ids, truth_ids, dag=cl_dag)
        pred_depths[hector_native_label] = compute_label_depths(pred_ids, dag=cl_dag)
        gt_depths_dict[hector_native_label] = compute_label_depths(truth_ids, dag=cl_dag)
        if native_summary is not None:
            accuracies[hector_native_label] = native_summary["accuracy"]
        print(f"{hector_native_label}: {len(pred_ids)} predictions")
    else:
        print("Hector native predictions not available; skipping hop distance for native.")

    for model_name in MODELS_TO_RUN:
        tables = prediction_tables.get(model_name, {})
        ppr_table = tables.get("ppr_candidate")
        if ppr_table is None or not hasattr(ppr_table, "columns"):
            print(f"{model_name}: PPR predictions not available; skipping hop distance.")
            continue
        display_name = DISPLAY_NAME_BY_MODEL.get(model_name, f"{model_name} (PPR)")
        pred_ids = ppr_table["predicted_cell_type_id"].astype(str)
        truth_ids = ppr_table["truth_cell_type_id"].astype(str)
        hop_data[display_name] = compute_hop_distance(pred_ids, truth_ids, dag=cl_dag)
        pred_depths[display_name] = compute_label_depths(pred_ids, dag=cl_dag)
        gt_depths_dict[display_name] = compute_label_depths(truth_ids, dag=cl_dag)
        ppr_summary = prediction_summaries.get(model_name)
        if ppr_summary is not None:
            accuracies[display_name] = ppr_summary["accuracy"]
        print(f"{display_name}: {len(pred_ids)} predictions")

    if random_baseline_n is not None and random_baseline_truth is not None:
        random_label = "Random Baseline"
        random_baseline_preds = generate_random_baseline(
            bridge_labels_for_baseline, random_baseline_n, seed=42
        )
        hop_data[random_label] = compute_hop_distance(
            random_baseline_preds, random_baseline_truth, dag=cl_dag
        )
        pred_depths[random_label] = compute_label_depths(
            random_baseline_preds, dag=cl_dag
        )
        gt_depths_dict[random_label] = compute_label_depths(
            random_baseline_truth, dag=cl_dag
        )
        accuracies[random_label] = float(
            (random_baseline_preds.values == random_baseline_truth.values).mean()
        )
        print(f"Random Baseline: {len(random_baseline_preds)} predictions")
    else:
        print("Random baseline skipped (no PPR predictions to size against).")

    # The hop distances are drawn in Figure 4 panel d (the hop curve with the
    # explainer schematic as an inset), which reads hop_data from
    # result/data/plot_cache/hop_data.pkl (written by the plot precompute stage).

    # Depth-conditioned outcomes are computed by supplement S5 itself from the
    # prediction CSVs on fixed depth bins (model_eval/plot/precompute.py:_depth_frame).

    # Figure 4 composition happens in the top-level main() so it can also see
    # clustering-stage in-memory state (clustering_summaries,
    # embedding_run_status). The data needed is returned below in the *state*
    # dict.

    # --- Section 10: Save outputs --------------------------------------------
    for name, tbl in [
        ("embedding_run_status.csv", embedding_run_status),
        ("cell_embeddings_inventory.csv", cell_embeddings_inventory),
        ("bridge_status.csv", bridge_status),
        ("prediction_status.csv", prediction_status),
        ("bridge_label_table.csv", bridge_label_table),
        ("candidate_label_table.csv", candidate_label_table),
        ("summary_table.csv", summary_table),
    ]:
        if tbl is not None and hasattr(tbl, "to_csv"):
            tbl.to_csv(OUTPUT_DIR / name, index=False)

    for model_name, model_tables in prediction_tables.items():
        prefix = MODEL_CONFIGS[model_name]["cell_embeddings_prefix"]
        for method_name, tbl in model_tables.items():
            if not hasattr(tbl, "to_csv"):
                continue
            tbl.to_csv(
                OUTPUT_DIR / f"{prefix}_{method_name}_predictions.csv",
                index=False,
            )

    state: dict[str, object] = {
        "prediction_summaries": prediction_summaries,
        "hop_data": hop_data,
        "prediction_tables": prediction_tables,
    }
    return pdf_paths, state


# ---------------------------------------------------------------------------
# Native-decoder comparison (Supplementary Fig. 11c)
# ---------------------------------------------------------------------------

def run_native_comparison_benchmark(
    *,
    run_native_comparison: bool,
    reuse_existing_native_comparison: bool,
) -> dict[str, object]:
    """HECTOR/OnClass/scCello scored with their native decoders instead of
    the shared read-out, plus the open-vocabulary stress test. Reads
    `hector_native_closed_set_predictions.csv`, written by
    `run_prediction_benchmark` above, so it must run after that — from
    cache is fine too, this only checks the file exists.

    Returns a status dict; writes `native_comparison_run_status.csv` and the
    prediction/summary CSVs Supplementary Fig. 11 panel c reads, all
    under `OUTPUT_DIR` like every other step in this file.
    """
    _banner("Native-decoder comparison (Supplementary Fig. 11c)")

    if not run_native_comparison:
        print("Native-decoder comparison skipped (--skip-native-comparison).")
        return {"status_rows": pd.DataFrame(), "ran": False}

    hector_native_path = OUTPUT_DIR / "hector_native_closed_set_predictions.csv"
    if not hector_native_path.exists():
        print(
            f"Native-decoder comparison skipped: {hector_native_path.name} not "
            "found. It needs Hector's native closed-set predictions -- run "
            "without --no-hector-native at least once first."
        )
        return {"status_rows": pd.DataFrame(), "ran": False}

    available_envs, conda_error = get_available_conda_envs()
    if conda_error:
        print(f"Conda discovery issue: {conda_error}")

    status_rows = []
    for step in NATIVE_COMPARISON_STEPS:
        check_path = step["check_path"]
        if reuse_existing_native_comparison and check_path.exists():
            print(f"[native-comparison] {step['name']}: reusing existing {check_path.name}")
            status_rows.append({
                "step": step["name"], "env": step["env"], "status": "reused_existing",
                "elapsed_seconds": 0.0, "returncode": 0, "log_path": "",
            })
            continue
        if step["env"] not in available_envs:
            print(f"[native-comparison] {step['name']}: conda env '{step['env']}' "
                  "not installed, skipping")
            status_rows.append({
                "step": step["name"], "env": step["env"], "status": "missing_env",
                "elapsed_seconds": 0.0, "returncode": None, "log_path": "",
            })
            continue
        record = run_native_comparison_job(
            step["name"], step["script"], step["env"], OUTPUT_DIR,
            extra_env=step.get("extra_env"),
        )
        status_rows.append(record)
        if record["status"] != "completed":
            print(f"[native-comparison] {step['name']} failed (see "
                  f"{record['log_path']}); later steps may be missing inputs.")

    summary = NATIVE_COMPARISON_SUMMARY_STEP
    if summary["env"] in available_envs:
        status_rows.append(
            run_native_comparison_job(
                summary["name"], summary["script"], summary["env"], OUTPUT_DIR,
            )
        )
    else:
        print(f"[native-comparison] {summary['name']}: conda env "
              f"'{summary['env']}' not installed, skipping")
        status_rows.append({
            "step": summary["name"], "env": summary["env"], "status": "missing_env",
            "elapsed_seconds": 0.0, "returncode": None, "log_path": "",
        })

    status_table = pd.DataFrame(status_rows)
    if not status_table.empty:
        status_table.to_csv(OUTPUT_DIR / "native_comparison_run_status.csv", index=False)
    _show("native_comparison_run_status", status_table)
    return {"status_rows": status_table, "ran": True}


# ---------------------------------------------------------------------------
# Cache fallback (when --skip-clustering or --skip-prediction is used but we
# still want to compose Figure 4 from prior runs' outputs)
# ---------------------------------------------------------------------------

def _load_clustering_state_from_cache() -> dict[str, object]:
    """Reconstruct the clustering-stage in-memory state from cached files.

    Returns the same ``{clustering_summaries, embedding_run_status}`` dict
    shape that ``run_clustering_benchmark`` would return live. Empty dict if
    the cache is absent.
    """
    state: dict[str, object] = {}
    clustering_summaries: dict[str, dict] = {}
    for model_name, cfg in MODEL_CONFIGS.items():
        prefix = cfg["cell_embeddings_prefix"]
        pkl = OUTPUT_DIR / f"{prefix}_clustering_metrics.pkl"
        if not pkl.exists():
            continue
        try:
            result = load_evaluation_results(str(pkl))
            if "summary" in result:
                clustering_summaries[model_name] = result["summary"]
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to load cached clustering summary for %s: %s",
                model_name, exc,
            )
    if clustering_summaries:
        state["clustering_summaries"] = clustering_summaries

    timing_csv = OUTPUT_DIR / "clustering_embedding_run_status.csv"
    if timing_csv.exists():
        try:
            state["embedding_run_status"] = pd.read_csv(timing_csv)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to load cached embedding run status: %s", exc,
            )
    if not state:
        print("  [info] No cached clustering state available.")
    else:
        print(f"  [info] Loaded cached clustering state for {len(clustering_summaries)} models.")
    return state


def _prediction_tables_with_cached_predictions(
    prediction_tables: dict[str, dict],
    methods: tuple[str, ...],
) -> dict[str, dict]:
    """In-memory prediction tables, backfilled from the CSVs already on disk.

    A step skipped this run (Hector's own inference under --no-hector-native,
    say) leaves its method missing from the in-memory tables even though the
    per-model CSV a previous run wrote is still there and still valid. Without
    this backfill the per-type accuracy CSV gets rewritten without those rows
    and the supplement draws the column blank, which reads as "the model got
    every cell type wrong" instead of "this table was not loaded".
    """
    merged = {model: dict(tables) for model, tables in prediction_tables.items()}
    for model_name, cfg in MODEL_CONFIGS.items():
        prefix = cfg["cell_embeddings_prefix"]
        for method in methods:
            if merged.get(model_name, {}).get(method) is not None:
                continue
            csv_path = OUTPUT_DIR / f"{prefix}_{method}_predictions.csv"
            if not csv_path.exists():
                continue
            try:
                merged.setdefault(model_name, {})[method] = pd.read_csv(csv_path)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Failed to read cached predictions %s: %s", csv_path.name, exc)
                continue
            print(f"  [info] {model_name}/{method} not computed this run; "
                  f"reusing cached {csv_path.name}")
    return merged


def _write_per_type_accuracy_csv(table: pd.DataFrame) -> None:
    """Write the per-type accuracy CSV unless doing so would drop rows.

    Second guard on the same failure as the backfill above: if the fresh table
    covers fewer (model, method) pairs than the file already on disk, keep the
    file and say so rather than silently blanking a supplement column.
    """
    path = OUTPUT_DIR / "per_held_out_type_accuracy.csv"
    new_pairs = set(zip(table["model"], table["method"]))
    if path.exists():
        try:
            existing = pd.read_csv(path)
            old_pairs = (set(zip(existing["model"], existing["method"]))
                         if {"model", "method"} <= set(existing.columns) else set())
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to read existing %s: %s", path.name, exc)
            old_pairs = set()
        lost = sorted(old_pairs - new_pairs)
        if lost:
            print(f"  [warning] Refusing to overwrite {path.name}: the new "
                  f"table is missing {len(lost)} model/method combination(s) "
                  f"the existing file has: "
                  + ", ".join(f"{m}/{k}" for m, k in lost))
            return
    table.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run both benchmarks and export every plot as a PDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip-clustering", action="store_true",
        help="Skip the clustering benchmark.",
    )
    parser.add_argument(
        "--skip-prediction", action="store_true",
        help="Skip the prediction benchmark.",
    )
    parser.add_argument(
        "--no-run-embedding", action="store_true",
        help=("Do not (re)generate any embeddings. Cached embedding files in "
              "result/ are used as-is; models without cached embeddings are "
              "skipped."),
    )
    parser.add_argument(
        "--no-reuse-embeddings", action="store_true",
        help=("Always regenerate embeddings even when a compatible cache is "
              "present. Has no effect when --no-run-embedding is set."),
    )
    parser.add_argument(
        "--no-run-evaluation", action="store_true",
        help=("Do not run clustering evaluation. Cached "
              "*_clustering_metrics.pkl files are used as-is."),
    )
    parser.add_argument(
        "--no-hector-native", action="store_true",
        help=("Skip the Hector native open/closed-set inference step in the "
              "prediction benchmark."),
    )
    parser.add_argument(
        "--skip-native-comparison", action="store_true",
        help=("Skip the native-decoder comparison (HECTOR/OnClass/scCello "
              "with their native decoders; Supplementary Fig. 11c)."),
    )
    parser.add_argument(
        "--no-reuse-native-comparison", action="store_true",
        help=("Always rerun every native-decoder-comparison step even when "
              "its output already exists. Has no effect when "
              "--skip-native-comparison is set."),
    )
    args = parser.parse_args(argv)

    os.chdir(MODEL_EVAL_DIR)  # several helpers expect model_eval as CWD

    all_pdfs: dict[str, Path] = {}
    clustering_state: dict[str, object] = {}
    prediction_state: dict[str, object] = {}

    if not args.skip_clustering:
        pdfs, clustering_state = run_clustering_benchmark(
            run_embedding=not args.no_run_embedding,
            reuse_existing_embeddings=not args.no_reuse_embeddings,
            run_evaluation=not args.no_run_evaluation,
        )
        all_pdfs.update(pdfs)
    else:
        print("\nSkipping clustering benchmark (--skip-clustering).")
        clustering_state = _load_clustering_state_from_cache()

    if not args.skip_prediction:
        pdfs, prediction_state = run_prediction_benchmark(
            run_embedding=not args.no_run_embedding,
            reuse_existing_ts_embeddings=not args.no_reuse_embeddings,
            run_hector_native=not args.no_hector_native,
        )
        all_pdfs.update(pdfs)
    else:
        print("\nSkipping prediction benchmark (--skip-prediction).")

    run_native_comparison_benchmark(
        run_native_comparison=not args.skip_native_comparison,
        reuse_existing_native_comparison=not args.no_reuse_native_comparison,
    )

    # --- Figure 4 + supplements (model_eval/plot package) ------------------
    # Reads the freshly written result/ files (no in-memory state needed).
    if prediction_state:
        try:
            # Both read-outs supplement S4 draws side by side: the shared one
            # every model is given, and HECTOR's own head on the same candidates.
            methods = ("ppr_candidate", "native_closed_set")
            per_type_source = _prediction_tables_with_cached_predictions(
                prediction_state["prediction_tables"], methods,
            )
            per_type_parts = []
            for method in methods:
                part = benchmark_visualize.build_per_type_accuracy_table(
                    per_type_source, method=method,
                )
                if not part.empty:
                    if "method" not in part.columns:
                        part.insert(3, "method", method)
                    per_type_parts.append(part)
            per_type_table = (pd.concat(per_type_parts, ignore_index=True)
                              if per_type_parts else pd.DataFrame())
            if not per_type_table.empty:
                _write_per_type_accuracy_csv(per_type_table)
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "per-type accuracy table failed: %s", exc)

    try:
        from plot import precompute as plot_precompute
        from plot import figure4 as plot_figure4
        from plot import supplements as plot_supplements
        plot_precompute.run_all()
        fig4_pdf = plot_figure4.compose_figure_4(out_dir=FIG_DIR)
        all_pdfs["figure_4"] = fig4_pdf
        print(f"\nFigure 4 -> {fig4_pdf}")
        for _name, _res in plot_supplements.render_all(out_dir=FIG_DIR).items():
            print(f"  supplement {_name}: {_res}")
    except Exception as exc:
        logging.getLogger(__name__).exception("plot stage failed: %s", exc)

    _banner("PDF summary")
    # The composed figure is the only key here; supplements are reported by
    # the render_all loop above.
    expected = ["figure_4"]
    for key in expected:
        path = all_pdfs.get(key)
        if path is None:
            print(f"  [skipped] {key}.pdf")
        else:
            print(f"  [written] {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
