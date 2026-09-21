from __future__ import annotations
import gc
import importlib
import importlib.util
import json
from math import gcd
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import anndata as ad
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
from scipy import sparse
from sklearn.decomposition import PCA

from helpers.downloaded_dataset_prep_helpers import ensure_directory, resolve_dataset_config


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# JAX/XLA's on-disk compilation cache is left disabled: it would write thousands
# of small kernel files without changing results, only the one-time GPU-compile
# cost.


DEFAULT_METHODS = (
    "Unintegrated",
    "fastMNN",
    "Harmony",
    "Scanorama",
    "scVI",
    "Seurat",
    "Hector",
)

RANDOM_SEED = 42
N_PCS = 50
# Every method's embedding is PCA-reduced to this dimensionality before scIB
# scoring, so all methods are compared on equal footing.
SCIB_EVALUATION_PCS = 50
N_HVGS = 2000
# Scanorama's RBF-kernel weight matrix scales with batch_size; its default 5000
# reaches many GB on large atlases and triggers OOM. Chunking it this way does
# not change the resulting embedding.
SCANORAMA_BATCH_SIZE = 1000
N_DIRECT_NEIGHBORS = 30
UMAP_MIN_DIST = 0.3

HECTOR_MODEL_NAME = "human"

METHOD_DISPLAY_NAMES = {"Hector": "HECTOR"}

# Metric columns from the scib-metrics Benchmarker (Leiden bio suite). Names must
# match its clean output names exactly; the saved score table is read back by them.
SCIB_METRIC_COLUMNS = [
    "Isolated labels",
    "Leiden NMI",
    "Leiden ARI",
    "Silhouette label",
    "cLISI",
    "BRAS",
    "iLISI",
    "KBET",
    "Graph connectivity",
]

SCIB_METRIC_GROUPS = {
    "Isolated labels": "Bio conservation",
    "Leiden NMI": "Bio conservation",
    "Leiden ARI": "Bio conservation",
    "Silhouette label": "Bio conservation",
    "cLISI": "Bio conservation",
    "BRAS": "Batch correction",
    "iLISI": "Batch correction",
    "KBET": "Batch correction",
    "Graph connectivity": "Batch correction",
}


_R_PACKAGE_CACHE: dict[tuple[str, ...], tuple[bool, str]] = {}


def _check_scvi_tools_installation() -> tuple[bool, str]:
    if importlib.util.find_spec("scvi") is None:
        return False, "Missing Python module: scvi-tools (import name: scvi)"

    try:
        importlib.import_module("scvi")
        scvi_model_module = importlib.import_module("scvi.model")
    except Exception as exc:
        return False, f"Could not import scvi: {exc}"

    has_model_scvi = hasattr(scvi_model_module, "SCVI")
    if has_model_scvi:
        return True, "OK"

    return (
        False,
        "Python module 'scvi.model' does not expose SCVI. "
        "Install a usable scvi-tools build.",
    )


def _check_hector_installation() -> tuple[bool, str]:
    if importlib.util.find_spec("hector") is None:
        return False, "Missing Python module: hector"

    if importlib.util.find_spec("huggingface_hub") is None:
        return False, "Missing Python module: huggingface_hub"

    try:
        hector_module = importlib.import_module("hector")
    except Exception as exc:
        return False, f"Could not import hector: {exc}"

    if not hasattr(hector_module, "HECTOR"):
        return False, "Installed hector package does not expose hector.HECTOR"

    if not hasattr(hector_module, "get_model_info"):
        return False, "Installed hector package does not expose hector.get_model_info"

    try:
        model_info = hector_module.get_model_info(HECTOR_MODEL_NAME)
    except Exception as exc:
        return (
            False,
            f"Installed hector package does not expose the registered '{HECTOR_MODEL_NAME}' model: {exc}",
        )

    repo_id = str(model_info.get("repo_id", "unknown"))
    filename = str(model_info.get("filename", "unknown"))
    return True, f"OK (registered remote model: {HECTOR_MODEL_NAME} -> {repo_id}/{filename})"


def result_paths(project_dir: Path | str, dataset_name: str) -> dict[str, Path]:
    # Flat `result/` tree; embeddings and UMAP coordinates share `result/embeddings/`.
    project_dir = Path(project_dir)
    result_root = project_dir / "result"
    embeddings_dir = result_root / "embeddings"
    figure_dir = result_root / "figure"
    return {
        "root": result_root,
        "embeddings_dir": embeddings_dir,
        "umap_dir": embeddings_dir,
        "figure_dir": figure_dir,
        "scib_raw_path": result_root / "scib_results_raw.tsv",
        "scib_ranking_path": result_root / "scib_method_ranking.tsv",
        "supp_benchmark_png_path": figure_dir / "supplementary_figure_8_integration_benchmark.png",
        "supp_benchmark_pdf_path": figure_dir / "supplementary_figure_8_integration_benchmark.pdf",
        "figure3_png_path": figure_dir / "figure3.png",
        "figure3_pdf_path": figure_dir / "figure3.pdf",
        "hector_predictions_path": result_root / "hector_predictions.tsv.gz",
    }


def embedding_output_path(project_dir: Path | str, dataset_name: str, method_name: str) -> Path:
    return result_paths(project_dir, dataset_name)["embeddings_dir"] / f"{method_name}.npy"


def umap_coord_path(project_dir: Path | str, dataset_name: str, method_name: str) -> Path:
    return result_paths(project_dir, dataset_name)["umap_dir"] / f"{method_name}__coords.npy"


def _effective_pca_components(adata: ad.AnnData, requested_components: int = N_PCS) -> int:
    max_components = min(requested_components, adata.n_obs - 1, int(adata.var["highly_variable"].sum()))
    return max(2, max_components)


def _check_r_packages(packages: tuple[str, ...]) -> tuple[bool, str]:
    if packages in _R_PACKAGE_CACHE:
        return _R_PACKAGE_CACHE[packages]

    if shutil.which("Rscript") is None:
        result = (False, "Rscript is not available on PATH.")
        _R_PACKAGE_CACHE[packages] = result
        return result

    package_vector = ", ".join(json.dumps(package) for package in packages)
    command = (
        f"pkgs <- c({package_vector}); "
        "ok <- vapply(pkgs, requireNamespace, logical(1), quietly = TRUE); "
        "cat(paste(names(ok), ok, sep='='), sep='\\n')"
    )
    completed = subprocess.run(
        ["Rscript", "-e", command],
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        result = (False, completed.stderr.strip() or "R package check failed.")
        _R_PACKAGE_CACHE[packages] = result
        return result

    parsed = {}
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        parsed[name.strip()] = value.strip() == "TRUE"

    missing = [package for package in packages if not parsed.get(package, False)]
    result = (False, f"Missing R packages: {', '.join(missing)}") if missing else (True, "OK")
    _R_PACKAGE_CACHE[packages] = result
    return result


def collect_method_availability() -> pd.DataFrame:
    records: list[dict[str, object]] = []

    python_requirements = {
        "Harmony": ("harmonypy",),
        "Scanorama": ("scanorama",),
        "scVI": ("scvi",),
    }
    r_requirements = {
        "fastMNN": ("anndataR", "rhdf5", "SingleCellExperiment", "batchelor"),
        "Seurat": ("anndataR", "rhdf5", "Seurat", "future"),
    }

    records.append({"method": "Unintegrated", "available": True, "detail": "Built-in PCA baseline."})

    hector_available, hector_detail = _check_hector_installation()
    records.append(
        {
            "method": "Hector",
            "available": hector_available,
            "detail": hector_detail,
        }
    )

    for method_name, modules in python_requirements.items():
        if method_name == "scVI":
            scvi_ok, scvi_detail = _check_scvi_tools_installation()
            records.append({"method": method_name, "available": scvi_ok, "detail": scvi_detail})
            continue

        missing_modules = [module for module in modules if importlib.util.find_spec(module) is None]
        records.append(
            {
                "method": method_name,
                "available": len(missing_modules) == 0,
                "detail": "OK" if not missing_modules else f"Missing Python modules: {', '.join(missing_modules)}",
            }
        )

    for method_name, packages in r_requirements.items():
        available, detail = _check_r_packages(packages)
        records.append({"method": method_name, "available": available, "detail": detail})

    return pd.DataFrame(records).sort_values("method").reset_index(drop=True)


def _validate_enabled_methods(enabled_methods: Sequence[str]) -> tuple[str, ...]:
    unknown_methods = [method_name for method_name in enabled_methods if method_name not in DEFAULT_METHODS]
    if unknown_methods:
        raise ValueError(f"Unknown methods requested: {', '.join(unknown_methods)}")

    method_positions = {method_name: index for index, method_name in enumerate(enabled_methods)}
    if "Harmony" in method_positions and "Unintegrated" not in method_positions:
        raise ValueError("Harmony requires Unintegrated to be included before it.")
    if "fastMNN" in method_positions and "Unintegrated" not in method_positions:
        raise ValueError("fastMNN requires Unintegrated to be included before it.")
    if "Harmony" in method_positions and method_positions["Unintegrated"] > method_positions["Harmony"]:
        raise ValueError("Unintegrated must appear before Harmony in enabled_methods.")
    if "fastMNN" in method_positions and method_positions["Unintegrated"] > method_positions["fastMNN"]:
        raise ValueError("Unintegrated must appear before fastMNN in enabled_methods.")

    return tuple(enabled_methods)


def _validate_method_availability(enabled_methods: Sequence[str]) -> None:
    availability_table = collect_method_availability().set_index("method")
    missing_method_details = []

    for method_name in enabled_methods:
        if not bool(availability_table.loc[method_name, "available"]):
            missing_method_details.append(f"{method_name}: {availability_table.loc[method_name, 'detail']}")

    if missing_method_details:
        raise RuntimeError("The requested integration methods are not available.\n" + "\n".join(missing_method_details))


def _copy_counts_to_csr(matrix) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr(copy=True)
    return sparse.csr_matrix(np.asarray(matrix))


def _is_nonnegative_integer_matrix(matrix) -> bool:
    # Rejects normalized/log1p/scaled matrices, which carry fractional or
    # negative values; the whole matrix is checked since one bad value is enough.
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).reshape(-1)
    if values.size == 0:
        return False
    if not np.isfinite(values).all():
        return False
    if values.min() < 0:
        return False
    return bool(np.array_equal(values, np.round(values)))


def _resolve_counts_matrix(adata: ad.AnnData, dataset_name: str) -> tuple[sparse.csr_matrix, str]:
    if "counts" in adata.layers:
        return _copy_counts_to_csr(adata.layers["counts"]), 'layers["counts"]'

    if adata.raw is not None:
        if adata.var_names.equals(adata.raw.var_names):
            return _copy_counts_to_csr(adata.raw.X), "anndata.raw.X"

        raw_var_names = pd.Index(adata.raw.var_names.astype(str))
        missing_mask = ~adata.var_names.isin(raw_var_names)
        if missing_mask.any():
            missing_count = int(missing_mask.sum())
            raise ValueError(
                f"{dataset_name} is missing {missing_count} benchmark features from anndata.raw, "
                "so raw counts cannot be aligned to the processed feature axis."
            )

        aligned_raw = adata.raw[:, adata.var_names].X
        return _copy_counts_to_csr(aligned_raw), "anndata.raw.X"

    # Falls back to adata.X only when it actually holds raw counts; downstream
    # code treats X as normalized log data, so a fractional/negative X here would
    # feed normalized values into count-based methods like scVI.
    if adata.X is not None and _is_nonnegative_integer_matrix(adata.X):
        return _copy_counts_to_csr(adata.X), "adata.X"

    raise ValueError(
        f"{dataset_name} is missing the required counts layer, has no anndata.raw matrix, "
        "and adata.X does not contain non-negative integer counts."
    )


def _resolve_feature_ids(adata: ad.AnnData) -> np.ndarray:
    resolved_feature_ids = pd.Series(adata.var_names.astype(str), index=adata.var_names)

    if "feature_id" in adata.var.columns:
        existing_feature_ids = adata.var["feature_id"].astype("string").fillna("").astype(str)
        existing_mask = existing_feature_ids != ""
        resolved_feature_ids.loc[existing_mask] = existing_feature_ids.loc[existing_mask].to_numpy()

    if "ensembl_gene_id" in adata.var.columns:
        ensembl_gene_ids = adata.var["ensembl_gene_id"].astype("string").fillna("").astype(str)
        ensembl_mask = ensembl_gene_ids != ""
        resolved_feature_ids.loc[ensembl_mask] = ensembl_gene_ids.loc[ensembl_mask].to_numpy()

    return resolved_feature_ids.to_numpy()


def load_benchmark_dataset(
    project_dir: Path | str,
    dataset_name: str,
    batch_obs_column: str,
    label_obs_column: str,
    min_cells_per_batch: int = 0,
) -> ad.AnnData:
    project_dir = Path(project_dir)
    processed_path = project_dir / "input" / "processed" / f"{dataset_name}.h5ad"
    adata = ad.read_h5ad(processed_path)

    batch_obs_column = str(batch_obs_column)
    label_obs_column = str(label_obs_column)
    if batch_obs_column not in adata.obs.columns:
        raise ValueError(f"{dataset_name} is missing the required batch column: {batch_obs_column}")
    if label_obs_column not in adata.obs.columns:
        raise ValueError(f"{dataset_name} is missing the required label column: {label_obs_column}")

    batch_values = adata.obs[batch_obs_column].astype("string").fillna("").astype(str)
    label_values = adata.obs[label_obs_column].astype("string").fillna("").astype(str)

    if (batch_values == "").any():
        raise ValueError(f"{dataset_name} has empty values in the batch column: {batch_obs_column}")

    min_cells_per_batch = int(min_cells_per_batch)
    removed_batch_names: list[str] = []
    removed_cell_count = 0
    if min_cells_per_batch > 0:
        batch_sizes = batch_values.value_counts()
        keep_batch_names = batch_sizes.index[batch_sizes >= min_cells_per_batch]
        removed_batch_names = batch_sizes.index[batch_sizes < min_cells_per_batch].astype(str).tolist()

        if removed_batch_names:
            keep_mask = batch_values.isin(keep_batch_names).to_numpy()
            removed_cell_count = int((~keep_mask).sum())
            adata = adata[keep_mask].copy()
            batch_values = adata.obs[batch_obs_column].astype("string").fillna("").astype(str)
            label_values = adata.obs[label_obs_column].astype("string").fillna("").astype(str)

    if adata.n_obs == 0:
        raise ValueError(f"{dataset_name} has no cells left after batch filtering.")
    if batch_values.nunique() < 2:
        raise ValueError(f"{dataset_name} needs at least two batches after filtering for integration benchmarking.")

    adata.obs = adata.obs.copy()
    adata.var = adata.var.copy()
    counts_matrix, counts_source = _resolve_counts_matrix(adata, dataset_name)
    adata.layers["counts"] = counts_matrix
    # Drop raw: layers["counts"] now holds it, and raw is a redundant copy that
    # is never read again downstream.
    adata.raw = None
    adata.uns["benchmark_counts_source"] = counts_source
    adata.uns["benchmark_min_cells_per_batch"] = min_cells_per_batch
    adata.uns["benchmark_removed_batch_count"] = len(removed_batch_names)
    adata.uns["benchmark_removed_batches"] = removed_batch_names
    adata.uns["benchmark_removed_cells"] = removed_cell_count
    adata.obs["source_cell_id"] = adata.obs_names.astype(str)
    adata.obs_names = pd.Index([f"{dataset_name}:{cell_id}" for cell_id in adata.obs["source_cell_id"]], dtype=str)
    adata.obs["cell_id"] = adata.obs_names.astype(str)

    adata.obs["benchmark_batch"] = batch_values.to_numpy()
    adata.obs["cell_type"] = label_values.to_numpy()
    adata.obs["cell_type_coarse"] = label_values.to_numpy()
    adata.obs["keep_for_label_metrics"] = (label_values != "").to_numpy(dtype=bool)
    adata.obs["benchmark_dataset"] = dataset_name
    adata.var["feature_id"] = _resolve_feature_ids(adata)

    return adata


def prepare_shared_hvg_adata(adata: ad.AnnData, n_hvgs: int = N_HVGS) -> ad.AnnData:
    # Built from the count matrix rather than adata.copy(), which would
    # duplicate every stored matrix and risk OOM on large atlases.
    prepared = ad.AnnData(
        X=_copy_counts_to_csr(adata.layers["counts"]),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    # Shares, not copies, the count layer: neither side mutates it in place.
    prepared.layers["counts"] = adata.layers["counts"]
    prepared.uns = dict(adata.uns)

    sc.pp.normalize_total(prepared, target_sum=1e4)
    sc.pp.log1p(prepared)
    sc.pp.highly_variable_genes(
        prepared,
        n_top_genes=min(int(n_hvgs), prepared.n_vars),
        flavor="cell_ranger",
        batch_key="benchmark_batch",
        subset=False,
    )
    prepared.uns["hvg_selection_method"] = "cell_ranger_batched"
    prepared.var["feature_id"] = _resolve_feature_ids(prepared)
    return prepared


def clear_runtime_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except ImportError:
        pass

    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass


def compute_hector_embedding(
    adata: ad.AnnData,
    predictions_output_path: Path | None = None,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="benchmark_hector_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        cache_path = _write_benchmark_input_h5ad(adata, tmp_dir / "panel_input.h5ad")
        output_path = tmp_dir / "hector_embedding.npy"
        pred_path = tmp_dir / "hector_predictions.tsv.gz"

        predict_block = ""
        if predictions_output_path is not None:
            predict_block = f"""
import pandas as pd
predictions = predictor.predict(adata, label_format="name")
predictor.write_predictions(adata, predictions, rare_rollup=True, max_cells=100)
keep_cols = [c for c in adata.obs.columns if c.startswith("hector")]
pred_df = adata.obs[keep_cols].copy()
pred_df.to_csv({json.dumps(str(pred_path))}, sep="\\t")
"""

        script_text = f"""
import anndata as ad
import hector
import numpy as np

predictor = hector.HECTOR(
    {json.dumps(HECTOR_MODEL_NAME)},
    verbose=True,
    auto_download=True,
)
adata = ad.read_h5ad({json.dumps(str(cache_path))})
if "ensembl_gene_id" in adata.var.columns:
    ensembl_gene_ids = adata.var["ensembl_gene_id"].astype("string").fillna("").astype(str)
    if "feature_id" not in adata.var.columns:
        adata.var["feature_id"] = adata.var_names.astype(str)
    feature_ids = adata.var["feature_id"].astype("string").fillna("").astype(str)
    ensembl_mask = ensembl_gene_ids != ""
    feature_ids.loc[ensembl_mask] = ensembl_gene_ids.loc[ensembl_mask]
    adata.var["feature_id"] = feature_ids
elif "feature_id" not in adata.var.columns:
    adata.var["feature_id"] = adata.var_names.astype(str)
# compute_cell_embeddings runs the encoder and writes into adata.obsm["X_hector"];
# it returns None, so the embedding is read from obsm below.
predictor.compute_cell_embeddings(adata)
embedding = np.asarray(adata.obsm["X_hector"], dtype=np.float32)
np.save({json.dumps(str(output_path))}, embedding)
{predict_block}
"""

        try:
            _run_python_embedding_script(
                script_text,
                tmp_dir,
                extra_env={"TF_CPP_MIN_LOG_LEVEL": "2"},
            )
        except subprocess.CalledProcessError as gpu_error:
            try:
                _run_python_embedding_script(
                    script_text,
                    tmp_dir,
                    extra_env={"CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "2"},
                )
            except subprocess.CalledProcessError as cpu_error:
                raise RuntimeError(
                    "Hector embedding failed for both the GPU-first workflow "
                    "and the CPU fallback.\n"
                    "GPU-first workflow error:\n"
                    f"{_format_subprocess_error(gpu_error)}\n\n"
                    "CPU fallback error:\n"
                    f"{_format_subprocess_error(cpu_error)}"
                ) from cpu_error

        if predictions_output_path is not None and pred_path.exists():
            predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pred_path, predictions_output_path)

        return np.load(output_path).astype(np.float32, copy=False)


def _write_benchmark_input_h5ad(
    adata: ad.AnnData,
    output_path: Path,
) -> Path:
    export_obs = adata.obs.copy()
    export_var = adata.var.copy()
    counts_matrix = adata.layers["counts"].copy()
    export_adata = ad.AnnData(X=adata.X.copy(), obs=export_obs, var=export_var)
    export_adata.layers["counts"] = counts_matrix
    # export_adata.raw is left unpopulated: no subprocess reads it, and a
    # full-gene raw would be a third copy of the count matrix.
    export_adata.uns = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_adata.write_h5ad(output_path)
    return output_path


def _run_r_embedding_script(script_text: str, work_dir: Path, output_path: Path) -> pd.DataFrame:
    script_path = work_dir / "run_method.R"
    script_path.write_text(script_text, encoding="utf-8")
    completed = subprocess.run(
        ["Rscript", str(script_path)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return pd.read_csv(output_path, sep="\t")


def _run_python_embedding_script(
    script_text: str,
    work_dir: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> None:
    script_path = work_dir / "run_method.py"
    script_path.write_text(script_text, encoding="utf-8")
    run_env = os.environ.copy()
    if extra_env:
        run_env.update(extra_env)
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
        env=run_env,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )


def _format_subprocess_error(exc: subprocess.CalledProcessError) -> str:
    stdout = (exc.output or "").strip()
    stderr = (exc.stderr or "").strip()
    pieces = [f"exit_code={exc.returncode}"]
    if stdout:
        pieces.append(f"stdout={stdout[-2000:]}")
    if stderr:
        pieces.append(f"stderr={stderr[-4000:]}")
    return "\n".join(pieces)


def compute_unintegrated_embedding(adata: ad.AnnData, n_components: int = N_PCS) -> np.ndarray:
    hvg_mask = adata.var["highly_variable"].to_numpy()
    pca_input = adata[:, hvg_mask].copy()
    effective_components = _effective_pca_components(pca_input, requested_components=n_components)
    sc.tl.pca(
        pca_input,
        n_comps=effective_components,
        svd_solver="arpack",
        random_state=RANDOM_SEED,
        mask_var=None,
    )
    return np.asarray(pca_input.obsm["X_pca"], dtype=np.float32)


def compute_harmony_embedding(
    adata: ad.AnnData,
    batch_key: str = "benchmark_batch",
    input_key: str = "Unintegrated",
) -> np.ndarray:
    if input_key not in adata.obsm:
        raise ValueError(f"Harmony requires adata.obsm[{input_key!r}] to be present.")

    with tempfile.TemporaryDirectory(prefix="benchmark_harmony_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        input_path = tmp_dir / "embedding.npy"
        batch_path = tmp_dir / "batches.tsv"
        output_path = tmp_dir / "harmony_embedding.npy"

        np.save(input_path, np.asarray(adata.obsm[input_key], dtype=np.float32))
        pd.DataFrame(
            {
                "cell_id": adata.obs_names.astype(str),
                batch_key: adata.obs[batch_key].astype(str).to_numpy(),
            }
        ).to_csv(batch_path, sep="\t", index=False)

        def build_script(use_gpu: bool) -> str:
            return f"""
import harmonypy
import numpy as np
import pandas as pd
import torch

embedding = np.load({json.dumps(str(input_path))})
obs = pd.read_csv({json.dumps(str(batch_path))}, sep="\\t")
device = "cuda" if {use_gpu!r} and torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.set_float32_matmul_precision("high")

harmony = harmonypy.run_harmony(
    embedding,
    obs,
    {json.dumps(batch_key)},
    random_state={int(RANDOM_SEED)},
    device=device,
)
corrected = np.asarray(harmony.Z_corr, dtype=np.float32)
if corrected.shape[0] != {adata.n_obs} and corrected.shape[1] == {adata.n_obs}:
    corrected = corrected.T
if corrected.shape[0] != {adata.n_obs}:
    raise ValueError(f"Harmony returned shape {{corrected.shape}}, expected first dimension {adata.n_obs}.")
np.save({json.dumps(str(output_path))}, corrected)
"""

        try:
            _run_python_embedding_script(build_script(use_gpu=True), tmp_dir)
        except subprocess.CalledProcessError as gpu_error:
            try:
                _run_python_embedding_script(build_script(use_gpu=False), tmp_dir)
            except subprocess.CalledProcessError as cpu_error:
                raise RuntimeError(
                    "Harmony integration failed for both the GPU-first workflow "
                    "and the CPU fallback.\n"
                    "GPU-first workflow error:\n"
                    f"{_format_subprocess_error(gpu_error)}\n\n"
                    "CPU fallback error:\n"
                    f"{_format_subprocess_error(cpu_error)}"
                ) from cpu_error

        return np.load(output_path).astype(np.float32, copy=False)


def compute_scanorama_embedding(
    adata: ad.AnnData,
    batch_key: str = "benchmark_batch",
    n_components: int = N_PCS,
) -> np.ndarray:
    import scanorama

    hvg_mask = adata.var["highly_variable"].to_numpy()
    scanorama_input = adata[:, hvg_mask].copy()
    batch_series = scanorama_input.obs[batch_key].astype("category")
    batch_categories = batch_series.cat.categories.tolist()

    adata_list = [scanorama_input[batch_series == batch].copy() for batch in batch_categories]
    scanorama.integrate_scanpy(adata_list, dimred=n_components, batch_size=SCANORAMA_BATCH_SIZE)

    embeddings = np.zeros((scanorama_input.n_obs, n_components), dtype=np.float32)
    for batch_name, batch_adata in zip(batch_categories, adata_list):
        mask = (batch_series == batch_name).to_numpy()
        embeddings[mask] = np.asarray(batch_adata.obsm["X_scanorama"], dtype=np.float32)

    return embeddings


def compute_scvi_embedding(
    adata: ad.AnnData,
    batch_key: str = "benchmark_batch",
    n_components: int = N_PCS,
) -> np.ndarray:
    scvi_ok, scvi_detail = _check_scvi_tools_installation()
    if not scvi_ok:
        raise ImportError(scvi_detail)

    with tempfile.TemporaryDirectory(prefix="benchmark_scvi_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        cache_path = _write_benchmark_input_h5ad(adata, tmp_dir / "panel_input.h5ad")
        output_path = tmp_dir / "scvi_embedding.npy"

        def build_script(use_gpu: bool) -> str:
            return f"""
import anndata as ad
import inspect
import numpy as np
import random
import scvi
import torch
from scvi.model import SCVI

if not hasattr(SCVI, "setup_anndata"):
    raise ImportError(
        "Python module 'scvi.model' does not expose a usable SCVI class. "
        "Install a usable scvi-tools build."
    )

adata = ad.read_h5ad({json.dumps(str(cache_path))})
hvg_mask = adata.var["highly_variable"].to_numpy()
adata = adata[:, hvg_mask].copy()
random.seed({int(RANDOM_SEED)})
np.random.seed({int(RANDOM_SEED)})
torch.manual_seed({int(RANDOM_SEED)})
if torch.cuda.is_available():
    torch.cuda.manual_seed_all({int(RANDOM_SEED)})
if hasattr(scvi, "settings"):
    scvi.settings.seed = {int(RANDOM_SEED)}
SCVI.setup_anndata(adata, layer="counts", batch_key={json.dumps(batch_key)})
model = SCVI(adata, n_layers=2, n_latent={int(n_components)}, gene_likelihood="nb")
use_gpu = {use_gpu!r} and torch.cuda.is_available()
train_signature = inspect.signature(model.train)
train_kwargs = {{}}
if "accelerator" in train_signature.parameters and "devices" in train_signature.parameters:
    if use_gpu:
        torch.set_float32_matmul_precision("high")
        train_kwargs["accelerator"] = "gpu"
        train_kwargs["devices"] = 1
    else:
        train_kwargs["accelerator"] = "cpu"
        train_kwargs["devices"] = 1
elif "use_gpu" in train_signature.parameters:
    train_kwargs["use_gpu"] = use_gpu
model.train(**train_kwargs)
latent = np.asarray(model.get_latent_representation(), dtype=np.float32)
if use_gpu:
    torch.cuda.empty_cache()
np.save({json.dumps(str(output_path))}, latent)
"""

        try:
            _run_python_embedding_script(build_script(use_gpu=True), tmp_dir)
        except subprocess.CalledProcessError as gpu_error:
            try:
                _run_python_embedding_script(build_script(use_gpu=False), tmp_dir)
            except subprocess.CalledProcessError as cpu_error:
                raise RuntimeError(
                    "scVI integration failed for both the GPU-first workflow "
                    "and the CPU fallback.\n"
                    "GPU-first workflow error:\n"
                    f"{_format_subprocess_error(gpu_error)}\n\n"
                    "CPU fallback error:\n"
                    f"{_format_subprocess_error(cpu_error)}"
                ) from cpu_error

        return np.load(output_path).astype(np.float32, copy=False)


def compute_fastmnn_embedding(
    adata: ad.AnnData,
    batch_key: str = "benchmark_batch",
    n_components: int = N_PCS,
) -> np.ndarray:
    packages_ok, detail = _check_r_packages(("anndataR", "rhdf5", "SingleCellExperiment", "batchelor"))
    if not packages_ok:
        raise ImportError(detail)

    with tempfile.TemporaryDirectory(prefix="benchmark_fastmnn_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        cache_path = _write_benchmark_input_h5ad(adata, tmp_dir / "panel_input.h5ad")
        output_path = tmp_dir / "fastmnn_embedding.tsv"

        script_text = f"""
        suppressPackageStartupMessages(library(anndataR))
        suppressPackageStartupMessages(library(SingleCellExperiment))
        suppressPackageStartupMessages(library(batchelor))

        sce <- anndataR::read_h5ad(
          {json.dumps(str(cache_path))},
          as = "SingleCellExperiment",
          x_mapping = "logcounts",
          assays_mapping = c(counts = "counts")
        )
        if (!({json.dumps(batch_key)} %in% colnames(colData(sce)))) {{
          stop("Missing batch key in panel h5ad.")
        }}
        if (!("logcounts" %in% assayNames(sce))) {{
          stop("AnnData X matrix was not mapped to a logcounts assay.")
        }}
        if (!("highly_variable" %in% colnames(rowData(sce)))) {{
          stop("Missing highly_variable flag in rowData(sce).")
        }}

        hvgs <- rownames(sce)[which(rowData(sce)$highly_variable %in% TRUE)]
        if (length(hvgs) == 0L) {{
          stop("No highly_variable genes were available for fastMNN.")
        }}
        batches <- factor(colData(sce)[[{json.dumps(batch_key)}]])
        batch_levels <- levels(batches)
        if (length(batch_levels) < 2L) {{
          stop("fastMNN requires at least two batches.")
        }}

        sce_list <- lapply(batch_levels, function(level_name) {{
          batch_sce <- sce[hvgs, batches == level_name]
          SingleCellExperiment(
            assays = list(logcounts = assay(batch_sce, "logcounts")),
            colData = colData(batch_sce),
            rowData = rowData(batch_sce)
          )
        }})

        corrected <- do.call(
          batchelor::fastMNN,
          c(
            sce_list,
            list(d = {int(n_components)}, correct.all = FALSE)
          )
        )

        embedding <- reducedDim(corrected, "corrected")
        colnames(embedding) <- paste0("PC", seq_len(ncol(embedding)))
        out_df <- data.frame(cell_id = colnames(corrected), embedding, check.names = FALSE)
        write.table(out_df, file = {json.dumps(str(output_path))}, sep = "\\t", quote = FALSE, row.names = FALSE)
        """

        embedding_df = _run_r_embedding_script(script_text, tmp_dir, output_path)
        aligned = embedding_df.set_index("cell_id").reindex(adata.obs_names)
        if aligned.isnull().any().any():
            raise ValueError("Returned fastMNN embedding is missing cells from the input AnnData.")
        return aligned.to_numpy(dtype=np.float32)


def _write_reduced_input_table(adata: ad.AnnData, output_path: Path, input_key: str, batch_key: str) -> Path:
    embedding = np.asarray(adata.obsm[input_key], dtype=np.float32)
    pc_columns = [f"PC{i}" for i in range(1, embedding.shape[1] + 1)]
    reduced_df = pd.DataFrame(embedding, columns=pc_columns)
    reduced_df.insert(0, "batch", adata.obs[batch_key].astype(str).to_numpy())
    reduced_df.insert(0, "cell_id", adata.obs_names.astype(str))
    reduced_df.to_csv(output_path, sep="\t", index=False)
    return output_path


def compute_fastmnn_reduced_embedding(
    adata: ad.AnnData,
    batch_key: str = "benchmark_batch",
    input_key: str = "Unintegrated",
) -> np.ndarray:
    if input_key not in adata.obsm:
        adata.obsm[input_key] = compute_unintegrated_embedding(adata)

    packages_ok, detail = _check_r_packages(("batchelor",))
    if not packages_ok:
        raise ImportError(detail)

    with tempfile.TemporaryDirectory(prefix="benchmark_fastmnn_reduced_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        input_path = _write_reduced_input_table(adata, tmp_dir / "reduced_input.tsv", input_key, batch_key)
        output_path = tmp_dir / "fastmnn_embedding.tsv"

        script_text = f"""
        suppressPackageStartupMessages(library(batchelor))

        input_df <- read.delim({json.dumps(str(input_path))}, check.names = FALSE, stringsAsFactors = FALSE)
        pc_columns <- setdiff(colnames(input_df), c("cell_id", "batch"))
        batch_levels <- unique(input_df$batch)

        reduced_batches <- lapply(batch_levels, function(level_name) {{
          batch_df <- input_df[input_df$batch == level_name, , drop = FALSE]
          batch_mat <- as.matrix(batch_df[, pc_columns, drop = FALSE])
          rownames(batch_mat) <- batch_df$cell_id
          batch_mat
        }})

        corrected <- do.call(batchelor::reducedMNN, reduced_batches)
        embedding <- corrected$corrected
        colnames(embedding) <- paste0("PC", seq_len(ncol(embedding)))
        out_df <- data.frame(cell_id = rownames(embedding), embedding, check.names = FALSE)
        write.table(out_df, file = {json.dumps(str(output_path))}, sep = "\\t", quote = FALSE, row.names = FALSE)
        """

        embedding_df = _run_r_embedding_script(script_text, tmp_dir, output_path)
        aligned = embedding_df.set_index("cell_id").reindex(adata.obs_names)
        if aligned.isnull().any().any():
            raise ValueError("Returned fastMNN reduced embedding is missing cells from the input AnnData.")
        return aligned.to_numpy(dtype=np.float32)


def compute_seurat_reference_rpca_embedding(
    adata: ad.AnnData,
    batch_key: str = "benchmark_batch",
    n_components: int = N_PCS,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="benchmark_seurat_reference_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        cache_path = _write_benchmark_input_h5ad(adata, tmp_dir / "panel_input.h5ad")
        output_path = tmp_dir / "seurat_reference_embedding.tsv"

        script_text = f"""
        options(future.globals.maxSize = Inf)

        suppressPackageStartupMessages(library(anndataR))
        suppressPackageStartupMessages(library(Seurat))
        suppressPackageStartupMessages(library(future))

        future::plan(future::sequential)

        seurat_obj <- anndataR::read_h5ad({json.dumps(str(cache_path))}, as = "Seurat")
        if (!({json.dumps(batch_key)} %in% colnames(seurat_obj[[]]))) {{
          stop("Missing batch key in panel h5ad.")
        }}

        batch_column <- {json.dumps(batch_key)}
        object_list <- SplitObject(seurat_obj, split.by = batch_column)
        object_list <- lapply(object_list, function(x) {{
          x <- NormalizeData(x, verbose = FALSE)
          x <- FindVariableFeatures(x, verbose = FALSE)
          x
        }})

        features <- SelectIntegrationFeatures(object.list = object_list, nfeatures = 2000)
        object_list <- lapply(object_list, function(x) {{
          x <- ScaleData(x, features = features, verbose = FALSE)
          x <- RunPCA(x, features = features, npcs = {int(n_components)}, verbose = FALSE)
          x
        }})

        batch_sizes <- vapply(object_list, function(x) as.integer(ncol(x)), integer(1))
        reference_indices <- order(batch_sizes, decreasing = TRUE)[seq_len(min(2L, length(object_list)))]
        k_weight <- min(100L, max(10L, floor(min(batch_sizes) / 2)))

        anchors <- FindIntegrationAnchors(
          object.list = object_list,
          anchor.features = features,
          reference = reference_indices,
          reduction = "rpca",
          dims = 1:{int(n_components)},
          k.filter = NA,
          verbose = FALSE
        )

        integrated <- IntegrateData(
          anchorset = anchors,
          dims = 1:{int(n_components)},
          k.weight = k_weight,
          verbose = FALSE
        )
        DefaultAssay(integrated) <- "integrated"
        integrated <- ScaleData(integrated, verbose = FALSE)
        integrated <- RunPCA(integrated, npcs = {int(n_components)}, verbose = FALSE)

        embedding <- Embeddings(integrated[["pca"]])
        out_df <- data.frame(cell_id = rownames(embedding), embedding, check.names = FALSE)
        write.table(out_df, file = {json.dumps(str(output_path))}, sep = "\\t", quote = FALSE, row.names = FALSE)
        """

        embedding_df = _run_r_embedding_script(script_text, tmp_dir, output_path)
        aligned = embedding_df.set_index("cell_id").reindex(adata.obs_names)
        if aligned.isnull().any().any():
            raise ValueError("Returned Seurat embedding is missing cells from the input AnnData.")
        return aligned.to_numpy(dtype=np.float32)


def compute_seurat_sketch_embedding(
    adata: ad.AnnData,
    batch_key: str = "benchmark_batch",
    n_components: int = N_PCS,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="benchmark_seurat_sketch_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        cache_path = _write_benchmark_input_h5ad(adata, tmp_dir / "panel_input.h5ad")
        output_path = tmp_dir / "seurat_sketch_embedding.tsv"

        sketch_cells = max(2000, min(5000, int(np.ceil(adata.n_obs / max(1, adata.obs[batch_key].nunique())))))

        script_text = f"""
        options(future.globals.maxSize = Inf)

        suppressPackageStartupMessages(library(anndataR))
        suppressPackageStartupMessages(library(Seurat))
        suppressPackageStartupMessages(library(future))

        future::plan(future::sequential)

        object <- anndataR::read_h5ad({json.dumps(str(cache_path))}, as = "Seurat")
        if (!({json.dumps(batch_key)} %in% colnames(object[[]]))) {{
          stop("Missing batch key in panel h5ad.")
        }}
        DefaultAssay(object) <- "RNA"
        object[["RNA"]] <- split(object[["RNA"]], f = object[[{json.dumps(batch_key)}]][, 1])

        object <- NormalizeData(object, verbose = FALSE)
        object <- FindVariableFeatures(object, verbose = FALSE)
        sketch_method <- if (requireNamespace("BPCells", quietly = TRUE)) "LeverageScore" else "Uniform"
        object <- SketchData(
          object = object,
          ncells = {int(sketch_cells)},
          sketched.assay = "sketch",
          method = sketch_method,
          cast = "dgCMatrix",
          verbose = FALSE
        )

        DefaultAssay(object) <- "sketch"
        object <- FindVariableFeatures(object, verbose = FALSE)
        object <- ScaleData(object, verbose = FALSE)
        object <- RunPCA(object, npcs = {int(n_components)}, verbose = FALSE)

        data_layers <- Layers(object[["sketch"]], search = "data")
        reference_indices <- seq_len(min(2L, length(data_layers)))
        object <- IntegrateLayers(
          object = object,
          method = RPCAIntegration,
          orig.reduction = "pca",
          new.reduction = "integrated.rpca",
          dims = 1:{int(n_components)},
          k.anchor = 20,
          reference = reference_indices,
          verbose = FALSE
        )

        sketch_data_layers <- Layers(object[["sketch"]], search = "data")
        if (length(sketch_data_layers) < 2L) {{
          object[["sketch"]] <- split(object[["sketch"]], f = object[[{json.dumps(batch_key)}]][, 1])
        }}
        object <- ProjectIntegration(
          object = object,
          sketched.assay = "sketch",
          assay = "RNA",
          reduction = "integrated.rpca",
          reduction.name = "integrated.rpca.full",
          verbose = FALSE
        )

        embedding <- Embeddings(object[["integrated.rpca.full"]])
        out_df <- data.frame(cell_id = rownames(embedding), embedding, check.names = FALSE)
        write.table(out_df, file = {json.dumps(str(output_path))}, sep = "\\t", quote = FALSE, row.names = FALSE)
        """

        embedding_df = _run_r_embedding_script(script_text, tmp_dir, output_path)
        aligned = embedding_df.set_index("cell_id").reindex(adata.obs_names)
        if aligned.isnull().any().any():
            raise ValueError("Returned Seurat sketch embedding is missing cells from the input AnnData.")
        return aligned.to_numpy(dtype=np.float32)


def compute_seurat_embedding(
    adata: ad.AnnData,
    batch_key: str = "benchmark_batch",
    n_components: int = N_PCS,
) -> np.ndarray:
    packages_ok, detail = _check_r_packages(("anndataR", "rhdf5", "Seurat", "future"))
    if not packages_ok:
        raise ImportError(detail)

    try:
        return compute_seurat_reference_rpca_embedding(
            adata,
            batch_key=batch_key,
            n_components=n_components,
        )
    except subprocess.CalledProcessError as reference_error:
        try:
            return compute_seurat_sketch_embedding(
                adata,
                batch_key=batch_key,
                n_components=n_components,
            )
        except subprocess.CalledProcessError as sketch_error:
            raise RuntimeError(
                "Seurat integration failed for both the reference-based RPCA workflow "
                "and the sketch-based fallback.\n"
                "Reference workflow error:\n"
                f"{_format_subprocess_error(reference_error)}\n\n"
                "Sketch workflow error:\n"
                f"{_format_subprocess_error(sketch_error)}"
            ) from sketch_error


def compute_method_embedding(
    adata: ad.AnnData,
    method_name: str,
    output_path: Path,
    *,
    force_recompute: bool = False,
    hector_predictions_path: Path | None = None,
) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_recompute:
        if method_name == "Hector" and hector_predictions_path is not None and not hector_predictions_path.exists():
            compute_hector_embedding(adata, predictions_output_path=hector_predictions_path)
            clear_runtime_memory()
        return np.load(output_path)

    if method_name == "Unintegrated":
        embedding = compute_unintegrated_embedding(adata)
    elif method_name == "Harmony":
        embedding = compute_harmony_embedding(adata)
    elif method_name == "Scanorama":
        embedding = compute_scanorama_embedding(adata)
    elif method_name == "scVI":
        embedding = compute_scvi_embedding(adata)
    elif method_name == "fastMNN":
        if adata.n_obs > 100000:
            embedding = compute_fastmnn_reduced_embedding(adata)
        else:
            try:
                embedding = compute_fastmnn_embedding(adata)
            except subprocess.CalledProcessError as exc:
                if exc.returncode == -9:
                    embedding = compute_fastmnn_reduced_embedding(adata)
                else:
                    raise RuntimeError(
                        "fastMNN integration failed.\n"
                        f"{_format_subprocess_error(exc)}"
                    ) from exc
    elif method_name == "Seurat":
        embedding = compute_seurat_embedding(adata)
    elif method_name == "Hector":
        embedding = compute_hector_embedding(adata, predictions_output_path=hector_predictions_path)
        clear_runtime_memory()
    else:
        raise ValueError(f"Unknown method: {method_name}")

    embedding = embedding.astype(np.float32, copy=False)
    np.save(output_path, embedding)
    return embedding


def prepare_embedding_for_evaluation(method_name: str, embedding: np.ndarray) -> np.ndarray:
    """Reduce an embedding to the common SCIB_EVALUATION_PCS dimensionality before scoring.

    Applied to every method so the scIB metrics compare them in the same-sized space. Embeddings
    already at or below SCIB_EVALUATION_PCS (the PCA-based methods at 50) are returned unchanged;
    higher-dimensional ones (Hector at 768) are PCA-reduced. ``method_name`` is accepted for
    call-site compatibility but no longer branches on the method.
    """
    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.ndim != 2:
        raise ValueError("scIB evaluation requires a 2D embedding matrix.")
    if embedding.shape[1] <= SCIB_EVALUATION_PCS:
        return embedding

    pca = PCA(
        n_components=SCIB_EVALUATION_PCS,
        svd_solver="randomized",
        random_state=RANDOM_SEED,
    )
    reduced_embedding = pca.fit_transform(embedding)
    return np.asarray(reduced_embedding, dtype=np.float32)


def compute_scib_result_tables_benchmarker(
    adata: ad.AnnData,
    evaluation_embeddings: dict[str, np.ndarray],
    method_order: Sequence[str],
    *,
    batch_key: str = "benchmark_batch",
    label_key: str = "cell_type_coarse",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every method with the canonical scib-metrics Benchmarker (Leiden bio suite).

    Replaces the KMeans-based per-method scoring. All embeddings must already be reduced to a
    common dimensionality (see prepare_embedding_for_evaluation / SCIB_EVALUATION_PCS). Bio
    conservation uses Leiden clustering with resolution optimization (the original scIB method,
    robust to embedding geometry/dimension) plus isolated_labels; batch correction keeps BRAS,
    iLISI, kBET and graph connectivity. pcr_comparison is disabled because its internal cuSolver
    PCA is unstable on some GPUs and it is not part of the reported batch set.

    Returns the same ``(raw_scib, ranking_scib)`` frames the figures consume: ``raw_scib`` has an
    ``Embedding`` column plus every SCIB_METRIC_COLUMNS metric and the Bio/Batch/Total aggregates;
    ``ranking_scib`` is sorted by Total with a Rank column.
    """
    for method_name in method_order:
        adata.obsm[method_name] = np.asarray(evaluation_embeddings[method_name], dtype=np.float32)

    benchmarker = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=list(method_order),
        bio_conservation_metrics=BioConservation(
            nmi_ari_cluster_labels_leiden=True,
            nmi_ari_cluster_labels_kmeans=False,
            isolated_labels=True,
            silhouette_label=True,
            clisi_knn=True,
        ),
        batch_correction_metrics=BatchCorrection(pcr_comparison=False),
        n_jobs=-1,
        progress_bar=False,
    )
    benchmarker.benchmark()

    # get_results returns rows = embeddings plus a trailing "Metric Type" row; columns = the
    # clean-named metrics plus the Bio/Batch/Total aggregates (raw, unscaled).
    results = benchmarker.get_results(min_max_scale=False, clean_names=True)
    raw_df = results.drop(index="Metric Type").astype(float)
    raw_df.index.name = "Embedding"
    raw_df = raw_df.reset_index()
    raw_df = raw_df.sort_values("Embedding").reset_index(drop=True)

    ranking_df = (
        raw_df.loc[:, ["Embedding", "Bio conservation", "Batch correction", "Total"]]
        .sort_values("Total", ascending=False)
        .reset_index(drop=True)
    )
    ranking_df.insert(0, "Rank", np.arange(1, len(ranking_df) + 1))
    return raw_df, ranking_df


def _generate_distinct_palette(n_categories: int) -> list[str]:
    if n_categories <= 0:
        return []

    qualitative_map_names = ["tab20", "tab20b", "tab20c", "Set1", "Set2", "Set3", "Paired", "Accent", "Dark2"]
    qualitative_colors: list[str] = []
    for map_name in qualitative_map_names:
        cmap = plt.get_cmap(map_name)
        qualitative_colors.extend(mcolors.to_hex(color) for color in getattr(cmap, "colors", []))

    unique_qualitative_colors = list(dict.fromkeys(qualitative_colors))
    selected_rgbs = [mcolors.to_rgb(color) for color in unique_qualitative_colors]

    if len(selected_rgbs) < n_categories:
        golden_ratio = 0.618033988749895
        hue = 0.11
        candidate_rgbs: list[tuple[float, float, float]] = []
        for index in range(max(n_categories * 12, 512)):
            hue = (hue + golden_ratio) % 1.0
            saturation = (0.65, 0.82, 0.95)[index % 3]
            value = (0.68, 0.82, 0.94)[(index // 3) % 3]
            candidate_rgbs.append(tuple(mcolors.hsv_to_rgb((float(hue), saturation, value))))

        for candidate_rgb in candidate_rgbs:
            distance = min(
                np.sum((np.asarray(candidate_rgb) - np.asarray(selected_rgb)) ** 2) for selected_rgb in selected_rgbs
            )
            if distance >= 0.015:
                selected_rgbs.append(candidate_rgb)
            if len(selected_rgbs) >= n_categories:
                break

        if len(selected_rgbs) < n_categories:
            for candidate_rgb in candidate_rgbs:
                selected_rgbs.append(candidate_rgb)
                if len(selected_rgbs) >= n_categories:
                    break

    base_palette = [mcolors.to_hex(color) for color in selected_rgbs[:n_categories]]
    if n_categories <= 2:
        return base_palette

    jump_step = max(2, round(n_categories / 1.61803398875))
    while gcd(jump_step, n_categories) != 1:
        jump_step += 1

    reordered_indices = []
    current_index = 0
    for _ in range(n_categories):
        reordered_indices.append(current_index)
        current_index = (current_index + jump_step) % n_categories

    return [base_palette[index] for index in reordered_indices]


def _build_palette(labels: pd.Series) -> dict[str, str]:
    categories = sorted(pd.Index(labels.fillna("Missing").astype(str).unique()).tolist())
    has_missing = "Missing" in categories
    if has_missing:
        categories = [category for category in categories if category != "Missing"]

    n_categories = len(categories)
    if n_categories <= 10:
        colors = [mcolors.to_hex(color) for color in sns.color_palette("colorblind", n_colors=n_categories)]
    elif n_categories <= 20:
        colors = [mcolors.to_hex(color) for color in sns.color_palette("tab20", n_colors=n_categories)]
    else:
        colors = _generate_distinct_palette(n_categories)

    palette = {category: color for category, color in zip(categories, colors)}
    if has_missing:
        palette["Missing"] = "#B8B8B8"
    return palette


def compute_umap_coordinates(
    embedding: np.ndarray,
    n_neighbors: int = N_DIRECT_NEIGHBORS,
    metric: str = "euclidean",
) -> np.ndarray:
    if embedding.shape[0] < 3:
        raise ValueError("Need at least three cells to compute a UMAP projection.")

    temp_adata = ad.AnnData(X=np.asarray(embedding, dtype=np.float32))
    # metric controls only this visualization graph, never scIB scoring, which
    # always uses euclidean.
    sc.pp.neighbors(
        temp_adata,
        n_neighbors=min(n_neighbors, embedding.shape[0] - 1),
        use_rep="X",
        metric=metric,
    )
    sc.tl.umap(temp_adata, min_dist=UMAP_MIN_DIST, random_state=RANDOM_SEED)
    coords = np.asarray(temp_adata.obsm["X_umap"], dtype=np.float32)
    del temp_adata
    clear_runtime_memory()
    return coords


def save_method_umaps(
    project_dir: Path | str,
    dataset_name: str,
    method_name: str,
    embedding: np.ndarray,
    *,
    force_recompute: bool = False,
) -> Path:
    # Computes and caches the 2D UMAP coordinates the figures are drawn from.
    coords_path = umap_coord_path(project_dir, dataset_name, method_name)

    if coords_path.exists() and not force_recompute:
        return coords_path

    # Hector's UMAP uses cosine (see compute_umap_coordinates); others stay euclidean.
    umap_metric = "cosine" if method_name == "Hector" else "euclidean"
    coords = compute_umap_coordinates(embedding, metric=umap_metric)
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(coords_path, coords)
    return coords_path


# Aggregate columns of the score table, in the order they are drawn.
SCIB_AGGREGATE_COLUMNS = ["Batch correction", "Bio conservation", "Total"]


def _draw_scib_table(ax, metric_frame: pd.DataFrame, method_order: tuple[str, ...]) -> None:
    """Draw the scIB score table into an axes the caller supplies.

    This was `benchmarker.plot_results_table`, which builds a figure of its own and so
    cannot be placed above the maps. The column definitions below are that function's,
    copied so the table keeps the look readers of the scIB benchmarks know: one shaded
    circle per metric, one bar per aggregate score, the metrics grouped under their two
    headings. Three things differ. The axes come from the caller. The method column
    prints HECTOR the way the rest of the paper spells it. And the two aggregate
    columns are drawn wider than the metric columns, because at this figure's width
    "Batch correction" and "Bio conservation" ran into each other at equal widths.
    """
    plot_frame = metric_frame.set_index("Embedding").loc[list(method_order)]
    plot_frame = plot_frame[SCIB_METRIC_COLUMNS + SCIB_AGGREGATE_COLUMNS].astype(float)
    plot_frame.index = [METHOD_DISPLAY_NAMES.get(name, name) for name in plot_frame.index]
    plot_frame["Method"] = plot_frame.index

    column_definitions = [
        ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
    ]
    column_definitions += [
        ColumnDefinition(
            metric_name,
            title=metric_name.replace(" ", "\n", 1),
            width=1,
            textprops={"ha": "center", "bbox": {"boxstyle": "circle", "pad": 0.25}},
            cmap=normed_cmap(plot_frame[metric_name], cmap=matplotlib.cm.PRGn, num_stds=2.5),
            group=SCIB_METRIC_GROUPS[metric_name],
            formatter="{:.2f}",
        )
        for metric_name in SCIB_METRIC_COLUMNS
    ]
    column_definitions += [
        ColumnDefinition(
            score_name,
            title=score_name.replace(" ", "\n", 1),
            width=1.3,
            plot_fn=bar,
            plot_kw={
                "cmap": matplotlib.cm.YlGnBu,
                "plot_bg_bar": False,
                "annotate": True,
                "textprops": {"fontsize": plt.rcParams["font.size"]},
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group="Aggregate score",
            border="left" if index == 0 else None,
        )
        for index, score_name in enumerate(SCIB_AGGREGATE_COLUMNS)
    ]

    Table(
        plot_frame,
        ax=ax,
        column_definitions=column_definitions,
        index_col="Method",
        cell_kw={"linewidth": 0, "edgecolor": "k"},
        row_dividers=True,
        footer_divider=True,
        textprops={"fontsize": plt.rcParams["font.size"], "ha": "center"},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
        column_border_kw={"linewidth": 1, "linestyle": "-"},
    ).autoset_fontcolors(colnames=plot_frame.columns)




def _plot_umap_strip(
    figure: plt.Figure,
    strip_spec,
    key_spec,
    coords_dict: dict[str, np.ndarray],
    labels: pd.Series,
    methods: tuple[str, ...],
    *,
    palette: dict[str, str] | None = None,
    key_columns: int,
    key_fontsize: float = 9,
    dot_area: float = 0.40,
) -> None:
    """One row of maps, one per method, with the colour key on its own strip underneath.

    The key sits below rather than to the right of the last map so that the seven maps
    have the whole figure width to share. `key_columns` sets how many entries a key row
    holds; it is passed in because the two keys here are different lengths -- fourteen
    cell types against thirty-nine samples.
    """
    labels = labels.astype(object).fillna("Missing").astype(str)
    palette = dict(palette) if palette is not None else _build_palette(labels)
    colors = labels.map(palette).to_numpy()
    inner = strip_spec.subgridspec(
        2, len(methods), height_ratios=[0.18, 1], wspace=0.15, hspace=0.03,
    )

    for method_index, method_name in enumerate(methods):
        coords = coords_dict[method_name]
        heading_ax = figure.add_subplot(inner[0, method_index])
        heading_ax.set_axis_off()
        ax = figure.add_subplot(inner[1, method_index])

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=dot_area,
            c=colors,
            linewidths=0,
            alpha=0.7,
            rasterized=True,
            clip_on=False,
        )

        heading_ax.text(
            0.5,
            0.5,
            METHOD_DISPLAY_NAMES.get(method_name, method_name),
            transform=heading_ax.transAxes,
            ha="center",
            va="center",
            fontsize=plt.rcParams["font.size"],
            fontweight="bold" if method_name == "Hector" else "normal",
        )

        # A square window (each method's own) rather than the data's own limits,
        # so every map is drawn the same size and none is scaled against another.
        x_centre = 0.5 * (coords[:, 0].min() + coords[:, 0].max())
        y_centre = 0.5 * (coords[:, 1].min() + coords[:, 1].max())
        half_span = 0.52 * max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
        ax.set_xlim(x_centre - half_span, x_centre + half_span)
        ax.set_ylim(y_centre - half_span, y_centre + half_span)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_visible(False)

    key_ax = figure.add_subplot(key_spec)
    key_ax.set_axis_off()
    key_ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=palette[label], label=label,
                   markersize=5 * key_fontsize / 9)
            for label in palette
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        ncol=key_columns,
        fontsize=key_fontsize,
        handletextpad=0.3,
        columnspacing=1.0,
        labelspacing=0.35,
    )




_FIGURE3_CT_PAL = {
    "T cell":         "#4E79A7",
    "B cell":         "#59A14F",
    "NK cell":        "#E15759",
    "Monocyte":       "#F28E2B",
    "Macrophage":     "#B07AA1",
    "Dendritic cell": "#EDC948",
    "Epithelial":     "#76B7B2",
    "Fibroblast":     "#9C755F",
    "Mesothelial":    "#FF9DA7",
    "Other stromal":  "#8C8C47",
    "Endothelial":    "#6C5CE7",
    "HSC/progenitor": "#4B0082",
    "Others":         "#D8D8D8",
}

_FIGURE3_TIS_PAL = {
    "PBMC":             "#596F91",
    "Lymph Node":       "#8A9A78",
    "Primary Tumor":    "#C17C62",
    "Metastatic Tumor": "#876580",
    "Ascites":          "#B69A50",
}

_FIGURE3_TIS_TEXT = {
    "PBMC":             "white",
    "Lymph Node":       "#202020",
    "Primary Tumor":    "#202020",
    "Metastatic Tumor": "white",
    "Ascites":          "#202020",
}


# Lineage-structured cell-type palette: related types share a hue family so
# lineages read at a glance, subtypes separated by lightness.
_SUPP_LINEAGE_PAL = {
    # T lymphoid -> blue family
    "CD4+ T":              "#7BA4C9",
    "CD8+ T":              "#2E5A87",
    # B lymphoid -> green ; NK -> red
    "B":                   "#59A14F",
    "NK":                  "#E15759",
    # Myeloid -> warm orange family
    "Monocyte":            "#F28E2B",
    "DC":                  "#F6C177",
    "Macrophage":          "#B5480E",
    # Stromal / mesenchymal -> brown family
    "Fibroblast":          "#9C755F",
    "Mesothelial cells":   "#C9A896",
    "Other stromal cells": "#6E4B34",
    # Vasculature -> indigo ; tumour epithelium -> teal standout
    "Endothelial cells":   "#6C5CE7",
    "Epithelial cells":    "#76B7B2",
    # Progenitors -> purple ; cycling state -> grey
    "HSC":                 "#C5B0D5",
    "Proliferative cells": "#AAAAAA",
}


def _anchored_palette(labels: pd.Series, anchor: dict[str, str]) -> dict[str, str]:
    """Colour the categories in `labels` from `anchor`, preserving the anchor's order
    (so the legend groups by lineage / tissue). Any category absent from the anchor is
    given a fallback colour from the default distinct palette, so other datasets still
    render."""
    present = set(pd.Series(labels).astype(str).unique())
    palette = {cat: anchor[cat] for cat in anchor if cat in present}
    extras = sorted(present - set(palette))
    if extras:
        fill = _build_palette(pd.Series(extras, name="extra"))
        for cat in extras:
            palette[cat] = fill.get(cat, "#B8B8B8")
    return palette

_FIGURE3_GROUPS = [
    "T cell", "B cell", "NK cell", "Monocyte", "Macrophage", "Dendritic cell",
    "Epithelial", "Fibroblast", "Mesothelial", "Other stromal", "Endothelial",
    "HSC/progenitor", "Others",
]

# The authors' `maintypes_2` classes, collapsed to the twelve the figure scores at.
# `Proliferative cells` is handled separately; see _FIGURE3_PROLIF_LINEAGE.
_FIGURE3_AUTHOR_MAP = {
    "CD4+ T": "T cell", "CD8+ T": "T cell", "B": "B cell", "NK": "NK cell",
    "Monocyte": "Monocyte", "Macrophage": "Macrophage", "DC": "Dendritic cell",
    "Epithelial cells": "Epithelial", "Fibroblast": "Fibroblast",
    "Mesothelial cells": "Mesothelial", "Other stromal cells": "Other stromal",
    "Endothelial cells": "Endothelial", "HSC": "HSC/progenitor",
}

# `Proliferative cells` is a cell-cycle state, not a scored class. Cells carrying
# it are re-assigned to the lineage in the authors' finer `Annotation` column
# below; cells with no finer lineage leave the per-cell comparison.
_FIGURE3_PROLIF_LINEAGE = {
    "M13_Proliferative-Macro-C3":    "Macrophage",
    "M15_Proliferative-Macro-FOLR2": "Macrophage",
    "S04_CAF-MKI67":                 "Fibroblast",
    "S16_Pericyte-MKI67":            "Other stromal",
    "S12_MC-MKI67":                  "Mesothelial",
    "E06_ENDO-MKI67":                "Endothelial",
}
_FIGURE3_PROLIF_CLASS = "Proliferative cells"
_FIGURE3_ANNOTATION_COLUMN = "Annotation"

_FIGURE3_TISSUES = ["PBMC", "Lymph Node", "Primary Tumor", "Metastatic Tumor", "Ascites"]

_FIGURE3_DISP = {
    "memory T cell": "Memory T", "regulatory T cell": "Treg",
    "CD8-positive, alpha-beta T cell": "CD8+ T",
    "CD4-positive, alpha-beta memory T cell": "CD4+ memory T",
    "effector memory CD8-positive, alpha-beta T cell": "Effector memory CD8+ T",
    "mucosal invariant T cell": "MAIT", "gamma-delta T cell": "γδ T",
    "memory B cell": "Memory B", "follicular B cell": "Follicular B",
    "plasma cell": "Plasma cell", "natural killer cell": "NK cell",
    "classical monocyte": "Classical mono.", "monocyte": "Monocyte",
    "CD14-positive, CD16-negative classical monocyte": "CD14+ mono.",
    "macrophage": "Macrophage", "alternatively activated macrophage": "M2 macrophage",
    "conventional dendritic cell": "cDC", "myeloid dendritic cell": "Myeloid DC",
    "epithelial cell": "Epithelial", "secretory cell": "Secretory",
    "glandular secretory epithelial cell": "Glandular secretory epi.",
    "fallopian tube secretory epithelial cell": "Fallopian tube secretory epi.",
    "ovarian surface epithelial cell": "Ovarian surface epi.",
    "fibroblast": "Fibroblast", "stromal cell": "Stromal cell",
    "vascular associated smooth muscle cell": "Vascular SMC",
    "mesothelial cell": "Mesothelial", "endothelial cell": "Endothelial",
    "vein endothelial cell": "Vein endothelial",
    "endothelial cell of lymphatic vessel": "Lymphatic endothelial",
    "endothelial tip cell": "Tip endothelial",
    "CD4-positive, alpha-beta memory T cell, CD45RO-positive": "CD4+ memory T (CD45RO+)",
    "effector memory CD8-positive, alpha-beta T cell, terminally differentiated": "TEMRA CD8+ T",
    "effector memory CD4-positive, alpha-beta T cell": "Effector memory CD4+ T",
    "CD8-positive, alpha-beta memory T cell": "CD8+ memory T",
    "CD8-positive, alpha-beta memory T cell, CD45RO-positive": "CD8+ memory T (CD45RO+)",
    "CD4-positive, alpha-beta cytotoxic T cell": "CD4+ cytotoxic T",
    "CD8-positive, alpha-beta cytotoxic T cell": "CD8+ cytotoxic T",
    "naive B cell": "Naive B",
    "IgG plasma cell": "IgG plasma cell",
    "mature natural killer cell": "Mature NK",
    "CD16-positive, CD56-dim natural killer cell, human": "CD16+ CD56-dim NK",
    "non-classical monocyte": "Non-classical mono.",
    "intermediate monocyte": "Intermediate mono.",
    "dendritic cell": "Dendritic cell",
    "stromal cell of ovary": "Stromal (ovary)",
}

_FIGURE3_PANEL_F_MIN_TOTAL = 1000
_FIGURE3_PANEL_F_PRESENCE_MIN = 10


def _load_harmonization_table(project_dir: Path) -> pd.DataFrame:
    """Load the table mapping each HECTOR Cell Ontology term to an author class.

    Columns: `hector_label`, `author_class`, plus `n_hector_cells` and `basis`,
    which document each row and are not read by this code.
    """
    path = project_dir / "input" / "reference" / "hector_label_harmonization.tsv"
    table = pd.read_csv(path, sep="\t")
    missing = {"hector_label", "author_class"} - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
    return table


def _harmonize_hector_labels(
    hector_labels: pd.Series,
    harmonization_table: pd.DataFrame,
) -> pd.Series:
    """Map each HECTOR prediction to one of `_FIGURE3_GROUPS`.

    The trailing `(CL:...)` identifier is stripped before lookup. The table already
    targets the scored classes, so `_FIGURE3_AUTHOR_MAP` is not applied here.
    """
    label_to_class = dict(zip(harmonization_table["hector_label"],
                              harmonization_table["author_class"]))
    clean = hector_labels.astype(str).str.replace(r"\s*\(CL:.*\)$", "", regex=True)
    mapped = clean.map(label_to_class)

    # An unmapped label is a gap in the harmonization table, not a real cell type;
    # report it and fall back to Others so the run still completes.
    unmapped = clean[mapped.isna()]
    if len(unmapped):
        counts = unmapped.value_counts()
        raise_note = "\n".join(f"    {n:>7,}  {lab}" for lab, n in counts.items())
        print(
            f"\n[figure3] WARNING — {len(counts)} HECTOR label(s) covering "
            f"{int(counts.sum()):,} cells have no row in "
            f"hector_label_harmonization.tsv and are being scored as Others.\n"
            f"    Add them to the table before trusting this figure's accuracy:\n"
            f"{raise_note}\n"
        )
        mapped = mapped.fillna("Others")
    return mapped


def _author_classes(
    meta: pd.DataFrame,
    label_obs_column: str,
) -> tuple[pd.Series, int]:
    """The authors' annotation, collapsed to the scored classes.

    Returns the class per cell — NaN where the cell has no scoreable class and
    leaves the per-cell comparison — and the number of such cells. Cells annotated
    `Proliferative cells` are re-assigned via `_FIGURE3_PROLIF_LINEAGE`.
    """
    labels = meta[label_obs_column].astype(str)
    classes = labels.map(_FIGURE3_AUTHOR_MAP)

    prolif = labels == _FIGURE3_PROLIF_CLASS
    if prolif.any():
        if _FIGURE3_ANNOTATION_COLUMN not in meta.columns:
            raise ValueError(
                f"{_FIGURE3_PROLIF_CLASS!r} cells need the "
                f"{_FIGURE3_ANNOTATION_COLUMN!r} column to be resolved to a lineage"
            )
        classes = classes.mask(
            prolif,
            meta.loc[:, _FIGURE3_ANNOTATION_COLUMN].astype(str).map(_FIGURE3_PROLIF_LINEAGE),
        )
    return classes, int(classes.isna().sum())


def build_figure3(
    project_dir: Path | str,
    dataset_name: str,
    *,
    batch_obs_column: str,
    label_obs_column: str,
    tissue_obs_column: str = "Groups",
    patient_obs_column: str = "Patients",
    sample_obs_column: str = "Samples",
    min_cells_per_batch: int = 0,
) -> Path:
    project_dir = Path(project_dir)
    paths = result_paths(project_dir, dataset_name)

    # Full 3:4 page; pixels target the same 180 x 240 mm insertion as the other figures.
    canvas_width_in = 270 / 25.4
    canvas_height_in = canvas_width_in * 4 / 3
    page_scale = canvas_width_in / (180 / 25.4)
    layout_scale = canvas_width_in / (270 / 25.4)
    geometry_scale = canvas_width_in / 14.0
    export_dpi = 600 / page_scale
    text_body = canvas_width_in * 72 / 85
    text_secondary = 0.9 * text_body
    text_emphasis = 1.2 * text_body
    text_panel = 1.8 * text_body

    benchmark_adata = load_benchmark_dataset(
        project_dir, dataset_name, batch_obs_column, label_obs_column,
        min_cells_per_batch=min_cells_per_batch,
    )
    meta = benchmark_adata.obs.copy()

    predictions = pd.read_csv(paths["hector_predictions_path"], sep="\t", index_col=0)
    meta["hector_prediction"] = predictions["hector_prediction"].reindex(meta.index).values
    meta["hector_label"] = meta["hector_prediction"].str.replace(r"\s*\(CL:.*\)$", "", regex=True)

    umap_coords = np.load(umap_coord_path(project_dir, dataset_name, "Hector"))
    meta["u1"], meta["u2"] = umap_coords[:, 0], umap_coords[:, 1]

    harmonization_table = _load_harmonization_table(project_dir)
    meta["hg"] = _harmonize_hector_labels(meta["hector_prediction"], harmonization_table)
    meta["ag"], n_unscored = _author_classes(meta, label_obs_column)
    if n_unscored:
        print(
            f"[figure3] {n_unscored:,} of {len(meta):,} cells leave the per-cell "
            f"comparison: the authors annotated them {_FIGURE3_PROLIF_CLASS!r} with no "
            f"lineage recorded, and that class is a cell-cycle state rather than a "
            f"cell type, so there is no cell type answer to score them against."
        )
    # Comparison panels use only cells with an author class; UMAP panels keep
    # the unscored ones too.
    scored = meta[meta["ag"].notna()]

    TISSUES = _FIGURE3_TISSUES
    GROUPS = _FIGURE3_GROUPS
    CT_PAL = _FIGURE3_CT_PAL
    TIS_PAL = _FIGURE3_TIS_PAL
    TIS_TEXT = _FIGURE3_TIS_TEXT

    n_ontology_labels = meta["hector_prediction"].nunique()
    n_patients = meta[patient_obs_column].nunique()
    n_samples = meta[sample_obs_column].nunique()

    tcounts = meta[tissue_obs_column].value_counts().reindex(TISSUES)

    fl2g = meta[["hector_label", "hg"]].drop_duplicates().set_index("hector_label")["hg"]

    # 1.60 square points per UMAP dot at 14 inches, calibrated against published
    # maps; scaled quadratically with the working width to preserve that density.
    UMAP_DOT_AREA = 1.60 * geometry_scale**2

    # Drawn opaque: overlapping translucent clusters blend into a hue belonging
    # to no cell type.
    UMAP_OPAQUE = 1.0

    # An even draw of 120,000 of the cohort's cells, shown only to display
    # distribution. Every number (panel E's accuracy, panel F and D's counts)
    # uses the full cohort and is not reconciled with this sample.
    np.random.seed(7)
    ds = meta.sample(min(120000, len(meta)), random_state=7).copy()
    XL = (meta["u1"].min() - 0.5, meta["u1"].max() + 0.5)
    YL = (meta["u2"].min() - 0.5, meta["u2"].max() + 0.5)
    map_half_span = max(XL[1] - XL[0], YL[1] - YL[0]) / 2
    x_center = (XL[0] + XL[1]) / 2
    y_center = (YL[0] + YL[1]) / 2
    XL = (x_center - map_half_span, x_center + map_half_span)
    YL = (y_center - map_half_span, y_center + map_half_span)

    hg_plot = [g for g in GROUPS if g in ds["hg"].unique()]
    ag_plot = [g for g in GROUPS if g in ds["ag"].unique()]
    shared_plot = [g for g in GROUPS if g in ds["hg"].unique() or g in ds["ag"].unique()]

    prev_rcparams = plt.rcParams.copy()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": text_body,
        "axes.titlesize": text_body,
        "axes.labelsize": text_body,
        "xtick.labelsize": text_secondary,
        "ytick.labelsize": text_secondary,
        "legend.fontsize": text_secondary,
        "axes.linewidth": 0.4 * page_scale,
        "xtick.major.width": 0.3 * page_scale,
        "ytick.major.width": 0.3 * page_scale,
        "xtick.major.size": 2.5 * page_scale,
        "ytick.major.size": 2.5 * page_scale,
    })

    PL = dict(fontsize=text_panel, fontweight="bold", va="top", ha="left")

    def _umap_ax(ax):
        ax.set_xlim(*XL); ax.set_ylim(*YL)
        ax.set_box_aspect(1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")
        for s in ax.spines.values():
            s.set_visible(False)

    # Four bands: banner, three UMAPs, their keys, then e and f side by side.
    fig = plt.figure(figsize=(canvas_width_in, canvas_height_in), facecolor="white")
    map_width_mm = canvas_width_in * 25.4 * 0.855 / 3.24
    map_height_mm = map_width_mm
    band_mm = [34 * layout_scale, 10 * layout_scale, map_height_mm,
               8 * layout_scale, 30 * layout_scale, 6 * layout_scale,
               100 * layout_scale]
    page_height_mm = canvas_height_in * 25.4
    plot_top = 1 - 10 * layout_scale / page_height_mm
    page_grid = GridSpec(
        len(band_mm), 1, figure=fig, height_ratios=band_mm, hspace=0,
        left=0.115, right=0.97, top=plot_top,
        bottom=plot_top - sum(band_mm) / page_height_mm,
    )
    outer = {(i, 0): page_grid[2 * i, 0] for i in range(4)}

    umap_row = outer[1, 0].subgridspec(1, 3, wspace=0.12)
    key_band = outer[2, 0]
    bottom_row = outer[3, 0].subgridspec(1, 2, wspace=0.10)

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(umap_row[0, 0])
    ax_c = fig.add_subplot(umap_row[0, 1])
    ax_d = fig.add_subplot(umap_row[0, 2])
    # Panel e occupies the top 90% of its original band; f retains full height.
    e_band = bottom_row[0, 0].subgridspec(2, 1, height_ratios=[9, 1], hspace=0)
    e_grid = e_band[0, 0].subgridspec(1, 2, width_ratios=[3.4, 1.1], wspace=0.12)
    ax_e = fig.add_subplot(e_grid[0, 0])
    ax_e_r = fig.add_subplot(e_grid[0, 1])
    # Reserve the left part of f for its long cell-type names, inside its own
    # panel. They must not grow over the tissue-accuracy bars in panel e.
    f_grid = bottom_row[0, 1].subgridspec(1, 2, width_ratios=[0.43, 1.0], wspace=0)
    ax_f = fig.add_subplot(f_grid[0, 1])

    # ── Panel A — Schematic banner ──
    ax_a.set_axis_off()
    ax_a.text(-0.02, 1.03, "a", transform=ax_a.transAxes, **PL)

    ax_a.text(0.015, 0.83, "Input cohort", transform=ax_a.transAxes,
              ha="left", va="center", fontsize=text_body, fontweight="bold")
    ax_a.text(0.015, 0.47,
              f"{meta.shape[0]:,} cells\n"
              f"{n_patients} patients · {n_samples} samples\n"
              f"{len(TISSUES)} tissue sites",
              transform=ax_a.transAxes, ha="left", va="center",
              fontsize=text_secondary, color="#555", linespacing=1.25)

    source_xs = [0.19, 0.245, 0.30, 0.355, 0.41]
    source_labels = ["PB", "PLN", "Pri.OT", "Met.Ome", "Ascites"]
    source_sample_counts = (
        meta.groupby(tissue_obs_column, observed=True)[sample_obs_column]
            .nunique()
            .reindex(TISSUES)
            .fillna(0)
            .astype(int)
    )
    ax_a.text(0.325, 0.95, "Study sources: 5 tissue sites",
              transform=ax_a.transAxes, ha="center", va="top",
              fontsize=text_body, fontweight="bold")

    for source_x, tissue, source_label in zip(source_xs, TISSUES, source_labels):
        source_box = FancyBboxPatch(
            (source_x, 0.35), 0.05, 0.33,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            lw=1.0 * geometry_scale, ec=TIS_PAL[tissue], fc=TIS_PAL[tissue],
            transform=ax_a.transAxes, zorder=2, clip_on=False,
        )
        ax_a.add_patch(source_box)
        ax_a.text(
            source_x + 0.025, 0.515,
            f"{source_label}\nn={source_sample_counts.loc[tissue]}",
            transform=ax_a.transAxes, ha="center", va="center", fontsize=text_secondary,
            color=TIS_TEXT[tissue], fontweight="bold",
        )

    ax_a.text(0.325, 0.12, "model input: expression matrices only",
              transform=ax_a.transAxes, ha="center", va="bottom",
              fontsize=text_secondary, color="#666", style="italic")
    ax_a.annotate(
        "", xy=(0.50, 0.51), xytext=(0.47, 0.51),
        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.2 * geometry_scale, mutation_scale=12 * geometry_scale),
        transform=ax_a.transAxes,
        clip_on=False,
    )

    hbox = FancyBboxPatch(
        (0.51, 0.18), 0.17, 0.65,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        lw=1.1 * geometry_scale, ec="#333", fc="#FAFAFA", transform=ax_a.transAxes, zorder=2,
        clip_on=False,
    )
    ax_a.add_patch(hbox)
    ax_a.text(0.595, 0.61, "HECTOR", transform=ax_a.transAxes,
              ha="center", va="center", fontsize=text_emphasis, fontweight="bold")
    ax_a.text(0.595, 0.48, "one model · one pass",
              transform=ax_a.transAxes, ha="center", va="center",
              fontsize=text_secondary, fontweight="bold", color="#444")
    ax_a.text(0.595, 0.30, "no tissue or batch\ninformation required",
              transform=ax_a.transAxes, ha="center", va="center",
              fontsize=text_secondary, color="#666")
    ax_a.annotate(
        "", xy=(0.735, 0.51), xytext=(0.69, 0.51),
        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.2 * geometry_scale, mutation_scale=12 * geometry_scale),
        transform=ax_a.transAxes,
        clip_on=False,
    )

    prediction_box = FancyBboxPatch(
        (0.745, 0.54), 0.235, 0.29,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        lw=0.8 * geometry_scale, ec="#8A8A8A", fc="white", transform=ax_a.transAxes, clip_on=False,
    )
    embedding_box = FancyBboxPatch(
        (0.745, 0.17), 0.235, 0.29,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        lw=0.8 * geometry_scale, ec="#8A8A8A", fc="white", transform=ax_a.transAxes, clip_on=False,
    )
    ax_a.add_patch(prediction_box)
    ax_a.add_patch(embedding_box)
    ax_a.text(0.76, 0.685,
              f"Cell type predictions\n({n_ontology_labels} distinct Cell Ontology terms)",
              transform=ax_a.transAxes, ha="left", va="center", fontsize=text_secondary)
    ax_a.text(0.76, 0.315, "768-dimensional cell embeddings",
              transform=ax_a.transAxes, ha="left", va="center", fontsize=text_secondary)

    # ── Panel B — HECTOR cell types ──
    ax_b.text(-0.05, 1.03, "b", transform=ax_b.transAxes, **PL)
    ax_b.text(0.5, -0.025, "HECTOR cell type predictions", transform=ax_b.transAxes, ha="center", va="top", fontsize=text_body)
    for g in hg_plot:
        m = ds["hg"] == g
        ax_b.scatter(ds.loc[m, "u1"], ds.loc[m, "u2"],
                     s=UMAP_DOT_AREA, c=CT_PAL[g], linewidths=0, alpha=UMAP_OPAQUE,
                     rasterized=True, clip_on=False)
    _umap_ax(ax_b)

    # ── Panel C — Author annotation ──
    ax_c.text(-0.05, 1.03, "c", transform=ax_c.transAxes, **PL)
    ax_c.text(0.5, -0.025, "Published annotation (Zheng et al.)", transform=ax_c.transAxes, ha="center", va="top", fontsize=text_body)
    for g in ag_plot:
        m = ds["ag"] == g
        ax_c.scatter(ds.loc[m, "u1"], ds.loc[m, "u2"],
                     s=UMAP_DOT_AREA, c=CT_PAL[g], linewidths=0, alpha=UMAP_OPAQUE,
                     rasterized=True, clip_on=False)
    _umap_ax(ax_c)

    # Keep each key beneath the maps it describes, with two entry rows.
    key_top = key_band.get_position(fig).y1
    ct_h = [Line2D([0], [0], marker="o", color="none", markerfacecolor=CT_PAL[g],
                   markeredgewidth=0, markersize=5 * geometry_scale, label=g) for g in shared_plot]
    fig.legend(
        handles=ct_h, loc="upper center",
        bbox_to_anchor=((ax_b.get_position().x0 + ax_c.get_position().x1) / 2, key_top),
        ncol=7, frameon=False, fancybox=False,
        framealpha=0.9, facecolor="white", edgecolor="#DDD",
        fontsize=text_secondary, columnspacing=0.7, handletextpad=0.2, handlelength=0.7,
    )

    # ── Panel D — Tissue site UMAP ──
    ax_d.text(-0.05, 1.03, "d", transform=ax_d.transAxes, **PL)
    ax_d.text(0.5, -0.025, "Tissue site of origin", transform=ax_d.transAxes, ha="center", va="top", fontsize=text_body)
    for tis in TISSUES:
        m = ds[tissue_obs_column] == tis
        ax_d.scatter(ds.loc[m, "u1"], ds.loc[m, "u2"],
                     s=UMAP_DOT_AREA, c=TIS_PAL[tis], linewidths=0, alpha=UMAP_OPAQUE,
                     rasterized=True, clip_on=False)
    _umap_ax(ax_d)

    tis_h = [Line2D([0], [0], marker="o", color="none", markerfacecolor=TIS_PAL[t],
                    markeredgewidth=0, markersize=5 * geometry_scale,
                    label=f"{t}\n({tcounts[t]:,})") for t in TISSUES]
    fig.legend(
        handles=tis_h, loc="upper center",
        bbox_to_anchor=((ax_d.get_position().x0 + ax_d.get_position().x1) / 2, key_top),
        ncol=3, frameon=False, fancybox=False,
        framealpha=0.9, facecolor="white", edgecolor="#DDD",
        fontsize=text_secondary, columnspacing=0.7, handletextpad=0.2, handlelength=0.7,
    )

    # ── Panel E — Per-cell agreement: confusion matrix + per-tissue accuracy ──
    ax_e.text(-0.12, 1 + 0.10 / 0.90, "e", transform=ax_e.transAxes, **PL)

    # The matrix is rectangular: `Others` occurs only on the prediction side, so it
    # is a column with no matching row.
    e_rows = [g for g in GROUPS if g in set(scored["ag"])]
    e_cols = [g for g in GROUPS if g in set(scored["hg"]) or g in e_rows]
    e_cm = pd.crosstab(scored["ag"], scored["hg"]).reindex(
        index=e_rows, columns=e_cols, fill_value=0)
    e_row_sums = e_cm.sum(axis=1).replace(0, 1)
    e_cm_pct = e_cm.div(e_row_sums, axis=0) * 100

    e_overall = (scored["hg"] == scored["ag"]).mean() * 100 if len(scored) else 0.0
    e_tis_acc = (
        scored.assign(_match=(scored["hg"] == scored["ag"]).astype(float))
              .groupby(tissue_obs_column, observed=True)["_match"]
              .mean() * 100
    ).reindex(TISSUES).fillna(0.0)

    e_cm_pct_display = np.ma.masked_less(e_cm_pct.values, 0.5)
    e_cmap = plt.get_cmap("Blues").copy()
    e_cmap.set_bad(color="white")
    im_e = ax_e.imshow(e_cm_pct_display, aspect="auto", cmap=e_cmap,
                       vmin=0, vmax=100, interpolation="nearest", clip_on=False)
    ax_e.set_xticks(np.arange(len(e_cols)))
    ax_e.set_yticks(np.arange(len(e_rows)))
    ax_e.set_xticklabels(e_cols, rotation=45, ha="right", fontsize=text_secondary)
    ax_e.set_yticklabels(e_rows, fontsize=text_secondary)
    for i, lbl in enumerate(e_cols):
        ax_e.get_xticklabels()[i].set_color(CT_PAL.get(lbl, "#444"))
    for i, lbl in enumerate(e_rows):
        ax_e.get_yticklabels()[i].set_color(CT_PAL.get(lbl, "#444"))
    ax_e.set_xlabel("HECTOR predicted broad label", fontsize=text_body, labelpad=3)
    ax_e.set_ylabel("Author broad label", fontsize=text_body, labelpad=6)
    for s in ax_e.spines.values():
        s.set_visible(False)
    ax_e.tick_params(axis="both", length=0)
    ax_e.grid(False)

    for i in range(len(e_rows)):
        for j in range(len(e_cols)):
            v = e_cm_pct.iloc[i, j]
            if v >= 5:
                ax_e.text(j, i, f"{v:.0f}", ha="center", va="center",
                          fontsize=text_secondary,
                          color="white" if v >= 55 else "#222")
    # The diagonal only runs over the classes that appear on both axes.
    e_diag = [(e_cols.index(g), e_rows.index(g)) for g in e_rows if g in e_cols]
    if e_diag:
        ax_e.plot([e_diag[0][0] - 0.5, e_diag[-1][0] + 0.5],
                  [e_diag[0][1] - 0.5, e_diag[-1][1] + 0.5],
                  color="#888", lw=0.6 * geometry_scale, linestyle="--", zorder=4, clip_on=False)

    y_t = np.arange(len(TISSUES))
    e_bar_colors = [TIS_PAL[t] for t in TISSUES]
    ax_e_r.barh(y_t, e_tis_acc.values, color=e_bar_colors, height=0.65,
                edgecolor="black", linewidth=0.4 * geometry_scale, clip_on=False)
    for i, (t, v) in enumerate(zip(TISSUES, e_tis_acc.values)):
        tissue_text_color = TIS_TEXT[t]
        tissue_label = t.replace("Metastatic Tumor", "Metastatic\nTumor")
        tissue_label = tissue_label.replace("Primary Tumor", "Primary\nTumor")
        ax_e_r.text(2.5, i - 0.11, tissue_label, va="center", ha="left", fontsize=text_secondary,
                    color=tissue_text_color, fontweight="bold", zorder=5)
        ax_e_r.text(2.5, i + 0.15, f"{v:.1f}%", va="center", ha="left",
                    fontsize=text_secondary, color=tissue_text_color, zorder=5)
    ax_e_r.set_yticks([])
    ax_e_r.set_xlim(0, 108)
    ax_e_r.set_xticks([0, 50, 100])
    ax_e_r.invert_yaxis()
    ax_e_r.set_xlabel("Accuracy %", fontsize=text_body, labelpad=6)
    ax_e_r.spines[["top", "right", "left"]].set_visible(False)
    ax_e_r.tick_params(axis="x", labelsize=text_secondary, length=2 * page_scale)
    ax_e_r.grid(False)

    # ── Panel F — Fine ontology labels assigned across every tissue site ──
    # Filter: labels with at least _FIGURE3_PANEL_F_MIN_TOTAL cells overall and
    # at least _FIGURE3_PANEL_F_PRESENCE_MIN cells in each of the five tissues.
    # This is purely data-driven — tissue-restricted predictions naturally fail
    # the presence test, so there is no hand-picked keyword list.
    ax_f.text(-0.12, 1.10, "f", transform=ax_f.transAxes, **PL)

    f_all_cts = pd.crosstab(meta["hector_label"], meta[tissue_obs_column]).reindex(
        columns=TISSUES, fill_value=0)
    f_totals_all = f_all_cts.sum(axis=1)
    f_present_all = (f_all_cts >= _FIGURE3_PANEL_F_PRESENCE_MIN).sum(axis=1) == len(TISSUES)
    f_keep = f_totals_all.index[(f_totals_all >= _FIGURE3_PANEL_F_MIN_TOTAL) & f_present_all]
    f_cts = f_all_cts.loc[f_keep]
    f_totals = f_cts.sum(axis=1)
    f_pct = f_cts.div(f_totals, axis=0) * 100
    f_broad = pd.Series([fl2g.get(l, "Others") for l in f_cts.index], index=f_cts.index)

    f_broad_rank = {b: i for i, b in enumerate(GROUPS)}
    f_order = pd.DataFrame({
        "broad": f_broad.map(f_broad_rank).fillna(99),
        "total": f_totals,
    }).sort_values(["broad", "total"], ascending=[True, False])
    f_cts = f_cts.loc[f_order.index]
    f_pct = f_pct.loc[f_order.index]
    f_totals = f_totals.loc[f_order.index]
    f_broad = f_broad.loc[f_order.index]

    n_f = len(f_cts)
    coverage = f_totals.sum() / len(meta) * 100 if len(meta) else 0.0

    y = np.arange(n_f)
    for i in range(n_f):
        if i % 2 == 1:
            ax_f.axhspan(i - 0.5, i + 0.5, color="#F7F7F7", zorder=0, clip_on=False)

    # Thin rules make transitions between broad biological lineages easier to scan.
    for i in range(1, n_f):
        if f_broad.iloc[i] != f_broad.iloc[i - 1]:
            ax_f.axhline(i - 0.5, color="#D0D0D0", linewidth=0.7 * geometry_scale, zorder=2, clip_on=False)

    left = np.zeros(n_f)
    for tis in TISSUES:
        seg = f_pct[tis].values
        ax_f.barh(y, seg, left=left, height=0.72, color=TIS_PAL[tis],
                  linewidth=0.35 * geometry_scale, edgecolor="white", zorder=3, clip_on=False)
        for i in range(n_f):
            if seg[i] >= 10:
                ax_f.text(left[i] + seg[i] / 2, i, f"{seg[i]:.0f}",
                          ha="center", va="center", fontsize=text_secondary,
                          color=TIS_TEXT[tis])
        left += seg

    for i in range(n_f):
        ax_f.text(121, i, f"{int(f_totals.iloc[i]):,}",
                  ha="right", va="center", fontsize=text_secondary, color="#222")

    ax_f.set_xlim(-1, 122)
    if n_f > 0:
        ax_f.set_ylim(n_f - 0.5, -0.5)
    ax_f.set_yticks(y)
    ax_f.set_yticklabels([_FIGURE3_DISP.get(l, l) for l in f_cts.index], fontsize=text_secondary)
    for i, lbl in enumerate(f_cts.index):
        ax_f.get_yticklabels()[i].set_color(CT_PAL.get(f_broad.iloc[i], "#444"))

    ax_f.set_xticks([0, 25, 50, 75, 100])
    ax_f.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=text_secondary)
    ax_f.set_xlabel("Tissue distribution within label (row %);\nright column: total cells", fontsize=text_body, labelpad=4)
    ax_f.tick_params(axis="y", length=0, width=0, pad=3,
                     left=False, right=False)
    ax_f.tick_params(axis="x", length=2 * page_scale)
    ax_f.yaxis.set_ticks_position("none")
    ax_f.spines[["top", "right", "left"]].set_visible(False)
    ax_f.spines["bottom"].set_color("#CCC")
    ax_f.spines["bottom"].set_linewidth(0.5 * geometry_scale)

    # ── Save ──
    ensure_directory(paths["figure_dir"])
    # dpi also sets the resolution of the rasterized UMAP layers in the PDF.
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(paths["figure3_pdf_path"], dpi=export_dpi, bbox_inches=None,
                    facecolor="white", transparent=False)
        fig.savefig(paths["figure3_png_path"], dpi=export_dpi, bbox_inches=None,
                    facecolor="white", transparent=False, pil_kwargs={"dpi": (600, 600)})
    plt.close(fig)

    for key, val in prev_rcparams.items():
        try:
            plt.rcParams[key] = val
        except (ValueError, KeyError):
            pass

    return paths["figure3_png_path"]


def build_benchmark_figure(
    project_dir: Path | str,
    dataset_name: str,
    *,
    dataset_display_name: str | None = None,
    batch_obs_column: str,
    label_obs_column: str,
    min_cells_per_batch: int = 0,
) -> Path:
    """The integration benchmark supplement: the scores in panel a, the maps in panel b.

    These were two separate figures until 2026-08-17 -- the scIB score table on its own,
    and the seven maps on their own -- and they are one figure with two panels now. They
    describe the same comparison, and the maps were never cited while the table was, so
    apart they read as a cited result and an orphan rather than as one benchmark. Nothing
    was dropped in the merge: panel a is the same nine metrics, two aggregate scores and
    total, and panel b the same seven maps in both colourings.

    `dataset_display_name` is accepted for the caller's sake and no longer drawn: it
    resolved to the input filename, which does not belong on a figure.
    """
    project_dir = Path(project_dir)
    paths = result_paths(project_dir, dataset_name)

    metric_frame = pd.read_csv(paths["scib_raw_path"], sep="\t")
    method_order = tuple(metric_frame.sort_values("Total", ascending=False)["Embedding"].astype(str).tolist())

    benchmark_adata = load_benchmark_dataset(
        project_dir,
        dataset_name,
        batch_obs_column,
        label_obs_column,
        min_cells_per_batch=min_cells_per_batch,
    )
    cell_type_labels = benchmark_adata.obs["cell_type_coarse"].astype(str)
    sample_labels = benchmark_adata.obs["benchmark_batch"].astype(str)
    cell_type_palette = _anchored_palette(cell_type_labels, _SUPP_LINEAGE_PAL)
    sample_palette = _build_palette(sample_labels)

    coords_dict = {
        method_name: np.load(umap_coord_path(project_dir, dataset_name, method_name))
        for method_name in method_order
    }

    FIGURE_WIDTH = 270 / 25.4
    FIGURE_HEIGHT = FIGURE_WIDTH * 4 / 3
    page_scale = FIGURE_WIDTH / (180 / 25.4)
    geometry_scale = FIGURE_WIDTH / 13.75
    export_dpi = 600 / page_scale
    text_body = FIGURE_WIDTH * 72 / 85
    text_secondary = 0.9 * text_body
    text_panel = 1.8 * text_body
    TABLE_HEIGHT = 80 / 25.4
    MAP_HEIGHT = FIGURE_WIDTH * 0.95 / (len(method_order) + 0.15 * (len(method_order) - 1)) * 1.18 * 1.015
    CELL_TYPE_KEY_HEIGHT = 15 / 25.4
    SAMPLE_KEY_HEIGHT = 28 / 25.4
    LETTER_MARGIN = 8 / 25.4  # room for the first panel letter above the table
    GAP_ABOVE_MAPS = 10 / 25.4
    GAP_BETWEEN_ROWS = 10 / 25.4
    row_heights = [
        LETTER_MARGIN,
        TABLE_HEIGHT,
        GAP_ABOVE_MAPS,
        MAP_HEIGHT,
        CELL_TYPE_KEY_HEIGHT,
        GAP_BETWEEN_ROWS,
        MAP_HEIGHT,
        SAMPLE_KEY_HEIGHT,
    ]

    previous_rcparams = plt.rcParams.copy()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": text_body,
        "pdf.fonttype": 42,
    })

    try:
        figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        grid = figure.add_gridspec(
            len(row_heights),
            1,
            height_ratios=row_heights,
            hspace=0.0,
            left=0.025,
            right=0.975,
            top=1 - 10 / 360,
            bottom=1 - 10 / 360 - sum(row_heights) / FIGURE_HEIGHT,
        )

        _draw_scib_table(figure.add_subplot(grid[1]), metric_frame, method_order)

        _plot_umap_strip(
            figure,
            grid[3],
            grid[4],
            coords_dict,
            cell_type_labels,
            methods=method_order,
            palette=cell_type_palette,
            key_columns=7,
            key_fontsize=text_secondary,
            dot_area=0.40 * geometry_scale**2,
        )
        _plot_umap_strip(
            figure,
            grid[6],
            grid[7],
            coords_dict,
            sample_labels,
            methods=method_order,
            palette=sample_palette,
            key_columns=10,
            key_fontsize=text_secondary,
            dot_area=0.40 * geometry_scale**2,
        )

        # Placed against the grid, not an axes: every map's equal aspect shrinks
        # its own box to fit the data, which would move text hung off it.
        panel_letter = {"fontsize": text_panel, "fontweight": "bold", "va": "bottom", "ha": "left"}
        row_label = {"fontsize": text_body, "va": "bottom", "ha": "left"}
        figure.text(0.004, grid[1].get_position(figure).y1, "a", **panel_letter)
        figure.text(0.004, grid[3].get_position(figure).y1, "b", **panel_letter)
        figure.text(0.004, grid[6].get_position(figure).y1, "c", **panel_letter)
        figure.text(0.030, grid[3].get_position(figure).y1, "Coloured by cell type", **row_label)
        figure.text(0.030, grid[6].get_position(figure).y1, "Coloured by sample", **row_label)

        ensure_directory(paths["figure_dir"])
        with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
            figure.savefig(paths["supp_benchmark_png_path"], dpi=export_dpi,
                           bbox_inches=None, facecolor="white", pil_kwargs={"dpi": (600, 600)})
            figure.savefig(paths["supp_benchmark_pdf_path"], dpi=export_dpi,
                           bbox_inches=None, facecolor="white")
        plt.close(figure)
    finally:
        plt.rcParams.update(previous_rcparams)

    return paths["supp_benchmark_png_path"]


def run_integration_stage(
    project_dir: Path | str,
    dataset_name: str,
    *,
    dataset_display_name: str | None = None,
    batch_obs_column: str,
    label_obs_column: str,
    min_cells_per_batch: int = 0,
    enabled_methods: Sequence[str] = DEFAULT_METHODS,
    force_recompute_embeddings: bool = False,
) -> dict:
    enabled_methods = _validate_enabled_methods(enabled_methods)
    _validate_method_availability(enabled_methods)

    project_dir = Path(project_dir)
    dataset_config = resolve_dataset_config(dataset_name, display_name=dataset_display_name)
    paths = result_paths(project_dir, dataset_name)
    for key in ["root", "embeddings_dir", "figure_dir"]:
        ensure_directory(paths[key])

    benchmark_adata = load_benchmark_dataset(
        project_dir,
        dataset_name,
        batch_obs_column,
        label_obs_column,
        min_cells_per_batch=min_cells_per_batch,
    )
    shared_hvg_adata = None
    if any(method_name in {"Unintegrated", "fastMNN", "Harmony", "Scanorama", "scVI"} for method_name in enabled_methods):
        shared_hvg_adata = prepare_shared_hvg_adata(benchmark_adata)

    shared_method_set = {"Unintegrated", "fastMNN", "Harmony", "Scanorama", "scVI"}
    enabled_sequence = list(enabled_methods)

    for method_index, method_name in enumerate(enabled_sequence):
        if method_name in shared_method_set:
            method_adata = shared_hvg_adata
            if method_name == "Unintegrated" and "Unintegrated" not in method_adata.obsm:
                method_adata.obsm["Unintegrated"] = compute_method_embedding(
                    method_adata,
                    "Unintegrated",
                    embedding_output_path(project_dir, dataset_name, "Unintegrated"),
                    force_recompute=force_recompute_embeddings,
                )
            if method_name in {"Harmony", "fastMNN"} and "Unintegrated" not in method_adata.obsm:
                method_adata.obsm["Unintegrated"] = np.load(
                    embedding_output_path(project_dir, dataset_name, "Unintegrated")
                )

            embedding = compute_method_embedding(
                method_adata,
                method_name,
                embedding_output_path(project_dir, dataset_name, method_name),
                force_recompute=force_recompute_embeddings,
            )
            if method_name == "Unintegrated":
                method_adata.obsm["Unintegrated"] = embedding

        else:
            # Seurat and Hector run heavy subprocesses that can use tens of GB;
            # release shared_hvg_adata once no later method still needs it.
            if shared_hvg_adata is not None and not any(
                later_method in shared_method_set
                for later_method in enabled_sequence[method_index:]
            ):
                shared_hvg_adata = None
                method_adata = None
                clear_runtime_memory()

            hector_pred_path = paths["hector_predictions_path"] if method_name == "Hector" else None
            embedding = compute_method_embedding(
                benchmark_adata,
                method_name,
                embedding_output_path(project_dir, dataset_name, method_name),
                force_recompute=force_recompute_embeddings,
                hector_predictions_path=hector_pred_path,
            )

        clear_runtime_memory()

    integration_report = {
        "dataset_name": dataset_name,
        "display_name": str(dataset_config["display_name"]),
        "processed_file": str((project_dir / "input" / "processed" / f"{dataset_name}.h5ad").relative_to(project_dir)),
        "n_obs": int(benchmark_adata.n_obs),
        "n_vars": int(benchmark_adata.n_vars),
        "batch_obs_column": str(batch_obs_column),
        "label_obs_column": str(label_obs_column),
        "min_cells_per_batch": int(min_cells_per_batch),
        "removed_batch_count": int(benchmark_adata.uns.get("benchmark_removed_batch_count", 0)),
        "removed_cell_count": int(benchmark_adata.uns.get("benchmark_removed_cells", 0)),
        "counts_layer_present": "counts" in benchmark_adata.layers,
        "counts_source": str(benchmark_adata.uns.get("benchmark_counts_source", "unknown")),
        "enabled_methods": list(enabled_methods),
    }
    return integration_report


def run_evaluation_stage(
    project_dir: Path | str,
    dataset_name: str,
    *,
    dataset_display_name: str | None = None,
    batch_obs_column: str,
    label_obs_column: str,
    tissue_obs_column: str = "Groups",
    patient_obs_column: str = "Patients",
    sample_obs_column: str = "Samples",
    min_cells_per_batch: int = 0,
    enabled_methods: Sequence[str] = DEFAULT_METHODS,
    force_recompute_evaluation: bool = False,
) -> dict:
    enabled_methods = _validate_enabled_methods(enabled_methods)

    project_dir = Path(project_dir)
    dataset_config = resolve_dataset_config(dataset_name, display_name=dataset_display_name)
    paths = result_paths(project_dir, dataset_name)
    for key in ["root", "embeddings_dir", "figure_dir"]:
        ensure_directory(paths[key])

    benchmark_adata = load_benchmark_dataset(
        project_dir,
        dataset_name,
        batch_obs_column,
        label_obs_column,
        min_cells_per_batch=min_cells_per_batch,
    )
    evaluation_embeddings: dict[str, np.ndarray] = {}

    for method_name in enabled_methods:
        embedding_path = embedding_output_path(project_dir, dataset_name, method_name)
        if not embedding_path.exists():
            raise FileNotFoundError(
                f"Missing saved embedding for {dataset_name} / {method_name}: {embedding_path}\n"
                "Run the integration stage first."
            )

        embedding = np.load(embedding_path)
        evaluation_embedding = prepare_embedding_for_evaluation(method_name, embedding)
        evaluation_embeddings[method_name] = evaluation_embedding

        save_method_umaps(
            project_dir,
            dataset_name,
            method_name,
            evaluation_embedding,
            force_recompute=force_recompute_evaluation,
        )
        clear_runtime_memory()

    # Score all methods at once with the scib-metrics Benchmarker (Leiden bio suite).
    raw_scib, ranking_scib = compute_scib_result_tables_benchmarker(
        benchmark_adata,
        evaluation_embeddings,
        tuple(enabled_methods),
        batch_key="benchmark_batch",
        label_key="cell_type_coarse",
    )
    raw_scib.to_csv(paths["scib_raw_path"], sep="\t", index=False)
    ranking_scib.to_csv(paths["scib_ranking_path"], sep="\t", index=False)

    supp_figure_path = build_benchmark_figure(
        project_dir,
        dataset_name,
        dataset_display_name=dataset_display_name,
        batch_obs_column=batch_obs_column,
        label_obs_column=label_obs_column,
        min_cells_per_batch=min_cells_per_batch,
    )

    figure3_path = None
    if paths["hector_predictions_path"].exists():
        figure3_path = build_figure3(
            project_dir,
            dataset_name,
            batch_obs_column=batch_obs_column,
            label_obs_column=label_obs_column,
            tissue_obs_column=tissue_obs_column,
            patient_obs_column=patient_obs_column,
            sample_obs_column=sample_obs_column,
            min_cells_per_batch=min_cells_per_batch,
        )

    evaluation_report = {
        "dataset_name": dataset_name,
        "display_name": str(dataset_config["display_name"]),
        "batch_obs_column": str(batch_obs_column),
        "label_obs_column": str(label_obs_column),
        "min_cells_per_batch": int(min_cells_per_batch),
        "removed_batch_count": int(benchmark_adata.uns.get("benchmark_removed_batch_count", 0)),
        "removed_cell_count": int(benchmark_adata.uns.get("benchmark_removed_cells", 0)),
        "enabled_methods": list(enabled_methods),
        "scib_raw_path": str(paths["scib_raw_path"].relative_to(project_dir)),
        "scib_ranking_path": str(paths["scib_ranking_path"].relative_to(project_dir)),
        "supp_benchmark_png_path": str(supp_figure_path.relative_to(project_dir)),
        "supp_benchmark_pdf_path": str(paths["supp_benchmark_pdf_path"].relative_to(project_dir)),
    }
    if figure3_path is not None:
        evaluation_report["figure3_png_path"] = str(figure3_path.relative_to(project_dir))
        evaluation_report["figure3_pdf_path"] = str(paths["figure3_pdf_path"].relative_to(project_dir))

    return evaluation_report
