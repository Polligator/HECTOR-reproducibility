#!/bin/bash
# Flexible environment setup for the multi-model benchmark.
#
# Four benchmark envs now install from explicit recipes plus benchmark-specific
# compatibility fixes:
#   - sccello
#   - scgpt
#   - geneformer
#   - scimilarity
#
# Hector remains supported through its legacy requirements file because it is
# outside the current rebuild campaign.
#
# The curated local repos and model assets already present under model_eval/
# remain the source of truth. This script never re-clones or refreshes them.
#
# Usage:
#   bash scripts/model_eval/setup_envs.sh [sccello|scgpt|geneformer|scimilarity|hector|all]
#
# Useful overrides:
#   RECREATE_EXISTING=0      # reuse an existing env instead of removing it first
#   VALIDATE_ENVS=0          # skip bridge/query embedding validation
#   TORCH_INDEX_URL=...      # choose a different official PyTorch wheel index
#   TORCH_INSTALL_SPECS=...  # e.g. "torch==2.10.0"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PATCHES_DIR="$SCRIPT_DIR/patches"
MANIFEST_DIR="$SCRIPT_DIR/install_manifests"
HECTOR_REQUIREMENTS_FILE="$SCRIPT_DIR/hector_requirements.txt"
VALIDATOR_PATH="$SCRIPT_DIR/validate_env_embeddings.py"

PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_INSTALL_SPECS="${TORCH_INSTALL_SPECS:-torch}"

RECREATE_EXISTING="${RECREATE_EXISTING:-1}"
VALIDATE_ENVS="${VALIDATE_ENVS:-1}"
VALIDATION_DEVICE="${VALIDATION_DEVICE:-auto}"
VALIDATION_ROOT="${VALIDATION_ROOT:-/tmp/zero_shot_benchmark_env_validation}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/tmp/zero_shot_benchmark_env_snapshots}"

BRIDGE_N_CELLS="${BRIDGE_N_CELLS:-200}"
QUERY_N_CELLS="${QUERY_N_CELLS:-100}"
GENEFORMER_VERSION="${GENEFORMER_VERSION:-104M}"

# Small compatibility constraints based on the currently working benchmark
# stacks. These are intentionally much narrower than the old full freezes.
SCCELLO_PIP_SPECS="${SCCELLO_PIP_SPECS:-numpy<2 anndata<0.11 scanpy<1.11 transformers<5 psutil wandb ipdb torchmetrics networkx}"
SCGPT_PIP_SPECS="${SCGPT_PIP_SPECS:-numpy<2 anndata<0.11 scanpy<1.11 transformers<5 orbax<0.1.8 ipython scgpt==0.2.4}"
GENEFORMER_PIP_SPECS="${GENEFORMER_PIP_SPECS:-anndata<0.13 scanpy<1.12 datasets<5 transformers<5}"
SCIMILARITY_PIP_SPECS="${SCIMILARITY_PIP_SPECS:-numpy<2 anndata<0.12 scanpy<1.12 scimilarity==0.4.1}"
EVALUATION_PIP_SPECS="${EVALUATION_PIP_SPECS:-scib-metrics leidenalg}"
JAX_CUDA13_PIP_SPECS="${JAX_CUDA13_PIP_SPECS:-jax[cuda13]==0.9.2}"
JAX_CUDA12_PIP_SPECS="${JAX_CUDA12_PIP_SPECS:-jax==0.4.30 jaxlib==0.4.30 jax-cuda12-plugin[with-cuda]==0.4.30}"

SCCELLO_REPO="$SCRIPT_DIR/../../models/scCello/scCello-repo"
GENEFORMER_REPO="$SCRIPT_DIR/../../models/geneformer"
SCGPT_CHECKPOINT="$SCRIPT_DIR/../../models/scGPT/checkpoints/scGPT_CP"
SCCELLO_CHECKPOINT="$SCRIPT_DIR/../../models/scCello/checkpoints/scCello-zeroshot"
SCIMILARITY_MODEL_DIR="$SCRIPT_DIR/../../models/scimilarity/model_v1.1"

CURRENT_SNAPSHOT_DIR=""
LAST_VALIDATION_SUMMARY=""
LAST_VALIDATION_OUTPUT_ROOT=""

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_command() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || die "Required command not found on PATH: $cmd"
}

require_path() {
    local path="$1"
    [ -e "$path" ] || die "Required path not found: $path"
}

conda_env_exists() {
    local env_name="$1"
    conda env list --json | python -c '
import json
import sys
from pathlib import Path

payload = json.load(sys.stdin)
target = sys.argv[1]
env_names = {Path(path).name for path in payload.get("envs", [])}
raise SystemExit(0 if target in env_names else 1)
' "$env_name"
}

bootstrap_python_tooling() {
    local env_name="$1"
    echo "Bootstrapping pip/setuptools/wheel in $env_name..."
    conda run -n "$env_name" python -m pip install --upgrade pip setuptools wheel
}

pip_install_specs() {
    local env_name="$1"
    local spec_string="$2"
    local -a specs=()

    read -r -a specs <<< "$spec_string"
    if [ ${#specs[@]} -eq 0 ]; then
        return 0
    fi

    echo "Installing packages in $env_name: ${specs[*]}"
    conda run -n "$env_name" python -m pip install \
        --index-url "$PYPI_INDEX_URL" \
        --extra-index-url "$TORCH_INDEX_URL" \
        "${specs[@]}"
}

install_torch_runtime() {
    local env_name="$1"
    local -a specs=()

    read -r -a specs <<< "$TORCH_INSTALL_SPECS"
    if [ ${#specs[@]} -eq 0 ]; then
        die "TORCH_INSTALL_SPECS must not be empty"
    fi

    echo "Installing torch runtime in $env_name: ${specs[*]}"
    conda run -n "$env_name" python -m pip install \
        --index-url "$TORCH_INDEX_URL" \
        --extra-index-url "$PYPI_INDEX_URL" \
        "${specs[@]}"
}

install_editable_package() {
    local env_name="$1"
    local package_path="$2"

    require_path "$package_path"
    echo "Installing editable package in $env_name: $package_path"
    conda run -n "$env_name" python -m pip install \
        --index-url "$PYPI_INDEX_URL" \
        --extra-index-url "$TORCH_INDEX_URL" \
        -e "$package_path"
}

install_torchtext_shim() {
    local env_name="$1"
    local shim_path="$PATCHES_DIR/torchtext_compat"

    require_path "$shim_path/setup.py"
    echo "Installing local torchtext compatibility shim in $env_name..."
    conda run -n "$env_name" python -m pip install "$shim_path"
}

install_requirements_file() {
    local env_name="$1"
    local requirements_file="$2"

    require_path "$requirements_file"
    echo "Installing requirements file in $env_name: $(basename "$requirements_file")"
    conda run -n "$env_name" python -m pip install \
        --index-url "$PYPI_INDEX_URL" \
        --extra-index-url "$TORCH_INDEX_URL" \
        --requirement "$requirements_file"
}

install_evaluation_stack() {
    local env_name="$1"

    echo "Installing shared clustering-evaluation stack in $env_name..."
    pip_install_specs "$env_name" "$EVALUATION_PIP_SPECS"
    install_pinned_jax_stack "$env_name"
}

install_pinned_jax_stack() {
    local env_name="$1"
    local jax_spec_string=""
    local -a specs=()

    case "$env_name" in
        hector)
            jax_spec_string="$JAX_CUDA13_PIP_SPECS"
            ;;
        scgpt|sccello)
            jax_spec_string="$JAX_CUDA12_PIP_SPECS"
            ;;
        *)
            return 0
            ;;
    esac

    read -r -a specs <<< "$jax_spec_string"
    if [ ${#specs[@]} -eq 0 ]; then
        return 0
    fi

    echo "Installing CUDA-enabled JAX stack in $env_name: ${specs[*]}"
    conda run -n "$env_name" env -u LD_LIBRARY_PATH python -m pip install \
        --upgrade \
        --index-url "$PYPI_INDEX_URL" \
        --extra-index-url "$TORCH_INDEX_URL" \
        "${specs[@]}"
}

verify_imports() {
    local env_name="$1"
    shift

    echo "Verifying imports in $env_name: $*"
    conda run --no-capture-output -n "$env_name" python - "$@" <<'PY'
import importlib
import sys

modules = sys.argv[1:]
for module_name in modules:
    importlib.import_module(module_name)

print(f"✓ import check passed: {' '.join(modules)}")
PY
}

verify_evaluation_stack() {
    local env_name="$1"

    verify_imports "$env_name" scib_metrics jax jaxlib igraph leidenalg
}

verify_hector_env() {
    local env_name="$1"
    local import_list="$2"
    local numba_cache="/tmp/numba_cache_${env_name}"
    local mpl_cache="/tmp/mpl_${env_name}"
    local xdg_cache="/tmp/xdg_cache_${env_name}"

    NUMBA_CACHE_DIR="$numba_cache" \
    MPLCONFIGDIR="$mpl_cache" \
    XDG_CACHE_HOME="$xdg_cache" \
    conda run --no-capture-output -n "$env_name" python - <<PY
import importlib
import os
from pathlib import Path

import tensorflow as tf

modules = "${import_list}".split()
for cache_var in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
    Path(os.environ[cache_var]).mkdir(parents=True, exist_ok=True)

for module_name in modules:
    importlib.import_module(module_name)

print(
    f"✓ Hector env verified: tensorflow={tf.__version__}, "
    f"gpus={len(tf.config.list_physical_devices('GPU'))}, imports={modules}"
)
PY
}

snapshot_existing_env() {
    local env_name="$1"
    local snapshot_dir="$SNAPSHOT_ROOT/${env_name}_$(date -u +"%Y%m%dT%H%M%SZ")"

    mkdir -p "$snapshot_dir"
    echo "Snapshotting existing $env_name env to $snapshot_dir..." >&2

    conda run -n "$env_name" python -m pip freeze > "$snapshot_dir/pip_freeze.txt" || true
    conda run -n "$env_name" python -m pip list --format=columns > "$snapshot_dir/pip_list.txt" || true

    case "$env_name" in
        sccello)
            conda run -n "$env_name" python -m pip show sccello torch transformers scanpy datasets > "$snapshot_dir/pip_show.txt" || true
            ;;
        scgpt)
            conda run -n "$env_name" python -m pip show scgpt torch transformers scanpy torchtext > "$snapshot_dir/pip_show.txt" || true
            ;;
        geneformer)
            conda run -n "$env_name" python -m pip show geneformer torch transformers scanpy datasets > "$snapshot_dir/pip_show.txt" || true
            ;;
        scimilarity)
            conda run -n "$env_name" python -m pip show scimilarity torch scanpy anndata > "$snapshot_dir/pip_show.txt" || true
            ;;
        hector)
            conda run -n "$env_name" python -m pip show tensorflow anndata numpy pandas > "$snapshot_dir/pip_show.txt" || true
            ;;
    esac

    echo "$snapshot_dir"
}

prepare_env() {
    local env_name="$1"
    local python_version="$2"

    CURRENT_SNAPSHOT_DIR=""

    if conda_env_exists "$env_name"; then
        CURRENT_SNAPSHOT_DIR="$(snapshot_existing_env "$env_name")"
        if [ "$RECREATE_EXISTING" = "1" ]; then
            echo "Removing existing $env_name env..."
            conda env remove -n "$env_name" -y
        else
            echo "Reusing existing $env_name env because RECREATE_EXISTING=0."
        fi
    fi

    if ! conda_env_exists "$env_name"; then
        echo "Creating $env_name env with Python $python_version..."
        conda create -n "$env_name" "python=$python_version" -y
    fi

    bootstrap_python_tooling "$env_name"
}

validate_model_env() {
    local model_key="$1"

    LAST_VALIDATION_SUMMARY=""
    LAST_VALIDATION_OUTPUT_ROOT=""

    if [ "$VALIDATE_ENVS" != "1" ]; then
        echo "Skipping validation for $model_key because VALIDATE_ENVS=0."
        return 0
    fi

    require_path "$VALIDATOR_PATH"
    mkdir -p "$VALIDATION_ROOT"

    LAST_VALIDATION_OUTPUT_ROOT="$VALIDATION_ROOT/${model_key}_$(date -u +"%Y%m%dT%H%M%SZ")"
    LAST_VALIDATION_SUMMARY="$VALIDATION_ROOT/${model_key}_latest.json"

    echo "Running forced fresh embedding validation for $model_key..."
    python "$VALIDATOR_PATH" \
        --model "$model_key" \
        --output-root "$LAST_VALIDATION_OUTPUT_ROOT" \
        --summary-out "$LAST_VALIDATION_SUMMARY" \
        --device "$VALIDATION_DEVICE" \
        --bridge-n-cells "$BRIDGE_N_CELLS" \
        --query-n-cells "$QUERY_N_CELLS" \
        --geneformer-version "$GENEFORMER_VERSION"

    echo "Validation summary saved to $LAST_VALIDATION_SUMMARY"
}

write_install_manifest() {
    local env_name="$1"
    local manifest_path="$MANIFEST_DIR/${env_name}_install_manifest.md"
    local install_source=""
    local fix_notes=""
    local key_packages=()
    local package=""
    local version=""

    mkdir -p "$MANIFEST_DIR"

    case "$env_name" in
        sccello)
            install_source="Installed torch from the configured PyTorch wheel index, then installed the curated local scCello repo from $SCCELLO_REPO with dependency resolution enabled."
            fix_notes="No repo refresh step is used. The benchmark keeps the curated local scCello source and local scCello checkpoint. Explicit compatibility installs cover repo-local import-time dependencies that are not declared in setup.py: psutil, wandb, ipdb, torchmetrics, and networkx. The shared clustering evaluator stack is installed separately so the notebook benchmark can run with scib-metrics. The evaluation JAX runtime is then pinned to the verified Python 3.9-compatible CUDA 12 stack."
            key_packages=(sccello torch transformers scanpy datasets rdflib safetensors anndata numpy psutil wandb ipdb torchmetrics networkx scib-metrics jax jaxlib jax-cuda12-plugin igraph leidenalg)
            ;;
        scgpt)
            install_source="Installed torch from the configured PyTorch wheel index, installed the local torchtext compatibility shim, then installed scgpt from PyPI with small compatibility constraints."
            fix_notes="The local torchtext shim remains explicit because upstream scgpt still expects torchtext on newer torch stacks. The shared clustering evaluator stack is installed separately so the notebook benchmark can run with scib-metrics. The evaluation JAX runtime is then pinned to the verified Python 3.9-compatible CUDA 12 stack."
            key_packages=(scgpt torch transformers scanpy anndata numpy torchtext scib-metrics jax jaxlib jax-cuda12-plugin igraph leidenalg)
            ;;
        geneformer)
            install_source="Installed torch from the configured PyTorch wheel index, then installed the curated local Geneformer repo from $GENEFORMER_REPO with dependency resolution enabled."
            fix_notes="Applied the CUDA compatibility patch against the repo-local Geneformer source that embed_geneformer.py imports at runtime. The shared clustering evaluator stack is installed separately so the notebook benchmark can run with scib-metrics."
            key_packages=(geneformer torch transformers scanpy anndata datasets loompy peft bitsandbytes scib-metrics jax jaxlib igraph leidenalg)
            ;;
        scimilarity)
            install_source="Installed torch from the configured PyTorch wheel index, then installed scimilarity from PyPI and used the curated local model assets under $SCIMILARITY_MODEL_DIR."
            fix_notes="No repo refresh step is used. Validation relies on the local scimilarity model_v1.1 assets already present in this workspace. The shared clustering evaluator stack is installed separately so the notebook benchmark can run with scib-metrics."
            key_packages=(scimilarity torch scanpy anndata numpy hnswlib scib-metrics jax jaxlib igraph leidenalg)
            ;;
        *)
            install_source="Legacy manifest path."
            fix_notes="No custom notes recorded."
            key_packages=(torch)
            ;;
    esac

    {
        echo "# ${env_name} install manifest"
        echo
        echo "- Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        echo "- Python: $(conda run -n "$env_name" python --version 2>&1 | tail -n 1)"
        echo "- Torch wheel index: \`$TORCH_INDEX_URL\`"
        echo "- Validation device request: \`$VALIDATION_DEVICE\`"
        if [ -n "$CURRENT_SNAPSHOT_DIR" ]; then
            echo "- Pre-rebuild snapshot: \`$CURRENT_SNAPSHOT_DIR\`"
        fi
        if [ -n "$LAST_VALIDATION_SUMMARY" ]; then
            echo "- Validation summary: \`$LAST_VALIDATION_SUMMARY\`"
            echo "- Validation outputs: \`$LAST_VALIDATION_OUTPUT_ROOT\`"
        fi
        echo
        echo "## Install source"
        echo
        echo "$install_source"
        echo
        echo "## Custom fixes"
        echo
        echo "$fix_notes"
        echo
        echo "## Key package versions"
        echo

        for package in "${key_packages[@]}"; do
            version="$(conda run --no-capture-output -n "$env_name" python -c '
from importlib import metadata
import sys

name = sys.argv[1]
try:
    print(metadata.version(name))
except metadata.PackageNotFoundError:
    print("not-installed")
' "$package" 2>/dev/null || true)"
            version="$(printf '%s\n' "$version" | sed '/^[[:space:]]*$/d' | tail -n 1)"
            if [ -z "$version" ]; then
                version="unknown"
            fi
            echo "- \`$package\`: \`$version\`"
        done
    } > "$manifest_path"

    echo "Wrote install manifest: $manifest_path"
}

setup_sccello() {
    echo "=== Setting up sccello environment ==="
    require_path "$SCCELLO_REPO/setup.py"
    require_path "$SCCELLO_CHECKPOINT"

    prepare_env sccello 3.9
    install_torch_runtime sccello
    pip_install_specs sccello "$SCCELLO_PIP_SPECS"
    install_editable_package sccello "$SCCELLO_REPO"
    install_evaluation_stack sccello

    verify_imports sccello scanpy sccello torch transformers datasets rdflib safetensors
    verify_evaluation_stack sccello
    validate_model_env sccello
    write_install_manifest sccello
    echo "sccello environment ready."
}

setup_scgpt() {
    echo "=== Setting up scgpt environment ==="
    require_path "$SCGPT_CHECKPOINT"

    prepare_env scgpt 3.9
    install_torch_runtime scgpt
    install_torchtext_shim scgpt
    pip_install_specs scgpt "$SCGPT_PIP_SPECS"
    install_evaluation_stack scgpt

    verify_imports scgpt scanpy scgpt transformers torch torchtext
    verify_evaluation_stack scgpt
    validate_model_env scgpt
    write_install_manifest scgpt
    echo "scgpt environment ready."
}

setup_geneformer() {
    echo "=== Setting up geneformer environment ==="
    require_path "$GENEFORMER_REPO/setup.py"
    require_path "$GENEFORMER_REPO/geneformer"

    prepare_env geneformer 3.11
    install_torch_runtime geneformer
    pip_install_specs geneformer "$GENEFORMER_PIP_SPECS"
    install_editable_package geneformer "$GENEFORMER_REPO"
    install_evaluation_stack geneformer

    echo "Applying Geneformer CUDA compatibility patch to repo-local source..."
    python "$PATCHES_DIR/patch_geneformer_cuda.py" --target "$GENEFORMER_REPO"

    verify_imports geneformer geneformer scanpy transformers datasets loompy peft bitsandbytes
    verify_evaluation_stack geneformer
    validate_model_env geneformer
    write_install_manifest geneformer
    echo "geneformer environment ready."
}

setup_scimilarity() {
    echo "=== Setting up scimilarity environment ==="
    require_path "$SCIMILARITY_MODEL_DIR"

    prepare_env scimilarity 3.10
    install_torch_runtime scimilarity
    pip_install_specs scimilarity "$SCIMILARITY_PIP_SPECS"
    install_evaluation_stack scimilarity

    verify_imports scimilarity scanpy scimilarity torch anndata hnswlib
    verify_evaluation_stack scimilarity
    validate_model_env scimilarity
    write_install_manifest scimilarity
    echo "scimilarity environment ready."
}

setup_hector() {
    echo "=== Setting up hector environment ==="
    prepare_env hector 3.11
    install_requirements_file hector "$HECTOR_REQUIREMENTS_FILE"
    install_pinned_jax_stack hector
    verify_hector_env hector "tensorflow anndata numpy pandas h5py scipy sklearn skimage tqdm networkx IPython charset_normalizer scib_metrics jax jaxlib igraph leidenalg"
    echo "hector environment ready."
}

main() {
    require_command conda
    require_command python
    require_path "$PROJECT_ROOT/input/TS_downsampled_cells.h5ad"
    require_path "$PROJECT_ROOT/input/Zero_shot_query_set.h5ad"
    require_path "$PROJECT_ROOT/input/cl.obo"

    case "${1:-all}" in
        sccello)     setup_sccello ;;
        scgpt)       setup_scgpt ;;
        geneformer)  setup_geneformer ;;
        scimilarity) setup_scimilarity ;;
        hector)      setup_hector ;;
        all)
            setup_sccello
            setup_scgpt
            setup_geneformer
            setup_scimilarity
            setup_hector
            ;;
        *)
            echo "Usage: $0 [sccello|scgpt|geneformer|scimilarity|hector|all]"
            exit 1
            ;;
    esac

    echo
    echo "Done. The four rebuilt benchmark envs now install from explicit recipes,"
    echo "local curated repos/assets, automated compatibility fixes, and forced fresh"
    echo "embedding validation. Legacy full requirements snapshots are no longer used"
    echo "for sccello/scgpt/geneformer/scimilarity."
    echo
    echo "Useful overrides:"
    echo "  RECREATE_EXISTING=0      # reuse an existing env instead of removing it"
    echo "  VALIDATE_ENVS=0          # skip forced fresh embedding validation"
    echo "  TORCH_INDEX_URL=...      # choose a different official torch wheel index"
    echo "  TORCH_INSTALL_SPECS=...  # choose a different torch package spec"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
