"""Infrastructure utilities for the benchmark workflow.

These helpers handle path resolution, device detection, conda environment
discovery, and embedding job execution — supporting infrastructure, not
evaluation logic.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np


_SCCELLO_PROTOCOL_NAMES = frozenset(
    {
        "10x 5' v1",
        "10x 5' v2",
        "10x 3' v1",
        "10x 3' v2",
        "10x 3' v3",
        "10x 3' transcription profiling",
        "10x 5' transcription profiling",
    }
)


def _cell_embeddings_prefix(model_cfg):
    """Return the configured output prefix for saved cell embeddings."""
    if "cell_embeddings_prefix" in model_cfg:
        return model_cfg["cell_embeddings_prefix"]
    raise KeyError("Model config must define 'cell_embeddings_prefix'.")


def find_benchmark_workspace_dir(
    start=None,
    max_up=6,
    workspace_dir_name="model_eval",
):
    """Walk up from *start* to locate the benchmark workspace directory."""
    start = Path(start or os.getcwd()).resolve()
    required = [
        "evaluate.py",
        "embed_hector.py",
        "embed_scgpt.py",
        "embed_sccello.py",
        "embed_geneformer.py",
        "embed_scimilarity.py",
    ]
    for candidate in [start] + list(start.parents)[:max_up]:
        if all((candidate / name).exists() for name in required):
            return candidate
        nested = candidate / workspace_dir_name
        if all((nested / name).exists() for name in required):
            return nested
    return start / workspace_dir_name


def find_existing_path(name, start=None, max_up=6):
    """Search upward from *start* for a file or directory named *name*."""
    start = Path(start or os.getcwd()).resolve()
    for candidate in [start] + list(start.parents)[:max_up]:
        path = candidate / name
        if path.exists():
            return path
    matches = list(start.parent.rglob(name))
    if matches:
        return matches[0]
    return start / name


def expected_cell_embeddings_path(model_cfg, output_dir, dataset_role):
    """Return the default output path for a dataset-role-specific embeddings file."""
    prefix = _cell_embeddings_prefix(model_cfg)
    return Path(output_dir) / f"{prefix}_{dataset_role}_embeddings.pkl"


def _is_sccello_model(model_cfg):
    """Return True when the model applies deterministic row filtering."""
    return (
        model_cfg.get("env") == "sccello"
        or _cell_embeddings_prefix(model_cfg) == "sccello"
    )


@lru_cache(maxsize=16)
def _read_dataset_obs_metadata_cached(dataset_path: str):
    """Load row-level dataset metadata once per dataset path."""
    try:
        import evaluate as evaluate_module
    except ImportError:
        from model_eval import evaluate as evaluate_module

    return evaluate_module.read_h5ad_observation_metadata(dataset_path)


def _resolve_saved_embedding_counts(data):
    """Extract the saved request/input/output counts from an embeddings payload."""
    sampled_obs_indices = data.get("sampled_obs_indices")

    saved_requested_n_cells = data.get("requested_n_cells")
    if saved_requested_n_cells is not None:
        saved_requested_n_cells = int(saved_requested_n_cells)
    elif sampled_obs_indices is not None:
        saved_requested_n_cells = len(sampled_obs_indices)

    saved_input_n_cells = data.get("input_n_cells")
    if saved_input_n_cells is not None:
        saved_input_n_cells = int(saved_input_n_cells)

    saved_output_n_cells = data.get("output_n_cells")
    if saved_output_n_cells is not None:
        saved_output_n_cells = int(saved_output_n_cells)
    else:
        saved_output_n_cells = int(data.get("n_cells", len(data.get("cell_types", []))))

    return {
        "requested_n_cells": saved_requested_n_cells,
        "input_n_cells": saved_input_n_cells,
        "output_n_cells": saved_output_n_cells,
    }


def _subsample_obs_metadata(obs_rows, requested_n_cells, seed=42):
    """Mirror the embedding scripts' deterministic subsampling step."""
    requested_n_cells = int(requested_n_cells)
    if requested_n_cells >= len(obs_rows):
        return obs_rows

    rng = np.random.RandomState(seed)
    sampled_indices = rng.choice(len(obs_rows), requested_n_cells, replace=False)
    return obs_rows.iloc[sampled_indices]


def _expected_output_n_cells(model_cfg, dataset_path, target_n_cells):
    """Return the expected saved output-row count for the current request."""
    target_n_cells = int(target_n_cells)
    if not _is_sccello_model(model_cfg):
        return target_n_cells

    obs_rows = _read_dataset_obs_metadata_cached(str(Path(dataset_path).resolve()))
    sampled_rows = _subsample_obs_metadata(obs_rows, target_n_cells)

    mask = np.ones(len(sampled_rows), dtype=bool)
    if "organism" in sampled_rows.columns:
        mask &= sampled_rows["organism"].astype(str).to_numpy() == "Homo sapiens"
    if "assay" in sampled_rows.columns:
        mask &= sampled_rows["assay"].isin(_SCCELLO_PROTOCOL_NAMES).to_numpy()
    if "is_primary_data" in sampled_rows.columns:
        mask &= sampled_rows["is_primary_data"].astype(bool).to_numpy()
    return int(mask.sum())


def _current_input_n_cells(dataset_path):
    """Return the current input row count for a dataset."""
    obs_rows = _read_dataset_obs_metadata_cached(str(Path(dataset_path).resolve()))
    return int(len(obs_rows))


def cell_embeddings_candidates(model_cfg, output_dir, dataset_path, dataset_role):
    """Return reuse candidates ordered from most to least specific."""
    prefix = _cell_embeddings_prefix(model_cfg)
    output_dir = Path(output_dir)
    dataset_name = Path(dataset_path).name
    dataset_role = str(dataset_role)

    candidates = [expected_cell_embeddings_path(model_cfg, output_dir, dataset_role)]

    if dataset_role == "bridge" and dataset_name == "TS_ALL_cells_10k.h5ad":
        candidates.append(output_dir / f"{prefix}_clustering_embeddings.pkl")
    elif dataset_role == "query" and dataset_name == "Minority_ontology.h5ad":
        candidates.append(output_dir / f"{prefix}_clustering_embeddings.pkl")
    elif dataset_role == "clustering":
        if dataset_name == "TS_ALL_cells_10k.h5ad":
            candidates.append(output_dir / f"{prefix}_bridge_embeddings.pkl")
        if dataset_name == "Minority_ontology.h5ad":
            candidates.append(output_dir / f"{prefix}_query_embeddings.pkl")

    candidates.append(output_dir / f"{prefix}_embeddings.pkl")

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate)
    return deduped


def load_cell_embeddings_result(path):
    """Load a saved cell-embeddings file and return the decoded payload."""
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def inspect_saved_cell_embeddings(candidate_path, model_cfg, dataset_path, target_n_cells):
    """Inspect one saved embeddings file against the requested dataset/count."""
    candidate_path = Path(candidate_path)
    dataset_name = Path(dataset_path).name
    result = {
        "path": candidate_path,
        "status": "missing",
        "saved_dataset_name": None,
        "saved_requested_n_cells": None,
        "saved_input_n_cells": None,
        "saved_output_n_cells": None,
        "message": f"{candidate_path.name}: file not found",
    }

    if not candidate_path.exists():
        return result

    data = load_cell_embeddings_result(candidate_path)
    if data is None:
        result["status"] = "unreadable"
        result["message"] = f"{candidate_path.name}: could not load pickle payload"
        return result

    saved_dataset_name = Path(str(data.get("dataset_name", data.get("source")))).name
    result["saved_dataset_name"] = saved_dataset_name

    saved_counts = _resolve_saved_embedding_counts(data)
    result["saved_requested_n_cells"] = saved_counts["requested_n_cells"]
    result["saved_input_n_cells"] = saved_counts["input_n_cells"]
    result["saved_output_n_cells"] = saved_counts["output_n_cells"]

    if saved_dataset_name != dataset_name:
        result["status"] = "dataset_mismatch"
        result["message"] = (
            f"{candidate_path.name}: dataset mismatch "
            f"({saved_dataset_name} != {dataset_name})"
        )
        return result

    current_input_n_cells = None
    if saved_counts["input_n_cells"] is not None or _is_sccello_model(model_cfg):
        current_input_n_cells = _current_input_n_cells(dataset_path)

    if (
        saved_counts["requested_n_cells"] is not None
        and int(saved_counts["requested_n_cells"]) != int(target_n_cells)
    ):
        result["status"] = "requested_n_cells_mismatch"
        result["message"] = (
            f"{candidate_path.name}: requested_n_cells mismatch "
            f"({saved_counts['requested_n_cells']} != {target_n_cells})"
        )
        return result

    if (
        saved_counts["input_n_cells"] is not None
        and current_input_n_cells is not None
        and int(saved_counts["input_n_cells"]) != int(current_input_n_cells)
    ):
        result["status"] = "input_n_cells_mismatch"
        result["message"] = (
            f"{candidate_path.name}: input_n_cells mismatch "
            f"({saved_counts['input_n_cells']} != {current_input_n_cells})"
        )
        return result

    expected_output_n_cells = _expected_output_n_cells(
        model_cfg,
        dataset_path,
        target_n_cells,
    )
    if int(saved_counts["output_n_cells"]) != int(expected_output_n_cells):
        result["status"] = "output_n_cells_mismatch"
        result["message"] = (
            f"{candidate_path.name}: output_n_cells mismatch "
            f"({saved_counts['output_n_cells']} != {expected_output_n_cells})"
        )
        return result

    result["status"] = "compatible"
    result["message"] = (
        f"{candidate_path.name}: compatible cache for {dataset_name} "
        f"with requested_n_cells={target_n_cells} and output_n_cells={expected_output_n_cells}"
    )
    return result


def summarize_embedding_diagnostics(diagnostics):
    """Compress candidate diagnostics into one short printable message."""
    if not diagnostics:
        return "No candidate embedding paths were checked."

    counts = {}
    examples = []
    for item in diagnostics:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
        if item["status"] != "compatible" and len(examples) < 3:
            examples.append(item["message"])

    count_summary = ", ".join(
        f"{status}={count}"
        for status, count in sorted(counts.items())
    )
    if not examples:
        return f"Candidate summary: {count_summary}."
    return f"Candidate summary: {count_summary}. Examples: {'; '.join(examples)}"


def inspect_reusable_cell_embeddings(
    model_cfg,
    output_dir,
    dataset_path,
    target_n_cells,
    dataset_role="clustering",
    candidate_paths=None,
):
    """Inspect candidate cache paths and return the first compatible match."""
    dataset_name = Path(dataset_path).name
    if candidate_paths is None:
        candidate_paths = cell_embeddings_candidates(
            model_cfg,
            output_dir,
            dataset_path,
            dataset_role,
        )

    diagnostics = [
        inspect_saved_cell_embeddings(candidate, model_cfg, dataset_path, target_n_cells)
        for candidate in candidate_paths
    ]
    selected_path = next(
        (item["path"] for item in diagnostics if item["status"] == "compatible"),
        None,
    )
    return {
        "selected_path": selected_path,
        "diagnostics": diagnostics,
        "summary": summarize_embedding_diagnostics(diagnostics),
    }


def _build_embedding_status_record(
    model_name,
    cfg,
    dataset_role,
    output_path,
    target_n_cells,
    status,
    *,
    elapsed_seconds=0.0,
    returncode=None,
    stderr_tail="",
    stdout_tail="",
    log_path=None,
    cache_action=None,
    cache_reason="",
):
    """Build a printable embedding status row."""
    return {
        "model": model_name,
        "dataset_role": dataset_role,
        "status": status,
        "env": cfg["env"],
        "output_path": str(output_path),
        "n_cells_requested": None if target_n_cells is None else int(target_n_cells),
        "elapsed_seconds": float(elapsed_seconds),
        "returncode": returncode,
        "stderr_tail": stderr_tail,
        "stdout_tail": stdout_tail,
        "log_path": None if log_path is None else str(log_path),
        "cache_action": cache_action,
        "cache_reason": cache_reason,
    }


def _clear_stale_embedding_outputs(output_path):
    """Remove an exact-output embedding artifact before forced regeneration.

    This helper is intentionally narrow: it only touches the exact output path
    and its paired `.log` file. We use it when reuse is disabled so a failed
    regeneration cannot silently fall through to stale embeddings from an
    earlier run.
    """
    output_path = Path(output_path)
    removed_any = False

    for candidate in (output_path, output_path.with_suffix(".log")):
        if candidate.exists():
            candidate.unlink()
            removed_any = True

    return removed_any


def ensure_cell_embeddings(
    model_name,
    cfg,
    dataset_role,
    dataset_path,
    output_dir,
    requested_n_cells,
    target_n_cells,
    device,
    *,
    available_envs,
    run_embedding,
    reuse_existing_embeddings,
):
    """Resolve one model's cache-vs-generation decision.

    Behavior is intentionally simple and per-model:
    - `reuse_existing_embeddings=True`: inspect that model's cache candidates in
      order and reuse the first compatible file; otherwise regenerate.
    - `reuse_existing_embeddings=False`: ignore caches and regenerate the exact
      output path.
    """
    output_path = expected_cell_embeddings_path(cfg, output_dir, dataset_role)
    reusable_path = None
    reuse_report = None

    if reuse_existing_embeddings:
        reuse_report = inspect_reusable_cell_embeddings(
            cfg,
            output_dir,
            dataset_path,
            target_n_cells,
            dataset_role=dataset_role,
        )
        reusable_path = reuse_report["selected_path"]

    if reusable_path is not None:
        record = _build_embedding_status_record(
            model_name,
            cfg,
            dataset_role,
            reusable_path,
            target_n_cells,
            "reused_existing",
            returncode=0,
            stdout_tail=f"Reused existing cell embeddings: {Path(reusable_path).name}",
            cache_action="reused_existing",
            cache_reason=next(
                (
                    item["message"]
                    for item in reuse_report["diagnostics"]
                    if item["status"] == "compatible"
                    and item["path"] == Path(reusable_path)
                ),
                f"{Path(reusable_path).name}: compatible cache selected",
            ),
        )
        return Path(reusable_path), record

    if not run_embedding:
        exact_output_report = inspect_reusable_cell_embeddings(
            cfg,
            output_dir,
            dataset_path,
            target_n_cells,
            dataset_role=dataset_role,
            candidate_paths=[output_path],
        )
        exact_output_path = exact_output_report["selected_path"]
        if exact_output_path is not None:
            record = _build_embedding_status_record(
                model_name,
                cfg,
                dataset_role,
                exact_output_path,
                target_n_cells,
                "cached_mode",
                stdout_tail=(
                    "Embedding execution skipped because RUN_EMBEDDING=False. "
                    "Using the exact expected cached output path."
                ),
                cache_action="used_exact_cached_output",
                cache_reason=exact_output_report["summary"],
            )
            return exact_output_path, record

        reuse_summary = (
            reuse_report["summary"]
            if reuse_existing_embeddings and reuse_report is not None
            else exact_output_report["summary"]
        )
        record = _build_embedding_status_record(
            model_name,
            cfg,
            dataset_role,
            output_path,
            target_n_cells,
            "missing_cached_output" if not reuse_existing_embeddings else "missing_reusable_embeddings",
            stderr_tail=(
                "Embedding execution skipped because RUN_EMBEDDING=False, but "
                "no compatible cached embeddings were found. "
                f"{reuse_summary}"
            ),
            cache_action="missing_cache",
            cache_reason=reuse_summary,
        )
        return None, record

    if cfg["env"] not in available_envs:
        record = _build_embedding_status_record(
            model_name,
            cfg,
            dataset_role,
            output_path,
            target_n_cells,
            "missing_env",
            stderr_tail=f"Conda env '{cfg['env']}' is not installed locally.",
            cache_action="missing_env",
            cache_reason=f"Conda env '{cfg['env']}' is not installed locally.",
        )
        return None, record

    stale_output_removed = False
    if not reuse_existing_embeddings:
        stale_output_removed = _clear_stale_embedding_outputs(output_path)

    record = run_embedding_job(
        model_name,
        cfg,
        dataset_role,
        dataset_path,
        output_path,
        requested_n_cells,
        device,
        reported_n_cells=target_n_cells,
    )

    if record["status"] == "completed":
        record["status"] = "generated"

    if reuse_existing_embeddings:
        record["cache_action"] = "generated_after_cache_check"
        record["cache_reason"] = (
            reuse_report["summary"]
            if reuse_report is not None
            else "No cache inspection summary was captured."
        )
    else:
        record["cache_action"] = "generated_reuse_disabled"
        record["cache_reason"] = "Cache reuse disabled; generated exact output path."

    if stale_output_removed:
        note = (
            "Removed stale exact-output embeddings before regeneration because "
            "reuse_existing_embeddings=False."
        )
        if record["stdout_tail"]:
            record["stdout_tail"] = f"{record['stdout_tail']}\n{note}"
        else:
            record["stdout_tail"] = note

    if reuse_existing_embeddings and reuse_report is not None and reusable_path is None:
        note = (
            "No compatible cached embeddings were found for this model, so "
            f"generation ran instead. {reuse_report['summary']}"
        )
        if record["stdout_tail"]:
            record["stdout_tail"] = f"{record['stdout_tail']}\n{note}"
        else:
            record["stdout_tail"] = note

    if record["status"] not in {"generated"}:
        return None, record

    if not output_path.exists():
        record["status"] = "failed_missing_output"
        message = f"Embedding job reported success but did not create {output_path}."
        if record["stderr_tail"]:
            record["stderr_tail"] = f"{record['stderr_tail']}\n{message}"
        else:
            record["stderr_tail"] = message
        return None, record

    return output_path, record


def resolve_default_device():
    """Detect whether a CUDA GPU is available; fall back to 'cpu'."""
    if not shutil.which("nvidia-smi"):
        return "cpu"
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "cpu"
    return "cuda" if result.returncode == 0 else "cpu"


def get_available_conda_envs():
    """Return (set_of_env_names, error_string_or_None)."""
    conda_path = shutil.which("conda")
    if not conda_path:
        return set(), "conda executable not found on PATH"
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        envs = {Path(p).name for p in payload.get("envs", [])}
        return envs, None
    except Exception as exc:
        return set(), str(exc)


def _summarize_tail(lines, max_chars=1000):
    """Join recent subprocess lines into a compact tail string."""
    if not lines:
        return ""
    return "\n".join(lines)[-max_chars:]


def _emit_subprocess_line(job_label, stream_name, line, log_handle, write_lock):
    """Write a subprocess log line to both stdout and the log file."""
    stream_suffix = "" if stream_name == "stdout" else f" {stream_name}"
    message = f"[{job_label}{stream_suffix}] {line}" if line else f"[{job_label}{stream_suffix}]"
    with write_lock:
        print(message, flush=True)
        if log_handle is not None:
            log_handle.write(message + "\n")
            log_handle.flush()


def _stream_subprocess_pipe(pipe, *, job_label, stream_name, log_handle, write_lock, tail_lines):
    """Continuously forward a child-process pipe until EOF."""
    try:
        for raw_line in iter(pipe.readline, ""):
            if raw_line == "":
                break
            line = raw_line.rstrip("\n")
            tail_lines.append(line)
            _emit_subprocess_line(job_label, stream_name, line, log_handle, write_lock)
    finally:
        pipe.close()


def _run_logged_subprocess(cmd, *, env, log_path, job_label):
    """Run a subprocess while streaming live logs and recording output tails."""
    stdout_tail_lines = deque(maxlen=80)
    stderr_tail_lines = deque(maxlen=80)
    write_lock = threading.Lock()

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_handle:
        _emit_subprocess_line(job_label, "stdout", f"Starting command: {' '.join(cmd)}", log_handle, write_lock)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )

        threads = []
        for stream_name, pipe, tail_lines in (
            ("stdout", process.stdout, stdout_tail_lines),
            ("stderr", process.stderr, stderr_tail_lines),
        ):
            if pipe is None:
                continue
            thread = threading.Thread(
                target=_stream_subprocess_pipe,
                kwargs={
                    "pipe": pipe,
                    "job_label": job_label,
                    "stream_name": stream_name,
                    "log_handle": log_handle,
                    "write_lock": write_lock,
                    "tail_lines": tail_lines,
                },
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        returncode = process.wait()
        for thread in threads:
            thread.join()

        elapsed = time.time() - t0
        status = "completed" if returncode == 0 else "failed"
        _emit_subprocess_line(
            job_label,
            "stdout",
            f"Finished with status={status}, returncode={returncode}, elapsed_seconds={elapsed:.2f}",
            log_handle,
            write_lock,
        )

    return {
        "returncode": int(returncode),
        "elapsed_seconds": round(elapsed, 2),
        "stdout_tail": _summarize_tail(stdout_tail_lines),
        "stderr_tail": _summarize_tail(stderr_tail_lines),
        "log_path": str(log_path),
    }


def run_embedding_job(
    model_name,
    cfg,
    dataset_role,
    dataset_path,
    output_path,
    n_cells,
    device,
    reported_n_cells=None,
):
    """Execute a single embedding subprocess and return a status dict."""
    clean_env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    cache_dirs = {
        "NUMBA_CACHE_DIR": Path("/tmp") / f"numba_cache_{cfg['env']}",
        "MPLCONFIGDIR": Path("/tmp") / f"mpl_{cfg['env']}",
        "XDG_CACHE_HOME": Path("/tmp") / f"xdg_cache_{cfg['env']}",
    }
    for env_key, cache_dir in cache_dirs.items():
        cache_dir.mkdir(parents=True, exist_ok=True)
        clean_env.setdefault(env_key, str(cache_dir))
    clean_env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [
        "conda", "run", "--no-capture-output", "-n", cfg["env"], "python", "-u", str(cfg["script"]),
        "--input", str(dataset_path),
        "--output", str(output_path),
        "--device", device,
    ]
    if n_cells is not None:
        cmd.extend(["--n-cells", str(int(n_cells))])
    for arg in cfg.get("extra_args", []):
        cmd.append(str(arg))
    log_path = Path(output_path).with_suffix(".log")
    job_label = f"embedding {model_name}/{dataset_role}"
    result = _run_logged_subprocess(cmd, env=clean_env, log_path=log_path, job_label=job_label)
    n_cells_requested = reported_n_cells if reported_n_cells is not None else n_cells
    return {
        "model": model_name,
        "dataset_role": dataset_role,
        "status": "completed" if result["returncode"] == 0 else "failed",
        "env": cfg["env"],
        "output_path": str(output_path),
        "n_cells_requested": None if n_cells_requested is None else int(n_cells_requested),
        "elapsed_seconds": result["elapsed_seconds"],
        "returncode": result["returncode"],
        "stderr_tail": result["stderr_tail"],
        "stdout_tail": result["stdout_tail"],
        "log_path": result["log_path"],
        "cache_action": None,
        "cache_reason": "",
    }


def run_native_comparison_job(step_name, script_path, env_name, output_dir,
                               extra_env=None):
    """Execute one native-decoder-comparison script and return a status dict.

    Mirrors `run_embedding_job`: a thin subprocess wrapper around
    `_run_logged_subprocess`. Unlike the embedding scripts, these take no CLI
    arguments — every path is resolved relative to the script's own location
    (`Path(__file__).resolve().parents[2]`) — so the command is just
    `conda run -n <env> python -u <script>`.
    """
    clean_env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    cache_dirs = {
        "NUMBA_CACHE_DIR": Path("/tmp") / f"numba_cache_{env_name}",
        "MPLCONFIGDIR": Path("/tmp") / f"mpl_{env_name}",
        "XDG_CACHE_HOME": Path("/tmp") / f"xdg_cache_{env_name}",
    }
    for env_key, cache_dir in cache_dirs.items():
        cache_dir.mkdir(parents=True, exist_ok=True)
        clean_env.setdefault(env_key, str(cache_dir))
    clean_env.setdefault("PYTHONUNBUFFERED", "1")
    if extra_env:
        clean_env.update(extra_env)

    cmd = ["conda", "run", "--no-capture-output", "-n", env_name,
           "python", "-u", str(script_path)]
    log_path = Path(output_dir) / f"native_comparison_{step_name}.log"
    job_label = f"native-comparison {step_name}"
    result = _run_logged_subprocess(cmd, env=clean_env, log_path=log_path, job_label=job_label)
    return {
        "step": step_name,
        "env": env_name,
        "status": "completed" if result["returncode"] == 0 else "failed",
        "elapsed_seconds": result["elapsed_seconds"],
        "returncode": result["returncode"],
        "log_path": result["log_path"],
    }


def run_clustering_evaluation_job(
    model_name,
    eval_env,
    script_path,
    embeddings_path,
    dataset_path,
    output_path,
    label_col,
    batch_col,
    n_neighbors,
    distance_metric,
    resolutions,
    random_state,
):
    """Execute a clustering-evaluation subprocess and return a status dict."""
    clean_env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    cache_dirs = {
        "NUMBA_CACHE_DIR": Path("/tmp") / f"numba_cache_{eval_env}",
        "MPLCONFIGDIR": Path("/tmp") / f"mpl_{eval_env}",
        "XDG_CACHE_HOME": Path("/tmp") / f"xdg_cache_{eval_env}",
    }
    for env_key, cache_dir in cache_dirs.items():
        cache_dir.mkdir(parents=True, exist_ok=True)
        clean_env.setdefault(env_key, str(cache_dir))

    resolution_arg = ",".join(str(float(x)) for x in resolutions)
    cmd = [
        "conda",
        "run",
        "-n",
        eval_env,
        "python",
        str(script_path),
        "--embeddings",
        str(embeddings_path),
        "--dataset",
        str(dataset_path),
        "--label-col",
        str(label_col),
        "--batch-col",
        str(batch_col),
        "--output",
        str(output_path),
        "--n-neighbors",
        str(int(n_neighbors)),
        "--distance-metric",
        str(distance_metric),
        "--resolutions",
        resolution_arg,
        "--random-state",
        str(int(random_state)),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=clean_env)
    elapsed = time.time() - t0
    return {
        "model": model_name,
        "status": "completed" if result.returncode == 0 else "failed",
        "eval_env": eval_env,
        "output_path": str(output_path),
        "label_col": str(label_col),
        "batch_col": str(batch_col),
        "n_neighbors": int(n_neighbors),
        "distance_metric": str(distance_metric),
        "elapsed_seconds": round(elapsed, 2),
        "returncode": int(result.returncode),
        "stderr_tail": (result.stderr or "")[-1000:],
        "stdout_tail": (result.stdout or "")[-1000:],
    }
