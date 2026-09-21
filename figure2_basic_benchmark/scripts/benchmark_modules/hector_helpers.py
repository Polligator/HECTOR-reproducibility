#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd

from benchmark_modules.benchmark_core import (
    MODEL_DISPLAY_ORDER,
    benchmark_target_labels,
    clear_score_metrics,
    compute_metrics,
    display_name,
    ensure_dir,
    harmonize_series,
    load_or_create_cell_ids,
    load_worker_config,
    measure_device,
    onehot_score_frame,
    save_json,
    save_label_audit,
    save_metrics_table,
    save_model_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal Hector benchmark helper.")
    parser.add_argument("--worker", choices=["benchmark"], required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def prepare_subset(adata, chosen_cells: list[str]):
    subset = adata[chosen_cells].copy()
    if subset.raw is not None:
        subset.X = subset.raw.X
    return subset


def run_hector_prediction(
    config: dict,
    adata,
    *,
    verbose: bool = False,
):
    import hector

    holder: dict = {}

    def _run_hector() -> None:
        holder["predictor"] = hector.HECTOR(config["hector"]["model_path"], verbose=verbose)
        holder["predictions"] = holder["predictor"].predict(
            adata,
            label_format="name",
            top_k=config["hector"]["top_k"],
            use_grit=config["hector"]["use_grit"],
            use_asymmetric_ppr=config["hector"]["use_asymmetric_ppr"],
            ppr_alpha=config["hector"]["ppr_alpha"],
            forward_weight=config["hector"]["forward_weight"],
        )

    t0 = time.perf_counter()
    device = measure_device(_run_hector)
    inference_seconds = time.perf_counter() - t0
    return holder["predictor"], holder["predictions"], inference_seconds, device


def _index_hector_predictions(
    predictions: pd.DataFrame,
    *,
    expected_cell_ids: pd.Index | list[str] | None = None,
    context: str,
) -> pd.DataFrame:
    table = predictions.copy()

    if "cell_id" in table.columns:
        table["cell_id"] = table["cell_id"].astype(str)
        table = table.drop_duplicates(subset="cell_id", keep="last").set_index("cell_id")
    else:
        table.index = pd.Index(table.index.astype(str), name="cell_id")

    if table.index.has_duplicates:
        table = table[~table.index.duplicated(keep="last")]

    if expected_cell_ids is not None:
        expected_index = pd.Index(pd.Index(expected_cell_ids).astype(str), name="cell_id")
        table = table.reindex(expected_index)
        missing_mask = table.isna().all(axis=1)
        if missing_mask.any():
            missing_ids = table.index[missing_mask].tolist()
            preview = ", ".join(missing_ids[:10])
            raise ValueError(
                f"Hector predictions for {context} are missing {missing_mask.sum()} expected cells. "
                f"Examples: {preview}"
            )

    if "top_1_prediction" not in table.columns:
        raise ValueError(f"Hector predictions for {context} are missing required column 'top_1_prediction'.")

    return table


def run_hector_benchmark(config: dict) -> None:
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("Hector step needs the 'sc' conda environment.") from exc

    output_dir = ensure_dir(config["results_dir"])
    print(f"Loading dataset from {config['data_path']} for Hector...")
    adata = ad.read_h5ad(config["data_path"])
    if config.get("drop_truth_labels"):
        adata = adata[~adata.obs[config["label_key"]].astype(str).isin(config["drop_truth_labels"])].copy()
    chosen_cells = load_or_create_cell_ids(
        adata,
        sample_size=config["sample_size"],
        random_state=config["random_state"],
        cell_id_file=config["cell_id_file"],
    )
    subset = prepare_subset(adata, chosen_cells)
    predictor, predictions, inference_seconds, inference_device = run_hector_prediction(config, subset, verbose=False)
    subset_cell_ids = pd.Index(subset.obs_names.astype(str), name="cell_id")
    predictions_indexed = _index_hector_predictions(
        predictions,
        expected_cell_ids=subset_cell_ids,
        context="benchmark",
    )

    strict_truth = subset.obs[config["label_key"]].astype(str).reset_index(drop=True)
    strict_pred = predictions_indexed["top_1_prediction"].astype(str).reset_index(drop=True)
    truth = harmonize_series(strict_truth, label_map=config["label_map_overrides"]).reset_index(drop=True)
    pred = harmonize_series(strict_pred, label_map=config["label_map_overrides"]).reset_index(drop=True)
    target_labels = benchmark_target_labels(truth)
    target_label_set = set(target_labels)
    score_frame = onehot_score_frame(pred, target_labels)

    metrics, per_class, confusion_long = compute_metrics(
        y_true=truth,
        y_pred=pred,
        score_frame=score_frame,
        strict_y_true=strict_truth,
        strict_y_pred=strict_pred,
    )
    metrics = clear_score_metrics(metrics)
    metrics.update(
        {
            "model_name": "HECTOR",
            "display_name": display_name("HECTOR"),
            "available": True,
            "score_source": "label_only_post_processed",
            "inference_seconds": inference_seconds,
            "inference_device": inference_device,
        }
    )

    predictions_bundle = pd.DataFrame(
        {
            "cell_id": predictions_indexed.index.astype(str),
            "true_label_raw": strict_truth,
            "true_label": truth,
            "pred_label_raw": strict_pred,
            "pred_label": pred,
            "pred_in_target_class": pred.astype(str).isin(target_label_set),
        }
    )
    save_model_artifacts(output_dir, "hector", "HECTOR", per_class, confusion_long, predictions_bundle)
    save_label_audit(output_dir, "hector", "HECTOR", strict_truth, truth, strict_pred, pred)

    metrics_df = pd.DataFrame([metrics])
    metrics_df["plot_rank"] = metrics_df["model_name"].map({name: idx for idx, name in enumerate(MODEL_DISPLAY_ORDER)})
    metrics_df = metrics_df.sort_values("plot_rank").drop(columns="plot_rank")
    save_metrics_table(output_dir, "hector", metrics_df)
    save_json(
        output_dir / "hector_metadata.json",
        {
            "dataset_key": config["dataset_key"],
            "model_selector": str(config["hector"]["model_path"]),
            "model_path": str(Path(predictor.model_path).resolve()),
            "sample_size": len(chosen_cells),
            "use_grit": config["hector"]["use_grit"],
            "use_asymmetric_ppr": config["hector"]["use_asymmetric_ppr"],
            "ppr_alpha": config["hector"]["ppr_alpha"],
            "forward_weight": config["hector"]["forward_weight"],
        },
    )
    print(f"Hector outputs saved under {output_dir}")


_CL_ID_RE = re.compile(r"^[A-Za-z]+:\d+$")
_CL_ID_PAREN_RE = re.compile(r"\(([A-Za-z]+:\d+)\)\s*$")
_INVALID_ID_TOKENS = {"", "nan", "none", "na", "n/a", "unknown", "<na>"}


def _is_valid_cl_id(value: str) -> bool:
    """True for a usable ontology ID (not blank / nan / 'unknown')."""
    return bool(value) and str(value).strip().lower() not in _INVALID_ID_TOKENS


def _build_name_to_cl_id(predictor) -> dict[str, str]:
    """Reverse HECTOR's id->name map into name->id (first id wins on ties)."""
    id_to_name = getattr(predictor, "id_to_name_map", None) or {}
    name_to_id: dict[str, str] = {}
    for cl_id, name in id_to_name.items():
        name_to_id.setdefault(str(name), str(cl_id))
    return name_to_id


def _label_to_cl_id(label: str, name_to_id: dict[str, str]) -> str | None:
    """Resolve a prediction label to its ontology ID.

    Handles every label_format HECTOR can emit: a bare name, the combined
    ``"name (CL:0000128)"`` form, or a raw ``"CL:0000128"`` ID.
    """
    text = str(label).strip()
    paren = _CL_ID_PAREN_RE.search(text)
    if paren:
        return paren.group(1)
    if _CL_ID_RE.match(text):
        return text
    return name_to_id.get(text)


def main() -> None:
    args = parse_args()
    config = load_worker_config(args.config)
    if args.worker == "benchmark":
        run_hector_benchmark(config)


if __name__ == "__main__":
    main()
