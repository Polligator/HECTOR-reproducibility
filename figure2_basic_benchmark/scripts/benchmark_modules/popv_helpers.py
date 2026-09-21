#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_modules.benchmark_core import (
    MODEL_DISPLAY_ORDER,
    POPV_METHOD_TO_OBS_KEY,
    REQUESTED_POPV_METHODS,
    RUNTIME_TRAINABLE_POPV_METHODS,
    aggregate_score_frame,
    benchmark_target_labels,
    clear_score_metrics,
    compute_metrics,
    display_name,
    ensure_dir,
    harmonize_series,
    load_or_create_cell_ids,
    load_worker_config,
    prepare_popv_score_frame,
    resolve_popv_raw_score_frame,
    save_json,
    save_label_audit,
    save_model_audit,
    save_model_artifacts,
    save_metrics_table,
    measure_device,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal POPV helper.")
    parser.add_argument("--worker", choices=["benchmark", "audit"], required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def prepare_subset(adata, chosen_cells: list[str]):
    subset = adata[chosen_cells].copy()
    if subset.raw is not None:
        subset.X = subset.raw.X
    return subset


def import_popv(force_cpu: bool = False):
    import os
    import sys

    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_pop")
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cwd = os.getcwd()
    sys.path = [path for path in sys.path if path not in ("", cwd)]
    import popv  # type: ignore

    popv.settings.return_probabilities = True
    if force_cpu:
        popv.settings.accelerator = "cpu"
        popv.settings.cuml = False
        popv.settings.device = None
    patch_popv_onclass_runtime(popv)
    return popv


def patch_popv_onclass_runtime(popv) -> None:
    import logging
    import os

    import numpy as np
    import pandas as pd
    import scipy
    from OnClass.OnClassModel import OnClassModel
    from popv import settings
    from popv.algorithms import _onclass as onclass_module

    if getattr(onclass_module.ONCLASS.predict, "_benchmark_onclass_patch", False):
        return

    def patched_predict(self, adata):
        logging.info(f'Computing Onclass. Storing prediction in adata.obs["{self.result_key}"]')
        adata.obs.loc[adata.obs["_dataset"] == "query", "self.labels_key"] = adata.uns["unknown_celltype_label"]

        train_idx = adata.obs["_ref_subsample"]
        train_x = adata[train_idx].X.copy() if self.layer_key is None else adata[train_idx].layers[self.layer_key].copy()
        if scipy.sparse.issparse(train_x):
            train_x = train_x.todense()

        cl_ontology_file = adata.uns["_cl_ontology_file"]
        nlp_emb_file = adata.uns["_nlp_emb_file"]
        train_model = OnClassModel(cell_type_nlp_emb_file=nlp_emb_file, cell_type_network_file=cl_ontology_file)

        model_path = os.path.join(adata.uns["_save_path_trained_models"], "OnClass") if adata.uns["_save_path_trained_models"] is not None else None

        if adata.uns["_prediction_mode"] == "retrain":
            train_y = adata[train_idx].obs[self.labels_key]
            _ = train_model.EmbedCellTypes(train_y)
            corr_train_feature, corr_train_genes = train_model.ProcessTrainFeature(
                train_x,
                train_y,
                adata.var_names,
                log_transform=False,
            )
            train_model.BuildModel(ngene=len(corr_train_genes))
            train_model.Train(
                corr_train_feature,
                train_y,
                save_model=model_path,
                max_iter=self.max_iter,
            )
        else:
            train_model.BuildModel(ngene=None, use_pretrain=model_path)

        subset = adata[adata.obs["_predict_cells"] == "relabel"]
        test_x = subset.X.copy() if self.layer_key is None else subset.layers[self.layer_key].copy()
        if self.return_probabilities:
            required_columns = {
                self.seen_result_key: pd.Series(index=subset.obs_names, dtype=str),
                self.result_key: pd.Series(index=subset.obs_names, dtype=str),
                f"{self.result_key}_probabilities": pd.Series(index=subset.obs_names, dtype=float),
                f"{self.seen_result_key}_probabilities": pd.Series(index=subset.obs_names, dtype=float),
            }
        else:
            required_columns = {
                self.seen_result_key: pd.Series(index=subset.obs_names, dtype=str),
                self.result_key: pd.Series(index=subset.obs_names, dtype=str),
            }

        result_df = pd.DataFrame(required_columns)
        result_df_probabilities = None
        result_df_seen_probabilities = None
        shard_size = int(settings.shard_size)
        for i in range(0, subset.n_obs, shard_size):
            tmp_x = test_x[i : i + shard_size]
            if scipy.sparse.issparse(test_x):
                tmp_x = tmp_x.todense()
            names_x = subset.obs_names[i : i + shard_size]
            corr_test_feature = train_model.ProcessTestFeature(
                test_feature=tmp_x,
                test_genes=subset.var_names,
                use_pretrain=model_path,
                log_transform=False,
            )

            if adata.uns["_prediction_mode"] == "fast":
                onclass_pred = train_model.Predict(
                    corr_test_feature,
                    use_normalize=False,
                    refine=False,
                    unseen_ratio=-0.0,
                )
                onclass_seen = np.argmax(onclass_pred, axis=1)
                pred_label_str = [train_model.i2co[ind] for ind in onclass_seen]
                result_df.loc[names_x, self.result_key] = pred_label_str
                result_df.loc[names_x, self.seen_result_key] = pred_label_str
                if self.return_probabilities:
                    result_df.loc[names_x, f"{self.result_key}_probabilities"] = np.max(onclass_pred, axis=1)
                    result_df.loc[names_x, f"{self.seen_result_key}_probabilities"] = np.max(onclass_pred, axis=1)
                    if result_df_probabilities is None:
                        result_df_probabilities = pd.DataFrame(
                            index=subset.obs_names,
                            columns=[train_model.i2co[ind] for ind in range(onclass_pred.shape[1])],
                            dtype=float,
                        )
                    if result_df_seen_probabilities is None:
                        result_df_seen_probabilities = result_df_probabilities.copy()
                    result_df_probabilities.loc[names_x, :] = onclass_pred
                    result_df_seen_probabilities.loc[names_x, :] = onclass_pred
            else:
                onclass_pred = train_model.Predict(
                    corr_test_feature,
                    use_normalize=False,
                    refine=True,
                    unseen_ratio=-1.0,
                )
                pred_label_str = [train_model.i2co[ind] for ind in onclass_pred[2]]
                result_df.loc[names_x, self.result_key] = pred_label_str
                onclass_seen = np.argmax(onclass_pred[0], axis=1)
                seen_label_str = [train_model.i2co[ind] for ind in onclass_seen]
                result_df.loc[names_x, self.seen_result_key] = seen_label_str

                if self.return_probabilities:
                    refined_scores = onclass_pred[1]
                    seen_scores = onclass_pred[0]
                    result_df.loc[names_x, f"{self.result_key}_probabilities"] = np.max(refined_scores, axis=1) / refined_scores.sum(1)
                    result_df.loc[names_x, f"{self.seen_result_key}_probabilities"] = np.max(seen_scores, axis=1)
                    if result_df_probabilities is None:
                        result_df_probabilities = pd.DataFrame(
                            index=subset.obs_names,
                            columns=[train_model.i2co[ind] for ind in range(refined_scores.shape[1])],
                            dtype=float,
                        )
                    if result_df_seen_probabilities is None:
                        result_df_seen_probabilities = pd.DataFrame(
                            index=subset.obs_names,
                            columns=[train_model.i2co[ind] for ind in range(seen_scores.shape[1])],
                            dtype=float,
                        )
                    result_df_probabilities.loc[names_x, :] = refined_scores
                    result_df_seen_probabilities.loc[names_x, :] = seen_scores

        for col in required_columns.keys():
            if col not in adata.obs.columns:
                if "probabilities" in col:
                    adata.obs[col] = pd.Series(dtype="float64")
                else:
                    adata.obs[col] = adata.uns["unknown_celltype_label"]
                    adata.obs[col] = adata.obs[col].astype(str)
        adata.obs.loc[adata.obs["_predict_cells"] == "relabel", result_df.columns] = result_df
        if self.return_probabilities and result_df_probabilities is not None:
            refined_key = f"{self.result_key}_probabilities"
            seen_matrix_key = f"{self.seen_result_key}_probability_matrix"
            refined_matrix_key = f"{self.result_key}_probability_matrix"
            adata.obsm[refined_key] = pd.DataFrame(
                np.nan,
                index=adata.obs_names,
                columns=result_df_probabilities.columns,
            )
            adata.obsm[refined_key].loc[adata.obs["_predict_cells"] == "relabel", :] = (
                result_df_probabilities.loc[adata.obs_names[adata.obs["_predict_cells"] == "relabel"], :]
            )
            adata.obsm[refined_matrix_key] = adata.obsm[refined_key].copy()
            if result_df_seen_probabilities is not None:
                adata.obsm[seen_matrix_key] = pd.DataFrame(
                    np.nan,
                    index=adata.obs_names,
                    columns=result_df_seen_probabilities.columns,
                )
                adata.obsm[seen_matrix_key].loc[adata.obs["_predict_cells"] == "relabel", :] = (
                    result_df_seen_probabilities.loc[adata.obs_names[adata.obs["_predict_cells"] == "relabel"], :]
                )

    patched_predict._benchmark_onclass_patch = True
    onclass_module.ONCLASS.predict = patched_predict


def _annotate_with_timing(
    hub_model,
    popv_module,
    query_adata,
    *,
    query_batch_key: str | None,
    prediction_mode: str,
    methods: list[str],
    save_path: str,
) -> tuple[object, dict[str, float], float]:
    """Replicate hub_model.annotate_data() with isolated per-method timing.

    Each method is run on its OWN fresh copy of the post-Process_Query object,
    so no method ever reuses an integration/embedding that another method left
    behind (e.g. the scVI latent), and one method's cost can never leak into
    another method's timer. The shared preprocessing (Process_Query, which
    builds X_pca and trains nothing) is timed once and returned separately.
    ``compute_umap`` is intentionally NOT called: it is visualization-only and
    is not consumed downstream (figures read X_umap from the source dataset).

    There is no error handling here on purpose -- if any method fails the
    exception propagates so the whole run stops (all methods must succeed).

    Returns (annotated_adata, method_timings, preprocess_seconds) where
    method_timings maps each method to its integration+predict seconds and
    "POPV" to the total pipeline time. preprocess_seconds is the shared
    Process_Query cost and is deliberately excluded from every method's time.
    """
    import os

    from popv.annotation import (
        compute_consensus,
        ontology_parent_onclass,
        ontology_vote_onclass,
    )
    from popv.preprocessing import Process_Query

    ref_adata = hub_model.adata if prediction_mode == "retrain" else hub_model.minified_adata
    setup_dict = hub_model.metadata.setup_dict

    t_total = time.perf_counter()

    # --- shared preprocessing (gene alignment, ref subsample, X_pca) ---
    t_pre = time.perf_counter()
    base = Process_Query(
        query_adata,
        ref_adata,
        query_batch_key=query_batch_key,
        ref_labels_key=setup_dict["ref_labels_key"],
        ref_batch_key=setup_dict["ref_batch_key"],
        unknown_celltype_label=setup_dict["unknown_celltype_label"],
        save_path_trained_models=hub_model._local_dir,
        cl_obo_folder=hub_model.ontology_dir,
        prediction_mode=prediction_mode,
        n_samples_per_label=100,
        hvg=None,
    ).adata
    preprocess_seconds = time.perf_counter() - t_pre

    # The pretrained reference ships its popv_* prediction columns as
    # Categorical with a fixed category set; pandas >= 2.2 refuses to write a
    # query prediction outside that set. Coerced to object so every method can
    # write freely. popv_labels stays categorical: methods read it via .cat.codes.
    for col in base.obs.columns:
        if (
            isinstance(base.obs[col].dtype, pd.CategoricalDtype)
            and col.startswith("popv_")
            and col != "popv_labels"
        ):
            base.obs[col] = base.obs[col].astype(object)

    if base.uns["_cl_obo_file"] is False and "ONCLASS" in methods:
        methods = [m for m in methods if m != "ONCLASS"]

    methods_kwargs = dict(hub_model.metadata.method_kwargs or {})
    method_timings: dict[str, float] = {}
    method_devices: dict[str, str] = {}
    all_prediction_keys: list[str] = []
    all_prediction_keys_seen: list[str] = []

    # Master object that accumulates every method's predictions for the
    # consensus. Each method itself runs on a throwaway copy of ``base``.
    annotated = base.copy()

    for method_name in methods:
        # Fresh, isolated copy: nothing a method computes can ride into the next
        # method's timer, and nothing it consumes was produced by a sibling.
        work = base.copy()
        current = getattr(popv_module.algorithms, method_name)(
            **methods_kwargs.pop(method_name, {})
        )

        def _run_method(current=current, work=work) -> None:
            current.compute_integration(work)
            current.predict(work)

        t0 = time.perf_counter()
        method_devices[method_name] = measure_device(_run_method)
        method_timings[method_name] = time.perf_counter() - t0

        # Merges this method's outputs into the master. Its prediction columns
        # may already exist in base, so copying only brand-new columns would
        # drop the query predictions; also copies anything under this method's
        # own result keys (ref values for reference cells, fresh predictions
        # for query cells). Copied positionally: work and base share cell order.
        owned = tuple(k for k in (current.result_key, current.seen_result_key) if k)

        def _is_owned(name: str) -> bool:
            return any(name == prefix or name.startswith(prefix) for prefix in owned)

        for col in work.obs.columns:
            if col not in base.obs.columns or _is_owned(col):
                annotated.obs[col] = work.obs[col].to_numpy()
        for key in work.obsm:
            if key not in base.obsm or _is_owned(key):
                annotated.obsm[key] = work.obsm[key]
        for key in work.uns:
            if key not in base.uns:
                annotated.uns[key] = work.uns[key]

        all_prediction_keys.append(current.result_key)
        all_prediction_keys_seen.append(current.seen_result_key)

    annotated.uns["prediction_keys"] = all_prediction_keys
    annotated.uns["prediction_keys_seen"] = all_prediction_keys_seen
    annotated.uns["methods"] = list(methods)
    annotated.uns["method_kwargs"] = methods_kwargs
    compute_consensus(annotated, all_prediction_keys_seen)
    if annotated.uns["_cl_obo_file"] is False:
        annotated.obs[["popv_prediction", "popv_prediction_score"]] = annotated.obs[
            ["popv_majority_vote_prediction", "popv_majority_vote_score"]
        ]
        annotated.obs[["popv_parent"]] = annotated.obs[["popv_majority_vote_prediction"]]
    else:
        ontology_vote_onclass(annotated, all_prediction_keys)
        ontology_parent_onclass(annotated, all_prediction_keys)

    popv_output_dir = os.path.join(save_path, "popv_output")
    os.makedirs(popv_output_dir, exist_ok=True)
    annotated[annotated.obs["_dataset"] == "query"].obs[
        [
            *all_prediction_keys,
            "popv_prediction",
            "popv_prediction_score",
            "popv_majority_vote_prediction",
            "popv_majority_vote_score",
            "popv_parent",
        ]
    ].to_csv(os.path.join(popv_output_dir, "predictions.csv"))

    method_timings["POPV"] = time.perf_counter() - t_total
    # The consensus spans every method, so its backend is genuinely mixed.
    method_devices["POPV"] = "mixed"
    return annotated, method_timings, method_devices, preprocess_seconds


def run_popv_workbench(config: dict) -> None:
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("POPV step needs the 'sc' conda environment.") from exc

    popv = import_popv(force_cpu=config["popv"].get("force_cpu", False))
    output_dir = ensure_dir(config["results_dir"])

    print(f"Loading dataset from {config['data_path']} for POPV...")
    adata = ad.read_h5ad(config["data_path"])
    if config.get("drop_truth_labels"):
        adata = adata[~adata.obs[config["label_key"]].astype(str).isin(config["drop_truth_labels"])].copy()
    chosen_cells = load_or_create_cell_ids(
        adata,
        sample_size=config["sample_size"],
        random_state=config["random_state"],
        cell_id_file=config["cell_id_file"],
    )
    query_adata = prepare_subset(adata, chosen_cells)

    cache_dir = config["popv"].get("cache_dir")
    pull_kwargs = {}
    if cache_dir:
        pull_kwargs["cache_dir"] = cache_dir
    hub_model = popv.hub.HubModel.pull_from_huggingface_hub(config["popv"]["repo"], **pull_kwargs)
    # A method runs if popv exposes its algorithm class AND either the hub model saved its
    # artifact (metadata.methods) or it trains entirely at inference time
    # (RUNTIME_TRAINABLE_POPV_METHODS). Methods missing from metadata that need a pretrained
    # artifact (e.g. Random_Forest -> rf_classifier.joblib) stay unavailable in inference mode.
    def _method_runnable(method: str) -> bool:
        return hasattr(popv.algorithms, method) and (
            method in hub_model.metadata.methods or method in RUNTIME_TRAINABLE_POPV_METHODS
        )

    available_methods = [method for method in REQUESTED_POPV_METHODS if _method_runnable(method)]
    missing_methods = [method for method in REQUESTED_POPV_METHODS if not _method_runnable(method)]
    prediction_mode_used = config["popv"]["prediction_mode"]
    runtime_missing_methods: list[str] = []
    runtime_error = None
    methods_to_run = available_methods

    if prediction_mode_used == "fast":
        from popv.annotation import algorithms_nt  # type: ignore

        methods_to_run = [method for method in available_methods if method in algorithms_nt.FAST_ALGORITHMS]
        runtime_missing_methods = [method for method in available_methods if method not in methods_to_run]

    print(f"Running POPV with methods: {methods_to_run}")
    # No fast-mode fallback: every method must run successfully in the
    # configured mode. If one fails, the exception propagates and the run stops
    # rather than silently downgrading to fast mode (which would record 1-epoch
    # timings in place of the real training cost).
    annotated, method_timings, method_devices, preprocess_seconds = _annotate_with_timing(
        hub_model,
        popv,
        query_adata,
        query_batch_key=config["batch_key"],
        prediction_mode=prediction_mode_used,
        methods=methods_to_run,
        save_path=str(output_dir / "popv_cache"),
    )

    query_mask = annotated.obs["_predict_cells"].astype(str).eq("relabel")
    query = annotated[query_mask].copy()
    strict_truth = query.obs[config["label_key"]].astype(str).reset_index(drop=True)
    truth = harmonize_series(strict_truth, label_map=config["label_map_overrides"]).reset_index(drop=True)
    target_labels = benchmark_target_labels(truth)
    target_label_set = set(target_labels)

    metrics_rows = []
    for model_name in ["POPV", *REQUESTED_POPV_METHODS]:
        obs_key = POPV_METHOD_TO_OBS_KEY.get(model_name)
        if obs_key is None or obs_key not in query.obs.columns:
            metrics_rows.append(
                {
                    "model_name": model_name,
                    "display_name": display_name(model_name),
                    "available": False,
                    "notes": "Prediction column unavailable in this hub model.",
                }
            )
            continue

        model_audit = {
            "dataset_key": config["dataset_key"],
            "model_name": model_name,
            "obs_key": obs_key,
            "n_cells": int(len(query)),
            "benchmark_target_labels": target_labels,
        }
        try:
            strict_pred = query.obs[obs_key].astype(str).reset_index(drop=True)
            pred = harmonize_series(strict_pred, label_map=config["label_map_overrides"]).reset_index(drop=True)
            score_argmax_pred = pred.copy()
            prob_key = f"{obs_key}_probabilities"
            model_audit["probability_key"] = prob_key
            skip_score_metrics = model_name in {"CELLTYPIST", "POPV"}
            if skip_score_metrics:
                classes = sorted(set(truth) | set(pred))
                score_frame = pd.DataFrame(
                    np.eye(len(classes))[pd.Categorical(pred, categories=classes).codes],
                    columns=classes,
                )
                score_source = "label_only_majority_vote"
                if model_name == "CELLTYPIST":
                    audit_reason = "CELLTYPIST uses majority_voting labels; probability_matrix is not used for ROC/PR/AUC."
                else:
                    audit_reason = "POPV consensus prediction does not expose a per-class probability matrix."
                model_audit["score_validation"] = {
                    "reason": audit_reason,
                    "fallback_classes": classes,
                }
            elif prob_key in query.obsm:
                raw_scores = query.obsm[prob_key]
                score_frame, score_argmax_pred, score_audit = prepare_popv_score_frame(
                    model_name=model_name,
                    raw_scores=raw_scores,
                    query_index=query.obs_names,
                    strict_pred=strict_pred,
                    pred=pred,
                    adata=query,
                    label_map=config["label_map_overrides"],
                )
                model_audit["score_validation"] = score_audit
                score_source = "probability_matrix"
            else:
                classes = sorted(set(truth) | set(pred))
                score_frame = pd.DataFrame(
                    np.eye(len(classes))[pd.Categorical(pred, categories=classes).codes],
                    columns=classes,
                )
                score_source = "hard_label_onehot_fallback"
                model_audit["score_validation"] = {
                    "fallback_classes": classes,
                    "reason": f"{prob_key} missing from query.obsm",
                }
            model_audit["score_source"] = score_source
            model_audit["metric_prediction_override_count"] = int(
                model_audit.get("score_validation", {}).get("aggregated_argmax_mismatch_count", 0)
            )
            model_audit["label_metric_prediction_source"] = "obs_prediction"
            model_audit["score_metric_source"] = "harmonized_score_matrix"
            off_target_pred_counts = (
                pred.astype(str).value_counts().loc[lambda s: ~s.index.isin(target_label_set)].sort_index().to_dict()
            )
            model_audit["off_target_pred_labels"] = sorted(off_target_pred_counts)
            model_audit["off_target_pred_counts"] = off_target_pred_counts

            metrics, per_class, confusion_long = compute_metrics(
                y_true=truth,
                y_pred=pred,
                score_frame=score_frame,
                strict_y_true=strict_truth,
                strict_y_pred=strict_pred,
            )
            if skip_score_metrics:
                metrics = clear_score_metrics(metrics)
                if model_name == "CELLTYPIST":
                    skip_reason = "CELLTYPIST majority_voting labels are not aligned to the per-cell probability matrix."
                else:
                    skip_reason = "POPV consensus prediction does not expose a per-class probability matrix; score-based metrics are not meaningful."
                model_audit["score_metrics_skipped"] = True
                model_audit["score_metrics_skip_reason"] = skip_reason
            else:
                model_audit["score_metrics_skipped"] = False
            metrics.update(
                {
                    "model_name": model_name,
                    "display_name": display_name(model_name),
                    "available": True,
                    "score_source": score_source,
                    "metric_prediction_override_count": model_audit["metric_prediction_override_count"],
                    "inference_seconds": method_timings.get(model_name),
                    "inference_device": method_devices.get(model_name),
                }
            )
            metrics_rows.append(metrics)
            predictions = pd.DataFrame(
                {
                    "cell_id": query.obs_names.astype(str),
                    "true_label_raw": strict_truth,
                    "true_label": truth,
                    "pred_label_raw": strict_pred,
                    "pred_label": pred,
                    "score_argmax_label": score_argmax_pred if prob_key in query.obsm and not skip_score_metrics else pred,
                    "pred_label_source": "obs_prediction",
                    "pred_in_target_class": pred.astype(str).isin(target_label_set),
                }
            )
            save_model_artifacts(output_dir, "popv", model_name, per_class, confusion_long, predictions)
            save_label_audit(output_dir, "popv", model_name, strict_truth, truth, strict_pred, pred)
            save_model_audit(output_dir, "popv", model_name, model_audit)
        except Exception as exc:
            model_audit["error"] = repr(exc)
            save_model_audit(output_dir, "popv", model_name, model_audit)
            raise

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["plot_rank"] = metrics_df["model_name"].map({name: idx for idx, name in enumerate(MODEL_DISPLAY_ORDER)})
    metrics_df = metrics_df.sort_values(["plot_rank", "model_name"], na_position="last").drop(columns="plot_rank")
    save_metrics_table(output_dir, "popv", metrics_df)

    save_json(
        output_dir / "popv_metadata.json",
        {
            "dataset_key": config["dataset_key"],
            "repo": config["popv"]["repo"],
            "sample_size": len(chosen_cells),
            "batch_key": config["batch_key"],
            "requested_methods": REQUESTED_POPV_METHODS,
            "available_methods": available_methods,
            "missing_methods": missing_methods,
            "prediction_mode_requested": config["popv"]["prediction_mode"],
            "prediction_mode_used": prediction_mode_used,
            "runtime_missing_methods": runtime_missing_methods,
            "runtime_error": runtime_error,
            "preprocess_seconds": preprocess_seconds,
            "method_timings": method_timings,
        },
    )
    save_json(
        output_dir / "run_context.json",
        {
            "dataset_key": config["dataset_key"],
            "cell_id_file": str(Path(config["cell_id_file"]).resolve()),
            "sample_size": len(chosen_cells),
            "random_state": config["random_state"],
            "data_path": str(Path(config["data_path"]).resolve()),
        },
    )

    cache_predictions = output_dir / "popv_cache" / "predictions.csv"
    if cache_predictions.exists():
        shutil.copy2(cache_predictions, output_dir / "popv_hub_predictions.csv")
    print(f"POPV outputs saved under {output_dir}")


def _write_stage_bundle(
    audit_dir: Path,
    model_name: str,
    stage_rows: pd.DataFrame,
    raw_scores: pd.DataFrame | None,
    harmonized_scores: pd.DataFrame | None,
    extra_scores: dict[str, pd.DataFrame] | None = None,
) -> None:
    stage_rows.to_csv(audit_dir / f"popv_audit_rows_{model_name}.csv", index=False)
    if raw_scores is not None:
        raw_scores.to_csv(audit_dir / f"popv_audit_raw_scores_{model_name}.csv", index=True)
    if harmonized_scores is not None:
        harmonized_scores.to_csv(audit_dir / f"popv_audit_harmonized_scores_{model_name}.csv", index=True)
    if extra_scores:
        for suffix, frame in extra_scores.items():
            frame.to_csv(audit_dir / f"popv_audit_{suffix}_{model_name}.csv", index=True)


def run_popv_audit(config: dict) -> None:
    output_dir = ensure_dir(Path(config["results_dir"]) / "popv_audit")

    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("POPV audit needs the 'sc' conda environment.") from exc

    popv = import_popv(force_cpu=config["popv"].get("force_cpu", False))
    adata = ad.read_h5ad(config["data_path"])
    if config.get("drop_truth_labels"):
        adata = adata[~adata.obs[config["label_key"]].astype(str).isin(config["drop_truth_labels"])].copy()
    chosen_cells = load_or_create_cell_ids(
        adata,
        sample_size=config["sample_size"],
        random_state=config["random_state"],
        cell_id_file=config["cell_id_file"],
    )
    query_adata = prepare_subset(adata, chosen_cells)

    cache_dir = config["popv"].get("cache_dir")
    pull_kwargs = {}
    if cache_dir:
        pull_kwargs["cache_dir"] = cache_dir
    hub_model = popv.hub.HubModel.pull_from_huggingface_hub(config["popv"]["repo"], **pull_kwargs)
    available_methods = [
        method for method in REQUESTED_POPV_METHODS
        if hasattr(popv.algorithms, method)
        and (method in hub_model.metadata.methods or method in RUNTIME_TRAINABLE_POPV_METHODS)
    ]
    prediction_mode_used = config["popv"]["prediction_mode"]
    methods_to_run = available_methods
    if prediction_mode_used == "fast":
        from popv.annotation import algorithms_nt  # type: ignore

        methods_to_run = [method for method in available_methods if method in algorithms_nt.FAST_ALGORITHMS]

    try:
        annotated = hub_model.annotate_data(
            query_adata=query_adata,
            query_batch_key=config["batch_key"],
            prediction_mode=prediction_mode_used,
            methods=methods_to_run,
            save_path=str(output_dir / "popv_cache"),
        )
    except Exception:
        from popv.annotation import algorithms_nt  # type: ignore

        prediction_mode_used = "fast"
        methods_to_run = [method for method in available_methods if method in algorithms_nt.FAST_ALGORITHMS]
        query_adata = prepare_subset(adata, chosen_cells)
        annotated = hub_model.annotate_data(
            query_adata=query_adata,
            query_batch_key=config["batch_key"],
            prediction_mode="fast",
            methods=methods_to_run,
            save_path=str(output_dir / "popv_cache"),
        )

    query_mask = annotated.obs["_predict_cells"].astype(str).eq("relabel")
    query = annotated[query_mask].copy()
    strict_truth = query.obs[config["label_key"]].astype(str).reset_index(drop=True)
    truth = harmonize_series(strict_truth, label_map=config["label_map_overrides"]).reset_index(drop=True)
    target_labels = benchmark_target_labels(truth)
    target_label_set = set(target_labels)

    for model_name in ["POPV", *REQUESTED_POPV_METHODS]:
        obs_key = POPV_METHOD_TO_OBS_KEY.get(model_name)
        audit_payload: dict[str, object] = {
            "dataset_key": config["dataset_key"],
            "model_name": model_name,
            "obs_key": obs_key,
            "n_cells": int(len(query)),
            "prediction_mode_used": prediction_mode_used,
            "benchmark_target_labels": target_labels,
        }
        raw_scores_frame = None
        harmonized_scores_frame = None
        stage_rows = None
        extra_score_frames: dict[str, pd.DataFrame] = {}
        try:
            if obs_key is None or obs_key not in query.obs.columns:
                audit_payload["available"] = False
                audit_payload["error"] = f"Prediction column missing: {obs_key}"
                save_json(output_dir / f"popv_audit_{model_name}.json", audit_payload)
                continue

            strict_pred = query.obs[obs_key].astype(str).reset_index(drop=True)
            pred = harmonize_series(strict_pred, label_map=config["label_map_overrides"]).reset_index(drop=True)
            prob_key = f"{obs_key}_probabilities"
            audit_payload["probability_key"] = prob_key
            audit_payload["available"] = True
            skip_score_metrics = model_name == "CELLTYPIST"

            if skip_score_metrics:
                classes = sorted(set(truth) | set(pred))
                harmonized_scores_frame = pd.DataFrame(
                    np.eye(len(classes))[pd.Categorical(pred, categories=classes).codes],
                    index=query.obs_names.astype(str),
                    columns=classes,
                )
                audit_payload["score_validation"] = {
                    "reason": "CELLTYPIST uses majority_voting labels; probability_matrix is not used for ROC/PR/AUC.",
                    "fallback_classes": classes,
                }
                stage_rows = pd.DataFrame(
                    {
                        "cell_id": query.obs_names.astype(str),
                        "truth_raw": strict_truth,
                        "truth_harmonized": truth,
                        "pred_raw": strict_pred,
                        "pred_harmonized_from_obs": pred,
                        "score_argmax_raw": [None] * len(query),
                        "score_argmax_harmonized": pred,
                    }
                )
            elif prob_key in query.obsm:
                raw_scores = query.obsm[prob_key]
                raw_scores_frame, raw_score_audit = resolve_popv_raw_score_frame(
                    model_name=model_name,
                    raw_scores=raw_scores,
                    query_index=query.obs_names,
                    adata=query,
                )
                harmonized_scores_frame = aggregate_score_frame(raw_scores_frame, label_map=config["label_map_overrides"])
                stage_rows = pd.DataFrame(
                    {
                        "cell_id": query.obs_names.astype(str),
                        "truth_raw": strict_truth,
                        "truth_harmonized": truth,
                        "pred_raw": strict_pred,
                        "pred_harmonized_from_obs": pred,
                        "score_argmax_raw": raw_scores_frame.idxmax(axis=1).astype(str).reset_index(drop=True),
                        "score_argmax_harmonized": harmonized_scores_frame.idxmax(axis=1).astype(str).reset_index(drop=True),
                    }
                )
                harmonized_scores_frame, _, score_audit = prepare_popv_score_frame(
                    model_name=model_name,
                    raw_scores=raw_scores,
                    query_index=query.obs_names,
                    strict_pred=strict_pred,
                    pred=pred,
                    adata=query,
                    label_map=config["label_map_overrides"],
                )
                harmonized_scores_frame.index = query.obs_names.astype(str)
                raw_scores_frame.index = query.obs_names.astype(str)
                audit_payload["score_resolution"] = raw_score_audit
                audit_payload["score_validation"] = score_audit
                if model_name == "ONCLASS":
                    seen_matrix_key = "popv_onclass_seen_probability_matrix"
                    refined_matrix_key = "popv_onclass_prediction_probability_matrix"
                    if seen_matrix_key in query.obsm:
                        seen_frame, seen_audit = resolve_popv_raw_score_frame(
                            model_name=f"{model_name}_seen_matrix",
                            raw_scores=query.obsm[seen_matrix_key],
                            query_index=query.obs_names,
                            adata=query,
                        )
                        extra_score_frames["raw_scores_seen_matrix"] = seen_frame
                        audit_payload["onclass_seen_matrix_resolution"] = seen_audit
                        audit_payload["onclass_seen_label_match_fraction"] = float(
                            (
                                seen_frame.idxmax(axis=1).astype(str).reset_index(drop=True)
                                == query.obs["popv_onclass_seen"].astype(str).reset_index(drop=True)
                            ).mean()
                        )
                    if refined_matrix_key in query.obsm:
                        refined_frame, refined_audit = resolve_popv_raw_score_frame(
                            model_name=f"{model_name}_refined_matrix",
                            raw_scores=query.obsm[refined_matrix_key],
                            query_index=query.obs_names,
                            adata=query,
                        )
                        extra_score_frames["raw_scores_refined_matrix"] = refined_frame
                        audit_payload["onclass_refined_matrix_resolution"] = refined_audit
                        audit_payload["onclass_refined_label_match_fraction"] = float(
                            (
                                refined_frame.idxmax(axis=1).astype(str).reset_index(drop=True)
                                == strict_pred.reset_index(drop=True)
                            ).mean()
                        )
            else:
                classes = sorted(set(truth) | set(pred))
                harmonized_scores_frame = pd.DataFrame(
                    np.eye(len(classes))[pd.Categorical(pred, categories=classes).codes],
                    index=query.obs_names.astype(str),
                    columns=classes,
                )
                audit_payload["score_validation"] = {
                    "reason": f"{prob_key} missing from query.obsm",
                    "fallback_classes": classes,
                }
                stage_rows = pd.DataFrame(
                    {
                        "cell_id": query.obs_names.astype(str),
                        "truth_raw": strict_truth,
                        "truth_harmonized": truth,
                        "pred_raw": strict_pred,
                        "pred_harmonized_from_obs": pred,
                        "score_argmax_raw": [None] * len(query),
                        "score_argmax_harmonized": harmonized_scores_frame.idxmax(axis=1).astype(str).reset_index(drop=True),
                    }
                )

            stage_rows["pred_harmonized"] = pred
            stage_rows["pred_label_source"] = "obs_prediction"
            stage_rows["pred_in_target_class"] = pred.astype(str).isin(target_label_set)
            audit_payload["metric_prediction_override_count"] = int(
                audit_payload.get("score_validation", {}).get("aggregated_argmax_mismatch_count", 0)
            )
            audit_payload["label_metric_prediction_source"] = "obs_prediction"
            audit_payload["score_metric_source"] = "harmonized_score_matrix"
            off_target_pred_counts = (
                pred.astype(str).value_counts().loc[lambda series: ~series.index.isin(target_label_set)].sort_index().to_dict()
            )
            audit_payload["off_target_pred_labels"] = sorted(off_target_pred_counts)
            audit_payload["off_target_pred_counts"] = off_target_pred_counts

            metrics, _, _ = compute_metrics(
                y_true=truth,
                y_pred=pred,
                score_frame=harmonized_scores_frame,
                strict_y_true=strict_truth,
                strict_y_pred=strict_pred,
            )
            if skip_score_metrics:
                metrics = clear_score_metrics(metrics)
                audit_payload["score_metrics_skipped"] = True
                audit_payload["score_metrics_skip_reason"] = "CELLTYPIST majority_voting labels are not aligned to the per-cell probability matrix."
            else:
                audit_payload["score_metrics_skipped"] = False
            audit_payload["metrics"] = metrics
            _write_stage_bundle(output_dir, model_name, stage_rows, raw_scores_frame, harmonized_scores_frame, extra_score_frames)
            save_json(output_dir / f"popv_audit_{model_name}.json", audit_payload)
        except Exception as exc:
            audit_payload["error"] = repr(exc)
            if stage_rows is not None:
                if "pred_harmonized" in stage_rows.columns and "score_argmax_harmonized" in stage_rows.columns:
                    mismatch_mask = stage_rows["pred_harmonized"] != stage_rows["score_argmax_harmonized"]
                    audit_payload["aggregated_argmax_mismatch_count"] = int(mismatch_mask.sum())
                    mismatch_rows = stage_rows.loc[mismatch_mask].copy()
                    mismatch_path = output_dir / f"popv_audit_mismatches_{model_name}.csv"
                    mismatch_rows.to_csv(mismatch_path, index=False)
                    audit_payload["mismatch_artifact"] = str(mismatch_path)
                _write_stage_bundle(output_dir, model_name, stage_rows, raw_scores_frame, harmonized_scores_frame, extra_score_frames)
            save_json(output_dir / f"popv_audit_{model_name}.json", audit_payload)
            continue


def main() -> None:
    args = parse_args()
    config = load_worker_config(args.config)
    if args.worker == "benchmark":
        run_popv_workbench(config)
    else:
        run_popv_audit(config)


if __name__ == "__main__":
    main()
