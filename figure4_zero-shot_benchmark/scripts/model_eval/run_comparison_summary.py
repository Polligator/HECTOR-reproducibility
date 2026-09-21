"""# Summarize the OnClass–scCello native zero-shot comparison

This analysis scores both models on the unchanged 66-type query set and on the
31-type subset that is unseen by both training resources and represented in
both native ontologies. Missing model outputs are padded as errors.
"""

from pathlib import Path
import sys

import pandas as pd


# ## 1. Resolve prediction tables and the shared evaluator
# The HECTOR and shared-decoder rows are context, while the native checkpoint
# rows are the direct comparison requested here.
figure_root = Path(__file__).resolve().parents[2]
model_eval_root = figure_root / "scripts/model_eval"
data_root = figure_root / "result/data"

if str(model_eval_root) not in sys.path:
    sys.path.insert(0, str(model_eval_root))

import evaluate as evaluate_module
from query_set_padding import pad_to_query_set

method_rows = [
    ("scCello", "shared_mean_pool_ppr", data_root / "sccello_ppr_candidate_predictions.csv", 66),
    ("scCello", "native_checkpoint_closed", data_root / "sccello_native_closed_set_predictions.csv", 60),
    ("scCello", "native_checkpoint_open", data_root / "sccello_native_open_set_predictions.csv", 2602),
    ("scCello", "native_checkpoint_strict_common31", data_root / "sccello_native_strict_common31_predictions.csv", 31),
    ("OnClass", "native_released_ensemble_closed", data_root / "onclass_native_released_ensemble_closed_set_predictions.csv", 54),
    ("OnClass", "native_released_ensemble_open", data_root / "onclass_native_released_ensemble_open_set_predictions.csv", 2149),
    ("OnClass", "native_released_ensemble_strict_common31", data_root / "onclass_native_released_ensemble_strict_common31_predictions.csv", 31),
    ("OnClass", "native_cellwise_ensemble_closed", data_root / "onclass_native_cellwise_ensemble_closed_set_predictions.csv", 54),
    ("OnClass", "native_cellwise_ensemble_open", data_root / "onclass_native_cellwise_ensemble_open_set_predictions.csv", 2149),
    ("OnClass", "native_cellwise_ensemble_strict_common31", data_root / "onclass_native_cellwise_ensemble_strict_common31_predictions.csv", 31),
    ("HECTOR", "native_closed_reference", data_root / "hector_native_closed_set_predictions.csv", 66),
    ("HECTOR", "native_open_reference", data_root / "hector_native_open_set_predictions.csv", 1407),
    ("HECTOR", "native_strict_common31", data_root / "hector_native_strict_common31_predictions.csv", 31),
]


# ## 2. Define the full and strict-common evaluation strata
# The strict flag was derived exclusively from training manifests and native
# ontology coverage, before any prediction result was inspected.
audit = pd.read_csv(data_root / "training_and_ontology_audit.csv")
audit["cell_type_id"] = audit["cell_type_id"].astype(str)
all_type_ids = audit["cell_type_id"].tolist()
strict_type_ids = audit.loc[
    audit["strict_common_zero_shot"].astype(bool), "cell_type_id"
].tolist()

subset_rows = [
    ("all_66_holdouts", all_type_ids),
    ("strict_truth_stratum_within_native_task", strict_type_ids),
]


# ## 3. Score every method with the same missing-row policy
# Accuracy is micro over cells; macro F1 and macro hierarchical F1 weight each
# held-out type equally. Coverage reports the fraction the model processed.
summary_rows = []
per_type_rows = []

for model_name, method_name, prediction_path, candidate_count in method_rows:
    predictions = pd.read_csv(prediction_path)
    predictions["truth_cell_type_id"] = predictions["truth_cell_type_id"].astype(str)

    for subset_name, subset_ids in subset_rows:
        is_common31_method = "strict_common31" in method_name
        if is_common31_method and subset_name == "all_66_holdouts":
            continue

        reported_subset_name = subset_name
        if is_common31_method:
            reported_subset_name = "strict_common31_identical_candidates"

        expected_table = audit.loc[audit["cell_type_id"].isin(subset_ids)].copy()
        expected_counts = expected_table.set_index("cell_type_id")["n_cells"]
        subset_predictions = predictions.loc[
            predictions["truth_cell_type_id"].isin(subset_ids)
        ].copy()

        summary = evaluate_module.summarize_prediction_table(
            subset_predictions,
            expected_query_counts=expected_counts,
        )
        hierarchical = summary["hierarchical_summary"]
        number_expected = int(expected_counts.sum())
        number_returned = len(subset_predictions)

        summary_rows.append(
            {
                "model": model_name,
                "method": method_name,
                "subset": reported_subset_name,
                "number_cell_types": len(subset_ids),
                "candidate_count": candidate_count,
                "number_expected_cells": number_expected,
                "number_predictions_returned": number_returned,
                "coverage": number_returned / number_expected,
                "accuracy_on_returned_cells": (
                    subset_predictions["truth_cell_type_id"].astype(str)
                    == subset_predictions["predicted_cell_type_id"].astype(str)
                ).mean(),
                "accuracy": summary["accuracy"],
                "macro_f1": summary["macro_f1"],
                "weighted_f1": summary["weighted_f1"],
                "micro_hierarchical_f1": hierarchical.loc["micro", "hierarchical_f1"],
                "macro_hierarchical_f1": hierarchical.loc["macro", "hierarchical_f1"],
                "micro_hierarchical_accuracy": hierarchical.loc["micro", "hierarchical_accuracy"],
                "macro_hierarchical_accuracy": hierarchical.loc["macro", "hierarchical_accuracy"],
                "micro_lca_depth": hierarchical.loc["micro", "lca_depth"],
                "macro_lca_depth": hierarchical.loc["macro", "lca_depth"],
            }
        )

        padded = pad_to_query_set(subset_predictions, expected_counts)
        padded["flat_correct"] = (
            padded["truth_cell_type_id"].astype(str)
            == padded["predicted_cell_type_id"].astype(str)
        ).astype(float)
        type_summary = padded.groupby("truth_cell_type_id", as_index=False).agg(
            number_scored=("predicted_cell_type_id", lambda values: int((values != "__not_returned__").sum())),
            accuracy=("flat_correct", "mean"),
            mean_hierarchical_f1=("hierarchical_f1", "mean"),
        )
        type_summary["model"] = model_name
        type_summary["method"] = method_name
        type_summary["subset"] = reported_subset_name
        type_summary["number_expected"] = type_summary[
            "truth_cell_type_id"
        ].map(expected_counts)
        type_summary["coverage"] = (
            type_summary["number_scored"] / type_summary["number_expected"]
        )
        per_type_rows.append(type_summary)


# ## 4. Save analysis-ready summary tables
# Sorting keeps native model rows adjacent and makes later plotting stable.
summary_table = pd.DataFrame(summary_rows)
summary_table = summary_table.sort_values(
    ["subset", "model", "method"], ignore_index=True
)
per_type_table = pd.concat(per_type_rows, ignore_index=True)
per_type_table = per_type_table.merge(
    audit[["cell_type_id", "cell_type"]],
    left_on="truth_cell_type_id",
    right_on="cell_type_id",
    how="left",
)

summary_path = data_root / "native_zero_shot_comparison_summary.csv"
per_type_path = data_root / "native_zero_shot_comparison_per_type.csv"
summary_table.to_csv(summary_path, index=False)
per_type_table.to_csv(per_type_path, index=False)

open_primary_methods = {
    ("HECTOR", "native_open_reference"),
    ("OnClass", "native_released_ensemble_open"),
    ("scCello", "native_checkpoint_open"),
}
open_primary_table = summary_table.loc[
    summary_table.apply(
        lambda row: (row["model"], row["method"]) in open_primary_methods,
        axis=1,
    )
].copy()
open_primary_path = data_root / "native_open_vocabulary_primary_summary.csv"
open_primary_table.to_csv(open_primary_path, index=False)

print(summary_table.to_string(index=False))
print(f"Saved: {summary_path}")
print(f"Saved: {per_type_path}")
print(f"Saved: {open_primary_path}")
