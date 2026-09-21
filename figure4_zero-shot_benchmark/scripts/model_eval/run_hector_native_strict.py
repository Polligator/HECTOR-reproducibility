"""# Run HECTOR on the identical strict common-zero-shot candidate set

This analysis applies HECTOR's native predictor to the frozen query set and
restricts its final decision to the same 31 audited candidate labels used for
the strict OnClass–scCello comparison.
"""

from pathlib import Path
import sys

import anndata as ad
import pandas as pd

from native_comparison_utils import build_query_metadata_from_frozen_predictions


# ## 1. Resolve frozen Figure 4 resources
# The strict candidate set comes from training manifests and ontology coverage,
# never from the HECTOR prediction output.
figure_root = Path(__file__).resolve().parents[2]
model_eval_root = figure_root / "scripts/model_eval"
query_path = figure_root / "input/Zero_shot_query_set.h5ad"
ontology_path = figure_root / "input/cl.obo"
frozen_prediction_path = (
    figure_root / "result/data/hector_native_closed_set_predictions.csv"
)
result_root = figure_root / "result/data"
audit_path = result_root / "training_and_ontology_audit.csv"

if str(model_eval_root) not in sys.path:
    sys.path.insert(0, str(model_eval_root))

import evaluate as evaluate_module
from cell_ontology_ppr import CellOntologyPPR
import hector


# ## 2. Load the strict candidate definition and complete query metadata
# HECTOR sees every query cell for inference; the evaluator retains only strict
# truth rows after prediction to preserve the same 31-label benchmark stratum.
audit = pd.read_csv(audit_path)
audit["cell_type_id"] = audit["cell_type_id"].astype(str)
strict_candidate_ids = audit.loc[
    audit["strict_common_zero_shot"].astype(bool), "cell_type_id"
].tolist()
strict_expected_counts = audit.loc[
    audit["strict_common_zero_shot"].astype(bool)
].set_index("cell_type_id")["n_cells"]
query_metadata = build_query_metadata_from_frozen_predictions(
    frozen_prediction_path
)
query_adata = ad.read_h5ad(query_path)

if query_adata.n_obs != len(query_metadata):
    raise RuntimeError("HECTOR query matrix and frozen metadata have different rows.")

ontology = CellOntologyPPR(ontology_path)
ontology.load_ontology()


# ## 3. Run the frozen HECTOR native predictor and restrict its candidate axis
# Exporting the native score matrix allows the final argmax to use exactly the
# same 31 candidates as OnClass and scCello.
hector_predictor = hector.HECTOR("human", verbose=True)
native_result = evaluate_module.run_hector_native_predictions(
    query_adata,
    query_metadata,
    hector_predictor=hector_predictor,
    candidate_ids=strict_candidate_ids,
    ontology=ontology,
    ontology_path=ontology_path,
    expected_query_counts=strict_expected_counts,
    strict_unseen=True,
    model_name="HECTOR",
    candidate_name_map={
        cell_type_id: ontology.get_name(cell_type_id)
        for cell_type_id in strict_candidate_ids
    },
)

if native_result["missing_candidate_ids"]:
    raise RuntimeError(
        "HECTOR is missing strict candidate IDs: "
        + ", ".join(native_result["missing_candidate_ids"])
    )


# ## 4. Save the aligned strict closed-set table
# This table contains 15,500 rows: 500 cells for each audited strict label.
strict_table = native_result["closed_table"]
strict_summary = native_result["closed_summary"]
strict_path = result_root / "hector_native_strict_common31_predictions.csv"
strict_table.to_csv(strict_path, index=False)

summary_path = result_root / "hector_native_strict_common31_summary.csv"
pd.DataFrame(
    [{
        "model": "HECTOR",
        "method": "native_strict_common31",
        "candidate_count": len(strict_candidate_ids),
        "accuracy": strict_summary["accuracy"],
        "macro_f1": strict_summary["macro_f1"],
        "weighted_f1": strict_summary["weighted_f1"],
        "macro_hierarchical_f1": strict_summary["hierarchical_summary"].loc[
            "macro", "hierarchical_f1"
        ],
    }]
).to_csv(summary_path, index=False)

print(f"Saved: {strict_path}")
print(f"Saved: {summary_path}")
