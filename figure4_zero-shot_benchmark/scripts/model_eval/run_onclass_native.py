"""# Run native pretrained OnClass zero-shot prediction

This analysis applies the six pretrained checkpoints used by OnClass's
released ensemble to the frozen 33,000-cell query set. Each checkpoint scores
its complete native ontology, after which CL-ID-aligned scores are combined
with both the released transductive normalization and a cellwise sensitivity.
"""

from pathlib import Path
import importlib.util
import sys

import anndata as ad
import numpy as np
import pandas as pd

from native_comparison_utils import (
    align_sparse_expression_to_checkpoint,
    build_query_metadata_from_frozen_predictions,
    load_onclass_source,
    predict_onclass_batches,
    predictions_from_candidate_scores,
)


# ## 1. Resolve frozen benchmark and official OnClass resources
# The six checkpoint names follow the ensemble order in the released utility.
figure_root = Path(__file__).resolve().parents[2]
model_eval_root = figure_root / "scripts/model_eval"
query_path = figure_root / "input/Zero_shot_query_set.h5ad"
current_ontology_path = figure_root / "input/cl.obo"
candidate_path = figure_root / "result/data/candidate_label_table.csv"
audit_path = figure_root / "result/data/training_and_ontology_audit.csv"
frozen_prediction_path = (
    figure_root / "result/data/hector_native_closed_set_predictions.csv"
)

onclass_source_root = figure_root / "models/OnClass/repo"
checkpoint_root = (
    figure_root / "models/OnClass/pretrained/OnClass_data_public/Pretrained_model"
)
checkpoint_names = [
    "OnClass_full_muris_facs",
    "OnClass_full_muris_droplet",
    "OnClass_full_microcebusBernard",
    "OnClass_full_microcebusStumpy",
    "OnClass_full_microcebusMartine",
    "OnClass_full_microcebusAntoine",
]

output_root = figure_root / "result/data"
output_root.mkdir(parents=True, exist_ok=True)
batch_size = 256

if str(model_eval_root) not in sys.path:
    sys.path.insert(0, str(model_eval_root))


# ## 2. Load the frozen query matrix and evaluation ontology
# Cell labels are retained only for final output assembly, never supplied to a
# checkpoint, a candidate-building operation, or a score-normalization gate.
query_data = ad.read_h5ad(query_path)
query_expression = query_data.X.tocsr()
query_gene_names = query_data.var["feature_name"].astype(str).to_numpy()
query_metadata = build_query_metadata_from_frozen_predictions(
    frozen_prediction_path
)

if query_data.n_obs != len(query_metadata):
    raise RuntimeError("Query expression and frozen metadata have different rows.")

evaluate_spec = importlib.util.spec_from_file_location(
    "figure4_evaluate_onclass",
    model_eval_root / "evaluate.py",
)
evaluate_module = importlib.util.module_from_spec(evaluate_spec)
sys.modules[evaluate_spec.name] = evaluate_module
evaluate_spec.loader.exec_module(evaluate_module)

ontology_module = sys.modules["cell_ontology_ppr"]
current_ontology = ontology_module.CellOntologyPPR(current_ontology_path)
current_ontology.load_ontology()


# ## 3. Establish a CL-ID reference shared by all checkpoints
# Seen labels are first in every checkpoint, so native column order differs.
# Alignment by ontology ID is required before any ensemble calculation.
checkpoint_payloads = []
checkpoint_ontology_sets = []
for checkpoint_name in checkpoint_names:
    checkpoint_stem = checkpoint_root / checkpoint_name
    payload = np.load(str(checkpoint_stem) + ".npz", allow_pickle=True)
    checkpoint_payloads.append(payload)
    checkpoint_ontology_sets.append(set(payload["co2i"].item()))

common_onclass_ids = set.intersection(*checkpoint_ontology_sets)
reference_ids = sorted(common_onclass_ids)
reference_to_index = {
    cell_type_id: index for index, cell_type_id in enumerate(reference_ids)
}

candidate_table = pd.read_csv(candidate_path)
all_closed_candidate_ids = candidate_table["cell_type_id"].astype(str).tolist()
native_closed_candidate_ids = [
    cell_type_id
    for cell_type_id in all_closed_candidate_ids
    if cell_type_id in common_onclass_ids
]
native_open_candidate_ids = sorted(
    common_onclass_ids & set(current_ontology.cell_types_onto)
)
audit_table = pd.read_csv(audit_path)
strict_common_candidate_ids = audit_table.loc[
    audit_table["strict_common_zero_shot"].astype(bool), "cell_type_id"
].astype(str).tolist()

print(f"Query cells: {query_data.n_obs:,}")
print(f"Closed candidates represented in all checkpoints: {len(native_closed_candidate_ids)}/66")
print(f"Open candidates valid in the frozen ontology: {len(native_open_candidate_ids):,}")


# ## 4. Score each checkpoint and align its native ontology columns
# OnClass's released propagation uses sixteen RWR networks. The composition
# gate is disabled because its required unseen fraction is unknown at inference.
OnClassModel, create_networks, extend_prediction = load_onclass_source(
    onclass_source_root
)

ensemble_released = np.zeros(
    (query_data.n_obs, len(reference_ids)), dtype=np.float32
)
ensemble_cellwise = np.zeros_like(ensemble_released)
checkpoint_qc_rows = []

for checkpoint_name, payload in zip(checkpoint_names, checkpoint_payloads):
    checkpoint_stem = checkpoint_root / checkpoint_name
    checkpoint_genes = payload["genes"].astype(str)
    model = OnClassModel.__new__(OnClassModel)
    model.BuildModel(ngene=None, use_pretrain=str(checkpoint_stem))

    aligned_expression = align_sparse_expression_to_checkpoint(
        query_expression,
        query_gene_names,
        checkpoint_genes,
    )
    propagation_networks = create_networks(
        model.co2i,
        model.ontology_dict,
        model.ontology_mat,
        model.co2vec_nlp_mat,
    )
    model_scores = predict_onclass_batches(
        model,
        aligned_expression,
        propagation_networks,
        extend_prediction,
        batch_size=batch_size,
    )

    model_ids = [str(model.i2co[index]) for index in range(int(model.nco))]
    model_indices = np.asarray([model.co2i[cell_type_id] for cell_type_id in reference_ids])
    aligned_scores = model_scores[:, model_indices]

    # The released ensemble normalizes each class across all query cells.
    released_scores = aligned_scores / (
        aligned_scores.sum(axis=0, keepdims=True) + 1.0
    )
    ensemble_released += released_scores.astype(np.float32)

    # This alternative never couples one query cell's score to another cell.
    cellwise_scale = aligned_scores.sum(axis=1, keepdims=True)
    cellwise_scores = aligned_scores / np.maximum(cellwise_scale, 1e-12)
    ensemble_cellwise += cellwise_scores.astype(np.float32)

    checkpoint_qc_rows.append(
        {
            "checkpoint": checkpoint_name,
            "number_seen_types": int(model.nseen),
            "number_ontology_types": int(model.nco),
            "checkpoint_genes": len(checkpoint_genes),
            "matched_query_genes": int(aligned_expression.getnnz(axis=0).astype(bool).sum()),
        }
    )
    print(f"Finished {checkpoint_name}")

    model.model.sess.close()
    del model_scores, aligned_scores, released_scores, cellwise_scores
    del propagation_networks, aligned_expression, model


# ## 5. Convert ensemble scores to closed- and open-vocabulary predictions
# Missing native ontology terms receive no synthetic score; affected truths
# therefore remain errors under the original 66-type denominator.
prediction_variants = {
    "released_ensemble": ensemble_released,
    "cellwise_ensemble": ensemble_cellwise,
}

saved_paths = []
for ensemble_name, ensemble_scores in prediction_variants.items():
    for vocabulary_name, candidate_ids in [
        ("closed_set", native_closed_candidate_ids),
        ("open_set", native_open_candidate_ids),
        ("strict_common31", strict_common_candidate_ids),
    ]:
        predicted_ids, top_scores = predictions_from_candidate_scores(
            ensemble_scores,
            reference_ids,
            candidate_ids,
        )
        prediction_table = evaluate_module.assemble_prediction_table(
            query_metadata,
            predicted_ids=predicted_ids,
            scores=top_scores,
            method=f"native_checkpoint_{ensemble_name}_{vocabulary_name}",
            model_name="OnClass",
            ontology=current_ontology,
        )
        prediction_table = evaluate_module.annotate_prediction_table(
            prediction_table,
            ontology_path=current_ontology_path,
        )

        output_path = output_root / (
            f"onclass_native_{ensemble_name}_{vocabulary_name}_predictions.csv"
        )
        prediction_table.to_csv(output_path, index=False)
        saved_paths.append(output_path)


# ## 6. Save checkpoint-level reproducibility diagnostics
# Gene overlap is reported per checkpoint because the lemma and mouse models
# use different input spaces.
checkpoint_qc = pd.DataFrame(checkpoint_qc_rows)
checkpoint_qc_path = output_root / "onclass_checkpoint_qc.csv"
checkpoint_qc.to_csv(checkpoint_qc_path, index=False)

for saved_path in saved_paths:
    print(f"Saved: {saved_path}")
print(f"Saved: {checkpoint_qc_path}")
