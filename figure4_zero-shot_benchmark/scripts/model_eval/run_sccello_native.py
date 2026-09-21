"""# Run native checkpoint-only scCello zero-shot prediction

This analysis re-encodes the frozen query cells with scCello's trained
transformed-CLS representation, compares them with the 398 prototypes stored
inside the checkpoint, and applies the paper's Spearman/PPR decoder without
using query labels to construct the candidate vocabulary.
"""

from pathlib import Path
import importlib.util
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from transformers.activations import ACT2FN

from native_comparison_utils import (
    build_query_metadata_from_frozen_predictions,
    compute_candidate_to_bridge_ppr_sparse,
    cosine_similarity_matrix,
    load_sccello_known_prototypes,
    load_sccello_representation_head,
    predict_from_spearman_profiles,
)


# ## 1. Resolve frozen benchmark and official scCello resources
# Absolute paths prevent the working directory from changing the inputs.
figure_root = Path(__file__).resolve().parents[2]
model_eval_root = figure_root / "scripts/model_eval"
sccello_repo = figure_root / "models/scCello/scCello-repo"
checkpoint_root = figure_root / "models/scCello/checkpoints/scCello-zeroshot"
query_path = figure_root / "input/Zero_shot_query_set.h5ad"
current_ontology_path = figure_root / "input/cl.obo"
native_ontology_path = sccello_repo / "data/cell_taxonomy/cl.owl"
candidate_path = figure_root / "result/data/candidate_label_table.csv"
audit_path = figure_root / "result/data/training_and_ontology_audit.csv"
frozen_prediction_path = (
    figure_root / "result/data/hector_native_closed_set_predictions.csv"
)
output_root = figure_root / "result/data"
output_root.mkdir(parents=True, exist_ok=True)
representation_path = output_root / "sccello_official_query_representations.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32 if device == "cuda" else 8

if str(model_eval_root) not in sys.path:
    sys.path.insert(0, str(model_eval_root))


# ## 2. Load the existing, verified tokenization implementation
# The benchmark module already reproduces scCello's official rank-tokenizer and
# deterministic 10x/primary-cell filter; only its pooling rule is replaced here.
embed_spec = importlib.util.spec_from_file_location(
    "figure4_embed_sccello",
    model_eval_root / "embed_sccello.py",
)
embed_module = importlib.util.module_from_spec(embed_spec)
sys.modules[embed_spec.name] = embed_module
embed_spec.loader.exec_module(embed_module)

ontology_spec = importlib.util.spec_from_file_location(
    "figure4_cell_ontology_ppr",
    model_eval_root / "cell_ontology_ppr.py",
)
ontology_module = importlib.util.module_from_spec(ontology_spec)
sys.modules[ontology_spec.name] = ontology_module
sys.modules["cell_ontology_ppr"] = ontology_module
ontology_spec.loader.exec_module(ontology_module)

evaluate_spec = importlib.util.spec_from_file_location(
    "figure4_evaluate",
    model_eval_root / "evaluate.py",
)
evaluate_module = importlib.util.module_from_spec(evaluate_spec)
sys.modules[evaluate_spec.name] = evaluate_module
evaluate_spec.loader.exec_module(evaluate_module)


# ## 3. Tokenize the complete query set
# Rows rejected by scCello's own input contract remain absent here and are
# counted as errors when the full 33,000-cell score is summarized.
query_data, sampled_obs_indices = embed_module.load_and_subsample(
    str(query_path),
    n_cells=None,
)
query_row_ids = embed_module._get_row_ids(query_data)
kept_row_indices = embed_module._get_sccello_kept_row_indices(query_data)

input_ids, attention_masks, cell_types, cell_type_ids, number_filtered = (
    embed_module.tokenize_with_sccello(query_data, max_length=2048)
)

if len(kept_row_indices) != len(cell_types):
    raise RuntimeError(
        "The deterministic scCello filter and tokenizer returned different row counts: "
        f"{len(kept_row_indices)} versus {len(cell_types)}."
    )

print(f"Device: {device}")
print(f"Query cells: {query_data.n_obs:,}")
print(f"Tokenized cells: {len(cell_types):,}")
print(f"Filtered cells: {number_filtered:,}")
expected_row_ids = [str(query_row_ids[index]) for index in kept_row_indices]


# ## 4. Generate the representation used by scCello's training objectives
# The published model takes the final-layer CLS token through ``cell_cls``,
# not a mean pool over token embeddings.
if representation_path.exists():
    with representation_path.open("rb") as handle:
        representation_payload = pickle.load(handle)
    if representation_payload["row_ids"] != expected_row_ids:
        raise RuntimeError("Cached scCello representations have a different row order.")
    cell_representations = representation_payload["embeddings"]
    print(f"Loaded cached representations: {representation_path}")
else:
    encoder, config_dict = embed_module.load_sccello_model(
        str(checkpoint_root),
        device=device,
    )
    config = encoder.config
    activation = ACT2FN[str(config.hidden_act)]
    representation_head = load_sccello_representation_head(
        checkpoint_root / "model.safetensors",
        config,
        activation,
        device,
    )

    cell_representations = []
    with torch.no_grad():
        for start in range(0, len(input_ids), batch_size):
            stop = min(start + batch_size, len(input_ids))
            token_batch = torch.tensor(input_ids[start:stop], device=device)
            mask_batch = torch.tensor(attention_masks[start:stop], device=device)
            hidden_states = encoder(
                input_ids=token_batch,
                attention_mask=mask_batch,
            )
            transformed_cls = representation_head(hidden_states[:, :1, :])[:, 0, :]
            cell_representations.append(transformed_cls.cpu().numpy())

            if start == 0 or stop == len(input_ids) or (start // batch_size) % 50 == 0:
                print(f"Encoded {stop:,}/{len(input_ids):,} cells")

    cell_representations = np.vstack(cell_representations).astype(np.float32)
    representation_payload = {
        "embeddings": cell_representations,
        "cell_types": list(cell_types),
        "cell_type_ids": list(cell_type_ids),
        "row_ids": expected_row_ids,
        "kept_row_indices": kept_row_indices,
        "source": query_path.name,
        "model": "scCello-zeroshot",
        "representation": "trained cell_cls transformation of final-layer CLS token",
    }
    with representation_path.open("wb") as handle:
        pickle.dump(representation_payload, handle)
    print(f"Saved representation cache: {representation_path}")


# ## 5. Recover the 398 known-type prototypes and their CL-ID order
# The order is fixed by the released pretraining manifest, not reconstructed
# from names or from the held-out query labels.
known_manifest_path = (
    sccello_repo / "data/new_pretrain/pretrain_frac100_clid2name.pkl"
)
with known_manifest_path.open("rb") as handle:
    known_id_to_index = pickle.load(handle)

known_ids = [None] * len(known_id_to_index)
for cell_type_id, index in known_id_to_index.items():
    known_ids[int(index)] = str(cell_type_id)

if any(cell_type_id is None for cell_type_id in known_ids):
    raise RuntimeError("scCello's known-type manifest does not define every index.")

known_prototypes = load_sccello_known_prototypes(
    checkpoint_root / "model.safetensors",
    number_known_types=len(known_ids),
)
query_to_known = cosine_similarity_matrix(
    cell_representations,
    known_prototypes,
)


# ## 6. Build native ontology profiles
# Closed prediction uses the prespecified 66 labels but only terms represented
# in scCello's own ontology can receive a score. Open prediction uses every
# native term that also remains valid in the frozen evaluation ontology.
native_ontology = ontology_module.CellOntologyPPR(native_ontology_path)
native_ontology.load_ontology()
current_ontology = ontology_module.CellOntologyPPR(current_ontology_path)
current_ontology.load_ontology()

candidate_table = pd.read_csv(candidate_path)
all_closed_candidate_ids = candidate_table["cell_type_id"].astype(str).tolist()
native_closed_candidate_ids = [
    cell_type_id
    for cell_type_id in all_closed_candidate_ids
    if cell_type_id in native_ontology.cell_types_onto
]
audit_table = pd.read_csv(audit_path)
strict_common_candidate_ids = audit_table.loc[
    audit_table["strict_common_zero_shot"].astype(bool), "cell_type_id"
].astype(str).tolist()
native_open_candidate_ids = sorted(
    set(native_ontology.cell_types_onto) & set(current_ontology.cell_types_onto)
)

closed_candidate_profiles = compute_candidate_to_bridge_ppr_sparse(
    native_ontology,
    native_closed_candidate_ids,
    known_ids,
    alpha=0.9,
)
open_candidate_profiles = compute_candidate_to_bridge_ppr_sparse(
    native_ontology,
    native_open_candidate_ids,
    known_ids,
    alpha=0.9,
)
closed_candidate_to_index = {
    cell_type_id: index
    for index, cell_type_id in enumerate(native_closed_candidate_ids)
}
strict_candidate_profiles = closed_candidate_profiles[
    [closed_candidate_to_index[cell_type_id] for cell_type_id in strict_common_candidate_ids]
]

print(f"Closed candidates represented natively: {len(native_closed_candidate_ids)}/66")
print(f"Open candidates represented in both ontologies: {len(native_open_candidate_ids):,}")


# ## 7. Make closed and open native predictions
# Spearman correlation is vectorized but mathematically identical to the
# per-cell loop in the released novel-cell-type script.
closed_predicted_ids, closed_scores = predict_from_spearman_profiles(
    query_to_known,
    closed_candidate_profiles,
    native_closed_candidate_ids,
)
open_predicted_ids, open_scores = predict_from_spearman_profiles(
    query_to_known,
    open_candidate_profiles,
    native_open_candidate_ids,
)
strict_predicted_ids, strict_scores = predict_from_spearman_profiles(
    query_to_known,
    strict_candidate_profiles,
    strict_common_candidate_ids,
)


# ## 8. Assemble Figure 4-compatible outputs
# The frozen HECTOR table supplies the authoritative row order and truth labels;
# scCello rows are selected only by the model's deterministic input filter.
full_query_metadata = build_query_metadata_from_frozen_predictions(
    frozen_prediction_path
)
query_metadata = full_query_metadata.iloc[kept_row_indices].reset_index(drop=True)

if query_metadata["row_id"].astype(str).tolist() != expected_row_ids:
    raise RuntimeError("scCello output rows do not align with the frozen query manifest.")

closed_table = evaluate_module.assemble_prediction_table(
    query_metadata,
    predicted_ids=closed_predicted_ids,
    scores=closed_scores,
    method="native_checkpoint_closed_set",
    model_name="scCello",
    ontology=current_ontology,
)
closed_table = evaluate_module.annotate_prediction_table(
    closed_table,
    ontology_path=current_ontology_path,
)

open_table = evaluate_module.assemble_prediction_table(
    query_metadata,
    predicted_ids=open_predicted_ids,
    scores=open_scores,
    method="native_checkpoint_open_set",
    model_name="scCello",
    ontology=current_ontology,
)
open_table = evaluate_module.annotate_prediction_table(
    open_table,
    ontology_path=current_ontology_path,
)

strict_table = evaluate_module.assemble_prediction_table(
    query_metadata,
    predicted_ids=strict_predicted_ids,
    scores=strict_scores,
    method="native_checkpoint_strict_common31_closed_set",
    model_name="scCello",
    ontology=current_ontology,
)
strict_table = evaluate_module.annotate_prediction_table(
    strict_table,
    ontology_path=current_ontology_path,
)


# ## 9. Save predictions and reproducibility metadata
# Intermediate representations are retained so decoder variants do not require
# another GPU pass through the model.
closed_path = output_root / "sccello_native_closed_set_predictions.csv"
open_path = output_root / "sccello_native_open_set_predictions.csv"
strict_path = output_root / "sccello_native_strict_common31_predictions.csv"
closed_table.to_csv(closed_path, index=False)
open_table.to_csv(open_path, index=False)
strict_table.to_csv(strict_path, index=False)

print(f"Saved: {closed_path}")
print(f"Saved: {open_path}")
print(f"Saved: {strict_path}")
print(f"Saved: {representation_path}")
