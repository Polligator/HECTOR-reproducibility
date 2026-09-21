"""# Audit native zero-shot eligibility

This analysis compares the frozen 66 HECTOR-held-out labels with the exact
supervised label manifests and native ontology vocabularies shipped by scCello
and the six OnClass checkpoints used in the paper's pretrained ensemble.
"""

from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd

from native_comparison_utils import checkpoint_ontology_ids, checkpoint_seen_ids


# ## 1. Resolve frozen inputs
# The benchmark label table is the authoritative list of the 66 query types.
figure_root = Path(__file__).resolve().parents[2]
candidate_path = figure_root / "result/data/candidate_label_table.csv"
sccello_root = figure_root / "models/scCello/scCello-repo"
onclass_root = (
    figure_root / "models/OnClass/pretrained/OnClass_data_public/Pretrained_model"
)
output_root = figure_root / "result/data"
output_root.mkdir(parents=True, exist_ok=True)

candidate_table = pd.read_csv(candidate_path)
candidate_table["cell_type_id"] = candidate_table["cell_type_id"].astype(str)


# ## 2. Read the exact scCello training manifest
# The CL-ID-to-index pickle fixes the 398 supervised cell types and their order.
sccello_manifest_path = (
    sccello_root / "data/new_pretrain/pretrain_frac100_clid2name.pkl"
)
with sccello_manifest_path.open("rb") as handle:
    sccello_id_to_index = pickle.load(handle)
sccello_seen_ids = {str(cell_type_id) for cell_type_id in sccello_id_to_index}

sccello_owl_path = sccello_root / "data/cell_taxonomy/cl.owl"
sccello_owl_text = sccello_owl_path.read_text(errors="ignore")
sccello_ontology_ids = {
    value.replace("_", ":") for value in re.findall(r"CL_\d+", sccello_owl_text)
}


# ## 3. Read the six OnClass paper checkpoints
# Allen Brain and the human lung model are not part of the six-dataset ensemble.
onclass_checkpoint_names = [
    "OnClass_full_muris_facs",
    "OnClass_full_muris_droplet",
    "OnClass_full_microcebusAntoine",
    "OnClass_full_microcebusBernard",
    "OnClass_full_microcebusMartine",
    "OnClass_full_microcebusStumpy",
]
onclass_npz_paths = [
    onclass_root / f"{checkpoint_name}.npz"
    for checkpoint_name in onclass_checkpoint_names
]

onclass_seen_by_checkpoint = {
    checkpoint_path.stem: checkpoint_seen_ids(checkpoint_path)
    for checkpoint_path in onclass_npz_paths
}
onclass_seen_union = set().union(*onclass_seen_by_checkpoint.values())
onclass_native_vocabularies = [
    checkpoint_ontology_ids(checkpoint_path)
    for checkpoint_path in onclass_npz_paths
]
onclass_common_ontology_ids = set.intersection(*onclass_native_vocabularies)


# ## 4. Annotate every held-out label
# Separate flags prevent "held out from HECTOR" from being mistaken for a
# method-independent zero-shot definition.
audit_table = candidate_table.copy()
audit_table["seen_by_sccello"] = audit_table["cell_type_id"].isin(sccello_seen_ids)
audit_table["in_sccello_native_ontology"] = audit_table["cell_type_id"].isin(
    sccello_ontology_ids
)
audit_table["seen_by_any_onclass_checkpoint"] = audit_table["cell_type_id"].isin(
    onclass_seen_union
)
audit_table["in_all_six_onclass_ontologies"] = audit_table["cell_type_id"].isin(
    onclass_common_ontology_ids
)

for checkpoint_name, seen_ids in onclass_seen_by_checkpoint.items():
    column_name = f"seen_by_{checkpoint_name.removeprefix('OnClass_full_')}"
    audit_table[column_name] = audit_table["cell_type_id"].isin(seen_ids)

audit_table["strict_common_zero_shot"] = (
    ~audit_table["seen_by_sccello"]
    & ~audit_table["seen_by_any_onclass_checkpoint"]
    & audit_table["in_sccello_native_ontology"]
    & audit_table["in_all_six_onclass_ontologies"]
)


# ## 5. Save and summarize the audit
# Counts are written as a separate compact table for manuscript and plotting code.
audit_output_path = output_root / "training_and_ontology_audit.csv"
audit_table.to_csv(audit_output_path, index=False)

summary_rows = [
    ("HECTOR-held-out types", len(audit_table)),
    ("seen by scCello", int(audit_table["seen_by_sccello"].sum())),
    (
        "seen by any OnClass paper checkpoint",
        int(audit_table["seen_by_any_onclass_checkpoint"].sum()),
    ),
    (
        "missing from scCello native ontology",
        int((~audit_table["in_sccello_native_ontology"]).sum()),
    ),
    (
        "missing from at least one OnClass native ontology",
        int((~audit_table["in_all_six_onclass_ontologies"]).sum()),
    ),
    ("strict common-zero-shot types", int(audit_table["strict_common_zero_shot"].sum())),
]
audit_summary = pd.DataFrame(summary_rows, columns=["criterion", "n_cell_types"])
audit_summary.to_csv(output_root / "training_and_ontology_audit_summary.csv", index=False)

print(audit_summary.to_string(index=False))
print(f"\nSaved: {audit_output_path}")
