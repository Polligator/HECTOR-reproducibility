# Build figure 3 — a single model annotating and integrating multi-tissue disease data.
#
#     conda run -n hector python run_figure3.py
#
# Generates each method's embedding, scores them with scIB, and builds figure 3
# and the integration supplement.
#
# Set `recompute = False` to redraw from the scores and UMAP coordinates already
# saved under result/. scIB scoring clusters with Leiden, which is not seed
# stable, so a full rerun shifts the printed scores slightly even with the
# model and data untouched.
#
# Outputs land in result/figure/: figure3.{pdf,png} and the integration
# benchmark supplement (scIB score table + per-method maps).

from pathlib import Path

from helpers.downloaded_dataset_prep_helpers import resolve_dataset_config
from helpers.single_dataset_benchmark_helpers import (
    DEFAULT_METHODS,
    build_benchmark_figure,
    build_figure3,
    run_evaluation_stage,
    run_integration_stage,
)


# ## 1. The dataset and its metadata columns
# Edit this block to point at a different dataset. `dataset_display_name = None`
# falls back to the known atlas label, otherwise to `dataset_name`.

project_dir = Path(__file__).resolve().parents[1]
dataset_name = "raw_object"  # human_pancreas_norm_complexbatch, immune_all_human, lung_atlas_public, combined_common_cells, combined_all_tissues
dataset_display_name = None
batch_obs_column = "Samples"  # immune_all_human: batch, lung_atlas_public: batch, human_pancreas_norm_complexbatch: tech, combined_common_cells: donor
label_obs_column = "maintypes_2"  # immune_all_human: final_annotation, lung_atlas_public: cell_type, human_pancreas_norm_complexbatch: celltype, combined_common_cells: harmonized_cell_type
tissue_obs_column = "Groups"
patient_obs_column = "Patients"
sample_obs_column = "Samples"
min_cells_per_batch = 50
enabled_methods = DEFAULT_METHODS


# ## 2. What to run
# `recompute` runs the benchmark before drawing; the four settings below apply
# only then, letting the two stages run separately or be forced to rebuild
# rather than reuse saved work. With `recompute = False`, the figure redraws
# from the saved score table and UMAP coordinates, so no number can move.

recompute = True

run_integration = True
run_evaluation = True
force_recompute_embeddings = False
force_recompute_evaluation = False

draw_supplement = True   # the supplement takes a few extra minutes on its own


# ## 3. Check the input
# Both paths read the processed atlas — the benchmark scores it, and the figures take
# their cell metadata from it. `prepare_downloaded_dataset.py` is what produces it.

processed_path = project_dir / "input" / "processed" / f"{dataset_name}.h5ad"
if not processed_path.exists():
    raise FileNotFoundError(
        f"Missing processed dataset: {processed_path}\n"
        "Run prepare_downloaded_dataset.py for this dataset first."
    )

dataset_config = resolve_dataset_config(dataset_name, display_name=dataset_display_name)
print(f"Benchmark dataset: {dataset_name} ({dataset_config['display_name']})")
print(
    {
        "mode": "recompute, then draw" if recompute else "draw from saved results",
        "dataset_display_name": dataset_config["display_name"],
        "batch_obs_column": batch_obs_column,
        "label_obs_column": label_obs_column,
        "min_cells_per_batch": min_cells_per_batch,
        "enabled_methods": enabled_methods,
    }
)
if recompute:
    print(
        "[figure3] Running the benchmark. The scIB scores printed on the figure will "
        "shift slightly from the saved ones, because Leiden clustering is not stable "
        "between runs. Set recompute = False to redraw without moving them."
    )


# ## 4. The benchmark, when asked for
# Stage one writes one saved embedding per method. Stage two reads them back, computes
# the scIB metric tables and the per-method UMAP coordinates, and draws both figures —
# so a recompute run needs nothing after it.

if recompute:
    if run_integration:
        integration_report = run_integration_stage(
            project_dir,
            dataset_name,
            dataset_display_name=dataset_display_name,
            batch_obs_column=batch_obs_column,
            label_obs_column=label_obs_column,
            min_cells_per_batch=min_cells_per_batch,
            enabled_methods=enabled_methods,
            force_recompute_embeddings=force_recompute_embeddings,
        )
        print(integration_report)

    if run_evaluation:
        evaluation_report = run_evaluation_stage(
            project_dir,
            dataset_name,
            dataset_display_name=dataset_display_name,
            batch_obs_column=batch_obs_column,
            label_obs_column=label_obs_column,
            tissue_obs_column=tissue_obs_column,
            patient_obs_column=patient_obs_column,
            sample_obs_column=sample_obs_column,
            min_cells_per_batch=min_cells_per_batch,
            enabled_methods=enabled_methods,
            force_recompute_evaluation=force_recompute_evaluation,
        )
        print(evaluation_report)


# ## 5. The figures, from what is already saved
# Only reached when not recomputing, since the evaluation stage above draws them itself.
# Both calls read the saved score table and the saved UMAP coordinates, so this path
# re-renders and nothing else.

else:
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
    print(f"wrote {figure3_path}")

    if draw_supplement:
        supplement_path = build_benchmark_figure(
            project_dir,
            dataset_name,
            dataset_display_name=dataset_display_name,
            batch_obs_column=batch_obs_column,
            label_obs_column=label_obs_column,
            min_cells_per_batch=min_cells_per_batch,
        )
        print(f"wrote {supplement_path}")
