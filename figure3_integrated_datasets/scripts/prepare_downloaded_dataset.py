# Prepare A Downloaded Dataset
# This analysis adds Ensembl gene IDs to one downloaded atlas object when they are missing and
# writes the processed `.h5ad` plus the gene-mapping reports used for benchmarking.

from pathlib import Path

from helpers.downloaded_dataset_prep_helpers import prepare_downloaded_dataset, resolve_dataset_config


# ## 1. Choose the downloaded atlas to prepare
# Edit this short config block for routine use. For the three atlas datasets already used in this
# workspace, leaving `source_filename=None` will pick the known file automatically. For any new
# dataset, the default input is `data/<dataset_name>.h5ad`, unless you set `source_filename`
# explicitly.

project_dir = Path(__file__).resolve().parents[1]
dataset_name = "Immune_ALL_human" #Immune_ALL_human, #Lung_atlas_public
source_filename = None
dataset_display_name = None
force_rebuild = False


# ## 2. Validate the requested dataset
# This step is open to any `.h5ad` dataset as long as the source file exists under `data/`.

dataset_config = resolve_dataset_config(
    dataset_name,
    source_filename=source_filename,
    display_name=dataset_display_name,
)
source_path = project_dir / "input" / dataset_config["source_filename"]
if not source_path.exists():
    raise FileNotFoundError(
        f"Missing source dataset: {source_path}\n"
        "Set `source_filename` explicitly if the input file name does not match the dataset name."
    )

print(f"Preparing dataset: {dataset_name} ({dataset_config['display_name']})")


# ## 3. Run the Ensembl-ID preparation step
# The prep step only standardizes the feature axis. It does not choose batch or label columns;
# those are configured later in the benchmark runner.

prepared_adata, prep_report = prepare_downloaded_dataset(
    project_dir,
    dataset_name,
    source_filename=source_filename,
    display_name=dataset_display_name,
    force_rebuild=force_rebuild,
)

print(
    {
        "dataset_name": prep_report["dataset_name"],
        "n_obs": prep_report["n_obs"],
        "n_vars": prep_report["n_vars"],
    }
)
print(prep_report["mapping_summary"])
