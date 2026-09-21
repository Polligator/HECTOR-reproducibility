from __future__ import annotations

"""Benchmark dataset configs + compute-if-missing inference, used by the figure
entry script run_figure2.py.

There is no separate benchmark runner: the prediction step (HECTOR in the
`hector` env + the POPV suite in the `popv` env) is triggered *by* the figure
script via ensure_inference(). Predictions land in a LOCAL, non-shipped cache (`.cache/`)
— the first figure run computes them; later runs reuse them; delete `.cache/` to
force a fresh recompute. Only the interpretable result tables + the figures (under
`result/`) are the kept deliverable.
"""

from pathlib import Path

from benchmark_modules import benchmark_core as core

# figure2_basic_benchmark/, holding input/, scripts/ and result/. Every path below
# is anchored to it, so a run does not depend on which folder it was started from.
FIGURE = Path(__file__).resolve().parents[2]

# Local prediction cache — NOT a shipped deliverable (safe to delete; recomputes).
CACHE_ROOT = FIGURE / ".cache"

POPV_CONDA_ENV = "popv"
HECTOR_CONDA_ENV = "hector"
POPV_FORCE_CPU = False

# (dataset_key, data_file in input/datasets/, POPV reference repo, HECTOR model)
_DATASETS = [
    ("kidney",            "human_kidney_normal_cells.h5ad", "popV/tabula_sapiens_Kidney",              "human"),
    ("lung",              "Human_BAL.h5ad",                 "popV/tabula_sapiens_Lung",                "human"),
    ("mouse_kidney",      "mouse_kidney_normal_cells.h5ad", "popV/tabula_muris_Kidney_10x",            "mouse"),
    ("mouse_aorta",       "mouse_VSMC.h5ad",                "popV/tabula_muris_Aorta",                 "mouse"),
    ("mouse_hippocampus", "mouse_hippocampus.h5ad",         "popV/tabula_muris_Brain_non-myeloid_cells", "mouse"),
    ("skin",              "HumanSkin_normal_cells.h5ad",    "popV/tabula_sapiens_Skin",                "human"),
]
_BY_KEY = {key: (key, data_file, popv_repo, model) for (key, data_file, popv_repo, model) in _DATASETS}

# The datasets that appear in the cross-tissue generalization supplement.
GENERALIZATION_KEYS = ["mouse_kidney", "lung", "skin", "mouse_aorta", "mouse_hippocampus"]

# Each dataset's supplementary figure number, the words it is called by in the
# manuscript, and the name its file carries. The numbers follow the order in which
# the supplements are first cited in the manuscript -- human first, then mouse, as
# the Results introduce them; GENERALIZATION_KEYS above is a rendering order and
# does not set numbering.
# The dataset keys are internal shorthand: "name" is what is printed on the figure
# and "file_stem" is what a reader matches to a citation, so neither repeats the key.
# Renumbering means changing this table and every citation in the manuscript
# together, or a citation will point at a file that no longer exists.
SUPPLEMENTS = {
    "kidney":            {"number": 2, "name": "Human kidney",
                          "file_stem": "supplementary_figure_2_human_kidney_detail"},
    "lung":              {"number": 3, "name": "Human bronchoalveolar lavage",
                          "file_stem": "supplementary_figure_3_human_lung"},
    "skin":              {"number": 4, "name": "Human skin",
                          "file_stem": "supplementary_figure_4_human_skin"},
    "mouse_kidney":      {"number": 5, "name": "Mouse kidney",
                          "file_stem": "supplementary_figure_5_mouse_kidney"},
    "mouse_aorta":       {"number": 6, "name": "Mouse aorta",
                          "file_stem": "supplementary_figure_6_mouse_aorta"},
    "mouse_hippocampus": {"number": 7, "name": "Mouse hippocampus",
                          "file_stem": "supplementary_figure_7_mouse_hippocampus"},
}


def display_name(dataset_key: str) -> str:
    """What the dataset is called in the manuscript, for anything a reader sees."""
    return SUPPLEMENTS[dataset_key]["name"]


def supplement_pdf_name(dataset_key: str) -> str:
    """File name for that dataset's supplement, carrying its number."""
    return SUPPLEMENTS[dataset_key]["file_stem"] + ".pdf"


def benchmark_config(dataset_key: str) -> dict:
    """Full benchmark config for a dataset, with results_dir pointing at the local
    prediction cache (.cache/<ds>/)."""
    key, data_file, popv_repo, hector_model = _BY_KEY[dataset_key]
    stem = Path(data_file).stem
    results_dir = CACHE_ROOT / key
    return {
        "dataset_key": key,
        "display_name": display_name(key),
        "figure_file_name": supplement_pdf_name(key),
        "data_path": str(FIGURE / "input" / "datasets" / data_file),
        "results_dir": str(results_dir),
        "cell_id_file": str(results_dir / "cell_ids.txt"),
        "label_key": core.LABEL_KEY,
        "batch_key": core.BATCH_KEY,
        "sample_size": core.SAMPLE_SIZE,
        "random_state": core.RANDOM_STATE,
        "harmonization_table": str(FIGURE / "input" / "harmonization" / f"{stem}.csv"),
        "popv": {"repo": popv_repo, "cache_dir": None,
                 "force_cpu": POPV_FORCE_CPU, "prediction_mode": "inference"},
        "hector": {"model_path": hector_model, "top_k": 3, "use_grit": True,
                   "use_asymmetric_ppr": False, "ppr_alpha": 0.25, "forward_weight": 0.6},
        "popv_conda_env": POPV_CONDA_ENV,
        "hector_conda_env": HECTOR_CONDA_ENV,
    }


def inference_present(dataset_key: str) -> bool:
    d = CACHE_ROOT / dataset_key
    return (d / "hector_metrics.csv").exists() and (d / "popv_metrics.csv").exists()


def ensure_inference(dataset_key: str) -> dict:
    """Return the dataset's benchmark config, first running HECTOR + POPV into the
    cache if its predictions aren't there yet (compute-if-missing)."""
    config = benchmark_config(dataset_key)
    if inference_present(dataset_key):
        print(f"  [{dataset_key}] using cached predictions ({CACHE_ROOT / dataset_key})")
    else:
        print(f"  [{dataset_key}] predictions not cached -> running inference (HECTOR + POPV)...")
        core.run_inference_for_dataset(config)
    return config
