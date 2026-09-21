"""
# Figure 5 — fetal retina trajectory and RNA–chromatin validation

This notebook-style analysis is the single source for main Figure 5.  With
``REFRESH_EXPENSIVE_RESULTS = False`` it verifies and reads frozen caches; changing the
constant to ``True`` rebuilds HECTOR, the matched flow fields, all five RNA–ATAC edge
programmes, and the Signac pileups before drawing panels a–e.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

# Scanpy/Numba and Matplotlib otherwise try to cache under read-only desktop locations.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/figure5_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/figure5_matplotlib_cache")

import anndata
import fitz
import h5py
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from pynndescent import NNDescent
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde, rankdata

# ## 1. Paths and the one execution switch
# All paths resolve from this file.  The default deliberately avoids fragment-file and R I/O.

REFRESH_EXPENSIVE_RESULTS = False

FIGURE5_DIRECTORY = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = FIGURE5_DIRECTORY / "input"
ATAC_DIRECTORY = INPUT_DIRECTORY / "atac"
FRAGMENT_DIRECTORY = ATAC_DIRECTORY / "fragments"
MULTIOME_DIRECTORY = ATAC_DIRECTORY / "multiome"
MULTIVELO_DIRECTORY = INPUT_DIRECTORY / "multivelo"
RESULT_DIRECTORY = FIGURE5_DIRECTORY / "result"
CACHE_DIRECTORY = RESULT_DIRECTORY / "cache"
PDF_DIRECTORY = RESULT_DIRECTORY / "pdf"
QC_DIRECTORY = RESULT_DIRECTORY / "qc"

ATLAS_PATH = INPUT_DIRECTORY / "HumanFetalRetina.h5ad"
REFGENE_PATH = ATAC_DIRECTORY / "refGene_hg38.txt.gz"
INPUT_MANIFEST_PATH = INPUT_DIRECTORY / "figure5_input_manifest.csv"
CACHE_MANIFEST_PATH = CACHE_DIRECTORY / "cache_manifest.json"
CELL_CACHE_PATH = CACHE_DIRECTORY / "hector_cells.parquet"
TREE_PDF_PATH = CACHE_DIRECTORY / "hector_radial_tree.pdf"
NEIGHBOR_CACHE_PATH = CACHE_DIRECTORY / "hector_neighbors_30.npz"
RNA_CURVE_PATH = CACHE_DIRECTORY / "rna_curves.csv"
CELL_GROUP_PATH = CACHE_DIRECTORY / "atac_cell_groups.csv"
GROUP_DEPTH_PATH = CACHE_DIRECTORY / "atac_group_depth.csv"
SIGNAC_PROFILE_PATH = CACHE_DIRECTORY / "signac_profiles.csv.gz"
SIGNAC_EFFECT_PATH = CACHE_DIRECTORY / "signac_effects.csv"
SIGNAC_SUMMARY_PATH = CACHE_DIRECTORY / "signac_summary.csv"
PARITY_REPORT_PATH = QC_DIRECTORY / "figure5_parity_report.json"
FINAL_PDF_PATH = PDF_DIRECTORY / "figure5.pdf"
FINAL_PNG_PATH = PDF_DIRECTORY / "figure5.png"

# The two external programs the refresh calls. Both are looked up on PATH, and
# both can be pointed somewhere else without editing this file, by exporting
# FIGURE5_TABIX or FIGURE5_RSCRIPT. The original run used the tabix shipped in a
# conda environment and the system Rscript at /usr/bin/Rscript.
TABIX_PATH = Path(os.environ.get("FIGURE5_TABIX") or shutil.which("tabix") or "tabix")
RSCRIPT_PATH = Path(os.environ.get("FIGURE5_RSCRIPT") or shutil.which("Rscript") or "/usr/bin/Rscript")

MIN_CELLS_NUMBER = 50
FALSE_DISCOVERY_RATE = 0.01
MINIMUM_CELLS_PER_GENE = 25
NUMBER_OF_SHUFFLES = 200
RANDOM_SEED = 20260822
GENES_PER_DIRECTION = 100
PROMOTER_HALF_WIDTH = 2_000
GENOMIC_BIN_SIZE = 50
NUMBER_OF_RNA_BINS = 12
NUMBER_OF_EDGE_GROUPS = 4
MINIMUM_ATAC_DEPTH = 1_000
MINIMUM_ELIGIBLE_CELLS = 80
CONTROL_SHORTLIST_WIDTH = 10
OPENNESS_FEATURE_WEIGHT = 2.0

ANALYSIS_SETTINGS = {
    "min_cells_number": MIN_CELLS_NUMBER,
    "false_discovery_rate": FALSE_DISCOVERY_RATE,
    "minimum_cells_per_gene": MINIMUM_CELLS_PER_GENE,
    "number_of_shuffles": NUMBER_OF_SHUFFLES,
    "random_seed": RANDOM_SEED,
    "genes_per_direction": GENES_PER_DIRECTION,
    "promoter_half_width": PROMOTER_HALF_WIDTH,
    "genomic_bin_size": GENOMIC_BIN_SIZE,
    "number_of_edge_groups": NUMBER_OF_EDGE_GROUPS,
    "minimum_atac_depth": MINIMUM_ATAC_DEPTH,
    "minimum_eligible_cells": MINIMUM_ELIGIBLE_CELLS,
    "control_shortlist_width": CONTROL_SHORTLIST_WIDTH,
    "openness_feature_weight": OPENNESS_FEATURE_WEIGHT,
    "signac_controls": "2026-08-25 openness-matched",
    "hector_environment": "hector",
    "hector_umap_backend": "cuml",
}

for directory in [CACHE_DIRECTORY, PDF_DIRECTORY, QC_DIRECTORY]:
    directory.mkdir(parents=True, exist_ok=True)


# ## 2. Frozen biological design and display vocabulary
# These choices were fixed before the ATAC outcome and are kept visibly in the analysis source.

EDGE_DESIGN = pd.DataFrame(
    [
        ("RGC", "RGC Precursor", "retinal ganglion cell", "midget ganglion cell of retina", "Donor_4", "Multi_Fetal_13W_FR"),
        ("Amacrine", "AC Precursor", "amacrine cell", "GABAergic amacrine cell", "Donor_6", "Multi_Fetal_14w5d_FR"),
        ("Horizontal", "HC Precursor", "retina horizontal cell", "H1 horizontal cell", "Donor_2", "Multi_Fetal_11w2d_FR;Multi_Fetal_11w2d_FR_2"),
        ("PRPC", "PRPC", "multi fate stem cell", "retinal progenitor cell", "Donor_2", "Multi_Fetal_11w2d_NR"),
        ("NRPC", "NRPC", "multi fate stem cell", "retinal progenitor cell", "Donor_2", "Multi_Fetal_11w2d_FR_2"),
    ],
    columns=["edge_key", "author_subclass", "trajectory_source", "trajectory_target", "discovery_donor", "discovery_libraries"],
)
EDGE_DESIGN["edge_name"] = EDGE_DESIGN["trajectory_source"] + " to " + EDGE_DESIGN["trajectory_target"] + " edge"

FRAGMENT_LIBRARIES = {
    "Donor_3": "Multiome_12w3d_FR",
    "Donor_4": "Multi_Fetal_13W_FR",
    "Donor_5": "Multiome_14w2d_FR",
    "Donor_6": "Multi_Fetal_14w5d_FR",
    "Donor_8": "Multi_Fetal_19W4d_FR",
    "Donor_10": "Multi_Fetal_20W2d_FR",
}

HECTOR_LINEAGES = [
    "amacrine cell",
    "retina horizontal cell",
    "retinal rod cell + retinal cone cell",
    "retinal bipolar neuron",
]
LINEAGE_ORDER = [
    "precursor cell",
    "camera-type eye photoreceptor cell",
    "retinal ganglion cell",
    "amacrine cell",
    "retinal bipolar neuron",
    "retina horizontal cell",
    "neuron associated cell",
]
CLASS_TO_DISPLAY = {
    "Cone": "photoreceptors", "Rod": "photoreceptors",
    "PRPC": "retinal progenitor cells", "NRPC": "retinal progenitor cells",
    "RGC": "retinal ganglion cells", "AC": "amacrine cells",
    "BC": "bipolar cells", "HC": "horizontal cells", "MG": "Müller glial cells",
}
CLASS_PALETTE = {
    "photoreceptors": "#E6853F", "retinal progenitor cells": "#9A9A9A",
    "retinal ganglion cells": "#5BA867", "amacrine cells": "#5081BC",
    "bipolar cells": "#3E9E9A", "horizontal cells": "#8C6FB0",
    "Müller glial cells": "#C25E5F",
}
FLOW_LINEAGE_PALETTE = {
    "precursor cell": CLASS_PALETTE["retinal progenitor cells"],
    "camera-type eye photoreceptor cell": CLASS_PALETTE["photoreceptors"],
    "retinal ganglion cell": CLASS_PALETTE["retinal ganglion cells"],
    "amacrine cell": CLASS_PALETTE["amacrine cells"],
    "retinal bipolar neuron": CLASS_PALETTE["bipolar cells"],
    "retina horizontal cell": CLASS_PALETTE["horizontal cells"],
    "neuron associated cell": CLASS_PALETTE["Müller glial cells"],
}


# ## 3. Small, explicit scientific helpers
# They live here—rather than in hidden project modules—so this file remains independently runnable.

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def settings_fingerprint() -> str:
    encoded = json.dumps(ANALYSIS_SETTINGS, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_week(value) -> float:
    match = re.search(r"(\d+)", str(value))
    return float(match.group(1)) if match else np.nan


def load_refgene_tss(path: Path) -> pd.DataFrame:
    accepted = {f"chr{value}" for value in list(range(1, 23)) + ["X"]}
    longest = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) < 13 or fields[2] not in accepted:
                continue
            symbol, chromosome, strand = fields[12], fields[2], fields[3]
            start, end = int(fields[4]), int(fields[5])
            if symbol not in longest or end - start > longest[symbol][1] - longest[symbol][0]:
                longest[symbol] = (start, end, chromosome, strand)
    rows = []
    for symbol, (start, end, chromosome, strand) in longest.items():
        rows.append((symbol, chromosome, start if strand == "+" else end, strand, end - start))
    return pd.DataFrame(rows, columns=["symbol", "chrom", "tss", "strand", "gene_length"])


def merged_regions(promoters: pd.DataFrame) -> pd.DataFrame:
    table = promoters[["chrom", "tss"]].dropna().copy()
    table["start"] = (table["tss"] - PROMOTER_HALF_WIDTH).clip(lower=0).astype(int)
    table["end"] = (table["tss"] + PROMOTER_HALF_WIDTH).astype(int)
    table = table.sort_values(["chrom", "start", "end"])
    merged = []
    for chromosome, block in table.groupby("chrom", sort=False):
        current_start = current_end = None
        for start, end in block[["start", "end"]].itertuples(index=False):
            if current_start is None:
                current_start, current_end = int(start), int(end)
            elif start <= current_end:
                current_end = max(current_end, int(end))
            else:
                merged.append((chromosome, current_start, current_end))
                current_start, current_end = int(start), int(end)
        if current_start is not None:
            merged.append((chromosome, current_start, current_end))
    return pd.DataFrame(merged, columns=["chrom", "start", "end"])


def load_atac_depth(matrix_path: Path) -> pd.DataFrame:
    multiome = sc.read_10x_h5(matrix_path, gex_only=False)
    multiome.var_names_make_unique()
    atac_mask = multiome.var["feature_types"].astype(str).eq("Peaks").to_numpy()
    atac_total = np.asarray(multiome.X[:, atac_mask].sum(axis=1)).ravel()
    return pd.DataFrame({"barcode": multiome.obs_names.to_numpy(), "atac_total": atac_total})


def assign_edge_groups(depth_table, edge_cells, library_name):
    depths = depth_table.set_index("barcode")["atac_total"]
    cells = edge_cells.copy()
    cells["barcode"] = cells["cell_id"].str.replace(f"^{library_name}_", "", regex=True)
    cells["atac_total"] = cells["barcode"].map(depths)
    cells = cells[cells["atac_total"].ge(MINIMUM_ATAC_DEPTH)].copy()
    if len(cells) < MINIMUM_ELIGIBLE_CELLS:
        return None, None, None
    cells["edge_group"] = pd.qcut(cells["relative_position"].rank(method="first"), NUMBER_OF_EDGE_GROUPS, labels=False)
    summary = cells.groupby("edge_group").agg(n_cells=("cell_id", "size"), group_depth=("atac_total", "sum")).reset_index()
    if summary["n_cells"].min() < MINIMUM_ELIGIBLE_CELLS // NUMBER_OF_EDGE_GROUPS:
        return None, None, None
    depths = summary.set_index("edge_group")["group_depth"].reindex(range(NUMBER_OF_EDGE_GROUPS)).to_numpy(float)
    return cells.reset_index(drop=True), depths, summary


def count_promoter_insertions(fragment_path, bed_path, cells, promoters):
    usable = promoters.dropna(subset=["chrom", "tss", "strand"]).reset_index(drop=True)
    number_of_bins = 2 * PROMOTER_HALF_WIDTH // GENOMIC_BIN_SIZE
    counts = np.zeros((NUMBER_OF_EDGE_GROUPS, len(usable), number_of_bins), dtype=np.float64)
    lookup = {}
    for chromosome, block in usable.groupby("chrom", sort=False):
        order = np.argsort(block["tss"].to_numpy(np.int64))
        lookup[chromosome] = (
            block["tss"].to_numpy(np.int64)[order],
            block.index.to_numpy(np.int64)[order],
            np.where(block["strand"].to_numpy(str)[order] == "-", -1, 1),
        )
    barcode_groups = {barcode.encode(): int(group) for barcode, group in cells[["barcode", "edge_group"]].itertuples(index=False)}
    process = subprocess.Popen([str(TABIX_PATH), "-R", str(bed_path), str(fragment_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    retained = 0
    for line in process.stdout:
        fields = line.rstrip().split(b"\t")
        group = barcode_groups.get(fields[3])
        chromosome = fields[0].decode()
        if group is None or chromosome not in lookup:
            continue
        tss, gene_indices, strand_signs = lookup[chromosome]
        start, end = int(fields[1]), int(fields[2])
        multiplicity = int(fields[4]) if len(fields) > 4 else 1
        retained += multiplicity
        for insertion in (start, end - 1):
            left = np.searchsorted(tss, insertion - PROMOTER_HALF_WIDTH, side="right")
            right = np.searchsorted(tss, insertion + PROMOTER_HALF_WIDTH, side="left")
            offsets = (insertion - tss[left:right]) * strand_signs[left:right]
            bins = (offsets + PROMOTER_HALF_WIDTH) // GENOMIC_BIN_SIZE
            np.add.at(counts[group], (gene_indices[left:right], bins), multiplicity)
    error = process.stderr.read().decode()
    if process.wait() != 0:
        raise RuntimeError(error)
    return usable, counts, retained


def latent_neighbors(latent_space):
    search = NNDescent(latent_space, n_neighbors=31, metric="cosine", n_jobs=24, random_state=0, low_memory=True)
    raw = np.asarray(search.neighbor_graph[0], dtype=np.int64)
    cleaned = np.empty((len(raw), 30), dtype=np.int64)
    for cell_index, row in enumerate(raw):
        cleaned[cell_index] = row[row != cell_index][:30]
    return cleaned


def hector_field(xy, lineages, ranks, neighbors):
    ranked = np.isin(lineages, LINEAGE_ORDER) & np.isfinite(ranks)
    safe_ranks = np.where(np.isfinite(ranks), ranks, 0)
    delta = safe_ranks[neighbors] - safe_ranks[:, None]
    scale = np.median(np.abs(delta))
    weights = np.tanh(delta / scale) if scale > 0 else np.sign(delta)
    arrows = np.zeros((len(xy), 2), dtype=np.float32)
    for start in range(0, len(xy), 20_000):
        stop = min(start + 20_000, len(xy))
        neighbor_block = neighbors[start:stop]
        steps = xy[neighbor_block] - xy[start:stop, None]
        lengths = np.linalg.norm(steps, axis=2, keepdims=True)
        unit_steps = np.divide(steps, lengths, out=np.zeros_like(steps), where=lengths > 0)
        same = (lineages[start:stop, None] == lineages[neighbor_block]) & ranked[start:stop, None]
        arrows[start:stop] = (np.where(same[:, :, None], weights[start:stop, :, None] * unit_steps, 0).sum(axis=1) / np.maximum(same.sum(axis=1), 1)[:, None])
    return arrows


def grid_field(xy, arrows, minimum_cells=5, x_edges=None, y_edges=None):
    if x_edges is None:
        x_edges = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 51)
    if y_edges is None:
        y_edges = np.linspace(xy[:, 1].min(), xy[:, 1].max(), 51)
    counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])
    total_u, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges], weights=arrows[:, 0])
    total_v, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges], weights=arrows[:, 1])
    blurred = gaussian_filter(counts, 1.0)
    field_u = gaussian_filter(total_u, 1.0) / np.maximum(blurred, 1e-9)
    field_v = gaussian_filter(total_v, 1.0) / np.maximum(blurred, 1e-9)
    thin = (blurred < minimum_cells) | (counts == 0)
    return x_edges[:-1] + np.diff(x_edges) / 2, y_edges[:-1] + np.diff(y_edges) / 2, np.where(thin, np.nan, field_u).T, np.where(thin, np.nan, field_v).T


def draw_flow_axis(axis, cells, field, title):
    xy = cells[["UMAP_1", "UMAP_2"]].to_numpy(float)
    lineages = cells["hector_lineage"].astype(object).to_numpy()
    ranks = pd.to_numeric(cells["hector_pseudotime"], errors="coerce").to_numpy()
    ranked = np.isin(lineages, LINEAGE_ORDER) & np.isfinite(ranks)
    colours = np.tile(mcolors.to_rgba("#C6C6C6"), (len(cells), 1))
    for lineage in LINEAGE_ORDER:
        mask = ranked & (lineages == lineage)
        colour = np.asarray(mcolors.to_rgb(FLOW_LINEAGE_PALETTE[lineage]))
        colours[mask, :3] = 1 - (1 - colour) * (0.55 + 0.45 * ranks[mask])[:, None]
    # The white background joins the same rasterised layer as the points, so the exported image is
    # fully opaque and carries no soft mask for Illustrator to flatten.  The streamlines below stay
    # vector because they are drawn after it.
    axis.set_facecolor("white"); axis.patch.set_rasterized(True)
    axis.scatter(xy[~ranked, 0], xy[~ranked, 1], s=0.25, c=colours[~ranked], linewidth=0, rasterized=True)
    order = np.flatnonzero(ranked)[np.argsort(ranks[ranked])]
    axis.scatter(xy[order, 0], xy[order, 1], s=0.25, c=colours[order], linewidth=0, rasterized=True)
    fields = list(field.values()) if isinstance(field, dict) else [field]
    for x, y, u, v in fields:
        speed = np.hypot(u, v)
        finite = speed[np.isfinite(speed)]
        if len(finite) == 0:
            continue
        width = 0.25 + 1.35 * np.clip(np.nan_to_num(speed) / max(np.nanpercentile(finite, 98), 1e-8), 0, 1)
        axis.streamplot(x, y, u, v, density=1.65, color="black", linewidth=width, arrowsize=0.8, minlength=0.08, maxlength=3)
    axis.set_title(title, fontsize=PLOT_TITLE_FONT_SIZE, fontweight="bold", pad=2)
    axis.set_aspect("equal")
    axis.set_xticks([]); axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


# ## 4. Input and cache validation
# Cached mode hashes only the compact manifest, not the 24 GB fragments; full refresh rechecks all files.

required_inputs = [ATLAS_PATH, REFGENE_PATH, ATAC_DIRECTORY / "library_manifest.csv"]
required_inputs += sorted(FRAGMENT_DIRECTORY.glob("*.tsv.gz")) + sorted(FRAGMENT_DIRECTORY.glob("*.tbi"))
required_inputs += sorted(MULTIOME_DIRECTORY.glob("*.h5")) + sorted(MULTIVELO_DIRECTORY.glob("*_cells.csv.gz"))
missing_inputs = [str(path) for path in required_inputs if not path.exists()]
if missing_inputs:
    raise FileNotFoundError("Figure 5 inputs are incomplete:\n" + "\n".join(missing_inputs))
if len(list(FRAGMENT_DIRECTORY.glob("*.tsv.gz"))) != 6 or len(list(FRAGMENT_DIRECTORY.glob("*.tbi"))) != 6:
    raise ValueError("Expected six ATAC fragments and six tabix indexes.")
if len(list(MULTIOME_DIRECTORY.glob("*.h5"))) != 8:
    raise ValueError("Expected eight multiome matrices.")
if len(list(MULTIVELO_DIRECTORY.glob("*_cells.csv.gz"))) != 8:
    raise ValueError("Expected eight compact MultiVelo tables.")

input_manifest = pd.read_csv(INPUT_MANIFEST_PATH)
if input_manifest["relative_path"].duplicated().any():
    raise ValueError("Input manifest contains duplicate paths.")
if REFRESH_EXPENSIVE_RESULTS:
    for manifest_row in input_manifest.itertuples(index=False):
        manifest_file = INPUT_DIRECTORY / manifest_row.relative_path
        if manifest_file.stat().st_size != int(manifest_row.size_bytes):
            raise ValueError(f"Input size differs from the manifest: {manifest_row.relative_path}")
        if sha256_file(manifest_file) != manifest_row.sha256:
            raise ValueError(f"Input checksum differs from the manifest: {manifest_row.relative_path}")

compact_manifest_hash = hashlib.sha256(INPUT_MANIFEST_PATH.read_bytes()).hexdigest()
current_settings_fingerprint = settings_fingerprint()


# ## 5. Refresh or load HECTOR cells
# The pseudotime step is independent of the supplementary five-tool benchmark.

if REFRESH_EXPENSIVE_RESULTS:
    import hector

    if Path(sys.executable).parent.parent.name != "hector":
        raise RuntimeError(
            "Full refresh must run with the Python of the conda environment named "
            "'hector' (environments/hector.yml); "
            f"the active interpreter is {sys.executable}."
        )
    if importlib.util.find_spec("cuml") is None:
        raise RuntimeError("Full refresh requires cuML in the HECTOR environment.")
    import tensorflow as tf
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Full refresh requires a visible NVIDIA GPU; CPU fallback is not accepted.")

    atlas = sc.read_h5ad(ATLAS_PATH)
    if atlas.n_obs != 226_506 or atlas.obs_names.duplicated().any():
        raise ValueError(f"Unexpected atlas cell index: {atlas.n_obs:,} rows.")
    predictor = hector.HECTOR("human")
    hector.create_trajectory_analysis(
        predictor=predictor, adata=atlas, output_file=str(TREE_PDF_PATH), title="",
        use_grit=True, layout_mode="radial", label_format="name",
        radial_sweep_angle_degrees=180, label_font_size=8,
        show_cloud_cell_counts=False, min_cells_number=MIN_CELLS_NUMBER,
        show_transitions=True, enable_atypical_detection=False,
        atypical_clusters=False, atypical_scatter=False,
        enable_inferred_paths=False, enable_elastic_layout=False,
    )
    predictor.reduce_dimensions(atlas, backend="cuml")
    hector.create_pseudotime_analysis(predictor=predictor, adata=atlas, lineages=HECTOR_LINEAGES)
    cells = atlas.obs.copy()
    cells["cell_id"] = atlas.obs_names
    cells["UMAP_1"] = atlas.obsm["X_umap"][:, 0]
    cells["UMAP_2"] = atlas.obsm["X_umap"][:, 1]
    if "lib" not in cells and "library" in cells:
        cells["lib"] = cells["library"]
    if "lib" not in cells:
        cells["lib"] = cells["cell_id"].str.rsplit("_", n=1).str[0]
    keep_columns = [
        "cell_id", "majorclass", "subclass", "development_stage", "donor_id", "lib", "tissue",
        "hector_prediction", "hector_prediction_confidence",
        "trajectory_state", "trajectory_source", "trajectory_target", "relative_position", "absolute_position",
        "hector_lineage", "hector_pseudotime", "UMAP_1", "UMAP_2",
    ]
    cells[[column for column in keep_columns if column in cells]].to_parquet(CELL_CACHE_PATH, index=False)
    x_hector = np.ascontiguousarray(atlas.obsm["X_hector"], dtype=np.float32)
    (QC_DIRECTORY / "hector_embedding_fingerprint.json").write_text(json.dumps({
        "python": sys.executable,
        "backend": "cuml",
        "tensorflow_gpus": [str(device) for device in tf.config.list_physical_devices("GPU")],
        "x_hector_shape": list(x_hector.shape),
        "x_hector_sha256": hashlib.sha256(x_hector.view(np.uint8)).hexdigest(),
        "umap_sha256": hashlib.sha256(np.ascontiguousarray(atlas.obsm["X_umap"], dtype=np.float32).view(np.uint8)).hexdigest(),
    }, indent=2))
else:
    required_caches = [CELL_CACHE_PATH, TREE_PDF_PATH, RNA_CURVE_PATH, CELL_GROUP_PATH, GROUP_DEPTH_PATH, SIGNAC_PROFILE_PATH, SIGNAC_EFFECT_PATH, SIGNAC_SUMMARY_PATH, CACHE_MANIFEST_PATH]
    missing_caches = [str(path) for path in required_caches if not path.exists()]
    if missing_caches:
        raise FileNotFoundError("Cached mode requires:\n" + "\n".join(missing_caches))
    cache_manifest = json.loads(CACHE_MANIFEST_PATH.read_text())
    if cache_manifest.get("input_manifest_sha256") != compact_manifest_hash:
        raise RuntimeError("Input manifest changed; set REFRESH_EXPENSIVE_RESULTS = True.")
    if cache_manifest.get("settings_sha256") != current_settings_fingerprint:
        raise RuntimeError("Analysis settings changed; set REFRESH_EXPENSIVE_RESULTS = True.")
    if cache_manifest.get("signac_controls") != "2026-08-25 openness-matched":
        raise RuntimeError("The Signac cache predates openness-matched controls.")
    if cache_manifest.get("schema_version") != 2:
        raise RuntimeError("The Figure 5 cache predates the consolidated two-script workflow.")
    if cache_manifest.get("hector_cell_cache_sha256") != sha256_file(CELL_CACHE_PATH):
        raise RuntimeError("The HECTOR cell cache fingerprint differs from its manifest.")
    cells = pd.read_parquet(CELL_CACHE_PATH)
    required_cell_cache_columns = {
        "cell_id", "hector_prediction", "hector_prediction_confidence",
        "hector_lineage", "hector_pseudotime", "UMAP_1", "UMAP_2",
    }
    missing_cell_cache_columns = required_cell_cache_columns.difference(cells.columns)
    if missing_cell_cache_columns:
        raise ValueError(
            "The HECTOR cell cache predates the consolidated supplementary workflow: "
            f"{sorted(missing_cell_cache_columns)}"
        )

if len(cells) != 226_506 or cells["cell_id"].duplicated().any():
    raise ValueError("The HECTOR cell cache must contain 226,506 unique cells.")
cells = cells.set_index("cell_id", drop=False)
if "lib" not in cells:
    cells["lib"] = cells["cell_id"].str.rsplit("_", n=1).str[0]


# ## 6. Refresh or load the five RNA–ATAC programmes
# Discovery uses RNA only.  Every local ATAC donor is retained or excluded by pre-outcome depth rules.

if REFRESH_EXPENSIVE_RESULTS:
    import hector

    coordinates = load_refgene_tss(REFGENE_PATH)
    depth_by_library = {}
    for library in FRAGMENT_LIBRARIES.values():
        matrix_path = MULTIOME_DIRECTORY / f"GSE268630_{library}_filtered_feature_bc_matrix.h5"
        depth_by_library[library] = load_atac_depth(matrix_path)

    rna_rows = []
    profile_rows = []
    effect_rows = []
    cell_group_rows = []
    group_depth_rows = []
    bed_workspace = tempfile.TemporaryDirectory(prefix="figure5_atac_beds_")
    bed_directory = Path(bed_workspace.name)

    for edge in EDGE_DESIGN.itertuples(index=False):
        discovery_libraries = edge.discovery_libraries.split(";")
        discovery_mask = cells["lib"].isin(discovery_libraries)
        discovery_mask &= cells["subclass"].eq(edge.author_subclass)
        edge_mask = discovery_mask & cells["trajectory_source"].eq(edge.trajectory_source)
        edge_mask &= cells["trajectory_target"].eq(edge.trajectory_target)
        edge_mask &= cells["relative_position"].notna()
        edge_cells = cells.loc[edge_mask].copy().sort_values("relative_position")
        if edge_cells.empty:
            raise ValueError(f"No discovery cells for {edge.edge_key}.")
        edge_cells["edge_order_percentile"] = (rankdata(edge_cells["relative_position"]) - 1) / max(len(edge_cells) - 1, 1)

        cell_indices = atlas.obs_names.get_indexer(edge_cells.index)
        if np.any(cell_indices < 0):
            raise ValueError(f"Atlas alignment failed for {edge.edge_key}.")
        expression = atlas[cell_indices].raw.to_adata()
        expression.var["symbol"] = expression.var["feature_name"].astype(str)
        expression.obs["hector_lineage"] = edge.edge_name
        expression.obs["hector_pseudotime"] = edge_cells["edge_order_percentile"].to_numpy()
        expression.obs["hector_detected_genes"] = np.asarray((expression.X > 0).sum(axis=1)).ravel().astype(float)

        gene_trends = hector.create_gene_trend_analysis(
            expression, lineages=[edge.edge_name], false_discovery_rate=FALSE_DISCOVERY_RATE,
            min_cells_per_gene=MINIMUM_CELLS_PER_GENE, n_shuffles=NUMBER_OF_SHUFFLES,
            hold_detected_genes_fixed=True, keep_all=True, n_jobs=None, seed=RANDOM_SEED,
        )
        gene_trends = gene_trends.drop_duplicates("symbol").copy()

        library_size = np.maximum(np.asarray(expression.X.sum(axis=1)).ravel().astype(float), 1)
        normalized = expression.X.multiply(1 / library_size[:, None]).tocsc()
        coordinate_rank = rankdata(edge_cells["edge_order_percentile"].to_numpy(float))
        centred_coordinate = coordinate_rank - coordinate_rank.mean()
        coordinate_norm = np.sqrt((centred_coordinate ** 2).sum())
        trend_correlation = np.zeros(expression.n_vars)
        for start in range(0, expression.n_vars, 2_000):
            stop = min(start + 2_000, expression.n_vars)
            ranked_expression = rankdata(np.asarray(normalized[:, start:stop].todense()), axis=0)
            ranked_expression -= ranked_expression.mean(axis=0)
            norms = np.sqrt((ranked_expression ** 2).sum(axis=0))
            trend_correlation[start:stop] = np.where(
                norms > 0,
                (ranked_expression * centred_coordinate[:, None]).sum(axis=0) / np.maximum(norms * coordinate_norm, 1e-12),
                0,
            )
        rank_table = pd.DataFrame({"symbol": expression.var["symbol"].astype(str).to_numpy(), "trend_correlation": trend_correlation}).drop_duplicates("symbol")
        genes = gene_trends.merge(rank_table, on="symbol", how="inner")
        genes = genes.merge(coordinates, on="symbol", how="inner")
        changing = genes[genes["q_value"].lt(FALSE_DISCOVERY_RATE)].sort_values("trend_correlation").copy()
        falling = changing.head(GENES_PER_DIRECTION)["symbol"]
        rising = changing.tail(GENES_PER_DIRECTION)["symbol"]
        genes["edge_pattern"] = "NOT_SIGNIFICANT"
        genes.loc[genes["symbol"].isin(falling), "edge_pattern"] = "EARLY"
        genes.loc[genes["symbol"].isin(rising), "edge_pattern"] = "LATE"
        genes.to_parquet(CACHE_DIRECTORY / f"{edge.edge_key.lower()}_gene_trends.parquet", index=False)

        genes["detection_fraction"] = genes["n_cells"] / len(edge_cells)
        targets = genes[genes["edge_pattern"].isin(["EARLY", "LATE"])].copy()
        candidates = genes[genes["q_value"].gt(0.5) & ~genes["symbol"].isin(targets["symbol"])].copy().reset_index(drop=True)
        target_features = np.column_stack([np.log10(targets["gene_length"]), targets["detection_fraction"]])
        candidate_features = np.column_stack([np.log10(candidates["gene_length"]), candidates["detection_fraction"]])
        scale = np.maximum(np.nanstd(np.vstack([target_features, candidate_features]), axis=0), 1e-6)
        distances = (((target_features[:, None] - candidate_features[None, :]) / scale) ** 2).sum(axis=2)
        width = min(CONTROL_SHORTLIST_WIDTH, distances.shape[1])
        nearest = np.argpartition(distances, width - 1, axis=1)[:, :width]
        shortlist = candidates.iloc[np.unique(nearest)].copy().reset_index(drop=True)
        candidate_pool = pd.concat([targets, shortlist], ignore_index=True)

        all_edge_mask = cells["subclass"].eq(edge.author_subclass)
        all_edge_mask &= cells["trajectory_source"].eq(edge.trajectory_source)
        all_edge_mask &= cells["trajectory_target"].eq(edge.trajectory_target)
        all_edge_mask &= cells["relative_position"].notna()
        all_edge_cells = cells.loc[all_edge_mask].copy()

        candidate_bed = bed_directory / f"{edge.edge_key.lower()}_candidate_regions.bed"
        merged_regions(candidate_pool).to_csv(candidate_bed, sep="\t", header=False, index=False)
        q1_by_donor = []
        edge_group_cache = {}
        for donor_id, library in FRAGMENT_LIBRARIES.items():
            library_cells = all_edge_cells[all_edge_cells["lib"].eq(library)].copy()
            assigned, group_depths, group_summary = assign_edge_groups(depth_by_library[library], library_cells, library)
            if assigned is None:
                continue
            edge_group_cache[donor_id] = (library, assigned, group_depths, group_summary)
            fragment_matches = sorted(FRAGMENT_DIRECTORY.glob(f"*_{library}_atac_fragments.tsv.gz"))
            usable, counts, _ = count_promoter_insertions(fragment_matches[0], candidate_bed, assigned, candidate_pool)
            q1_by_donor.append(pd.Series(counts[0].sum(axis=1) / group_depths[0] * 1e6, index=usable["symbol"]))
        mean_q1 = pd.concat(q1_by_donor, axis=1).mean(axis=1).rename("mean_q1_cpm")
        targets = targets.merge(mean_q1, left_on="symbol", right_index=True, how="inner")
        shortlist = shortlist.merge(mean_q1, left_on="symbol", right_index=True, how="inner")
        if len(targets) != 2 * GENES_PER_DIRECTION:
            raise ValueError(f"{edge.edge_key} lost target promoters during openness measurement.")

        def control_features(table):
            return np.column_stack([np.log10(table["gene_length"]), table["detection_fraction"], np.log1p(table["mean_q1_cpm"])])

        target_values = control_features(targets)
        candidate_values = control_features(shortlist)
        scale = np.maximum(np.nanstd(np.vstack([target_values, candidate_values]), axis=0), 1e-6)
        squared = ((target_values[:, None] - candidate_values[None, :]) / scale) ** 2
        assignment_cost = (squared * np.array([1.0, 1.0, OPENNESS_FEATURE_WEIGHT])).sum(axis=2)
        target_indices, candidate_indices = linear_sum_assignment(assignment_cost)
        controls = shortlist.iloc[candidate_indices].copy().reset_index(drop=True)
        controls["edge_pattern"] = "CTRL"
        controls["matched_target"] = targets.iloc[target_indices]["symbol"].to_numpy()
        promoters = pd.concat([targets, controls], ignore_index=True)
        promoters.to_csv(CACHE_DIRECTORY / f"{edge.edge_key.lower()}_promoters.csv", index=False)

        bin_index = pd.qcut(pd.Series(edge_cells["edge_order_percentile"].to_numpy()).rank(method="first"), NUMBER_OF_RNA_BINS, labels=False)
        symbols = pd.Index(expression.var["symbol"].astype(str))
        for set_name in ["EARLY", "LATE"]:
            for symbol in promoters.loc[promoters["edge_pattern"].eq(set_name), "symbol"]:
                columns = np.flatnonzero(symbols == symbol)
                counts = np.asarray(expression.X[:, columns].sum(axis=1)).ravel()
                values = np.log1p(counts / library_size * 10_000)
                curve = np.array([values[bin_index == b].mean() for b in range(NUMBER_OF_RNA_BINS)])
                span = curve.max() - curve.min()
                if span <= 0:
                    continue
                scaled_curve = (curve - curve.min()) / span
                for b in range(NUMBER_OF_RNA_BINS):
                    rna_rows.append({"edge_key": edge.edge_key, "set": set_name, "symbol": symbol, "bin": b, "edge_position": edge_cells["edge_order_percentile"].to_numpy()[bin_index == b].mean(), "scaled_expression": scaled_curve[b]})

        query_bed = bed_directory / f"{edge.edge_key.lower()}_regions.bed"
        merged_regions(promoters).to_csv(query_bed, sep="\t", header=False, index=False)
        for donor_id, (library, assigned, group_depths, group_summary) in edge_group_cache.items():
            cell_group_rows.append(pd.DataFrame({"edge_key": edge.edge_key, "donor_id": donor_id, "library": library, "barcode": assigned["barcode"], "group": "Q" + (assigned["edge_group"] + 1).astype(str), "relative_position": assigned["relative_position"]}))
            for group, depth in enumerate(group_depths):
                group_depth_rows.append({"edge_key": edge.edge_key, "donor_id": donor_id, "group": f"Q{group + 1}", "group_depth": depth, "group_cells": int((assigned["edge_group"] == group).sum())})
            fragment_path = sorted(FRAGMENT_DIRECTORY.glob(f"*_{library}_atac_fragments.tsv.gz"))[0]
            usable, counts, _ = count_promoter_insertions(fragment_path, query_bed, assigned, promoters)
            coverage = counts / group_depths[:, None, None] * 1e6
            offsets = np.linspace(-PROMOTER_HALF_WIDTH, PROMOTER_HALF_WIDTH, coverage.shape[2])
            totals = {}
            for set_name in ["EARLY", "LATE", "CTRL"]:
                mask = usable["edge_pattern"].eq(set_name).to_numpy()
                totals[set_name] = coverage[:, mask].sum(axis=2).mean(axis=1)
                for group in range(NUMBER_OF_EDGE_GROUPS):
                    profile_rows.append(pd.DataFrame({"edge_key": edge.edge_key, "donor_id": donor_id, "set": set_name, "group": f"Q{group + 1}", "offset_from_tss": offsets, "mean_cpm_per_gene": coverage[group, mask].mean(axis=0)}))
            for set_name in ["EARLY", "LATE"]:
                effect_rows.append({"edge_key": edge.edge_key, "donor_id": donor_id, "set": set_name, "target_minus_control": float(np.log2(totals[set_name][-1] / totals[set_name][0]) - np.log2(totals["CTRL"][-1] / totals["CTRL"][0]))})

    bed_workspace.cleanup()
    pd.DataFrame(rna_rows).to_csv(RNA_CURVE_PATH, index=False)
    pd.concat(cell_group_rows, ignore_index=True).to_csv(CELL_GROUP_PATH, index=False)
    pd.DataFrame(group_depth_rows).to_csv(GROUP_DEPTH_PATH, index=False)
    pd.concat(profile_rows, ignore_index=True).to_csv(CACHE_DIRECTORY / "tabix_profiles.csv.gz", index=False, compression="gzip")
    pd.DataFrame(effect_rows).to_csv(CACHE_DIRECTORY / "tabix_effects.csv", index=False)
else:
    rna_curves = pd.read_csv(RNA_CURVE_PATH)

rna_curves = pd.read_csv(RNA_CURVE_PATH)


# ## 7. Signac pileup through its public Footprint API
# The R source is temporary at runtime; the durable, editable source remains this Python file.

SIGNAC_PROGRAM = r'''
suppressPackageStartupMessages({
  library(Signac); library(Seurat); library(GenomicRanges); library(BSgenome.Hsapiens.UCSC.hg38)
})
args <- commandArgs(trailingOnly=TRUE)
cache_directory <- args[[1]]
fragment_directory <- args[[2]]
fragment_libraries <- c(Donor_3="Multiome_12w3d_FR", Donor_4="Multi_Fetal_13W_FR",
  Donor_5="Multiome_14w2d_FR", Donor_6="Multi_Fetal_14w5d_FR",
  Donor_8="Multi_Fetal_19W4d_FR", Donor_10="Multi_Fetal_20W2d_FR")
cell_groups <- read.csv(file.path(cache_directory, "atac_cell_groups.csv"), stringsAsFactors=FALSE)
group_depth <- read.csv(file.path(cache_directory, "atac_group_depth.csv"), stringsAsFactors=FALSE)
profile_rows <- list(); effect_rows <- list(); promoter_half_width <- 2000
for (edge_key in unique(cell_groups$edge_key)) {
  promoters <- read.csv(file.path(cache_directory, paste0(tolower(edge_key), "_promoters.csv")), stringsAsFactors=FALSE)
  promoters <- promoters[!duplicated(promoters$symbol),]
  tss_by_set <- lapply(c("EARLY","LATE","CTRL"), function(set_name) {
    block <- promoters[promoters$edge_pattern == set_name,]
    GRanges(seqnames=block$chrom, ranges=IRanges(start=block$tss, width=1), strand=block$strand)
  }); names(tss_by_set) <- c("EARLY","LATE","CTRL")
  for (donor_id in names(fragment_libraries)) {
    donor_cells <- cell_groups[cell_groups$edge_key == edge_key & cell_groups$donor_id == donor_id,]
    if (nrow(donor_cells) == 0) next
    library_name <- fragment_libraries[[donor_id]]
    fragment_path <- Sys.glob(file.path(fragment_directory, paste0("*_", library_name, "_atac_fragments.tsv.gz")))[1]
    dummy <- Matrix::Matrix(1, nrow=2, ncol=nrow(donor_cells), sparse=TRUE,
      dimnames=list(c("chr1-1-2","chr1-3-4"), donor_cells$barcode))
    fragment_object <- CreateFragmentObject(path=fragment_path, cells=donor_cells$barcode, validate.fragments=FALSE)
    assay <- CreateChromatinAssay(counts=dummy, fragments=fragment_object, min.cells=0, min.features=0)
    object <- CreateSeuratObject(assay, assay="ATAC")
    donor_group <- setNames(donor_cells$group, donor_cells$barcode)
    edge_depth <- group_depth[group_depth$edge_key == edge_key & group_depth$donor_id == donor_id,]
    depth <- setNames(edge_depth$group_depth, edge_depth$group)
    totals <- list()
    for (set_name in c("EARLY","LATE","CTRL")) {
      fp <- Footprint(object[["ATAC"]], genome=BSgenome.Hsapiens.UCSC.hg38,
        regions=tss_by_set[[set_name]], key="fp", upstream=promoter_half_width,
        downstream=promoter_half_width, compute.expected=FALSE, in.peaks=FALSE,
        cells=donor_cells$barcode, verbose=FALSE)
      pileup <- GetAssayData(fp, layer="positionEnrichment")[["fp"]][donor_cells$barcode,,drop=FALSE]
      by_group <- rowsum(as.matrix(pileup), group=donor_group[rownames(pileup)])
      cpm <- by_group / as.numeric(depth[rownames(by_group)]) * 1e6
      cpm <- cpm[c("Q1","Q2","Q3","Q4"),,drop=FALSE]
      totals[[set_name]] <- rowSums(cpm)
      for (group in rownames(cpm)) profile_rows[[length(profile_rows)+1]] <- data.frame(
        edge_key=edge_key, donor_id=donor_id, set=set_name, group=group,
        offset_from_tss=as.integer(colnames(cpm)), mean_cpm_per_gene=as.numeric(cpm[group,])/length(tss_by_set[[set_name]]))
    }
    for (set_name in c("EARLY","LATE")) effect_rows[[length(effect_rows)+1]] <- data.frame(
      edge_key=edge_key, donor_id=donor_id, set=set_name,
      target_minus_control=log2(totals[[set_name]][["Q4"]]/totals[[set_name]][["Q1"]]) - log2(totals[["CTRL"]][["Q4"]]/totals[["CTRL"]][["Q1"]]))
  }
}
profiles <- do.call(rbind, profile_rows); effects <- do.call(rbind, effect_rows)
write.csv(profiles, file.path(cache_directory, "signac_profiles.csv"), row.names=FALSE)
write.csv(effects, file.path(cache_directory, "signac_effects.csv"), row.names=FALSE)
summary <- aggregate(target_minus_control ~ edge_key + set, effects,
  FUN=function(v) c(mean=mean(v), se=sd(v)/sqrt(length(v)), donors=length(v)))
summary <- do.call(data.frame, summary); names(summary) <- c("edge_key","set","mean","se","donors")
write.csv(summary, file.path(cache_directory, "signac_summary.csv"), row.names=FALSE)
'''

if REFRESH_EXPENSIVE_RESULTS:
    with tempfile.TemporaryDirectory(prefix="figure5_signac_") as temporary_directory:
        r_path = Path(temporary_directory) / "figure5_signac.R"
        r_path.write_text(SIGNAC_PROGRAM)
        signac_log = subprocess.run([str(RSCRIPT_PATH), str(r_path), str(CACHE_DIRECTORY), str(FRAGMENT_DIRECTORY)], check=True, text=True, capture_output=True)
        (QC_DIRECTORY / "signac_refresh.log").write_text(signac_log.stdout + "\n" + signac_log.stderr)
    raw_signac_profiles = pd.read_csv(CACHE_DIRECTORY / "signac_profiles.csv")
    raw_signac_profiles.to_csv(SIGNAC_PROFILE_PATH, index=False, compression="gzip")
    (CACHE_DIRECTORY / "signac_profiles.csv").unlink()

signac_profiles = pd.read_csv(SIGNAC_PROFILE_PATH)
signac_effects = pd.read_csv(SIGNAC_EFFECT_PATH)
signac_summary = pd.read_csv(SIGNAC_SUMMARY_PATH)


# ## 8. Matched HECTOR and MultiVelo flow fields
# Coordinates are joined by barcode.  MultiVelo vectors are transformed from the
# deposited Figure 5 basis into the freshly verified GPU UMAP basis before gridding.

if REFRESH_EXPENSIVE_RESULTS:
    with h5py.File(ATLAS_PATH, "r") as atlas_file:
        atlas_index = np.array([value.decode() if isinstance(value, bytes) else str(value) for value in atlas_file["obs"]["_index"][:]])
        latent_space = np.asarray(atlas_file["obsm"]["X_scVI"][:], dtype=np.float32)
    latent_order = pd.Index(atlas_index).get_indexer(cells.index)
    if np.any(latent_order < 0):
        raise ValueError("HECTOR cells are missing from X_scVI.")
    neighbors = latent_neighbors(latent_space[latent_order])
    np.savez_compressed(NEIGHBOR_CACHE_PATH, neighbor_indices=neighbors)
else:
    if not NEIGHBOR_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing neighbour cache: {NEIGHBOR_CACHE_PATH}")
    neighbors = np.load(NEIGHBOR_CACHE_PATH)["neighbor_indices"]

xy = cells[["UMAP_1", "UMAP_2"]].to_numpy(float)
lineage_values = cells["hector_lineage"].astype(object).to_numpy()
rank_values = pd.to_numeric(cells["hector_pseudotime"], errors="coerce").to_numpy()
hector_arrows = hector_field(xy, lineage_values, rank_values, neighbors)
common_x_edges = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 51)
common_y_edges = np.linspace(xy[:, 1].min(), xy[:, 1].max(), 51)
hector_grid_field = grid_field(
    xy, hector_arrows, minimum_cells=max(5, round(len(xy) / 10_000)),
    x_edges=common_x_edges, y_edges=common_y_edges,
)

required_velocity_columns = {"cell_id", "analysis_cell", "umap_1", "umap_2", "velocity_umap_1", "velocity_umap_2"}
velocity_by_class = {}
for class_name in ["prpc", "nrpc", "rgc", "hc", "cone", "ac", "rod", "bc"]:
    velocity_table = pd.read_csv(MULTIVELO_DIRECTORY / f"{class_name}_cells.csv.gz")
    if not required_velocity_columns.issubset(velocity_table.columns):
        raise ValueError(f"MultiVelo schema mismatch for {class_name}.")
    if velocity_table["cell_id"].duplicated().any():
        raise ValueError(f"MultiVelo contains duplicate barcodes for {class_name}.")
    velocity_table = velocity_table[velocity_table["analysis_cell"]].copy()
    velocity_table[["UMAP_1", "UMAP_2"]] = cells.reindex(velocity_table["cell_id"])[["UMAP_1", "UMAP_2"]].to_numpy()
    if velocity_table[["UMAP_1", "UMAP_2"]].isna().any().any():
        raise ValueError(f"MultiVelo barcodes do not align with HECTOR for {class_name}.")
    velocity_by_class[class_name] = velocity_table

all_velocity_cells = pd.concat(velocity_by_class.values(), ignore_index=True)
finite_alignment = np.isfinite(all_velocity_cells[["umap_1", "umap_2", "UMAP_1", "UMAP_2"]]).all(axis=1)
old_velocity_basis = all_velocity_cells.loc[finite_alignment, ["umap_1", "umap_2"]].to_numpy(float)
new_hector_basis = all_velocity_cells.loc[finite_alignment, ["UMAP_1", "UMAP_2"]].to_numpy(float)
affine_design = np.column_stack([old_velocity_basis, np.ones(len(old_velocity_basis))])
velocity_basis_transform = np.linalg.lstsq(affine_design, new_hector_basis, rcond=None)[0]
alignment_residual = np.linalg.norm(affine_design @ velocity_basis_transform - new_hector_basis, axis=1)
if np.quantile(alignment_residual, 0.95) > 0.75:
    raise ValueError("The deposited MultiVelo coordinate basis does not match the verified HECTOR UMAP.")

multivelo_sources = {
    "precursor cell": ["prpc", "nrpc"],
    "camera-type eye photoreceptor cell": ["cone", "rod"],
    "retinal ganglion cell": ["rgc"],
    "amacrine cell": ["ac"],
    "retinal bipolar neuron": ["bc"],
    "retina horizontal cell": ["hc"],
}
multivelo_grid_field = {}
for lineage_name, class_names in multivelo_sources.items():
    velocity_cells = pd.concat([velocity_by_class[class_name] for class_name in class_names], ignore_index=True)
    finite = np.isfinite(velocity_cells[["UMAP_1", "UMAP_2", "velocity_umap_1", "velocity_umap_2"]]).all(axis=1)
    velocity_cells = velocity_cells.loc[finite].copy()
    old_vectors = velocity_cells[["velocity_umap_1", "velocity_umap_2"]].to_numpy(float)
    transformed_vectors = old_vectors @ velocity_basis_transform[:2, :]
    vector_length = np.linalg.norm(transformed_vectors, axis=1)
    nonzero = vector_length > 0
    velocity_xy = velocity_cells.loc[nonzero, ["UMAP_1", "UMAP_2"]].to_numpy(float)
    velocity_arrows = transformed_vectors[nonzero] / vector_length[nonzero, None]
    multivelo_grid_field[lineage_name] = grid_field(
        velocity_xy, velocity_arrows, minimum_cells=4,
        x_edges=common_x_edges, y_edges=common_y_edges,
    )

(QC_DIRECTORY / "multivelo_umap_alignment.json").write_text(json.dumps({
    "matched_cells": int(len(old_velocity_basis)),
    "affine_transform": velocity_basis_transform.tolist(),
    "residual_median": float(np.median(alignment_residual)),
    "residual_p95": float(np.quantile(alignment_residual, 0.95)),
    "coordinate_source": "barcode-aligned verified GPU HECTOR UMAP",
}, indent=2))


# ## 9. Final cache fingerprint and parity evidence
# A refresh is accepted only when the displayed edge has six donors and both methods agree in sign.

tabix_effects = pd.read_csv(CACHE_DIRECTORY / "tabix_effects.csv")
amacrine_signac = signac_effects[signac_effects["edge_key"].eq("Amacrine")]
amacrine_tabix = tabix_effects[tabix_effects["edge_key"].eq("Amacrine")]
if amacrine_signac["donor_id"].nunique() != 6:
    raise ValueError("The displayed amacrine Signac result must contain six donors.")
for set_name, expected_sign in [("EARLY", -1), ("LATE", 1)]:
    signac_mean = amacrine_signac.loc[amacrine_signac["set"].eq(set_name), "target_minus_control"].mean()
    tabix_mean = amacrine_tabix.loc[amacrine_tabix["set"].eq(set_name), "target_minus_control"].mean()
    if np.sign(signac_mean) != expected_sign or np.sign(tabix_mean) != expected_sign:
        raise ValueError(f"Signac/tabix directional check failed for {set_name}.")

if REFRESH_EXPENSIVE_RESULTS:
    cache_manifest = {
        "schema_version": 2,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_manifest_sha256": compact_manifest_hash,
        "settings_sha256": current_settings_fingerprint,
        "hector_cell_cache_sha256": sha256_file(CELL_CACHE_PATH),
        "signac_controls": "2026-08-25 openness-matched",
        "atlas_cells": int(len(cells)),
        "amacrine_signac_donors": int(amacrine_signac["donor_id"].nunique()),
    }
    CACHE_MANIFEST_PATH.write_text(json.dumps(cache_manifest, indent=2))

parity_report = {
    "input_manifest_sha256": compact_manifest_hash,
    "settings_sha256": current_settings_fingerprint,
    "python": platform.python_version(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "anndata": importlib.metadata.version("anndata"),
    "atlas_cells": int(len(cells)),
    "hector_ranked_cells": int(cells["hector_pseudotime"].notna().sum()),
    "signac_summary": signac_summary.to_dict(orient="records"),
    "amacrine_tabix": amacrine_tabix.groupby("set")["target_minus_control"].mean().to_dict(),
    "amacrine_signac": amacrine_signac.groupby("set")["target_minus_control"].mean().to_dict(),
}
PARITY_REPORT_PATH.write_text(json.dumps(parity_report, indent=2))


# ## 10. Assemble panels a–e
# The figure reads from direct tool output to developmental order, independent flow, and mechanism.

PANEL_LABEL_FONT_SIZE = 15
PLOT_TITLE_FONT_SIZE = 10
TEXT_FONT_SIZE = 8
PANEL_LABEL_X = 0.035

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": TEXT_FONT_SIZE,
    "font.weight": "normal",
    "axes.titlesize": PLOT_TITLE_FONT_SIZE,
    "axes.labelsize": TEXT_FONT_SIZE,
    "xtick.labelsize": TEXT_FONT_SIZE,
    "ytick.labelsize": TEXT_FONT_SIZE,
    "legend.fontsize": TEXT_FONT_SIZE,
    "legend.title_fontsize": TEXT_FONT_SIZE,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

figure = plt.figure(figsize=(10.8, 15.5), facecolor="white")
layout = GridSpec(4, 20, figure=figure, height_ratios=[1.45, 0.82, 1.0, 0.72], hspace=0.30, wspace=0.32)

# --- Panel a: reserve the exact rectangle for vector insertion after Matplotlib export.
tree_axis = figure.add_subplot(layout[0, :])
tree_axis.axis("off")

# --- Panel b: trajectory positions by post-conception week.
ridge_groups = [
    ("photoreceptors", ("Cone", "Rod")), ("retinal ganglion cells", ("RGC",)),
    ("amacrine cells", ("AC",)), ("bipolar cells", ("BC",)),
    ("horizontal cells", ("HC",)), ("Müller glial cells", ("MG",)),
]
weeks = [9, 11, 12, 13, 14, 15, 17, 20, 21, 24]
ridge_layout = GridSpecFromSubplotSpec(1, 6, subplot_spec=layout[1, 1:19], wspace=0.32)
ridge_cells = cells[cells["relative_position"].notna()].copy()
ridge_cells["week"] = ridge_cells["development_stage"].map(parse_week)
target_text = ridge_cells["trajectory_target"].fillna("").astype(str).str.lower()
ridge_cells["target_majorclass"] = np.select(
    [
        target_text.str.contains("bipolar"), target_text.str.contains("rod"),
        target_text.str.contains("mueller|muller"), target_text.str.contains("horizontal"),
        target_text.str.contains("amacrine"), target_text.str.contains("ganglion"),
        target_text.str.contains("cone|photoreceptor"), target_text.str.contains("progenitor"),
    ],
    ["BC", "Rod", "MG", "HC", "AC", "RGC", "Cone", "NRPC"],
    default="Other",
)
week_colours = mpl.colormaps["viridis"](np.linspace(0.1, 0.9, len(weeks)))
for column, (group_name, class_codes) in enumerate(ridge_groups):
    axis = figure.add_subplot(ridge_layout[0, column])
    block = ridge_cells[ridge_cells["target_majorclass"].isin(class_codes)]
    x_grid = np.linspace(0, 1, 300)
    for row, (week, colour) in enumerate(zip(weeks, week_colours)):
        week_cells = block[block["week"].eq(week)]
        donor_values = []
        week_rng = np.random.default_rng(RANDOM_SEED + column * 100 + week)
        for _, donor_cells in week_cells.groupby("donor_id", observed=True):
            values_for_donor = donor_cells["relative_position"].dropna().to_numpy()
            if len(values_for_donor):
                donor_values.append(week_rng.choice(values_for_donor, size=500, replace=True))
        values = np.concatenate(donor_values) if donor_values else np.array([])
        raw_count = len(week_cells)
        y = len(weeks) - 1 - row
        if len(values) >= 5 and np.ptp(values) > 0:
            density = gaussian_kde(values)(x_grid)
            density = density / density.max() * 0.48
            thin = raw_count < 100
            axis.fill_between(x_grid, y, y + density, color=colour, alpha=0.12 if thin else 0.40, linewidth=0)
            axis.plot(x_grid, y + density, color=colour, linewidth=0.6 if thin else 0.85, linestyle=(0, (2.2, 1.4)) if thin else "-")
            count_label = f"{raw_count / 1_000:.1f}k" if raw_count >= 1_000 else str(raw_count)
            axis.text(1.02, y, count_label, transform=axis.get_yaxis_transform(), fontsize=TEXT_FONT_SIZE, color="0.5", va="center")
    axis.set_title(group_name, fontsize=PLOT_TITLE_FONT_SIZE, fontweight="bold", pad=2)
    axis.set_xlim(0, 1); axis.set_ylim(-0.15, len(weeks) - 0.35)
    axis.set_xticks([0, 0.5, 1]); axis.tick_params(labelsize=TEXT_FONT_SIZE, length=2)
    axis.set_yticks(range(len(weeks)))
    axis.set_yticklabels([f"W{week}" for week in reversed(weeks)] if column == 0 else [])
    axis.grid(axis="x", color="0.91", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    if column == 0:
        axis.set_ylabel("Post-conception week", fontsize=TEXT_FONT_SIZE)
        panel_b_axis = axis
    if column == 2:
        axis.set_xlabel("Trajectory position", fontsize=TEXT_FONT_SIZE)

# --- Panel c: identical embedding sliced by week.
week_layout = GridSpecFromSubplotSpec(2, 5, subplot_spec=layout[2, :11], wspace=0.02, hspace=0.08)
display_class = cells["majorclass"].map(CLASS_TO_DISPLAY).fillna(cells["majorclass"].astype(str))
x_center = (cells["UMAP_1"].min() + cells["UMAP_1"].max()) / 2
y_center = (cells["UMAP_2"].min() + cells["UMAP_2"].max()) / 2
embedding_span = max(cells["UMAP_1"].max() - cells["UMAP_1"].min(), cells["UMAP_2"].max() - cells["UMAP_2"].min())
x_limits = x_center - embedding_span / 2, x_center + embedding_span / 2
y_limits = y_center - embedding_span / 2, y_center + embedding_span / 2
rng = np.random.default_rng(0)
panel_c_axes = []
for index, week in enumerate(weeks):
    axis = figure.add_subplot(week_layout[index // 5, index % 5])
    panel_c_axes.append(axis)
    # The white background joins the same rasterised layer as the points, so the semi-transparent
    # dots are blended into an opaque image that carries no soft mask.  Without this the exported
    # image is a transparency group, which Illustrator flattens to its own low raster resolution.
    axis.set_facecolor("white"); axis.patch.set_rasterized(True)
    mask = cells["development_stage"].map(parse_week).eq(week).to_numpy()
    positions = cells.loc[mask, ["UMAP_1", "UMAP_2"]].to_numpy()
    classes = display_class[mask].to_numpy()
    order = rng.permutation(len(positions))
    colours = [CLASS_PALETTE.get(value, "#BBBBBB") for value in classes[order]]
    axis.scatter(positions[order, 0], positions[order, 1], s=0.35, c=colours, alpha=0.65, linewidth=0, rasterized=True)
    axis.set_xlim(x_limits); axis.set_ylim(y_limits); axis.set_title(f"Week {week}", fontsize=PLOT_TITLE_FONT_SIZE, fontweight="bold")
    axis.set_aspect("equal", adjustable="box"); axis.set_box_aspect(1)
    axis.set_anchor("S" if index < 5 else "N")
    axis.set_xticks([]); axis.set_yticks([])
    for spine in axis.spines.values(): spine.set_visible(False)
    if index == 0: panel_c_axis = axis

# --- Panel d: matched HECTOR and MultiVelo directional fields.
flow_layout = GridSpecFromSubplotSpec(1, 2, subplot_spec=layout[2, 11:], wspace=0.00)
hector_axis = figure.add_subplot(flow_layout[0, 0])
multivelo_axis = figure.add_subplot(flow_layout[0, 1])
draw_flow_axis(hector_axis, cells, hector_grid_field, "HECTOR")
draw_flow_axis(multivelo_axis, cells, multivelo_grid_field, "MultiVelo")
shared_embedding_legend = [
    Line2D([], [], marker="o", linestyle="None", markersize=4.5, color=CLASS_PALETTE[label], label=label)
    for label in [
        "retinal progenitor cells", "photoreceptors", "retinal ganglion cells",
        "amacrine cells", "bipolar cells", "horizontal cells", "Müller glial cells",
    ]
]
figure.legend(
    handles=shared_embedding_legend, loc="center", bbox_to_anchor=(0.5, 0.264),
    ncol=7, frameon=False, fontsize=TEXT_FONT_SIZE, handletextpad=0.25, columnspacing=0.9,
)

# --- Panel e: RNA and chromatin views of the displayed amacrine edge.
bottom_layout = GridSpecFromSubplotSpec(1, 3, subplot_spec=layout[3, 1:19], wspace=0.34)
rna_axis = figure.add_subplot(bottom_layout[0, 0])
amacrine_rna = rna_curves[rna_curves["edge_key"].eq("Amacrine")]
amacrine_promoters = pd.read_csv(CACHE_DIRECTORY / "amacrine_promoters.csv")
direction_colormaps = {"EARLY": mpl.colormaps["viridis"].reversed(), "LATE": mpl.colormaps["plasma"]}
for set_name in ["EARLY", "LATE"]:
    cmap = direction_colormaps[set_name]
    individual = amacrine_rna[amacrine_rna["set"].eq(set_name)]
    for _, gene_curve in individual.groupby("symbol"):
        gene_curve = gene_curve.sort_values("bin")
        rna_axis.plot(
            gene_curve["edge_position"], gene_curve["scaled_expression"],
            color=cmap(0.5), alpha=0.16, linewidth=0.6,
        )
    mean_curve = individual.groupby("bin", as_index=False).agg(
        edge_position=("edge_position", "first"), scaled_expression=("scaled_expression", "mean")
    ).sort_values("bin")
    points = mean_curve[["edge_position", "scaled_expression"]].to_numpy()
    segments = np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)
    gradient = LineCollection(segments, cmap=cmap, norm=mcolors.Normalize(0, 1), linewidth=2.2)
    gradient.set_array(mean_curve["edge_position"].to_numpy())
    rna_axis.add_collection(gradient)
    last = mean_curve.iloc[-1]
    label_offset = (0, 14) if set_name == "LATE" else (0, -8)
    rna_axis.annotate(
        "turning on" if set_name == "LATE" else "turning off",
        xy=(last["edge_position"], last["scaled_expression"]), xytext=label_offset,
        textcoords="offset points", fontsize=TEXT_FONT_SIZE, color=cmap(0.75),
        va="bottom" if set_name == "LATE" else "top", ha="right",
    )
rna_axis.set(xlabel="Position along the edge (rank)", ylabel="Scaled RNA expression", xlim=(0, 1), ylim=(-0.05, 1.05))
rna_axis.set_title("RNA, the genes selected", fontsize=PLOT_TITLE_FONT_SIZE, fontweight="bold")
rna_axis.spines[["top", "right"]].set_visible(False)

# --- Panel e continued: control-adjusted chromatin for both RNA programmes.
signac_profiles["bin"] = (signac_profiles["offset_from_tss"] // GENOMIC_BIN_SIZE) * GENOMIC_BIN_SIZE
rebinned = signac_profiles.groupby(["edge_key", "donor_id", "set", "group", "bin"], as_index=False)["mean_cpm_per_gene"].sum()
donor_mean = rebinned.groupby(["edge_key", "set", "group", "bin"], as_index=False)["mean_cpm_per_gene"].mean()
control_only = donor_mean[donor_mean["set"].eq("CTRL")][
    ["edge_key", "group", "bin", "mean_cpm_per_gene"]
].rename(columns={"mean_cpm_per_gene": "control_cpm"})
donor_mean = donor_mean.merge(control_only, on=["edge_key", "group", "bin"], how="left")
donor_mean["height"] = donor_mean["mean_cpm_per_gene"] - donor_mean["control_cpm"]
height_quarter_mean = donor_mean.groupby(["edge_key", "set", "bin"], as_index=False)["height"].mean().rename(columns={"height": "height_quarter_mean"})
donor_mean = donor_mean.merge(height_quarter_mean, on=["edge_key", "set", "bin"], how="left")
donor_mean["centered_height"] = donor_mean["height"] - donor_mean["height_quarter_mean"]
quarter_colour_position = {
    "EARLY": {"Q1": 0.0, "Q2": 0.33, "Q3": 0.67, "Q4": 1.0},
    # Interior plasma samples keep all four turning-on quarters visibly within
    # the purple-magenta-orange sequence used by the RNA programme.
    "LATE": {"Q1": 0.20, "Q2": 0.40, "Q3": 0.60, "Q4": 0.80},
}
group_labels = {"Q1": "Q1  earliest quarter", "Q2": "Q2", "Q3": "Q3", "Q4": "Q4  latest quarter"}
chromatin_axes = [figure.add_subplot(bottom_layout[0, 1]), figure.add_subplot(bottom_layout[0, 2])]
for axis, set_name in zip(chromatin_axes, ["EARLY", "LATE"]):
    cmap = direction_colormaps[set_name]
    target = donor_mean[donor_mean["edge_key"].eq("Amacrine") & donor_mean["set"].eq(set_name)]
    for group in ["Q1", "Q2", "Q3", "Q4"]:
        target_curve = target[target["group"].eq(group)].sort_values("bin")
        axis.plot(
            target_curve["bin"] / 1_000,
            gaussian_filter1d(target_curve["centered_height"].to_numpy(), sigma=1.0),
            color=cmap(quarter_colour_position[set_name][group]), linewidth=1.4,
            label=group_labels[group],
        )
    effects = amacrine_signac[amacrine_signac["set"].eq(set_name)]["target_minus_control"]
    expected = -1 if set_name == "EARLY" else 1
    n_genes = int(amacrine_promoters["edge_pattern"].eq(set_name).sum())
    axis.set_title(f"chromatin, genes turning {'off' if set_name == 'EARLY' else 'on'} ({n_genes})\nchange {effects.mean():+.3f}, {int((np.sign(effects) == expected).sum())} of {len(effects)} donors agree", fontsize=PLOT_TITLE_FONT_SIZE, fontweight="bold")
    axis.axvline(0, color="0.6", linestyle="--", linewidth=0.7); axis.axhline(0, color="0.85", linewidth=0.6)
    data_y_min, data_y_max = axis.get_ylim()
    axis.set_ylim(data_y_min, data_y_max + 0.30 * (data_y_max - data_y_min))
    axis.set_xlabel("Distance from transcription start (kb)")
    axis.spines[["top", "right"]].set_visible(False)
    if set_name == "EARLY": axis.set_ylabel("Tn5 insertions per million,\nminus control, own average", fontsize=TEXT_FONT_SIZE, labelpad=1)
    axis.legend(
        title=f"turning {'off' if set_name == 'EARLY' else 'on'}, by quarter",
        fontsize=TEXT_FONT_SIZE, title_fontsize=TEXT_FONT_SIZE,
        loc="upper right", bbox_to_anchor=(1.06, 1.0),
        handlelength=1.0, frameon=False,
    )

figure.subplots_adjust(top=0.985, bottom=0.045, left=0.055, right=0.985)
for week_axis in panel_c_axes:
    week_position = week_axis.get_position()
    week_axis.set_position([
        week_position.x0,
        week_position.y0 + 0.015,
        week_position.width,
        week_position.height,
    ])
for flow_axis in [hector_axis, multivelo_axis]:
    flow_position = flow_axis.get_position()
    flow_axis.set_position([
        flow_position.x0,
        flow_position.y0 + 0.009,
        flow_position.width,
        flow_position.height,
    ])
for bottom_axis in [rna_axis, *chromatin_axes]:
    bottom_position = bottom_axis.get_position()
    bottom_axis.set_position([
        bottom_position.x0,
        bottom_position.y0 + 0.028,
        bottom_position.width,
        bottom_position.height,
    ])
panel_c_label_y = panel_c_axis.get_position().y1 + 0.005
figure.text(PANEL_LABEL_X, tree_axis.get_position().y1 + 0.004, "a", fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold", va="bottom")
figure.text(PANEL_LABEL_X, panel_b_axis.get_position().y1 + 0.005, "b", fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold", va="bottom")
figure.text(PANEL_LABEL_X, panel_c_label_y, "c", fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold", va="bottom")
figure.text(hector_axis.get_position().x0 - 0.025, panel_c_label_y, "d", fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold", va="bottom")
figure.text(PANEL_LABEL_X, rna_axis.get_position().y1 + 0.010, "e", fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold", va="bottom")
tree_rectangle = tree_axis.get_position()
figure.savefig(FINAL_PDF_PATH, dpi=400)
plt.close(figure)


# ## 11. Preserve the radial tree as vector artwork and render one canonical preview
# PyMuPDF overlays the original vector page into panel a's reserved rectangle.

output_document = fitz.open(FINAL_PDF_PATH)
tree_document = fitz.open(TREE_PDF_PATH)
page = output_document[0]
page_width, page_height = page.rect.width, page.rect.height
tree_page = tree_document[0]
legend_top = min(
    text_block[1]
    for text_block in tree_page.get_text("blocks")
    if str(text_block[4]).strip().startswith("Legend")
)
occupied_rectangles = []
for text_block in tree_page.get_text("blocks"):
    block_text = str(text_block[4]).strip()
    if block_text and block_text != "HECTOR Trajectory Analysis" and text_block[1] < legend_top:
        occupied_rectangles.append(fitz.Rect(text_block[:4]))
for image_information in tree_page.get_image_info():
    occupied_rectangles.append(fitz.Rect(image_information["bbox"]))
if occupied_rectangles:
    source_rectangle = fitz.Rect(
        max(tree_page.rect.x0, min(rectangle.x0 for rectangle in occupied_rectangles) - 6),
        max(tree_page.rect.y0, min(rectangle.y0 for rectangle in occupied_rectangles) - 6),
        min(tree_page.rect.x1, max(rectangle.x1 for rectangle in occupied_rectangles) + 6),
        min(tree_page.rect.y1, legend_top - 2, max(rectangle.y1 for rectangle in occupied_rectangles) + 6),
    )
else:
    source_rectangle = tree_page.rect
target_rectangle = fitz.Rect(
    (tree_rectangle.x0 + 0.01 * tree_rectangle.width) * page_width,
    (1 - tree_rectangle.y1 + 0.03 * tree_rectangle.height) * page_height,
    (tree_rectangle.x1 - 0.01 * tree_rectangle.width) * page_width,
    (1 - tree_rectangle.y0 + 0.14 * tree_rectangle.height) * page_height,
)
page.show_pdf_page(target_rectangle, tree_document, 0, keep_proportion=True, overlay=True, clip=source_rectangle)

temporary_pdf = FINAL_PDF_PATH.with_suffix(".vector.pdf")
output_document.save(temporary_pdf, garbage=4, deflate=True)
output_document.close(); tree_document.close()
temporary_pdf.replace(FINAL_PDF_PATH)

rendered = fitz.open(FINAL_PDF_PATH)
pixmap = rendered[0].get_pixmap(dpi=300, alpha=False)
pixmap.save(FINAL_PNG_PATH)
rendered.close()

print(f"Wrote {FINAL_PDF_PATH}")
print(f"Wrote {FINAL_PNG_PATH}")
print(f"Parity report: {PARITY_REPORT_PATH}")
