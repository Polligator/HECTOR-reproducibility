"""Precompute for the Figure-4 plot package — rebuild every cache from source.

Each panel cache under ``result/data/plot_cache/`` is rebuilt from the benchmark
outputs in ``result/data/`` + the Cell Ontology (``cl.obo``) + the source
``.h5ad``:

* ``hop_data.pkl`` and ``depth_rows.csv`` — from the prediction CSVs +
  ontology (deterministic; reproduce the cache exactly).
* ``donor.pkl`` / ``broad_class.pkl`` — per-cell labels joined from the source
  ``TS_downsampled_cells.h5ad`` obs (deterministic; value-exact).
* ``umap_coords.pkl`` — UMAP over the bridge embeddings. Compartment labels are
  deterministic; the 2-D coordinates are not seed-stable across versions, so an
  existing cache is reused (keeps the locked panel-a layout) and rebuilt only
  when absent.

``run_all`` rebuilds whatever is missing (or everything when ``force=True``),
loading the ontology once and sharing it across builders.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DATA_DIR, PLOT_CACHE_DIR, ONTOLOGY_OBO, PROJECT_ROOT

# (series label, prediction CSV) — order matches the locked hop-curve legend.
_HOP_SPEC = [
    ("HECTOR (own head)",       "hector_native_closed_set_predictions.csv"),
    ("HECTOR (own head, open)", "hector_native_open_set_predictions.csv"),
    ("HECTOR (embedding)",      "hector_ppr_candidate_predictions.csv"),
    ("scimilarity",             "scimilarity_ppr_candidate_predictions.csv"),
    ("scGPT",                   "scgpt_ppr_candidate_predictions.csv"),
    ("scCello",                 "sccello_ppr_candidate_predictions.csv"),
    ("Geneformer",              "geneformer_104m_ppr_candidate_predictions.csv"),
]


def _ensure_paths() -> None:
    """Put model_eval/ and scripts/ on sys.path (for evaluate + ontology)."""
    for p in (str(PROJECT_ROOT / "scripts" / "model_eval"), str(PROJECT_ROOT / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_dag():
    _ensure_paths()
    from Hierarchical_metrics.cell_ontology_metrics import load_cell_ontology
    return load_cell_ontology(str(ONTOLOGY_OBO))


def _cache_is_current(cache, inputs) -> bool:
    """True when *cache* exists and no input file is newer than it.

    Without this a cache built from an earlier benchmark run is returned intact
    after the predictions change, and the panel silently draws the old numbers.
    That is how panel d kept plotting a superseded query set: the run succeeded,
    the figure looked plausible, and nothing pointed at the stale file.
    """
    if not cache.exists():
        return False
    cache_time = cache.stat().st_mtime
    for path in inputs:
        path = Path(path)
        if path.exists() and path.stat().st_mtime > cache_time:
            return False
    return True


def build_hop_data(dag=None, force: bool = False):
    """Rebuild hop_data.pkl from the prediction CSVs + cl.obo (deterministic)."""
    cache = PLOT_CACHE_DIR / "hop_data.pkl"
    inputs = [DATA_DIR / csv for _, csv in _HOP_SPEC] + [ONTOLOGY_OBO]
    if _cache_is_current(cache, inputs) and not force:
        return cache
    _ensure_paths()
    from evaluate import compute_hop_distance, generate_random_baseline
    if dag is None:
        dag = _load_dag()

    # Query cells a model never returned a prediction for are counted as errors,
    # the same convention panel c uses: scCello's tokenizer discards cells it
    # cannot represent, and a cell it will not process is one it has not named.
    # They enter as a distance no k can reach rather than being left out of the
    # denominator, which would otherwise score that model only on the cells it
    # chose to answer.
    expected = pd.read_csv(DATA_DIR / "candidate_label_table.csv")
    n_query_cells = int(expected["n_cells"].sum())

    series: dict[str, pd.Series] = {}
    truth = None
    for label, csv in _HOP_SPEC:
        df = pd.read_csv(DATA_DIR / csv, low_memory=False)
        hops = compute_hop_distance(
            df["predicted_cell_type_id"], df["truth_cell_type_id"], dag=dag
        ).reset_index(drop=True)
        missing = n_query_cells - len(hops)
        if missing > 0:
            print(f"[hop_data] {label}: {missing:,} query cells not returned, counted as errors")
            hops = pd.concat([hops, pd.Series([np.nan] * missing)], ignore_index=True)
        series[label] = hops
        if truth is None or len(df) == n_query_cells:
            truth = df["truth_cell_type_id"].reset_index(drop=True)

    # Random baseline: random guesses over the held-out label space.
    rng_pred = generate_random_baseline(sorted(set(truth)), n_samples=len(truth), seed=42)
    series["Random Baseline"] = compute_hop_distance(
        pd.Series(rng_pred).reset_index(drop=True), truth, dag=dag
    ).reset_index(drop=True)

    PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(series, f)
    return cache


# Every model is binned on the same fixed depth ranges, covering the full
# depth range the held-out set spans (1-10), so all models share one partition.
_DEPTH_BINS = [
    (0, "Shallow\n(1-4)", (1, 4)),
    (1, "Intermediate\n(5-6)", (5, 6)),
    (2, "Deep\n(7-10)", (7, 10)),
]
_DEPTH_COLUMNS = [
    "model", "depth_bin_rank", "depth_bin_label", "n_total",
    "n_exact", "n_near", "n_far", "frac_exact", "frac_near", "frac_far",
]
# (label as supplement S5 names it, prediction CSV)
_DEPTH_SPEC = [
    ("Hector (Native)",       "hector_native_closed_set_predictions.csv"),
    ("Hector (Native, open)", "hector_native_open_set_predictions.csv"),
    ("Hector (PPR)",          "hector_ppr_candidate_predictions.csv"),
    ("scimilarity (PPR)",     "scimilarity_ppr_candidate_predictions.csv"),
    ("scGPT (PPR)",           "scgpt_ppr_candidate_predictions.csv"),
    ("scCello (PPR)",         "sccello_ppr_candidate_predictions.csv"),
    ("Geneformer (PPR)",      "geneformer_104m_ppr_candidate_predictions.csv"),
]

# Aliases: the frame covers every model, not just native, despite the name.
_NATIVE_DEPTH_BINS = _DEPTH_BINS
_NATIVE_DEPTH_COLUMNS = _DEPTH_COLUMNS


def _bin_rows(model_label, truth_ids, hops, depth_of):
    """Count exact / near / far outcomes per depth bin for one model."""
    depth = np.array([depth_of.get(str(t), -1) for t in truth_ids])
    hops = np.asarray(hops, dtype=float)
    rows = []
    for rank, label, (lo, hi) in _DEPTH_BINS:
        sel = (depth >= lo) & (depth <= hi)
        h = hops[sel]
        n_total = int(sel.sum())
        if n_total == 0:
            continue
        n_exact = int((h == 0).sum())
        n_near = int(((h >= 1) & (h <= 2)).sum())
        # Everything else is far: 3 or more hops, the -1 disconnected sentinel,
        # and the NaN standing for a query cell the model never returned.
        n_far = n_total - n_exact - n_near
        rows.append({
            "model": model_label, "depth_bin_rank": rank, "depth_bin_label": label,
            "n_total": n_total, "n_exact": n_exact, "n_near": n_near, "n_far": n_far,
            "frac_exact": n_exact / n_total, "frac_near": n_near / n_total,
            "frac_far": n_far / n_total,
        })
    return rows


def _depth_frame(dag) -> pd.DataFrame:
    """Depth-binned exact/near/far outcomes for every model, on shared bins.

    hop = shortest-path distance pred->truth; depth = depth of the true term.
    Query cells a model never returned are added back as failures, the same
    convention panels c and d use. Assumes ``_ensure_paths`` has run.
    """
    from evaluate import (compute_hop_distance, compute_label_depths,
                          generate_random_baseline)

    expected = pd.read_csv(DATA_DIR / "candidate_label_table.csv")
    expected["cell_type_id"] = expected["cell_type_id"].astype(str)
    depths = compute_label_depths(expected["cell_type_id"], dag=dag)
    depth_of = dict(zip(expected["cell_type_id"], depths))
    n_query_cells = int(expected["n_cells"].sum())

    rows = []
    truth_full = None
    for model_label, csv in _DEPTH_SPEC:
        path = DATA_DIR / csv
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        truth_ids = df["truth_cell_type_id"].astype(str).tolist()
        hops = compute_hop_distance(
            df["predicted_cell_type_id"], df["truth_cell_type_id"], dag=dag
        ).to_numpy(dtype=float)

        have = pd.Series(truth_ids).value_counts()
        for cell_type_id, n_expected in zip(expected["cell_type_id"], expected["n_cells"]):
            missing = int(n_expected) - int(have.get(cell_type_id, 0))
            if missing > 0:
                truth_ids.extend([cell_type_id] * missing)
                hops = np.concatenate([hops, np.full(missing, np.nan)])
        if truth_full is None or len(truth_ids) == n_query_cells:
            truth_full = list(truth_ids)
        rows.extend(_bin_rows(model_label, truth_ids, hops, depth_of))

    if truth_full is not None:
        rng_pred = generate_random_baseline(
            sorted(set(truth_full)), n_samples=len(truth_full), seed=42)
        rng_hops = compute_hop_distance(
            pd.Series(rng_pred), pd.Series(truth_full), dag=dag).to_numpy(dtype=float)
        rows.extend(_bin_rows("Random Baseline", truth_full, rng_hops, depth_of))

    return pd.DataFrame(rows, columns=_DEPTH_COLUMNS)


def build_native_depth_rows(dag=None, force: bool = False):
    """Rebuild depth_rows.csv — every model, shared depth bins — from the CSVs + cl.obo."""
    cache = PLOT_CACHE_DIR / "depth_rows.csv"
    inputs = [DATA_DIR / csv for _, csv in _DEPTH_SPEC] + [ONTOLOGY_OBO]
    if _cache_is_current(cache, inputs) and not force:
        return cache
    _ensure_paths()
    if dag is None:
        dag = _load_dag()
    frame = _depth_frame(dag)
    PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return cache


# Source atlas whose obs carries the per-cell donor + broad-class labels.
SOURCE_H5AD = PROJECT_ROOT / "input" / "TS_downsampled_cells.h5ad"

# model -> bridge embeddings pickle. Each pkl's row_ids equal the source obs
# observation_joinid, so a join on it recovers per-cell labels for every model
# (including the scCello subset that drops some bridge cells).
_BRIDGE_PKL = {
    "Hector": "hector_bridge_embeddings.pkl",
    "scimilarity": "scimilarity_bridge_embeddings.pkl",
    "scGPT": "scgpt_bridge_embeddings.pkl",
    "scCello": "sccello_bridge_embeddings.pkl",
    "geneformer": "geneformer_104m_bridge_embeddings.pkl",
}
_JOIN_OBS_COL = "observation_joinid"
_DONOR_OBS_COL = "donor_id"
_BROAD_OBS_COL = "broad_cell_class"


def _decode_obs_column(obs, name):
    """Decode one AnnData obs column (categorical or plain) to a list of str/None."""
    import h5py
    o = obs[name]
    if isinstance(o, h5py.Group) and "categories" in o and "codes" in o:
        cats = [c.decode() if isinstance(c, bytes) else c for c in o["categories"][:]]
        codes = o["codes"][:]
        return [cats[c] if c >= 0 else None for c in codes]
    return [v.decode() if isinstance(v, bytes) else v for v in o[:]]


def _read_obs_columns(path, columns):
    """Read the given obs columns from an .h5ad via h5py (obs is cheap to read)."""
    import h5py
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        return {c: _decode_obs_column(obs, c) for c in columns}


def _per_cell_obs_labels(columns):
    """Map each model's bridge-embedding rows to source-obs labels.

    Joins each bridge pkl's ``row_ids`` to the source obs ``observation_joinid``
    (verified to reproduce the shipped caches exactly for all models). Returns
    ``{column: {model: ndarray(object)}}`` keyed in the locked model order.
    """
    obs = _read_obs_columns(SOURCE_H5AD, [_JOIN_OBS_COL, *columns])
    pos = {str(j): i for i, j in enumerate(obs[_JOIN_OBS_COL])}
    out = {c: {} for c in columns}
    for model, pkl in _BRIDGE_PKL.items():
        with open(DATA_DIR / pkl, "rb") as f:
            row_ids = pickle.load(f)["row_ids"]
        idx = [pos[str(r)] for r in row_ids]
        for c in columns:
            col = obs[c]
            out[c][model] = np.array([col[i] for i in idx], dtype=object)
    return out


def build_donor_broad_class(force: bool = False):
    """Rebuild donor.pkl + broad_class.pkl from the source .h5ad obs."""
    donor_cache = PLOT_CACHE_DIR / "donor.pkl"
    broad_cache = PLOT_CACHE_DIR / "broad_class.pkl"
    if donor_cache.exists() and broad_cache.exists() and not force:
        return donor_cache, broad_cache
    labels = _per_cell_obs_labels([_DONOR_OBS_COL, _BROAD_OBS_COL])
    PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(donor_cache, "wb") as f:
        pickle.dump(labels[_DONOR_OBS_COL], f)
    with open(broad_cache, "wb") as f:
        pickle.dump(labels[_BROAD_OBS_COL], f)
    return donor_cache, broad_cache


# Ordered lineage anchors: (compartment, [CL ids]). First match (in this order)
# wins. Recovered from the original precompute_umap builder.
_COMPARTMENT_ANCHORS = [
    ("Immune / blood",       ["CL:0000988"]),
    ("Endothelial",          ["CL:0000115"]),
    ("Muscle",               ["CL:0000187"]),
    ("Neural / glial",       ["CL:0000540", "CL:0000125"]),
    ("Epithelial",           ["CL:0000066"]),
    ("Stromal / fibroblast", ["CL:0000499", "CL:0000057", "CL:0008019", "CL:0000669", "CL:0000136"]),
    ("Germ",                 ["CL:0000586"]),
]
# Per-model UMAP distance metric (HECTOR's embedding space is angular).
_UMAP_METRIC = {"Hector": "cosine", "scimilarity": "euclidean", "scGPT": "euclidean",
                "scCello": "euclidean", "geneformer": "euclidean"}
_UMAP_KW = dict(n_neighbors=15, min_dist=0.3, random_state=0)


def _compartment_map(unique_ids, dag):
    """Map each CL id -> coarse lineage compartment ('Other' if unanchored).

    Works on ontology *names*: anchors and each cell's id are resolved to names
    via ``dag.graph['id_to_name']``, then a cell joins the first compartment
    whose anchor is among its inclusive ancestors. Returns ``(map, n_other)``.
    """
    from Hierarchical_metrics.cell_ontology_metrics import _inclusive_ancestor_set
    id_to_name = dag.graph.get("id_to_name", {})
    anchor_name_sets = []
    for comp, ids in _COMPARTMENT_ANCHORS:
        names = {id_to_name.get(i) for i in ids}
        anchor_name_sets.append((comp, {n for n in names if n is not None and n in dag}))

    out = {}
    n_other = 0
    for cid in unique_ids:
        name = id_to_name.get(cid)
        comp = "Other"
        if name is not None and name in dag:
            anc = _inclusive_ancestor_set(dag, name)
            for comp_name, anchors in anchor_name_sets:
                if anchors & anc:
                    comp = comp_name
                    break
        if comp == "Other":
            n_other += 1
        out[cid] = comp
    return out, n_other


def build_umap_coords(dag=None, force: bool = False):
    """Rebuild umap_coords.pkl from the bridge embeddings.

    UMAP is re-derived only when the cache is absent (or ``force``); when present
    it is kept so the published panel-a orientation stays locked. A fresh layout
    may be re-oriented (UMAP is not seed-stable across versions); the per-cell
    compartment labels are deterministic (ontology-derived).
    """
    cache_path = PLOT_CACHE_DIR / "umap_coords.pkl"
    if cache_path.exists() and not force:
        return cache_path
    _ensure_paths()
    if dag is None:
        dag = _load_dag()
    import umap

    raw = {}
    all_ids = set()
    for model, pkl in _BRIDGE_PKL.items():
        with open(DATA_DIR / pkl, "rb") as f:
            raw[model] = pickle.load(f)
        all_ids.update(raw[model]["cell_type_ids"])
    comp_map, _ = _compartment_map(all_ids, dag)

    cache = {"compartment_order": [c for c, _ in _COMPARTMENT_ANCHORS] + ["Other"]}
    for model, d in raw.items():
        emb = np.asarray(d["embeddings"], dtype=np.float32)
        comps = np.array([comp_map[c] for c in d["cell_type_ids"]], dtype=object)
        metric = _UMAP_METRIC.get(model, "euclidean")
        reducer = umap.UMAP(metric=metric, **_UMAP_KW)
        coords = np.asarray(reducer.fit_transform(emb), dtype=np.float32)
        cache[model] = {
            "coords": coords,
            "compartment": comps,
            "cell_type": np.array(d["cell_types"], dtype=object),
        }
    PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    return cache_path


def run_all(force: bool = False) -> None:
    """Rebuild every plot cache from source (missing ones, or all when forced).

    The ontology is loaded once and shared across builders. Builders whose cache
    already exists short-circuit, so a warm ``plot_cache/`` is a fast no-op.
    """
    PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    caches = ("hop_data.pkl", "depth_rows.csv", "donor.pkl",
              "broad_class.pkl", "umap_coords.pkl")
    need_rebuild = force or any(not (PLOT_CACHE_DIR / c).exists() for c in caches)
    dag = _load_dag() if need_rebuild else None

    build_hop_data(dag=dag, force=force)
    build_native_depth_rows(dag=dag, force=force)
    build_donor_broad_class(force=force)
    build_umap_coords(dag=dag, force=force)
