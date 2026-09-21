"""Utilities for evaluating cell type predictions against the Cell Ontology."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import networkx as nx
import pandas as pd

RAW_CATEGORY_ORDER = ("exact", "parent", "child", "sibling", "no match")
DISPLAY_NAME_MAP = {
    "exact": "Exact Match",
    "parent": "Parent Match",
    "child": "Child Match",
    "sibling": "Sibling Match",
    "no match": "No Match",
}
DISPLAY_CATEGORY_ORDER = tuple(DISPLAY_NAME_MAP[label] for label in RAW_CATEGORY_ORDER)
HIERARCHICAL_METRIC_COLUMNS = (
    "hierarchical_precision",
    "hierarchical_recall",
    "hierarchical_f1",
    "hierarchical_accuracy",
    "hierarchical_jaccard",
    "lca_depth",
    "flat_accuracy",
)
TERMINAL_ONTOLOGY_ID_PATTERN = re.compile(r"\((?P<ontology_id>[A-Za-z][A-Za-z0-9_]*:\d+)\)\s*$")


def load_cell_ontology(obofile: str | Path) -> nx.DiGraph:
    """Load a Cell Ontology OBO or OWL file into the DAG used by the metrics."""
    path = Path(obofile).expanduser().resolve()
    base_graph = _load_child_to_parent_dag(path)
    graph = base_graph.reverse(copy=True)
    graph.graph["orientation"] = "parent_to_child"
    graph.graph["source_obofile"] = str(path)
    return graph


def classify_ontology_matches(
    pred: Iterable[str] | pd.Series | str,
    truth: Iterable[str] | pd.Series | str,
    *,
    obofile: str | Path | None = None,
    dag: nx.DiGraph | None = None,
) -> pd.Series | str:
    """Classify prediction-ground-truth pairs with an immediate-relationship scheme."""
    graph = _resolve_matching_dag(obofile=obofile, dag=dag)
    pred_series, truth_series, scalar_input = _coerce_pair_inputs(pred, truth)
    pred_series = _canonicalize_labels(pred_series, graph, role="Prediction")
    truth_series = _canonicalize_labels(truth_series, graph, role="Ground-truth")
    _validate_labels(pred_series, graph, role="Prediction")
    _validate_labels(truth_series, graph, role="Ground-truth")

    pair_cache: dict[tuple[str, str], str] = {}
    labels = []
    for pred_label, truth_label in zip(pred_series.tolist(), truth_series.tolist()):
        key = (pred_label, truth_label)
        cached = pair_cache.get(key)
        if cached is not None:
            labels.append(cached)
            continue

        if pred_label == truth_label:
            match = "exact"
        else:
            pred_parents = set(graph.predecessors(pred_label))
            truth_parents = set(graph.predecessors(truth_label))
            if pred_parents & truth_parents:
                match = "sibling"
            elif pred_label in truth_parents:
                match = "parent"
            elif truth_label in pred_parents:
                match = "child"
            else:
                match = "no match"

        pair_cache[key] = match
        labels.append(match)

    result = pd.Series(labels, index=pred_series.index, name="ontology_match")
    if scalar_input:
        return result.iloc[0]
    return result


def summarize_ontology_matches(
    categories: Iterable[str] | pd.Series | str,
    *,
    normalize: bool = True,
    display_labels: bool = True,
    group_labels: Iterable[str] | pd.Series | str | None = None,
    averaging: str = "micro",
) -> pd.Series | pd.DataFrame:
    """Summarize coarse ontology matches in a fixed display order."""
    category_series = _coerce_category_input(categories)
    unknown = sorted(set(category_series.dropna()) - set(RAW_CATEGORY_ORDER))
    if unknown:
        raise ValueError(f"Unknown ontology match categories: {unknown}")

    averaging = _validate_averaging_mode(averaging)
    if averaging != "micro" and not normalize:
        raise ValueError("Macro averaging is only supported for normalized ontology category summaries.")

    if averaging == "micro":
        return _summarize_ontology_matches_micro(
            category_series,
            normalize=normalize,
            display_labels=display_labels,
        )

    group_series = _coerce_group_labels(group_labels, category_series.index)
    per_group = []
    for _, subset in category_series.groupby(group_series):
        per_group.append(
            _summarize_ontology_matches_micro(
                subset,
                normalize=True,
                display_labels=display_labels,
            )
        )

    macro_summary = pd.DataFrame(per_group).mean(axis=0)
    macro_summary.name = "macro"
    if averaging == "macro":
        return macro_summary

    micro_summary = _summarize_ontology_matches_micro(
        category_series,
        normalize=True,
        display_labels=display_labels,
    )
    micro_summary.name = "micro"
    return pd.DataFrame([micro_summary, macro_summary])


def compute_fine_ontology_distance(
    pred: Iterable[str] | pd.Series | str,
    truth: Iterable[str] | pd.Series | str,
    *,
    obofile: str | Path | None = None,
    dag: nx.DiGraph | None = None,
) -> pd.Series | int | str:
    """Compute signed and sibling-style ontology distances between predictions and truth."""
    graph = _resolve_matching_dag(obofile=obofile, dag=dag)
    pred_series, truth_series, scalar_input = _coerce_pair_inputs(pred, truth)
    pred_series = _canonicalize_labels(pred_series, graph, role="Prediction")
    truth_series = _canonicalize_labels(truth_series, graph, role="Ground-truth")
    _validate_labels(pred_series, graph, role="Prediction")
    _validate_labels(truth_series, graph, role="Ground-truth")

    undirected_graph = nx.Graph(graph)
    pair_cache: dict[tuple[str, str], int | str] = {}
    distances: list[int | str] = []

    for pred_label, truth_label in zip(pred_series.tolist(), truth_series.tolist()):
        key = (pred_label, truth_label)
        cached = pair_cache.get(key)
        if cached is not None:
            distances.append(cached)
            continue

        if pred_label == truth_label:
            score: int | str = 0
        elif nx.has_path(graph, source=pred_label, target=truth_label):
            score = nx.shortest_path_length(graph, source=pred_label, target=truth_label) - 1
        elif nx.has_path(graph, source=truth_label, target=pred_label):
            score = nx.shortest_path_length(graph, source=truth_label, target=pred_label) - 1
            score *= -1
        else:
            try:
                path = next(
                    nx.algorithms.simple_paths.shortest_simple_paths(
                        undirected_graph, source=pred_label, target=truth_label
                    ),
                    None,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path = None

            if path is None:
                score = 1000
            else:
                score = f"{len(path) - 1}_sib"

        pair_cache[key] = score
        distances.append(score)

    result = pd.Series(distances, index=pred_series.index, name="fine_ontology_distance")
    if scalar_input:
        return result.iloc[0]
    return result


def compute_hierarchical_metrics(
    pred: Iterable[str] | pd.Series | str,
    truth: Iterable[str] | pd.Series | str,
    *,
    obofile: str | Path | None = None,
    dag: nx.DiGraph | None = None,
) -> pd.DataFrame | pd.Series:
    """Compute per-cell hierarchical overlap metrics and baselines."""
    graph = _resolve_matching_dag(obofile=obofile, dag=dag)
    pred_series, truth_series, scalar_input = _coerce_pair_inputs(pred, truth)
    pred_series = _canonicalize_labels(pred_series, graph, role="Prediction")
    truth_series = _canonicalize_labels(truth_series, graph, role="Ground-truth")
    _validate_labels(pred_series, graph, role="Prediction")
    _validate_labels(truth_series, graph, role="Ground-truth")

    pair_cache: dict[tuple[str, str], dict[str, float | int]] = {}
    records: list[dict[str, float | int]] = []

    for pred_label, truth_label in zip(pred_series.tolist(), truth_series.tolist()):
        key = (pred_label, truth_label)
        cached = pair_cache.get(key)
        if cached is not None:
            records.append(cached)
            continue

        pred_ancestors = _inclusive_ancestor_set(graph, pred_label)
        truth_ancestors = _inclusive_ancestor_set(graph, truth_label)
        intersection = pred_ancestors & truth_ancestors
        union = pred_ancestors | truth_ancestors

        hierarchical_precision = len(intersection) / len(pred_ancestors)
        hierarchical_recall = len(intersection) / len(truth_ancestors)
        if hierarchical_precision + hierarchical_recall == 0:
            hierarchical_f1 = 0.0
        else:
            hierarchical_f1 = (
                2
                * hierarchical_precision
                * hierarchical_recall
                / (hierarchical_precision + hierarchical_recall)
            )
        hierarchical_accuracy = len(intersection) / len(union)

        row = {
            "hierarchical_precision": hierarchical_precision,
            "hierarchical_recall": hierarchical_recall,
            "hierarchical_f1": hierarchical_f1,
            "hierarchical_accuracy": hierarchical_accuracy,
            "hierarchical_jaccard": hierarchical_accuracy,
            "lca_depth": _deepest_common_ancestor_depth(graph, pred_label, truth_label),
            "flat_accuracy": int(pred_label == truth_label),
        }
        pair_cache[key] = row
        records.append(row)

    result = pd.DataFrame(records, index=pred_series.index)
    if scalar_input:
        return result.iloc[0]
    return result


def summarize_hierarchical_metrics(
    metrics: pd.DataFrame | pd.Series | None = None,
    *,
    pred: Iterable[str] | pd.Series | str | None = None,
    truth: Iterable[str] | pd.Series | str | None = None,
    obofile: str | Path | None = None,
    dag: nx.DiGraph | None = None,
    group_labels: Iterable[str] | pd.Series | str | None = None,
    averaging: str = "micro",
) -> pd.Series | pd.DataFrame:
    """Summarize hierarchical metrics as simple per-cell means."""
    averaging = _validate_averaging_mode(averaging)

    if metrics is None:
        if pred is None or truth is None:
            raise ValueError("Pass either 'metrics' or both 'pred' and 'truth'.")
        graph = _resolve_matching_dag(obofile=obofile, dag=dag)
        _, truth_series, _ = _coerce_pair_inputs(pred, truth)
        truth_series = _canonicalize_labels(truth_series, graph, role="Ground-truth")
        _validate_labels(truth_series, graph, role="Ground-truth")
        metrics_frame = compute_hierarchical_metrics(
            pred,
            truth,
            obofile=obofile,
            dag=graph,
        )
        if isinstance(metrics_frame, pd.Series):
            metrics_frame = metrics_frame.to_frame().T
        group_series = truth_series
    else:
        if pred is not None or truth is not None:
            raise ValueError("Pass either 'metrics' or raw 'pred'/'truth' inputs, not both.")
        if isinstance(metrics, pd.Series):
            metrics_frame = metrics.to_frame().T
        else:
            metrics_frame = pd.DataFrame(metrics).copy()
        if averaging == "micro":
            group_series = None
        else:
            group_series = _coerce_group_labels(group_labels, metrics_frame.index)

    if "hierarchical_accuracy" in metrics_frame and "hierarchical_jaccard" not in metrics_frame:
        metrics_frame["hierarchical_jaccard"] = metrics_frame["hierarchical_accuracy"]
    if "hierarchical_jaccard" in metrics_frame and "hierarchical_accuracy" not in metrics_frame:
        metrics_frame["hierarchical_accuracy"] = metrics_frame["hierarchical_jaccard"]

    missing_columns = [
        column for column in HIERARCHICAL_METRIC_COLUMNS if column not in metrics_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Hierarchical metric frame is missing required columns: {missing_columns}"
        )

    micro_summary = metrics_frame.loc[:, HIERARCHICAL_METRIC_COLUMNS].mean()
    micro_summary.name = "micro"
    if averaging == "micro":
        return micro_summary

    if group_series is None:
        raise ValueError("Pass 'group_labels' when requesting macro or both averaging from a metrics frame.")

    macro_summary = (
        metrics_frame.loc[:, HIERARCHICAL_METRIC_COLUMNS]
        .groupby(group_series)
        .mean()
        .mean(axis=0)
    )
    macro_summary.name = "macro"
    if averaging == "macro":
        return macro_summary

    return pd.DataFrame([micro_summary, macro_summary])


@lru_cache(maxsize=None)
def _load_child_to_parent_dag(obofile: Path) -> nx.DiGraph:
    path = Path(obofile)
    if not path.exists():
        raise FileNotFoundError(f"Ontology file not found: {path}")

    if path.suffix.lower() == ".owl":
        graph, metadata = _load_child_to_parent_dag_from_owl(path)
    else:
        graph, metadata = _load_child_to_parent_dag_from_obo(path)

    graph.graph["orientation"] = "child_to_parent"
    graph.graph["source_obofile"] = str(path)
    graph.graph.update(metadata)
    return graph


def _load_child_to_parent_dag_from_obo(path: Path) -> tuple[nx.DiGraph, dict[str, object]]:
    terms = list(_iter_obo_terms(path))
    id_to_name = {
        term["id"]: term["name"]
        for term in terms
        if term["id"].startswith("CL:") and term["name"] and not term["is_obsolete"]
    }
    synonyms = {
        term["id"]: list(term.get("synonyms", []))
        for term in terms
        if term["id"] in id_to_name and term.get("synonyms")
    }
    graph = nx.DiGraph()
    for term_name in id_to_name.values():
        graph.add_node(term_name)

    for term in terms:
        child_id = term["id"]
        if child_id not in id_to_name:
            continue
        child_name = id_to_name[child_id]
        for parent_id in term["is_a"]:
            parent_name = id_to_name.get(parent_id)
            if parent_name is not None:
                graph.add_edge(child_name, parent_name)

    metadata = _build_lookup_metadata(id_to_name, synonyms)
    return graph, metadata


def _load_child_to_parent_dag_from_owl(path: Path) -> tuple[nx.DiGraph, dict[str, object]]:
    tree = ET.parse(path)
    root = tree.getroot()

    rdf = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
    rdfs = "{http://www.w3.org/2000/01/rdf-schema#}"
    owl = "{http://www.w3.org/2002/07/owl#}"

    id_to_name: dict[str, str] = {}
    synonyms: dict[str, list[str]] = {}
    child_to_parents: dict[str, list[str]] = {}

    for elem in root.iter(f"{owl}Class"):
        about = elem.get(f"{rdf}about", "")
        if not about or "CL_" not in about:
            continue

        child_id = about.split("/")[-1].replace("_", ":")
        label = None
        syns: list[str] = []
        parents: list[str] = []

        for child in elem:
            if child.tag == f"{rdfs}label" and child.text:
                label = child.text
            elif "Synonym" in child.tag and child.text:
                syns.append(child.text)
            elif child.tag == f"{rdfs}subClassOf":
                resource = child.get(f"{rdf}resource", "")
                if resource and "CL_" in resource:
                    parents.append(resource.split("/")[-1].replace("_", ":"))

        if label:
            id_to_name[child_id] = label
            if syns:
                synonyms[child_id] = syns
            if parents:
                child_to_parents[child_id] = parents

    graph = nx.DiGraph()
    for term_name in id_to_name.values():
        graph.add_node(term_name)

    for child_id, parents in child_to_parents.items():
        child_name = id_to_name.get(child_id)
        if child_name is None:
            continue
        for parent_id in parents:
            parent_name = id_to_name.get(parent_id)
            if parent_name is not None:
                graph.add_edge(child_name, parent_name)

    metadata = _build_lookup_metadata(id_to_name, synonyms)
    return graph, metadata


def _build_lookup_metadata(
    id_to_name: dict[str, str],
    synonyms: dict[str, list[str]],
) -> dict[str, object]:
    name_to_id = {name: clid for clid, name in id_to_name.items()}
    normalized_name_to_name: dict[str, str] = {}
    ambiguous_normalized_names: dict[str, set[str]] = {}

    for canonical_name in id_to_name.values():
        key = canonical_name.strip().casefold()
        existing = normalized_name_to_name.get(key)
        if existing is None:
            normalized_name_to_name[key] = canonical_name
        elif existing != canonical_name:
            ambiguous_normalized_names.setdefault(key, {existing}).add(canonical_name)

    alias_to_name: dict[str, str] = {}
    for clid, syns in synonyms.items():
        canonical_name = id_to_name[clid]
        for syn in syns:
            alias_to_name.setdefault(_normalize_alias(syn), canonical_name)

    root_name = id_to_name.get("CL:0000000", "cell")
    return {
        "id_to_name": id_to_name,
        "name_to_id": name_to_id,
        "alias_to_name": alias_to_name,
        "normalized_name_to_name": normalized_name_to_name,
        "ambiguous_normalized_names": ambiguous_normalized_names,
        "root_name": root_name,
    }


def _iter_obo_terms(path: Path):
    current: dict[str, object] | None = None

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line == "[Term]":
                if current is not None and current.get("id"):
                    yield current
                current = {
                    "id": "",
                    "name": "",
                    "is_a": [],
                    "synonyms": [],
                    "is_obsolete": False,
                }
                continue

            if line.startswith("[") and line.endswith("]"):
                if current is not None and current.get("id"):
                    yield current
                current = None
                continue

            if current is None:
                continue

            if line.startswith("id: "):
                current["id"] = line[4:]
            elif line.startswith("name: "):
                current["name"] = line[6:]
            elif line.startswith("is_a: "):
                parent_id = line[6:].split()[0]
                current["is_a"].append(parent_id)
            elif line.startswith("synonym: "):
                parts = line.split('"')
                if len(parts) >= 3:
                    current["synonyms"].append(parts[1])
            elif line == "is_obsolete: true":
                current["is_obsolete"] = True

    if current is not None and current.get("id"):
        yield current


def _resolve_matching_dag(
    *, obofile: str | Path | None, dag: nx.DiGraph | None
) -> nx.DiGraph:
    if dag is None and obofile is None:
        raise ValueError("Pass either 'obofile' or 'dag'.")
    if dag is None:
        return load_cell_ontology(obofile)

    orientation = dag.graph.get("orientation")
    if orientation == "child_to_parent":
        converted = dag.reverse(copy=True)
        converted.graph["orientation"] = "parent_to_child"
        return converted
    return dag


def _coerce_pair_inputs(
    pred: Iterable[str] | pd.Series | str,
    truth: Iterable[str] | pd.Series | str,
) -> tuple[pd.Series, pd.Series, bool]:
    pred_is_scalar = _is_scalar_label(pred)
    truth_is_scalar = _is_scalar_label(truth)
    if pred_is_scalar != truth_is_scalar:
        raise ValueError("Prediction and ground-truth inputs must both be scalar or both be iterable.")

    preserve_index = (
        isinstance(pred, pd.Series)
        and isinstance(truth, pd.Series)
        and pred.index.equals(truth.index)
    )

    pred_values = [pred] if pred_is_scalar else list(pred)
    truth_values = [truth] if truth_is_scalar else list(truth)

    pred_series = pd.Series(pred_values, name="prediction")
    truth_series = pd.Series(truth_values, name="ground_truth")

    if len(pred_series) != len(truth_series):
        raise ValueError(
            f"Prediction and ground-truth lengths differ: {len(pred_series)} != {len(truth_series)}"
        )

    if preserve_index:
        pred_series.index = pred.index
        truth_series.index = truth.index

    return pred_series, truth_series, pred_is_scalar and truth_is_scalar


def _coerce_category_input(categories: Iterable[str] | pd.Series | str) -> pd.Series:
    if _is_scalar_label(categories):
        return pd.Series([categories], name="ontology_match")
    return pd.Series(list(categories), name="ontology_match")


def _summarize_ontology_matches_micro(
    category_series: pd.Series,
    *,
    normalize: bool,
    display_labels: bool,
) -> pd.Series:
    counts = category_series.value_counts(normalize=normalize)
    index = DISPLAY_CATEGORY_ORDER if display_labels else RAW_CATEGORY_ORDER
    values = []
    for raw_label in RAW_CATEGORY_ORDER:
        values.append(counts.get(raw_label, 0.0 if normalize else 0))

    summary = pd.Series(values, index=index)
    summary.name = "fraction" if normalize else "count"
    return summary


def _validate_averaging_mode(averaging: str) -> str:
    if averaging not in {"micro", "macro", "both"}:
        raise ValueError("averaging must be one of: 'micro', 'macro', 'both'.")
    return averaging


def _coerce_group_labels(
    group_labels: Iterable[str] | pd.Series | str | None,
    index: pd.Index,
) -> pd.Series:
    if group_labels is None:
        raise ValueError("Pass 'group_labels' when requesting macro or both averaging.")

    if _is_scalar_label(group_labels):
        series = pd.Series([group_labels], index=index)
    else:
        values = list(group_labels)
        if len(values) != len(index):
            raise ValueError(
                f"Grouping label length differs from metric length: {len(values)} != {len(index)}"
            )
        series = pd.Series(values, index=index)

    if series.isna().any():
        raise ValueError("Grouping labels must not contain missing values.")
    return series


def _inclusive_ancestor_set(dag: nx.DiGraph, node: str) -> frozenset[str]:
    cache = dag.graph.setdefault("_inclusive_ancestor_cache", {})
    cached = cache.get(node)
    if cached is not None:
        return cached

    ancestors = frozenset(nx.ancestors(dag, node) | {node})
    cache[node] = ancestors
    return ancestors


def _node_depth(dag: nx.DiGraph, node: str, *, root: str | None = None) -> int:
    root = root or dag.graph.get("root_name", "cell")
    cache = dag.graph.setdefault("_node_depth_cache", {})
    key = (root, node)
    cached = cache.get(key)
    if cached is not None:
        return cached

    try:
        depth = nx.shortest_path_length(dag, source=root, target=node)
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        raise ValueError(
            f"Ontology node {node!r} is not connected to root {root!r} in the DAG."
        ) from exc

    cache[key] = depth
    return depth


def _deepest_common_ancestor_depth(
    dag: nx.DiGraph,
    pred_label: str,
    truth_label: str,
    *,
    root: str | None = None,
) -> int:
    common_ancestors = _inclusive_ancestor_set(dag, pred_label) & _inclusive_ancestor_set(
        dag, truth_label
    )
    if not common_ancestors:
        raise ValueError(
            f"Prediction {pred_label!r} and ground truth {truth_label!r} have no common ancestors."
        )

    return max(_node_depth(dag, node, root=root) for node in common_ancestors)


def _canonicalize_labels(labels: pd.Series, dag: nx.DiGraph, *, role: str) -> pd.Series:
    id_to_name = dag.graph.get("id_to_name", {})
    alias_to_name = dag.graph.get("alias_to_name", {})
    normalized_name_to_name = dag.graph.get("normalized_name_to_name", {})
    ambiguous_normalized_names = dag.graph.get("ambiguous_normalized_names", {})

    normalized = []
    for label in labels:
        if pd.isna(label):
            normalized.append(label)
            continue

        if label in dag:
            normalized.append(label)
            continue

        if not isinstance(label, str):
            normalized.append(label)
            continue

        stripped = label.strip()
        if stripped in dag:
            normalized.append(stripped)
            continue

        if stripped in id_to_name:
            normalized.append(id_to_name[stripped])
            continue

        terminal_id = _extract_terminal_ontology_id(stripped)
        if terminal_id is not None and terminal_id in id_to_name:
            normalized.append(id_to_name[terminal_id])
            continue

        alias_key = _normalize_alias(stripped)
        if alias_key in alias_to_name:
            normalized.append(alias_to_name[alias_key])
            continue

        key = stripped.casefold()
        ambiguous_matches = ambiguous_normalized_names.get(key)
        if ambiguous_matches:
            raise ValueError(
                f"{role} label {label!r} matches multiple ontology nodes by case-insensitive lookup: "
                f"{sorted(ambiguous_matches)}."
            )

        normalized.append(normalized_name_to_name.get(key, label))

    return pd.Series(normalized, index=labels.index, name=labels.name)


def _normalize_alias(label: str) -> str:
    return label.strip().casefold().replace("_", " ").replace("-", " ")


def _extract_terminal_ontology_id(label: str) -> str | None:
    match = TERMINAL_ONTOLOGY_ID_PATTERN.search(label.strip())
    if match is None:
        return None
    return match.group("ontology_id")


def _is_scalar_label(value: object) -> bool:
    if isinstance(value, (str, bytes)):
        return True
    try:
        iter(value)
    except TypeError:
        return True
    return False


def _validate_labels(labels: pd.Series, dag: nx.DiGraph, *, role: str) -> None:
    missing = []
    suggestions = []
    id_to_name = dag.graph.get("id_to_name", {})
    normalized_name_to_name = dag.graph.get("normalized_name_to_name", {})
    alias_to_name = dag.graph.get("alias_to_name", {})

    for label in pd.unique(labels):
        if pd.isna(label):
            missing.append("<NA>")
            continue
        if label in dag:
            continue

        missing.append(str(label))
        if isinstance(label, str):
            stripped = label.strip()
            if stripped in id_to_name:
                suggestions.append(f"{label!r} -> {id_to_name[stripped]!r}")
            else:
                terminal_id = _extract_terminal_ontology_id(stripped)
                if terminal_id is not None and terminal_id in id_to_name:
                    suggestions.append(f"{label!r} -> {id_to_name[terminal_id]!r}")
                else:
                    normalized = normalized_name_to_name.get(stripped.casefold())
                    if normalized is not None and normalized != label:
                        suggestions.append(f"{label!r} -> {normalized!r}")
                    else:
                        alias = alias_to_name.get(_normalize_alias(stripped))
                        if alias is not None and alias != label:
                            suggestions.append(f"{label!r} -> {alias!r}")

    if not missing:
        return

    message = f"{role} labels not found in ontology DAG: {sorted(missing)}."
    if suggestions:
        message += f" Possible ontology resolutions: {sorted(set(suggestions))}."
    raise ValueError(message)
