#!/usr/bin/env python3
"""Cell Ontology utilities for scCello-style Personalized PageRank scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_OBO = _SCRIPT_DIR.parents[1] / "input" / "cl.obo"


class CellOntologyPPR:
    """Cell Ontology Personalized PageRank for cell type similarity.

    Uses a pinned local ontology file so runs do not depend on network access.
    Labels are canonicalized to CL IDs; names are retained as display
    metadata. The default source is the local ``cl.obo`` used by the
    hierarchical metrics code so both benchmark stages stay aligned.
    """

    ALPHA = 0.9
    THRESHOLD = 1e-4

    def __init__(self, ontology_path: str | Path | None = None):
        self.ontology_path = Path(ontology_path or _DEFAULT_OBO).expanduser().resolve()
        self.G: Optional[nx.Graph] = None
        self.cell_types_onto: dict[str, str] = {}
        self.synonyms: dict[str, list[str]] = {}
        self.label_to_id: dict[str, str] = {}
        self.parents_by_id: dict[str, list[str]] = {}
        self.children_by_id: dict[str, list[str]] = {}
        self._candidate_to_bridge_cache: dict[tuple[tuple[str, ...], tuple[str, ...], float], np.ndarray] = {}

    def load_ontology(self) -> None:
        """Load and parse the pinned local ontology."""
        if not self.ontology_path.exists():
            raise FileNotFoundError(
                f"Cell ontology file not found: {self.ontology_path}"
            )

        suffix = self.ontology_path.suffix.lower()
        if suffix == ".obo":
            self._parse_obo(self.ontology_path)
        elif suffix == ".owl":
            self._parse_owl(self.ontology_path)
        else:
            raise ValueError(
                f"Unsupported ontology format for PPR: {self.ontology_path}"
            )

        self.label_to_id = {}
        for onto_id, label in self.cell_types_onto.items():
            self.label_to_id[self._normalize(label)] = onto_id
        for onto_id, syns in self.synonyms.items():
            for syn in syns:
                norm = self._normalize(syn)
                self.label_to_id.setdefault(norm, onto_id)

        print(
            f"Loaded local Cell Ontology from {self.ontology_path} "
            f"with {len(self.cell_types_onto)} terms and "
            f"{self.G.number_of_edges()} edges"
        )

    def _parse_owl(self, owl_file: Path) -> None:
        """Parse OWL/XML to extract CL IDs, primary labels, synonyms, and edges."""
        tree = ET.parse(owl_file)
        root = tree.getroot()

        self.G = nx.Graph()
        self.cell_types_onto = {}
        self.synonyms = {}
        self.parents_by_id = {}
        self.children_by_id = {}
        self._candidate_to_bridge_cache.clear()

        rdf = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
        rdfs = "{http://www.w3.org/2000/01/rdf-schema#}"
        owl = "{http://www.w3.org/2002/07/owl#}"

        for elem in root.iter(f"{owl}Class"):
            about = elem.get(f"{rdf}about", "")
            if not about or "CL_" not in about:
                continue

            raw_id = about.split("/")[-1]
            cell_id = raw_id.replace("_", ":")

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
                        parent_id = resource.split("/")[-1].replace("_", ":")
                        parents.append(parent_id)
                        self.G.add_edge(parent_id, cell_id)

            if label:
                self.cell_types_onto[cell_id] = label
                self.G.add_node(cell_id, label=label)
                if syns:
                    self.synonyms[cell_id] = syns
                self.parents_by_id[cell_id] = parents
                for parent_id in parents:
                    self.children_by_id.setdefault(parent_id, []).append(cell_id)
                self.children_by_id.setdefault(cell_id, [])

    def _parse_obo(self, obo_file: Path) -> None:
        """Parse OBO to extract CL IDs, primary labels, synonyms, and is_a edges."""
        self.G = nx.Graph()
        self.cell_types_onto = {}
        self.synonyms = {}
        self.parents_by_id = {}
        self.children_by_id = {}
        self._candidate_to_bridge_cache.clear()

        child_to_parents: dict[str, list[str]] = {}
        for term in self._iter_obo_terms(obo_file):
            cell_id = term["id"]
            label = term["name"]
            if (
                not cell_id.startswith("CL:")
                or not label
                or term.get("is_obsolete", False)
            ):
                continue

            self.cell_types_onto[cell_id] = label
            self.G.add_node(cell_id, label=label)

            syns = list(term.get("synonyms", []))
            if syns:
                self.synonyms[cell_id] = syns

            child_to_parents[cell_id] = [
                parent_id
                for parent_id in term.get("is_a", [])
                if isinstance(parent_id, str) and parent_id.startswith("CL:")
            ]
            self.parents_by_id[cell_id] = list(child_to_parents[cell_id])
            self.children_by_id.setdefault(cell_id, [])

        for child_id, parents in child_to_parents.items():
            for parent_id in parents:
                if parent_id in self.cell_types_onto:
                    self.G.add_edge(parent_id, child_id)
                    self.children_by_id.setdefault(parent_id, []).append(child_id)

    def _iter_obo_terms(self, path: Path):
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
                    current["is_a"].append(line[6:].split()[0])
                elif line.startswith("synonym: "):
                    parts = line.split('"')
                    if len(parts) >= 3:
                        current["synonyms"].append(parts[1])
                elif line == "is_obsolete: true":
                    current["is_obsolete"] = True

        if current is not None and current.get("id"):
            yield current

    @staticmethod
    def _normalize(value: str) -> str:
        return value.lower().replace("_", " ").replace("-", " ").strip()

    def get_name(self, ontology_id: str) -> str:
        """Return the canonical display name for a CL ID."""
        return self.cell_types_onto.get(ontology_id, ontology_id)

    def get_parents(self, ontology_id: str) -> list[str]:
        """Return the immediate CL-ID parents for *ontology_id*."""
        return list(self.parents_by_id.get(ontology_id, []))

    def get_children(self, ontology_id: str) -> list[str]:
        """Return the immediate CL-ID children for *ontology_id*."""
        return list(self.children_by_id.get(ontology_id, []))

    def resolve_term(self, label_or_id: str, threshold: float = 0.95) -> Optional[tuple[str, str, float]]:
        """Resolve a CL ID or label/synonym to ``(id, name, score)``."""
        if not isinstance(label_or_id, str):
            return None

        stripped = label_or_id.strip()
        if stripped in self.cell_types_onto:
            return stripped, self.cell_types_onto[stripped], 1.0

        norm = self._normalize(stripped)
        if norm in self.label_to_id:
            onto_id = self.label_to_id[norm]
            return onto_id, self.cell_types_onto[onto_id], 1.0

        best_match = None
        best_score = 0.0
        for label_norm, onto_id in self.label_to_id.items():
            if norm in label_norm or label_norm in norm:
                score = len(norm) / max(len(norm), len(label_norm))
                if score > best_score:
                    best_match = onto_id
                    best_score = score
        if best_match and best_score >= threshold:
            return best_match, self.cell_types_onto[best_match], best_score
        return None

    def personalized_pagerank(
        self,
        seed_nodes: list[str],
        alpha: float = 0.9,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict[str, float]:
        """Run Personalized PageRank on the undirected Cell Ontology graph."""
        personalization = {
            seed: 1.0 / len(seed_nodes) for seed in seed_nodes if seed in self.G
        }
        if not personalization:
            return {}

        return nx.pagerank(
            self.G,
            alpha=alpha,
            personalization=personalization,
            max_iter=max_iter,
            tol=tol,
        )

    def compute_candidate_to_bridge_similarity(
        self,
        candidate_ids: list[str],
        bridge_ids: list[str],
        alpha: float = 0.9,
    ) -> np.ndarray:
        """Compute raw PPR similarity from candidate labels to bridge labels."""
        if self.G is None:
            raise RuntimeError("Call load_ontology() before computing PPR similarity.")

        candidate_ids = [str(candidate_id) for candidate_id in candidate_ids]
        bridge_ids = [str(bridge_id) for bridge_id in bridge_ids]
        cache_key = (tuple(candidate_ids), tuple(bridge_ids), float(alpha))
        cached = self._candidate_to_bridge_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        s_candidate = np.zeros((len(candidate_ids), len(bridge_ids)), dtype=np.float32)

        for i, candidate_id in enumerate(candidate_ids):
            if candidate_id not in self.G:
                continue
            pr = self.personalized_pagerank([candidate_id], alpha=alpha)
            for j, bridge_id in enumerate(bridge_ids):
                if bridge_id in pr:
                    s_candidate[i, j] = pr[bridge_id]

        self._candidate_to_bridge_cache[cache_key] = s_candidate
        return s_candidate.copy()

    def resolve_labels_to_ids(
        self,
        labels: list[str],
        *,
        strict: bool = True,
        threshold: float = 0.95,
    ) -> list[Optional[str]]:
        """Resolve a list of labels/IDs to ontology IDs."""
        resolved: list[Optional[str]] = []
        missing: list[str] = []
        for label in labels:
            if label is None:
                resolved.append(None)
                continue
            result = self.resolve_term(label, threshold=threshold)
            if result is None:
                resolved.append(None)
                missing.append(str(label))
            else:
                resolved.append(result[0])

        if strict and missing:
            raise ValueError(
                f"Could not resolve {len(missing)} labels to Cell Ontology IDs: "
                f"{missing[:10]}"
            )
        return resolved


def normalize_ppr_rows(candidate_to_bridge: np.ndarray) -> np.ndarray:
    """Divide each candidate's row by its own total across the tracked bridges.

    Raw PPR mass favours candidates sitting near well-connected ontology hubs
    regardless of the bridge being compared: their mass is large everywhere,
    not just near a true match. Rescaling each row by its own total puts every
    candidate on the same 0-1 scale before columns are compared, so a
    candidate is judged against how it usually scores, not against candidates
    whose raw scale is unrelated. This does not require re-running PPR; it
    only rescales values `compute_candidate_to_bridge_similarity` already
    returned, so callers comparing across candidates should apply this before
    taking a column-wise winner, rather than comparing raw mass directly.
    """
    matrix = np.asarray(candidate_to_bridge, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1.0, row_sums)
    return matrix / safe_sums


def resolve_column_winners(
    matrix: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, dict[frozenset[int], list[int]]]:
    """Per-bridge winning candidate, with exact ties split across the tied rows.

    Returns (winners, tie_groups): winners[j] is the row index that wins
    column j; tie_groups maps the exact set of tied row indices to the list
    of columns where they tied (empty when there are no ties).

    Ties are detected by exact float equality. That is the correct test here,
    not merely a convenient one: a tie between two rows can only arise when
    two candidates are exact structural mirror images in the ontology graph
    (identical connections, so the same deterministic PPR walk produces
    bit-identical values for both), never from floating-point noise between
    otherwise-unrelated quantities. A tolerance-based test would risk treating
    two genuinely different, merely-close candidates as tied and discarding
    real ranking information between them.

    Tied columns are split across the tied rows by a permutation seeded from
    (seed, the sorted tied row indices) -- not from row/candidate order, which
    would otherwise silently and systematically favour whichever tied
    candidate's ID happens to sort first, every time the same pair ties. Each
    tie group's resolution is independent of any other tie group and of how
    many bridges are being scored, and is exactly reproducible for a fixed
    seed. Nothing assumes there is only ever one tied pair, or that a tied
    group has only two members.
    """
    matrix = np.asarray(matrix)
    n_bridge = matrix.shape[1]
    winners = np.empty(n_bridge, dtype=np.int64)
    col_max = matrix.max(axis=0)

    tie_groups: dict[frozenset[int], list[int]] = {}
    for col in range(n_bridge):
        tied = np.flatnonzero(matrix[:, col] == col_max[col])
        if tied.size == 1:
            winners[col] = tied[0]
        else:
            tie_groups.setdefault(frozenset(tied.tolist()), []).append(col)

    for tied_rows, cols in tie_groups.items():
        rows_sorted = sorted(tied_rows)
        rng = np.random.default_rng([seed, *rows_sorted])
        ordered_rows = rng.permutation(rows_sorted)
        for position, col in enumerate(sorted(cols)):
            winners[col] = ordered_rows[position % len(ordered_rows)]

    return winners, tie_groups
