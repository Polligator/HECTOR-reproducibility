"""Cell Ontology (CL) OBO parsing.

All that remains of a module that used to build an ontology tree and compare it
against the confusion-derived Ward clustering (Baker's gamma, cophenetic
correlation, tanglegram crossings). That comparison was dropped: it scored two
trees built on incomparable notions of distance, over only the shared leaf
vocabulary and the cells that survived that filter, and Baker's gamma reads near
zero for perfectly reasonable trees.

The deep-dive uses this parser to resolve each ontology ID to its canonical name.
Panel D does its own parsing in panel_d_flow.parse_cell_ontology, which is kept
separate on purpose so that panel stays self-contained.
"""

from __future__ import annotations

import io
import re
from collections import defaultdict


def parse_obo_file(file_content):
    """Parse OBO file content to extract term IDs, names, and is_a relationships."""
    relationships = defaultdict(list)
    names = {}
    obsolete_terms = set()
    current_term_id = None
    obo_file = io.StringIO(file_content)
    for line in obo_file:
        line = line.strip()
        if not line:
            continue
        if line == "[Term]":
            current_term_id = None
        elif line.startswith("id:") and not current_term_id:
            current_term_id = line.split("id:", 1)[1].strip()
        elif line.startswith("name:") and current_term_id:
            names[current_term_id] = line.split("name:", 1)[1].strip()
        elif line.startswith("is_a:") and current_term_id:
            match = re.match(r"is_a:\s*(\S+)", line)
            if match:
                relationships[current_term_id].append(match.group(1))
        elif line.startswith("is_obsolete: true") and current_term_id:
            obsolete_terms.add(current_term_id)
        elif line.startswith("[Typedef]"):
            current_term_id = None

    final_relationships = defaultdict(list)
    final_names = {tid: name for tid, name in names.items() if tid not in obsolete_terms}
    for child_id, parent_ids in relationships.items():
        if child_id in final_names:
            valid_parents = [p_id for p_id in parent_ids if p_id in final_names]
            if valid_parents:
                final_relationships[child_id].extend(valid_parents)

    return final_relationships, final_names, obsolete_terms


