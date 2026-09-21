"""Padding for query cells a model never returned a prediction for.

Shared by the accuracy panel (``plot/panels/accuracy.py``) and the summary-table
builder (``evaluate.summarize_prediction_table``) so the two cannot silently
diverge on how a missing cell is scored. They already had, once: scCello's
headline accuracy in ``summary_table.csv`` was computed over only the 20,360 of
33,000 query cells its tokenizer kept, while the same number in ``figure_4.pdf``
was computed over all 33,000, through this same padding, in the panel alone.

A cell a model will not process is a cell it has not annotated, so it is
counted as an error in every metric here: every column in
``HIERARCHICAL_METRIC_COLUMNS`` reads 0.0 for a padded row — including
``lca_depth``, whose 0 means only the ontology root is shared, the least
specific match the metric can express — and ``predicted_cell_type_id`` is set
to ``UNSCORED`` so it never collides with a real prediction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_DIR.parent          # scripts/, which carries Hierarchical_metrics
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from Hierarchical_metrics.cell_ontology_metrics import HIERARCHICAL_METRIC_COLUMNS  # noqa: E402

# Marks a query cell a model never returned a prediction for.
UNSCORED = "__not_returned__"

_PAD_VALUES = {column: 0.0 for column in HIERARCHICAL_METRIC_COLUMNS}


def expected_query_counts_from_table(candidate_label_table: pd.DataFrame) -> pd.Series:
    """How many query cells each held-out cell type contributes, by CL ID."""
    ids = candidate_label_table["cell_type_id"].astype(str)
    return candidate_label_table.set_index(ids)["n_cells"]


def expected_query_counts(data_dir: str | Path) -> pd.Series:
    """Same, read from the ``candidate_label_table.csv`` a benchmark run writes."""
    table = pd.read_csv(Path(data_dir) / "candidate_label_table.csv")
    return expected_query_counts_from_table(table)


def pad_to_query_set(df: pd.DataFrame, expected: pd.Series) -> pd.DataFrame:
    """Add a failed row for every query cell a model did not return.

    scCello's tokenizer discards cells it cannot represent — 12,640 of 33,000
    in the shipped run, taking 21 of the 66 held-out cell types with it
    entirely. Without this, a model's per-cell accuracy would average only
    over the cells it chose to answer, while a macro metric computed
    separately might already be scoring its unreached types as zero — grading
    the same model two different ways from the same predictions.
    """
    have = df.groupby(df["truth_cell_type_id"].astype(str)).size()
    gaps = []
    for cell_type_id, n_expected in expected.items():
        missing = int(n_expected) - int(have.get(cell_type_id, 0))
        if missing <= 0:
            continue
        gap = {
            "truth_cell_type_id": [cell_type_id] * missing,
            "predicted_cell_type_id": [UNSCORED] * missing,
        }
        gap.update({column: [value] * missing for column, value in _PAD_VALUES.items()})
        gaps.append(pd.DataFrame(gap))
    if not gaps:
        return df
    return pd.concat([df, *gaps], ignore_index=True)
