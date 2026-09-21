"""Figure-4 supplements. Each module exposes ``render(out_dir)``.

Three figures and one table, down from seven figures on 2026-08-14:

  landmark_space    was S1 (scIB table) + S3 (the two UMAP rows) + S7 (timing),
                    three views of one subject
  per_type          was S4
  depth             was S5

S6, the vocabulary listing, was a list drawn as a figure and is now a
supplementary table supplied with the manuscript. It is not rendered from here.

S2, four scatter panels of five points each, was cut: its fourth panel was the
mean of the other three, its correlations do not reach significance at that
size, and it was the only place a neighbourhood ROC-AUC read-out appeared.

Manuscript numbering is fixed by the manuscript, not here.
"""
from __future__ import annotations

from importlib import import_module

from ..paths import RESULT_DIR

_SUPPLEMENT_MODULES = ["landmark_space", "per_type", "depth"]


def render_all(out_dir=RESULT_DIR) -> dict:
    """Render every supplement to ``out_dir``; never raise, report per-supplement."""
    results = {}
    for name in _SUPPLEMENT_MODULES:
        try:
            module = import_module(f"{__name__}.{name}")
            results[name] = str(module.render(out_dir))
        except Exception as exc:  # noqa: BLE001 — one bad supplement shouldn't kill the rest
            results[name] = f"FAILED: {type(exc).__name__}: {exc}"
    return results
