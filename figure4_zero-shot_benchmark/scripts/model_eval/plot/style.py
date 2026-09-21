"""Locked Figure-4 plotting conventions (Lancet-extended palette, 2026-06-21)."""

MODEL_ORDER = ["Hector", "scimilarity", "scGPT", "scCello", "geneformer"]

MODEL_DISPLAY = {
    "Hector": "HECTOR", "scimilarity": "SCimilarity", "scGPT": "scGPT",
    "scCello": "scCello", "geneformer": "Geneformer",
}

MODEL_COLOR = {
    "Hector": "#FB8072",        # HECTOR (PPR) — light salmon-red
    "scimilarity": "#925E9F",   # purple
    "scGPT": "#0099B4",         # Lancet teal-blue
    "scCello": "#00468B",       # Lancet deep blue
    "geneformer": "#42B540",    # Lancet green
    "OnClass": "#ADB6B6",       # Lancet gray — native-decoder comparison only,
                                 # not part of the five-model embedding roster
                                 # above, so not added to MODEL_ORDER
}

# HECTOR's three arms. All are red-family, so the reader sees at a glance which
# bars and lines are ours; the three shades separate what is being asked of the
# model. The old name for the open arm's colour was HECTOR_PPR, which said the
# opposite of what it marks — the open arm never touches the shared read-out.
HECTOR_NATIVE = "#ED0000"       # own head, choosing among the 66 held-out types
HECTOR_NATIVE_OPEN = "#FB8072"  # own head, choosing among all ~1,400 terms
HECTOR_EMBEDDING = "#C97B72"    # HECTOR's embedding through the shared read-out
HECTOR_PPR = HECTOR_NATIVE_OPEN  # deprecated alias, kept for older scripts

# The true cell type, wherever the figure marks one: the gold star at the centre
# of every panel-e ring, and the query cell in panel d's inset. One definition,
# because a reader who learns the mark in one panel should not have to learn it
# again in the other. Distance from the truth is a separate scale and stays
# blue -- see the node colours in panels/hop.py.
TRUTH_FILL = "#FFB300"
TRUTH_EDGE = "#1A1A1A"

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 1.0, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 350, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "pdf.fonttype": 42, "svg.fonttype": "none",
}

# Full-page reference: retain the entire 3:4 canvas when exporting/inserting.
# Text is proportional to page width, independent of the occupied plot area.
# At this insertion size, every full-page PNG has the same 600-PPI resolution.
# Supplements retain their own styles until they receive a separate review.
MAIN_WIDTH_IN = 180 / 25.4
MAIN_HEIGHT_IN = 240 / 25.4
MAIN_EXPORT_DPI = 600
MAIN_TEXT = MAIN_WIDTH_IN * 72 / 85
MAIN_SECONDARY = 0.9 * MAIN_TEXT
MAIN_EMPHASIS = 1.2 * MAIN_TEXT
MAIN_PANEL_LETTER = 1.8 * MAIN_TEXT
