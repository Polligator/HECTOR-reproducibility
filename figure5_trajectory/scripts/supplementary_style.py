"""Page and typography conventions shared with the standardized main figures."""

from matplotlib.text import Text

PAGE_WIDTH_IN = 180 / 25.4
PAGE_HEIGHT_IN = 240 / 25.4
EXPORT_DPI = 600
TEXT_SIZE = PAGE_WIDTH_IN * 72 / 85
SECONDARY_SIZE = 0.9 * TEXT_SIZE
EMPHASIS_SIZE = 1.2 * TEXT_SIZE
PANEL_SIZE = 1.8 * TEXT_SIZE


def standardize_figure(figure):
    """Apply printed-size type roles and strokes before full-canvas export."""
    for text in figure.findobj(match=Text):
        original_size = text.get_fontsize()
        if text.get_text() in {"a", "b", "c", "d"} and original_size >= 14:
            text.set_fontsize(PANEL_SIZE)
        elif original_size >= 9:
            text.set_fontsize(EMPHASIS_SIZE)
        elif original_size < 6:
            text.set_fontsize(SECONDARY_SIZE)
        else:
            text.set_fontsize(TEXT_SIZE)
        text.set_fontfamily("sans-serif")
    for axis in figure.axes:
        axis.tick_params(width=0.6, length=2, pad=2)
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)
        for gridline in [*axis.get_xgridlines(), *axis.get_ygridlines()]:
            gridline.set_linewidth(0.5)
