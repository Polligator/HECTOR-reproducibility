"""Draw how well the model does on each cell type it was trained on.

    python make_figure.py                 human, the default
    python make_figure.py mouse
    python make_figure.py --colour-check   how the fill colours look to a
                                           colour-blind reader, and no figure

Writes ../../result/validation_ontology_projection/<species>/figure.pdf, figure.png,
drawing.html and placement.csv.
The boxes are the Cell Ontology, nested exactly as the training composition
figure nests it; the fill is how well the model does on that cell type.
"""

from pathlib import Path
from io import BytesIO
import os
import re
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

import tree

HERE = Path(__file__).resolve().parent
FIGURE = HERE.parents[1]          # figure1/, holding input/, scripts/ and result/
OUTPUT = FIGURE / "result" / HERE.name

# The headless browser used to save the figure needs a scratch profile below a
# visible directory on an ordinary disk, not a cloud-drive mount.
# HECTOR_RENDER_SCRATCH overrides the list below.
BROWSER_SCRATCH_CHOICES = [OUTPUT / ".render_home",
                           Path.home() / "Desktop" / ".hector_render_scratch"]


def on_an_ordinary_disk(where):
    """Whether a path sits on a real filesystem rather than a mounted cloud drive."""
    mounts = {}
    for line in Path("/proc/mounts").read_text().splitlines():
        parts = line.split()
        if len(parts) > 2:
            mounts[parts[1]] = parts[2]
    walk = where.resolve()
    while True:
        if str(walk) in mounts:
            return not mounts[str(walk)].startswith("fuse")
        if walk.parent == walk:
            return True
        walk = walk.parent


def browser_scratch():
    """Where the browser may keep the scratch profile it needs to draw anything."""
    told = os.environ.get("HECTOR_RENDER_SCRATCH")
    choices = [Path(told)] if told else BROWSER_SCRATCH_CHOICES
    for where in choices:
        if where.parent.exists() and on_an_ordinary_disk(where.parent):
            where.mkdir(parents=True, exist_ok=True)
            return where
    raise RuntimeError(
        "the headless browser that draws this figure needs a scratch folder on an "
        "ordinary disk, below a visible directory in the home directory. None of these "
        "will do: " + ", ".join(str(where) for where in choices)
        + ". Set HECTOR_RENDER_SCRATCH to one that will.")

# ============================================================== size and type

# Full 3:4 working canvas, intended for insertion at the same 180 mm width.
PAGE_WIDTH_MM = 180.0
PAGE_HEIGHT_MM = 240.0
PAGE_MARGIN_MM = 7.0
TILES_HEIGHT_MM = 170.0
DISTRIBUTIONS_HEIGHT_MM = 40.0  # the three score distributions beneath them
GAP_MM = 16.0                   # holds the colour bar
EXPORT_PPI = 600

TYPEFACE = "Arial, Helvetica, Nimbus Sans, sans-serif"
NAME_PT = 6.2        # secondary group names; browser output stays above 6 pt
AXIS_PT = 8.2        # numbers and titles on the score distributions
BAR_PT = AXIS_PT
PANEL_PT = 12.2

# A name is drawn only where its box has room for it at NAME_PT; a small box is
# left blank rather than shrinking the type below the journal's print floor.
NAMES_BELOW_THIS_SIZE_ARE_DROPPED = NAME_PT

# ============================================================== colour

SCORE_FLOOR = None   # worked out from the data, rounded down to a tenth
SCORE_CEILING = 1.0

# Dark at the low end, pale at the top, so the model's worst cell types are the
# dark marks on a pale page. Not red-to-green: that ramp is dark at both ends,
# so a red-blind reader cannot tell a fail from a pass.
SCORE_COLOURS = [
    [0.00, "#3b0f70"], [0.25, "#8c2981"], [0.50, "#de4968"],
    [0.75, "#fe9f6d"], [1.00, "#fde0c8"],
]

# Boxes that only hold other boxes carry no score of their own.
STRUCTURE_FILL = "#ffffff"
STRUCTURE_EDGE = "#9a9a9a"
TILE_EDGE = "#ffffff"

DISTRIBUTION_COLOURS = {"Accuracy": "#4c4c4c", "Precision": "#4c4c4c", "F1": "#4c4c4c"}

# ============================================================== the web version

# Sizes for the browser-opened drawing, in browser pixels, not held to the
# journal's 5 pt print floor: a reader can zoom.
WEB_TYPE_PX = 10
WEB_SMALLEST_TYPE_PX = 7
WEB_BREADCRUMB_PX = 24

# The drawing scrolls vertically only, never horizontally. 1.2x taller than wide
# names 292 of 655 boxes on a 1600 px window (227 at 1:1); past 1.2 the gain
# flattens.
WEB_TALL_AS_A_SHARE_OF_WIDTH = 1.2
WEB_HEIGHT_BEFORE_RESIZING_PX = 1200

MM_PER_POINT = 25.4 / 72.0

# The headless browser writes at 72 points to 96 pixels, so sizes are enlarged
# by a third going in, or the page renders at three-quarters scale.
POINTS_PER_BROWSER_PIXEL = 0.75


def millimetres_to_points(millimetres):
    return millimetres / MM_PER_POINT


def as_browser_pixels(points):
    """A size in points, in the units the browser is given."""
    return points / POINTS_PER_BROWSER_PIXEL


def break_over_lines(name, lines):
    """A name split at spaces into that many lines, as evenly as the words allow."""
    words = name.split()
    if lines <= 1 or len(words) < 2:
        return [name]
    broken, remaining = [], list(words)
    for left in range(lines, 0, -1):
        take = -(-len(remaining) // left)
        broken.append(" ".join(remaining[:take]))
        remaining = remaining[take:]
        if not remaining:
            break
    return broken


def name_that_fits(name, box_width, box_height, holds_others):
    """The name broken to fit inside its box at NAME_PT, or nothing if it cannot.

    A box that holds other boxes has only the strip along its top to write in, so
    its name has to go on one line. A cell type's own box is written across, and
    is nearly square, so a name written on one line asks for a box several times
    wider than it is tall and is turned away for want of width it never needed.
    Broken into a block, the same name asks for a shape the box actually has.
    """
    line_height = NAME_PT * LINE_SPACING
    room_across = box_width - 2 * NAME_INSET_SIDES_PT
    # The header strip is room the first line already has.
    room_down = box_height - header_height() - NAME_INSET_BOTTOM_PT + line_height
    if room_across <= 0 or room_down < line_height:
        return ""

    most = 1 if holds_others else MOST_LINES_PER_NAME
    for lines in range(1, min(most, len(name.split())) + 1):
        broken = break_over_lines(name, lines)
        widest = max(text_width_at_one_point(line) for line in broken) * NAME_PT
        # A box that holds others is written on its header strip only.
        tall = line_height * (1 if holds_others else len(broken))
        if widest <= room_across and tall <= room_down:
            return "<br>".join(broken)
    return ""


def header_height():
    """The strip along the top of every box, which its name is written in."""
    return NAME_PT * LINE_SPACING + HEADER_CLEARANCE_PT


def text_width_at_one_point(text):
    """How wide a line of text is, in points, set at one point."""
    global _ARIAL
    if _ARIAL is None:
        from PIL import ImageFont
        import subprocess
        where = subprocess.run(["fc-match", "-f", "%{file}", "Arial"],
                               capture_output=True, text=True).stdout.strip()
        _ARIAL = ImageFont.truetype(where, 100)
    return _ARIAL.getlength(text) / 100.0


_ARIAL = None

# How far apart the lines of a name sit, as a share of the type size.
LINE_SPACING = 1.15

# A box that holds other boxes is named wherever it has room, however deep it
# sits; a cell type's own box is never named — only 3 of 650 have room for their
# own name at this size, so naming those three would be an arbitrary exception.
# Every box's identity is in the drawing's hover text and in placement.csv.

# How many lines a name may be broken over.
MOST_LINES_PER_NAME = 4

# Fixed here rather than left to the drawing package, so the layout does not
# change between the numbering pass and the final render. 1.2 pt names 64 of the
# boxes that fit a name; below it boxes start to look jammed together.
NAME_INSET_SIDES_PT = 1.2
NAME_INSET_BOTTOM_PT = 1.2

# Extra clearance below the header strip so descenders (g, y) aren't cropped.
HEADER_CLEARANCE_PT = 2.4


def box_geometry(figure, tiles):
    """The size of every box as the drawing actually lays it out, in points.

    The drawing package decides the sizes itself and will not say what it chose,
    so the figure is drawn once with every box named by its number, and the
    numbers are read back off the drawing. Nothing else asks it where a box is;
    guessing, and hoping the guess matched, is how a name ends up outside its own
    box.
    """
    probe = go.Figure(figure)
    probe.data[0].labels = [str(number) for number in range(len(tiles))]
    probe.data[0].textfont.size = 1
    probe.update_layout(uniformtext=dict(minsize=1, mode="show"))

    was_home = os.environ.get("HOME")
    os.environ["HOME"] = str(browser_scratch())
    try:
        drawing = probe.to_image(format="svg").decode()
    finally:
        if was_home is None:
            del os.environ["HOME"]
        else:
            os.environ["HOME"] = was_home

    corners = re.compile(r'<path class="surface" d="M([\d.eE+-]+),([\d.eE+-]+)'
                         r'L([\d.eE+-]+),([\d.eE+-]+)L([\d.eE+-]+),([\d.eE+-]+)')
    sizes = {}
    # Excludes "slicetext" groups, whose class also starts with "slice".
    for chunk in re.split(r'<g class="slice(?!text)', drawing)[1:]:
        found = corners.search(chunk)
        written = "".join(re.findall(r'<text[^>]*class="slicetext"[^>]*>([^<]*)</text>', chunk))
        number = re.sub(r"[^0-9]", "", written)
        if not found or not number:
            continue
        left, top, right, _, _, bottom = (float(value) for value in found.groups())
        sizes[int(number)] = (abs(right - left) * POINTS_PER_BROWSER_PIXEL,
                              abs(bottom - top) * POINTS_PER_BROWSER_PIXEL)

    # A box too small for even a number is recorded as having no room for a name.
    # Losing most boxes this way would mean the reading itself is broken.
    if len(sizes) < 0.8 * len(tiles):
        raise ValueError(f"read only {len(sizes)} boxes back off the drawing of {len(tiles)}")
    return [sizes.get(number, (0.0, 0.0)) for number in range(len(tiles))]


def score_floor(placed=None):
    """The bottom of the colour scale: below every score of either species.

    Worked out from both species together, not from the one being drawn, so a
    colour means the same score on the human figure and the mouse one. Taken
    from each species alone it came out at 0.2 for human and 0.3 for mouse, and
    the same shade of red then stood for two different scores.
    """
    if SCORE_FLOOR is not None:
        return SCORE_FLOOR
    wanted = [tree.TILE_SCORE, *tree.DISTRIBUTION_SCORES]
    lowest = min(tree.load_scores(species)[column].min()
                 for species in tree.SCORE_FILES
                 if (tree.INPUT / tree.SCORE_FILES[species]).exists()
                 for column in wanted)
    return np.floor(lowest * 10) / 10


def draw(species, placed, tiles, names, release):
    """The boxes, the colour bar and the three score distributions, on one page."""
    floor = score_floor(placed)

    page_height_mm = PAGE_HEIGHT_MM
    content_height_mm = PAGE_HEIGHT_MM - 2 * PAGE_MARGIN_MM
    page_width = as_browser_pixels(millimetres_to_points(PAGE_WIDTH_MM))
    page_height = as_browser_pixels(millimetres_to_points(page_height_mm))
    tiles_bottom = (GAP_MM + DISTRIBUTIONS_HEIGHT_MM) / content_height_mm
    distributions_top = DISTRIBUTIONS_HEIGHT_MM / content_height_mm

    fill = [STRUCTURE_FILL if pd.isna(value) else value for value in tiles["score"]]
    # A box that holds other boxes is outlined; a cell type's own tile is
    # separated from its neighbours by white space instead.
    holds_others = tiles["score"].isna()
    edge_colour = np.where(holds_others, STRUCTURE_EDGE, TILE_EDGE)
    edge_width = np.where(holds_others, as_browser_pixels(0.4), as_browser_pixels(0.25))
    # A box that holds others is named on the box around them, so its own tile
    # inside is left blank rather than printing the name twice.
    label = np.where(tiles["node"].astype(str).str.endswith(":itself"), "", tiles["name"])

    figure = go.Figure()
    figure.add_trace(go.Treemap(
        ids=tiles["node"], parents=tiles["inside"], labels=label, values=tiles["size"],
        branchvalues="total",
        pathbar=dict(visible=False),
        root=dict(color="white"),
        tiling=dict(pad=as_browser_pixels(0.9)),
        marker=dict(
            colors=fill, colorscale=SCORE_COLOURS, cmin=floor, cmax=SCORE_CEILING,
            showscale=True,
            pad=dict(t=as_browser_pixels(header_height()),
                     l=as_browser_pixels(NAME_INSET_SIDES_PT),
                     r=as_browser_pixels(NAME_INSET_SIDES_PT),
                     b=as_browser_pixels(NAME_INSET_BOTTOM_PT)),
            line=dict(width=edge_width, color=edge_colour),
            colorbar=dict(
                title=dict(text=f"{tree.TILE_SCORE} on test cells", side="top",
                           font=dict(size=as_browser_pixels(BAR_PT))),
                orientation="h", x=0.5, xanchor="center",
                y=distributions_top + (GAP_MM * 0.55) / content_height_mm,
                yanchor="middle", len=0.34,
                thickness=as_browser_pixels(millimetres_to_points(2.2)),
                tickfont=dict(size=as_browser_pixels(BAR_PT)), outlinewidth=0,
                ticklen=as_browser_pixels(1.5), tickwidth=as_browser_pixels(0.4)),
        ),
        # The score goes in what the drawing says when you point at a box, never
        # on the box: printed on the page it would fill the figure with numbers.
        customdata=np.stack([
            tiles["name"], tiles["node"].astype(str).str.replace(":itself", "", regex=False),
            [f"{score:.3f}" if not pd.isna(score) else "not a cell type in its own right"
             for score in tiles["score"]]], axis=-1),
        hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>"
                      + tree.TILE_SCORE + " %{customdata[2]}<extra></extra>",
        textfont=dict(family=TYPEFACE, size=as_browser_pixels(NAME_PT), color="#111111"),
        textposition="top left",
        domain=dict(x=[0, 1], y=[tiles_bottom, 1]),
    ))

    span = SCORE_CEILING - floor
    for position, metric in enumerate(tree.DISTRIBUTION_SCORES):
        values = placed[metric].to_numpy()
        left = position / len(tree.DISTRIBUTION_SCORES)
        width = 1 / len(tree.DISTRIBUTION_SCORES)
        horizontal, vertical = f"x{position + 2}", f"y{position + 2}"

        figure.add_trace(go.Box(
            y=values, x0=0, name="", width=0.42,
            marker=dict(color=DISTRIBUTION_COLOURS[metric]),
            line=dict(color=DISTRIBUTION_COLOURS[metric], width=as_browser_pixels(0.5)),
            fillcolor="rgba(0,0,0,0)", boxpoints=False, showlegend=False,
            hoverinfo="y", xaxis=horizontal, yaxis=vertical))

        spread = np.random.default_rng(0).uniform(-0.16, 0.16, size=len(values))
        figure.add_trace(go.Scatter(
            x=spread, y=values, mode="markers",
            marker=dict(size=as_browser_pixels(1.8), color=values, colorscale=SCORE_COLOURS,
                        cmin=floor, cmax=SCORE_CEILING, showscale=False,
                        # The palest fills are nearly the colour of the page, so
                        # the dots need an outline to be countable at all.
                        line=dict(width=as_browser_pixels(0.2), color="#8a8a8a")),
            showlegend=False, hovertemplate="%{y:.3f}<extra></extra>",
            xaxis=horizontal, yaxis=vertical))

        figure.update_layout(**{
            f"xaxis{position + 2}": dict(
                domain=[left + 0.06, left + width - 0.06], range=[-0.55, 0.55],
                showticklabels=False, showgrid=False, zeroline=False,
                title=dict(text=metric.replace("_", " "),
                           font=dict(size=as_browser_pixels(AXIS_PT))),
                ticks="", showline=False),
            f"yaxis{position + 2}": dict(
                domain=[0, distributions_top],
                range=[floor - 0.02 * span, SCORE_CEILING + 0.02 * span],
                dtick=0.1, tickformat=".1f", showgrid=True, gridcolor="#e6e6e6",
                gridwidth=as_browser_pixels(0.3), zeroline=False, anchor=horizontal,
                showticklabels=position == 0, ticks="outside",
                ticklen=as_browser_pixels(1.5), tickwidth=as_browser_pixels(0.4),
                showline=True, linewidth=as_browser_pixels(0.4), linecolor="#4c4c4c",
                tickfont=dict(size=as_browser_pixels(AXIS_PT))),
        })

    figure.update_layout(
        width=page_width, height=page_height,
        paper_bgcolor="white", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=as_browser_pixels(millimetres_to_points(PAGE_MARGIN_MM)), l=as_browser_pixels(1),
                    r=as_browser_pixels(1),
                    b=as_browser_pixels(millimetres_to_points(PAGE_MARGIN_MM))),
        font=dict(family=TYPEFACE, size=as_browser_pixels(AXIS_PT), color="#111111"),
        uniformtext=dict(minsize=as_browser_pixels(NAMES_BELOW_THIS_SIZE_ARE_DROPPED),
                         mode="hide"),
    )
    figure.add_annotation(
        text="<b>" + {"human": "a", "mouse": "b"}[species] + "</b>",
        x=0, y=1.002, xref="paper", yref="paper", xanchor="left", yanchor="bottom",
        showarrow=False, font=dict(family=TYPEFACE, size=as_browser_pixels(PANEL_PT)),
    )

    figure = name_the_boxes(figure, tiles)
    return figure, floor


def name_the_boxes(figure, tiles):
    """Name every box whose name fits it, having measured the box first.

    Left to itself the drawing package shrinks a name until it fits and hides it
    below a floor, and the floor it applies is one size for the whole figure — so
    a single cramped box drags every other name down with it, and asking for
    smaller type printed fewer names rather than more. Deciding here, box by box,
    is what makes the type size mean what it says.
    """
    boxes = box_geometry(figure, tiles)
    written = []
    for (width, height), name, node, score in zip(
            boxes, tiles["name"], tiles["node"].astype(str), tiles["score"]):
        holds_others = pd.isna(score)
        if not holds_others or node == tree.EVERYTHING:
            written.append("")
        else:
            written.append(name_that_fits(name, width, height, True))

    figure.data[0].labels = written
    figure.update_layout(uniformtext=dict(minsize=as_browser_pixels(NAME_PT), mode="show"))
    return figure


def draw_for_the_web(species, placed, tiles, names, release):
    """The same drawing, to be opened in a browser rather than printed.

    A page has no edge to run out of, so this one is not sized to 180 mm and
    nothing is left off it for want of room. Every box is given its name and the
    browser writes the ones that fit the window; opening a box gives its contents
    the whole window, and names that were too small to write appear. That is the
    reason for the two versions: the printed one is the look of the thing, and
    this is where a reader goes to find a particular cell type.
    """
    floor = score_floor(placed)
    scored = placed.set_index("cell_type_ontology_term_id")

    fill, told, edge_colour, edge_width, label = [], [], [], [], []
    for name, node, score in zip(tiles["name"], tiles["node"].astype(str), tiles["score"]):
        term = node.replace(":itself", "")
        holds_others = pd.isna(score) and not node.endswith(":itself")
        fill.append(STRUCTURE_FILL if pd.isna(score) else score)
        edge_colour.append(STRUCTURE_EDGE if holds_others else TILE_EDGE)
        edge_width.append(1.0 if holds_others else 0.6)
        label.append("" if node.endswith(":itself") or node == tree.EVERYTHING else name)

        if term in scored.index:
            row = scored.loc[term]
            told.append(
                f"<b>{name}</b><br>{term}<br><br>"
                f"F1 {row['F1']:.3f}<br>"
                f"accuracy {row['Accuracy']:.3f}<br>"
                f"precision {row['Precision']:.3f}<br><br>"
                f"{int(row['Correct']):,} of {int(row['Total']):,} test cells "
                f"of this type were called correctly<br>"
                f"{int(row['cells_used_in_training']):,} cells of it went into training<br>"
                f"drawn inside {row['drawn_inside'] or 'nothing — it is a broad term'}")
        else:
            told.append(f"<b>{name}</b><br>{term}<br><br>"
                        "a broad term, not a cell type the model trained on")

    figure = go.Figure(go.Treemap(
        ids=tiles["node"], parents=tiles["inside"], labels=label, values=tiles["size"],
        branchvalues="total",
        # The pathbar (unlike on the printed figure) is the way back out of an
        # opened box.
        pathbar=dict(visible=True, thickness=WEB_BREADCRUMB_PX,
                     textfont=dict(family=TYPEFACE, size=WEB_TYPE_PX)),
        root=dict(color="white"),
        tiling=dict(pad=1),
        marker=dict(
            colors=fill, colorscale=SCORE_COLOURS, cmin=floor, cmax=SCORE_CEILING,
            showscale=True,
            line=dict(width=edge_width, color=edge_colour),
            pad=dict(t=WEB_TYPE_PX * 1.7, l=3, r=3, b=3),
            colorbar=dict(title=dict(text=f"{tree.TILE_SCORE}", side="top",
                                     font=dict(size=WEB_TYPE_PX)),
                          thickness=14, len=0.4, y=0.5, tickfont=dict(size=WEB_TYPE_PX),
                          outlinewidth=0),
        ),
        customdata=told,
        hovertemplate="%{customdata}<extra></extra>",
        hoverlabel=dict(align="left", font=dict(family=TYPEFACE, size=WEB_TYPE_PX)),
        textfont=dict(family=TYPEFACE, size=WEB_TYPE_PX, color="#111111"),
        textposition="top left",
    ))
    figure.update_layout(
        # Initial size before the page resizes to the actual window; without it,
        # Plotly's 700x450 default draws first and most names are missing.
        autosize=True, height=WEB_HEIGHT_BEFORE_RESIZING_PX,
        margin=dict(t=4, l=4, r=4, b=4),
        paper_bgcolor="white",
        font=dict(family=TYPEFACE, color="#111111"),
        uniformtext=dict(minsize=WEB_SMALLEST_TYPE_PX, mode="hide"),
    )
    return figure


def save_the_web_version(figure, species, placed, release):
    """Write it as a page that fills whatever window it is opened in."""
    folder = OUTPUT / species
    folder.mkdir(parents=True, exist_ok=True)
    where = folder / "drawing.html"

    figure.write_html(
        where,
        # Carried in the file rather than fetched from elsewhere, so the page
        # keeps working wherever it is put and however long it sits there.
        include_plotlyjs=True, full_html=True,
        default_width="100%", default_height="100%",
        config=dict(responsive=True, displaylogo=False,
                    toImageButtonOptions=dict(format="svg", filename=f"{species}_ontology")),
    )

    heading = (
        f"<h1>{species.capitalize()} model: how well HECTOR does on each of the "
        f"{len(placed):,} cell types it was trained on</h1>"
        "<p>The boxes are the Cell Ontology, one box for every cell type. "
        "Colour is the F1 score on cells the model was not trained on: pale where it "
        "does well, dark where it does badly. "
        "<b>Click a box to open it</b> and its contents fill the window, which is how "
        "the smaller cell types get room for their names; the strip along the top is "
        "the way back out. Point at any box for its name, its ontology identifier, "
        "its scores, and how many cells of it the model saw."
        f"<br>Cell Ontology release {release}.</p>")

    page = where.read_text()
    page = page.replace("<body>", f"""<body>
<style>
  html, body {{ margin: 0; font-family: Arial, Helvetica, sans-serif;
                color: #111; overflow-x: hidden; }}
  #heading {{ padding: 10px 14px 6px; }}
  #heading h1 {{ font-size: 17px; font-weight: 600; margin: 0 0 4px; }}
  #heading p {{ font-size: 13px; line-height: 1.45; margin: 0; max-width: 62em;
                color: #333; }}
</style>
<div id="heading">{heading}</div>""", 1)

    # Sized in JS, not CSS: the drawing measures its own box once, at creation,
    # before the page settles, and never rereads it.
    page = page.replace("</body>", """<script>
  var TALL_AS_A_SHARE_OF_WIDTH = %s;
  function sizeTheDrawing() {
    var drawing = document.querySelector('.plotly-graph-div');
    var heading = document.getElementById('heading');
    if (!drawing || !window.Plotly) { return; }
    var wide = window.innerWidth;
    Plotly.relayout(drawing, {
      width: wide,
      height: Math.max(window.innerHeight - heading.offsetHeight - 10,
                       Math.round(wide * TALL_AS_A_SHARE_OF_WIDTH))
    });
  }
  window.addEventListener('resize', sizeTheDrawing);
  window.addEventListener('load', sizeTheDrawing);
  sizeTheDrawing();
</script>
</body>""" % WEB_TALL_AS_A_SHARE_OF_WIDTH, 1)
    where.write_text(page)
    return where


def count_boxes_drawn(figure):
    """How many boxes the drawing actually put on the page, and how many carry a score.

    Counted from what the browser draws rather than from the finished PDF: the
    tools that read a PDF back merge and skip shapes, and reading 233 boxes out
    of a page that holds 863 is what sent this hunting for a fault that was not
    there.
    """
    drawing = figure.to_image(format="svg").decode()
    fills = re.findall(r'class="surface"[^>]*style="[^"]*fill:\s*([^;"]+)', drawing)
    white = sum(1 for fill in fills
                if fill.strip().replace(" ", "") in ("rgb(255,255,255)", "white", "#ffffff"))
    return {"total": len(fills), "white": white, "coloured": len(fills) - white}


def check_and_save(figure, species, placed, tiles, floor):
    """Write the figure, having refused to if it would not say what it means."""
    folder = OUTPUT / species
    folder.mkdir(parents=True, exist_ok=True)

    lowest = min(placed[column].min() for column in tree.DISTRIBUTION_SCORES)
    if lowest < floor:
        raise ValueError(f"the score axis starts at {floor} but a cell type scores "
                         f"{lowest:.3f}, so it would not be drawn")
    if tiles.loc[tiles["inside"] == "", "size"].sum() != len(placed):
        raise ValueError("the boxes no longer add up to every cell type")

    pdf_path = folder / "figure.pdf"
    was_home = os.environ.get("HOME")
    os.environ["HOME"] = str(browser_scratch())
    try:
        # A box squeezed to nothing is dropped without a word by the drawing
        # package, so verify every trained cell type is actually on the page.
        drawn = count_boxes_drawn(figure)
        if drawn["coloured"] != len(placed):
            raise ValueError(f"{drawn['coloured']} cell types were drawn, not {len(placed)}")
        named_cell_types = [name for name, score, written
                            in zip(tiles["name"], tiles["score"], figure.data[0].labels)
                            if written and not pd.isna(score)]
        if named_cell_types:
            raise ValueError("a cell type's own box was named, which the figure does not do: "
                             + ", ".join(named_cell_types[:3]))
        if drawn["white"] != (~tiles["score"].notna()).sum():
            raise ValueError(f"{drawn['white']} boxes hold others, expected "
                             f"{(~tiles['score'].notna()).sum()}")

        figure.write_image(pdf_path, format="pdf")
        # Render native pixels once; attaching PPI metadata does not resample them.
        png_bytes = figure.to_image(format="png", scale=EXPORT_PPI / 96)
        with Image.open(BytesIO(png_bytes)) as raster:
            raster.save(folder / "figure.png", dpi=(EXPORT_PPI, EXPORT_PPI))
    finally:
        if was_home is None:
            del os.environ["HOME"]
        else:
            os.environ["HOME"] = was_home

    import fitz
    page = fitz.open(pdf_path)[0]
    if abs(page.rect.width - millimetres_to_points(PAGE_WIDTH_MM)) > 0.5:
        raise ValueError(f"the page came out {page.rect.width * MM_PER_POINT:.1f} mm across, "
                         f"not {PAGE_WIDTH_MM}")
    if not page.get_text().strip():
        raise ValueError("the figure saved with no text in it")
    return pdf_path


def colour_check(placed, floor):
    """How far apart the fill colours stay for a colour-blind reader."""
    ontology = tree.ontology

    def fill_at(score):
        position = (score - floor) / (SCORE_CEILING - floor)
        stops = [stop for stop, _ in SCORE_COLOURS]
        below = max(index for index, stop in enumerate(stops) if stop <= position or index == 0)
        above = min(below + 1, len(stops) - 1)
        if above == below:
            return SCORE_COLOURS[below][1]
        share = (position - stops[below]) / (stops[above] - stops[below])
        start = np.array(ontology.hex_to_lab(SCORE_COLOURS[below][1]))
        end = np.array(ontology.hex_to_lab(SCORE_COLOURS[above][1]))
        return ontology.lab_to_hex(start + share * (end - start))

    print(f"fill colours from {floor:.1f} to {SCORE_CEILING:.1f}\n")
    steps = np.linspace(floor, SCORE_CEILING, 8)
    for view in ontology.VIEWS:
        seen = [fill_at(score) if view == "as it is"
                else ontology.simulate_colour_blindness(fill_at(score), view)
                for score in steps]
        lightness = [ontology.hex_to_lab(colour)[0] for colour in seen]
        rises = all(later < earlier + 1e-6 for earlier, later in zip(lightness, lightness[1:]))
        print(f"  {view:<28} " + " ".join(seen))
        print(f"  {'':<28} lightness " + " ".join(f"{value:5.1f}" for value in lightness)
              + ("   steadily darker" if rises else "   NOT in order"))
        print()


def main():
    arguments = [word for word in sys.argv[1:]]
    wanted_colour_check = "--colour-check" in arguments
    species = next((word for word in arguments if not word.startswith("-")), "human")
    if species not in tree.SCORE_FILES:
        raise SystemExit(f"no such species: {species}")

    placed, names, release = tree.place(species)
    floor = score_floor(placed)

    if wanted_colour_check:
        colour_check(placed, floor)
        return

    tiles = tree.as_tiles(placed, names)
    figure, floor = draw(species, placed, tiles, names, release)
    pdf_path = check_and_save(figure, species, placed, tiles, floor)
    web_path = save_the_web_version(draw_for_the_web(species, placed, tiles, names, release),
                                    species, placed, release)

    folder = pdf_path.parent
    placed.to_csv(folder / "placement.csv", index=False)

    holds_others = (placed["cell_type_ontology_term_id"]
                    .isin(placed["drawn_inside_cl_id"]).sum())
    named = sum(1 for written in figure.data[0].labels if written)
    could = sum(1 for node, score in zip(tiles["node"], tiles["score"])
                if pd.isna(score) and node != tree.EVERYTHING)
    print(f"{species}: {len(placed)} cell types in {placed['broad_cl_id'].nunique()} "
          f"broad terms, {holds_others} of them holding other cell types")
    print(f"  {named} of the {could} boxes that hold others are named, "
          "wherever the box had the room")
    print(f"  {tree.TILE_SCORE} from {placed[tree.TILE_SCORE].min():.3f} to "
          f"{placed[tree.TILE_SCORE].max():.3f}, colour scale from {floor:.1f}")
    print(f"  ontology release {release}")
    print(f"  printed  {pdf_path}")
    print(f"  in a browser  {web_path}")


if __name__ == "__main__":
    main()
