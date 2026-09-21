#!/usr/bin/env Rscript

# Draw one species' training set as a Voronoi treemap.
#
# Everything it needs is given on the command line, so nothing has to sit beside
# it. Called by make_figure.py; not meant to be run by hand.
#
# Usage: Rscript voronoi_treemap.r <cells.csv> <output_dir> <name> [min font] [smallest cell]

library(voronoiTreemap)
library(dplyr)
library(htmlwidgets)
library(jsonlite)

# -------------------------------
# Parse command-line arguments
# -------------------------------
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  cat("Usage: Rscript voronoi_treemap.r <cells.csv> <output_dir> <name> [min font] [smallest cell]\n")
  cat("  csv_path: Path to input CSV file with columns h1,h2,h3,color,weight,codes\n")
  cat("  output_dir: Directory to save output files\n")
  cat("  output_name: Base name for output files (without extension)\n")
  quit(status = 1)
}

csv_path <- args[1]
output_dir <- args[2]
output_name <- args[3]

# The smallest size a cell name is allowed to be set at, in the units the drawing is
# laid out in. A cell whose name will not fit at this size is left unnamed rather than
# named unreadably.
#
# The drawing is 940 units across. On screen it can be zoomed, so a small floor is
# fine and names most of the cells. On the page it cannot, so the floor has to be set
# from the size the circle is actually printed at: a circle 85 mm across makes one
# unit 0.09 mm, so six point type is about 23 units. Passing that in is what tells the
# two versions apart.
label_min_font <- if (length(args) >= 4) as.numeric(args[4]) else 4.5

# Smallest cell the layout will draw, as a fraction of the largest cell beside it.
# Cells below it are silently enlarged, so this has to stay low for area to mean the
# training cell count. It cannot go much lower: below about 0.002 the layout fails to
# place the smallest term and the page raises "Cannot read properties of undefined".
smallest_cell_ratio <- if (length(args) >= 5) args[5] else "3e-3"

# Validate input file exists
if (!file.exists(csv_path)) {
  cat(sprintf("ERROR: Input CSV file not found: %s\n", csv_path))
  quit(status = 1)
}

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat(sprintf("Processing: %s\n", csv_path))
cat(sprintf("Output directory: %s\n", output_dir))
cat(sprintf("Output name: %s\n", output_name))

# -------------------------------
# Colour
# -------------------------------
# Colour is not decided here. The table arrives with its colour column filled, one
# colour per broad Cell Ontology term and the same in every species, so that a term
# keeps its colour across panels. This script only checks that it is filled.

# -------------------------------
# Load and process CSV data
# -------------------------------
tryCatch({
  df_raw <- read.csv(csv_path, stringsAsFactors = FALSE)
  
  # Validate required columns
  required_cols <- c("h1", "h2", "h3", "color", "weight", "codes")
  missing_cols <- setdiff(required_cols, names(df_raw))
  
  if (length(missing_cols) > 0) {
    cat(sprintf("ERROR: Missing required columns: %s\n", paste(missing_cols, collapse = ", ")))
    quit(status = 1)
  }
  
  # Refuse to draw without colours rather than choose them here, which would give a
  # term a different colour in each species.
  if (any(is.na(df_raw$color) | !nzchar(trimws(df_raw$color)))) {
    cat("ERROR: the colour column is empty. Build the table with make_figure.py,\n")
    cat("       which fills it from the colours worked out in ontology.py.\n")
    quit(status = 1)
  }
  per_term <- tapply(df_raw$color, df_raw$h2, function(x) length(unique(x)))
  if (any(per_term != 1)) {
    cat(sprintf("ERROR: these broad terms are given more than one colour: %s\n",
                paste(names(per_term)[per_term != 1], collapse = ", ")))
    quit(status = 1)
  }

  # Ensure correct column types
  df_input <- df_raw %>%
    select(h1, h2, h3, color, weight, codes) %>%
    mutate(
      across(c(h1, h2, h3, color, codes), as.character),
      weight = as.numeric(weight)
    )
  
  cat(sprintf("Loaded %d rows from CSV\n", nrow(df_input)))
  
}, error = function(e) {
  cat(sprintf("ERROR reading CSV: %s\n", e$message))
  quit(status = 1)
})

# Build JSON and render widget
# -------------------------------
tryCatch({
  cat("DEBUG: Starting voronoi treemap generation...\n")
  cat(sprintf("DEBUG: Processing %d rows\n", nrow(df_input)))
  
  set.seed(123)
  
  cat("DEBUG: Step 1/4 - Creating voronoi input from dataframe...\n")
  vt_node <- vt_input_from_df(df_input, scaleToPerc = TRUE)
  cat("DEBUG: ✓ vt_input_from_df completed\n")
  
  cat("DEBUG: Step 2/4 - Exporting to JSON...\n")
  g_json  <- vt_export_json(vt_node)
  cat("DEBUG: ✓ vt_export_json completed\n")
  
  cat("DEBUG: Step 3/4 - Creating D3 widget (this may take a while)...\n")
  w <- vt_d3(
    g_json,
    width  = 1600,
    height = 1000,
    seed   = 3,
    legend = FALSE,  # Custom legend will be added
    # title  = sprintf("Voronoi Treemap: %s", output_name),
    label  = FALSE,  # Disable leaf labels, we'll add h2 labels via post-processing
    color_border = "#ffffff",
    size_border  = "1px"
  )
  cat("DEBUG: ✓ vt_d3 completed\n")
  
  # Save HTML output
  cat("DEBUG: Step 4/4 - Saving HTML widget...\n")
  html_path <- file.path(output_dir, paste0(output_name, ".html"))
  saveWidget(w, html_path, selfcontained = TRUE)
  cat(sprintf("✓ Saved HTML: %s\n", html_path))
  
  # Build the legend: every broad term, largest first, with its share of this species.
  # Both the term and its colour are read off the table that was just drawn, so the
  # legend cannot disagree with the picture.
  cat("DEBUG: Building the legend...\n")

  h2_summary <- df_raw %>%
    group_by(h2, color) %>%
    summarize(total_weight = sum(weight, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(total_weight)) %>%
    mutate(percent = 100 * total_weight / sum(total_weight))

  legend_row <- function(swatch, label) {
    sprintf(
      '<div style="display: flex; align-items: center; margin: 4px 0;">
        <div style="width: 18px; height: 18px; background-color: %s; margin-right: 8px; border: 1px solid #ccc; flex: none;"></div>
        <span style="font-size: 13px; color: #333;">%s</span>
      </div>', swatch, label)
  }

  legend_items <- character(0)
  for (i in seq_len(nrow(h2_summary))) {
    legend_items <- c(legend_items, legend_row(
      h2_summary$color[i],
      sprintf("%s &nbsp;<span style='color:#777'>%.1f%%</span>",
              gsub("_", " ", h2_summary$h2[i]), h2_summary$percent[i])))
  }

  legend_html <- sprintf("
    <style>
      html, body { margin: 0; min-width: 0; }
      #htmlwidget_container { box-sizing: border-box; width: min(100%%, 1400px); margin: 0 auto; padding: 12px 16px 0; }
      #custom-h2-legend { box-sizing: border-box; width: min(100%% - 32px, 1400px); max-width: none !important; margin: 14px auto 24px; display: flex; flex-wrap: nowrap; align-items: center; gap: 14px; overflow-x: auto; white-space: nowrap; }
      #custom-h2-legend > div { flex: none; }
      #custom-h2-legend > div:not(:first-child) { margin: 0 !important; }
      @media (max-width: 700px) { #htmlwidget_container { padding-inline: 8px; } #custom-h2-legend { width: calc(100%% - 16px); gap: 10px; } }
    </style>
    <div id='custom-h2-legend' style='
      background: white;
      padding: 15px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
      font-family: Arial, sans-serif;
    '>
      <div style='font-weight: bold; margin-bottom: 10px; font-size: 14px; color: #333;'>Broad Cell Ontology term</div>
      %s
    </div>
  ", paste(legend_items, collapse = "\n"))
  
  # Read and modify HTML
  html_content <- readLines(html_path, warn = FALSE)
  html_text <- paste(html_content, collapse = "\n")

  # Lower the layout's smallest-cell floor, which otherwise enlarges small cells and
  # breaks the correspondence between a cell's area and its count. The floor exists
  # to keep tiny cells catchable by the mouse; the count matters more.
  #
  # The page carries its own copy of the layout code, so this changes this drawing
  # only and leaves the installed package alone.
  # The number of layout passes is left at its default of fifty. Raising it makes the
  # areas more accurate but the page then computes for minutes in a real browser and
  # will not open. The headless render below fast-forwards the page's clock and so
  # hides that entirely: anything changed here must be checked in a real browser.
  from <- "var DEFAULT_MIN_WEIGHT_RATIO = 0.01;"
  to   <- paste0("var DEFAULT_MIN_WEIGHT_RATIO = ", smallest_cell_ratio, ";")
  found <- lengths(regmatches(html_text, gregexpr(from, html_text, fixed = TRUE)))
  html_text <- gsub(from, to, html_text, fixed = TRUE)
  cat(sprintf("✓ Smallest-cell floor set to %s in %d place(s)\n", smallest_cell_ratio, found))
  
  # Insert legend before closing </body> tag
  html_text <- sub("</body>", paste0(legend_html, "\n</body>"), html_text, fixed = TRUE)
  
  # ----------------------------------
  # Name the cells
  # ----------------------------------
  # Every cell is offered a name, wrapped over up to four lines and set at the
  # largest size that fits inside it. A cell is left unnamed only when even the
  # smallest readable size will not fit, and no name is ever truncated. Which cells
  # end up named therefore follows from how large they are drawn.
  #
  # Label only the visible cell. Each one has a second, invisible copy underneath
  # for catching the mouse; labelling both draws every name twice.
  cat("DEBUG: Naming the cells...\n")

  label_script <- paste0("
  <script>
  (function() {
    setTimeout(function() {
      var svg = document.querySelector('svg');
      if (!svg) { console.log('No SVG found'); return; }
      var NS = 'http://www.w3.org/2000/svg';

      var MIN_FONT = ", label_min_font, ";   // nothing smaller than this is readable
      var MAX_FONT = 13;
      var MAX_LINES = 4;
      var LINE_SPACING = 1.12;
      var MARGIN = 0.94;      // leave a little air between the name and the cell edge

      // Only the drawn cells. The copies that catch the mouse carry the same name
      // and labelling them too is what drew every name twice.
      var cells = svg.querySelectorAll('path.cell');

      // What the mouse shows. The drawing package writes the cell type's name and
      // its share of the species as a percentage. The number of cells is what the
      // drawing is actually made of, so show that instead. It is the number that
      // entered training, so it stops at 50,000 for the commonest cell types, and
      // the wording says so rather than leaving a capped number to be read as a
      // count of everything there is.
      var retitled = 0;
      svg.querySelectorAll('path.hoverer').forEach(function(hoverer) {
        var data = hoverer.__data__;
        var caption = hoverer.querySelector('title');
        if (!data || !data.data || !caption) return;
        var howMany = Number(data.data.code);
        if (!isFinite(howMany)) return;
        // A cell type whose name is also a broad term's name arrives with two spaces
        // on the end, which is how the drawing package keeps the two apart.
        caption.textContent = data.data.name.trim() + '\\n' +
          howMany.toLocaleString('en-US') + ' cells used in training';
        retitled++;
      });
      console.log('Rewrote ' + retitled + ' of ' + cells.length + ' pop-ups');
      svg.setAttribute('data-hovers-rewritten', retitled + ' of ' + cells.length);

      // Text width is proportional to font size, so measure each line once at a
      // large size and scale from there rather than laying out text repeatedly.
      var ruler = document.createElementNS(NS, 'text');
      ruler.setAttribute('font-family', 'Arial, sans-serif');
      ruler.setAttribute('font-size', '100px');
      ruler.setAttribute('font-weight', 'bold');
      ruler.setAttribute('visibility', 'hidden');
      svg.appendChild(ruler);
      var measured = {};
      function widthOf(line) {
        if (measured[line] === undefined) {
          ruler.textContent = line;
          measured[line] = ruler.getComputedTextLength();
        }
        return measured[line];
      }

      function cornersOf(path) {
        var nums = path.getAttribute('d').match(/-?[0-9.]+(?:e-?[0-9]+)?/gi);
        var pts = [];
        if (!nums) return pts;
        for (var i = 0; i + 1 < nums.length; i += 2) pts.push([+nums[i], +nums[i + 1]]);
        return pts;
      }

      function middleOf(pts) {
        var twiceArea = 0, x = 0, y = 0;
        for (var i = 0, n = pts.length; i < n; i++) {
          var a = pts[i], b = pts[(i + 1) % n];
          var cross = a[0] * b[1] - b[0] * a[1];
          twiceArea += cross; x += (a[0] + b[0]) * cross; y += (a[1] + b[1]) * cross;
        }
        if (Math.abs(twiceArea) < 1e-9) return null;
        return { x: x / (3 * twiceArea), y: y / (3 * twiceArea),
                 area: Math.abs(twiceArea) / 2 };
      }

      function isInside(pts, x, y) {
        var inside = false;
        for (var i = 0, j = pts.length - 1; i < pts.length; j = i++) {
          var a = pts[i], b = pts[j];
          if (((a[1] > y) !== (b[1] > y)) &&
              (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0])) inside = !inside;
        }
        return inside;
      }

      // How much room there is around a point before the cell's edge is reached.
      function roomAt(pts, x, y) {
        var least = Infinity;
        for (var i = 0, n = pts.length; i < n; i++) {
          var a = pts[i], b = pts[(i + 1) % n];
          var dx = b[0] - a[0], dy = b[1] - a[1];
          var along = dx * dx + dy * dy;
          var t = along ? Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / along)) : 0;
          var ex = a[0] + t * dx - x, ey = a[1] + t * dy - y;
          least = Math.min(least, Math.sqrt(ex * ex + ey * ey));
        }
        return least;
      }

      // The roomiest point in the cell, which is where the name goes. Starting at
      // the middle and searching outwards keeps the name off the edges of cells
      // that are long and thin.
      function roomiestPoint(pts) {
        var mid = middleOf(pts);
        if (!mid) return null;
        var best = { x: mid.x, y: mid.y, room: isInside(pts, mid.x, mid.y) ? roomAt(pts, mid.x, mid.y) : 0 };
        var xs = pts.map(function(p) { return p[0]; }), ys = pts.map(function(p) { return p[1]; });
        var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
        var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
        var step = Math.max((x1 - x0), (y1 - y0)) / 8;
        for (var pass = 0; pass < 2; pass++) {
          var cx = best.x, cy = best.y;
          for (var gx = -2; gx <= 2; gx++) {
            for (var gy = -2; gy <= 2; gy++) {
              var px = cx + gx * step, py = cy + gy * step;
              if (!isInside(pts, px, py)) continue;
              var room = roomAt(pts, px, py);
              if (room > best.room) best = { x: px, y: py, room: room };
            }
          }
          step /= 2.5;
        }
        best.area = mid.area;
        return best;
      }

      // Break the name into at most this many lines, keeping the longest line as
      // short as it can be, so the block of text is as square as possible.
      function breakInto(words, howManyLines) {
        var n = words.length;
        if (howManyLines > n) return null;
        var best = null;
        function join(from, to) { return words.slice(from, to).join(' '); }
        function search(from, linesLeft, sofar, widest) {
          if (best !== null && widest >= best.widest) return;
          if (linesLeft === 1) {
            var last = join(from, n);
            var w = Math.max(widest, widthOf(last));
            if (best === null || w < best.widest) best = { lines: sofar.concat([last]), widest: w };
            return;
          }
          for (var cut = from + 1; cut <= n - linesLeft + 1; cut++) {
            var line = join(from, cut);
            search(cut, linesLeft - 1, sofar.concat([line]), Math.max(widest, widthOf(line)));
          }
        }
        search(0, howManyLines, [], 0);
        return best;
      }

      // The largest size at which the whole name fits inside the room available.
      // The block of text has to sit within a circle of that radius, so its half
      // diagonal is what has to fit.
      function fitName(name, room) {
        var words = name.split(/\\s+/).filter(function(w) { return w.length; });
        var bestFit = null;
        for (var lines = 1; lines <= MAX_LINES; lines++) {
          var split = breakInto(words, lines);
          if (!split) break;
          var halfWide = split.widest / 200;
          var halfTall = lines * LINE_SPACING / 2;
          var size = room * MARGIN / Math.sqrt(halfWide * halfWide + halfTall * halfTall);
          size = Math.min(size, MAX_FONT);
          if (bestFit === null || size > bestFit.font) bestFit = { lines: split.lines, font: size };
        }
        return (bestFit && bestFit.font >= MIN_FONT) ? bestFit : null;
      }

      // Dark text on the pale cells, light text on the deep ones.
      function inkFor(cell) {
        var fill = window.getComputedStyle(cell).fill;
        var parts = fill.match(/[0-9]+/g);
        if (!parts) return { ink: '#ffffff', halo: 'rgba(0,0,0,0.65)' };
        var brightness = 0.299 * parts[0] + 0.587 * parts[1] + 0.114 * parts[2];
        return brightness > 150
          ? { ink: '#1a1a1a', halo: 'rgba(255,255,255,0.75)' }
          : { ink: '#ffffff', halo: 'rgba(0,0,0,0.65)' };
      }

      var named = 0, unnamed = 0, holder = null;
      var waiting = [];
      cells.forEach(function(cell) {
        var data = cell.__data__;
        if (!data || !data.data || !data.data.name) { unnamed++; return; }

        // Write what the cell is onto the cell itself. Nothing in the saved drawing
        // said which cell type a shape was or which broad term it belonged to, so the
        // file could only be read as anonymous shapes. With this the static figure can
        // be built from the drawing, and the shapes can be picked out by name in
        // Illustrator.
        cell.setAttribute('data-cell-type', data.data.name.trim());
        cell.setAttribute('data-cells', data.data.code);
        if (data.parent && data.parent.data && data.parent.data.name) {
          cell.setAttribute('data-broad-term', data.parent.data.name.trim());
        }

        var pts = cornersOf(cell);
        if (pts.length < 3) { unnamed++; return; }
        var spot = roomiestPoint(pts);
        if (!spot) { unnamed++; return; }
        var fit = fitName(data.data.name, spot.room);
        if (!fit) { unnamed++; return; }

        var colours = inkFor(cell);
        var text = document.createElementNS(NS, 'text');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('dominant-baseline', 'middle');
        text.setAttribute('font-size', fit.font.toFixed(2) + 'px');
        text.setAttribute('font-family', 'Arial, sans-serif');
        text.setAttribute('font-weight', 'bold');
        text.setAttribute('fill', colours.ink);
        text.setAttribute('stroke', colours.halo);
        text.setAttribute('stroke-width', (fit.font * 0.07).toFixed(3) + 'px');
        text.setAttribute('paint-order', 'stroke');
        text.setAttribute('pointer-events', 'none');

        var top = -(fit.lines.length - 1) / 2 * fit.font * LINE_SPACING;
        fit.lines.forEach(function(line, i) {
          var part = document.createElementNS(NS, 'tspan');
          part.setAttribute('x', spot.x.toFixed(2));
          part.setAttribute('y', (spot.y + top + i * fit.font * LINE_SPACING).toFixed(2));
          part.textContent = line;
          text.appendChild(part);
        });

        holder = cell.parentElement;
        waiting.push(text);
        named++;
      });

      // Put every name in after all the cells, so no cell is drawn over a name.
      if (holder) waiting.forEach(function(text) { holder.appendChild(text); });
      ruler.remove();
      svg.setAttribute('data-cells-named', named + ' of ' + (named + unnamed));
      console.log('Named ' + named + ' of ' + (named + unnamed) + ' cells');

      // Measure how much room the finished drawing actually takes up, and record it on the
      // drawing for the saving step to read. Only the measurement is taken here: nothing is
      // moved or resized, because on this page the drawing is meant to fill its container
      // and resizing it would slide it out from under the legend.
      try {
        var drawnBox = svg.getBBox();
        var margin = 10;
        svg.setAttribute('data-fitbox',
          (drawnBox.x - margin) + ' ' + (drawnBox.y - margin) + ' ' +
          (drawnBox.width + 2 * margin) + ' ' + (drawnBox.height + 2 * margin));
      } catch(e) {
        console.log('Could not measure the drawing:', e);
      }
    }, 3000);
  })();
  </script>
  ")

  # Insert script before closing </body> tag
  html_text <- sub("</body>", paste0(label_script, "\n</body>"), html_text, fixed = TRUE)

  # The widget package writes a fixed 1600 x 1000 canvas.  On a normal web page
  # that leaves unused space around the drawing and lets the legend overlap it.
  # Fit the finished Voronoi cells to the page width, then keep the legend as a
  # separate horizontal strip beneath the plot.  The same data-fitbox continues
  # to drive the standalone SVG export below.
  responsive_layout_script <- paste0("
  <script>
  window.addEventListener('load', function () {
    setTimeout(function () {
      var widget = document.querySelector('.d3vt.html-widget');
      var svg = widget && widget.querySelector('svg');
      if (!widget || !svg) return;
      var measured = (svg.getAttribute('data-fitbox') || '').trim().split(/\\s+/).map(Number);
      if (measured.length !== 4 || measured.some(function (value) { return !isFinite(value); })) return;
      var availableWidth = Math.max(320, widget.parentElement.clientWidth);
      var aspect = measured[2] / measured[3];
      widget.style.width = '100%';
      widget.style.height = Math.ceil(availableWidth / aspect) + 'px';
      svg.setAttribute('viewBox', measured.join(' '));
      svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
      svg.style.width = '100%';
      svg.style.height = '100%';
    }, 3400);
  });
  </script>
  ")
  html_text <- sub("</body>", paste0(responsive_layout_script, "\n</body>"), html_text, fixed = TRUE)

  writeLines(html_text, html_path)
  cat("✓ Added the legend and the cell names to the page\n")
  cat("DEBUG: HTML processing completed successfully\n")
  
  # Save the SVG by opening the page in the browser and reading back what it drew.
  #
  # The browser's clock is run forward rather than waited on. While the layout is
  # being computed the page cannot answer anything at all, however long it is given,
  # so waiting in real time never returns a drawing.
  browser_path <- tryCatch(chromote::find_chrome(), error = function(e) "")

  if (nzchar(browser_path)) {
    tryCatch({
      dom_file <- tempfile(fileext = ".html")

      # The page sits at the same address on every run, and the browser will happily show a
      # copy it kept from a previous run. That is how a freshly drawn plot could still be
      # saved carrying the previous run's numbers. Giving the address a different ending
      # each time forces the browser to read the page that was just written.
      file_url <- paste0("file://", normalizePath(html_path), "?drawn=", as.integer(Sys.time()))

      cat("Rendering the drawing in the browser...\n")
      system2(
        browser_path,
        args = c(
          "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
          "--window-size=1600,1000",
          "--virtual-time-budget=180000",   # run the page's clock forward, do not wait in real time
          "--dump-dom",
          shQuote(file_url)
        ),
        stdout = dom_file,
        stderr = FALSE
      )

      page_text <- paste(readLines(dom_file, warn = FALSE), collapse = "\n")
      unlink(dom_file)

      svg_match <- regmatches(page_text, regexpr("<svg.*</svg>", page_text))
      if (length(svg_match) == 0) {
        stop("the finished page held no drawing")
      }

      svg_text <- svg_match[1]

      # On the page the drawing is sized by whatever contains it, so it carries no width,
      # height or viewBox of its own, only a 'width:100%; height:100%' style. Saved on its
      # own there is nothing to fill, so it opened off centre with part of it outside the
      # canvas. Give the saved copy the box the page measured for it. This is done here
      # rather than on the page itself, so the page keeps its own layout and the legend
      # stays clear of the drawing.
      fitted <- regmatches(svg_text, regexpr('data-fitbox="[^"]+"', svg_text))
      if (length(fitted) == 0) {
        stop("the drawing was read back before it had been measured")
      }
      fit_box <- gsub('data-fitbox="|"', '', fitted[1])
      fit_size <- as.numeric(strsplit(fit_box, " ")[[1]])

      open_tag <- regmatches(svg_text, regexpr("^<svg[^>]*>", svg_text))[1]
      new_tag <- sub('style="[^"]*"', "", open_tag)
      # The responsive web page has already fitted the live SVG to its container.
      # Remove those browser-only dimensions before giving the standalone file its
      # own measured box; otherwise the root element carries duplicate attributes
      # and is invalid SVG XML.
      new_tag <- sub(' viewBox="[^"]*"', "", new_tag)
      new_tag <- sub(' width="[^"]*"', "", new_tag)
      new_tag <- sub(' height="[^"]*"', "", new_tag)
      new_tag <- sub(' preserveAspectRatio="[^"]*"', "", new_tag)
      new_tag <- sub(
        "<svg",
        sprintf('<svg viewBox="%s" width="%.0f" height="%.0f"', fit_box, fit_size[3], fit_size[4]),
        new_tag
      )
      svg_text <- sub("^<svg[^>]*>", new_tag, svg_text)

      if (!grepl("xmlns=", svg_text, fixed = TRUE)) {
        svg_text <- sub("<svg", "<svg xmlns=\"http://www.w3.org/2000/svg\"", svg_text)
      }
      if (!grepl("xmlns:xlink=", svg_text, fixed = TRUE)) {
        svg_text <- sub("<svg", "<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\"", svg_text)
      }

      counted <- regmatches(svg_text, regexpr('data-cells-named="[^"]+"', svg_text))
      if (length(counted) > 0) {
        cat(sprintf("✓ Named %s cells\n", gsub('data-cells-named="|"', '', counted[1])))
      }
      hovers <- regmatches(svg_text, regexpr('data-hovers-rewritten="[^"]+"', svg_text))
      if (length(hovers) > 0) {
        cat(sprintf("✓ Pop-ups now showing cell numbers: %s\n",
                    gsub('data-hovers-rewritten="|"', '', hovers[1])))
      }

      svg_path <- file.path(output_dir, paste0(output_name, ".svg"))
      writeLines(paste0("<?xml version=\"1.0\" standalone=\"no\"?>\n", svg_text), svg_path)
      cat(sprintf("✓ Saved SVG: %s (%d shapes)\n", svg_path,
                  lengths(regmatches(svg_text, gregexpr("<path", svg_text, fixed = TRUE)))))

    }, error = function(e) {
      cat(sprintf("Warning: Could not generate SVG: %s\n", e$message))
    })
  } else {
    cat("Note: no browser found - SVG export skipped\n")
  }
  
  cat("✓ Voronoi treemap generation completed successfully\n")
  
}, error = function(e) {
  cat(sprintf("ERROR generating treemap: %s\n", e$message))
  quit(status = 1)
})
