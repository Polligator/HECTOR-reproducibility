# Figure 1 — inputs

Everything this figure needs is already in `input/`. Nothing to download.

```
input/training_ontology_projection/    cl.obo, human/mouse cell counts, reviewed assignments
input/validation_ontology_projection/  human/mouse per-class scores
```

`cl.obo` is Cell Ontology release 2025-07-30
(https://github.com/obophenotype/cell-ontology). It is the same file as figure
2's. Figure 4 uses a different, extended one — do not substitute it here.

The cell counts, reviewed assignments and class scores were produced by the
authors from the HECTOR training set and validation run.

## Running

Needs `environments/hector.yml`, and R with `voronoiTreemap`, `dplyr`,
`htmlwidgets` and `jsonlite`. The training drawing additionally needs matplotlib
replaced by the PyPI wheel — see the README; its PDF is byte-identical to the
published one only with that build. The validation drawing does not care: it
renders through Plotly and a headless browser, using `kaleido`, which is in
`hector.yml`.

```bash
conda run -n hector python scripts/training_ontology_projection/make_figure.py
conda run -n hector python scripts/validation_ontology_projection/make_figure.py human
conda run -n hector python scripts/validation_ontology_projection/make_figure.py mouse
```

The training drawing does both species in one run; the validation drawing does
one per run and defaults to `human`.

The validation drawing renders through a headless browser, which needs a scratch
profile directory on an ordinary disk, not a cloud-drive mount. It tries
`result/validation_ontology_projection/.render_home` first; set
`HECTOR_RENDER_SCRATCH` to override.

Panel C of the published figure is the training drawing's PDF, placed into the
Illustrator artwork by hand. There is no assembly script.

## Verified

Running this copy reproduces all three published PDFs. The training drawing's
three output tables and its PDF are byte-identical to the originals; the two
validation PDFs differ by two bytes each, inside the headless browser's version
string, because the browser updated between runs.
