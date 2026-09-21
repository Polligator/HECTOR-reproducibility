# HECTOR paper — analysis and figure code

This repository contains every script that produced Figures 1–5 of the HECTOR
manuscript and their supplementary figures, together with every small input those
scripts read. No cached intermediate result is included: once the large datasets
listed below are downloaded into the directories the code expects, running the
commands below regenerates every figure from scratch.

The supplementary tables are supplied with the manuscript, so the code that
builds those workbooks is not included here.

Figure 1 needs no download at all — it runs as shipped.

## Layout

```
figure1/                        training composition and validation score drawings
figure2_basic_benchmark/        annotation accuracy against PopV
figure3_integrated_datasets/    multi-tissue integration benchmark
figure4_zero-shot_benchmark/    zero-shot held-out cell type benchmark
figure5_trajectory/             fetal retina trajectories and RNA-chromatin validation
environments/                   conda environment specifications
```

Every figure directory has the same shape:

- `scripts/` — all the code for that figure, entry point included.
- `input/` — everything the code reads and never writes. The ontology files,
  label harmonization tables, gene references and manifests are shipped here;
  the large single-cell datasets are not. Each figure's `INPUTS.md` says what to
  download and where to put it.
- `result/` — everything the code writes. Created on the first run.

`figure4_zero-shot_benchmark/models/` is a fourth directory holding the five
published models HECTOR is compared against. It is neither code nor a result, so
it is not shipped; see that figure's `INPUTS.md` for the downloads.

## Dependencies

HECTOR itself is a published package and is not vendored here:

```bash
pip install hector-sc
```

The figures were produced with `hector-sc`.

Everything else is captured in `environments/`, exported directly from the conda
environments that produced the manuscript figures.

```bash
conda env create -f environments/hector.yml
conda run -n hector python -m pip install --force-reinstall --no-deps "matplotlib==3.10.7"
```

**The second line is required to reproduce Figure 1's training drawing exactly** —
its layout depends on which FreeType build matplotlib carries, and the PyPI
wheel matches the published figure; conda's own build does not.

| Environment | Used by |
|---|---|
| `hector.yml` | Figures 1, 2, 3, 4 and 5 |
| `popv.yml` | Figure 2 (the PopV comparison suite) |
| `geneformer.yml`, `scgpt.yml`, `sccello.yml`, `scimilarity.yml` | Figure 4 comparator models — see that figure's `INPUTS.md` for how to build them |
| `traj.yml` | one Supplementary Figure 12 panel — see Figure 5's `INPUTS.md` |

`palantir` and `cytotrace2-py` cannot live in `hector`: installing them there
forces NumPy and JAX down to versions that break the cuML/TensorFlow/scib-metrics
stack Figures 3–5 depend on, measured rather than assumed. `traj.yml` gives the
five trajectory tools an environment of their own instead.

Non-conda requirements (R packages, `tabix`) are listed in the `INPUTS.md` of
whichever figure needs them.

## Running

Run each command from that figure's own directory.

**Figure 1** — two independent drawings, each with its own entry point. Panel C
is a standalone PDF placed into the Illustrator artwork by hand; there is no
assembly step.

```bash
conda run -n hector python scripts/training_ontology_projection/make_figure.py
conda run -n hector python scripts/validation_ontology_projection/make_figure.py human
conda run -n hector python scripts/validation_ontology_projection/make_figure.py mouse
```

The training drawing does both species in one run. The validation drawing does
one species per run and defaults to `human`, so it is run twice.

**Figure 2** — computes predictions if they are missing, then draws. The first
run calls HECTOR in the `hector` environment and the PopV suite in the `popv`
environment, caching predictions under `.cache/`; later runs reuse them. Add
`--main-only` to stop after the main figure and the kidney tables.

```bash
conda run -n hector python scripts/run_figure2.py
```

**Figure 3** — runs the integration benchmark, then draws. `recompute = True` at
the top of the entry point runs the whole benchmark; set it to `False` to redraw
from the saved score table and UMAP coordinates. The scIB scoring clusters with
Leiden, which is not seed-stable, so a full rerun moves the printed scores
slightly even with the model and the data untouched.

```bash
conda run -n hector python scripts/run_figure3.py
```

**Figure 4** — embeds every model, runs the shared read-out and the native
decoders, then draws. Add `--skip-clustering --skip-prediction` to redraw from
prediction files that already exist.

```bash
conda run -n hector python scripts/run_figure4.py
```

**Figure 5** — two linear scripts with no command-line interface. Each has one
`REFRESH_EXPENSIVE_RESULTS` constant at the top. With it `False` the script
validates fingerprints and redraws from `result/cache/`; with it `True` the
script rebuilds its own analyses. A full refresh of the main figure needs an
NVIDIA GPU. A full refresh of the supplementary figures additionally needs the
`traj` environment described above, for one panel of Supplementary Figure
12; without it every other supplementary panel still rebuilds.

```bash
conda run -n hector python scripts/figure5.py
conda run -n hector python scripts/figure5_supplementary.py
```

Because this repository ships no cached results, the first run of each Figure 5
script must be a refresh: set `REFRESH_EXPENSIVE_RESULTS = True`, run it once to
build `result/cache/`, then set it back to `False` for subsequent redraws. Run
`figure5.py` before `figure5_supplementary.py` — the supplementary script reads
six files the main one writes into `result/cache/` and refuses to start without
them.

Supplementary Figures 12–14 use a fixed 180 × 240 mm canvas and shared
typography and line weights defined in `figure5_trajectory/scripts/supplementary_style.py`.
Figure 13 occupies only the top half of that page, with the lower half blank;
font sizes remain unchanged. PNG previews are exported at 600 dpi.

## External programs Figure 5 calls

Each is found automatically and can be overridden with an environment variable.

| Variable | Default |
|---|---|
| `FIGURE5_TABIX` | `tabix` on `PATH` |
| `FIGURE5_RSCRIPT` | `Rscript` on `PATH` |
| `FIGURE5_TRAJPY_PYTHON`, `FIGURE5_TRAJ_RSCRIPT` | the `bin/python` and `bin/Rscript` of a conda environment named `traj` beside the running interpreter |

Figure 5's main refresh also checks that the running interpreter belongs to a
conda environment named `hector`, so create it under that name.

## Inputs

`required_inputs.csv` lists every file in one table: figure, relative path,
whether it is shipped here, size in bytes, SHA-256, and source. Each figure's
`INPUTS.md` gives the same downloads with the file names to save them under.

What has to be downloaded, in short:

| Figure | Download |
|---|---|
| 1 | nothing |
| 2 | six query datasets from CZ CELLxGENE Discover; PopV reference models pull themselves at run time |
| 3 | the ovarian cancer atlas from Mendeley Data |
| 4 | the Tabula Sapiens reference panel, the held-out query set, and five model releases |
| 5 | the fetal retina RNA atlas and the GSE268630 ATAC fragments and multiome matrices |

Two points worth stating plainly, because a wrong file here changes the numbers:

- **Figures 1 and 2 share one Cell Ontology file, and Figure 4 uses a different
  one.** Both are Cell Ontology release 2025-07-30
  (https://github.com/obophenotype/cell-ontology), but Figure 4's carries 197
  extra terms (3,523 against 3,326), because the compared models can name terms
  no release contains. All three are shipped here. Figure 5 uses no ontology file.
- **Figure 2's harmonization tables and Figure 3's and Figure 5's label maps are
  hand-curated by the authors.** They are inputs, not derived files, and the code
  that built them is not part of this repository.

HECTOR's training transcriptomes, and Figure 4's query cells, were drawn from the
CZ CELLxGENE Census version 2025-11-08
(https://chanzuckerberg.github.io/cellxgene-census/).

## Licence

The analysis source code is released under the Apache License, Version 2.0;
see [LICENSE](LICENSE). Third-party dependencies and datasets retain their
respective licences.
