# Figure 2 — inputs

`input/cl.obo` and the six tables in `input/harmonization/` ship here. The six
query datasets must be downloaded into `input/datasets/` under the exact file
names below — the code derives each harmonization table's name from its
dataset's name.

## Download

All six are on CZ CELLxGENE Discover. The collection link is the citable one;
the file link is the exact dataset version used.

| Save as | Collection | File |
|---|---|---|
| `human_kidney_normal_cells.h5ad` | [9c9d04c4](https://cellxgene.cziscience.com/collections/9c9d04c4-8899-417f-bb6f-6107dcadf14f) | https://datasets.cellxgene.cziscience.com/9beaa89c-608a-4ef7-a816-2e6d42b1a0c2.h5ad |
| `mouse_kidney_normal_cells.h5ad` | [9c9d04c4](https://cellxgene.cziscience.com/collections/9c9d04c4-8899-417f-bb6f-6107dcadf14f) | https://datasets.cellxgene.cziscience.com/1da2eb55-ae34-420a-a204-5e1657a4535d.h5ad |
| `Human_BAL.h5ad` | [30704a65](https://cellxgene.cziscience.com/collections/30704a65-1b50-409b-9cf6-1f26bf644c99) | — |
| `HumanSkin_normal_cells.h5ad` | [b1fd6a09](https://cellxgene.cziscience.com/collections/b1fd6a09-eb76-44ca-822d-68318548094c) | https://datasets.cellxgene.cziscience.com/5b293ff5-baa6-465e-b03b-043d9b892850.h5ad |
| `mouse_VSMC.h5ad` | [7a3044e4](https://cellxgene.cziscience.com/collections/7a3044e4-6b16-4693-9504-212d9a573f80) | https://datasets.cellxgene.cziscience.com/17f8fb18-a666-4882-b876-0a9aabc1a796.h5ad |
| `mouse_hippocampus.h5ad` | [d245b35b](https://cellxgene.cziscience.com/collections/d245b35b-3cc1-4f47-aed6-68dfdadebb5f) | https://datasets.cellxgene.cziscience.com/9940f1a7-5fc8-4e52-aba4-6dbba8c90303.h5ad |

Sizes and SHA-256 checksums are in `../required_inputs.csv`.

The PopV reference models are pulled from https://huggingface.co/popV at run
time, so the first run needs network access. Nothing to download by hand.

## Shipped here

- `input/cl.obo` — Cell Ontology release 2025-07-30, unmodified, 3,326 `CL:`
  terms. Figure 4's is a different file with 197 more terms.
- `input/harmonization/*.csv` — six hand-curated tables mapping each dataset's
  author labels onto the shared biological classes the benchmark scores against.
  They are inputs, not derived files; the code that built them is not part of
  this repository.

## Running

```bash
conda run -n hector python scripts/run_figure2.py
```

Computes predictions if missing, then draws. HECTOR runs in the `hector`
environment and the PopV suite in `popv`, launched as subprocesses, so both must
exist under those names. Predictions cache under `.cache/`; delete it to force a
recompute. `--main-only` stops after the main figure and the kidney tables.

Several PopV methods are not deterministic — scANVI moves about 0.7 accuracy
points between runs on the same cells — so a recompute will not reproduce the
published numbers to the last decimal.
