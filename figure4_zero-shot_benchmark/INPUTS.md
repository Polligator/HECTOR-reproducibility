# Figure 4 — inputs

`input/cl.obo` ships here. Two datasets and five model releases must be
downloaded.

## Download — data

Both files are hosted together on figshare:
https://figshare.com/s/43ae877369dd6ff972c0

| Save as | What it is |
|---|---|
| `input/TS_downsampled_cells.h5ad` | the Tabula Sapiens reference panel, downsampled; originally from [collection e5f58829](https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5) |
| `input/Zero_shot_query_set.h5ad` | the 66 held-out human cell types, drawn by the authors from CZ CELLxGENE Census version 2025-11-08 |

Sizes and SHA-256 checksums are in `../required_inputs.csv`.

The query set is the 66 held-out human cell types, 500 cells each, 33,000 cells
in total. The selection rule is recorded inside the file: a type is included if
it is absent from the checkpoint's 650 trained classes, present among its 1,407
nameable classes, has more than 500 primary Census cells, and is not a Tabula
Sapiens landmark type. Normal donors were sampled first within each type.
Supplementary Table 4 lists the 66 query types and the 144 reference-panel types
with their Cell Ontology identifiers and analyzed cell counts.

The Tabula Sapiens columns the code reads are `cell_type` for the label and
`donor_id` for the batch.

## Download — the compared models

Each is a public release from its own authors. Place them as:

```
models/geneformer/          Geneformer-V2-104M, Geneformer-V2-316M, and the geneformer repository
models/scGPT/checkpoints/scGPT_CP
models/scCello/checkpoints/scCello-zeroshot, and scCello-repo
models/scimilarity/model_v1.1
models/OnClass/pretrained/OnClass_data_public, and repo
```

`scripts/model_eval/setup_envs.sh` builds the four comparator conda environments
and expects this exact layout — it installs the scCello and Geneformer Python
packages from the repositories inside `models/`, which is why those two appear in
`environments/*.yml` with version `0.1.0` rather than a PyPI version.

OnClass is used only for the strict native-decoder comparison in panel b, not for
the shared read-out.

## The ontology file

`input/cl.obo` is **not the same file as figure 2's.** It is Cell Ontology
release 2025-07-30 (https://github.com/obophenotype/cell-ontology) extended by
the authors with 197 further terms, 3,523 `CL:` terms in total, because the
compared models can name terms no release of the ontology contains. Using the
plain release here drops predictions the benchmark has to score.

## Running

```bash
conda run -n hector python scripts/run_figure4.py
```

Embeds every model, runs the shared read-out and the native decoders, then
draws. `--skip-clustering --skip-prediction` redraws from prediction files
already under `result/data/`.

Each comparator model is embedded by a subprocess launched in its own conda
environment (`geneformer`, `scgpt`, `sccello`, `scimilarity`), so those must
exist under those names. HECTOR runs in `hector` alongside the entry point.

`scripts/model_eval/hector_requirements.txt` is the pinned package list the
`hector` environment was built from, kept here because `setup_envs.sh` reads it.
