# Figure 3 — inputs

The three files in `input/reference/` ship here. One dataset must be downloaded.

## Download

The multi-tissue ovarian cancer atlas (Zheng et al., Nature Cancer 4:1138–1156,
2023, doi:10.1038/s43018-023-00599-8):

- processed object — Mendeley Data, https://doi.org/10.17632/rc47y6m9mp.1
- raw sequencing data — GSA-Human, accession PRJCA005422

Convert the deposited `raw_object.rds` Seurat object with
`scripts/prepare_downloaded_dataset.py`, which also does the gene identifier
mapping, and save the result as `input/processed/raw_object.h5ad`. Size and
SHA-256 for the converted file are in `../required_inputs.csv`.

The entry point reads the source object's own columns by name: `Samples` for
batch, `maintypes_2` for the label, `Groups` for tissue, `Patients` for donor.

## Shipped here

- `hector_label_harmonization.tsv` — maps HECTOR's Cell Ontology output onto the
  atlas authors' classes. Curated by hand; the code that built it is not part of
  this repository.
- `hgnc_complete_set.txt` and `biomart_hsapiens_gene_reference.tsv.gz` — the gene
  symbol and Ensembl references. The code re-downloads these when absent, so the
  copies here pin the gene mapping to the versions the published figure used.
  HGNC and Ensembl both change between releases.

## Running

```bash
conda run -n hector python scripts/run_figure3.py
```

Runs the benchmark, then draws. `recompute = True` near the top of the entry
point; set it to `False` to redraw from the saved score table and UMAP
coordinates under `result/`.

The scIB scoring clusters with Leiden, which is not seed-stable, so a full rerun
moves the printed scores slightly even with the model and the data untouched.
That is why the redraw path exists.

Needs the competitor integration methods installed in the same environment —
scVI, Harmony, Scanorama, fastMNN, Seurat. The entry point checks for each and
reports which are missing rather than failing part-way through.
