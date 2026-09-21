# Figure 5 — inputs

The label map, the eight MultiVelo tables, the promoter annotation and both
manifests ship here. The RNA atlas, the ATAC fragments and the multiome matrices
must be downloaded.

`input/figure5_input_manifest.csv` carries the relative path, size, SHA-256 and
per-sample accession of every input, including the ones shipped. The code reads
it and refuses to run when a file does not match.

## Download

| Save under | Source |
|---|---|
| `input/HumanFetalRetina.h5ad` | [CZ CELLxGENE collection 5900dda8](https://cellxgene.cziscience.com/collections/5900dda8-2dc3-4770-b604-084eac1c2c82), also GEO [GSE268630](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE268630) |
| `input/atac/fragments/` | GEO GSE268630 — six fragment files and their tabix indexes; the per-sample GSM accessions are in the manifest |
| `input/atac/multiome/` | GEO GSE268630 — eight Cell Ranger ARC matrices |

The published ordering used for comparison is on Zenodo,
https://doi.org/10.5281/zenodo.11412907.

Source papers: doi:10.1038/s41467-024-50853-5 (the fetal retina dataset) and
doi:10.1038/s41588-025-02454-1 (the human retina atlas reference). Neither is
read by the code.

## Shipped here

- `hector_label_harmonization.csv` — HECTOR-to-author class map, curated by hand.
- `multivelo/*.csv.gz` — eight deposited class-wise velocity tables. These are
  immutable source inputs, not derived caches: the original class-specific
  MultiVelo objects are not available, so these tables are the only form of that
  result.
- `atac/refGene_hg38.txt.gz` — strand-aware promoter annotation.
- `atac/library_manifest.csv` — the GEO sample list.

Figure 5 uses no Cell Ontology file.

## Running

Two linear scripts, no command-line interface:

```bash
conda run -n hector python scripts/figure5.py                 # main Figure 5
conda run -n hector python scripts/figure5_supplementary.py   # Supplementary Figures 12-14
```

**Run `figure5.py` first.** The supplementary script never imports or executes
the main one, but it reads six files the main one writes into `result/cache/`
(`hector_cells.parquet`, `cache_manifest.json`, `rna_curves.csv`,
`signac_profiles.csv.gz`, `signac_effects.csv`, `tabix_effects.csv`) and refuses
to start without them.

Each script has one constant at the top, `REFRESH_EXPENSIVE_RESULTS`. **This
repository ships no cached results, so your first run of each must set it to
`True`.** Once that run has built `result/cache/`, set it back to `False`; the
cached path then validates fingerprints and redraws in seconds without reading
the ATAC fragments, opening the 3-GB atlas, or launching R.

A full refresh of the main figure reruns HECTOR on an NVIDIA GPU with the cuML
UMAP backend and must run with the Python of a conda environment named `hector`,
which it checks before doing any work. New caches are promoted only after
numerical parity checks pass.

One supplementary panel needs a second environment: Supplementary Figure 12's
comparison of HECTOR against five documented trajectory tools runs DPT, Palantir,
CytoTRACE 2, Monocle 3 and Slingshot as subprocesses. All five fit in one
environment of their own, `environments/traj.yml`, which the script finds
automatically. They cannot go in `hector` — see the README for the measured
reason. Without that environment every other supplementary panel still rebuilds.

Three external programs are located automatically and can be overridden with
`FIGURE5_TABIX`, `FIGURE5_RSCRIPT`, and `FIGURE5_TRAJPY_PYTHON` /
`FIGURE5_TRAJ_RSCRIPT`. `tabix` comes from HTSlib or samtools. The Signac step
needs R with `Signac`, `Seurat`, `GenomicRanges` and
`BSgenome.Hsapiens.UCSC.hg38`.
