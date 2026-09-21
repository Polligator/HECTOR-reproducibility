from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import requests
from scipy import sparse


ENSEMBL_PATTERN = re.compile(r"^ENSG[0-9]+(?:\.[0-9]+)?$")
HELPER_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = HELPER_DIR.parents[1] / "input" / "reference"

DATASET_SPECS = {
    "immune_all_human": {
        "display_name": "Immune",
        "source_filename": "Immune_ALL_human.h5ad",
    },
    "lung_atlas_public": {
        "display_name": "Lung",
        "source_filename": "Lung_atlas_public.h5ad",
    },
    "human_pancreas_norm_complexbatch": {
        "display_name": "Pancreas",
        "source_filename": "human_pancreas_norm_complexBatch.h5ad",
    },
}


def resolve_dataset_config(
    dataset_name: str,
    *,
    source_filename: str | None = None,
    display_name: str | None = None,
) -> dict[str, str]:
    known_spec = DATASET_SPECS.get(dataset_name, {})
    resolved_source_filename = source_filename or known_spec.get("source_filename") or f"{dataset_name}.h5ad"
    resolved_display_name = display_name or known_spec.get("display_name") or dataset_name
    return {
        "dataset_name": dataset_name,
        "source_filename": str(resolved_source_filename),
        "display_name": str(resolved_display_name),
    }


def ensure_directory(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json_report(report: dict, destination: Path | str) -> Path:
    destination = Path(destination)
    ensure_directory(destination.parent)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def request_get_with_retries(
    url: str,
    *,
    stream: bool = False,
    timeout: int = 120,
    attempts: int = 6,
    backoff_seconds: int = 5,
    **kwargs,
) -> requests.Response:
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url,
                stream=stream,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
                **kwargs,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt == attempts:
                break
            time.sleep(backoff_seconds * attempt)

    raise last_error


def _normalize_obs_index_column_conflict(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    index_name = adata.obs.index.name

    if index_name is None or index_name not in adata.obs.columns:
        return adata

    index_values = pd.Index(adata.obs.index.astype(str))
    column_values = adata.obs[index_name].astype(str)

    if column_values.equals(pd.Series(index_values, index=adata.obs.index)):
        adata.obs[index_name] = index_values.to_numpy()
    else:
        adata.obs[f"source_{index_name}"] = column_values.to_numpy()
        adata.obs = adata.obs.drop(columns=[index_name])

    return adata


def _download_hgnc_complete_set(reference_dir: Path) -> Path:
    reference_dir = ensure_directory(reference_dir)
    destination = reference_dir / "hgnc_complete_set.txt"

    if destination.exists():
        return destination

    urls = [
        "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
        "https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt",
    ]

    for url in urls:
        try:
            response = request_get_with_retries(url, timeout=120)
            destination.write_bytes(response.content)
            return destination
        except Exception:
            continue

    raise RuntimeError("Failed to download the HGNC complete set reference.")


def _load_biomart_reference(reference_dir: Path) -> pd.DataFrame:
    reference_dir = ensure_directory(reference_dir)
    destination = reference_dir / "biomart_hsapiens_gene_reference.tsv.gz"

    if destination.exists():
        biomart = pd.read_csv(destination, sep="\t")
    else:
        biomart_query = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Attribute name="ensembl_gene_id" />
    <Attribute name="external_gene_name" />
  </Dataset>
</Query>
"""
        biomart_response = None

        for host in ["https://www.ensembl.org", "https://useast.ensembl.org", "https://asia.ensembl.org"]:
            try:
                response = request_get_with_retries(
                    f"{host}/biomart/martservice",
                    timeout=120,
                    params={"query": biomart_query},
                )
                biomart_response = response.text
                break
            except Exception:
                continue

        if biomart_response is None:
            raise RuntimeError("Failed to download the Ensembl BioMart gene reference.")

        biomart = pd.read_csv(
            io.StringIO(biomart_response),
            sep="\t",
            names=["ensembl_gene_id", "gene_symbol"],
        )
        biomart = biomart.dropna(subset=["ensembl_gene_id"]).copy()
        biomart["ensembl_gene_id"] = biomart["ensembl_gene_id"].astype(str).str.replace(r"\.[0-9]+$", "", regex=True)
        biomart["gene_symbol"] = biomart["gene_symbol"].fillna("").astype(str)
        biomart.to_csv(destination, sep="\t", index=False, compression="gzip")

    biomart = biomart.drop_duplicates().copy()
    biomart["ensembl_gene_id"] = biomart["ensembl_gene_id"].astype(str).str.replace(r"\.[0-9]+$", "", regex=True)
    biomart["gene_symbol"] = biomart["gene_symbol"].fillna("").astype(str)
    return biomart


def _load_hgnc_reference(reference_dir: Path) -> pd.DataFrame:
    destination = _download_hgnc_complete_set(reference_dir)
    hgnc = pd.read_csv(destination, sep="\t", low_memory=False)
    required_columns = ["symbol", "alias_symbol", "prev_symbol", "ensembl_gene_id", "entrez_id"]
    available_columns = [column for column in required_columns if column in hgnc.columns]
    hgnc = hgnc[available_columns].copy()
    hgnc["symbol"] = hgnc["symbol"].fillna("").astype(str)
    hgnc["ensembl_gene_id"] = hgnc["ensembl_gene_id"].fillna("").astype(str).str.replace(r"\.[0-9]+$", "", regex=True)
    if "entrez_id" in hgnc.columns:
        hgnc["entrez_id"] = (
            hgnc["entrez_id"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
    return hgnc


def _build_symbol_maps(
    reference_dir: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]], dict[str, str], dict[str, str]]:
    biomart = _load_biomart_reference(reference_dir)
    hgnc = _load_hgnc_reference(reference_dir)

    symbol_to_ens_df = biomart.loc[biomart["gene_symbol"] != "", ["gene_symbol", "ensembl_gene_id"]].drop_duplicates()
    symbol_counts = symbol_to_ens_df.groupby("gene_symbol")["ensembl_gene_id"].nunique()
    symbol_to_ens_df = symbol_to_ens_df.loc[symbol_to_ens_df["gene_symbol"].isin(symbol_counts.index[symbol_counts == 1])]
    biomart_exact_symbol_map = dict(zip(symbol_to_ens_df["gene_symbol"], symbol_to_ens_df["ensembl_gene_id"], strict=True))

    hgnc_exact_df = hgnc.loc[(hgnc["symbol"] != "") & (hgnc["ensembl_gene_id"] != ""), ["symbol", "ensembl_gene_id"]].drop_duplicates()
    hgnc_counts = hgnc_exact_df.groupby("symbol")["ensembl_gene_id"].nunique()
    hgnc_exact_df = hgnc_exact_df.loc[hgnc_exact_df["symbol"].isin(hgnc_counts.index[hgnc_counts == 1])]
    hgnc_exact_symbol_map = dict(zip(hgnc_exact_df["symbol"], hgnc_exact_df["ensembl_gene_id"], strict=True))

    ensembl_to_symbol = (
        biomart.loc[biomart["gene_symbol"] != "", ["ensembl_gene_id", "gene_symbol"]]
        .drop_duplicates()
        .drop_duplicates(subset=["ensembl_gene_id"])
        .set_index("ensembl_gene_id")["gene_symbol"]
        .to_dict()
    )

    alias_map: dict[str, set[str]] = {}
    for _, row in hgnc.iterrows():
        ensembl_gene_id = row.get("ensembl_gene_id", "")
        if not ensembl_gene_id:
            continue

        alias_tokens = []
        for column in ["alias_symbol", "prev_symbol"]:
            raw_value = row.get(column, "")
            if pd.isna(raw_value) or not raw_value:
                continue
            alias_tokens.extend(token.strip() for token in str(raw_value).split("|"))

        for alias in alias_tokens:
            if not alias:
                continue
            alias_map.setdefault(alias, set()).add(ensembl_gene_id)

    if "entrez_id" in hgnc.columns:
        entrez_df = hgnc.loc[
            (hgnc["entrez_id"] != "") & (hgnc["ensembl_gene_id"] != ""),
            ["entrez_id", "ensembl_gene_id"],
        ].drop_duplicates()
        entrez_counts = entrez_df.groupby("entrez_id")["ensembl_gene_id"].nunique()
        entrez_df = entrez_df.loc[entrez_df["entrez_id"].isin(entrez_counts.index[entrez_counts == 1])]
        entrez_to_ens_map = dict(zip(entrez_df["entrez_id"], entrez_df["ensembl_gene_id"], strict=True))
    else:
        entrez_to_ens_map = {}

    return biomart_exact_symbol_map, hgnc_exact_symbol_map, alias_map, ensembl_to_symbol, entrez_to_ens_map


def _pick_source_series(
    adata: ad.AnnData,
    *,
    feature_id_column: str | None,
    gene_symbol_column: str | None,
) -> pd.DataFrame:
    source = pd.DataFrame(index=adata.var_names.astype(str))
    source["original_var_name"] = adata.var_names.astype(str)

    if feature_id_column and feature_id_column in adata.var.columns:
        source["source_gene_id"] = adata.var[feature_id_column].astype(str).to_numpy()
    else:
        source["source_gene_id"] = adata.var_names.astype(str)

    if gene_symbol_column and gene_symbol_column in adata.var.columns:
        source["source_gene_symbol"] = adata.var[gene_symbol_column].astype(str).to_numpy()
    elif "gene_symbol" in adata.var.columns:
        source["source_gene_symbol"] = adata.var["gene_symbol"].astype(str).to_numpy()
    else:
        source["source_gene_symbol"] = adata.var_names.astype(str)

    return source


def _map_features(
    source_frame: pd.DataFrame,
    reference_dir: Path,
) -> pd.DataFrame:
    (
        biomart_exact_symbol_map,
        hgnc_exact_symbol_map,
        alias_symbol_map,
        ensembl_to_symbol,
        entrez_to_ens_map,
    ) = _build_symbol_maps(reference_dir)

    mapping = source_frame.copy()
    mapping["ensembl_gene_id"] = ""
    mapping["resolved_gene_symbol"] = ""
    mapping["mapping_status"] = "unmapped"

    for index, row in mapping.iterrows():
        source_gene_id = row["source_gene_id"]
        source_symbol = row["source_gene_symbol"]

        if ENSEMBL_PATTERN.match(source_gene_id):
            ensembl_gene_id = re.sub(r"\.[0-9]+$", "", source_gene_id)
            mapping.at[index, "ensembl_gene_id"] = ensembl_gene_id
            mapping.at[index, "resolved_gene_symbol"] = ensembl_to_symbol.get(ensembl_gene_id, source_symbol)
            mapping.at[index, "mapping_status"] = "ensembl_version_stripped" if "." in source_gene_id else "ensembl_direct"
            continue

        if source_gene_id in entrez_to_ens_map:
            ensembl_gene_id = entrez_to_ens_map[source_gene_id]
            mapping.at[index, "ensembl_gene_id"] = ensembl_gene_id
            mapping.at[index, "resolved_gene_symbol"] = ensembl_to_symbol.get(ensembl_gene_id, source_symbol)
            mapping.at[index, "mapping_status"] = "entrez_exact"
            continue

        if source_symbol in biomart_exact_symbol_map:
            ensembl_gene_id = biomart_exact_symbol_map[source_symbol]
            mapping.at[index, "ensembl_gene_id"] = ensembl_gene_id
            mapping.at[index, "resolved_gene_symbol"] = ensembl_to_symbol.get(ensembl_gene_id, source_symbol)
            mapping.at[index, "mapping_status"] = "symbol_exact"
            continue

        if source_symbol in hgnc_exact_symbol_map:
            ensembl_gene_id = hgnc_exact_symbol_map[source_symbol]
            mapping.at[index, "ensembl_gene_id"] = ensembl_gene_id
            mapping.at[index, "resolved_gene_symbol"] = ensembl_to_symbol.get(ensembl_gene_id, source_symbol)
            mapping.at[index, "mapping_status"] = "symbol_exact_hgnc"
            continue

        alias_hits = alias_symbol_map.get(source_symbol, set())
        if len(alias_hits) == 1:
            ensembl_gene_id = next(iter(alias_hits))
            mapping.at[index, "ensembl_gene_id"] = ensembl_gene_id
            mapping.at[index, "resolved_gene_symbol"] = ensembl_to_symbol.get(ensembl_gene_id, source_symbol)
            mapping.at[index, "mapping_status"] = "symbol_alias"
            continue

        if len(alias_hits) > 1:
            mapping.at[index, "mapping_status"] = "ambiguous_alias"

    return mapping


def _collapse_duplicate_ensembl_ids(adata: ad.AnnData, mapping: pd.DataFrame) -> tuple[ad.AnnData, pd.DataFrame]:
    keep_mask = mapping["ensembl_gene_id"] != ""
    mapped = mapping.loc[keep_mask].copy()

    if mapped.empty:
        raise ValueError("No features could be mapped to Ensembl gene IDs.")

    kept_var_positions = np.where(keep_mask.to_numpy())[0]
    group_codes, unique_gene_ids = pd.factorize(mapped["ensembl_gene_id"], sort=True)
    aggregation = sparse.coo_matrix(
        (
            np.ones(group_codes.shape[0], dtype=np.int64),
            (np.arange(group_codes.shape[0]), group_codes),
        ),
        shape=(group_codes.shape[0], unique_gene_ids.shape[0]),
    ).tocsr()

    matrix = adata.X[:, kept_var_positions]
    matrix = matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
    collapsed_matrix = matrix @ aggregation

    collapsed_var = (
        mapped.assign(_group=group_codes)
        .groupby("_group", sort=True)
        .agg(
            ensembl_gene_id=("ensembl_gene_id", "first"),
            gene_symbol=("resolved_gene_symbol", "first"),
            mapping_status=("mapping_status", lambda values: "merged" if len(set(values)) > 1 else values.iloc[0]),
            merged_from=("original_var_name", lambda values: "|".join(values.astype(str))),
        )
        .set_index("ensembl_gene_id")
    )

    collapsed = ad.AnnData(X=collapsed_matrix, obs=adata.obs.copy(), var=collapsed_var)
    for column in adata.obs.columns:
        collapsed.obs[column] = adata.obs[column].to_numpy()

    if "counts" in adata.layers:
        layer_counts = adata.layers["counts"][:, kept_var_positions]
        layer_counts = layer_counts.tocsr() if sparse.issparse(layer_counts) else sparse.csr_matrix(layer_counts)
        collapsed.layers["counts"] = layer_counts @ aggregation

    return collapsed, mapped


def standardize_gene_ids(
    adata: ad.AnnData,
    reference_dir: Path | str,
    *,
    feature_id_column: str | None = None,
    gene_symbol_column: str | None = None,
) -> tuple[ad.AnnData, pd.DataFrame, dict]:
    reference_dir = Path(reference_dir)
    source_frame = _pick_source_series(
        adata,
        feature_id_column=feature_id_column,
        gene_symbol_column=gene_symbol_column,
    )
    mapping = _map_features(source_frame, reference_dir)
    standardized, mapped = _collapse_duplicate_ensembl_ids(adata, mapping)

    summary = {
        "input_features": int(mapping.shape[0]),
        "mapped_features": int((mapping["ensembl_gene_id"] != "").sum()),
        "unmapped_features": int((mapping["mapping_status"] == "unmapped").sum()),
        "ambiguous_alias_features": int((mapping["mapping_status"] == "ambiguous_alias").sum()),
        "duplicate_ensembl_collapses": int(mapped["ensembl_gene_id"].duplicated().sum()),
        "mapping_status_counts": mapping["mapping_status"].value_counts().to_dict(),
    }

    standardized.uns["gene_id_mapping_summary"] = json.loads(json.dumps(summary))
    standardized.var_names = standardized.var.index.astype(str)
    standardized.var["source_feature"] = standardized.var["merged_from"].to_numpy()
    standardized.obs.index = standardized.obs.index.astype(str)
    standardized.var.index = standardized.var.index.astype(str)
    standardized.obs.index.name = "cell_barcode"
    standardized.var.index.name = "gene_id"

    return standardized, mapping, summary


def dataset_output_paths(
    project_dir: Path | str,
    dataset_name: str,
    *,
    source_filename: str | None = None,
) -> dict[str, Path]:
    project_dir = Path(project_dir)
    dataset_config = resolve_dataset_config(dataset_name, source_filename=source_filename)
    prep_dir = project_dir / "result" / dataset_name / "prep"

    return {
        "source_path": project_dir / "input" / dataset_config["source_filename"],
        "processed_path": project_dir / "input" / "processed" / f"{dataset_name}.h5ad",
        "prep_dir": prep_dir,
        "summary_path": prep_dir / "prep_summary.json",
        "mapping_path": prep_dir / "gene_id_mapping.tsv.gz",
        "reference_dir": REFERENCE_DIR,
    }


def prepare_downloaded_dataset(
    project_dir: Path | str,
    dataset_name: str,
    *,
    source_filename: str | None = None,
    display_name: str | None = None,
    force_rebuild: bool = False,
) -> tuple[ad.AnnData, dict]:
    project_dir = Path(project_dir)
    dataset_config = resolve_dataset_config(
        dataset_name,
        source_filename=source_filename,
        display_name=display_name,
    )
    output_paths = dataset_output_paths(
        project_dir,
        dataset_name,
        source_filename=dataset_config["source_filename"],
    )

    if (
        output_paths["processed_path"].exists()
        and output_paths["summary_path"].exists()
        and not force_rebuild
    ):
        existing_adata = ad.read_h5ad(output_paths["processed_path"])
        existing_report = json.loads(output_paths["summary_path"].read_text(encoding="utf-8"))
        return existing_adata, existing_report

    adata = ad.read_h5ad(output_paths["source_path"])
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)
    adata.obs.index.name = "cell_barcode"
    adata.var.index.name = "feature_id"
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata = _normalize_obs_index_column_conflict(adata)

    standardized, mapping, mapping_summary = standardize_gene_ids(
        adata,
        output_paths["reference_dir"],
        gene_symbol_column="gene_symbol" if "gene_symbol" in adata.var.columns else None,
    )

    standardized.obs.index = standardized.obs.index.astype(str)
    standardized.var.index = standardized.var.index.astype(str)
    standardized.obs.index.name = "cell_barcode"
    standardized.var.index.name = "gene_id"
    standardized.obs_names_make_unique()
    standardized.var_names_make_unique()
    standardized = _normalize_obs_index_column_conflict(standardized)

    ensure_directory(output_paths["processed_path"].parent)
    ensure_directory(output_paths["prep_dir"])
    standardized.write_h5ad(output_paths["processed_path"])
    mapping.to_csv(output_paths["mapping_path"], sep="\t", index=False, compression="gzip")

    report = {
        "dataset_name": dataset_name,
        "display_name": str(dataset_config["display_name"]),
        "source_file": str(output_paths["source_path"].relative_to(project_dir)),
        "processed_file": str(output_paths["processed_path"].relative_to(project_dir)),
        "n_obs": int(standardized.n_obs),
        "n_vars": int(standardized.n_vars),
        "obs_columns": sorted(str(column) for column in standardized.obs.columns),
        "var_columns": sorted(str(column) for column in standardized.var.columns),
        "mapping_summary": mapping_summary,
        "uns_keys": sorted(str(key) for key in standardized.uns.keys()),
    }
    write_json_report(report, output_paths["summary_path"])

    return standardized, report
