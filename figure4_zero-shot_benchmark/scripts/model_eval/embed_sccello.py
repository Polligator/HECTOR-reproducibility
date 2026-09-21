#!/usr/bin/env python3
"""scCello embedding generator using official tokenization pipeline.

Uses the scCello package's ProcessSingleCellData.from_h5ad_adata() for proper
gene tokenization (Geneformer-style rank encoding with median normalization),
then feeds tokenized sequences through the scCello-zeroshot checkpoint.

The model architecture (PrototypeContrastiveModel) extends BertModel with custom
embedding layers (tf_class, tf_superclass, expbin). These are replicated here as
a standalone class to avoid circular import issues in the scCello package.

Usage:
    conda run -n sccello python model_eval/embed_sccello.py
    conda run -n sccello python model_eval/embed_sccello.py --n-cells 500 --device cuda
"""

import argparse
import functools
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPreTrainedModel,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _SCRIPT_DIR.parents[1]   # the figure folder
_SCCELLO_REPO = _SCRIPT_DIR.parents[1] / "models" / "scCello" / "scCello-repo"
_DEFAULT_INPUT = _WORKSPACE_ROOT / "input" / "TS_ALL_cells_10k.h5ad"
_DEFAULT_CHECKPOINT = _SCRIPT_DIR.parents[1] / "models" / "scCello" / "checkpoints" / "scCello-zeroshot"
_DEFAULT_OUTPUT = _WORKSPACE_ROOT / "result" / "sccello_embeddings.pkl"

MAX_CELLS = 100_000
SEED = 42

if str(_SCCELLO_REPO) not in sys.path:
    sys.path.insert(0, str(_SCCELLO_REPO))

from sccello.src.data.data_proc import PROTOCOL_DICT, ProcessSingleCellData
from sccello.src.utils import data_loading


# ---------------------------------------------------------------------------
# Custom model matching scCello's PrototypeContrastiveModel architecture
# ---------------------------------------------------------------------------
class ScCelloEmbeddings(nn.Module):
    """Replicates PrototypeContrastiveEmbeddings from scCello.

    Adds tf_class, tf_superclass, and expbin embedding layers on top of
    standard BERT word + position embeddings. In the pretrained checkpoint
    these are single-entry embeddings (vocab_size=1) always indexed at 0,
    acting as learned bias vectors.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size,
            padding_idx=config.pad_token_id,
            scale_grad_by_freq=getattr(config, "scale_grad_by_freq", False),
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.tf_class_embeddings = nn.Embedding(1, config.hidden_size)
        self.tf_superclass_embeddings = nn.Embedding(1, config.hidden_size)
        self.expbin_embeddings = nn.Embedding(1, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

    def forward(self, input_ids, **kwargs):
        """Forward pass matching scCello's embedding logic."""
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.word_embeddings(input_ids)

        zeros = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        tf_class_emb = self.tf_class_embeddings(zeros)
        tf_superclass_emb = self.tf_superclass_embeddings(zeros)
        expbin_emb = self.expbin_embeddings(zeros)

        embeddings = inputs_embeds + tf_class_emb + tf_superclass_emb + expbin_emb
        if self.position_embedding_type == "absolute":
            embeddings = embeddings + self.position_embeddings(position_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ScCelloBertPooler(nn.Module):
    """Standard BERT pooler (first token → dense → tanh)."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        return self.activation(self.dense(hidden_states[:, 0]))


class ScCelloModel(BertPreTrainedModel):
    """Standalone replica of scCello's PrototypeContrastiveModel.

    Matches the checkpoint key structure (bert.embeddings.*, bert.encoder.*,
    bert.pooler.*) so weights load directly with strict=True on the bert.* subset.
    """

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = ScCelloEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = ScCelloBertPooler(config)
        self.post_init()

    def forward(self, input_ids, attention_mask=None):
        """Minimal forward for embedding extraction."""
        if attention_mask is not None:
            extended_mask = self.get_extended_attention_mask(
                attention_mask, input_ids.shape
            )
        else:
            extended_mask = None

        embedding_output = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_mask)
        sequence_output = encoder_outputs[0]
        return sequence_output


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_and_subsample(input_path: str, n_cells: Optional[int], seed: int = SEED):
    """Load h5ad and optionally subsample *n_cells*."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    adata = sc.read_h5ad(str(path), backed="r")
    total_cells = int(adata.n_obs)

    if n_cells is None:
        target_n_cells = total_cells
    else:
        target_n_cells = int(n_cells)

    if n_cells is not None and target_n_cells > MAX_CELLS:
        raise ValueError(f"Max supported: {MAX_CELLS:,}, got {target_n_cells:,}")
    if target_n_cells > total_cells:
        raise ValueError(
            f"Requested {target_n_cells:,} cells but dataset only has {total_cells:,}."
        )

    if target_n_cells == total_cells:
        indices = None
        adata = adata.to_memory()
    else:
        np.random.seed(seed)
        indices = np.random.choice(total_cells, target_n_cells, replace=False)
        adata = adata[indices].to_memory()

    if "decontXcounts" in adata.layers:
        adata.X = adata.layers["decontXcounts"].copy()

    sampled_obs_indices = None if indices is None else indices.astype(int).tolist()
    return adata, sampled_obs_indices


def _get_row_ids(adata) -> list[str]:
    for key in ("observation_joinid", "soma_joinid", "_index"):
        if key in adata.obs:
            return adata.obs[key].astype(str).tolist()
    return adata.obs_names.astype(str).tolist()


def _get_sccello_kept_row_indices(adata) -> list[int]:
    mask = np.ones(adata.n_obs, dtype=bool)
    if "organism" in adata.obs:
        mask &= adata.obs["organism"].astype(str).to_numpy() == "Homo sapiens"
    if "assay" in adata.obs:
        mask &= adata.obs["assay"].isin(PROTOCOL_DICT.keys()).to_numpy()
    if "is_primary_data" in adata.obs:
        mask &= adata.obs["is_primary_data"].astype(bool).to_numpy()
    return np.flatnonzero(mask).tolist()


@functools.lru_cache(maxsize=1)
def _get_sccello_vocab_ids() -> frozenset[str]:
    token_dict = data_loading.get_prestored_data("token_dictionary_pkl")
    return frozenset(str(gene_id) for gene_id in token_dict if gene_id != "<cls>")


def _normalize_gene_ids(values) -> np.ndarray:
    values = np.asarray(values, dtype=str)
    normalized = []
    for value in values:
        if value in {"", "None", "nan"}:
            normalized.append("")
            continue
        normalized.append(value.split(".", 1)[0])
    return np.asarray(normalized, dtype=object)


def _score_gene_id_source(values, valid_ids: frozenset[str]) -> tuple[float, float, np.ndarray]:
    normalized = _normalize_gene_ids(values)
    hit_rate = float(np.mean([gene_id in valid_ids for gene_id in normalized]))
    ensembl_rate = float(np.mean([gene_id.startswith("ENSG") for gene_id in normalized]))
    return hit_rate, ensembl_rate, normalized


def _prepare_sccello_gene_ids(adata) -> None:
    """Populate adata.var['ensembl_ids'] when the dataset already exposes Ensembl IDs.

    Some benchmark files store true Ensembl IDs in columns like ``feature_id`` while
    ``var_names`` are numeric placeholders. In that case scCello's default
    ``map_gene_name2id(var_names)`` path silently produces almost-empty token
    sequences. This helper picks the best direct Ensembl-ID source if one exists,
    otherwise it leaves ``ensembl_ids`` unset so scCello can fall back to gene-name
    mapping on ``var_names``.
    """

    valid_ids = _get_sccello_vocab_ids()
    candidates = []

    if "ensembl_ids" in adata.var:
        candidates.append(("ensembl_ids", adata.var["ensembl_ids"].astype(str).to_numpy()))

    for col in ("feature_id", "ensembl_id", "gene_ids", "gene_id"):
        if col in adata.var:
            candidates.append((col, adata.var[col].astype(str).to_numpy()))

    candidates.append(("var_names", np.asarray(adata.var_names.astype(str), dtype=object)))

    scored = []
    for name, values in candidates:
        hit_rate, ensembl_rate, normalized = _score_gene_id_source(values, valid_ids)
        scored.append((name, hit_rate, ensembl_rate, normalized))

    best_name, best_hit_rate, best_ensembl_rate, best_ids = max(
        scored, key=lambda item: (item[1], item[2])
    )

    if best_hit_rate > 0:
        adata.var["ensembl_ids"] = best_ids
        print(
            "  Gene IDs: using "
            f"{best_name} as ensembl_ids "
            f"(vocab hit rate {best_hit_rate:.1%}, ENSG rate {best_ensembl_rate:.1%})"
        )
        return

    if "ensembl_ids" in adata.var:
        adata.var = adata.var.drop(columns=["ensembl_ids"])

    print(
        "  Gene IDs: no direct Ensembl-ID source found; "
        "falling back to var_names gene-name mapping"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_sccello_model(checkpoint_dir: str, device: str = "cpu"):
    """Load scCello model from safetensors checkpoint.

    Returns the ScCelloModel with weights from the bert.* subset of the
    checkpoint, plus the config dict.
    """
    ckpt = Path(checkpoint_dir)
    config_path = ckpt / "config.json"
    weights_path = ckpt / "model.safetensors"
    for p in (config_path, weights_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    with open(config_path) as f:
        config_dict = json.load(f)
    if config_dict.get("type_vocab_size") is None:
        config_dict["type_vocab_size"] = 2

    config = BertConfig(**config_dict)
    model = ScCelloModel(config)

    state_dict = load_file(str(weights_path))
    model_keys = set(model.state_dict().keys())

    mapped = {}
    for k, v in state_dict.items():
        clean = k[len("bert."):] if k.startswith("bert.") else k
        if clean in model_keys:
            mapped[clean] = v

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (expected for pooler if unused)")

    model.eval()
    model.to(device)
    return model, config_dict


# ---------------------------------------------------------------------------
# Tokenization via scCello package
# ---------------------------------------------------------------------------
def tokenize_with_sccello(adata, max_length: int = 2048):
    """Tokenize cells using scCello's ProcessSingleCellData.from_h5ad_adata().

    Uses the official scCello tokenization pipeline which:
    1. Filters genes to those in scCello's vocabulary (~25k Ensembl IDs)
    2. Applies size-factor normalization (scaling to 10000)
    3. Normalizes by gene median expression (from pretrained dictionary)
    4. Ranks genes by normalized expression (descending)
    5. Maps Ensembl IDs to token IDs via the pretrained vocabulary
    6. Prepends <cls> token (id=25426)

    Tabula Sapiens is a cellxgene dataset, so is_cellxgene=True is used.
    This filters out Smart-seq2/3 cells (~3.5%) that aren't in the protocol dict.

    Args:
        adata: AnnData with raw counts and cellxgene metadata.
        max_length: Maximum sequence length (2048 for scCello).

    Returns:
        Tuple of (input_ids, attention_masks, cell_types, cell_type_ids,
        n_filtered) where
        input_ids and attention_masks are numpy arrays of shape (n_cells, max_length).
    """
    adata = adata.copy()
    if "organism" not in adata.obs:
        adata.obs["organism"] = "Homo sapiens"
    _prepare_sccello_gene_ids(adata)

    n_before = adata.n_obs
    dataset = ProcessSingleCellData.from_h5ad_adata(
        adata, "benchmark", is_cellxgene=True, require_raw=False, batch_size=512
    )
    n_after = dataset.num_rows
    n_filtered = n_before - n_after

    cell_types = dataset["cell_type"]
    cell_type_ids = dataset["cell_ct_ontology"]
    all_token_ids = dataset["gene_token_ids"]

    # Pad/truncate to max_length and build attention masks
    input_ids = np.zeros((n_after, max_length), dtype=np.int64)
    attention_masks = np.zeros((n_after, max_length), dtype=np.int64)

    for i, tokens in enumerate(all_token_ids):
        seq_len = min(len(tokens), max_length)
        input_ids[i, :seq_len] = tokens[:seq_len]
        attention_masks[i, :seq_len] = 1

    seq_lengths = attention_masks.sum(axis=1)
    if seq_lengths.size and int(np.median(seq_lengths)) <= 1:
        raise RuntimeError(
            "scCello tokenization produced median sequence length <= 1 token. "
            "This usually means the dataset's gene identifiers were not mapped "
            "to Ensembl IDs correctly."
        )

    return input_ids, attention_masks, cell_types, cell_type_ids, n_filtered


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------
def generate_embeddings(adata, checkpoint_dir: str, device: str = "cpu",
                        batch_size: int = 32):
    """Generate 256-dim scCello embeddings via official tokenization + model.

    Args:
        adata: AnnData with raw counts and cellxgene metadata.
        checkpoint_dir: Path to scCello-zeroshot checkpoint.
        device: 'cpu' or 'cuda'.
        batch_size: Inference batch size.

    Returns:
        Tuple of (embeddings, cell_types, cell_type_ids, kept_row_indices).
    """
    model, config_dict = load_sccello_model(checkpoint_dir, device)
    max_length = config_dict["max_position_embeddings"]
    hidden_size = config_dict["hidden_size"]

    print(f"  Model: vocab={config_dict['vocab_size']}, hidden={hidden_size}, "
          f"max_len={max_length}")
    print(f"  Device: {device}")

    # Tokenize using scCello pipeline
    print(f"  Tokenizing {adata.n_obs} cells (scCello pipeline)...")
    kept_row_indices = _get_sccello_kept_row_indices(adata)
    input_ids, attention_masks, cell_types, cell_type_ids, n_filtered = tokenize_with_sccello(
        adata, max_length
    )
    n_cells = input_ids.shape[0]
    if n_filtered > 0:
        print(f"  Filtered {n_filtered} cells (non-10x assay)")
    print(f"  Tokenized {n_cells} cells")

    seq_lengths = attention_masks.sum(axis=1)
    print(f"  Tokens/cell: min={seq_lengths.min()}, "
          f"median={int(np.median(seq_lengths))}, max={seq_lengths.max()}")

    # Forward pass — mean-pool over non-padding positions
    print(f"  Generating embeddings (batch_size={batch_size})...")
    embeddings = []

    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            ids = torch.tensor(input_ids[start:end], device=device)
            mask = torch.tensor(attention_masks[start:end], device=device)

            hidden = model(input_ids=ids, attention_mask=mask)  # (batch, seq, 256)

            mask_exp = mask.unsqueeze(-1).float()
            pooled = (hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
            embeddings.append(pooled.cpu().numpy())

            if (start // batch_size) % 10 == 0:
                print(f"    Processed {end}/{n_cells} cells")

    embeddings = np.vstack(embeddings)

    if np.isnan(embeddings).any():
        raise RuntimeError(f"Embeddings contain NaN values")

    print(f"  Embedding shape: {embeddings.shape}")
    return embeddings, list(cell_types), list(cell_type_ids), kept_row_indices


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    """CLI entry point for scCello embedding generation."""
    parser = argparse.ArgumentParser(
        description="Generate scCello embeddings for Tabula Sapiens benchmark"
    )
    parser.add_argument("--input", "-i", default=str(_DEFAULT_INPUT),
                        help=f"Input h5ad file (default: {_DEFAULT_INPUT})")
    parser.add_argument("--n-cells", "-n", type=int, default=None,
                        help="Number of cells to subsample; omit to use all cells")
    parser.add_argument("--output", "-o", default=str(_DEFAULT_OUTPUT),
                        help=f"Output pickle path (default: {_DEFAULT_OUTPUT})")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Compute device (default: cpu)")
    args = parser.parse_args()

    print("=" * 60)
    print("scCello Embedding Generation")
    print("=" * 60)

    if args.n_cells is None:
        print("\n[1/4] Loading all cells...")
    else:
        print(f"\n[1/4] Loading and subsampling {args.n_cells} cells...")
    adata, sampled_obs_indices = load_and_subsample(args.input, args.n_cells)
    requested_n_cells = int(args.n_cells) if args.n_cells is not None else int(adata.n_obs)
    input_n_cells = int(adata.n_obs)
    print(f"  Shape: {adata.shape}")
    print(f"  Unique cell types: {len(set(adata.obs['cell_type'].values))}")
    row_ids = _get_row_ids(adata)
    dataset_name = Path(args.input).name

    print("\n[2/4] Generating embeddings...")
    embeddings, cell_types, cell_type_ids, kept_row_indices = generate_embeddings(
        adata, str(_DEFAULT_CHECKPOINT), args.device
    )
    kept_row_ids = [row_ids[idx] for idx in kept_row_indices]

    print(f"\n[3/4] Validation passed")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Cell types: {len(set(cell_types))} unique")
    print(f"  Sample (cell 0, first 5 dims): {embeddings[0, :5]}")

    print(f"\n[4/4] Saving results...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "embeddings": embeddings,
        "cell_types": cell_types,
        "cell_type_ids": cell_type_ids,
        "row_ids": kept_row_ids,
        "dataset_name": dataset_name,
        "sampled_obs_indices": sampled_obs_indices,
        "kept_row_indices": kept_row_indices,
        "requested_n_cells": requested_n_cells,
        "input_n_cells": input_n_cells,
        "output_n_cells": int(embeddings.shape[0]),
        "model": "scCello-zeroshot",
        "source": dataset_name,
        "n_cells": len(cell_types),
        "embedding_dim": embeddings.shape[1],
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"  Saved to {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
