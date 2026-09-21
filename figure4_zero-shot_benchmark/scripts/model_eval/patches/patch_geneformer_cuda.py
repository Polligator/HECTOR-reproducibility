#!/usr/bin/env python3
"""Patch Geneformer source files for CUDA-compatible embedding extraction.

The benchmark's embed_geneformer.py imports the curated local repo under
model_eval/geneformer, so the patch target must match the code path actually
executed by the benchmark. This helper can patch either:

- a repo root containing `geneformer/emb_extractor.py`, or
- an installed `geneformer` package discovered from the current environment.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def patch_emb_extractor(path: str) -> bool:
    """Patch emb_extractor.py to use model device instead of hardcoded 'cpu'."""
    with open(path) as f:
        content = f.read()

    if "model_device = next(model.parameters()).device" in content:
        print(f"  emb_extractor.py: already patched")
        return True

    # 1. Add model_device detection before the loop
    old1 = '    overall_max_len = 0\n\n    for i in trange('
    new1 = (
        '    overall_max_len = 0\n\n'
        '    # Detect model device once for consistent tensor placement\n'
        '    model_device = next(model.parameters()).device\n\n'
        '    for i in trange('
    )
    if old1 not in content:
        print(f"  emb_extractor.py: cannot find loop anchor, skipping")
        return False
    content = content.replace(old1, new1, 1)

    # 2. Fix original_lens device
    content = content.replace(
        'original_lens = torch.tensor(minibatch["length"], device="cpu")',
        'original_lens = torch.tensor(minibatch["length"], device=model_device)',
        1,
    )

    # 3. Fix input_ids and attention_mask device
    old3 = (
        '        with torch.no_grad():\n'
        '            outputs = model(\n'
        '                input_ids=input_data_minibatch.to("cpu"),\n'
        '                attention_mask=pu.gen_attention_mask(minibatch),\n'
        '            )'
    )
    new3 = (
        '        # Send inputs to the same device as the model (fixes CPU/CUDA mismatch)\n'
        '        with torch.no_grad():\n'
        '            outputs = model(\n'
        '                input_ids=input_data_minibatch.to(model_device),\n'
        '                attention_mask=pu.gen_attention_mask(minibatch, device=model_device),\n'
        '            )'
    )
    if old3 not in content:
        print(f"  emb_extractor.py: cannot find forward pass anchor, skipping")
        return False
    content = content.replace(old3, new3, 1)

    with open(path, "w") as f:
        f.write(content)
    print(f"  emb_extractor.py: patched")
    return True


def patch_perturber_utils(path: str) -> bool:
    """Patch gen_attention_mask to accept a device parameter."""
    with open(path) as f:
        content = f.read()

    if 'def gen_attention_mask(minibatch_encoding, max_len=None, device="cpu")' in content:
        print(f"  perturber_utils.py: already patched")
        return True

    old = (
        'def gen_attention_mask(minibatch_encoding, max_len=None):\n'
        '    if max_len is None:\n'
        '        max_len = max(minibatch_encoding["length"])\n'
        '    original_lens = minibatch_encoding["length"]\n'
        '    attention_mask = [\n'
        '        [1] * original_len + [0] * (max_len - original_len)\n'
        '        if original_len <= max_len\n'
        '        else [1] * max_len\n'
        '        for original_len in original_lens\n'
        '    ]\n'
        '    return torch.tensor(attention_mask, device="cpu")'
    )
    new = (
        'def gen_attention_mask(minibatch_encoding, max_len=None, device="cpu"):\n'
        '    """Generate attention mask for a minibatch.\n'
        '\n'
        '    Patched to accept a device parameter for CUDA compatibility.\n'
        '    """\n'
        '    if max_len is None:\n'
        '        max_len = max(minibatch_encoding["length"])\n'
        '    original_lens = minibatch_encoding["length"]\n'
        '    attention_mask = [\n'
        '        [1] * original_len + [0] * (max_len - original_len)\n'
        '        if original_len <= max_len\n'
        '        else [1] * max_len\n'
        '        for original_len in original_lens\n'
        '    ]\n'
        '    return torch.tensor(attention_mask, device=device)'
    )

    if old not in content:
        print(f"  perturber_utils.py: cannot find gen_attention_mask anchor, skipping")
        return False

    content = content.replace(old, new, 1)

    with open(path, "w") as f:
        f.write(content)
    print(f"  perturber_utils.py: patched")
    return True


def resolve_package_dir(target: str | None) -> Path:
    """Resolve the package directory that should be patched."""
    if target is not None:
        path = Path(target).resolve()
        if (path / "geneformer" / "emb_extractor.py").exists():
            return path / "geneformer"
        if (path / "emb_extractor.py").exists():
            return path
        raise FileNotFoundError(
            f"Could not find Geneformer source under target path: {path}"
        )

    try:
        import geneformer  # noqa: F401
    except ImportError:
        print("ERROR: geneformer is not installed and no --target path was provided")
        sys.exit(1)

    return Path(importlib.util.find_spec("geneformer").submodule_search_locations[0])


def main():
    """Find and patch the requested Geneformer source path."""
    parser = argparse.ArgumentParser(
        description="Patch Geneformer source files for CUDA compatibility."
    )
    parser.add_argument(
        "--target",
        default=None,
        help=(
            "Optional repo root or package directory to patch. If omitted, patch "
            "the installed geneformer package in the current environment."
        ),
    )
    args = parser.parse_args()

    pkg_dir = resolve_package_dir(args.target)
    print(f"Geneformer package at: {pkg_dir}")

    emb_path = str(pkg_dir / "emb_extractor.py")
    pu_path = str(pkg_dir / "perturber_utils.py")

    ok1 = patch_emb_extractor(emb_path)
    ok2 = patch_perturber_utils(pu_path)

    if ok1 and ok2:
        print("Geneformer CUDA patches applied successfully.")
    else:
        print("WARNING: Some patches could not be applied.")
        sys.exit(1)


if __name__ == "__main__":
    main()
