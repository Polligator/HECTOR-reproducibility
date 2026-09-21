"""Minimal torchtext compatibility shim for scGPT.

Provides the Vocab class that scGPT's gene_tokenizer.py requires,
without depending on the deprecated torchtext package (removed in PyTorch 2.5+).
"""

from . import vocab

__version__ = "0.0.1"
