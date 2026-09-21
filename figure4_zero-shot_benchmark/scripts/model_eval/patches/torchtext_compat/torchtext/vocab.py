"""Minimal torchtext.vocab replacement for scGPT compatibility.

Replaces the deprecated torchtext.vocab module with a standalone Vocab class
that provides the same interface used by scGPT's GeneVocab (gene_tokenizer.py).

The original torchtext.vocab.Vocab is a C++ backed token↔index mapping.
This pure-Python replacement covers the subset of the API that scGPT uses:
  - __getitem__(token) → index
  - __contains__(token) → bool
  - __len__() → int
  - set_default_index(index)
  - insert_token(token, index)
  - get_stoi() → dict
  - vocab property
"""

from collections import OrderedDict
from typing import Dict, List, Optional


class Vocab:
    """Pure-Python replacement for torchtext.vocab.Vocab.

    Provides bidirectional token↔index mapping with a default index
    for out-of-vocabulary tokens.
    """

    def __init__(self, ordered_dict_or_vocab):
        """Initialize from an OrderedDict (token→freq) or another Vocab's internal dict.

        When constructed via the `vocab()` factory function, receives an OrderedDict
        of token→freq pairs. When constructed via copy (e.g., GeneVocab(existing_vocab)),
        receives the internal _stoi dict.
        """
        if isinstance(ordered_dict_or_vocab, dict):
            self._stoi: Dict[str, int] = dict(ordered_dict_or_vocab)
            self._itos: Dict[int, str] = {v: k for k, v in self._stoi.items()}
        else:
            self._stoi = {}
            self._itos = {}
        self._default_index: Optional[int] = None

    @property
    def vocab(self):
        """Return internal stoi dict (used by GeneVocab's super().__init__)."""
        return self._stoi

    def __getitem__(self, token: str) -> int:
        if token in self._stoi:
            return self._stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(f"Token '{token}' not found in vocabulary")

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def __len__(self) -> int:
        return len(self._stoi)

    def __call__(self, tokens: List[str]) -> List[int]:
        """Look up indices for a list of tokens (matches torchtext Vocab behavior)."""
        return [self[t] for t in tokens]

    def set_default_index(self, index: int) -> None:
        """Set the default index for OOV tokens."""
        self._default_index = index

    def insert_token(self, token: str, index: int) -> None:
        """Insert a token at a specific index."""
        self._stoi[token] = index
        self._itos[index] = token

    def get_stoi(self) -> Dict[str, int]:
        """Return the string-to-index mapping."""
        return dict(self._stoi)

    def get_itos(self) -> List[str]:
        """Return the index-to-string list."""
        max_idx = max(self._itos.keys()) if self._itos else -1
        return [self._itos.get(i, "") for i in range(max_idx + 1)]


def vocab(ordered_dict: OrderedDict, min_freq: int = 1) -> Vocab:
    """Factory function matching torchtext.vocab.vocab().

    Builds a Vocab from an OrderedDict of token→frequency pairs,
    filtering by min_freq and assigning consecutive indices.
    """
    stoi = {}
    idx = 0
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            stoi[token] = idx
            idx += 1
    return Vocab(stoi)
