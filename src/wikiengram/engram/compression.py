"""Tokenizer compression ("canonical ids") for Engram n-gram hashing.

Ported from DeepSeek-V4.1-Flash ``inference/engram.py`` (MIT License, Copyright (c) 2023 DeepSeek).
Tokens that normalize alike (NFKC, NFD, accents stripped, lowercased, whitespace collapsed) share one
compressed id, so " The", "the" and "THE" hash identically. Every hash multiplier depends on the size of
the compressed vocab, so the size is stored with every Engram checkpoint and checked on load.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# a private-use char, so a token that is exactly one space survives Strip() instead of collapsing to the
# empty string and merging with unrelated tokens (same trick as the reference implementation)
_SENTINEL = ""


def _normalizer():
    from tokenizers import Regex, normalizers

    return normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), _SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(_SENTINEL, " "),
        ]
    )


def _backend_and_size(tokenizer) -> tuple[object, int]:
    """Accept a transformers fast tokenizer or a raw ``tokenizers.Tokenizer``."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        return backend, len(tokenizer)
    # raw tokenizers.Tokenizer
    return tokenizer, tokenizer.get_vocab_size(with_added_tokens=True)


def build_compressed_token_map(tokenizer) -> tuple[np.ndarray, int]:
    """Return ``(lookup, compressed_vocab_size)`` where ``lookup[raw_id] = compressed_id``.

    Mirrors the reference exactly: decode with the raw Rust tokenizer (no clean-up of spaces), key partial
    UTF-8 byte tokens by their raw token string, assign new ids densely in first-seen order.
    """
    normalizer = _normalizer()
    backend, size = _backend_and_size(tokenizer)
    key_to_new: dict[str, int] = {}
    lookup = np.zeros(size, dtype=np.int64)
    for token_id in range(size):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def tokenizer_fingerprint(tokenizer) -> str:
    """A stable hash of the tokenizer vocabulary, used as a cache key for the compressed map."""
    backend, size = _backend_and_size(tokenizer)
    vocab = backend.get_vocab(with_added_tokens=True)
    payload = json.dumps(sorted(vocab.items(), key=lambda kv: kv[1]), ensure_ascii=False)
    return hashlib.sha256(f"{size}|{payload}".encode()).hexdigest()[:16]


def load_or_build_token_map(tokenizer, cache_dir: str | Path | None = None) -> tuple[np.ndarray, int]:
    """Building the map decodes every token once (seconds for a 150k vocab), so cache it on disk."""
    if cache_dir is None:
        return build_compressed_token_map(tokenizer)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"token_map_{tokenizer_fingerprint(tokenizer)}.npz"
    if path.exists():
        data = np.load(path)
        return data["lookup"], int(data["size"])
    lookup, size = build_compressed_token_map(tokenizer)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, lookup=lookup, size=np.int64(size))
    tmp.replace(path)
    return lookup, size
