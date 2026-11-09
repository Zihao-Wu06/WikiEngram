"""General-text corpora for base Engram adaptation, locality replay and background row statistics.

Sources:
* a local ``.txt`` (blank-line separated documents) or ``.jsonl`` (``{"text": ...}``) file;
* ``hf:<repo>/<file.parquet>[:column]`` - a parquet file from the Hugging Face Hub (needs ``pyarrow``);
* ``wikitext-2`` / ``wikitext-103`` - shortcuts for Salesforce/wikitext raw splits (CC BY-SA 3.0, downloaded
  at run time, not redistributed).
"""

from __future__ import annotations

import json
from pathlib import Path

_SHORTCUTS = {
    "wikitext-2": ("Salesforce/wikitext", "wikitext-2-raw-v1/{split}-00000-of-00001.parquet", "text"),
    "wikitext-103": ("Salesforce/wikitext", "wikitext-103-raw-v1/{split}-00000-of-00002.parquet", "text"),
}


def _wikitext_docs(lines: list[str]) -> list[str]:
    """Wikitext stores one paragraph per row with ' = Title = ' headings; regroup into articles."""
    docs, cur = [], []
    for line in lines:
        s = line.strip()
        if s.startswith("= ") and s.endswith(" =") and not s.startswith("= ="):
            if cur:
                docs.append(" ".join(cur))
            cur = [s.strip("= ").strip() + "."]
        elif s:
            cur.append(s)
    if cur:
        docs.append(" ".join(cur))
    return docs


def load_texts(source: str, split: str = "train", limit_docs: int | None = None) -> list[str]:
    """Documents of ``source``. Wikitext has real splits; for other sources the last 5% of documents form the
    ``validation`` split and the rest ``train``, so evaluation never reuses training text."""
    if source in _SHORTCUTS:
        repo, pattern, col = _SHORTCUTS[source]
        return _wikitext_docs(_read_parquet_column(repo, pattern.format(split=split), col))[:limit_docs]
    if source.startswith("hf:"):
        spec = source[3:]
        col = "text"
        if ":" in spec:
            spec, col = spec.rsplit(":", 1)
        repo, _, file = spec.partition("/")
        repo2, _, file2 = file.partition("/")
        docs = _read_parquet_column(f"{repo}/{repo2}", file2, col)
    else:
        path = Path(source)
        if path.suffix == ".jsonl":
            docs = [json.loads(line)["text"] for line in path.read_text().splitlines() if line.strip()]
        else:
            docs = [d.strip() for d in path.read_text().split("\n\n") if d.strip()]
    return _split(docs, split)[:limit_docs]


def _split(docs: list[str], split: str) -> list[str]:
    if len(docs) < 2:
        return docs
    n_train = min(len(docs) - 1, max(1, int(len(docs) * 0.95)))
    return docs[:n_train] if split == "train" else docs[n_train:]


def _read_parquet_column(repo: str, filename: str, column: str) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading Hub parquet corpora needs pyarrow: pip install 'wikiengram[data]'") from e
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")
    return [t for t in pq.read_table(local, columns=[column]).column(column).to_pylist() if t]
