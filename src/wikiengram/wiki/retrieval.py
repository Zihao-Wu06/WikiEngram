"""Lexical retrieval over claims: BM25 plus entity matching. No embeddings, by design.

Tokens are casefolded unicode words with a light English suffix stemmer ("lives"/"living" -> "liv"); runs of
CJK characters become character unigrams and bigrams so Chinese/Japanese text is searchable without a
segmenter. First-person words map to the token "user", matching claims about the user ("Where do I live?").
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Hashable, Sequence
from itertools import groupby, pairwise
from typing import Generic, TypeVar

from .models import Claim, is_cjk

K = TypeVar("K", bound=Hashable)

_WORD = re.compile(r"[^\W_]+")
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "of",
    "to", "and", "or", "it", "its", "this", "that", "these", "those",
})
_FIRST_PERSON = {"i": "user", "me": "user", "my": "user", "mine": "user", "myself": "user", "我": "user"}
_VOWELS = frozenset("aeiouy")


def stem(word: str) -> str:
    """Tiny suffix stripper, consistent across inflections: works/working/worked -> work, lives/lived -> liv."""
    if len(word) <= 3 or not (word.isascii() and word.isalpha()):
        return word
    w = word
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ing") and len(w) > 5 and _VOWELS & set(w[:-3]):
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 4 and not w.endswith("eed"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    """Search tokens of ``text`` (see module docstring)."""
    tokens: list[str] = []
    for m in _WORD.finditer(unicodedata.normalize("NFKC", text).casefold()):
        for cjk, run in groupby(m.group(), key=is_cjk):
            chars = "".join(run)
            if cjk:
                tokens.extend(_FIRST_PERSON.get(ch, ch) for ch in chars)
                tokens.extend(a + b for a, b in pairwise(chars))
                continue
            word = _FIRST_PERSON.get(chars, chars)
            if word in _STOPWORDS or (len(word) == 1 and word.isalpha()):
                continue
            tokens.append(stem(word))
    return tokens


class BM25(Generic[K]):
    """Okapi BM25 over pre-tokenized documents keyed by ``K``."""

    def __init__(self, docs: Sequence[tuple[K, Sequence[str]]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.keys = [key for key, _ in docs]
        self.lengths = [len(tokens) for _, tokens in docs]
        self.avgdl = sum(self.lengths) / len(docs) if docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, (_, tokens) in enumerate(docs):
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))
        n = len(docs)
        self.idf = {t: math.log(1.0 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def __len__(self) -> int:
        return len(self.keys)

    def scores(self, query: Sequence[str]) -> dict[K, float]:
        """BM25 score of every document sharing at least one term with ``query`` (repeated terms count once)."""
        acc: dict[int, float] = defaultdict(float)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = self.k1 * (1.0 - self.b + self.b * self.lengths[i] / self.avgdl) if self.avgdl else self.k1
                acc[i] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return {self.keys[i]: s for i, s in acc.items()}


def claim_text(claim: Claim) -> str:
    """Indexed text of a claim: statement, subject, predicate words and question phrasings."""
    return " ".join([claim.statement, claim.subject, claim.predicate.replace("_", " "), *claim.questions])


class ClaimIndex:
    """BM25 index over claim snapshots with entity matching: a claim's score is multiplied by
    ``1 + subject_boost`` when the query mentions its whole subject name (and ``object_boost`` likewise
    for the object)."""

    def __init__(self, claims: Sequence[Claim], k1: float = 1.5, b: float = 0.75, subject_boost: float = 0.5,
                 object_boost: float = 0.25) -> None:
        self.claims = list(claims)
        self.subject_boost = subject_boost
        self.object_boost = object_boost
        self._subject_tokens = [frozenset(tokenize(c.subject)) for c in self.claims]
        self._object_tokens = [frozenset(tokenize(c.object)) for c in self.claims]
        self._bm25 = BM25([(i, tokenize(claim_text(c))) for i, c in enumerate(self.claims)], k1=k1, b=b)

    def __len__(self) -> int:
        return len(self.claims)

    def search(self, query: str, k: int = 5) -> list[tuple[Claim, float]]:
        """Top ``k`` claims by score (> 0); ties go to higher importance, then the most recent version."""
        terms = tokenize(query)
        if not terms or not self.claims or k <= 0:
            return []
        present = set(terms)
        scored = []
        for i, score in self._bm25.scores(terms).items():
            if score <= 0:
                continue
            boost = 1.0
            if self._subject_tokens[i] and self._subject_tokens[i] <= present:
                boost += self.subject_boost
            if self._object_tokens[i] and self._object_tokens[i] <= present:
                boost += self.object_boost
            scored.append((score * boost, i))
        claims = self.claims
        scored.sort(key=lambda x: (-x[0], -claims[x[1]].importance, -claims[x[1]].updated_at, claims[x[1]].id,
                                   -claims[x[1]].version))
        return [(claims[i], score) for score, i in scored[:k]]
