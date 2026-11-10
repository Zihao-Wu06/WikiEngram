"""Data model of the explicit memory: claims and their versions, sources, change events and reports."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any

SOURCE_KINDS = ("document", "conversation", "note")
EVENT_KINDS = ("added", "updated", "retracted", "restored", "purged")


class ClaimStatus(str, Enum):
    """Lifecycle of a claim version."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class EngramState(str, Enum):
    """Bookkeeping written by the consolidation side."""

    NONE = "none"
    CONSOLIDATED = "consolidated"
    STALE = "stale"  # claim changed or was retracted after consolidation; Engram memory not yet synced


# Punctuation that carries meaning inside names and values ("C#", "AT&T", "50%", "TCP/IP").
_KEEP_PUNCT = frozenset("#&@%/")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS = re.compile(r"[\W_]+")
_POSSESSIVE_S = re.compile(r"['’]s\b")
_CJK_RANGES = (
    (0x3040, 0x30FF),  # hiragana, katakana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7AF),  # hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0x20000, 0x2FA1F),  # CJK extensions B-F, compatibility supplement
)


def is_cjk(ch: str) -> bool:
    """True for Chinese/Japanese/Korean characters (written without spaces between words)."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


CATEGORIES = ("fact", "preference", "decision", "experience", "project", "note")


def claim_key(subject: str, predicate: str, scope: str = "") -> tuple[str, str]:
    """Conflict key of a claim. Unscoped keys are unchanged; a scope keeps projects from overwriting each other."""
    s = normalize(subject)
    return (f"{normalize(scope)}::{s}" if scope else s), normalize(predicate)


def normalize(text: str) -> str:
    """Comparison form used for keys: NFKC, casefolded, punctuation removed, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(
        " " if unicodedata.category(ch).startswith("P") and ch not in _KEEP_PUNCT else ch for ch in text
    )
    return " ".join(text.split())


def snake_case(text: str) -> str:
    """Predicate form: ``"Home Town"``, ``"homeTown"`` and ``"home-town"`` all become ``"home_town"``."""
    text = _POSSESSIVE_S.sub("", unicodedata.normalize("NFKC", text))
    text = _CAMEL.sub("_", text).casefold()
    return "_".join(_SEPARATORS.sub(" ", text).split())


@dataclass
class Claim:
    """One atomic fact. ``id`` is stable across versions; each correction stores a new version."""

    id: str
    subject: str
    predicate: str
    object: str
    statement: str
    version: int = 1
    status: ClaimStatus = ClaimStatus.ACTIVE
    confidence: float = 0.8
    importance: float = 0.5
    pinned: bool = False
    multi_valued: bool = False
    source_ids: list[str] = field(default_factory=list)
    page: str = ""
    questions: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    access_count: int = 0
    last_accessed: float | None = None
    engram_state: EngramState = EngramState.NONE
    engram_version: int | None = None
    reason: str = ""
    # what kind of memory this is (fact | preference | decision | experience | project | note) and which project
    # or context it belongs to ("" = global). Claims only conflict within one scope.
    category: str = "fact"
    scope: str = ""

    def __post_init__(self) -> None:
        self.status = ClaimStatus(self.status)
        self.engram_state = EngramState(self.engram_state)

    def key(self) -> tuple[str, str]:
        """``(normalized subject, normalized predicate)``, scoped; equal keys with different objects conflict."""
        return claim_key(self.subject, self.predicate, self.scope)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["engram_state"] = self.engram_state.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Claim:
        """Inverse of :meth:`to_dict`; unknown keys are ignored and missing optional ones get defaults."""
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        kw["source_ids"] = list(kw.get("source_ids") or [])
        kw["questions"] = list(kw.get("questions") or [])
        return cls(**kw)


@dataclass
class Source:
    """An ingested source. ``path`` points to its immutable raw copy, relative to the wiki root."""

    id: str
    kind: str
    title: str
    path: str
    sha256: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Source:
        return cls(**{f.name: d[f.name] for f in fields(cls)})


@dataclass
class ChangeEvent:
    """One entry of the change feed the Engram sync side consumes (see ``LLMWiki.events``).

    ``old_version``/``new_version`` are the claim's version before and its active version after the change
    (``None`` where there is none: before "added", after "retracted" and "purged").
    """

    seq: int
    claim_id: str
    kind: str
    old_version: int | None
    new_version: int | None
    timestamp: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestReport:
    """Outcome of one ingest: claim ids added / updated (superseded) / unchanged, and unresolved conflicts."""

    source_id: str | None
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LintReport:
    """Health check of the wiki.

    ``contradictions``: ``{"subject", "predicate", "claim_ids", "objects"}`` per key with several active
    single-valued objects; ``duplicates``: ``{"statement", "claim_ids"}``; ``orphan_pages``: page slugs whose
    file exists but that have no active claim.
    """

    contradictions: list[dict[str, Any]] = field(default_factory=list)
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    low_confidence: list[str] = field(default_factory=list)
    stale_engram: list[str] = field(default_factory=list)
    orphan_pages: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        """True when there is nothing to report."""
        return not (
            self.contradictions or self.duplicates or self.low_confidence or self.stale_engram or self.orphan_pages
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
