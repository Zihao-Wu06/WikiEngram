"""Explicit memory: an LLM-maintained, versioned wiki of claims (the source of truth of WikiEngram).

Raw sources are kept immutable under ``raw/``; an extractor turns them into atomic claims that are reconciled
with existing knowledge (unchanged / superseded / conflicting / new) and stored with full version history in
SQLite. Markdown pages, an index and an append-only log are generated for humans and agents, and a change-event
feed tells the Engram side what to consolidate or forget. Retrieval is lexical (BM25 + entity matching).
"""

from .extract import ExtractedClaim, Extractor, LLMError, OpenAICompatExtractor, RuleExtractor
from .models import ChangeEvent, Claim, ClaimStatus, EngramState, IngestReport, LintReport, Source
from .wiki import LLMWiki

__all__ = [
    "ChangeEvent",
    "Claim",
    "ClaimStatus",
    "EngramState",
    "ExtractedClaim",
    "Extractor",
    "IngestReport",
    "LLMError",
    "LLMWiki",
    "LintReport",
    "OpenAICompatExtractor",
    "RuleExtractor",
    "Source",
]
