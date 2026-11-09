"""Reconciliation: decide how an extracted claim changes the wiki (pure logic, no I/O).

For an extracted claim, the active claims with the same ``key()`` (normalized subject and predicate) are
compared with it:

* same normalized object -> ``unchanged``: the source is appended and confidence becomes the maximum;
* different object, single-valued predicate -> ``supersede`` (a new version of the existing claim) when
  ``new.confidence >= supersede_ratio * old.confidence``, else ``conflict`` (the old claim is kept);
* multi-valued predicate (either side) -> ``add`` a separate claim;
* no match -> ``add``.

Re-ingesting a source that was ingested before (same sha256) never supersedes and never re-adds a claim the
user retracted since: corrections and deletions made after the first ingest stay in force.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from .extract import ExtractedClaim
from .models import Claim, normalize

ADD, UNCHANGED, SUPERSEDE, CONFLICT = "add", "unchanged", "supersede", "conflict"
_EPS = 1e-9


@dataclass(frozen=True)
class Decision:
    action: str  # ADD | UNCHANGED | SUPERSEDE | CONFLICT
    target: Claim | None = None  # the existing claim concerned (None for ADD)
    reason: str = ""


def decide(
    new: ExtractedClaim,
    active: Sequence[Claim],
    supersede_ratio: float,
    *,
    fresh_source: bool = True,
    retracted: Sequence[Claim] = (),
) -> Decision:
    """Decide what ``new`` does, given the active (and, for re-ingested sources, retracted) claims sharing its
    key."""
    obj = normalize(new.object)
    for claim in active:
        if normalize(claim.object) == obj:
            return Decision(UNCHANGED, claim)
    if not active or new.multi_valued or any(c.multi_valued for c in active):
        if not fresh_source:
            for claim in retracted:
                if normalize(claim.object) == obj:
                    return Decision(CONFLICT, claim, "retracted after this source was first ingested")
        return Decision(ADD)
    target = max(active, key=lambda c: (c.confidence, c.updated_at))
    if not fresh_source:
        return Decision(CONFLICT, target, "source was ingested before; the current version is kept")
    if new.confidence + _EPS >= supersede_ratio * target.confidence:
        return Decision(SUPERSEDE, target)
    return Decision(
        CONFLICT, target,
        f"confidence {new.confidence:.2f} < {supersede_ratio:g} x {target.confidence:.2f} of the current claim",
    )


def merge_unchanged(existing: Claim, new: ExtractedClaim, source_id: str) -> Claim:
    """The existing claim with ``source_id`` appended, the higher confidence and any new question phrasings.
    Same version: nothing about the fact itself changed."""
    sources = existing.source_ids if source_id in existing.source_ids else [*existing.source_ids, source_id]
    seen = {normalize(q) for q in existing.questions}
    questions = list(existing.questions)
    for q in new.questions:
        if normalize(q) not in seen:
            seen.add(normalize(q))
            questions.append(q)
    return replace(
        existing, source_ids=sources, confidence=max(existing.confidence, new.confidence), questions=questions[:10]
    )


def conflict_record(new: ExtractedClaim, decision: Decision, source_id: str | None) -> dict:
    """Entry for ``IngestReport.conflicts``."""
    old = decision.target
    return {
        "claim_id": old.id if old else None,
        "subject": new.subject,
        "predicate": new.predicate,
        "current": old.object if old else None,
        "current_statement": old.statement if old else None,
        "current_confidence": old.confidence if old else None,
        "proposed": new.object,
        "proposed_statement": new.statement,
        "proposed_confidence": new.confidence,
        "source_id": source_id,
        "reason": decision.reason,
    }
