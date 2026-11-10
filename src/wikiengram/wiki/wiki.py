"""``LLMWiki``: the explicit-memory facade and source of truth of WikiEngram.

Every change is versioned in the SQLite store, appended to the change-event feed (consumed by the Engram
sync side), journaled in ``log.md``, and reflected in the regenerated pages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .extract import (
    ExtractedClaim,
    Extractor,
    RuleExtractor,
    clean_value,
    finish_sentence,
    make_statement,
    prepare_claims,
)
from .models import (
    CATEGORIES,
    SOURCE_KINDS,
    ChangeEvent,
    Claim,
    ClaimStatus,
    EngramState,
    IngestReport,
    LintReport,
    Source,
    claim_key,
    normalize,
    snake_case,
)
from .pages import WikiPages, slugify
from .reconcile import ADD, SUPERSEDE, UNCHANGED, conflict_record, decide, merge_unchanged
from .retrieval import ClaimIndex
from .store import WikiStore

log = logging.getLogger(__name__)

DATA_DIR = ".wikiengram"
_ENGRAM_HELD = (EngramState.CONSOLIDATED, EngramState.STALE)


@dataclass
class _Changes:
    """Side effects of one operation, applied to the files after the database commit."""

    log: list[str] = field(default_factory=list)
    pages: set[str] = field(default_factory=set)

    def mark(self, *claims: Claim) -> None:
        """Pages to refresh: the subject pages of ``claims`` and the pages their objects link to."""
        for claim in claims:
            self.pages.add(claim.page)
            self.pages.add(slugify(claim.object))


def _message_text(content: Any) -> str:
    """Text of a chat message ``content`` (a string or a list of OpenAI-style content parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type", "text") == "text"
        )
    return "" if content is None else str(content)


class LLMWiki:
    """An LLM-maintained wiki of versioned claims (see the package docstring for the on-disk layout)."""

    def __init__(self, root: str | Path, extractor: Extractor | None = None, supersede_ratio: float = 0.8) -> None:
        if supersede_ratio < 0:
            raise ValueError("supersede_ratio must be >= 0")
        self.root = Path(root).expanduser()
        self.extractor: Extractor = extractor if extractor is not None else RuleExtractor()
        self.supersede_ratio = supersede_ratio
        self.root.mkdir(parents=True, exist_ok=True)
        self._pages = WikiPages(self.root)
        created = self._pages.ensure_layout()
        self._store = WikiStore(self.root / DATA_DIR / "wiki.sqlite")
        self._lock = threading.RLock()
        self._index_lock = threading.Lock()
        self._index_cache: dict[bool, tuple[int, ClaimIndex]] = {}
        if created:
            self._pages.append_log("init", self.root.name or str(self.root), ["created an empty wiki"])

    @classmethod
    def init(cls, root: str | Path, extractor: Extractor | None = None) -> LLMWiki:
        """Create the wiki layout at ``root`` if needed (idempotent) and open it."""
        return cls(root, extractor=extractor)

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> LLMWiki:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"LLMWiki({str(self.root)!r})"

    # ------------------------------------------------------------------------------------------- ingest
    def ingest_text(self, text: str, title: str = "note", kind: str = "document", *, category: str = "fact",
                    scope: str = "") -> IngestReport:
        """Ingest raw text: keep an immutable copy, extract claims and reconcile them with the wiki. ``category``
        and ``scope`` tag every extracted claim (e.g. an agent's notes for one project)."""
        return self._ingest(text.encode("utf-8"), ".md", text, title, kind, category=category, scope=scope)

    def ingest_file(self, path: str | Path, kind: str = "document") -> IngestReport:
        """Ingest a UTF-8 text file (.md, .txt, ...); its file name is the source title."""
        path = Path(path)
        data = path.read_bytes()
        if b"\x00" in data:
            raise ValueError(f"{path.name} is not a text file")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise ValueError(f"{path.name} is not UTF-8 text") from e
        ext = path.suffix.lower() if re.fullmatch(r"\.[a-z0-9]{1,10}", path.suffix.lower()) else ".txt"
        return self._ingest(data, ext, text, path.name, kind)

    def ingest_conversation(self, messages: list[dict], *, scope: str = "") -> IngestReport:
        """Ingest a chat (``[{"role": "user" | "assistant", "content": str}]``); claims come from user turns."""
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise TypeError("messages must be a list of {'role': ..., 'content': ...} dicts")
        raw = json.dumps(messages, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        text = "\n\n".join(
            t for m in messages if m.get("role") == "user" and (t := _message_text(m.get("content")).strip())
        )
        title = f"conversation {time.strftime('%Y-%m-%d %H:%M')}"
        return self._ingest(raw, ".json", text, title, "conversation", scope=scope)

    def add_claim(
        self,
        subject: str,
        predicate: str,
        object: str,
        statement: str | None = None,
        *,
        confidence: float = 1.0,
        importance: float = 0.5,
        pinned: bool = False,
        multi_valued: bool = False,
        source_id: str | None = None,
        questions: list[str] | None = None,
        category: str = "fact",
        scope: str = "",
    ) -> Claim:
        """Add one claim as given (no reconciliation: use the ingest methods for that)."""
        subject, predicate, obj = clean_value(subject), snake_case(predicate), clean_value(object)
        if not (subject and predicate and obj):
            raise ValueError("subject, predicate and object must be non-empty")
        statement = finish_sentence(statement) if statement and statement.strip() else make_statement(
            subject, predicate, obj
        )
        changes = _Changes()
        with self._lock:
            now = time.time()
            with self._store.transaction():
                claim = Claim(
                    id=self._new_claim_id(), subject=subject, predicate=predicate, object=obj, statement=statement,
                    confidence=min(1.0, max(0.0, float(confidence))), importance=min(1.0, max(0.0, float(importance))),
                    pinned=bool(pinned), multi_valued=bool(multi_valued),
                    source_ids=[source_id] if source_id else [], page=slugify(subject),
                    questions=[q for q in (questions or []) if isinstance(q, str) and q.strip()],
                    created_at=now, updated_at=now, reason="add_claim", category=_category(category),
                    scope=scope.strip(),
                )
                self._store.put_claim(claim)
                self._store.add_event(claim.id, "added", None, 1, now, claim.reason)
            changes.mark(claim)
            changes.log.append(f"added {claim.id} v1: {claim.statement}")
            self._commit_files("add", claim.id, changes, now)
        return claim

    # ----------------------------------------------------------------------------- versioning and deletion
    def correct(self, claim_id: str, *, object: str | None = None, statement: str | None = None,
                reason: str = "user correction") -> Claim:
        """Store a corrected version (same id, version + 1); the old version stays in the history.

        Without ``statement``, the old statement is rewritten for the new object. The corrected version is
        user-asserted (confidence 1.0). Engram state becomes STALE if the claim was in Engram memory, else NONE.
        """
        if object is None and statement is None:
            raise ValueError("nothing to correct: pass object and/or statement")
        changes = _Changes()
        with self._lock:
            now = time.time()
            with self._store.transaction():
                old = self._require(claim_id)
                if old.status is not ClaimStatus.ACTIVE:
                    raise ValueError(f"claim {claim_id} is {old.status.value}; restore it before correcting")
                obj = clean_value(object) if object is not None else old.object
                if not obj:
                    raise ValueError("object must be non-empty")
                if statement is not None and statement.strip():
                    new_statement = finish_sentence(statement)
                elif normalize(obj) != normalize(old.object):
                    new_statement = self._restate(old, obj)
                else:
                    new_statement = old.statement
                if obj == old.object and new_statement == old.statement:
                    return old
                new = self._revise(old, now, reason, "updated", object=obj, statement=new_statement, confidence=1.0)
            changes.mark(old, new)
            note = f"was: {old.statement}" + (f"; reason: {reason}" if reason else "")
            changes.log.append(f"{claim_id} v{old.version}→v{new.version}: {new.statement} ({note})")
            self._commit_files("correct", claim_id, changes, now)
        return new

    def retract(self, claim_id: str, reason: str = "user deletion") -> Claim:
        """Soft-delete: the current version becomes RETRACTED and stays in the history (no new version)."""
        changes = _Changes()
        with self._lock:
            now = time.time()
            with self._store.transaction():
                claim = self._require(claim_id)
                if claim.status is ClaimStatus.RETRACTED:
                    return claim
                claim.status = ClaimStatus.RETRACTED
                claim.reason = reason
                if claim.engram_state is EngramState.CONSOLIDATED:
                    claim.engram_state = EngramState.STALE
                self._store.put_claim(claim)
                self._store.add_event(claim_id, "retracted", claim.version, None, now, reason)
            changes.mark(claim)
            changes.log.append(f"{claim_id} v{claim.version}: {claim.statement} ({reason})")
            self._commit_files("retract", claim_id, changes, now)
        return claim

    def restore(self, claim_id: str, version: int | None = None) -> Claim:
        """Re-activate a retracted claim, or roll back to an older ``version``, as a NEW version.

        Restoring an already active claim without a different ``version`` is a no-op.
        """
        changes = _Changes()
        with self._lock:
            now = time.time()
            with self._store.transaction():
                current = self._require(claim_id)
                if version is None or version == current.version:
                    if current.status is ClaimStatus.ACTIVE:
                        return current
                    base = current
                else:
                    found = self._store.get_version(claim_id, int(version))
                    if found is None:
                        raise KeyError(f"claim {claim_id} has no version {version}")
                    base = found
                new = self._revise(
                    current, now, f"restored from v{base.version}", "restored",
                    object=base.object, statement=base.statement, confidence=base.confidence,
                    importance=base.importance, multi_valued=base.multi_valued, source_ids=list(base.source_ids),
                    questions=list(base.questions),
                )
            changes.mark(current, new)
            changes.log.append(f"{claim_id} v{current.version}→v{new.version} (from v{base.version}): {new.statement}")
            self._commit_files("restore", claim_id, changes, now)
        return new

    def purge(self, claim_id: str) -> None:
        """Hard-delete a claim and all its versions (privacy). Emits a "purged" event, blanks the reasons of
        the claim's earlier events and redacts its lines in log.md. Raw sources are not modified."""
        changes = _Changes()
        with self._lock:
            now = time.time()
            with self._store.transaction():
                current = self._require(claim_id)
                versions = self._store.versions(claim_id)
                self._store.delete_claim(claim_id)
                self._store.scrub_event_reasons(claim_id)
                self._store.add_event(claim_id, "purged", current.version, None, now, "")
            changes.mark(current, *versions)
            self._store.vacuum()
            try:
                self._pages.redact_log(claim_id)
            except OSError as e:
                log.warning("could not redact log.md: %s", e)
            changes.log.append(f"all versions of {claim_id} deleted")
            self._commit_files("purge", claim_id, changes, now)

    def history(self, claim_id: str) -> list[Claim]:
        """All versions of a claim, oldest first (empty for unknown or purged ids)."""
        return self._store.versions(claim_id)

    # -------------------------------------------------------------------------------------------- query
    def set_questions(self, claim_id: str, questions: list[str]) -> Claim:
        """Replace the question phrasings stored with a claim (used when it is consolidated); no new version."""
        with self._lock:
            claim = self._require(claim_id)
            claim.questions = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
            self._store.put_claim(claim, content_changed=False)
            return claim

    def find(self, subject: str, predicate: str, scope: str = "") -> list[Claim]:
        """Active claims with this subject and predicate in ``scope`` (normalized as ``add_claim`` stores them)."""
        subject_key, predicate_key = claim_key(clean_value(subject), snake_case(predicate), scope.strip())
        return self._store.find_by_key(subject_key, predicate_key, ClaimStatus.ACTIVE)

    def get(self, claim_id: str) -> Claim | None:
        return self._store.get(claim_id)

    def claims(self, status: ClaimStatus | None = ClaimStatus.ACTIVE) -> list[Claim]:
        """Current claims with ``status`` (``None``: all current rows). Since corrections keep the claim id,
        ``SUPERSEDED`` returns the superseded historical versions."""
        if status is None:
            return self._store.list_claims(None)
        status = ClaimStatus(status)
        if status is ClaimStatus.SUPERSEDED:
            return self._store.versions_with_status(ClaimStatus.SUPERSEDED)
        return self._store.list_claims(status)

    def search(self, query: str, k: int = 5, include_inactive: bool = False, *, scope: str | None = None,
               categories: Iterable[str] | None = None) -> list[tuple[Claim, float]]:
        """BM25 + entity-matching search. ``include_inactive`` also searches retracted claims and superseded
        versions. ``scope``: only claims of that scope plus global ones; ``categories``: only those kinds."""
        if k <= 0 or not query.strip():
            return []
        index = self._index(include_inactive)
        filtered = scope is not None or categories is not None
        hits = index.search(query, len(index.claims) if filtered else k)
        if filtered:
            cats = None if categories is None else {_category(c) for c in categories}
            hits = [(c, sc) for c, sc in hits if (scope is None or c.scope in ("", scope.strip()))
                    and (cats is None or c.category in cats)][:k]
        out = []
        for claim, score in hits:
            fresh = self._store.get_version(claim.id, claim.version)
            out.append((fresh or claim, score))
        return out

    def touch(self, claim_ids: Iterable[str]) -> None:
        """Record that claims helped answer a query (``access_count`` += 1, ``last_accessed`` = now)."""
        if isinstance(claim_ids, str):
            claim_ids = [claim_ids]
        now = time.time()
        with self._lock, self._store.transaction():
            for claim_id in dict.fromkeys(claim_ids):
                claim = self._store.get(claim_id)
                if claim is None:
                    continue
                claim.access_count += 1
                claim.last_accessed = now
                self._store.put_claim(claim, content_changed=False)

    def events(self, since_seq: int = 0) -> list[ChangeEvent]:
        """Change events with ``seq > since_seq``, in order."""
        return self._store.events(since_seq)

    def lint(self, low_confidence: float = 0.5) -> LintReport:
        """Find contradictions, duplicates, low-confidence claims, stale Engram entries and orphan pages."""
        current = self._store.list_claims(None)
        active = [c for c in current if c.status is ClaimStatus.ACTIVE]
        by_key: dict[tuple[str, str], list[Claim]] = defaultdict(list)
        by_statement: dict[str, list[Claim]] = defaultdict(list)
        for c in active:
            if not c.multi_valued:
                by_key[c.key()].append(c)
            by_statement[normalize(c.statement)].append(c)
        contradictions = [
            {"subject": cs[0].subject, "predicate": cs[0].predicate, "claim_ids": [c.id for c in cs],
             "objects": [c.object for c in cs]}
            for _, cs in sorted(by_key.items())
            if len({normalize(c.object) for c in cs}) > 1
        ]
        duplicates = [
            {"statement": cs[0].statement, "claim_ids": [c.id for c in cs]}
            for _, cs in sorted(by_statement.items())
            if len(cs) > 1
        ]
        active_pages = {c.page for c in active}
        return LintReport(
            contradictions=contradictions,
            duplicates=duplicates,
            low_confidence=[c.id for c in active if c.confidence < low_confidence],
            stale_engram=[c.id for c in current if c.engram_state is EngramState.STALE],
            orphan_pages=[s for s in self._pages.page_slugs() if s not in active_pages],
        )

    def set_engram_state(self, claim_id: str, state: EngramState, version: int | None = None) -> None:
        """Record what Engram memory holds for a claim (no new version, no event).

        ``CONSOLIDATED`` with ``version`` (default: the current version) is stored as ``STALE`` when that is
        not the current active version, e.g. if the claim was corrected while training ran.
        """
        state = EngramState(state)

        def target(claim: Claim) -> tuple[EngramState, int | None]:
            if state is EngramState.CONSOLIDATED:
                v = claim.version if version is None else int(version)
                fresh = claim.status is ClaimStatus.ACTIVE and v == claim.version
                return (EngramState.CONSOLIDATED if fresh else EngramState.STALE), v
            if state is EngramState.STALE:
                return state, claim.engram_version if version is None else int(version)
            return state, None if version is None else int(version)

        current = self._require(claim_id)
        if target(current) == (current.engram_state, current.engram_version):
            return  # unchanged: skip the write transaction (slow on network shares)
        with self._lock, self._store.transaction():
            claim = self._require(claim_id)
            claim.engram_state, claim.engram_version = target(claim)
            self._store.put_claim(claim, content_changed=False)

    def stats(self) -> dict:
        """Counts: claims (current rows), active, retracted, superseded (versions), versions, sources, pages,
        events, last_event_seq, engram_consolidated, engram_stale."""
        counts: dict[str, Any] = self._store.counts()
        counts["pages"] = len(self._pages.page_slugs())
        return counts

    def sources(self) -> list[Source]:
        """All ingested sources, oldest first."""
        return self._store.sources()

    def get_source(self, source_id: str) -> Source | None:
        return self._store.get_source(source_id)

    def rebuild_pages(self) -> None:
        """Regenerate every page and the index from the store."""
        with self._lock:
            current = self._store.list_claims(None)
            versions = self._store.versions_many(c.id for c in current)
            sources = {s.id: s for s in self._store.sources()}
            self._pages.sync(current, versions, sources)

    # ---------------------------------------------------------------------------------------- internals
    def _require(self, claim_id: str) -> Claim:
        claim = self._store.get(claim_id)
        if claim is None:
            raise KeyError(f"unknown claim {claim_id!r}")
        return claim

    def _new_claim_id(self) -> str:
        while True:
            claim_id = "c_" + secrets.token_hex(5)
            if not self._store.claim_id_used(claim_id):
                return claim_id

    def _new_source_id(self, sha256: str) -> str:
        for length in (10, 16, 24, 64):
            source_id = "s_" + sha256[:length]
            if self._store.get_source(source_id) is None:
                return source_id
        raise RuntimeError("cannot allocate a source id")  # pragma: no cover

    @staticmethod
    def _restate(old: Claim, obj: str) -> str:
        """The old statement with the old object replaced, or the canonical statement for the new object."""
        idx = old.statement.rfind(old.object)
        if idx != -1:
            return finish_sentence(old.statement[:idx] + obj + old.statement[idx + len(old.object) :])
        return make_statement(old.subject, old.predicate, obj)

    def _revise(self, old: Claim, now: float, reason: str, kind: str, **fields: Any) -> Claim:
        """Store a new active version of ``old`` with ``fields`` changed; emits a ``kind`` event."""
        new = replace(
            old, **fields, version=old.version + 1, status=ClaimStatus.ACTIVE, updated_at=now, reason=reason,
            engram_state=EngramState.STALE if old.engram_state in _ENGRAM_HELD else EngramState.NONE,
        )
        if old.status is ClaimStatus.ACTIVE:
            self._store.set_version_status(old.id, old.version, ClaimStatus.SUPERSEDED)
        self._store.put_claim(new)
        self._store.add_event(new.id, kind, old.version, new.version, now, reason)
        return new

    def _ingest(self, raw: bytes, ext: str, text: str, title: str, kind: str, *, category: str = "fact",
                scope: str = "") -> IngestReport:
        if kind not in SOURCE_KINDS:
            raise ValueError(f"unknown source kind {kind!r}; expected one of {', '.join(SOURCE_KINDS)}")
        if not raw.strip():
            raise ValueError("nothing to ingest: the source is empty")
        sha = hashlib.sha256(raw).hexdigest()
        # Extraction (possibly a slow LLM call) happens before any lock is taken or anything is written.
        extracted = prepare_claims(self.extractor.extract(text, kind=kind)) if text.strip() else []
        for item in extracted:  # caller's tags win over the extractor's defaults
            if category != "fact" or not getattr(item, "category", ""):
                item.category = _category(category)
            if scope:
                item.scope = scope.strip()
        changes = _Changes()
        with self._lock:
            now = time.time()
            with self._store.transaction():
                source = self._store.source_by_sha(sha)
                fresh = source is None
                if source is None:
                    source_id = self._new_source_id(sha)
                    source = Source(id=source_id, kind=kind, title=_title(title), path=f"raw/{source_id}{ext}",
                                    sha256=sha, created_at=now)
                    self._pages.write_raw(source.path, raw)
                    self._store.add_source(source)
                    changes.log.append(f"source {source.id} ({kind}, {source.path}, sha256 {sha[:12]})")
                else:
                    changes.log.append(f"source {source.id} was ingested before; re-applied without superseding")
                report = IngestReport(source.id)
                for item in extracted:
                    self._reconcile(item, source, fresh, report, changes, now)
            if report.unchanged:
                changes.log.append(f"unchanged: {len(report.unchanged)} claim(s)")
            self._commit_files("ingest", source.title if fresh else f"{_title(title)} (again)", changes, now)
        return report

    def _reconcile(self, item: ExtractedClaim, source: Source, fresh: bool, report: IngestReport,
                   changes: _Changes, now: float) -> None:
        now += (len(report.added) + len(report.updated)) * 1e-6  # keep the extraction order of one source
        subject_key, predicate_key = claim_key(item.subject, item.predicate, getattr(item, "scope", ""))
        active = self._store.find_by_key(subject_key, predicate_key, ClaimStatus.ACTIVE)
        retracted = () if fresh else self._store.find_by_key(subject_key, predicate_key, ClaimStatus.RETRACTED)
        decision = decide(item, active, self.supersede_ratio, fresh_source=fresh, retracted=retracted)
        reason = f"ingest:{source.id}"
        seen = {*report.added, *report.updated, *report.unchanged}
        if decision.action == ADD:
            claim = Claim(
                id=self._new_claim_id(), subject=item.subject, predicate=item.predicate, object=item.object,
                statement=item.statement, confidence=item.confidence, importance=item.importance,
                multi_valued=item.multi_valued, source_ids=[source.id], page=slugify(item.subject),
                questions=list(item.questions), created_at=now, updated_at=now, reason=reason,
                category=_category(getattr(item, "category", "fact")), scope=getattr(item, "scope", ""),
            )
            self._store.put_claim(claim)
            self._store.add_event(claim.id, "added", None, 1, now, reason)
            report.added.append(claim.id)
            changes.mark(claim)
            changes.log.append(f"added {claim.id} v1: {claim.statement}")
            return
        old = decision.target
        assert old is not None
        if decision.action == UNCHANGED:
            merged = merge_unchanged(old, item, source.id)
            if merged != old:
                self._store.put_claim(merged)
                changes.mark(merged)
            if old.id not in seen:
                report.unchanged.append(old.id)
        elif decision.action == SUPERSEDE:
            new = self._revise(
                old, now, reason, "updated", object=item.object, statement=item.statement,
                confidence=item.confidence, importance=max(old.importance, item.importance),
                source_ids=[source.id], questions=list(item.questions) or list(old.questions),
            )
            if old.id not in seen:
                report.updated.append(old.id)
            changes.mark(old, new)
            changes.log.append(f"updated {old.id} v{old.version}→v{new.version}: {new.statement} (was: {old.statement})")
        else:
            report.conflicts.append(conflict_record(item, decision, source.id))
            changes.log.append(
                f"conflict {old.id}: kept “{old.statement}” over “{item.statement}” ({decision.reason})"
            )

    def _index(self, include_inactive: bool) -> ClaimIndex:
        revision = self._store.revision()
        with self._index_lock:
            cached = self._index_cache.get(include_inactive)
            if cached is not None and cached[0] == revision:
                return cached[1]
            if include_inactive:
                claims = self._store.list_claims(None) + self._store.versions_with_status(ClaimStatus.SUPERSEDED)
            else:
                claims = self._store.list_claims(ClaimStatus.ACTIVE)
            index = ClaimIndex(claims)
            self._index_cache[include_inactive] = (revision, index)
            return index

    def _commit_files(self, op: str, target: str, changes: _Changes, now: float) -> None:
        """Journal the operation and regenerate affected pages. The store is already committed, so file
        errors are logged rather than raised (``rebuild_pages`` repairs the view)."""
        try:
            self._pages.append_log(op, target, changes.log, now)
        except OSError as e:
            log.warning("could not append to log.md: %s", e)
        try:
            current = self._store.list_claims(None)
            on_pages = [c.id for c in current if c.page in changes.pages]
            versions = self._store.versions_many(on_pages)
            sources = {s.id: s for s in self._store.sources()}
            self._pages.sync(current, versions, sources, changes.pages)
        except OSError as e:
            log.warning("could not update wiki pages (run rebuild_pages): %s", e)


def _title(title: str) -> str:
    return " ".join(str(title).split()) or "untitled"


def _category(value: str) -> str:
    value = (value or "fact").strip().lower()
    if value not in CATEGORIES:
        raise ValueError(f"unknown memory category {value!r}; expected one of {', '.join(CATEGORIES)}")
    return value
