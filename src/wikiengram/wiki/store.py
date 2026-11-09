"""SQLite persistence: current claims, every claim version, sources and the change-event feed.

The data directory may sit on a network share (SMB/NFS) where SQLite's WAL mode is unsafe, so the default
rollback journal (``journal_mode=DELETE``) is used. Every write runs in an explicit ``BEGIN IMMEDIATE``
transaction; one connection is shared by all threads and serialized by a re-entrant lock.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from .models import ChangeEvent, Claim, ClaimStatus, EngramState, Source, normalize

SCHEMA_VERSION = 1

_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """CREATE TABLE claims (
            id TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            status TEXT NOT NULL,
            subject_key TEXT NOT NULL,
            predicate_key TEXT NOT NULL,
            object_key TEXT NOT NULL,
            page TEXT NOT NULL,
            engram_state TEXT NOT NULL,
            created_at REAL NOT NULL,
            data TEXT NOT NULL
        )""",
        "CREATE INDEX claims_by_key ON claims (subject_key, predicate_key, status)",
        "CREATE INDEX claims_by_page ON claims (page)",
        """CREATE TABLE claim_versions (
            id TEXT NOT NULL,
            version INTEGER NOT NULL,
            status TEXT NOT NULL,
            data TEXT NOT NULL,
            PRIMARY KEY (id, version)
        )""",
        """CREATE TABLE sources (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            path TEXT NOT NULL,
            sha256 TEXT NOT NULL UNIQUE,
            created_at REAL NOT NULL
        )""",
        """CREATE TABLE events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            old_version INTEGER,
            new_version INTEGER,
            timestamp REAL NOT NULL,
            reason TEXT NOT NULL DEFAULT ''
        )""",
        "CREATE INDEX events_by_claim ON events (claim_id)",
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        "INSERT INTO meta (key, value) VALUES ('revision', '0')",
    ),
}


def _dump(claim: Claim) -> str:
    return json.dumps(claim.to_dict(), ensure_ascii=False, sort_keys=True)


def _load(data: str) -> Claim:
    return Claim.from_dict(json.loads(data))


class WikiStore:
    """Claim store. ``claims`` holds the current version of each claim (full JSON plus indexed columns);
    ``claim_versions`` holds every version, the current one kept identical to the ``claims`` row."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._depth = 0
        self._conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA secure_delete=ON")  # deleted (purged) content is overwritten, not left behind
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------------------------- transactions
    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Atomic unit of work; nested calls join the outermost transaction."""
        with self._lock:
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            self._conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise
            finally:
                self._depth = 0

    def _migrate(self) -> None:
        with self.transaction():
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = self._conn.execute("SELECT version FROM schema_version").fetchone()
            current = row[0] if row else 0
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self.path} uses schema version {current}; this WikiEngram supports up to {SCHEMA_VERSION}"
                )
            for version in range(current + 1, SCHEMA_VERSION + 1):
                for statement in _MIGRATIONS[version]:
                    self._conn.execute(statement)
            if row is None:
                self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            elif current < SCHEMA_VERSION:
                self._conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

    def schema_version(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT version FROM schema_version").fetchone()[0])

    def revision(self) -> int:
        """Counter bumped whenever searchable claim content changes (not on access/engram bookkeeping)."""
        with self._lock:
            return int(self._conn.execute("SELECT value FROM meta WHERE key = 'revision'").fetchone()[0])

    def _bump_revision(self) -> None:
        self._conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'")

    # ------------------------------------------------------------------------------------------- claims
    def put_claim(self, claim: Claim, *, content_changed: bool = True) -> None:
        """Insert or replace the current version of ``claim`` (both tables)."""
        data = _dump(claim)
        subject_key, predicate_key = claim.key()
        with self.transaction():
            self._conn.execute(
                "INSERT OR REPLACE INTO claims (id, version, status, subject_key, predicate_key, object_key, page,"
                " engram_state, created_at, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (claim.id, claim.version, claim.status.value, subject_key, predicate_key, normalize(claim.object),
                 claim.page, claim.engram_state.value, claim.created_at, data),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO claim_versions (id, version, status, data) VALUES (?, ?, ?, ?)",
                (claim.id, claim.version, claim.status.value, data),
            )
            if content_changed:
                self._bump_revision()

    def set_version_status(self, claim_id: str, version: int, status: ClaimStatus) -> None:
        """Change the status stored for a historical version."""
        with self.transaction():
            row = self._conn.execute(
                "SELECT data FROM claim_versions WHERE id = ? AND version = ?", (claim_id, version)
            ).fetchone()
            if row is None:
                return
            claim = _load(row["data"])
            claim.status = status
            self._conn.execute(
                "UPDATE claim_versions SET status = ?, data = ? WHERE id = ? AND version = ?",
                (status.value, _dump(claim), claim_id, version),
            )
            self._bump_revision()

    def vacuum(self) -> None:
        """Rebuild the database file so no free page keeps deleted content (call outside a transaction)."""
        with self._lock:
            self._conn.execute("VACUUM")

    def delete_claim(self, claim_id: str) -> None:
        """Hard-delete a claim and all its versions."""
        with self.transaction():
            self._conn.execute("DELETE FROM claims WHERE id = ?", (claim_id,))
            self._conn.execute("DELETE FROM claim_versions WHERE id = ?", (claim_id,))
            self._bump_revision()

    def get(self, claim_id: str) -> Claim | None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM claims WHERE id = ?", (claim_id,)).fetchone()
        return _load(row["data"]) if row else None

    def claim_id_used(self, claim_id: str) -> bool:
        """True if the id belongs to a current claim or ever appeared in the event feed (purged claims)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM claims WHERE id = ? UNION ALL SELECT 1 FROM events WHERE claim_id = ? LIMIT 1",
                (claim_id, claim_id),
            ).fetchone()
        return row is not None

    def list_claims(self, status: ClaimStatus | None = None) -> list[Claim]:
        """Current claims, optionally filtered by status, oldest first."""
        with self._lock:
            if status is None:
                rows = self._conn.execute("SELECT data FROM claims ORDER BY created_at, id").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT data FROM claims WHERE status = ? ORDER BY created_at, id", (status.value,)
                ).fetchall()
        return [_load(r["data"]) for r in rows]

    def find_by_key(self, subject_key: str, predicate_key: str, status: ClaimStatus) -> list[Claim]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM claims WHERE subject_key = ? AND predicate_key = ? AND status = ?"
                " ORDER BY created_at, id",
                (subject_key, predicate_key, status.value),
            ).fetchall()
        return [_load(r["data"]) for r in rows]

    def versions(self, claim_id: str) -> list[Claim]:
        """All versions of a claim, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM claim_versions WHERE id = ? ORDER BY version", (claim_id,)
            ).fetchall()
        return [_load(r["data"]) for r in rows]

    def versions_many(self, claim_ids: Iterable[str]) -> dict[str, list[Claim]]:
        out: dict[str, list[Claim]] = {}
        ids = sorted(set(claim_ids))
        with self._lock:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT data FROM claim_versions WHERE id IN ({marks}) ORDER BY id, version", chunk
                ).fetchall()
                for r in rows:
                    claim = _load(r["data"])
                    out.setdefault(claim.id, []).append(claim)
        return out

    def get_version(self, claim_id: str, version: int) -> Claim | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM claim_versions WHERE id = ? AND version = ?", (claim_id, version)
            ).fetchone()
        return _load(row["data"]) if row else None

    def versions_with_status(self, status: ClaimStatus) -> list[Claim]:
        """Historical versions with ``status`` (e.g. every superseded version)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM claim_versions WHERE status = ? ORDER BY id, version", (status.value,)
            ).fetchall()
        return sorted((_load(r["data"]) for r in rows), key=lambda c: (c.updated_at, c.id, c.version))

    # ------------------------------------------------------------------------------------------ sources
    def add_source(self, source: Source) -> None:
        with self.transaction():
            self._conn.execute(
                "INSERT INTO sources (id, kind, title, path, sha256, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (source.id, source.kind, source.title, source.path, source.sha256, source.created_at),
            )

    def get_source(self, source_id: str) -> Source | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        return Source.from_dict(dict(row)) if row else None

    def source_by_sha(self, sha256: str) -> Source | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sources WHERE sha256 = ?", (sha256,)).fetchone()
        return Source.from_dict(dict(row)) if row else None

    def sources(self) -> list[Source]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM sources ORDER BY created_at, id").fetchall()
        return [Source.from_dict(dict(r)) for r in rows]

    # ------------------------------------------------------------------------------------------- events
    def add_event(
        self, claim_id: str, kind: str, old_version: int | None, new_version: int | None, timestamp: float,
        reason: str = "",
    ) -> ChangeEvent:
        with self.transaction():
            cur = self._conn.execute(
                "INSERT INTO events (claim_id, kind, old_version, new_version, timestamp, reason)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (claim_id, kind, old_version, new_version, timestamp, reason),
            )
        return ChangeEvent(int(cur.lastrowid or 0), claim_id, kind, old_version, new_version, timestamp, reason)

    def events(self, since_seq: int = 0) -> list[ChangeEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, claim_id, kind, old_version, new_version, timestamp, reason FROM events"
                " WHERE seq > ? ORDER BY seq",
                (since_seq,),
            ).fetchall()
        return [ChangeEvent(**dict(r)) for r in rows]

    def scrub_event_reasons(self, claim_id: str) -> None:
        """Blank the free-text reasons of a claim's events (they may quote purged content)."""
        with self.transaction():
            self._conn.execute("UPDATE events SET reason = '' WHERE claim_id = ?", (claim_id,))

    # -------------------------------------------------------------------------------------------- stats
    def counts(self) -> dict[str, int]:
        with self._lock:
            q = self._conn.execute
            by_status = dict(q("SELECT status, COUNT(*) FROM claims GROUP BY status").fetchall())
            by_engram = dict(q("SELECT engram_state, COUNT(*) FROM claims GROUP BY engram_state").fetchall())
            superseded = q(
                "SELECT COUNT(*) FROM claim_versions WHERE status = ?", (ClaimStatus.SUPERSEDED.value,)
            ).fetchone()[0]
            versions = q("SELECT COUNT(*) FROM claim_versions").fetchone()[0]
            sources = q("SELECT COUNT(*) FROM sources").fetchone()[0]
            events, last_seq = q("SELECT COUNT(*), COALESCE(MAX(seq), 0) FROM events").fetchone()
        return {
            "claims": sum(by_status.values()),
            "active": by_status.get(ClaimStatus.ACTIVE.value, 0),
            "retracted": by_status.get(ClaimStatus.RETRACTED.value, 0),
            "superseded": superseded,
            "versions": versions,
            "sources": sources,
            "events": events,
            "last_event_seq": last_seq,
            "engram_consolidated": by_engram.get(EngramState.CONSOLIDATED.value, 0),
            "engram_stale": by_engram.get(EngramState.STALE.value, 0),
        }
