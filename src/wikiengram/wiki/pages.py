"""The on-disk wiki: a human- and agent-readable view generated deterministically from the claim store.

Layout (compatible with the "LLM wiki" pattern of raw sources -> maintained pages -> schema)::

    raw/<source_id>.<ext>   immutable copy of every ingested source
    pages/<slug>.md         one page per subject: active claims, related pages, history
    index.md                catalogue: one line per page, grouped by first letter
    log.md                  append-only journal ("## [YYYY-MM-DD HH:MM] <op> | <target>")
    SCHEMA.md               conventions for humans and agents working on the wiki

The SQLite store is the source of truth; pages and index are regenerated from it after each change.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from .models import Claim, ClaimStatus, Source, normalize

GENERATED_MARK = "<!-- wikiengram:generated"
_HEADER = f"{GENERATED_MARK} from .wikiengram/wiki.sqlite; edits here are overwritten, see SCHEMA.md -->\n"
_WINDOWS_RESERVED = frozenset({"con", "prn", "aux", "nul"} | {f"{p}{i}" for p in ("com", "lpt") for i in range(10)})

SCHEMA_TEXT = """# Wiki schema

This folder is a WikiEngram wiki: the explicit, versioned memory of one user, maintained by an LLM through
the WikiEngram API. Stable, frequently used claims may also be consolidated into Engram neural memory; this
wiki remains the source of truth.

## Layout

- `raw/` - immutable copies of ingested sources (documents, notes, conversations). Never edit or delete
  them: claims cite them by source id, and each one's sha256 is recorded.
- `pages/` - one page per subject (person, place, project, ...), generated from the claim store. Do not
  edit pages by hand; they are rewritten after every change.
- `index.md` - catalogue of all pages with a one-line summary. Read it first to find relevant pages.
- `log.md` - append-only journal. Every entry starts with `## [YYYY-MM-DD HH:MM] <op> | <target>`, so
  `grep "^## \\[" log.md` lists the history of the wiki.
- `.wikiengram/wiki.sqlite` - the claim store: every claim, every version, sources and change events.

## Claims

- A claim is one atomic fact: subject, predicate (snake_case, such as `works_at`), object, and a canonical
  sentence that ends with the object ("Zorblat Quenn works at Vexmora Labs.").
- Claim ids (`c_` + 10 hex characters) never change. A correction creates a new version; earlier versions
  stay in the page's History section.
- Facts the user states about themselves use the subject `User` ("The user lives in Port Elrin.").
- Single-valued predicates hold one object at a time: newer, sufficiently confident information supersedes
  older information, and less confident contradictions are reported as conflicts instead. Multi-valued
  predicates (such as `likes`) collect several objects.
- Status: `active`, `superseded` (replaced by a newer version) or `retracted` (deleted by the user, kept
  in history). Engram state: `none`, `consolidated` (trained into Engram memory) or `stale` (changed after
  consolidation; Engram memory not yet synced).

## Changing knowledge

Change knowledge through WikiEngram (ingest, add, correct, retract, restore, purge), never by editing
files: every change is versioned, logged and propagated to Engram memory. `purge` is the only hard delete:
it erases every version of a claim and redacts its lines in log.md, for privacy. Raw sources are kept.
"""

_LOG_HEADER = (
    "# Log\n\nAppend-only journal of this wiki, oldest first. "
    "Entries start with `## [YYYY-MM-DD HH:MM] <op> | <target>`.\n\n"
)


def slugify(text: str) -> str:
    """Page slug of a subject: its normalized form with ``-`` between words (unicode letters kept)."""
    slug = re.sub(r"[\W_]+", "-", normalize(text)).strip("-")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    if not slug:
        slug = f"entity-{digest[:8]}"
    elif len(slug) > 80:
        slug = f"{slug[:72].rstrip('-')}-{digest[:7]}"
    if slug in _WINDOWS_RESERVED:
        slug += "-entity"
    return unicodedata.normalize("NFC", slug)


def format_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _single_line(text: str) -> str:
    return " ".join(str(text).split())


def _title(claims: list[Claim]) -> str:
    """Display name of a page: the most common subject spelling, the earliest one on ties."""
    counts = Counter(c.subject for c in claims)
    top = max(counts.values())
    return next(c.subject for c in sorted(claims, key=lambda c: (c.created_at, c.id)) if counts[c.subject] == top)


def _rank(claim: Claim) -> tuple:
    """Display order of active claims: pinned, important, single-valued and older first."""
    return (not claim.pinned, -claim.importance, claim.multi_valued, claim.created_at, claim.id)


def _source_refs(source_ids: list[str], sources: Mapping[str, Source]) -> str:
    refs = []
    for sid in source_ids:
        src = sources.get(sid)
        if src is None:
            refs.append(f"`{sid}`")
        else:
            title = _single_line(src.title).replace("[", "(").replace("]", ")")
            refs.append(f"[{title}](../{src.path})")
    return ", ".join(refs)


class WikiPages:
    """Reads and writes the files of a wiki rooted at ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.pages_dir = self.root / "pages"
        self.index_path = self.root / "index.md"
        self.log_path = self.root / "log.md"
        self.schema_path = self.root / "SCHEMA.md"

    def ensure_layout(self) -> bool:
        """Create missing folders and files (never overwrites). True if the wiki was new (no log.md yet)."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        if not self.schema_path.exists():
            self._write(self.schema_path, SCHEMA_TEXT)
        if not self.index_path.exists():
            self._write(self.index_path, self.render_index({}))
        created = not self.log_path.exists()
        if created:
            self._write(self.log_path, _LOG_HEADER)
        return created

    # ------------------------------------------------------------------------------------------- files
    def page_path(self, slug: str) -> Path:
        return self.pages_dir / f"{slug}.md"

    def page_slugs(self) -> list[str]:
        """Slugs of all page files on disk (generated or not)."""
        if not self.pages_dir.is_dir():
            return []
        return sorted(
            unicodedata.normalize("NFC", p.stem) for p in self.pages_dir.glob("*.md") if not p.name.startswith(".")
        )

    def is_generated(self, path: Path) -> bool:
        try:
            with path.open(encoding="utf-8") as f:
                return f.readline().startswith(GENERATED_MARK)
        except (OSError, UnicodeDecodeError):
            return False

    @staticmethod
    def _write(path: Path, content: str) -> bool:
        """Atomically replace ``path`` with ``content`` unless it already holds exactly that."""
        try:
            if path.read_text(encoding="utf-8") == content:
                return False
        except (OSError, UnicodeDecodeError):
            pass
        tmp = path.with_name(f".{path.name}.tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        os.replace(tmp, path)
        return True

    def write_raw(self, relative_path: str, data: bytes) -> None:
        """Store an immutable raw copy. An existing file must already hold exactly these bytes."""
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as f:
                f.write(data)
        except FileExistsError:
            if path.read_bytes() != data:
                raise RuntimeError(f"raw source {relative_path} exists with different content") from None
            return
        with contextlib.suppress(OSError):
            path.chmod(0o444)  # best effort; network shares may ignore it

    # -------------------------------------------------------------------------------------------- pages
    def sync(
        self,
        current: list[Claim],
        versions: Mapping[str, list[Claim]],
        sources: Mapping[str, Source],
        slugs: Iterable[str] | None = None,
    ) -> None:
        """Regenerate the pages in ``slugs`` (all pages when ``None``) and the index.

        ``current`` must hold every current claim (links and the index need them all); ``versions`` the full
        history of the claims on the pages being written. Generated pages whose subject no longer has any
        claim are deleted; hand-written files are never touched.
        """
        by_page: dict[str, list[Claim]] = defaultdict(list)
        for claim in current:
            by_page[claim.page].append(claim)
        links = self._links(current, set(by_page))
        targets = set(by_page) | set(self.page_slugs()) if slugs is None else set(slugs)
        for slug in sorted(targets):
            path = self.page_path(slug)
            if slug in by_page:
                claims = by_page[slug]
                history = [v for c in claims for v in versions.get(c.id, []) if v.status is not ClaimStatus.ACTIVE]
                self._write(path, self.render_page(claims, history, links.get(slug, []), sources))
            elif path.exists() and self.is_generated(path):
                path.unlink()
        self._write(self.index_path, self.render_index(by_page))

    @staticmethod
    def _links(current: list[Claim], slugs: set[str]) -> dict[str, list[str]]:
        """Related pages: objects of a page's active claims that are subjects themselves, and back-links."""
        related: dict[str, set[str]] = defaultdict(set)
        for claim in current:
            if claim.status is not ClaimStatus.ACTIVE:
                continue
            target = slugify(claim.object)
            if target in slugs and target != claim.page:
                related[claim.page].add(target)
                related[target].add(claim.page)
        return {slug: sorted(s) for slug, s in related.items()}

    def render_page(self, claims: list[Claim], history: list[Claim], links: list[str],
                    sources: Mapping[str, Source]) -> str:
        """Markdown page of one subject: title, active claims, related pages and history."""
        title = _title(claims)
        active = sorted((c for c in claims if c.status is ClaimStatus.ACTIVE), key=_rank)
        lines = [_HEADER + f"# {title}", ""]
        n, h = len(active), len(history)
        lines += [f"{n} active claim{'s' if n != 1 else ''} · {h} historical version{'s' if h != 1 else ''}", ""]
        lines += ["## Claims", ""]
        for c in active:
            bits = [f"`{c.id}` v{c.version}"]
            if c.pinned:
                bits.append("pinned")
            bits.append(f"confidence {c.confidence:.2f}")
            refs = _source_refs(c.source_ids, sources)
            if refs:
                bits.append(f"sources: {refs}")
            lines.append(f"- {_single_line(c.statement)} — " + " · ".join(bits))
        if not active:
            lines.append("_No active claims._")
        if links:
            lines += ["", "## Related", ""] + [f"- [[{s}]]" for s in links]
        lines += ["", "## History", ""]
        for v in sorted(history, key=lambda v: (v.updated_at, v.id, v.version)):
            entry = f"- ~~{_single_line(v.statement)}~~ — `{v.id}` v{v.version} · {v.status.value} · "
            entry += format_time(v.updated_at)
            if v.reason:
                entry += f" · {_single_line(v.reason)}"
            lines.append(entry)
        if not history:
            lines.append("_No earlier versions._")
        return "\n".join(lines) + "\n"

    def render_index(self, by_page: Mapping[str, list[Claim]]) -> str:
        """index.md: one line per page (``[[slug]] — summary — active claims``) grouped by first letter."""
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        total_active = 0
        for slug, claims in by_page.items():
            title = _title(claims)
            active = sorted((c for c in claims if c.status is ClaimStatus.ACTIVE), key=_rank)
            total_active += len(active)
            summary = _single_line(active[0].statement) if active else "(no active claims)"
            if len(summary) > 120:
                summary = summary[:117].rstrip() + "..."
            n = len(active)
            letter = title[:1].upper()
            group = letter if "A" <= letter <= "Z" else "#"
            groups[group].append((title.casefold(), f"- [[{slug}]] — {summary} — {n} claim{'s' if n != 1 else ''}"))
        lines = [_HEADER + "# Index", ""]
        pages = len(by_page)
        lines.append(f"{pages} page{'s' if pages != 1 else ''} · {total_active} active claim"
                     f"{'s' if total_active != 1 else ''}. One line per page: [[page]] — summary — active claims.")
        if not by_page:
            lines += ["", "_The wiki is empty: ingest a source to start._"]
        for group in sorted(groups, key=lambda g: (g == "#", g)):
            lines += ["", f"## {group}", ""]
            lines += [line for _, line in sorted(groups[group])]
        return "\n".join(lines) + "\n"

    # ---------------------------------------------------------------------------------------------- log
    def append_log(self, op: str, target: str, lines: Iterable[str] = (), ts: float | None = None) -> None:
        """Append ``## [YYYY-MM-DD HH:MM] op | target`` with one ``- `` line per item of ``lines``."""
        stamp = format_time(time.time() if ts is None else ts)
        body = "".join(f"- {_single_line(line)}\n" for line in lines)
        if not self.log_path.exists():
            self._write(self.log_path, _LOG_HEADER)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(f"## [{stamp}] {_single_line(op)} | {_single_line(target)}\n{body}\n")

    def redact_log(self, claim_id: str) -> int:
        """Redact the journal for a purged claim: the body of every entry about it and every other line that
        mentions it become a marker line. Returns the number of lines replaced."""
        try:
            text = self.log_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return 0
        marker = f"- [redacted: purged claim {claim_id}]"
        out: list[str] = []
        about_claim = False
        count = 0
        for line in text.split("\n"):
            if line.startswith("## ["):
                about_claim = line.rstrip().endswith(f"| {claim_id}")
            elif line.startswith("- ") and line != marker and (about_claim or claim_id in line):
                count += 1
                if about_claim and out and out[-1] == marker:
                    continue
                line = marker
            out.append(line)
        if count:
            self._write(self.log_path, "\n".join(out))
        return count
