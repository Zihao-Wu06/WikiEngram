# Wiki schema

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
  `grep "^## \[" log.md` lists the history of the wiki.
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
