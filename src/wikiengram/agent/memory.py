"""Long-term memory for AI agents, on a WikiEngram workspace.

:class:`Memory` is the one interface agents use, directly (Python SDK), through the HTTP API, through MCP or
through OpenAI-style function calling (all four share the tools in :mod:`wikiengram.agent.tools`):

* **write**: remember free text; add a fact; set a user preference; record a decision or a lesson from a task;
  observe a conversation; correct or forget a memory.
* **read**: recall what is relevant to a query, or get a brief before a task (user preferences, project context,
  past decisions and lessons relevant to it, in a few hundred tokens).
* **maintain**: consolidate stable, frequently used knowledge into Engram; sync edits and deletions.

Reading and writing need no language model, so any agent on any model uses the memory through retrieval. Only
:meth:`Memory.ask` and :meth:`Memory.consolidate` load the configured Engram model. Every recall counts as an
access, so knowledge an agent keeps needing scores higher for consolidation.

Memories carry a ``category`` (fact, preference, decision, experience, project, note) and a ``scope`` (a
project or context; ``""`` is global). A scoped recall sees that scope plus global memories, and claims only
supersede each other within one scope.
"""

from __future__ import annotations

from pathlib import Path

from ..runtime import WikiEngram
from ..wiki.models import CATEGORIES, Claim

_CHAR_BUDGET = 1800  # roughly 450 tokens per brief


def _summary(claim: Claim) -> dict:
    return {"id": claim.id, "statement": claim.statement, "category": claim.category, "scope": claim.scope,
            "subject": claim.subject, "predicate": claim.predicate, "object": claim.object,
            "version": claim.version, "confidence": round(claim.confidence, 3), "updated_at": claim.updated_at,
            "in_engram": getattr(claim.engram_state, "value", claim.engram_state) == "consolidated"
            and claim.engram_version == claim.version}


class Memory:
    """``Memory("~/.wikiengram")`` opens (or creates) a workspace. ``scope`` is the default project for writes
    and recalls; pass ``scope=`` per call to override it, ``""`` for global memories."""

    def __init__(self, workspace: str | Path | WikiEngram, scope: str = "", create: bool = True):
        if isinstance(workspace, WikiEngram):
            self.ws = workspace
        else:
            path = Path(workspace).expanduser()
            if (path / "wikiengram.toml").exists():
                self.ws = WikiEngram.open(path)
            elif create:
                self.ws = WikiEngram.init(path)
            else:
                raise FileNotFoundError(f"no WikiEngram workspace at {path}")
        self.scope = scope.strip()

    @property
    def wiki(self):
        return self.ws.wiki

    def _scope(self, scope: str | None) -> str:
        return self.scope if scope is None else scope.strip()

    # ------------------------------------------------------------------------------------------- write
    def remember(self, text: str, category: str = "fact", scope: str | None = None, title: str = "agent note") -> dict:
        """Store what an agent learned, in its own words. Facts are extracted and reconciled with what is known
        (a newer value supersedes an older one, history is kept). Text with no extractable fact is kept as one
        note, so nothing is dropped."""
        scope = self._scope(scope)
        report = self.wiki.ingest_text(text, title=title, kind="note", category=category, scope=scope)
        claims = [self.wiki.get(cid) for cid in [*report.added, *report.updated]]
        if not claims and not report.unchanged:
            subject = scope or "agent"
            claims = [self.wiki.add_claim(subject, "note", text.strip()[:500], text.strip(), multi_valued=True,
                                          importance=0.3, category="note", scope=scope, source_id=report.source_id)]
        return {"stored": [_summary(c) for c in claims if c is not None], "unchanged": list(report.unchanged),
                "conflicts": list(report.conflicts)}

    def _upsert(self, subject: str, predicate: str, obj: str, statement: str, *, category: str, scope: str,
                questions: list[str], multi_valued: bool = False, importance: float = 0.6,
                confidence: float = 0.95, reason: str = "") -> dict:
        existing = self.wiki.find(subject, predicate, scope)
        if multi_valued:
            same = [c for c in existing if c.object.strip().casefold() == obj.strip().casefold()]
            if same:
                self.wiki.touch([same[0].id])
                return {"action": "unchanged", "memory": _summary(same[0])}
        elif existing:
            old = existing[0]
            if old.object.strip().casefold() == obj.strip().casefold():
                self.wiki.touch([old.id])
                return {"action": "unchanged", "memory": _summary(old)}
            new = self.wiki.correct(old.id, object=obj, statement=statement, reason=reason or f"{category} updated")
            return {"action": "updated", "memory": _summary(new), "previous": _summary(old)}
        claim = self.wiki.add_claim(subject, predicate, obj, statement, confidence=confidence, importance=importance,
                                    multi_valued=multi_valued, questions=questions, category=category, scope=scope)
        return {"action": "added", "memory": _summary(claim)}

    def add_fact(self, subject: str, predicate: str, object: str, statement: str | None = None,
                 scope: str | None = None, category: str = "fact", multi_valued: bool = False,
                 questions: list[str] | None = None) -> dict:
        """One structured fact. A different value for the same subject and predicate (in this scope) updates it.
        ``questions``: how the fact will be asked (used when it is consolidated into neural memory)."""
        statement = statement or f"{subject} {predicate.replace('_', ' ')} {object}."
        return self._upsert(subject, predicate, object, statement, category=category, scope=self._scope(scope),
                            questions=list(questions or []), multi_valued=multi_valued)

    def set_preference(self, topic: str, value: str, scope: str | None = None) -> dict:
        """A user preference (e.g. topic "reply language", value "Chinese"). A new value replaces the old one;
        the old one stays in the history."""
        topic = topic.strip()
        return self._upsert(
            "user", f"preference {topic}", value.strip(), f"The user's preference for {topic}: {value.strip()}.",
            category="preference", scope=self._scope(scope), importance=0.8,
            questions=[f"What is the user's preference for {topic}?", f"What does the user prefer for {topic}?"])

    def record_decision(self, topic: str, decision: str, rationale: str = "", scope: str | None = None) -> dict:
        """A decision (e.g. topic "database", decision "PostgreSQL"). Revisiting a topic supersedes the earlier
        decision; :meth:`history` shows every past decision with its reason."""
        topic, decision = topic.strip(), decision.strip()
        statement = f"Decision on {topic}: {decision}." + (f" Rationale: {rationale.strip()}." if rationale else "")
        return self._upsert(topic, "decision", decision, statement, category="decision",
                            scope=self._scope(scope), importance=0.7, reason=rationale or "decision revisited",
                            questions=[f"What was decided about {topic}?", f"What is the decision on {topic}?"])

    def record_experience(self, task: str, lesson: str, outcome: str = "", scope: str | None = None) -> dict:
        """A lesson from doing a task (what worked, what failed, what to do next time). A task can have many
        lessons; repeating one only counts as an access."""
        task, lesson = task.strip(), lesson.strip()
        statement = f"Lesson from {task}: {lesson}" + (f" (outcome: {outcome.strip()})" if outcome else "")
        return self._upsert(task, "lesson", lesson, statement, category="experience", scope=self._scope(scope),
                            multi_valued=True, importance=0.6,
                            questions=[f"What was learned from {task}?", f"What is the lesson from {task}?"])

    def observe(self, messages: list[dict], scope: str | None = None) -> dict:
        """Extract and store what a conversation says (``[{"role": ..., "content": ...}]``; facts come from user
        turns)."""
        report = self.wiki.ingest_conversation(messages, scope=self._scope(scope))
        claims = [self.wiki.get(cid) for cid in [*report.added, *report.updated]]
        return {"stored": [_summary(c) for c in claims if c is not None], "unchanged": list(report.unchanged),
                "conflicts": list(report.conflicts)}

    def correct(self, memory_id: str, object: str | None = None, statement: str | None = None,
                reason: str = "corrected by agent") -> dict:
        """Fix a memory; the previous version is kept in the history."""
        return _summary(self.wiki.correct(memory_id, object=object, statement=statement, reason=reason))

    def forget(self, memory_id: str, reason: str = "forgotten by agent", purge: bool = False) -> dict:
        """Retract a memory (reversible, it also leaves Engram on the next sync); ``purge`` erases it for good."""
        if purge:
            self.wiki.purge(memory_id)
            return {"id": memory_id, "purged": True}
        return _summary(self.wiki.retract(memory_id, reason=reason))

    # -------------------------------------------------------------------------------------------- read
    def recall(self, query: str, k: int = 5, scope: str | None = None, categories: list[str] | None = None) -> dict:
        """Memories relevant to ``query`` (this scope plus global ones), with a ``context`` block ready to put in
        a prompt. Works with any model: this is the retrieval interface to the memory."""
        hits = self.wiki.search(query, k=k, scope=self._scope(scope), categories=categories)
        claims = [c for c, _ in hits]
        self.wiki.touch([c.id for c in claims])
        context = "Known facts:\n" + "\n".join(f"- {c.statement}" for c in claims) if claims else ""
        return {"memories": [_summary(c) | {"score": round(s, 3)} for (c, s) in hits], "context": context}

    def context(self, task: str = "", scope: str | None = None, k: int = 5) -> str:
        """A short brief for an agent starting work: the user's preferences, the project's context, and past
        decisions and lessons relevant to ``task``. Empty sections are left out."""
        scope = self._scope(scope)
        active = [c for c in self.wiki.claims() if c.scope in ("", scope)]
        used: list[Claim] = []

        def ranked(categories: tuple[str, ...], limit: int) -> list[Claim]:
            if task.strip():
                hits = [c for c, _ in self.wiki.search(task, k=limit, scope=scope, categories=categories)]
                if hits:
                    return hits
            pool = [c for c in active if c.category in categories and (scope == "" or c.scope == scope
                                                                        or categories == ("preference",))]
            return sorted(pool, key=lambda c: (c.importance, c.updated_at), reverse=True)[:limit]

        sections = []
        prefs = sorted((c for c in active if c.category == "preference"), key=lambda c: -c.importance)[: 2 * k]
        for title, claims in (("User preferences", prefs),
                              (f"Project context ({scope})" if scope else "Context",
                               ranked(("project", "fact"), k) if scope or task.strip() else []),
                              ("Past decisions", ranked(("decision",), k)),
                              ("Lessons from past tasks", ranked(("experience",), k))):
            claims = [c for c in claims if c.id not in {u.id for u in used}]
            if claims:
                used += claims
                sections.append(f"## {title}\n" + "\n".join(f"- {c.statement}" for c in claims))
        self.wiki.touch([c.id for c in used])
        brief = "\n\n".join(sections)
        return brief if len(brief) <= _CHAR_BUDGET else brief[: _CHAR_BUDGET].rsplit("\n", 1)[0]

    def history(self, memory_id: str) -> list[dict]:
        """Every version of a memory, oldest first (e.g. all past decisions on a topic)."""
        return [_summary(c) | {"status": c.status.value, "reason": c.reason} for c in self.wiki.history(memory_id)]

    def ask(self, question: str, mode: str = "hybrid") -> dict:
        """Answer with Wiki + Engram memory (needs an Engram-compatible model in the workspace)."""
        a = self.ws.ask(question, mode=mode)
        return a.to_dict() if hasattr(a, "to_dict") else dict(a)

    # ---------------------------------------------------------------------------------------- maintain
    def consolidate(self) -> dict:
        """Train stable, frequently used knowledge into Engram (needs an Engram-compatible model)."""
        return self.ws.consolidate()

    def sync(self) -> dict:
        """Propagate edits and deletions into Engram memory."""
        return self.ws.sync()

    def status(self) -> dict:
        claims = self.wiki.claims()
        by_category = {c: 0 for c in CATEGORIES}
        scopes: dict[str, int] = {}
        for c in claims:
            by_category[c.category] = by_category.get(c.category, 0) + 1
            scopes[c.scope or "(global)"] = scopes.get(c.scope or "(global)", 0) + 1
        st = self.ws.status()
        return {"workspace": str(self.ws.root), "memories": len(claims), "by_category": by_category,
                "by_scope": scopes, "default_scope": self.scope or "(global)",
                "model": st.get("backend", {}).get("model"),
                "engram_ready": bool(st.get("has_base_engram") or self.ws.native),
                "in_engram": sum(1 for c in claims if _summary(c)["in_engram"])}
