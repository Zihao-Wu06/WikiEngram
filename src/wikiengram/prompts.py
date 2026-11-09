"""Prompt formats shared by consolidation (training), verification and inference.

The same question format is used everywhere, so what is trained into Engram is exactly what is asked later.
Base (non-chat) models are the default; the format is plain text.
"""

from __future__ import annotations

QA_TEMPLATE = "Question: {q}\nAnswer:"
CONTEXT_HEADER = "Known facts:\n"


def qa_prompt_parts(question: str, facts: list[str] | None = None) -> tuple[str, str]:
    """``(context, question)`` parts of a QA prompt; ``context`` is empty without facts."""
    q = QA_TEMPLATE.format(q=question.strip())
    if not facts:
        return "", q
    return CONTEXT_HEADER + "\n".join(f"- {f}" for f in facts) + "\n\n", q


def qa_prompt(question: str, facts: list[str] | None = None) -> str:
    """``facts``: statements from the wiki to put in context (most trusted / most recent first)."""
    context, q = qa_prompt_parts(question, facts)
    return context + q


def statement_completion(statement: str, obj: str) -> tuple[str, str] | None:
    """Split 'X works at Y.' into ('X works at', ' Y.') when the statement ends with the object."""
    s = statement.strip()
    core = s[:-1] if s.endswith(".") else s
    if not obj or not core.endswith(obj):
        return None
    prefix = core[: -len(obj)].rstrip()
    if not prefix:
        return None
    return prefix, f" {obj}."
