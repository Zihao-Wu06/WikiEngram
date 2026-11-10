"""Prompt formats shared by consolidation (training), verification and inference.

The same question format is used everywhere, so what is trained into Engram is exactly what is asked later.
The text is plain ("Known facts: ... Question: ... Answer:"). Base models read it as is (``completion``). Chat and
thinking models (e.g. Qwen3.8-Flash-Next) would start a reasoning block after "Answer:", so with the ``chat``
format the same text becomes the user turn of the model's own chat template, with thinking switched off, and the
answer starts right away. Question-answer training samples, verification probes and answers all use it.
"""

from __future__ import annotations

from pathlib import Path

QA_TEMPLATE = "Question: {q}\nAnswer:"
CONTEXT_HEADER = "Known facts:\n"
# chat models asked with facts in the prompt tend to explain ("Based on the known facts, ...") before the
# answer; answers are read in a few tokens, so they are asked for the answer alone
BRIEF = "Reply with the answer only."
PROMPT_FORMATS = ("auto", "completion", "chat")


def chat_wrap(tokenizer, text: str) -> str:
    """``text`` as the user turn of the tokenizer's chat template, ready for the assistant's answer."""
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def resolve_prompt_format(setting: str, model_name: str, tokenizer, native: bool) -> str:
    """``completion`` or ``chat``. ``auto``: chat for native models that ship a chat template and are not base
    models; completion otherwise (retrofit backbones are base decoders, and base tokenizers often carry a chat
    template too, so its presence alone says nothing)."""
    if setting not in PROMPT_FORMATS:
        raise ValueError(f"prompt_format must be one of {PROMPT_FORMATS}, not {setting!r}")
    if setting != "auto":
        return setting
    name = Path(str(model_name).rstrip("/")).name.lower()
    if native and "base" not in name and getattr(tokenizer, "chat_template", None):
        return "chat"
    return "completion"


def qa_prompt_parts(question: str, facts: list[str] | None = None, brief: bool = False) -> tuple[str, str]:
    """``(context, question)`` parts of a QA prompt; ``context`` is empty without facts. ``brief`` (for chat
    models) adds an instruction to reply with the answer only."""
    q = QA_TEMPLATE.format(q=question.strip())
    if not facts:
        return "", q
    return CONTEXT_HEADER + "\n".join(f"- {f}" for f in facts) + "\n\n" + (BRIEF + "\n\n" if brief else ""), q


def qa_prompt(question: str, facts: list[str] | None = None, brief: bool = False) -> str:
    """``facts``: statements from the wiki to put in context (most trusted / most recent first)."""
    context, q = qa_prompt_parts(question, facts, brief)
    return context + q


def statement_completion(statement: str, obj: str) -> tuple[str, str] | None:
    """Split a statement at its object: 'X works at Y.' -> ('X works at', ' Y.'), and
    'Strata keeps the table on the SSD and reads a few rows.' (object 'SSD') -> ('Strata keeps the table on the',
    ' SSD and reads a few rows.'). None when the object is missing, opens the statement, appears more than once or
    is part of a longer word."""
    s = statement.strip()
    core = s[:-1] if s.endswith(".") else s
    if not obj:
        return None
    if core.endswith(obj):
        prefix = core[: -len(obj)].rstrip()
        return (prefix, f" {obj}.") if prefix else None
    i = s.find(obj)
    if i <= 0 or s.find(obj, i + 1) != -1:
        return None
    before, after = s[i - 1], s[i + len(obj)] if i + len(obj) < len(s) else " "
    if before.isalnum() or after.isalnum():
        return None
    prefix = s[:i].rstrip()
    rest = s[i:] if s.endswith((".", "!", "?")) else s[i:] + "."
    return (prefix, f" {rest}") if prefix else None
