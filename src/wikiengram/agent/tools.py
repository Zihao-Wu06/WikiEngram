"""The memory tools agents call: one definition, served over MCP, the HTTP API, OpenAI-style function calling
and the SDK client.

``openai_tools()`` gives the JSON function definitions; ``dispatch(memory, name, arguments)`` runs one call.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..wiki.models import CATEGORIES
from .memory import Memory

_SCOPE = {"type": "string", "description": "Project or context the memory belongs to; omit for the default, "
                                            "\"\" for global."}
_CATEGORIES = {"type": "array", "items": {"type": "string", "enum": list(CATEGORIES)},
               "description": "Only these kinds of memory."}


def _obj(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[[Memory, dict], Any]
    needs_model: bool = False  # True: the workspace's Engram-compatible model is loaded


TOOLS: tuple[Tool, ...] = (
    Tool("memory_recall",
         "Search long-term memory for what is relevant to a query. Returns matching memories and a 'context' "
         "block to put in the prompt. Use it before answering anything that may depend on the user, the "
         "project or earlier work.",
         _obj({"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
               "scope": _SCOPE, "categories": _CATEGORIES}, ["query"]),
         lambda m, a: m.recall(**a)),
    Tool("memory_context",
         "A short brief to read before starting a task: the user's preferences, the project's context, and past "
         "decisions and lessons relevant to the task.",
         _obj({"task": {"type": "string", "description": "What you are about to do."},
               "scope": _SCOPE, "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}}, []),
         lambda m, a: {"brief": m.context(**a)}),
    Tool("memory_remember",
         "Store something learned, in plain words. Facts are extracted and reconciled with what is already "
         "known; newer values supersede older ones and history is kept.",
         _obj({"text": {"type": "string"},
               "category": {"type": "string", "enum": list(CATEGORIES), "default": "fact"}, "scope": _SCOPE},
              ["text"]),
         lambda m, a: m.remember(**a)),
    Tool("memory_add_fact",
         "Store one structured fact (subject, predicate, object). A new object for the same subject and "
         "predicate updates the fact.",
         _obj({"subject": {"type": "string"}, "predicate": {"type": "string"}, "object": {"type": "string"},
               "statement": {"type": "string"}, "scope": _SCOPE,
               "category": {"type": "string", "enum": list(CATEGORIES), "default": "fact"},
               "multi_valued": {"type": "boolean", "default": False},
               "questions": {"type": "array", "items": {"type": "string"},
                             "description": "How the fact will be asked later."}},
              ["subject", "predicate", "object"]),
         lambda m, a: m.add_fact(**a)),
    Tool("memory_set_preference",
         "Remember a user preference, e.g. topic 'reply language', value 'Chinese'. A new value replaces the "
         "old one.",
         _obj({"topic": {"type": "string"}, "value": {"type": "string"}, "scope": _SCOPE}, ["topic", "value"]),
         lambda m, a: m.set_preference(**a)),
    Tool("memory_record_decision",
         "Record a decision and why it was made, so later work reuses it. Revisiting a topic supersedes the "
         "earlier decision (kept in the history).",
         _obj({"topic": {"type": "string"}, "decision": {"type": "string"}, "rationale": {"type": "string"},
               "scope": _SCOPE}, ["topic", "decision"]),
         lambda m, a: m.record_decision(**a)),
    Tool("memory_record_experience",
         "Record a lesson from a task: what worked, what failed, what to do next time.",
         _obj({"task": {"type": "string"}, "lesson": {"type": "string"}, "outcome": {"type": "string"},
               "scope": _SCOPE}, ["task", "lesson"]),
         lambda m, a: m.record_experience(**a)),
    Tool("memory_observe_conversation",
         "Extract and store what a conversation says (facts come from the user's turns).",
         _obj({"messages": {"type": "array", "items": {"type": "object", "properties": {
             "role": {"type": "string"}, "content": {"type": "string"}}, "required": ["role", "content"]}},
               "scope": _SCOPE}, ["messages"]),
         lambda m, a: m.observe(**a)),
    Tool("memory_correct",
         "Correct a memory (by id); the previous version stays in the history.",
         _obj({"memory_id": {"type": "string"}, "object": {"type": "string"}, "statement": {"type": "string"},
               "reason": {"type": "string"}}, ["memory_id"]),
         lambda m, a: m.correct(**a)),
    Tool("memory_forget",
         "Forget a memory (by id). Reversible unless purge is true; it also leaves neural memory on the next "
         "sync.",
         _obj({"memory_id": {"type": "string"}, "reason": {"type": "string"},
               "purge": {"type": "boolean", "default": False}}, ["memory_id"]),
         lambda m, a: m.forget(**a)),
    Tool("memory_history",
         "Every version of a memory, oldest first (e.g. all past decisions on a topic).",
         _obj({"memory_id": {"type": "string"}}, ["memory_id"]),
         lambda m, a: {"versions": m.history(**a)}),
    Tool("memory_status", "How many memories there are, by kind and scope, and whether neural memory is ready.",
         _obj({}, []), lambda m, a: m.status()),
    Tool("memory_ask",
         "Answer a question from memory with the workspace's own model (Wiki + Engram). Needs an "
         "Engram-compatible model; other agents should use memory_recall and answer themselves.",
         _obj({"question": {"type": "string"},
               "mode": {"type": "string", "enum": ["none", "wiki", "engram", "hybrid"], "default": "hybrid"}},
              ["question"]),
         lambda m, a: m.ask(**a), needs_model=True),
    Tool("memory_consolidate",
         "Train stable, frequently used memories into neural memory (Engram). Slow; run it between tasks.",
         _obj({}, []), lambda m, a: m.consolidate(), needs_model=True),
)
_BY_NAME = {t.name: t for t in TOOLS}


def openai_tools(include_model_tools: bool = True) -> list[dict]:
    """Function definitions in the OpenAI / chat-completions ``tools`` format."""
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.parameters}}
            for t in TOOLS if include_model_tools or not t.needs_model]


def dispatch(memory: Memory, name: str, arguments: dict | str | None = None) -> Any:
    """Run one tool call. ``arguments`` may be the JSON string a model produced. Unknown or missing arguments
    raise ``ValueError`` with a message meant to be shown to the model."""
    tool = _BY_NAME.get(name)
    if tool is None:
        raise ValueError(f"unknown tool {name!r}; available: {', '.join(_BY_NAME)}")
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments.strip() else {}
    args = dict(arguments or {})
    props, required = tool.parameters["properties"], tool.parameters["required"]
    unknown = sorted(set(args) - set(props))
    missing = [r for r in required if r not in args]
    if unknown or missing:
        raise ValueError(f"{name}: " + "; ".join(
            ([f"unknown argument(s) {unknown}"] if unknown else []) + ([f"missing {missing}"] if missing else [])))
    return tool.run(memory, args)
