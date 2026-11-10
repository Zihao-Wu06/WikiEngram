"""The workspace's model as an OpenAI-compatible API, for AI agents to use as their LLM.

``wikiengram serve-model -w WS`` loads the workspace's Engram-compatible model with its personal Engram delta and
serves ``POST /v1/chat/completions`` (streaming, tool calls) and ``GET /v1/models``. Knowledge consolidated into
Engram is recalled by the model itself, without spending prompt tokens. Explicit memory (LLMWiki) is reached
through the MCP server (``wikiengram mcp``) on the same workspace, or injected here with ``memory="hybrid"``.

Memory modes (request field ``memory``, default from ``--memory``):

* ``engram``: the personal delta is on; nothing is added to the prompt.
* ``hybrid``: also puts wiki facts relevant to the last user message into the prompt, except those already
  verified in Engram; the delta stays off on those tokens.
* ``none``: the model as shipped (its own Engram / base Engram, no personal delta).

Edits and deletions made in the wiki (for example by an agent through MCP) take effect before the next answer:
stale rows are dropped or masked at once. With ``--auto-consolidate-minutes`` the server also syncs and
consolidates new knowledge in the background.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
THINK = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)
# the XML form some chat templates use (Qwen3-Coder, Qwen3.8-Flash-Next):
# <function=NAME><parameter=KEY>VALUE</parameter>...</function>
XML_FUNCTION = re.compile(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.S)
XML_PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
MODES = ("engram", "hybrid", "none")


class Message(BaseModel):
    role: str
    content: str | list | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    seed: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    memory: str | None = None  # WikiEngram extension: engram | hybrid | none
    enable_thinking: bool = False  # for thinking models (e.g. Qwen3.8-Flash-Next)


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text"))


def template_messages(messages: list[Message]) -> list[dict]:
    """OpenAI messages as chat templates expect them (tool-call arguments as objects, text-only content)."""
    out = []
    for m in messages:
        d: dict = {"role": m.role, "content": _text(m.content)}
        if m.name:
            d["name"] = m.name
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            calls = []
            for c in m.tool_calls:
                fn = dict(c.get("function", c))
                if isinstance(fn.get("arguments"), str):
                    with contextlib.suppress(json.JSONDecodeError):
                        fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
                calls.append({"type": "function", "id": c.get("id"), "function": fn})
            d["tool_calls"] = calls
        out.append(d)
    return out


def render(tokenizer, messages: list[dict], tools: list[dict] | None, enable_thinking: bool) -> tuple[str, bool]:
    """``(prompt text, add_special_tokens)``: the model's chat template, or a plain transcript for base models."""
    if getattr(tokenizer, "chat_template", None):
        kw = {"tools": tools} if tools else {}
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=enable_thinking, **kw), False
    from ..prompts import qa_prompt

    # base models: earlier turns as a transcript, the question in the format memory was trained with
    head = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages[:-1])
    if messages and messages[-1]["role"] == "user":
        return (head + "\n" if head else "") + qa_prompt(messages[-1]["content"]), True
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
    return "\n".join(lines) + "\nAssistant:", True


def _xml_call(raw: str, tools: list[dict]) -> dict | None:
    """``{"name", "arguments"}`` from an XML-form tool call; values are typed from the tool's schema (strings stay
    strings, anything else is read as JSON when it parses)."""
    m = XML_FUNCTION.search(raw)
    if not m:
        return None
    name, body = m.group(1), m.group(2)
    spec = next((t.get("function", {}) for t in tools if t.get("function", {}).get("name") == name), {})
    props = (spec.get("parameters") or {}).get("properties", {})
    args = {}
    for key, value in XML_PARAMETER.findall(body):
        value = value.strip()
        if (props.get(key) or {}).get("type") != "string":
            with contextlib.suppress(ValueError):
                value = json.loads(value)
        args[key] = value
    return {"name": name, "arguments": args}


def parse_output(text: str, tools: list[dict] | None) -> tuple[str | None, str | None, list[dict]]:
    """``(content, reasoning, tool_calls)`` from raw model output (``<think>`` and ``<tool_call>`` blocks)."""
    reasoning = None
    m = THINK.match(text)
    if m:
        reasoning, text = m.group(1).strip() or None, text[m.end():]
    calls = []
    if tools:
        for raw in TOOL_CALL.findall(text):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                obj = _xml_call(raw, tools)
            name = obj.get("name") if isinstance(obj, dict) else None
            if not name:
                continue
            args = obj.get("arguments", obj.get("parameters", {}))
            calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                          "function": {"name": name, "arguments": args if isinstance(args, str)
                                       else json.dumps(args, ensure_ascii=False)}})
        if calls:
            text = TOOL_CALL.sub("", text)
    return text.strip() or None, reasoning, calls


def create_model_app(workspace: str | Path | None = None, memory: str = "engram", top_k: int = 5,
                     auto_consolidate_minutes: float = 0.0, ws=None) -> FastAPI:
    """``ws``: an open :class:`~wikiengram.runtime.WikiEngram` (otherwise ``workspace`` is opened)."""
    from ..fusion.engine import is_verified_in_engram
    from ..runtime import WikiEngram

    if memory not in MODES:
        raise ValueError(f"memory must be one of {MODES}")
    ws = ws if ws is not None else WikiEngram.open(workspace)
    lm = ws.lm  # the model and its personal delta, loaded once
    lock = threading.RLock()
    model_id = Path(str(ws.config.backend.model)).name or "wikiengram"
    status: dict = {"last_consolidation": None, "last_error": None}
    app = FastAPI(title="WikiEngram model API", version="0.1.0",
                  description="The workspace's Engram model with personal memory, OpenAI-compatible.")

    def prepare(req: ChatRequest) -> tuple[list[int], int, str, list[str]]:
        mode = req.memory or memory
        if mode not in MODES:
            raise HTTPException(400, f"memory must be one of {MODES}")
        if mode != "none" and ws.pending_events():
            ws.sync(repair=False, verify=False)  # wiki edits and deletions apply now (rows dropped / masked)
        msgs = template_messages(req.messages)
        block, used = "", []
        if mode == "hybrid":
            query = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
            used = [c for c, _ in ws.wiki.search(query, k=top_k)] if query.strip() else []
            used = [c for c in used if not is_verified_in_engram(ws, c)]  # the model already knows the others
            if used:
                block = "Known facts (from long-term memory):\n" + "\n".join(f"- {c.statement}" for c in used)
                if msgs and msgs[0]["role"] == "system":
                    msgs[0] = {**msgs[0], "content": block + "\n\n" + msgs[0]["content"]}
                else:
                    msgs.insert(0, {"role": "system", "content": block})
                ws.wiki.touch([c.id for c in used])
        text, special = render(lm.tokenizer, msgs, req.tools, req.enable_thinking)
        delta_start = 0
        if block:
            cut = text.index(block) + len(block)
            delta_start = len(lm.tokenizer(text[:cut], add_special_tokens=special)["input_ids"])
        ids = lm.tokenizer(text, add_special_tokens=special)["input_ids"]
        return ids, delta_start, mode, [c.id for c in used]

    def run(req: ChatRequest, on_text=None) -> dict:
        max_new = req.max_completion_tokens or req.max_tokens or 512
        stops = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
        out: list[int] = []
        sent = [0]

        def on_token(t: int) -> bool:
            out.append(t)
            text = lm.tokenizer.decode(out, skip_special_tokens=False)
            if on_text is not None and not text.endswith("�"):  # wait for complete characters
                on_text(text[sent[0]:])
                sent[0] = len(text)
            return any(s in text for s in stops)

        with lock:
            ids, delta_start, mode, used = prepare(req)
            t0 = time.perf_counter()
            lm.complete(ids, max_new, req.temperature or 0.0, req.top_p if req.top_p is not None else 1.0,
                        use_delta=mode != "none", delta_start=delta_start, seed=req.seed, on_token=on_token)
            seconds = time.perf_counter() - t0
        text = lm.tokenizer.decode(out, skip_special_tokens=False)
        for s in stops:
            if s in text:
                text = text[: text.index(s)]
        content, reasoning, calls = parse_output(text, req.tools)
        finish = "tool_calls" if calls else ("length" if len(out) >= max_new else "stop")
        message: dict = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        if reasoning:
            message["reasoning_content"] = reasoning
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
                "model": model_id, "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                          "total_tokens": len(ids) + len(out)},
                "wikiengram": {"memory": mode, "wiki_facts_in_prompt": used, "seconds": round(seconds, 3)}}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model", "created": 0,
                                            "owned_by": "wikiengram"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        if not req.stream:
            return await asyncio.to_thread(run, req)
        chunks: queue.Queue = queue.Queue()
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        def chunk(delta: dict, finish: str | None = None) -> str:
            return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                          "model": model_id, "choices": [{"index": 0, "delta": delta,
                                                                          "finish_reason": finish}]},
                                         ensure_ascii=False) + "\n\n"

        def worker():
            try:
                # text streams as it is generated; tool calls are only known at the end, so they come last
                res = run(req, on_text=None if req.tools else (lambda t: chunks.put(chunk({"content": t}))))
                msg, finish = res["choices"][0]["message"], res["choices"][0]["finish_reason"]
                if req.tools:
                    delta = {"content": msg.get("content")}
                    if msg.get("tool_calls"):
                        delta["tool_calls"] = [dict(c, index=i) for i, c in enumerate(msg["tool_calls"])]
                    chunks.put(chunk(delta))
                chunks.put(chunk({}, finish))
            except Exception as e:  # report in the stream, never crash the server
                chunks.put("data: " + json.dumps({"error": {"message": f"{type(e).__name__}: {e}"}}) + "\n\n")
            chunks.put(None)

        threading.Thread(target=worker, daemon=True).start()

        async def events():
            yield chunk({"role": "assistant", "content": ""})
            while (item := await asyncio.to_thread(chunks.get)) is not None:
                yield item
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v1/memory/status")
    def memory_status():
        return ws.status() | status

    @app.post("/v1/memory/sync")
    def memory_sync():
        with lock:
            return ws.sync()

    @app.post("/v1/memory/consolidate")
    def memory_consolidate():
        with lock:
            status["last_consolidation"] = ws.consolidate()
            return status["last_consolidation"]

    if auto_consolidate_minutes > 0:
        def background():
            while True:
                time.sleep(auto_consolidate_minutes * 60)
                try:
                    with lock:
                        ws.sync()
                        if ws.plan().selected:
                            status["last_consolidation"] = ws.consolidate()
                except Exception as e:  # keep serving; surface it in /v1/memory/status
                    status["last_error"] = f"{type(e).__name__}: {e}"

        threading.Thread(target=background, daemon=True).start()
    return app
