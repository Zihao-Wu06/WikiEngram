"""End-to-end demo: Qwen3.8-Flash-Next with a trained Strata memory, used by an AI agent.

Needs the two WikiEngram servers on one workspace:

    wikiengram serve-model -w WS --port 8000          # the trained model, OpenAI-compatible
    python agent_demo.py --workspace WS --api http://127.0.0.1:8000/v1 --out transcript.md

The script (1) asks Strata questions to the model as shipped (``memory=none``), with its trained memory alone
(``memory=engram``, nothing added to the prompt) and with both memories (``memory=hybrid``: wiki facts that Engram
has not verifiably learned go into the prompt), then (2) runs an agent session: the model gets the LLMWiki memory
tools from the MCP server (``wikiengram mcp``, spawned over stdio) and decides itself when to use them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

# Written for this demo (not training phrasings), each about one fact in the memory: (question, expected answer).
QUESTIONS = [
    ("What is Strata's default local endpoint?", "http://127.0.0.1:8080"),
    ("Where does Strata keep the n-gram table?", "SSD"),
    ("How many experts does Qwen3.8-Flash-Next have in Strata?", "24,576"),
    ("Which command checks whether the Strata server is running?", "strata launch --status"),
    ("Which environment variable turns on Strata's KV prefetch?", "STRATA_KV_PREFETCH=1"),
    ("How many more experts does each extra GB of VRAM hold in Strata?", "700"),
    ("Which tool of Strata's MCP server runs a short speed test?", "strata_benchmark"),
    ("Which NVIDIA driver do Strata's CUDA 13 packages need?", "580 or newer"),
]
AGENT_TASK = ("I want to run Qwen3.8-Flash-Next with Strata on a PC with a 16 GB graphics card. Check our project "
              "memory first, then tell me where the model's parts will live. Afterwards record the decision that "
              "we use the 16 GB card for this project.")
SYSTEM = ("You are a coding agent with long-term memory tools. Use memory_recall or memory_context before answering "
          "questions about the project, and record decisions with memory_record_decision. Answer briefly.")


def _schema(tool) -> dict:
    """A tool's JSON schema (``input_schema`` in mcp 2.x, ``inputSchema`` before)."""
    return getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {"type": "object"}


def chat(api: str, messages: list[dict], memory: str, tools: list[dict] | None = None, max_tokens: int = 48) -> dict:
    body = {"model": "wikiengram", "messages": messages, "memory": memory, "max_tokens": max_tokens,
            "temperature": 0.0}
    if tools:
        body["tools"] = tools
    t = time.perf_counter()
    r = httpx.post(f"{api}/chat/completions", json=body, timeout=3600)
    r.raise_for_status()
    out = r.json()
    out["seconds"] = round(time.perf_counter() - t, 1)
    return out


async def agent_session(api: str, workspace: str, log: list[str]) -> None:
    from mcp.client.stdio import StdioServerParameters, stdio_client

    try:
        from mcp import ClientSession
    except ImportError:  # pragma: no cover
        from mcp.client.session import ClientSession

    # the MCP client starts servers with a minimal environment: pass ours on, so the server imports this same
    # WikiEngram (e.g. a source checkout on PYTHONPATH) rather than whatever else is installed
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "wikiengram.cli", "mcp", "-w", workspace, "--scope", "strata"],
                                   env=dict(os.environ))
    async with stdio_client(params) as (r, w), ClientSession(r, w) as mcp:
        await mcp.initialize()
        listed = (await mcp.list_tools()).tools
        tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                   "parameters": _schema(t)}}
                 for t in listed if t.name in ("memory_recall", "memory_context", "memory_record_decision")]
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": AGENT_TASK}]
        log += ["### Agent session (model API + LLMWiki MCP)", "", f"**User:** {AGENT_TASK}", ""]
        for _ in range(6):
            res = chat(api, messages, "engram", tools=tools, max_tokens=256)
            msg = res["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}
                            if calls else {"role": "assistant", "content": msg.get("content") or ""})
            if not calls:
                log += [f"**Agent:** {msg.get('content')}", "", f"_({res['seconds']} s)_", ""]
                break
            for call in calls:
                args = json.loads(call["function"]["arguments"] or "{}")
                result = await mcp.call_tool(call["function"]["name"], args)
                text = "".join(getattr(c, "text", "") for c in result.content)
                log += [f"**Agent → MCP** `{call['function']['name']}({json.dumps(args, ensure_ascii=False)})`", "",
                        "```", text[:800], "```", ""]
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": text[:4000]})


def main() -> None:
    from wikiengram.utils.text import answer_in

    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--out", default="transcript.md")
    ap.add_argument("--agent-only", action="store_true", help="skip the question table, run the agent session")
    a = ap.parse_args()
    log = ["## Strata memory in Qwen3.8-Flash-Next", "",
           "### Questions asked through the model API (plain chat messages, phrasings not used in training)", "",
           "| Question | Expected | As shipped | Engram only (0 context tokens) | LLMWiki + Engram |",
           "|---|---|---|---|---|"]
    score = {"none": 0, "engram": 0, "hybrid": 0}

    def cell(r, expected, mode):
        text = (r["choices"][0]["message"]["content"] or "").replace("\n", " ").replace("|", "/").strip()
        ok = answer_in(text, expected)
        score[mode] += ok
        tokens = r["usage"]["prompt_tokens"]
        return f"{'✓' if ok else '✗'} {text[:90]} <sub>({tokens} prompt tokens, {r['seconds']} s)</sub>"

    if a.agent_only:
        log = log[:2]
    else:
        for q, expected in QUESTIONS:
            cells = [cell(chat(a.api, [{"role": "user", "content": q}], m), expected, m)
                     for m in ("none", "engram", "hybrid")]
            log.append(f"| {q} | `{expected}` | " + " | ".join(cells) + " |")
            print(log[-1], flush=True)
        n = len(QUESTIONS)
        log += [f"| **Correct** | | **{score['none']}/{n}** | **{score['engram']}/{n}** | **{score['hybrid']}/{n}** |",
                ""]
        Path(a.out).write_text("\n".join(log) + "\n")  # kept even if the agent session fails
    asyncio.run(agent_session(a.api, a.workspace, log))
    Path(a.out).write_text("\n".join(log) + "\n")
    print(f"transcript -> {a.out}")


if __name__ == "__main__":
    main()
