"""LLMWiki as an MCP server: AI agents (Claude Code, Cursor, Codex, any MCP client) read and write long-term
memory through tools.

No model is loaded, so this server is light and works next to any LLM. The Engram-compatible model, served by
``wikiengram serve-model`` (OpenAI-compatible) on the same workspace, consolidates what is written here into neural
memory and applies edits and deletions to it.

Run ``wikiengram mcp -w ~/.wikiengram`` (stdio) or add ``--transport streamable-http --port 8766``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from .memory import Memory
from .tools import TOOLS

INSTRUCTIONS = """Long-term memory shared across sessions (WikiEngram).
- At the start of a task call memory_context with the task; before answering questions that may depend on the
  user, the project or earlier work, call memory_recall.
- When you learn something durable, store it: memory_set_preference for user preferences, memory_record_decision
  for decisions and their reasons, memory_record_experience for lessons from a task, memory_remember or
  memory_add_fact for other facts.
- When a memory turns out wrong or outdated, fix it with memory_correct or memory_forget (by id)."""

_METHODS = {"memory_observe_conversation": "observe"}


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2

        return MCPServer
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP  # mcp 1.x

            return FastMCP
        except ImportError as e:  # pragma: no cover
            raise ImportError("the MCP server needs the mcp package: pip install 'wikiengram[mcp]'") from e


def build_server(memory: Memory, include_model_tools: bool = False):
    """An MCP server exposing the memory tools (tool names and descriptions from :mod:`.tools`, argument schemas
    from the :class:`Memory` methods' signatures)."""
    server = _server_class()("wikiengram", instructions=INSTRUCTIONS)
    for tool in TOOLS:
        if tool.needs_model and not include_model_tools:
            continue
        fn = getattr(memory, _METHODS.get(tool.name, tool.name.removeprefix("memory_")))
        server.add_tool(fn, name=tool.name, description=tool.description)
    return server


def serve(workspace: str | Path, scope: str = "", transport: str = "stdio", host: str = "127.0.0.1",
          port: int = 8766, include_model_tools: bool = False) -> None:
    server = build_server(Memory(workspace, scope=scope), include_model_tools)
    if transport == "stdio":
        server.run("stdio")
    elif _run_takes_address(server):  # mcp 2.x
        server.run(transport, host=host, port=port)
    else:  # mcp 1.x: host and port are server settings
        server.settings.host, server.settings.port = host, port
        server.run(transport)


def _run_takes_address(server) -> bool:
    params = inspect.signature(server.run).parameters.values()
    return any(p.kind is p.VAR_KEYWORD or p.name == "port" for p in params)
