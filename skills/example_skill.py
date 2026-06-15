"""Example user skill for arc-llama.

This file demonstrates how to register a custom tool. Copy it to your configured
skills directory (default ``~/.config/arc-llama/skills``) and restart the server
to make the ``get_timestamp`` tool available to the agent.

See SKILLS.md for details.
"""
from __future__ import annotations

from datetime import datetime, timezone

from arc_llama.agent.tools import ToolResult, tool


@tool(
    name="get_timestamp",
    description="Return the current UTC time in ISO 8601 format.",
    parameters={"type": "object", "properties": {}},
)
def _get_timestamp(arguments: dict, ctx) -> ToolResult:
    return ToolResult(datetime.now(timezone.utc).isoformat())
