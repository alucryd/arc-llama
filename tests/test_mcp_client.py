"""Tests for MCP client manager."""
from __future__ import annotations

import pytest

from arc_llama.agent.mcp_client import MCPClientManager, MCPServerConfig


@pytest.mark.asyncio
async def test_mcp_manager_raises_without_optional_dependency() -> None:
    manager = MCPClientManager([MCPServerConfig(name="test", command="echo")])
    try:
        import mcp  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="mcp"):
            await manager.start()
        return

    # If mcp is installed, this test would require a real server; skip.
    pytest.skip("mcp package is installed; skipping no-dependency assertion")
