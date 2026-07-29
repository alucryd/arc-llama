"""Tests for arc_llama.cli — arcllama entry point."""
from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from arc_llama.cli import arcllama_main
from arc_llama.config import Config


def test_arcllama_main_loads_config_and_passes_to_tui():
    cfg = Config()
    cfg.server.host = "127.0.0.1"
    cfg.server.port = 11437

    with (
        patch("arc_llama.cli._experimental_agent_enabled", return_value=True),
        patch("arc_llama.cli.load_config", return_value=cfg) as mock_load_config,
        patch("arc_llama.cli.run_agent_tui") as mock_run,
    ):
        runner = CliRunner()
        result = runner.invoke(arcllama_main, ["--model", "qwen"])

    assert result.exit_code == 0, result.output
    mock_load_config.assert_called_once_with()
    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["config"] is cfg
    assert kwargs["model"] == "qwen"
