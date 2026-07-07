"""Tests for repo map and semantic search helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from arc_llama.agent.repo_map import SemanticIndex, build_repo_map


def test_build_repo_map(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def hello():\n    pass\n\nclass Greeter:\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")

    text = build_repo_map(tmp_path, max_entries=50)
    assert "src/main.py" in text
    assert "hello" in text
    assert "Greeter" in text
    assert "README.md" in text


def test_semantic_search_requires_optional_dependency(tmp_path: Path) -> None:
    index = SemanticIndex(tmp_path / "idx")
    # If fastembed is installed this will index; if not it raises RuntimeError.
    try:
        import fastembed  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="semantic"):
            index.index(tmp_path)
        return

    # When the optional dep is present, exercise the full flow.
    (tmp_path / "main.py").write_text(
        "def authenticate():\n    pass\n\ndef login():\n    pass\n",
        encoding="utf-8",
    )
    stats = index.index(tmp_path)
    assert stats["indexed_files"] == 1
    results = index.search(tmp_path, "authentication logic", top_k=2)
    assert len(results) <= 2
    assert any("authenticate" in r["path"] or "authenticate" in r["snippet"] for r in results)
