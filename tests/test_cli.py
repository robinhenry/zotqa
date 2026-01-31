"""Tests for the CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """Create a sample corpus for testing."""
    papers_dir = tmp_path / "papers"

    paper1_dir = papers_dir / "PAPER001"
    paper1_dir.mkdir(parents=True)

    (paper1_dir / "metadata.json").write_text(
        json.dumps(
            {
                "key": "PAPER001",
                "title": "Test Paper",
                "year": 2024,
                "authors": ["Test Author"],
                "abstract": "This is a test abstract.",
                "tags": ["test"],
            }
        )
    )

    (paper1_dir / "notes.md").write_text("Test notes about the paper.")

    return tmp_path


class TestCLI:
    """Tests for CLI commands."""

    def test_help(self) -> None:
        """Test --help flag."""
        result = subprocess.run(
            [sys.executable, "-m", "zotqa.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Query your Zotero library" in result.stdout
        assert "export" in result.stdout
        assert "index" in result.stdout
        assert "info" in result.stdout
        assert "ask" in result.stdout
        assert "init-prompts" in result.stdout

    def test_info_no_index(self, tmp_path: Path) -> None:
        """Test info command with no index."""
        result = subprocess.run(
            [sys.executable, "-m", "zotqa.cli", "info", "--index-dir", str(tmp_path / "index")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "No index found" in result.stdout

    def test_ask_no_index(self, tmp_path: Path) -> None:
        """Test ask command with no index."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "zotqa.cli",
                "ask",
                "test question",
                "--index-dir",
                str(tmp_path / "index"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "No index found" in result.stdout or "Error" in result.stdout

    def test_index_missing_corpus(self, tmp_path: Path) -> None:
        """Test index command with missing corpus."""
        result = subprocess.run(
            [sys.executable, "-m", "zotqa.cli", "index", str(tmp_path / "nonexistent")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "not found" in result.stdout
