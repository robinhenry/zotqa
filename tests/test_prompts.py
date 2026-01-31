"""Tests for the prompts module."""

from pathlib import Path

import pytest

from zotqa.prompts import get_user_prompts_dir, init_user_prompts, load_prompt


class TestPrompts:
    """Tests for prompt loading and initialization."""

    def test_load_default_prompts(self) -> None:
        """Test loading default prompts."""
        system_prompt = load_prompt("system")
        user_prompt = load_prompt("user")

        assert len(system_prompt) > 0
        assert len(user_prompt) > 0
        assert "research assistant" in system_prompt.lower()
        assert "{context}" in user_prompt
        assert "{question}" in user_prompt

    def test_get_user_prompts_dir(self) -> None:
        """Test getting user prompts directory."""
        prompts_dir = get_user_prompts_dir()
        # Should be a valid path that includes "zotqa" and "prompts"
        assert isinstance(prompts_dir, Path)
        assert "zotqa" in str(prompts_dir).lower()
        assert "prompts" in str(prompts_dir).lower()

    def test_init_user_prompts(self, tmp_path: Path, monkeypatch) -> None:
        """Test initializing user prompts."""
        # Redirect config dir to tmp_path
        prompts_dir = tmp_path / "prompts"
        monkeypatch.setattr("zotqa.prompts._USER_CONFIG_DIR", prompts_dir)

        # Initialize prompts
        result = init_user_prompts()

        assert result == prompts_dir
        assert prompts_dir.exists()
        assert (prompts_dir / "system.md").exists()
        assert (prompts_dir / "user.md").exists()

        # Check content was copied
        system_content = (prompts_dir / "system.md").read_text()
        assert len(system_content) > 0
        assert "research assistant" in system_content.lower()

    def test_init_user_prompts_already_exists(self, tmp_path: Path, monkeypatch) -> None:
        """Test initializing user prompts when they already exist."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "system.md").write_text("custom system prompt")

        monkeypatch.setattr("zotqa.prompts._USER_CONFIG_DIR", prompts_dir)

        # Should raise error without force
        with pytest.raises(FileExistsError):
            init_user_prompts(force=False)

        # Should succeed with force
        result = init_user_prompts(force=True)
        assert result == prompts_dir

        # Content should be overwritten with default
        system_content = (prompts_dir / "system.md").read_text()
        assert system_content != "custom system prompt"

    def test_load_custom_prompts(self, tmp_path: Path, monkeypatch) -> None:
        """Test loading custom user prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(parents=True)

        custom_system = "Custom system prompt"
        custom_user = "Custom user prompt with {context} and {question}"

        (prompts_dir / "system.md").write_text(custom_system)
        (prompts_dir / "user.md").write_text(custom_user)

        monkeypatch.setattr("zotqa.prompts._USER_CONFIG_DIR", prompts_dir)

        # Should load custom prompts
        assert load_prompt("system") == custom_system
        assert load_prompt("user") == custom_user

    def test_fallback_to_defaults(self, tmp_path: Path, monkeypatch) -> None:
        """Test fallback to default prompts when user prompts don't exist."""
        prompts_dir = tmp_path / "prompts"
        monkeypatch.setattr("zotqa.prompts._USER_CONFIG_DIR", prompts_dir)

        # User prompts don't exist, should load defaults
        system_prompt = load_prompt("system")
        user_prompt = load_prompt("user")

        assert len(system_prompt) > 0
        assert len(user_prompt) > 0
        assert "research assistant" in system_prompt.lower()
