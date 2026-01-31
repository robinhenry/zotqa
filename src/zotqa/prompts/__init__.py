"""LLM prompt templates for zotqa."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        name: Name of the prompt file (without .md extension)

    Returns:
        The prompt content as a string
    """
    path = _PROMPTS_DIR / f"{name}.md"
    return path.read_text().strip()


def get_system_prompt() -> str:
    """Get the system prompt for the query LLM."""
    return load_prompt("system")


def get_user_prompt_template() -> str:
    """Get the user prompt template for the query LLM."""
    return load_prompt("user")
