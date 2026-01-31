"""LLM prompt templates for zotqa."""

import shutil
from pathlib import Path

from platformdirs import user_config_dir

_DEFAULT_PROMPTS_DIR = Path(__file__).parent
_USER_CONFIG_DIR = Path(user_config_dir("zotqa", appauthor=False)) / "prompts"


def get_user_prompts_dir() -> Path:
    """Get the user's prompts directory."""
    return _USER_CONFIG_DIR


def load_prompt(name: str) -> str:
    """Load a prompt template, checking user config first, then defaults.

    Args:
        name: Name of the prompt file (without .md extension)

    Returns:
        The prompt content as a string
    """
    # Check user config directory first
    user_path = _USER_CONFIG_DIR / f"{name}.md"
    if user_path.exists():
        return user_path.read_text().strip()

    # Fall back to default prompts
    default_path = _DEFAULT_PROMPTS_DIR / f"{name}.md"
    return default_path.read_text().strip()


def init_user_prompts(force: bool = False) -> Path:
    """Copy default prompts to user config directory for customization.

    Args:
        force: If True, overwrite existing user prompts

    Returns:
        Path to the user prompts directory

    Raises:
        FileExistsError: If user prompts already exist and force=False
    """
    if _USER_CONFIG_DIR.exists() and not force:
        existing_files = list(_USER_CONFIG_DIR.glob("*.md"))
        if existing_files:
            raise FileExistsError(f"User prompts already exist at {_USER_CONFIG_DIR}. " "Use force=True to overwrite.")

    # Create directory
    _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Copy all .md files from defaults
    for prompt_file in _DEFAULT_PROMPTS_DIR.glob("*.md"):
        dest = _USER_CONFIG_DIR / prompt_file.name
        shutil.copy2(prompt_file, dest)

    return _USER_CONFIG_DIR


def get_system_prompt() -> str:
    """Get the system prompt for the query LLM."""
    return load_prompt("system")


def get_user_prompt_template() -> str:
    """Get the user prompt template for the query LLM."""
    return load_prompt("user")
