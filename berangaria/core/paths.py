"""The single place where project paths are resolved.

This module imports nothing from the project, so both `berangaria.config` and
`berangaria.core.state` can use it without creating a cycle.

Relative paths in the config and in environment variables are resolved against
the repository root rather than the current working directory: the bot starts as
`python -m berangaria`, from the container, and from scripts in `scripts/`, and
all three have to reach the same config.yaml, bot_state.db, and bot.log.
"""
from pathlib import Path

# berangaria/core/paths.py -> berangaria/core -> berangaria -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str) -> str:
    """Absolute path: a relative value is resolved against the repository root."""
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)
