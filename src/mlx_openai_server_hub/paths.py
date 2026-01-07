"""Runtime path helpers for MLX OpenAI Server Hub."""

from __future__ import annotations

from pathlib import Path

from .const import DEFAULT_BASE_PATH


class _RuntimePaths:
    """Keep track of runtime directories derived from the selected base path."""

    def __init__(self, base_path: Path):
        self.base_path = base_path

    @property
    def log_root(self) -> Path:
        return self.base_path / "logs"

    @property
    def pid_dir(self) -> Path:
        return self.base_path / "pids"

    @property
    def models_file(self) -> Path:
        return self.base_path / "models.yaml"


_PATHS = _RuntimePaths(DEFAULT_BASE_PATH)


def base_path() -> Path:
    """Return the currently active base path (defaults to ~/mlx-openai-server-hub)."""

    return _PATHS.base_path


def log_root() -> Path:
    """Return the path where per-model and process logs live."""

    return _PATHS.log_root


def pid_dir() -> Path:
    """Return the path where PID metadata files are stored."""

    return _PATHS.pid_dir


def models_config_file() -> Path:
    """Return the resolved models.yaml path for the active base path."""

    return _PATHS.models_file


def set_base_path(path: Path | str) -> Path:
    """Update the base path and derived runtime directories."""

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()

    _PATHS.base_path = resolved
    return _PATHS.base_path
