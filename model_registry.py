"""Model registry utilities.

This module provides a tiny registry that loads model definitions from a
YAML file (`models.yaml`) and validates them against the MLXServerConfig
dataclass. It exposes a `REGISTRY` singleton used by the CLI to enumerate and
instantiate per-model configurations.
"""

from __future__ import annotations

from collections.abc import Iterable
import copy
from dataclasses import dataclass, fields
from pathlib import Path

from app.config import MLXServerConfig
import yaml

from const import LOG_ROOT, MODELS_CONFIG_FILE


class ModelRegistryError(RuntimeError):
    """Raised when the models configuration file is invalid."""


@dataclass(slots=True)
class ModelEntry:
    """A registry entry representing a configured model.

    Attributes:
        name: Short nickname for the model used by the CLI.
        config: An `MLXServerConfig` instance with model settings.
        default: Whether this model should be started when no names are
            provided to the `start` command.

    """

    name: str
    config: MLXServerConfig
    default: bool


class ModelRegistry:
    """Loads and validates MLX models from a YAML definition file.

    The registry reads a top-level mapping with a `models:` list. Each entry is
    validated for required fields and against the fields available on
    `app.config.MLXServerConfig`. Valid entries are stored as `ModelEntry`
    instances and exposed through iteration and lookup helpers.
    """

    def __init__(self, config_file: Path | str = MODELS_CONFIG_FILE):
        """Initialize the registry and load entries from `config_file`.

        Args:
            config_file: Path to the YAML file describing models. Defaults to
                the module-level `MODELS_CONFIG_FILE`.

        """
        self.config_file = Path(config_file)
        self._entries: dict[str, ModelEntry] = {}
        self._ordered_names: list[str] = []
        self._default_names: list[str] = []
        self._config_fields = {field.name: field for field in fields(MLXServerConfig)}
        self.reload()

    def reload(self) -> None:
        """Reload and validate models from the YAML configuration file.

        This replaces the registry's internal entries with the contents of the
        YAML file. Several validation errors raise `ModelRegistryError` to
        signal problems to the caller (typically the CLI).
        """

        if not self.config_file.exists():
            raise ModelRegistryError(f"Models file not found: {self.config_file}")

        with self.config_file.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        if not isinstance(raw, dict):
            raise ModelRegistryError("models.yaml must contain a mapping with a 'models' list")

        items = raw.get("models")
        if not isinstance(items, list):
            raise ModelRegistryError("models.yaml must define a 'models' list")

        entries: dict[str, ModelEntry] = {}
        ordered_names: list[str] = []
        default_names: list[str] = []

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise ModelRegistryError(f"Entry #{idx + 1} must be a mapping")

            name = item.get("name")
            if not name:
                raise ModelRegistryError(f"Entry #{idx + 1} is missing required field 'name'")
            if name in entries:
                raise ModelRegistryError(f"Duplicate model name '{name}'")

            model_path = item.get("model_path")
            if not model_path:
                raise ModelRegistryError(f"Model '{name}' is missing required field 'model_path'")

            default = bool(item.get("default", False))

            init_kwargs: dict[str, object] = {"model_path": model_path}
            post_init: dict[str, object] = {}

            for key, value in item.items():
                if key in {"name", "model_path", "default"}:
                    continue
                field = self._config_fields.get(key)
                if field is None:
                    raise ModelRegistryError(
                        f"Unknown MLXServerConfig option '{key}' in model '{name}'"
                    )
                if field.init:
                    init_kwargs[key] = value
                else:
                    post_init[key] = value

            config = MLXServerConfig(**init_kwargs)
            for key, value in post_init.items():
                setattr(config, key, value)

            if not config.no_log_file and not config.log_file:
                config.log_file = str(Path(LOG_ROOT) / f"{name}.log")

            entries[name] = ModelEntry(name=name, config=config, default=default)
            ordered_names.append(name)
            if default:
                default_names.append(name)

        if not ordered_names:
            raise ModelRegistryError("models.yaml does not define any models")
        if not default_names:
            raise ModelRegistryError("models.yaml must mark at least one model as default")

        self._entries = entries
        self._ordered_names = ordered_names
        self._default_names = default_names

    def all_entries(self) -> Iterable[ModelEntry]:
        """Yield all registered ModelEntry objects in the configured order."""
        for name in self._ordered_names:
            yield self._entries[name]

    def default_entries(self) -> Iterable[ModelEntry]:
        """Yield ModelEntry objects that are marked as defaults."""
        for name in self._default_names:
            yield self._entries[name]

    def get_entry(self, name: str) -> ModelEntry:
        """Return the ModelEntry for `name` or raise `ModelRegistryError`.

        Args:
            name: The model nickname to look up.

        Raises:
            ModelRegistryError: if the name is not registered.

        """
        try:
            return self._entries[name]
        except KeyError as exc:  # pragma: no cover - handled by CLI error path
            raise ModelRegistryError(f"Unknown model nickname '{name}'") from exc

    def build_model_map(self, names: list[str] | None) -> dict[str, MLXServerConfig]:
        """Build a mapping of model name -> MLXServerConfig for startup.

        If `names` is provided, those specific models are returned. Otherwise
        the models marked as defaults are used. Returned config objects are
        deep-copied to ensure callers can mutate them (for example to assign
        ports) without affecting the registry.
        """
        if names:
            selected = [self.get_entry(name) for name in names]
        else:
            selected = list(self.default_entries())

        if not selected:
            raise ModelRegistryError("No models selected for startup")

        models: dict[str, MLXServerConfig] = {}
        for entry in selected:
            models[entry.name] = copy.deepcopy(entry.config)
        return models

    def default_names(self) -> list[str]:
        """Return a list of default model names in configured order."""
        return list(self._default_names)

    def ordered_names(self) -> list[str]:
        """Return a list of all model names in configured order."""
        return list(self._ordered_names)


REGISTRY = ModelRegistry()
