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

from .const import LOG_ROOT, MODELS_CONFIG_FILE


class ModelRegistryError(RuntimeError):
    """Raised when the models configuration file is invalid."""


@dataclass(slots=True)
class ModelEntry:
    """Container for a single registry entry.

    Attributes:
        name: nickname for the model
        config: MLXServerConfig instance for the model
        default: whether this model is a default selection

    """

    name: str
    config: MLXServerConfig
    default: bool


class ModelRegistry:
    """Registry that loads and validates model definitions from YAML.

    The registry exposes iteration and lookup helpers and can build
    per-model `MLXServerConfig` instances for runtime use.
    """

    def __init__(self, config_file: Path | str = MODELS_CONFIG_FILE):
        """Load and validate models from `config_file` into an in-memory registry.

        The constructor prepares internal bookkeeping and calls `reload()`
        to populate the registry immediately.
        """
        self.config_file = Path(config_file)
        self._entries: dict[str, ModelEntry] = {}
        self._ordered_names: list[str] = []
        self._default_names: list[str] = []
        self._config_fields = {field.name: field for field in fields(MLXServerConfig)}
        self.reload()

    def reload(self) -> None:
        """Reload and validate the YAML models configuration file.

        This reads `models.yaml` (or the supplied path), validates its
        structure and builds internal `ModelEntry` objects. Errors in
        the file raise `ModelRegistryError` with a helpful message.
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
        """Yield all `ModelEntry` objects in the order defined in the file."""
        for name in self._ordered_names:
            yield self._entries[name]

    def default_entries(self) -> Iterable[ModelEntry]:
        """Yield entries that are marked as default in the configuration."""
        for name in self._default_names:
            yield self._entries[name]

    def get_entry(self, name: str) -> ModelEntry:
        """Return the `ModelEntry` for `name` or raise `ModelRegistryError`.

        This performs a lookup in the registry and raises a human-readable
        error when the requested name does not exist.
        """
        try:
            return self._entries[name]
        except KeyError as exc:
            raise ModelRegistryError(f"Unknown model nickname '{name}'") from exc

    def build_model_map(self, names: list[str] | None) -> dict[str, MLXServerConfig]:
        """Return a mapping of model name -> MLXServerConfig for selection.

        If `names` is provided, those specific entries are returned; when
        `names` is None the registry's default entries are used. Each
        returned config is a deep copy to allow per-process mutation.
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
        """Return a list of names marked as default in the same order."""
        return list(self._default_names)

    def ordered_names(self) -> list[str]:
        """Return a list of all configured model names in file order."""
        return list(self._ordered_names)


REGISTRY = ModelRegistry()
