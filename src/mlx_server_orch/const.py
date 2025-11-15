"""Constants used across the MLX server orchestrator.

This module defines filesystem paths and configuration file locations that
are referenced by the CLI and registry code. Keeping these values in a
single place makes them easy to override in tests or alternate deployments.
"""

from pathlib import Path

__version__ = "0.1.0"

# Root directory for the package (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory where per-model and process logs are written
LOG_ROOT = BASE_DIR / "logs"

# Default location of the models YAML file (can be overridden)
MODELS_CONFIG_FILE = BASE_DIR / "models.yaml"

DEFAULT_STARTING_PORT = 5005
