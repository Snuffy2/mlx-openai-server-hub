"""MLX OpenAI Server Hub package.

This package contains the CLI and supporting modules used to launch and
manage MLX OpenAI Server workers. The package itself has no runtime
side-effects; functionality is exposed through `main` and the registry
helpers.
"""

from .const import __version__

__all__ = ["__version__"]
