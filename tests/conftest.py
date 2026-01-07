"""Pytest configuration with stubs for missing mlx_openai_server dependency."""

from __future__ import annotations

import sys
import types
from typing import Any

from mlx_openai_server_hub import const


class MLXServerConfig:
    """Minimal stand-in for mlx_openai_server.config.MLXServerConfig.

    The real package is not installed in CI for hub-only testing, so this
    lightweight shim provides the attributes the hub code relies on.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Populate config fields with provided values or hub defaults."""
        self.name: str | None = kwargs.get("name")
        self.model_path: str = str(kwargs["model_path"])
        self.model_type: str = str(kwargs.get("model_type", const.DEFAULT_MODEL_TYPE))
        self.port: int = int(kwargs.get("port", const.DEFAULT_MODEL_STARTING_PORT))
        self.host: str = str(kwargs.get("host", const.DEFAULT_BIND_HOST))
        self.context_length: int = int(kwargs.get("context_length", const.DEFAULT_CONTEXT_LENGTH))
        self.max_concurrency: int = int(
            kwargs.get("max_concurrency", const.DEFAULT_MAX_CONCURRENCY)
        )
        self.queue_timeout: int = int(kwargs.get("queue_timeout", const.DEFAULT_QUEUE_TIMEOUT))
        self.queue_size: int = int(kwargs.get("queue_size", const.DEFAULT_QUEUE_SIZE))
        self.config_name: str | None = kwargs.get("config_name")
        self.quantize: Any = kwargs.get("quantize", const.DEFAULT_QUANTIZE)
        self.disable_auto_resize: bool = bool(
            kwargs.get("disable_auto_resize", const.DEFAULT_DISABLE_AUTO_RESIZE)
        )
        self.lora_paths: list[str] | None = kwargs.get("lora_paths")
        self.lora_scales: list[str] | None = kwargs.get("lora_scales")
        self.enable_auto_tool_choice: bool = bool(
            kwargs.get("enable_auto_tool_choice", const.DEFAULT_ENABLE_AUTO_TOOL_CHOICE)
        )
        self.tool_call_parser: str | None = kwargs.get("tool_call_parser")
        self.reasoning_parser: str | None = kwargs.get("reasoning_parser")
        self.message_converter: str | None = kwargs.get("message_converter")
        self.trust_remote_code: bool = bool(
            kwargs.get("trust_remote_code", const.DEFAULT_TRUST_REMOTE_CODE)
        )
        self.chat_template_file: str | None = kwargs.get("chat_template_file")
        self.jit_enabled: bool = bool(kwargs.get("jit_enabled", const.DEFAULT_JIT_ENABLED))
        self.no_log_file: bool = bool(kwargs.get("no_log_file", const.DEFAULT_NO_LOG_FILE))
        self.log_file: str | None = kwargs.get("log_file", const.DEFAULT_LOG_FILE)
        self.log_level: str = str(kwargs.get("log_level", const.DEFAULT_LOG_LEVEL))
        self.group: str | None = kwargs.get("group")
        self.max_tokens: int | None = kwargs.get("max_tokens")
        self.is_default_model: bool = bool(
            kwargs.get("is_default_model", const.DEFAULT_IS_DEFAULT_MODEL)
        )
        self.enable_status_page: bool = bool(
            kwargs.get("enable_status_page", const.DEFAULT_ENABLE_STATUS_PAGE)
        )
        self.debug: bool = bool(kwargs.get("debug", False))

    # mimic attribute presence without extra behavior


# Inject stub module before tests import hub.config
mlx_openai_server = types.ModuleType("mlx_openai_server")
mlx_openai_server.config = types.ModuleType("mlx_openai_server.config")
mlx_openai_server.config.MLXServerConfig = MLXServerConfig

sys.modules["mlx_openai_server"] = mlx_openai_server
sys.modules["mlx_openai_server.config"] = mlx_openai_server.config
