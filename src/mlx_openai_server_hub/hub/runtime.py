"""Runtime state management for the hub daemon."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import httpx
from loguru import logger

from mlx_openai_server_hub.const import (
    DEFAULT_SIDECAR_HEALTH_INTERVAL,
    DEFAULT_SIDECAR_HEALTH_TIMEOUT,
    DEFAULT_SIDECAR_SHUTDOWN_TIMEOUT,
    HUB_POLL_INTERVAL_SECONDS,
)
from mlx_openai_server_hub.hub.config import HubConfigError, MLXHubConfig, load_hub_config


class HubRuntimeError(RuntimeError):
    """Raised when runtime operations cannot be completed."""


@dataclass(slots=True)
class ModelProcessState:
    """Process-level runtime status for a configured model."""

    name: str
    config: Any
    port: int
    host: str
    jit_enabled: bool
    group: str | None
    status: str = "configured"
    process: subprocess.Popen[bytes] | None = None
    return_code: int | None = None
    last_error: str | None = None
    start_timestamp: float | None = None
    last_active: float | None = None

    def is_running(self) -> bool:
        """Return True when the managed subprocess is alive."""

        return self.process is not None and self.process.poll() is None


@dataclass
class HubRuntime:
    """Holds hub configuration and mutable runtime state for the daemon."""

    config_path: Path
    hub_config: MLXHubConfig
    persisted_ports: dict[str, int] = field(default_factory=dict)
    server: Any | None = None

    def __post_init__(self) -> None:
        """Seed model state from the loaded hub configuration."""

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._refresh_group_lookup()
        self._started_at = time.time()
        self._model_state: dict[str, ModelProcessState] = {}
        for model in self.hub_config.models:
            if model.name is None:
                raise HubRuntimeError("Hub models must have a name after validation")
            jit_enabled = bool(getattr(model, "jit_enabled", False))
            self._model_state[model.name] = ModelProcessState(
                name=model.name,
                config=model,
                port=model.port,
                host=model.host,
                jit_enabled=jit_enabled,
                group=getattr(model, "group", None),
                status="configured" if jit_enabled else "stopped",
            )

        self._should_exit = False
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="hub-monitor", daemon=True
        )
        self._monitor_thread.start()

    @classmethod
    def from_path(cls, config_path: Path) -> HubRuntime:
        """Convenience constructor that loads hub.yaml from ``config_path``."""

        persisted_ports = cls._persisted_ports_from_path(config_path)
        hub_config = load_hub_config(config_path, persisted_ports=persisted_ports)
        return cls(config_path=config_path, hub_config=hub_config, persisted_ports=persisted_ports)

    @property
    def should_exit(self) -> bool:
        """Return True when the daemon should stop."""

        return self._should_exit

    def attach_server(self, server: Any) -> None:
        """Attach the running server instance for lifecycle control."""

        self.server = server

    def request_shutdown(self) -> None:
        """Signal the daemon to stop."""

        self._should_exit = True
        if self.server is not None:
            try:
                self.server.should_exit = True
            except AttributeError:
                logger.debug("Attached server does not expose should_exit; skipping flag")
        self._stop_event.set()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1)

    def reload_config(self) -> None:
        """Reload hub.yaml and reconcile running processes."""

        self.persisted_ports = self._persisted_ports_from_config()
        try:
            new_config = load_hub_config(self.config_path, persisted_ports=self.persisted_ports)
        except HubConfigError as exc:
            raise HubRuntimeError(str(exc)) from exc

        new_states: dict[str, ModelProcessState] = {}
        with self._lock:
            current_names = set(self._model_state)
            new_names = {model.name for model in new_config.models if model.name is not None}

            # Stop models removed from configuration
            for removed in current_names - new_names:
                logger.info("Stopping removed model '%s'", removed)
                self._stop_process(self._model_state[removed], kill=False)

            for model in new_config.models:
                if model.name is None:
                    raise HubRuntimeError("Hub models must have a name after validation")
                jit_enabled = bool(getattr(model, "jit_enabled", False))
                existing = self._model_state.get(model.name)
                new_state = ModelProcessState(
                    name=model.name,
                    config=model,
                    port=model.port,
                    host=model.host,
                    jit_enabled=jit_enabled,
                    group=getattr(model, "group", None),
                    status="configured" if jit_enabled else "stopped",
                )

                if existing is not None:
                    if (
                        self._config_matches(existing.config, model)
                        and existing.process is not None
                    ):
                        new_state.process = existing.process
                        new_state.status = existing.status
                        new_state.last_error = existing.last_error
                        new_state.start_timestamp = existing.start_timestamp
                        new_state.last_active = existing.last_active
                    else:
                        self._stop_process(existing, kill=False)

                new_states[model.name] = new_state

            self._model_state = new_states
            self.hub_config = new_config
            self._refresh_group_lookup()

        logger.info(
            "Reloaded hub config: %s model(s), %s group(s)",
            len(self.hub_config.models),
            len(self.hub_config.groups),
        )
        self.start_initial_models()

    def status_payload(self) -> dict[str, Any]:
        """Return a serializable status payload for HTTP responses."""

        self._refresh_process_states()

        models = [
            {
                "name": state.name,
                "port": state.port,
                "host": state.host,
                "jit_enabled": state.jit_enabled,
                "group": state.group,
                "status": state.status,
                "pid": state.process.pid if state.process else None,
                "return_code": state.return_code,
                "last_error": state.last_error,
                "started_at": state.start_timestamp,
                "last_active": state.last_active,
                "uptime_seconds": (time.time() - state.start_timestamp)
                if state.start_timestamp
                else None,
                "supervisor_log": str(self._supervisor_log_path(state.name)),
            }
            for state in self._model_state.values()
        ]

        return {
            "host": self.hub_config.host,
            "port": self.hub_config.port,
            "model_starting_port": self.hub_config.model_starting_port,
            "enable_status_page": self.hub_config.enable_status_page,
            "log_level": self.hub_config.log_level,
            "models": models,
            "groups": self._group_statuses(),
            "started_at": self._started_at,
        }

    def start_model(self, name: str) -> None:
        """Start or restart the backing MLX server process for a model."""

        state = self._get_model_state(name)
        with self._lock:
            eviction_target = self._pick_eviction_candidate(state)
            if state.is_running():
                state.status = "running"
                state.last_error = None
                state.last_active = time.time()
                logger.info(
                    "Model '%s' already running on %s:%s (pid=%s)",
                    name,
                    state.host,
                    state.port,
                    state.process.pid if state.process else "?",
                )
                return

            state.status = "starting"
            state.last_error = None
            state.return_code = None

        if eviction_target is not None:
            logger.info(
                "Group '%s' is at capacity (%s); unloading '%s' before starting '%s'",
                eviction_target.group,
                self._group_lookup[eviction_target.group].max_loaded
                if eviction_target.group
                else "n/a",
                eviction_target.name,
                name,
            )
            self._stop_process(eviction_target, kill=False)

        cmd = self._build_launch_command(state.config)
        log_path = self._supervisor_log_path(state.name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("a", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=log_handle,
                    env=self._inherit_env(),
                )
        except OSError as exc:
            with self._lock:
                state.status = "failed"
                state.last_error = str(exc)
            raise HubRuntimeError(f"Failed to start model '{name}': {exc}") from exc

        with self._lock:
            state.process = process
            state.start_timestamp = time.time()
            state.last_active = state.start_timestamp
            state.return_code = None

        logger.info("Start requested for model '%s' on %s:%s", name, state.host, state.port)
        healthy = self._wait_for_health(state)
        with self._lock:
            if healthy:
                state.status = "running"
                state.last_error = None
                state.last_active = time.time()
                logger.info(
                    "Model '%s' is healthy on %s:%s (pid=%s)",
                    name,
                    state.host,
                    state.port,
                    state.process.pid if state.process else "?",
                )
            else:
                state.status = "failed"
                state.last_error = f"Health check failed for {name}"
                self._stop_process(state, kill=True)
                state.status = "failed"
                raise HubRuntimeError(f"Model '{name}' failed health checks")

    def stop_model(self, name: str) -> None:
        """Stop the backing process for a model if it is running."""

        state = self._get_model_state(name)
        with self._lock:
            self._stop_process(state, kill=False)
        logger.info("Stop requested for model '%s'", name)

    def load_model(self, name: str) -> None:
        """Load a JIT-enabled model by starting its process."""

        state = self._get_model_state(name)
        if not state.jit_enabled:
            logger.info("Model '%s' is not JIT-enabled; delegating to start", name)
        self.start_model(name)

    def unload_model(self, name: str) -> None:
        """Unload a JIT-enabled model by stopping its process."""

        state = self._get_model_state(name)
        if not state.jit_enabled:
            logger.info("Model '%s' is not JIT-enabled; delegating to stop", name)
        self.stop_model(name)

    def stop_all_models(self) -> None:
        """Stop all managed model processes."""

        for state in list(self._model_state.values()):
            with self._lock:
                self._stop_process(state, kill=False)
        logger.info("Stop requested for all configured models")

    def _monitor_loop(self) -> None:
        """Background loop to enforce idle unload policies."""

        while not self._stop_event.wait(HUB_POLL_INTERVAL_SECONDS):
            self._refresh_process_states()
            self._enforce_idle_unload()

    def _refresh_process_states(self) -> None:
        """Synchronize runtime status with the underlying subprocess state."""

        with self._lock:
            for state in self._model_state.values():
                process = state.process
                if process is None:
                    if state.status not in {"configured", "stopped"}:
                        state.status = "stopped"
                    continue

                return_code = process.poll()
                if return_code is None:
                    continue

                state.return_code = return_code
                state.process = None
                state.last_active = time.time()
                if return_code not in (None, 0):
                    state.status = "failed"
                    state.last_error = state.last_error or f"Process exited with code {return_code}"
                else:
                    state.status = "stopped"
                    if state.last_error and "exited with code" in state.last_error:
                        state.last_error = None

    def _group_statuses(self) -> list[dict[str, Any]]:
        """Return group summaries including running counts."""

        summaries: list[dict[str, Any]] = []
        for group in getattr(self.hub_config, "groups", []):
            members = [state for state in self._model_state.values() if state.group == group.name]
            running = [state for state in members if state.is_running()]
            summaries.append(
                {
                    "name": group.name,
                    "max_loaded": group.max_loaded,
                    "idle_unload_trigger_min": group.idle_unload_trigger_min,
                    "running": len(running),
                    "total": len(members),
                }
            )
        return summaries

    def _enforce_idle_unload(self) -> None:
        """Unload JIT-enabled models that have been idle past their group threshold."""

        now = time.time()
        with self._lock:
            for state in self._model_state.values():
                if not state.jit_enabled or not state.is_running() or state.group is None:
                    continue

                group = self._group_lookup.get(state.group)
                if group is None or group.idle_unload_trigger_min is None:
                    continue

                last_active = state.last_active or state.start_timestamp or now
                idle_seconds = group.idle_unload_trigger_min * 60
                if now - last_active >= idle_seconds:
                    logger.info(
                        "Auto-unloading idle model '%s' in group '%s' after %ss idle",
                        state.name,
                        state.group,
                        int(now - last_active),
                    )
                    self._stop_process(state, kill=False)

    def _refresh_group_lookup(self) -> None:
        """Rebuild the internal mapping of group name to config."""

        self._group_lookup = {group.name: group for group in getattr(self.hub_config, "groups", [])}

    def _get_model_state(self, name: str) -> ModelProcessState:
        """Return the runtime state for a model or raise a runtime error."""

        try:
            return self._model_state[name]
        except KeyError as exc:
            raise HubRuntimeError(f"Unknown model '{name}'") from exc

    def _pick_eviction_candidate(self, target: ModelProcessState) -> ModelProcessState | None:
        """Determine whether a running model must be unloaded to satisfy group caps."""

        if target.group is None:
            return None

        group = self._group_lookup.get(target.group)
        if group is None or group.max_loaded is None:
            return None

        running = [
            state
            for state in self._model_state.values()
            if state.group == target.group and state.is_running() and state.name != target.name
        ]
        if len(running) < group.max_loaded:
            return None

        # Evict the oldest running model in the group
        def _age(state: ModelProcessState) -> float:
            return state.start_timestamp or 0.0

        return min(running, key=_age)

    def _build_launch_command(self, config: Any) -> list[str]:
        """Construct the mlx-openai-server launch command for a model."""

        args: list[str] = [
            sys.executable,
            "-m",
            "mlx_openai_server.main",
            "launch",
            "--model-path",
            str(config.model_path),
            "--model-type",
            str(config.model_type),
            "--port",
            str(config.port),
            "--host",
            str(config.host),
            "--max-concurrency",
            str(config.max_concurrency),
            "--queue-timeout",
            str(config.queue_timeout),
            "--queue-size",
            str(config.queue_size),
            "--log-level",
            str(config.log_level),
        ]

        if getattr(config, "context_length", None):
            args.extend(["--context-length", str(config.context_length)])

        if getattr(config, "config_name", None):
            args.extend(["--config-name", str(config.config_name)])

        if getattr(config, "quantize", None) is not None:
            args.extend(["--quantize", str(config.quantize)])

        if getattr(config, "disable_auto_resize", False):
            args.append("--disable-auto-resize")

        if getattr(config, "log_file", None):
            args.extend(["--log-file", str(config.log_file)])

        if getattr(config, "no_log_file", False):
            args.append("--no-log-file")

        lora_paths = getattr(config, "lora_paths", None)
        lora_scales = getattr(config, "lora_scales", None)
        if lora_paths:
            args.extend(["--lora-paths", ",".join(str(path) for path in lora_paths)])
        if lora_scales:
            args.extend(["--lora-scales", ",".join(str(scale) for scale in lora_scales)])

        if getattr(config, "enable_auto_tool_choice", False):
            args.append("--enable-auto-tool-choice")

        if getattr(config, "tool_call_parser", None):
            args.extend(["--tool-call-parser", str(config.tool_call_parser)])

        if getattr(config, "reasoning_parser", None):
            args.extend(["--reasoning-parser", str(config.reasoning_parser)])

        if getattr(config, "message_converter", None):
            args.extend(["--message-converter", str(config.message_converter)])

        if getattr(config, "trust_remote_code", False):
            args.append("--trust-remote-code")

        if getattr(config, "chat_template_file", None):
            args.extend(["--chat-template-file", str(config.chat_template_file)])

        if getattr(config, "debug", False):
            args.append("--debug")

        return args

    def _wait_for_health(self, state: ModelProcessState) -> bool:
        """Poll the model process health endpoint until ready or timed out."""

        deadline = time.time() + DEFAULT_SIDECAR_HEALTH_TIMEOUT
        host = "127.0.0.1" if state.host in {"0.0.0.0", "::"} else state.host
        url = f"http://{host}:{state.port}/health"
        while time.time() < deadline:
            if not state.is_running():
                return False

            try:
                response = httpx.get(url, timeout=5.0)
                if response.status_code == 200:
                    return True
            except httpx.RequestError:
                pass

            time.sleep(DEFAULT_SIDECAR_HEALTH_INTERVAL)

        return state.is_running()

    def _stop_process(self, state: ModelProcessState, *, kill: bool) -> None:
        """Terminate the process for a model and update status."""

        if state.process is None:
            state.status = "stopped"
            state.return_code = None
            state.last_active = time.time()
            return

        process = state.process
        if not kill:
            state.status = "stopping"
            process.terminate()
        else:
            state.status = "stopping"
            process.kill()

        try:
            process.wait(timeout=DEFAULT_SIDECAR_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process for model '%s' did not exit cleanly", state.name)

        state.return_code = process.returncode
        state.process = None
        state.last_active = time.time()
        if process.returncode not in (None, 0):
            state.status = "failed"
            state.last_error = state.last_error or f"Process exited with code {process.returncode}"
        else:
            state.status = "stopped"

    def _supervisor_log_path(self, name: str) -> Path:
        """Return the log path for supervisor output for a model."""

        return Path(self.hub_config.log_path) / f"{name}.supervisor.log"

    def _config_matches(self, lhs: Any, rhs: Any) -> bool:
        """Compare two MLXServerConfig objects for process compatibility."""

        keys = [
            "model_path",
            "model_type",
            "port",
            "host",
            "context_length",
            "max_concurrency",
            "queue_timeout",
            "queue_size",
            "config_name",
            "quantize",
            "disable_auto_resize",
            "lora_paths",
            "lora_scales",
            "enable_auto_tool_choice",
            "tool_call_parser",
            "reasoning_parser",
            "message_converter",
            "trust_remote_code",
            "chat_template_file",
            "jit_enabled",
        ]
        return all(getattr(lhs, key, None) == getattr(rhs, key, None) for key in keys)

    def _inherit_env(self) -> dict[str, str]:
        """Return a shallow copy of the environment for subprocesses."""

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def start_initial_models(self) -> None:
        """Start all non-JIT models configured in hub.yaml."""

        for state in list(self._model_state.values()):
            if state.jit_enabled:
                continue
            try:
                self.start_model(state.name)
            except HubRuntimeError as exc:
                logger.error("Failed to start model '%s': %s", state.name, exc)

    def _persisted_ports_from_config(self) -> dict[str, int]:
        """Build a map of model name to port from the current configuration."""

        ports: dict[str, int] = {}
        for model in self.hub_config.models:
            if model.name is None:
                continue
            ports[model.name] = model.port
        return ports

    @staticmethod
    def _persisted_ports_from_path(config_path: Path) -> dict[str, int]:
        """Extract persisted ports by reading the existing config if present."""

        if not config_path.exists():
            return {}

        try:
            loaded = load_hub_config(config_path)
        except HubConfigError:
            return {}

        ports: dict[str, int] = {}
        for model in loaded.models:
            if model.name is None:
                continue
            ports[model.name] = model.port
        return ports
