"""Click-based CLI for mlx-openai-server-hub.

This module mirrors the Create-hub command surface while loading and validating
`hub.yaml`, dispatching operational commands to the hub daemon for orchestration.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, TypeVar

import click
from loguru import logger
import uvicorn

from mlx_openai_server_hub.const import DEFAULT_HUB_CONFIG_PATH, __version__
from mlx_openai_server_hub.hub.client import HubClient, HubClientError
from mlx_openai_server_hub.hub.config import HubConfigError, MLXHubConfig, load_hub_config
from mlx_openai_server_hub.hub.runtime import HubRuntime
from mlx_openai_server_hub.hub.server import create_app


def configure_cli_logging() -> None:
    """Configure Loguru for CLI stdout usage."""

    logger.remove()
    logger.add(lambda msg: click.echo(msg, nl=False), format="{message}", colorize=False)


@dataclass(slots=True)
class HubCLIContext:
    """Runtime context shared across Click commands."""

    config_path: Path
    hub_config: MLXHubConfig
    client: HubClient | None = None


def _resolve_config_path(config: str | Path | None) -> Path:
    """Resolve the hub.yaml path, falling back to the default location."""

    if config is None:
        return DEFAULT_HUB_CONFIG_PATH
    candidate = Path(config).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate


def _build_context(config: str | Path | None) -> HubCLIContext:
    """Load the hub configuration and return a Click context payload."""

    config_path = _resolve_config_path(config)
    try:
        hub_config = load_hub_config(config_path)
    except HubConfigError as exc:
        raise click.UsageError(str(exc)) from exc
    return HubCLIContext(config_path=config_path, hub_config=hub_config)


def _render_status(config: MLXHubConfig) -> str:
    """Render a human-readable status summary from the loaded configuration."""

    status = [
        f"Hub binding: {config.host}:{config.port}",
        f"Status page: {'enabled' if config.enable_status_page else 'disabled'}",
        f"Models: {len(config.models)} configured",
    ]
    for model in config.models:
        parts = [
            f"  - {model.name}",
            f"port={model.port}",
            f"path={model.model_path}",
        ]
        if model.group:
            parts.append(f"group={model.group}")
        if model.jit_enabled:
            parts.append("jit=yes")
        status.append(" ".join(parts))
    if config.groups:
        status.append(f"Groups: {len(config.groups)} configured")
        for group in config.groups:
            bits = [f"  - {group.name}"]
            if group.max_loaded is not None:
                bits.append(f"max_loaded={group.max_loaded}")
            if group.idle_unload_trigger_min is not None:
                bits.append(f"idle_unload_trigger_min={group.idle_unload_trigger_min}")
            status.append(" ".join(bits))
    return "\n".join(status)


def _client_from_ctx(
    ctx: HubCLIContext, host_override: str | None, port_override: int | None
) -> HubClient:
    """Construct a HubClient using CLI overrides or config defaults."""

    host = host_override or ctx.hub_config.host
    if host in {"0.0.0.0", "::"}:  # connectable loopback default
        host = "127.0.0.1"

    port = port_override or ctx.hub_config.port
    return HubClient(host=host, port=port)


TResult = TypeVar("TResult")


def _require_client(ctx: HubCLIContext) -> HubClient:
    """Return the initialized HubClient or raise a friendly Click error."""

    if ctx.client is None:
        raise click.ClickException("Hub client is not initialized; start the daemon first")
    return ctx.client


def _daemon_call[TResult](fn: Callable[[], TResult]) -> TResult:
    """Execute a daemon call and translate HubClient errors into Click errors."""

    try:
        return fn()
    except HubClientError as exc:
        raise click.ClickException(str(exc)) from exc


def _render_daemon_status(payload: dict[str, Any]) -> str:
    """Render status returned by the daemon for CLI output."""

    lines = [
        f"Hub binding: {payload.get('host')}:{payload.get('port')}",
        f"Status page: {'enabled' if payload.get('enable_status_page') else 'disabled'}",
        f"Model port seed: {payload.get('model_starting_port')}",
    ]
    models = payload.get("models", [])
    lines.append(f"Models: {len(models)} configured")
    for model in models:
        parts = [
            f"  - {model.get('name')}",
            f"status={model.get('status')}",
            f"port={model.get('host')}:{model.get('port')}",
        ]
        if model.get("pid"):
            parts.append(f"pid={model.get('pid')}")
        if model.get("group"):
            parts.append(f"group={model.get('group')}")
        if model.get("jit_enabled"):
            parts.append("jit=yes")
        if model.get("last_error"):
            parts.append(f"error={model.get('last_error')}")
        lines.append(" ".join(parts))

    groups = payload.get("groups") or []
    if groups:
        lines.append(f"Groups: {len(groups)} configured")
        for group in groups:
            snippet = [f"  - {group.get('name')}"]
            running = group.get("running")
            total = group.get("total")
            max_loaded = group.get("max_loaded")
            if running is not None and total is not None:
                snippet.append(f"running={running}/{max_loaded or total}")
            if group.get("idle_unload_trigger_min"):
                snippet.append(f"idle_unload_trigger_min={group.get('idle_unload_trigger_min')}")
            lines.append(" ".join(snippet))
    return "\n".join(lines)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "config",
    "--config",
    type=click.Path(exists=False, dir_okay=False, path_type=str),
    default=None,
    help=("Path to hub.yaml (default: ~/mlx-openai-server-hub/hub.yaml). Supports relative paths."),
)
@click.option(
    "controller_host",
    "--host",
    type=str,
    default=None,
    help="Controller host override for CLI->daemon requests (default: hub.yaml host or loopback).",
)
@click.option(
    "controller_port",
    "--port",
    type=int,
    default=None,
    help="Controller port override for CLI->daemon requests (default: hub.yaml port).",
)
@click.version_option(version=__version__, prog_name="mlx-openai-server-hub")
@click.pass_context
def cli(
    ctx: click.Context, config: str | None, controller_host: str | None, controller_port: int | None
) -> None:
    """mlx-openai-server-hub command-line interface."""

    configure_cli_logging()
    ctx.obj = _build_context(config)
    ctx.obj.client = _client_from_ctx(ctx.obj, controller_host, controller_port)


@cli.command("start")
@click.pass_obj
def start_command(ctx: HubCLIContext) -> None:
    """Start the hub daemon and its managed models."""

    runtime = HubRuntime.from_path(ctx.config_path)
    app = create_app(runtime)

    config = uvicorn.Config(
        app,
        host=runtime.hub_config.host,
        port=runtime.hub_config.port,
        log_level=runtime.hub_config.log_level.lower(),
        lifespan="on",
    )
    server = uvicorn.Server(config)
    runtime.attach_server(server)

    click.echo(f"Starting hub on {runtime.hub_config.host}:{runtime.hub_config.port}")
    runtime.start_initial_models()
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:  # pragma: no cover - interactive control path
        click.echo("Hub interrupted; shutting down")
    finally:
        runtime.stop_all_models()


@cli.command("reload")
@click.pass_obj
def reload_command(ctx: HubCLIContext) -> None:
    """Reload hub configuration and restart as needed."""

    client = _require_client(ctx)
    _daemon_call(client.reload)
    click.echo("Reload requested")


@cli.command("shutdown")
@click.pass_obj
def shutdown_command(ctx: HubCLIContext) -> None:
    """Shut down the hub daemon and stop managed models."""

    client = _require_client(ctx)
    _daemon_call(client.shutdown)
    click.echo("Shutdown requested")


@cli.command("status")
@click.pass_obj
def status_command(ctx: HubCLIContext) -> None:
    """Display status derived from the loaded hub configuration."""
    client = ctx.client
    if client is None:
        summary = _render_status(ctx.hub_config)
        click.echo(summary)
        return

    try:
        payload = _daemon_call(client.status)
    except click.ClickException as exc:
        click.echo(f"Hub daemon unreachable: {exc}. Showing config view instead.\n")
        summary = _render_status(ctx.hub_config)
        click.echo(summary)
        return

    summary = _render_daemon_status(payload)
    click.echo(summary)


@cli.command("start-model")
@click.argument("name")
@click.pass_obj
def start_model_command(ctx: HubCLIContext, name: str) -> None:
    """Start a specific model by name."""

    client = _require_client(ctx)
    _daemon_call(lambda: client.start_model(name))
    click.echo(f"Start requested for model '{name}'")


@cli.command("stop-model")
@click.argument("name")
@click.pass_obj
def stop_model_command(ctx: HubCLIContext, name: str) -> None:
    """Stop a specific model by name."""

    client = _require_client(ctx)
    _daemon_call(lambda: client.stop_model(name))
    click.echo(f"Stop requested for model '{name}'")


@cli.command("load-model")
@click.argument("name")
@click.pass_obj
def load_model_command(ctx: HubCLIContext, name: str) -> None:
    """Load a model into memory via the daemon."""

    client = _require_client(ctx)
    _daemon_call(lambda: client.load_model(name))
    click.echo(f"Load requested for model '{name}'")


@cli.command("unload-model")
@click.argument("name")
@click.pass_obj
def unload_model_command(ctx: HubCLIContext, name: str) -> None:
    """Unload a model from memory via the daemon."""

    client = _require_client(ctx)
    _daemon_call(lambda: client.unload_model(name))
    click.echo(f"Unload requested for model '{name}'")


@cli.command("stop")
@click.pass_obj
def stop_command(ctx: HubCLIContext) -> None:
    """Stop all managed models while keeping the daemon running."""

    client = _require_client(ctx)
    _daemon_call(client.stop_all_models)
    click.echo("Stop-all requested")


@cli.command("watch")
@click.option("interval", "--interval", type=float, default=5.0, show_default=True)
@click.pass_obj
def watch_command(ctx: HubCLIContext, interval: float) -> None:
    """Periodically display configuration status (daemon wiring pending)."""

    client = _require_client(ctx)
    click.echo("Watching hub status (Ctrl+C to exit)...")
    try:
        while True:
            payload = _daemon_call(client.status)
            click.echo(_render_daemon_status(payload))
            time.sleep(interval)
    except KeyboardInterrupt:  # pragma: no cover - interactive control path
        click.echo("Stopped watching")


def main() -> None:
    """Entry point for the console script."""

    cli.main(standalone_mode=True)


if __name__ == "__main__":  # pragma: no cover
    main()
