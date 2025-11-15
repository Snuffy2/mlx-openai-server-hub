"""MLX OpenAI Server Orchestrator CLI.

Command-line interface to start, stop, list, and inspect multiple MLX OpenAI
server worker instances on the same host. It manages per-process logging,
PID metadata, and port assignment so multiple servers can coexist.
"""

import argparse
import asyncio
import atexit
import contextlib
import json
import logging
import multiprocessing
import multiprocessing.util as mp_util
import os
from pathlib import Path
import re
import signal
import sys
import time

from loguru import logger

from const import LOG_ROOT
from model_registry import REGISTRY, ModelRegistryError

STARTING_PORT = 5005
PID_DIR = Path(LOG_ROOT) / "pids"


def configure_cli_logging() -> None:
    """Configure Loguru for command-line output.

    The CLI uses a simplified Loguru sink that prints only the message text
    (no timestamps) so interactive invocations remain clean.
    """
    with contextlib.suppress(Exception):
        logger.remove()
    logger.add(sys.stdout, format="{message}", colorize=False)


def refresh_model_registry() -> None:
    """Reload the in-memory model registry and exit on validation errors.

    This helper ensures that CLI commands operate against the latest
    `models.yaml` contents. If the registry cannot be loaded the CLI exits
    with a non-zero status after logging the problem.
    """
    try:
        REGISTRY.reload()
    except ModelRegistryError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse.ArgumentParser for the CLI.

    Returns:
        Configured ArgumentParser with the subcommands `start`, `stop`,
        `models`, `status`, and `help`.

    """
    parser = argparse.ArgumentParser(
        prog="mlx-server-orch",
        description="MLX OpenAI Server Orchestrator — manage MLX model server processes.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser(
        "start",
        help="Start one or more models (runs in background).",
    )
    start_parser.add_argument(
        "names",
        nargs="*",
        metavar="name",
        help="Model nickname(s) to start.",
    )

    parser.start_parser = start_parser

    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop running model processes.",
    )
    stop_parser.add_argument(
        "names",
        nargs="*",
        metavar="name",
        help="Model nickname(s) to stop (default: all running).",
    )

    subparsers.add_parser("models", help="List available model nicknames and paths.")
    subparsers.add_parser("status", help="Show running models and their PIDs.")
    subparsers.add_parser("help", help="Show this help message.")

    return parser


def _process_worker(name, model):
    # Configure per-process logging BEFORE importing/initializing heavy modules
    try:
        configure_process_logging(name=name)
    except (OSError, PermissionError) as exc:  # pragma: no cover - logging best-effort
        logger.exception(f"Failed to configure process logging for {name}: {exc}")

    with contextlib.suppress(Exception):
        os.setsid()

    _register_pid_cleanup(name)

    # Worker runs in a fresh process; import modules inside the process
    from app.main import start as _start  # noqa: PLC0415

    # The `model` argument is passed as a pickled MLXServerConfig instance
    # (or another config object) from the parent process. Use it directly.
    asyncio.run(_start(model))


"""Worker entrypoint invoked inside a spawned process.

The function configures per-process logging, registers PID cleanup hooks,
imports the MLX server startup routine and runs it with the provided
configuration object.
"""


def configure_process_logging(name: str | None = None):
    """Configure per-process stdout/stderr and Loguru logging.

    Call this as the very first action in a child process.
    """
    Path(LOG_ROOT).mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    suffix = name or str(pid)

    # Single combined log file for this process (captures stdout, stderr, and app logs)
    app_log = Path(LOG_ROOT) / f"{suffix}-app.log"

    # Open one file object (append, line-buffered)
    app_file = app_log.open("a", buffering=1, encoding="utf-8")

    # Replace OS-level FDs so C extensions and subprocesses also go to the file
    os.dup2(app_file.fileno(), 1)
    os.dup2(app_file.fileno(), 2)

    # Replace Python-level sys.stdout/sys.stderr to capture print() etc.
    sys.stdout = app_file
    sys.stderr = app_file

    # Configure Loguru to write to the same path. Use the path (string) so
    # rotation and other file-based options are supported by Loguru.
    with contextlib.suppress(Exception):
        logger.remove()
    # Disable Loguru rotation to keep a single continuous file per process
    # Disable colorization in file logs and write to the file path
    logger.add(str(app_log), backtrace=True, diagnose=True, enqueue=False, colorize=False)

    # Route stdlib logging into Loguru
    # ansi_re = re.compile(r"\x1B\[[0-9;]*[mK]")
    ansi_re = re.compile(r"\x1B?\[[0-9;]*[mK]")

    def _strip_ansi(s: str) -> str:
        return ansi_re.sub("", s)

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except (LookupError, ValueError):
                level = record.levelno
            msg = record.getMessage()
            msg = _strip_ansi(msg)
            logger.opt(depth=6, exception=record.exc_info).log(level, msg)

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Ensure files get flushed/closed at exit
    def _close_files():
        with contextlib.suppress(Exception):
            app_file.flush()
        with contextlib.suppress(Exception):
            app_file.close()

    atexit.register(_close_files)


def _register_pid_cleanup(name: str) -> None:
    """Register an atexit hook to remove the PID metadata file for `name`.

    The PID metadata file tracks the pid/port for a running model and should
    be removed when the worker process exits.
    """
    ensure_runtime_dirs()
    pid_path = pid_file(name)

    def _cleanup():
        with contextlib.suppress(FileNotFoundError):
            pid_path.unlink()

    atexit.register(_cleanup)


def pid_file(name: str) -> Path:
    """Return the path to the PID metadata file for `name`."""
    return PID_DIR / f"{name}.json"


def ensure_runtime_dirs() -> None:
    """Ensure runtime directories (logs and pid dir) exist.

    Creates `LOG_ROOT` and the PID directory if they are missing.
    """
    Path(LOG_ROOT).mkdir(parents=True, exist_ok=True)
    PID_DIR.mkdir(parents=True, exist_ok=True)


def assign_ports(
    models: dict[str, object], reserved_ports: set[int] | None = None
) -> dict[str, object]:
    """Assign TCP ports to model configs avoiding reserved ports.

    Models may optionally include a `port` field. Ports already present and
    not equal to the MLX default (8000) will be preserved if not reserved.
    Any model without an explicit port will be assigned a free port starting
    at `STARTING_PORT` and incrementing to avoid collisions with
    `reserved_ports`.

    Returns the same `models` mapping with `port` attributes set on config
    objects.
    """
    used_ports: dict[str, int] = {}
    claimed_ports: set[int] = set(reserved_ports or set())

    for name, model in models.items():
        port = getattr(model, "port", None)
        if port and port != 8000 and port not in claimed_ports:
            claimed_ports.add(port)
            used_ports[name] = port

    for name, model in models.items():
        port = getattr(model, "port", None)
        if port is None or port == 8000 or used_ports.get(name) != port:
            temp_port = STARTING_PORT
            while temp_port in claimed_ports:
                temp_port += 1
            setattr(model, "port", temp_port)
            claimed_ports.add(temp_port)
            used_ports[name] = temp_port
            logger.info(
                f"Assigning port {getattr(model, 'port', 'unknown')} to {name} "
                f"({getattr(model, 'model_path', 'unknown')})"
            )

    return models


def write_pid_metadata(name: str, pid: int, model) -> None:
    """Write a small JSON file describing a running model process.

    The metadata contains pid, model_path, port and a timestamp. These files
    are used by the CLI across invocations to discover running servers.
    """
    ensure_runtime_dirs()
    metadata = {
        "pid": pid,
        "name": name,
        "model_path": getattr(model, "model_path", ""),
        "port": getattr(model, "port", None),
        "started_at": time.time(),
    }
    pid_file(name).write_text(json.dumps(metadata))


def load_pid_metadata(name: str) -> dict | None:
    """Load and return the PID metadata dict for `name` or None if missing.

    Corrupt metadata files are removed and a warning is logged.
    """
    try:
        return json.loads(pid_file(name).read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        logger.warning(f"PID metadata for {name} is corrupt; removing.")
        with contextlib.suppress(FileNotFoundError):
            pid_file(name).unlink()
        return None


def discover_running_models() -> dict[str, dict]:
    """Return a mapping of model name -> PID metadata for discovered processes.

    This scans the `logs/pids` directory for JSON metadata files and returns a
    mapping suitable for status and stop actions.
    """
    ensure_runtime_dirs()
    running = {}
    for path in PID_DIR.glob("*.json"):
        name = path.stem
        meta = load_pid_metadata(name)
        if not meta:
            continue
        running[name] = meta
    return running


def process_alive(pid: int) -> bool:
    """Return True if a process with `pid` appears to exist.

    Uses `os.kill(pid, 0)` as a lightweight liveness probe. Permission errors
    are treated as the process existing.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def start_models(names: list[str] | None, detach: bool = True) -> None:
    """Start the requested models and optionally detach.

    Args:
        names: Optional list of model nicknames to start. If omitted, the
            registry's default models are used.
        detach: If True the CLI will detach and return immediately after
            launching child processes. Otherwise the parent process
            supervises child lifecycles.

    """
    refresh_model_registry()
    ensure_runtime_dirs()
    running = discover_running_models()
    reserved_ports: set[int] = set()
    for meta in running.values():
        pid = meta.get("pid")
        port = meta.get("port")
        if not port:
            continue
        if pid and not process_alive(pid):
            continue
        reserved_ports.add(port)
    try:
        requested_models = REGISTRY.build_model_map(names or None)
    except ModelRegistryError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc

    models = assign_ports(requested_models, reserved_ports=reserved_ports)
    processes: list[tuple[str, multiprocessing.Process]] = []
    started = 0

    for name, model in models.items():
        existing = running.get(name)
        if existing and process_alive(existing["pid"]):
            logger.warning(f"Model {name} already running with PID {existing['pid']}")
            continue

        logger.info(
            f"Starting MLX server {name} ({getattr(model, 'model_path', 'unknown')}) "
            f"on port {getattr(model, 'port', 'unknown')}"
        )
        proc = multiprocessing.Process(
            target=_process_worker,
            args=(name, model),
            name=f"server-{name}",
        )
        proc.start()
        write_pid_metadata(name, proc.pid, model)
        if detach:
            _detach_process(proc)
        else:
            processes.append((name, proc))
        started += 1

    if started == 0:
        logger.info("No models started.")
        return

    if detach:
        logger.info(f"Started {started} model(s) and detaching.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    supervise_processes(processes)


def _detach_process(proc: multiprocessing.Process) -> None:
    """Adjust multiprocessing internals to avoid auto-joining the child.

    This is a best-effort operation used when detaching; on some platforms
    it may be preferable to use OS-level daemonization instead.
    """
    with contextlib.suppress(Exception):
        mp_util._children.discard(proc)  # noqa: SLF001 - need to avoid auto-join on exit


def supervise_processes(processes: list[tuple[str, multiprocessing.Process]]) -> None:
    """Block and supervise the given list of (name, Process) tuples.

    Installs signal handlers to forward shutdown to children and ensures a
    graceful termination sequence.
    """
    names = ", ".join(name for name, _ in processes)
    logger.info(f"Supervising model processes: {names}")

    def _shutdown(signum, frame):  # noqa: ARG001
        logger.info(f"Shutdown signal received ({signum}); stopping models.")
        for name, proc in processes:
            if proc.is_alive():
                try:
                    os.kill(proc.pid, signal.SIGINT)
                except ProcessLookupError:
                    logger.info(f"Process for {name} already exited.")
        for name, proc in processes:
            proc.join(timeout=10)
            if proc.is_alive():
                logger.warning(f"Process {name} did not exit; terminating.")
                proc.terminate()
                proc.join(timeout=5)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        for _, proc in processes:
            proc.join()
    finally:
        for name, _ in processes:
            with contextlib.suppress(FileNotFoundError):
                pid_file(name).unlink()


def stop_models(names: list[str]) -> None:
    """Stop the named models, or all running models if `names` is empty.

    Sends SIGINT/SIGTERM (and SIGKILL if necessary) to child processes and
    removes their PID metadata files.
    """
    running = discover_running_models()
    target_names = names or list(running.keys())
    if not target_names:
        logger.info("No running models found.")
        return

    for name in target_names:
        meta = running.get(name)
        if not meta:
            logger.warning(f"Model {name} is not running.")
            continue
        pid = meta["pid"]
        if not process_alive(pid):
            logger.info(f"Model {name} had stale PID {pid}; cleaning up.")
            with contextlib.suppress(FileNotFoundError):
                pid_file(name).unlink()
            continue
        logger.info(f"Stopping model {name} (pid={pid})...")
        _terminate_process(pid)
        with contextlib.suppress(FileNotFoundError):
            pid_file(name).unlink()


def _terminate_process(pid: int) -> None:
    """Attempt to gracefully terminate a process, escalating to SIGKILL.

    Sends SIGINT then SIGTERM, waiting for the process to exit after each
    signal. If the process remains, a SIGKILL is sent when available.
    """
    for sig, timeout in (
        (signal.SIGINT, 5),
        (signal.SIGTERM, 5),
    ):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        _wait_for_exit(pid, timeout)
        if not process_alive(pid):
            return

    if hasattr(signal, "SIGKILL"):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        _wait_for_exit(pid, 5)


def _wait_for_exit(pid: int, timeout: int) -> None:
    """Block for up to `timeout` seconds waiting for `pid` to exit."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_alive(pid):
            return
        time.sleep(0.2)


def show_models() -> None:
    """Print configured models and their paths to the CLI logger."""
    refresh_model_registry()
    default_names = set(REGISTRY.default_names())
    for entry in REGISTRY.all_entries():
        label = " (default)" if entry.name in default_names else ""
        logger.info(f"{entry.name}{label} -> {getattr(entry.config, 'model_path', 'unknown')}")


def status_models() -> None:
    """Show runtime status for discovered model processes."""
    running = discover_running_models()
    if not running:
        logger.info("No models are currently running.")
        return

    for name, meta in running.items():
        pid = meta.get("pid")
        port = meta.get("port")
        alive = process_alive(pid) if pid else False
        state = "running" if alive else "stopped"
        logger.info(f"{name} -> {state} (pid={pid}, port={port})")
        if not alive:
            with contextlib.suppress(FileNotFoundError):
                pid_file(name).unlink()


def show_help(parser: argparse.ArgumentParser) -> None:
    """Print the main help text and detailed `start` subparser help."""
    parser.print_help()
    start_parser = getattr(parser, "start_parser", None)
    if start_parser:
        logger.info("")
        logger.info("Start command options:")
        start_parser.print_help()


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint; parse arguments and dispatch commands.

    The `argv` parameter is provided for easier testing; when None the
    interpreter's `sys.argv` is used.
    """
    configure_cli_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "start":
        start_models(args.names, detach=True)
    elif args.command == "stop":
        stop_models(args.names)
    elif args.command == "models":
        show_models()
    elif args.command == "status":
        status_models()
    elif args.command == "help":
        show_help(parser)
    else:  # pragma: no cover - argparse enforces
        parser.print_help()


if __name__ == "__main__":
    main()
