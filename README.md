# MLX OpenAI Server Hub

`mlx-openai-server-hub` mirrors the Create-hub workflow from `mlx-openai-server` while wrapping the JIT-and-Auto-Unload branch. It reads `~/mlx-openai-server-hub/hub.yaml`, starts a FastAPI controller, launches model workers, and serves the status dashboard at `/hub/`.

## Quick start

```bash
git clone https://github.com/Snuffy2/mlx-openai-server-hub.git
cd mlx-openai-server-hub
python -m venv .venv
./.venv/bin/python -m pip install -e .

# create your hub.yaml in ~/mlx-openai-server-hub/hub.yaml
cp models.yaml-example ~/mlx-openai-server-hub/hub.yaml

# launch the hub daemon and non-JIT models
./.venv/bin/python -m mlx_openai_server_hub.main start

# inspect status (CLI)
./.venv/bin/python -m mlx_openai_server_hub.main status
# or open http://127.0.0.1:8000/hub/ if status pages are enabled
```

## hub.yaml reference

Hub-level settings live at `~/mlx-openai-server-hub/hub.yaml` (override with `--config`). Defaults mirror the Create-hub branch:

| Field | Default | Description |
| --- | --- | --- |
| `host` | `0.0.0.0` | Bind host for the controller and status page |
| `port` | `8000` | Controller port |
| `model_starting_port` | `47850` | Starting port used when models omit an explicit port |
| `log_level` | `INFO` | Log level passed to model workers |
| `log_path` | `~/mlx-openai-server-hub/logs` | Root directory for hub and model logs |
| `enable_status_page` | `true` | Serve the status dashboard at `/hub/` |
| `models` | _(required)_ | List of model entries (see below) |
| `groups` | `[]` | Optional group constraints for JIT/auto-unload |

### Model entries

Each model entry is validated against `MLXServerConfig` from `mlx-openai-server`. Key options:

| Field | Required | Default | Notes |
| --- | :---: | --- | --- |
| `name` | ✓ | — | Slug used for CLI and status output |
| `model_path` | ✓ | — | HF model id or local path |
| `port` |  | auto-allocated | Reserved from `model_starting_port` if omitted (persisted across reloads) |
| `host` |  | hub `host` | Worker bind host |
| `jit_enabled` |  | `false` | When true the worker stays off until explicitly loaded and can auto-unload when idle |
| `group` |  | — | Optional group slug used for max-loaded/idle-unload rules |
| `log_file` |  | `<log_path>/<name>.log` | Auto-filled unless `no_log_file` is true |
| `model_type`, `context_length`, `max_concurrency`, `queue_timeout`, `queue_size`, `quantize`, `config_name`, `lora_paths`, `lora_scales`, `enable_auto_tool_choice`, `tool_call_parser`, `reasoning_parser`, `message_converter`, `trust_remote_code`, `chat_template_file`, `disable_auto_resize`, `debug`, `no_log_file` |  | varies | Passed through to `mlx-openai-server launch` |

### Groups

Groups bound to JIT-enabled models let you cap concurrency and unload idle workers:

| Field | Description |
| --- | --- |
| `name` | Slug matching member model `group` values |
| `max_loaded` | Maximum simultaneously running members; oldest running model is unloaded to make room |
| `idle_unload_trigger_min` | Minutes of inactivity before auto-unloading; requires all members to set `jit_enabled: true` |

## CLI commands

All commands accept `--config` to point at an alternate `hub.yaml`.

| Command | Purpose |
| --- | --- |
| `start` | Start the FastAPI hub daemon and all non-JIT models |
| `reload` | Re-read `hub.yaml` and reconcile running workers |
| `shutdown` | Stop all workers and request daemon shutdown |
| `status` | Show hub binding, model state, and group summaries |
| `start-model <name>` | Start a specific worker |
| `stop-model <name>` | Stop a specific worker |
| `load-model <name>` | Load a JIT-enabled worker |
| `unload-model <name>` | Unload a JIT-enabled worker |
| `stop` | Stop all managed workers but keep the daemon alive |
| `watch` | Repeatedly poll `/hub/status` (CLI client) |

## Status page

When `enable_status_page` is true the dashboard is available at `http://<host>:<port>/hub/`. It exposes start/stop/load/unload controls, group chips, and live counters backed by `/hub/status`.

## Development

- Keep the virtual environment in `.venv` and run tools with `./.venv/bin/python ...`.
- Format/lint with `pre-commit run --all-files` (ruff) and type-check with `./.venv/bin/python -m mypy src/` when available.
- Run tests: `./.venv/bin/python -m pytest tests/`.
