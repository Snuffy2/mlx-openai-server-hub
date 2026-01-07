# Implement Full Create-hub Parity

## Plan 

Rebuild mlx-openai-server-hub to match Create-hub CLI, config, orchestration, and status UI (same port/auth), aligned to JIT-and-Auto-Unload APIs; use config at ~/mlx-openai-server-hub/hub.yaml.

## Initial request

It turns out that the mlx-openai-server branch Create-hub (https://github.com/Snuffy2/mlx-openai-server/tree/Create-hub) is not going to be implemented. Thus, I'd like to move it to a separate app that works as a wrapper around mlx-openai-server and functions as close as possible to what the hub does in the Create hub branch. 

I had initially started the hub as a separate app called mlx-server-orch. I renamed this in GitHub to mlx-openai-server-hub (https://github.com/Snuffy2/mlx-openai-server-hub). Now, I'd like you to update mlx-openai-server-hub to function as close to the hub function in the Create hub branch as possible. It should use the same commands as `mlx-openai-server hub <command>` except it will be called as `mlx-openai-server-hub <command>`. The web status page should be the same with the same buttons/options.

mlx-openai-server-hub was never in use, so there doesn't need to be any legacy considerations.

When updating mlx-openai-server-hub, base it on this branch of mlx-openai-server that will eventually end up in production: https://github.com/Snuffy2/mlx-openai-server/tree/JIT-and-Auto-Unload

mlx-openai-server-hub should look for the configuration file in `~/mlx-openai-server-hub`. The configuration file should be called `hub.yaml` and use the same options that `hub.yaml` used in Create-hub

## Guidelines

* Follow AGENTS.md

* Update this prompt file as the plan progresses. Mark steps as complete and add any additional steps or changes as they are determined.

* There do not need to be any legacy or backward compatibility considerations for this plan.

* Once all steps are complete, mark the entire plan as complete.

## Steps

1. Catalog Create-hub specs: pull mlx-openai-server Create-hub branch to copy the hub Click group/subcommands, hub.yaml schema/defaults, and status UI routes/assets (port/auth). **(done – CLI commands, hub.yaml defaults, daemon endpoints, and status UI buttons collected)**

2. Map current code: review main.py, const.py, paths.py, model_registry.py to list gaps vs Create-hub commands, config, and UI. **(done – current hub uses argparse-only CLI, models.yaml registry, no web UI, no hub daemon)**

3. Align CLI/config: implement mlx-openai-server-hub <command> parity (start/reload/status/shutdown/load/unload/start-model/stop-model), read ~/mlx-openai-server-hub/hub.yaml, and mirror Create-hub defaults/flags including status UI enable/port/auth. **(done – Click CLI parity wired; hub.yaml loader with defaults/port allocation; daemon start uses FastAPI/uvicorn; status flag plumbed)**

4. Recreate orchestration: port allocation, model registry, start/stop/load/unload handlers, reload/shutdown flow, health/status endpoints, adapted to JIT-and-Auto-Unload server APIs and lifecycle (JIT load, auto-unload). **(done – runtime now tracks subprocess return codes, refreshes liveness in the monitor loop, exposes richer status payload with uptime/log paths, and keeps group caps/idle unload enforcement active)**

5. Rebuild status UI: add routes/templates/static JS mirroring Create-hub buttons/options and endpoints; bind to the same default port/auth behavior as Create-hub. **(done – status dashboard refreshed with gradient styling, auto-refresh toggle, group chips, actionable buttons, and live counts backed by the new status payload)**

6. Update docs: README usage for mlx-openai-server-hub, config reference for hub.yaml fields, and notes on JIT-and-Auto-Unload compatibility. **(done – README now documents hub.yaml defaults, CLI parity, status UI access, and JIT/auto-unload expectations)**

7. Build out integration tests: CLI command tests, config loading tests, orchestration behavior tests (start/stop/load/unload/reload/shutdown), and status UI endpoint tests. **(done – pytest covers hub config validation, runtime lifecycle with stubs, and FastAPI control/status routes)**

## Plan Status

All steps are complete; parity work is finished and verified via pytest.
