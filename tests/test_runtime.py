"""Runtime lifecycle tests for the hub daemon."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from mlx_openai_server_hub.hub import config as hub_config_module
from mlx_openai_server_hub.hub.runtime import HubRuntime


class FakeProcess:
    """Minimal process stand-in used to avoid spawning real workers."""

    def __init__(self) -> None:
        """Initialize the fake process with default metadata.

        Returns
        -------
        None
            Constructor sets initial pid and return code placeholders.

        """

        self.pid = 12345
        self.returncode: int | None = None

    def poll(self) -> int | None:
        """Return the current return code.

        Returns
        -------
        int | None
            The current return code value.

        """

        return self.returncode

    def terminate(self) -> None:
        """Simulate graceful shutdown.

        Returns
        -------
        None
            Updates the return code to a zero exit status.

        """

        self.returncode = 0

    def kill(self) -> None:
        """Simulate forced shutdown.

        Returns
        -------
        None
            Updates the return code to a non-zero exit status.

        """

        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int | None:
        """Return the return code once set.

        Parameters
        ----------
        timeout : float | None
            Ignored timeout parameter to mirror subprocess interface.

        Returns
        -------
        int | None
            The return code after termination or kill.

        """

        return self.returncode


@pytest.fixture
def hub_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Prepare a hub.yaml with two models and patched port availability.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest for writing hub.yaml fixtures.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch helper to bypass port availability checks.

    Returns
    -------
    Path
        Path to the generated hub.yaml file.

    """

    monkeypatch.setattr(hub_config_module, "is_port_available", lambda host, port: True)

    config_path = tmp_path / "hub.yaml"
    config_path.write_text(
        f"host: 127.0.0.1\n"
        f"port: 19080\n"
        f"log_path: {tmp_path / 'logs'}\n"
        "models:\n"
        "  - name: alpha\n"
        "    model_path: /models/a\n"
        "    port: 19081\n"
        "  - name: beta\n"
        "    model_path: /models/b\n"
        "    port: 19082\n"
        "    jit_enabled: true\n"
        "    group: runners\n"
        "groups:\n"
        "  - name: runners\n"
        "    max_loaded: 1\n"
        "    idle_unload_trigger_min: 1\n"
    )

    return config_path


def test_start_stop_and_status(monkeypatch: pytest.MonkeyPatch, hub_config_path: Path) -> None:
    """Exercise start/stop/load/unload flows without spawning real servers.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest helper to patch subprocess and runtime methods.
    hub_config_path : Path
        Path to the generated hub.yaml fixture.

    Returns
    -------
    None
        This test validates status transitions without real processes.

    """

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    runtime = HubRuntime.from_path(hub_config_path)
    monkeypatch.setattr(runtime, "_build_launch_command", lambda cfg: ["python", "--version"])
    monkeypatch.setattr(runtime, "_wait_for_health", lambda state: True)

    runtime.start_model("alpha")
    alpha_payload = {model["name"]: model for model in runtime.status_payload()["models"]}
    assert alpha_payload["alpha"]["status"] == "running"

    runtime.stop_model("alpha")
    alpha_payload = {model["name"]: model for model in runtime.status_payload()["models"]}
    assert alpha_payload["alpha"]["status"] == "stopped"

    runtime.load_model("beta")
    beta_payload = {model["name"]: model for model in runtime.status_payload()["models"]}
    assert beta_payload["beta"]["status"] == "running"

    runtime.unload_model("beta")
    beta_payload = {model["name"]: model for model in runtime.status_payload()["models"]}
    assert beta_payload["beta"]["status"] == "stopped"

    payload = runtime.status_payload()
    assert payload["host"] == "127.0.0.1"
    assert len(payload["models"]) == 2
    assert {model["name"] for model in payload["models"]} == {"alpha", "beta"}

    runtime.request_shutdown()
