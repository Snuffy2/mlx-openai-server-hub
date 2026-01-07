"""CLI surface tests for mlx-openai-server-hub."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
import pytest

from mlx_openai_server_hub import main
from mlx_openai_server_hub.hub import config as hub_config_module


def test_status_command_falls_back_to_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure `status` prints config summary when daemon is unreachable.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for writing hub.yaml fixtures.
    monkeypatch : pytest.MonkeyPatch
        Pytest helper to override client behavior and port availability.

    Returns
    -------
    None
        Verifies CLI output includes the config-derived status summary.

    """

    monkeypatch.setattr(hub_config_module, "is_port_available", lambda host, port: True)

    def _raise_status(self: main.HubClient) -> None:
        raise main.HubClientError("down")

    monkeypatch.setattr(main.HubClient, "status", _raise_status)

    config_path = tmp_path / "hub.yaml"
    config_path.write_text(
        "host: 127.0.0.1\n"
        "port: 18080\n"
        f"log_path: {tmp_path / 'logs'}\n"
        "models:\n"
        "  - name: alpha\n"
        "    model_path: /models/a\n"
        "    port: 19001\n"
    )

    runner = CliRunner()
    result = runner.invoke(main.cli, ["--config", str(config_path), "status"])

    assert result.exit_code == 0
    assert "Hub daemon unreachable" in result.output
    assert "Hub binding: 127.0.0.1:18080" in result.output
