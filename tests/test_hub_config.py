"""Tests for hub configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_openai_server_hub.hub import config as hub_config_module
from mlx_openai_server_hub.hub.config import HubConfigError, load_hub_config


def test_load_hub_config_applies_defaults_and_persisted_ports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure defaults, persisted ports, and group wiring apply correctly.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest for writing hub.yaml fixtures.
    monkeypatch : pytest.MonkeyPatch
        Pytest helper to override port availability checks.

    Returns
    -------
    None
        This test asserts the loaded configuration matches expectations.

    """

    config_path = tmp_path / "hub.yaml"
    config_path.write_text(
        "host: 127.0.0.1\n"
        "port: 18080\n"
        "model_starting_port: 19000\n"
        f"log_path: {tmp_path / 'logs'}\n"
        "models:\n"
        "  - name: alpha\n"
        "    model_path: /models/a\n"
        "    port: 19001\n"
        "  - name: beta\n"
        "    model_path: /models/b\n"
        "    jit_enabled: true\n"
        "    group: runners\n"
        "groups:\n"
        "  - name: runners\n"
        "    max_loaded: 1\n"
    )

    monkeypatch.setattr(hub_config_module, "is_port_available", lambda host, port: True)

    hub = load_hub_config(config_path, persisted_ports={"beta": 19005})

    assert hub.host == "127.0.0.1"
    assert hub.port == 18080
    assert hub.model_starting_port == 19000
    assert hub.enable_status_page is True
    assert hub.log_path == Path(tmp_path / "logs")

    model_map = {model.name: model for model in hub.models}
    assert model_map["alpha"].port == 19001
    assert model_map["alpha"].log_file
    assert model_map["beta"].port == 19005
    assert model_map["beta"].jit_enabled is True
    assert model_map["beta"].group == "runners"

    assert hub.groups[0].name == "runners"
    assert hub.groups[0].max_loaded == 1


def test_idle_unload_group_requires_jit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate idle_unload_trigger_min requires JIT-capable members.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest for writing hub.yaml fixtures.
    monkeypatch : pytest.MonkeyPatch
        Pytest helper to override port availability checks.

    Returns
    -------
    None
        This test expects a HubConfigError to be raised for invalid groups.

    """

    config_path = tmp_path / "hub.yaml"
    config_path.write_text(
        "models:\n"
        "  - name: alpha\n"
        "    model_path: /models/a\n"
        "    port: 19501\n"
        "    group: slow\n"
        "groups:\n"
        "  - name: slow\n"
        "    max_loaded: 2\n"
        "    idle_unload_trigger_min: 5\n"
        "    \n"
    )

    monkeypatch.setattr(hub_config_module, "is_port_available", lambda host, port: True)

    with pytest.raises(HubConfigError):
        load_hub_config(config_path)
