"""Tests for FastAPI server wiring around the hub runtime."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from mlx_openai_server_hub.hub.config import MLXHubConfig
from mlx_openai_server_hub.hub.runtime import HubRuntimeError
from mlx_openai_server_hub.hub.server import create_app


class StubRuntime:
    """Runtime stub capturing calls for assertions."""

    def __init__(self) -> None:
        """Initialize the stub runtime with default payload.

        Returns
        -------
        None
            Establishes the default call buffer and hub configuration.

        """

        self.calls: list[str] = []
        self.hub_config = MLXHubConfig(enable_status_page=True)

    def status_payload(self) -> dict[str, Any]:
        """Return a minimal status payload.

        Returns
        -------
        dict[str, Any]
            Serializable payload similar to the real runtime output.

        """

        return {
            "host": "127.0.0.1",
            "port": 8000,
            "model_starting_port": 47850,
            "enable_status_page": self.hub_config.enable_status_page,
            "log_level": self.hub_config.log_level,
            "models": [],
            "groups": [],
            "started_at": 0.0,
        }

    def reload_config(self) -> None:
        """Record reload requests.

        Returns
        -------
        None
            Appends a reload marker to the call log.

        """

        self.calls.append("reload")

    def request_shutdown(self) -> None:
        """Record shutdown requests.

        Returns
        -------
        None
            Appends a shutdown marker to the call log.

        """

        self.calls.append("shutdown")

    def start_model(self, name: str) -> None:
        """Record start-model requests.

        Parameters
        ----------
        name : str
            Model identifier captured for assertions.

        Returns
        -------
        None
            Appends a start marker to the call log.

        """

        self.calls.append(f"start:{name}")

    def stop_model(self, name: str) -> None:
        """Record stop-model requests.

        Parameters
        ----------
        name : str
            Model identifier captured for assertions.

        Returns
        -------
        None
            Appends a stop marker to the call log.

        """

        self.calls.append(f"stop:{name}")

    def load_model(self, name: str) -> None:
        """Record load-model requests.

        Parameters
        ----------
        name : str
            Model identifier captured for assertions.

        Returns
        -------
        None
            Appends a load marker to the call log.

        """

        self.calls.append(f"load:{name}")

    def unload_model(self, name: str) -> None:
        """Record unload-model requests.

        Parameters
        ----------
        name : str
            Model identifier captured for assertions.

        Returns
        -------
        None
            Appends an unload marker to the call log.

        """

        self.calls.append(f"unload:{name}")

    def stop_all_models(self) -> None:
        """Record stop-all requests.

        Returns
        -------
        None
            Appends a stop-all marker to the call log.

        """

        self.calls.append("stop-all")


def test_status_and_control_routes() -> None:
    """Verify core routes proxy to runtime and return payloads.

    Returns
    -------
    None
        Ensures HTTP routes exercise runtime control methods successfully.

    """

    runtime = StubRuntime()
    client = TestClient(create_app(runtime))

    response = client.get("/hub/status")
    assert response.status_code == 200
    assert response.json()["host"] == "127.0.0.1"

    response = client.post("/hub/reload")
    assert response.status_code == 200
    assert "reload" in runtime.calls

    client.post("/hub/models/demo/start")
    client.post("/hub/models/demo/stop")
    client.post("/hub/models/demo/load")
    client.post("/hub/models/demo/unload")
    client.post("/hub/models/stop-all")

    assert runtime.calls == [
        "reload",
        "start:demo",
        "stop:demo",
        "load:demo",
        "unload:demo",
        "stop-all",
    ]

    response = client.get("/hub/")
    assert response.status_code == 200


def test_error_handling_from_runtime() -> None:
    """Ensure runtime errors propagate as HTTP 400 responses.

    Returns
    -------
    None
        Confirms FastAPI surfaces HubRuntimeError instances as 400s.

    """

    class ExplodingRuntime(StubRuntime):
        def reload_config(self) -> None:  # type: ignore[override]
            """Raise a HubRuntimeError to simulate reload failure.

            Returns
            -------
            None
                Always raises an exception.

            Raises
            ------
            HubRuntimeError
                Emulates a runtime reload failure path.

            """

            raise HubRuntimeError("boom")

    runtime = ExplodingRuntime()
    client = TestClient(create_app(runtime))

    response = client.post("/hub/reload")
    assert response.status_code == 400
    assert "boom" in response.text
