"""HTTP client for interacting with the hub daemon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class HubClientError(RuntimeError):
    """Raised when a hub daemon request fails."""


@dataclass(slots=True)
class HubClient:
    """Lightweight synchronous HTTP client for hub control endpoints."""

    host: str
    port: int
    timeout: float = 10.0

    @property
    def base_url(self) -> str:
        """Return the fully qualified base URL for the hub daemon."""

        return f"http://{self.host}:{self.port}"

    def _request(self, method: str, path: str, json_body: dict[str, Any] | None = None) -> Any:
        """Execute an HTTP request against the daemon and return JSON."""

        url = f"{self.base_url}{path}"
        try:
            response = httpx.request(method, url, json=json_body, timeout=self.timeout)
        except httpx.RequestError as exc:  # pragma: no cover - network surface
            raise HubClientError(f"Failed to reach hub daemon at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            message = response.text.strip() or response.reason_phrase
            raise HubClientError(f"Hub daemon error {response.status_code}: {message}")

        if response.content:
            return response.json()
        return None

    def status(self) -> dict[str, Any]:
        """Fetch hub status information."""

        payload = self._request("GET", "/hub/status")
        if not isinstance(payload, dict):
            raise HubClientError("Unexpected response schema from /hub/status")
        return payload

    def reload(self) -> None:
        """Trigger a hub reload (config apply + model sync)."""

        self._request("POST", "/hub/reload")

    def shutdown(self) -> None:
        """Request hub shutdown."""

        self._request("POST", "/hub/shutdown")

    def start_model(self, name: str) -> None:
        """Start a specific model."""

        self._request("POST", f"/hub/models/{name}/start")

    def stop_model(self, name: str) -> None:
        """Stop a specific model."""

        self._request("POST", f"/hub/models/{name}/stop")

    def load_model(self, name: str) -> None:
        """Load a model into memory."""

        self._request("POST", f"/hub/models/{name}/load")

    def unload_model(self, name: str) -> None:
        """Unload a model from memory."""

        self._request("POST", f"/hub/models/{name}/unload")

    def stop_all_models(self) -> None:
        """Stop all managed models while keeping the daemon alive."""

        self._request("POST", "/hub/models/stop-all")
