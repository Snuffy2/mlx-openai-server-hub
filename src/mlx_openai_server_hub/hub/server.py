"""FastAPI application for the hub daemon."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from mlx_openai_server_hub.hub.runtime import HubRuntime, HubRuntimeError


def create_app(runtime: HubRuntime) -> FastAPI:
    """Construct the FastAPI app bound to a ``HubRuntime`` instance."""

    app = FastAPI(title="mlx-openai-server-hub", version="0.1.0")

    if runtime.hub_config.enable_status_page:
        templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

        @app.get("/hub/", response_class=HTMLResponse)
        async def status_page(request: Request) -> HTMLResponse:
            return templates.TemplateResponse(request, "status.html.jinja")

    @app.get("/hub/status")
    async def get_status() -> dict[str, object]:
        return runtime.status_payload()

    @app.post("/hub/reload")
    async def reload_hub() -> dict[str, object]:
        try:
            runtime.reload_config()
        except HubRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return runtime.status_payload()

    @app.post("/hub/shutdown")
    async def shutdown_hub() -> dict[str, str]:
        runtime.request_shutdown()
        return {"detail": "shutdown requested"}

    @app.post("/hub/models/{name}/start")
    async def start_model(name: str) -> dict[str, str]:
        try:
            runtime.start_model(name)
        except HubRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"detail": f"start requested for '{name}'"}

    @app.post("/hub/models/{name}/stop")
    async def stop_model(name: str) -> dict[str, str]:
        try:
            runtime.stop_model(name)
        except HubRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"detail": f"stop requested for '{name}'"}

    @app.post("/hub/models/{name}/load")
    async def load_model(name: str) -> dict[str, str]:
        try:
            runtime.load_model(name)
        except HubRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"detail": f"load requested for '{name}'"}

    @app.post("/hub/models/{name}/unload")
    async def unload_model(name: str) -> dict[str, str]:
        try:
            runtime.unload_model(name)
        except HubRuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"detail": f"unload requested for '{name}'"}

    @app.post("/hub/models/stop-all")
    async def stop_all_models() -> dict[str, str]:
        runtime.stop_all_models()
        return {"detail": "stop requested for all models"}

    return app
