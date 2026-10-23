"""Optional local HTTP service and bundled standalone UI."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager
from importlib.resources import files

from .contracts import SCHEMA_VERSION, TERMINAL, Search
from .runtime import BusyError, SearchManager


def create_app(config, *, manager=None, token=None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, Response, StreamingResponse
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    # FastAPI resolves annotations in this module's globals.
    globals()["Request"] = Request
    runtime = manager or SearchManager(config)
    expected_token = token if token is not None else os.environ.get("GROUNDED_STREAM_TOKEN", "")

    @asynccontextmanager
    async def lifespan(app):
        runtime.start()
        try:
            yield
        finally:
            await asyncio.to_thread(runtime.close)

    app = FastAPI(
        title="Grounded Stream", version="0.2.0", lifespan=lifespan, docs_url=None, redoc_url=None
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.middleware("http")
    async def boundary(request, call_next):
        origin = request.headers.get("origin")
        expected_origin = f"{request.url.scheme}://{request.headers.get('host', '')}"
        if request.headers.get("sec-fetch-site") == "cross-site" or (
            origin and origin != expected_origin
        ):
            return Response("Cross-origin requests are not allowed.", status_code=403)
        if request.url.path.startswith("/v1") and expected_token:
            supplied = request.headers.get("authorization", "")
            if not secrets.compare_digest(
                supplied.encode("utf-8"), ("Bearer " + expected_token).encode("utf-8")
            ):
                return Response("Authentication required.", status_code=401)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return files("grounded_stream").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/static/{name}", include_in_schema=False)
    def asset(name: str):
        if name not in {"app.js", "style.css"}:
            raise HTTPException(404)
        media = "text/javascript" if name.endswith(".js") else "text/css"
        return Response(
            files("grounded_stream").joinpath("static", name).read_text(encoding="utf-8"),
            media_type=media,
        )

    @app.get("/v1/health")
    def health():
        return runtime.health()

    @app.get("/v1/capabilities")
    def capabilities():
        return {
            "schema_version": SCHEMA_VERSION,
            "detection": True,
            "continuous_search": True,
            "latest_snapshots": True,
            "detector_backend": config.detector.backend,
            "tracking": False,
            "segmentation": False,
            "speech": False,
            "history": False,
            "max_sources": 8,
            "max_targets": 8,
            "max_duration_seconds": 120,
            "max_pending": runtime.max_pending,
            "retention_seconds": runtime.retention_seconds,
            "sources": [s.public() for s in config.sources],
        }

    @app.post(
        "/v1/searches",
        status_code=202,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["targets", "source_ids"],
                            "properties": {
                                "targets": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 8,
                                    "items": {"type": "string", "minLength": 1, "maxLength": 120},
                                },
                                "source_ids": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 8,
                                    "items": {"type": "string"},
                                },
                                "duration_seconds": {
                                    "anyOf": [
                                        {"type": "number", "minimum": 1, "maximum": 120},
                                        {"type": "null"},
                                    ],
                                    "default": 10,
                                    "description": "Null runs until stopped or all sources end.",
                                },
                                "timeout_seconds": {
                                    "type": "number",
                                    "minimum": 5,
                                    "maximum": 600,
                                    "default": 180,
                                },
                            },
                        }
                    }
                },
            }
        },
    )
    async def search(request: Request):
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise HTTPException(415, "Use application/json.")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 8192:
                raise HTTPException(413, "Search request exceeds 8 KiB.")
        try:
            query = Search.parse(json.loads(body), {s.id for s in config.sources})
            return runtime.submit(query)
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except BusyError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "2"}) from exc

    @app.get("/v1/searches/{jid}")
    def status(jid: str, latest: bool = False):
        try:
            return runtime.snapshot(jid, latest_only=latest)
        except KeyError as exc:
            raise HTTPException(404, "Search expired or does not exist.") from exc

    @app.delete("/v1/searches/{jid}")
    def cancel(jid: str):
        try:
            return runtime.cancel(jid)
        except KeyError as exc:
            raise HTTPException(404, "Search expired or does not exist.") from exc

    @app.get("/v1/searches/{jid}/evidence/{sid}")
    def evidence(jid: str, sid: str, frame_id: str):
        try:
            jpeg, current_frame = runtime.evidence(jid, sid)
        except KeyError as exc:
            raise HTTPException(404, "Preview expired or is not available.") from exc
        # Never draw an old box over a newer frame from a changing preview endpoint.
        if current_frame != frame_id:
            raise HTTPException(409, "A newer frame is available. Refresh the search result.")
        return Response(jpeg, media_type="image/jpeg", headers={"X-Frame-ID": current_frame})

    @app.get("/v1/searches/{jid}/events")
    async def events(jid: str, request: Request):
        status(jid)

        async def stream():
            previous = None
            while not await request.is_disconnected():
                try:
                    snapshot = runtime.snapshot(jid)
                except KeyError:
                    yield "event: expired\ndata: {}\n\n"
                    return
                value = json.dumps(snapshot)
                if value != previous:
                    yield f"event: search\ndata: {value}\n\n"
                    previous = value
                else:
                    yield ": heartbeat\n\n"
                if snapshot["status"] in TERMINAL:
                    return
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    app.state.manager = runtime
    return app
