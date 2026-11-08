"""OpenAI-compatible gateway in front of Ollama.

Adds what a raw Ollama port does not have:
API keys, per-key rate limiting, a model allowlist, JSON logs and metrics.
Any OpenAI client (Python SDK, Open WebUI, n8n, LangChain...) can use it
by pointing base_url to http://<host>:8080/v1.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .auth import check_key, extract_bearer, key_id
from .config import Settings
from .metrics import Metrics
from .ratelimit import RateLimiter

log = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    limiter = RateLimiter(settings.rate_limit_per_minute)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            base_url=settings.upstream_url,
            timeout=settings.upstream_timeout,
            transport=transport,
        )
        yield
        await app.state.client.aclose()

    app = FastAPI(title="Local LLM gateway", version="1.0.0", lifespan=lifespan)

    def authenticate(authorization: str | None) -> str:
        matched = check_key(extract_bearer(authorization), settings.api_key_hashes)
        if not matched:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        kid = key_id(matched)
        if not limiter.allow(kid):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(limiter.retry_after(kid))},
            )
        return kid

    def write_log(**fields) -> None:
        log.info(json.dumps({"ts": round(time.time(), 3), **fields}, ensure_ascii=False))

    @app.get("/health")
    async def health(request: Request):
        try:
            r = await request.app.state.client.get("/api/tags")
            r.raise_for_status()
            models = [m.get("name") for m in r.json().get("models", [])]
            return {"status": "ok", "upstream": "ok", "models": models}
        except httpx.HTTPError as exc:
            return JSONResponse({"status": "degraded", "upstream": str(exc)}, status_code=503)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        return metrics.render()

    @app.get("/v1/models")
    async def list_models(request: Request, authorization: str | None = Header(default=None)):
        authenticate(authorization)
        r = await request.app.state.client.get("/v1/models")
        data = r.json()
        if settings.allowed_models:
            data["data"] = [m for m in data.get("data", []) if m.get("id") in settings.allowed_models]
        return data

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
        kid = authenticate(authorization)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Body must be JSON")
        model = str(body.get("model", ""))
        if not model or not isinstance(body.get("messages"), list):
            raise HTTPException(status_code=400, detail="'model' and 'messages' are required")
        if settings.allowed_models and model not in settings.allowed_models:
            raise HTTPException(status_code=403, detail=f"Model '{model}' is not allowed")

        client: httpx.AsyncClient = request.app.state.client
        started = time.perf_counter()
        extra = {"messages": body["messages"]} if settings.log_prompts else {}

        if body.get("stream"):
            upstream_req = client.build_request("POST", "/v1/chat/completions", json=body)
            upstream = await client.send(upstream_req, stream=True)

            async def relay():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await upstream.aclose()
                    latency = time.perf_counter() - started
                    metrics.record(kid, model, upstream.status_code, latency)
                    write_log(key=kid, model=model, status=upstream.status_code,
                              latency_s=round(latency, 3), stream=True, **extra)

            return StreamingResponse(relay(), status_code=upstream.status_code,
                                     media_type=upstream.headers.get("content-type", "text/event-stream"))

        try:
            upstream = await client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            metrics.record(kid, model, 502, time.perf_counter() - started)
            raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")

        latency = time.perf_counter() - started
        data = upstream.json()
        usage = data.get("usage") or {}
        metrics.record(kid, model, upstream.status_code, latency,
                       usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        write_log(key=kid, model=model, status=upstream.status_code, latency_s=round(latency, 3),
                  prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                  **({**extra, "answer": data} if settings.log_prompts else {}))
        return JSONResponse(data, status_code=upstream.status_code)

    return app


app = create_app()
