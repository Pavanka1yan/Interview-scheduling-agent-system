"""Logging utilities with request ID support."""

from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestIDFilter(logging.Filter):
    """Attach request ID from context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        record.request_id = request_id_ctx.get()
        return True


class JsonFormatter(logging.Formatter):
    """Format logs as JSON."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        log_record = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if getattr(record, "request_id", None):
            log_record["request_id"] = record.request_id
        return json.dumps(log_record)


def init_logging(level: str) -> None:
    """Initialize application logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIDFilter())
    logging.basicConfig(level=level, handlers=[handler])


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that manages request IDs."""

    async def dispatch(
        self, request: Request, call_next
    ):  # type: ignore[override]
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        token = request_id_ctx.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        request_id_ctx.reset(token)
        return response
