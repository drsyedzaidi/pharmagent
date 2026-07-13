"""Structured JSON logging — one line per record, no external deps.

Production logs go to stdout as JSON so they aggregate cleanly (Cloud Logging,
Loki, Datadog, etc.). Call ``configure_logging()`` once at startup.
"""
from __future__ import annotations

import json
import logging
import logging.config
import os

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render a LogRecord (plus any ``extra=`` fields) as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():        # structured extras
            if k not in _RESERVED and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def configure_logging(level: str | None = None) -> None:
    lvl = (level or os.environ.get("PHARMAGENT_LOG_LEVEL", "INFO")).upper()
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"json": {"()": "app.core.logging_config.JsonFormatter"}},
        "handlers": {"stdout": {"class": "logging.StreamHandler",
                                "formatter": "json", "stream": "ext://sys.stdout"}},
        "root": {"handlers": ["stdout"], "level": lvl},
        # uvicorn's access logger is replaced by our request middleware log.
        "loggers": {"uvicorn.access": {"handlers": ["stdout"], "level": "WARNING",
                                       "propagate": False}},
    })
