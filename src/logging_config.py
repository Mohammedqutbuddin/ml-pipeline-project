"""
Structured (JSON) logging setup. Real log aggregation systems (Datadog,
CloudWatch, ELK) expect JSON lines, not free-text print statements — this
gives every service in the pipeline (training, serving, monitoring) a
consistent, machine-parseable log format.

Usage:
    from src.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("model_loaded", extra={"model_version": "3"})
"""
import json
import logging
import sys
from datetime import datetime, timezone

_RESERVED = set(logging.LogRecord(
    "", 0, "", 0, "", (), None
).__dict__.keys())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # include any structured fields passed via `extra=`
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != "message":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
