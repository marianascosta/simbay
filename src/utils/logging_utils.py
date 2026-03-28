import json
import logging
import resource
import sys
from typing import Any

from .settings import LOG_LEVEL_NAME

_LOG_KEY_ORDER = (
    "run_id",
    "msg",
    "backend",
    "stage",
    "substage",
    "step",
    "steps",
    "timestamp",
    "level",
    "logger",
)

logger = logging.getLogger("simbay")


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self._run_id
        return True


class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, dict):
            payload = dict(record.msg)
        else:
            payload = {"msg": record.getMessage()}

        if "event" in payload and "msg" not in payload:
            payload["msg"] = payload.pop("event")

        payload.setdefault("run_id", getattr(record, "run_id", "unknown"))
        payload.setdefault("level", record.levelname)
        payload.setdefault("logger", record.name)
        payload.setdefault("timestamp", self.formatTime(record, self.datefmt))
        ordered_payload: dict[str, Any] = {}
        for key in _LOG_KEY_ORDER:
            if key in payload:
                ordered_payload[key] = payload[key]
        for key in sorted(payload):
            if key not in ordered_payload:
                ordered_payload[key] = payload[key]
        return json.dumps(ordered_payload, default=str)


def setup_logging(run_id: str = "unknown") -> logging.Logger:
    """
    Configure application logging to stdout.
    """
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    level = getattr(logging, LOG_LEVEL_NAME, logging.INFO)

    logger.setLevel(level)
    logger.propagate = False

    formatter = _StructuredFormatter(
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    run_id_filter = _RunIdFilter(run_id)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(run_id_filter)

    logger.addHandler(stream_handler)
    return logger


def extend_logging_data(logging_data: dict[str, Any], **updates: Any) -> dict[str, Any]:
    merged = dict(logging_data)
    merged.update(updates)
    return merged


def get_process_memory_bytes() -> int:
    """
    Return the current process resident-set-size approximation from `resource`.

    On macOS `ru_maxrss` is reported in bytes. On Linux it is reported in KiB.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage * 1024)


def format_bytes(num_bytes: float) -> str:
    """
    Render a byte count in human-readable units.
    """
    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes:.2f} B"
