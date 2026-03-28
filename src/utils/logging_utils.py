import json
import logging
import resource
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from .settings import LOG_FALLBACK_DIR
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


def setup_logging(log_dir: str | Path = "logs", run_id: str = "unknown") -> logging.Logger:
    """
    Configure application logging for both stdout and a rotating file.
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

    preferred_log_path = Path(log_dir)
    fallback_log_path = Path(LOG_FALLBACK_DIR)
    fallback_log_path = fallback_log_path / str(run_id)
    file_handler: RotatingFileHandler | None = None
    file_target: Path | None = None

    for candidate in (preferred_log_path, fallback_log_path):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            file_target = candidate / "simbay.log"
            file_handler = RotatingFileHandler(
                file_target,
                maxBytes=5 * 1024 * 1024,
                backupCount=5,
            )
            break
        except OSError:
            file_handler = None
            file_target = None

    if file_handler is None or file_target is None:
        raise RuntimeError(
            "Failed to initialize file logging. "
            f"Tried {preferred_log_path / 'simbay.log'} and {fallback_log_path / 'simbay.log'}."
        )

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(run_id_filter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    if file_target.parent != preferred_log_path:
        logger.warning(
            {
                "msg": "Primary logs directory is not writable; using fallback log path.",
                "requested_log_dir": str(preferred_log_path),
                "fallback_log_file": str(file_target),
            }
        )
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
