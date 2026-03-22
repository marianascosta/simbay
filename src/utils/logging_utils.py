import logging
import os
import resource
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self._run_id
        return True


def setup_logging(log_dir: str | Path = "logs", run_id: str = "unknown") -> logging.Logger:
    """
    Configure application logging for both stdout and a rotating file.
    """
    logger = logging.getLogger("simbay")
    if logger.handlers:
        return logger

    level_name = os.getenv("SIMBAY_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger.setLevel(level)
    logger.propagate = False

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s run_id=%(run_id)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    run_id_filter = _RunIdFilter(run_id)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(run_id_filter)

    file_handler = RotatingFileHandler(
        log_path / "simbay.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(run_id_filter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


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
