import logging
import os
import sys
from ctypes import Structure
from ctypes import WinDLL
from ctypes import byref
from ctypes import c_size_t
from ctypes import sizeof
from ctypes import windll
from ctypes.wintypes import DWORD
from ctypes.wintypes import HANDLE
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None


def setup_logging(log_dir: str | Path = "logs") -> logging.Logger:
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
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_path / "simbay.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def get_process_memory_bytes() -> int:
    """
    Return the current process resident-set-size approximation.

    On macOS `ru_maxrss` is reported in bytes. On Linux it is reported in KiB.
    """
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(usage)
        return int(usage * 1024)

    if sys.platform == "win32":
        return _get_process_memory_bytes_windows()

    return 0


class PROCESS_MEMORY_COUNTERS(Structure):
    _fields_ = [
        ("cb", DWORD),
        ("PageFaultCount", DWORD),
        ("PeakWorkingSetSize", c_size_t),
        ("WorkingSetSize", c_size_t),
        ("QuotaPeakPagedPoolUsage", c_size_t),
        ("QuotaPagedPoolUsage", c_size_t),
        ("QuotaPeakNonPagedPoolUsage", c_size_t),
        ("QuotaNonPagedPoolUsage", c_size_t),
        ("PagefileUsage", c_size_t),
        ("PeakPagefileUsage", c_size_t),
    ]


def _get_process_memory_bytes_windows() -> int:
    psapi = WinDLL("Psapi.dll")
    kernel32 = windll.kernel32

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = sizeof(PROCESS_MEMORY_COUNTERS)
    process: HANDLE = kernel32.GetCurrentProcess()

    success = psapi.GetProcessMemoryInfo(
        process,
        byref(counters),
        counters.cb,
    )
    if not success:
        return 0

    return int(counters.WorkingSetSize)


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
