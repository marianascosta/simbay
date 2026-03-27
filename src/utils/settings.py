from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

_TRUE_VALUES = {"1", "true", "yes", "on"}
_VALID_BACKENDS = {"cpu", "mujoco-warp"}


def _read_bool(name: str, default: bool = False) -> bool:
    default_value = "1" if default else "0"
    return os.getenv(name, default_value).lower() in _TRUE_VALUES


def _read_backend() -> str:
    backend = os.getenv("SIMBAY_BACKEND", "cpu").lower()
    if backend not in _VALID_BACKENDS:
        raise SystemExit("Unsupported backend. Set SIMBAY_BACKEND to `cpu` or `mujoco-warp`.")
    return backend


RUN_ID = os.getenv("SIMBAY_RUN_ID") or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
HEADLESS = _read_bool("SIMBAY_HEADLESS", default=True)
BACKEND = _read_backend()
NUM_PARTICLES = int(os.getenv("SIMBAY_PARTICLES", "100"))
LOG_LEVEL_NAME = "INFO"
LOG_FALLBACK_DIR = Path("/tmp/simbay-logs")
SYSTEM_METRICS_INTERVAL_SECONDS = 1.0
OTEL_ENDPOINT = "http://tempo:4317"
