from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np

_TRUE_VALUES = {"1", "true", "yes", "on"}

_VALID_BACKENDS = {"mujoco", "mujoco-warp"}


def _read_bool(name: str, default: bool = False) -> bool:
    default_value = "1" if default else "0"
    return os.getenv(name, default_value).lower() in _TRUE_VALUES


RUN_ID = os.getenv("SIMBAY_RUN_ID") or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
HEADLESS = _read_bool("SIMBAY_HEADLESS", default=True)
_backend_value = os.getenv("SIMBAY_BACKEND", "mujoco").lower()
BACKEND = _backend_value if _backend_value in _VALID_BACKENDS else "mujoco"
NUM_PARTICLES = int(os.getenv("SIMBAY_PARTICLES", "100"))
WARP_MEASUREMENT_VARIANCE = max(
    float(os.getenv("SIMBAY_WARP_MEASUREMENT_VARIANCE", "0.1")),
    1.0e-6,
)
LOG_LEVEL_NAME = "INFO"
LOG_FALLBACK_DIR = Path("/tmp/simbay-logs")
SYSTEM_METRICS_INTERVAL_SECONDS = 1.0
OTEL_ENDPOINT = "http://tempo:4317"
MATPLOTLIB_BACKEND = "Agg"
os.environ.setdefault("MPLBACKEND", MATPLOTLIB_BACKEND)

DEFAULT_OBJECT_PROPS = {
    "type": "box",
    "pos": (0.3, 0.1, 0.022),
    "size": (0.02, 0.02, 0.02),
    "mass": 2.50,
    "friction": (5.0, 0.5, 0.001),
}

FRANKA_HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
