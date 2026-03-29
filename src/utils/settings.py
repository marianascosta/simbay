from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _read_bool(name: str, default: bool = False) -> bool:
    default_value = "1" if default else "0"
    return os.getenv(name, default_value).lower() in _TRUE_VALUES

RUN_ID = os.getenv("SIMBAY_RUN_ID") or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
HEADLESS = _read_bool("SIMBAY_HEADLESS", default=True)
BACKEND = os.getenv("SIMBAY_BACKEND", "cpu").lower()
ENVIRONMENT = os.getenv("SIMBAY_ENVIRONMENT", "dev").lower()
NUM_PARTICLES = int(os.getenv("SIMBAY_PARTICLES", "100"))
LOG_LEVEL_NAME = "INFO"
LOG_FALLBACK_DIR = Path("/tmp/simbay-logs")
SYSTEM_METRICS_INTERVAL_SECONDS = 1.0
OTEL_ENDPOINT = "http://tempo:4317"
OBSERVABILITY_ENABLED = ENVIRONMENT == "docker-dev"
PARTICLE_HISTORY_ENABLED = _read_bool("SIMBAY_PARTICLE_HISTORY", default=False)
WARP_LIKELIHOOD_DEBUG_ENABLED = _read_bool("SIMBAY_WARP_LIKELIHOOD_DEBUG", default=False)
PARTICLE_AUDIT_ENABLED = _read_bool("SIMBAY_PARTICLE_AUDIT", default=False)
PARTICLE_AUDIT_SAMPLE_EVERY = max(1, int(os.getenv("SIMBAY_PARTICLE_AUDIT_SAMPLE_EVERY", "10")))
PARTICLE_AUDIT_TOPK = max(1, int(os.getenv("SIMBAY_PARTICLE_AUDIT_TOPK", "8")))
PARTICLE_INVARIANT_CHECKS_ENABLED = _read_bool("SIMBAY_PARTICLE_INVARIANT_CHECKS", default=False)
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
