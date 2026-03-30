from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _read_bool(name: str, default: bool = False) -> bool:
    default_value = "1" if default else "0"
    return os.getenv(name, default_value).lower() in _TRUE_VALUES


def _read_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return float(default)


RUN_ID = os.getenv("SIMBAY_RUN_ID") or datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
HEADLESS = _read_bool("SIMBAY_HEADLESS", default=True)
BACKEND = os.getenv("SIMBAY_BACKEND", "cpu").lower()
ENVIRONMENT = os.getenv("SIMBAY_ENVIRONMENT", "dev").lower()
NUM_PARTICLES = int(os.getenv("SIMBAY_PARTICLES", "100"))
LOG_LEVEL_NAME = "INFO"
LOG_FALLBACK_DIR = Path("/tmp/simbay-logs")
SYSTEM_METRICS_INTERVAL_SECONDS = 1.0
OTEL_ENDPOINT = os.getenv("SIMBAY_OTEL_ENDPOINT", "http://tempo:4317").strip() or "http://tempo:4317"
OBSERVABILITY_ENABLED = _read_bool(
    "SIMBAY_OBSERVABILITY_ENABLED",
    default=ENVIRONMENT == "docker-dev",
)
PARTICLE_HISTORY_ENABLED = _read_bool("SIMBAY_PARTICLE_HISTORY", default=False)
WARP_LIKELIHOOD_DEBUG_ENABLED = _read_bool("SIMBAY_WARP_LIKELIHOOD_DEBUG", default=False)
_WARP_LIKELIHOOD_SPACE_RAW = os.getenv("SIMBAY_WARP_LIKELIHOOD_SPACE", "wrench").strip().lower()
WARP_LIKELIHOOD_SPACE = _WARP_LIKELIHOOD_SPACE_RAW if _WARP_LIKELIHOOD_SPACE_RAW in {"force", "wrench"} else "wrench"
WARP_MEASUREMENT_VARIANCE = max(_read_float("SIMBAY_WARP_MEASUREMENT_VARIANCE", 1.0), 1e-6)
WARP_FORCE_MEASUREMENT_VARIANCE = max(
    _read_float("SIMBAY_WARP_FORCE_MEASUREMENT_VARIANCE", WARP_MEASUREMENT_VARIANCE),
    1e-6,
)
WARP_TORQUE_MEASUREMENT_VARIANCE = max(
    _read_float("SIMBAY_WARP_TORQUE_MEASUREMENT_VARIANCE", WARP_MEASUREMENT_VARIANCE),
    1e-6,
)
try:
    WARP_RESAMPLE_WARMUP_STEPS = max(int(os.getenv("SIMBAY_WARP_RESAMPLE_WARMUP_STEPS", "0")), 0)
except ValueError:
    WARP_RESAMPLE_WARMUP_STEPS = 0
WARP_SET_CONST_ON_MASS_UPDATE = _read_bool("SIMBAY_WARP_SET_CONST_ON_MASS_UPDATE", default=False)
try:
    WARP_MAX_REPAIRED_WORLD_FRACTION_FOR_UPDATE = float(
        os.getenv("SIMBAY_WARP_MAX_REPAIRED_WORLD_FRACTION_FOR_UPDATE", "0.25")
    )
except ValueError:
    WARP_MAX_REPAIRED_WORLD_FRACTION_FOR_UPDATE = 0.25
WARP_MAX_REPAIRED_WORLD_FRACTION_FOR_UPDATE = min(
    max(WARP_MAX_REPAIRED_WORLD_FRACTION_FOR_UPDATE, 0.0),
    1.0,
)
_WARP_MEASUREMENT_CONTACT_MODE_RAW = os.getenv("SIMBAY_WARP_MEASUREMENT_CONTACT_MODE", "full").strip().lower()
WARP_MEASUREMENT_CONTACT_MODE = (
    _WARP_MEASUREMENT_CONTACT_MODE_RAW
    if _WARP_MEASUREMENT_CONTACT_MODE_RAW in {"simplified", "full"}
    else "full"
)
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
