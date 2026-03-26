"""
State estimation module providing particle filter implementations
and model interfaces for robot tracking.
"""

from .base import ParticleEnvironment
from .mujoco_particle_filter import FrankaMuJoCoEnv
from .particle_filter import ParticleFilter

__all__ = [
    "ParticleEnvironment",
    "FrankaMuJoCoEnv",
    "ParticleFilter",
    "FrankaWarpEnv",
    "WarpParticleFilter",
]


def __getattr__(name: str):
    if name in {"WarpParticleFilter", "FrankaWarpEnv"}:
        from .warp_filter import WarpParticleFilter
        from .warp_particle_filter import FrankaWarpEnv

        exports = {
            "WarpParticleFilter": WarpParticleFilter,
            "FrankaWarpEnv": FrankaWarpEnv,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
