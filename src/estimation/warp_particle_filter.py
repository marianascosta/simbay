"""
FrankaWarpEnv, a ParticleEnvironment backed by MuJoCo Warp.
"""

from __future__ import annotations

import logging
import os

import mujoco
import numpy as np

from src.utils import DEFAULT_OBJECT_PROPS
from src.utils import load_mujoco_model
from src.utils import modify_object_properties
from src.utils.mjx_utils import prepare_model_for_mjx

from .base import ParticleEnvironment
from .warp_batch import WarpBatch


class FrankaWarpEnv(ParticleEnvironment):
    """Batched MJWarp particle environment for mass estimation."""

    def __init__(
        self,
        limits: tuple[float, float],
        num_particles: int,
        nconmax: int = 128,
        njmax: int = 256,
    ):
        self.logger = logging.getLogger("simbay.warp_env")
        self.min, self.max = limits
        self._num_particles = num_particles
        self.std_dev = 0.005

        xml_path = os.path.join("assets", "franka_fr3_v2", "scene.xml")
        self._mj_model, self._mj_data = load_mujoco_model(xml_path)
        modify_object_properties(self._mj_model, self._mj_data, "object", DEFAULT_OBJECT_PROPS)
        prepare_model_for_mjx(self._mj_model)

        self.block_body_id = mujoco.mj_name2id(
            self._mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "object",
        )
        force_sensor_id = mujoco.mj_name2id(
            self._mj_model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "hand_force",
        )
        self.force_adr = int(self._mj_model.sensor_adr[force_sensor_id])

        self._nconmax = nconmax
        self._njmax = njmax
        self._batch: WarpBatch | None = None
        self._rng = np.random.default_rng(42)
        self._masses = np.empty((0,), dtype=np.float32)
        self._step_count = 0

    @property
    def num_particles(self) -> int:
        return self._num_particles

    def initialize_particles(self) -> np.ndarray:
        masses = self._rng.uniform(self.min, self.max, size=self._num_particles).astype(np.float32)
        self._batch = WarpBatch(
            self._mj_model,
            self._mj_data,
            masses,
            self.block_body_id,
            nconmax=self._nconmax,
            njmax=self._njmax,
        )
        self.logger.info("warp_runtime_warmup_start particles=%d", self._num_particles)
        self._batch.warmup()
        self.logger.info("warp_runtime_warmup_done")
        self._masses = masses.copy()
        self._step_count = 0
        return masses

    def propagate(self, particles: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        if self._batch is None:
            raise RuntimeError("Warp batch must be initialized before propagation.")

        noise = self._rng.standard_normal(self._num_particles).astype(np.float32) * self.std_dev
        next_particles = np.clip(
            np.asarray(particles, dtype=np.float32) + noise,
            self.min,
            self.max,
        ).astype(np.float32)
        self._batch.step(control_input, next_particles)
        self._masses = next_particles.copy()

        self._step_count += 1
        if self._step_count % 500 == 0:
            self.logger.info("warp_propagate step=%d", self._step_count)

        return next_particles

    def predict_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        if self._batch is None:
            raise RuntimeError("Warp batch must be initialized before rollout.")
        controls = np.asarray(trajectory, dtype=np.float32)
        if controls.size == 0:
            return self._masses.copy()

        noise = self._rng.standard_normal((controls.shape[0], self._num_particles)).astype(
            np.float32
        ) * self.std_dev
        mass_trajectory = np.clip(
            np.cumsum(noise, axis=0) + self._masses[np.newaxis, :],
            self.min,
            self.max,
        ).astype(np.float32)
        self._batch.rollout(controls, mass_trajectory)
        self._masses = mass_trajectory[-1].copy()

        self._step_count += int(controls.shape[0])
        if self._step_count and self._step_count % 500 == 0:
            self.logger.info("warp_rollout_complete step=%d", self._step_count)

        return self._masses.copy()

    def compute_likelihoods(self, particles: np.ndarray, observation: np.ndarray) -> np.ndarray:
        del particles
        if self._batch is None:
            raise RuntimeError("Warp batch must be initialized before likelihood evaluation.")
        sim_forces = self._batch.sensor_slice(self.force_adr, 3)
        diff = np.asarray(observation, dtype=np.float32) - sim_forces
        dist_sq = np.sum(diff**2, axis=1)
        return np.exp(-0.5 * dist_sq / 1.0)

    def resample_states(self, indexes: np.ndarray) -> None:
        if self._batch is None:
            raise RuntimeError("Warp batch must be initialized before resampling.")
        indexes_np = np.asarray(indexes, dtype=np.int32)
        self._batch.resample(indexes_np)
        self._masses = self._masses[indexes_np]

    def warmup_runtime(self, rollout_lengths) -> list[int]:
        if self._batch is None:
            raise RuntimeError("Warp batch must be initialized before warmup.")
        normalized_lengths = sorted({int(length) for length in rollout_lengths if int(length) > 0})
        for length in normalized_lengths:
            self._batch.warmup_rollout(length)
        self.logger.info(
            "warp_runtime_rollout_warmup_done particles=%d rollout_lengths=%s",
            self._num_particles,
            normalized_lengths,
        )
        return normalized_lengths

    def memory_profile(self) -> dict[str, int | float | str]:
        if self._batch is not None:
            return self._batch.memory_profile()
        return {
            "execution_platform": "uninitialized",
            "execution_device": "uninitialized",
            "default_jax_platform": "n/a",
            "default_jax_device": "n/a",
            "device_fallback_applied": False,
            "bytes_in_use": 0,
            "peak_bytes_in_use": 0,
            "bytes_limit": 0,
        }
