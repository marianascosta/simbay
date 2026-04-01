"""
FrankaWarpEnv, a ParticleEnvironment backed by MuJoCo Warp.
"""

from __future__ import annotations

import logging
import os

import mujoco
import numpy as np

from src.utils.settings import DEFAULT_OBJECT_PROPS
from src.utils.settings import WARP_MEASUREMENT_VARIANCE
from src.utils.mujoco_utils import load_mujoco_model
from src.utils.mujoco_utils import modify_object_properties
from src.utils.mujoco_utils import prepare_model_for_warp
from src.utils.profiling import annotate

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
        logging_data: dict[str, object] | None = None,
    ):
        self.logger = logging.getLogger("simbay.warp_env")
        self.logging_data = dict(logging_data or {})
        self.min, self.max = limits
        self._num_particles = num_particles
        self.std_dev = 0.005
        self._measurement_variance = float(WARP_MEASUREMENT_VARIANCE)

        xml_path = os.path.join("assets", "franka_fr3_v2", "scene.xml")
        self._mj_model, self._mj_data = load_mujoco_model(xml_path)
        modify_object_properties(
            self._mj_model, self._mj_data, "object", DEFAULT_OBJECT_PROPS
        )
        prepare_model_for_warp(self._mj_model)

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
        self._last_measurement_diagnostics: dict[str, float] = {}
        self._invalid_sensor_events = 0
        self._invalid_state_events = 0
        self._sensor_invalid_active = False
        self._state_invalid_active = False
        self._first_invalid_sensor_step: int | None = None
        self._first_invalid_state_step: int | None = None

    @property
    def num_particles(self) -> int:
        return self._num_particles

    def initialize_particles(self) -> np.ndarray:
        masses = self._rng.uniform(self.min, self.max, size=self._num_particles).astype(
            np.float32
        )
        self._batch = WarpBatch(
            self._mj_model,
            self._mj_data,
            masses,
            self.block_body_id,
            nconmax=self._nconmax,
            njmax=self._njmax,
            logging_data=self.logging_data,
        )
        self.logger.info(
            {
                **self.logging_data,
                "event": "warp_runtime_warmup_start",
                "msg": (
                    f"Started warming up the Warp runtime with {self._num_particles} particles."
                ),
                "measurement_variance": self._measurement_variance,
            }
        )
        self._batch.warmup()
        self.logger.info(
            {
                **self.logging_data,
                "event": "warp_runtime_warmup_done",
                "msg": "Finished warming up the Warp runtime.",
            }
        )
        self._masses = masses.copy()
        self._step_count = 0
        return masses

    def propagate(self, particles: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        if self._batch is None:
            raise RuntimeError("Warp batch must be initialized before propagation.")

        noise = (
            self._rng.standard_normal(self._num_particles).astype(np.float32)
            * self.std_dev
        )
        next_particles = np.clip(
            np.asarray(particles, dtype=np.float32) + noise,
            self.min,
            self.max,
        ).astype(np.float32)
        with annotate("warp_batch_step"):
            self._batch.step(control_input, next_particles)
        self._masses = next_particles.copy()

        self._step_count += 1

        return next_particles

    def predict_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        if self._batch is None:
            raise RuntimeError("Warp batch must be initialized before rollout.")
        controls = np.asarray(trajectory, dtype=np.float32)
        if controls.size == 0:
            return self._masses.copy()

        noise = (
            self._rng.standard_normal((controls.shape[0], self._num_particles)).astype(
                np.float32
            )
            * self.std_dev
        )
        mass_trajectory = np.clip(
            np.cumsum(noise, axis=0) + self._masses[np.newaxis, :],
            self.min,
            self.max,
        ).astype(np.float32)
        with annotate("warp_batch_rollout"):
            self._batch.rollout(controls, mass_trajectory)
        self._masses = mass_trajectory[-1].copy()

        self._step_count += int(controls.shape[0])

        return self._masses.copy()

    def compute_likelihoods(
        self, particles: np.ndarray, observation: np.ndarray
    ) -> np.ndarray:
        del particles
        if self._batch is None:
            raise RuntimeError(
                "Warp batch must be initialized before likelihood evaluation."
            )
        with annotate("warp_sensor_slice"):
            sim_forces = self._batch.sensor_slice(self.force_adr, 3)
        observation_np = np.asarray(observation, dtype=np.float32)
        sim_force_finite = np.isfinite(sim_forces)
        valid_force_particle = np.all(sim_force_finite, axis=1)

        diff = np.full_like(sim_forces, np.nan, dtype=np.float32)
        if np.any(valid_force_particle):
            diff[valid_force_particle] = (
                observation_np - sim_forces[valid_force_particle]
            )

        dist_sq = np.full((self._num_particles,), np.nan, dtype=np.float32)
        if np.any(valid_force_particle):
            dist_sq[valid_force_particle] = np.sum(
                diff[valid_force_particle] ** 2, axis=1
            )

        log_likelihoods = np.full(
            (self._num_particles,), -np.inf, dtype=np.float32
        )
        if np.any(valid_force_particle):
            log_likelihoods[valid_force_particle] = (
                -0.5 * dist_sq[valid_force_particle] / self._measurement_variance
            )
            finite_log_likelihoods = np.isfinite(log_likelihoods)
            log_likelihoods[finite_log_likelihoods] = (
                log_likelihoods[finite_log_likelihoods]
                - np.max(log_likelihoods[finite_log_likelihoods])
            )

        likelihoods = np.exp(log_likelihoods).astype(np.float32, copy=False)
        diff_finite = np.isfinite(diff)
        likelihood_finite = np.isfinite(likelihoods)
        sim_force_nonfinite_count = int(
            sim_forces.size - np.count_nonzero(sim_force_finite)
        )
        diff_nonfinite_count = int(diff.size - np.count_nonzero(diff_finite))
        likelihood_nonfinite_count = int(
            likelihoods.size - np.count_nonzero(likelihood_finite)
        )
        sensor_invalid_now = (
            sim_force_nonfinite_count > 0
            or diff_nonfinite_count > 0
            or likelihood_nonfinite_count > 0
        )
        sensor_invalid_transition = (
            sensor_invalid_now and not self._sensor_invalid_active
        )
        if sensor_invalid_transition:
            self._invalid_sensor_events += 1
            if self._first_invalid_sensor_step is None:
                self._first_invalid_sensor_step = self._step_count
        self._sensor_invalid_active = sensor_invalid_now
        state_invalid_now = False
        state_nonfinite_counts = {
            "qpos_nonfinite_count": 0,
            "qvel_nonfinite_count": 0,
            "sensordata_nonfinite_count": 0,
            "ctrl_nonfinite_count": 0,
        }
        if self._batch is not None:
            state_nonfinite_counts = self._batch.state_nonfinite_counts()
            state_invalid_now = any(
                value > 0 for value in state_nonfinite_counts.values()
            )
            state_invalid_transition = (
                state_invalid_now and not self._state_invalid_active
            )
            if state_invalid_transition:
                self._invalid_state_events += 1
                if self._first_invalid_state_step is None:
                    self._first_invalid_state_step = self._step_count
            self._state_invalid_active = state_invalid_now
        else:
            state_invalid_transition = False
        contact_counts = (
            self._batch.contact_counts()
            if self._batch is not None
            else np.zeros((0,), dtype=np.int32)
        )

        safe_sim_forces = np.where(sim_force_finite, sim_forces, 0.0)
        force_norms = np.linalg.norm(safe_sim_forces, axis=1)
        diff_norms = np.full((self._num_particles,), np.nan, dtype=np.float32)
        if np.any(valid_force_particle):
            diff_norms[valid_force_particle] = np.linalg.norm(
                diff[valid_force_particle], axis=1
            )
        sim_force_signal_particle_ratio = (
            float(np.mean(force_norms[valid_force_particle] > 1e-3))
            if np.any(valid_force_particle)
            else 0.0
        )
        contact_metric_available = float(contact_counts.size > 0)
        contact_force_mismatch = float(
            contact_counts.size > 0
            and float(np.max(contact_counts)) == 0.0
            and sim_force_signal_particle_ratio > 0.0
        )

        if sensor_invalid_transition:
            self.logger.warning(
                {
                    **self.logging_data,
                    "event": "warp_invalid_sensor_state",
                    "msg": (
                        "Detected non-finite values in the predicted force path for "
                        "this Warp batch step. This means one or more entries in "
                        "`sim_forces`, `diff`, or `likelihoods` were not finite."
                    ),
                    "step": self._step_count,
                    "valid_force_particle_ratio": float(
                        np.mean(valid_force_particle)
                    )
                    if valid_force_particle.size
                    else 0.0,
                    "invalid_force_particle_ratio": (
                        1.0 - float(np.mean(valid_force_particle))
                        if valid_force_particle.size
                        else 0.0
                    ),
                    "sim_force_nonfinite_count": sim_force_nonfinite_count,
                    "diff_nonfinite_count": diff_nonfinite_count,
                    "likelihood_nonfinite_count": likelihood_nonfinite_count,
                    "observation_norm": float(np.linalg.norm(observation_np)),
                    "sim_force_signal_particle_ratio": sim_force_signal_particle_ratio,
                    "contact_count_mean": (
                        float(np.mean(contact_counts)) if contact_counts.size else 0.0
                    ),
                    "contact_force_mismatch": contact_force_mismatch,
                }
            )
        if state_invalid_transition:
            self.logger.warning(
                {
                    **self.logging_data,
                    "event": "warp_invalid_backend_state",
                    "msg": (
                        "Detected non-finite values in the Warp backend state for "
                        "this batch step. This means one or more entries in "
                        "`qpos`, `qvel`, `sensordata`, or `ctrl` were not finite."
                    ),
                    "step": self._step_count,
                    "qpos_nonfinite_count": int(
                        state_nonfinite_counts["qpos_nonfinite_count"]
                    ),
                    "qvel_nonfinite_count": int(
                        state_nonfinite_counts["qvel_nonfinite_count"]
                    ),
                    "sensordata_nonfinite_count": int(
                        state_nonfinite_counts["sensordata_nonfinite_count"]
                    ),
                    "ctrl_nonfinite_count": int(
                        state_nonfinite_counts["ctrl_nonfinite_count"]
                    ),
                    "invalid_state_events_total": int(self._invalid_state_events),
                }
            )

        self._last_measurement_diagnostics = {
            "obs_fx": float(observation_np[0]),
            "obs_fy": float(observation_np[1]),
            "obs_fz": float(observation_np[2]),
            "obs_norm": float(np.linalg.norm(observation_np)),
            "sim_force_norm_min": float(np.nanmin(force_norms))
            if force_norms.size
            else 0.0,
            "sim_force_norm_max": float(np.nanmax(force_norms))
            if force_norms.size
            else 0.0,
            "sim_force_norm_mean": float(np.nanmean(force_norms))
            if force_norms.size
            else 0.0,
            "sim_force_axis_std_x": (
                float(np.std(sim_forces[valid_force_particle, 0]))
                if np.any(valid_force_particle)
                else 0.0
            ),
            "sim_force_axis_std_y": (
                float(np.std(sim_forces[valid_force_particle, 1]))
                if np.any(valid_force_particle)
                else 0.0
            ),
            "sim_force_axis_std_z": (
                float(np.std(sim_forces[valid_force_particle, 2]))
                if np.any(valid_force_particle)
                else 0.0
            ),
            "diff_norm_min": float(np.nanmin(diff_norms))
            if np.any(valid_force_particle)
            else 0.0,
            "diff_norm_max": float(np.nanmax(diff_norms))
            if np.any(valid_force_particle)
            else 0.0,
            "diff_norm_mean": float(np.nanmean(diff_norms))
            if np.any(valid_force_particle)
            else 0.0,
            "sim_force_finite_ratio": (
                float(np.mean(sim_force_finite)) if sim_force_finite.size else 0.0
            ),
            "diff_finite_ratio": (
                float(np.mean(diff_finite)) if diff_finite.size else 0.0
            ),
            "likelihood_finite_ratio": (
                float(np.mean(likelihood_finite)) if likelihood_finite.size else 0.0
            ),
            "valid_force_particle_ratio": (
                float(np.mean(valid_force_particle))
                if valid_force_particle.size
                else 0.0
            ),
            "invalid_force_particle_ratio": (
                1.0 - float(np.mean(valid_force_particle))
                if valid_force_particle.size
                else 0.0
            ),
            "all_particles_invalid": float(not np.any(valid_force_particle)),
            "sim_force_signal_particle_ratio": sim_force_signal_particle_ratio,
            "contact_count_mean": (
                float(np.mean(contact_counts)) if contact_counts.size else 0.0
            ),
            "contact_count_max": (
                float(np.max(contact_counts)) if contact_counts.size else 0.0
            ),
            "active_contact_particle_ratio": (
                float(np.mean(contact_counts > 0)) if contact_counts.size else 0.0
            ),
            "contact_metric_available": contact_metric_available,
            "contact_force_mismatch": contact_force_mismatch,
            "sim_force_nonfinite_count": float(sim_force_nonfinite_count),
            "diff_nonfinite_count": float(diff_nonfinite_count),
            "likelihood_nonfinite_count": float(likelihood_nonfinite_count),
            "invalid_sensor_events": float(self._invalid_sensor_events),
            "invalid_state_events": float(self._invalid_state_events),
            "first_invalid_sensor_step": float(
                self._first_invalid_sensor_step
                if self._first_invalid_sensor_step is not None
                else -1
            ),
            "first_invalid_state_step": float(
                self._first_invalid_state_step
                if self._first_invalid_state_step is not None
                else -1
            ),
            "qpos_nonfinite_count": float(
                state_nonfinite_counts["qpos_nonfinite_count"]
            ),
            "qvel_nonfinite_count": float(
                state_nonfinite_counts["qvel_nonfinite_count"]
            ),
            "sensordata_nonfinite_count": float(
                state_nonfinite_counts["sensordata_nonfinite_count"]
            ),
            "ctrl_nonfinite_count": float(
                state_nonfinite_counts["ctrl_nonfinite_count"]
            ),
            "repaired_world_count": 0.0,
            "likelihood_min": float(np.min(likelihoods)) if likelihoods.size else 0.0,
            "likelihood_max": float(np.max(likelihoods)) if likelihoods.size else 0.0,
            "likelihood_mean": float(np.mean(likelihoods)) if likelihoods.size else 0.0,
            "likelihood_std": float(np.std(likelihoods)) if likelihoods.size else 0.0,
            "likelihood_range": (
                float(np.max(likelihoods) - np.min(likelihoods))
                if likelihoods.size
                else 0.0
            ),
            "measurement_variance": float(self._measurement_variance),
        }

        return likelihoods

    def mean_particle_sensor_reads(self) -> np.ndarray:
        if self._batch is None:
            return np.zeros((3,), dtype=np.float32)
        sim_forces = self._batch.sensor_slice(self.force_adr, 3)
        return np.mean(sim_forces, axis=0, dtype=np.float32)

    def last_measurement_diagnostics(self) -> dict[str, float]:
        return self._last_measurement_diagnostics.copy()

    def resample_states(self, indexes: np.ndarray) -> None:
        if self._batch is None:
            raise RuntimeError("Warp batch must be initialized before resampling.")
        indexes_np = np.asarray(indexes, dtype=np.int32)
        self._batch.resample(indexes_np)
        self._masses = self._masses[indexes_np]

    def warmup_runtime(self, rollout_lengths) -> list[int]:
        if self._batch is None:
            raise RuntimeError("Warp batch must be initialized before warmup.")
        normalized_lengths = sorted(
            {int(length) for length in rollout_lengths if int(length) > 0}
        )
        for length in normalized_lengths:
            self._batch.warmup_rollout(length)
        self.logger.info(
            {
                **self.logging_data,
                "event": "warp_runtime_rollout_warmup_done",
                "msg": "Finished warming up Warp rollout execution.",
            }
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
            "state_bytes_estimate": 0,
        }
