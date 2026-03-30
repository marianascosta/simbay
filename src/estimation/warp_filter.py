import logging

import numpy as np

from src.utils.logging_utils import get_process_memory_bytes
from src.utils.profiling import annotate
from src.utils.settings import WARP_RESAMPLE_WARMUP_STEPS

from .warp_particle_filter import FrankaWarpEnv


def _normalize_weights(weights: np.ndarray, likelihoods: np.ndarray) -> np.ndarray:
    updated = weights * likelihoods
    updated = np.nan_to_num(
        updated,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    total = float(np.sum(updated))
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(weights, 1.0 / weights.shape[0])
    return updated / total


def _effective_sample_size(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(weights**2))


def _estimate_particles(particles: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(particles * weights, axis=0))


def _uniform_weight_metrics(weights: np.ndarray) -> tuple[float, float, bool]:
    if weights.size == 0:
        return 0.0, 0.0, False
    uniform = 1.0 / weights.shape[0]
    deviations = np.abs(weights - uniform)
    l1_distance = float(np.sum(deviations))
    max_deviation = float(np.max(deviations))
    return l1_distance, max_deviation, bool(max_deviation <= 1e-6)


def _systematic_resample(
    weights: np.ndarray,
    particles: np.ndarray,
    offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = weights.shape[0]
    positions = (np.arange(count, dtype=weights.dtype) + offset) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    indexes = np.searchsorted(cumulative, positions, side="right")
    resampled_particles = particles[indexes]
    uniform_weights = np.full_like(weights, 1.0 / count)
    return uniform_weights, resampled_particles, indexes


def _update_and_optionally_resample(
    weights: np.ndarray,
    particles: np.ndarray,
    likelihoods: np.ndarray,
    resample_offset: float,
    *,
    allow_resample: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    updated_weights = _normalize_weights(weights, likelihoods)
    effective_sample_size = _effective_sample_size(updated_weights)
    should_resample = allow_resample and effective_sample_size < (weights.shape[0] / 2.0)

    if should_resample:
        next_weights, next_particles, indexes = _systematic_resample(
            updated_weights,
            particles,
            resample_offset,
        )
        reported_effective_sample_size = float(weights.shape[0])
        return next_weights, next_particles, reported_effective_sample_size, indexes, True

    indexes = np.arange(weights.shape[0], dtype=np.int32)
    return updated_weights, particles, effective_sample_size, indexes, False


class WarpParticleFilter:
    """NumPy-backed particle filter for the Warp environment."""

    def __init__(self, env: FrankaWarpEnv, logging_data: dict[str, object] | None = None):
        self.logger = logging.getLogger("simbay.warp_particle_filter")
        self.env = env
        self.logging_data = dict(logging_data or {})

        init_memory_before = get_process_memory_bytes()
        self.particles = self.env.initialize_particles().astype(np.float32, copy=False)
        init_memory_after = get_process_memory_bytes()
        self.N = int(self.particles.shape[0])

        self.weights = np.full((self.N,), 1.0 / self.N, dtype=np.float64)
        self._rng = np.random.default_rng()
        self._effective_sample_size = float(self.N)
        self._step_index = 0
        self._resample_count = 0
        self._skipped_invalid_updates = 0
        self._resample_warmup_steps = int(WARP_RESAMPLE_WARMUP_STEPS)

        state_bytes_total = int(self.particles.nbytes + self.weights.nbytes)
        self.state_bytes_total = state_bytes_total
        self.state_bytes_per_particle = self.state_bytes_total / self.N if self.N else 0.0
        self.process_memory_per_particle_estimate = (
            max(init_memory_after - init_memory_before, 0) / self.N if self.N else 0.0
        )

        self.logger.info(
            {
                **self.logging_data,
                "event": "warp_particle_filter_initialized",
                "msg": f"Initialised the Warp particle filter with {self.N} particles.",
                "particles": self.N,
            }
        )

    def warmup_runtime(self, rollout_lengths: list[int]) -> list[int]:
        warmed_rollout_lengths = self.env.warmup_runtime(rollout_lengths)
        measurement_dim = int(getattr(self.env, "measurement_dim", 3))
        zero_observation = np.zeros((measurement_dim,), dtype=self.particles.dtype)
        likelihoods = self.env.compute_likelihoods(self.particles, zero_observation)
        weights = _normalize_weights(self.weights, likelihoods)
        effective_sample_size = _effective_sample_size(weights)
        estimate = _estimate_particles(self.particles, weights)
        _ = (effective_sample_size, estimate)
        self.logger.info(
            {
                **self.logging_data,
                "event": "warp_filter_runtime_warmup_done",
                "msg": "Finished warming up the Warp particle filter runtime.",
            }
        )
        return warmed_rollout_lengths

    def set_contact_mode(self, *, simplified_contacts: bool) -> None:
        self.env.set_contact_mode(simplified_contacts=simplified_contacts)

    def predict(self, control_input) -> None:
        self.particles = self.env.propagate(self.particles, control_input)

    def predict_trajectory(self, trajectory) -> None:
        self.particles = self.env.predict_trajectory(trajectory)

    def predict_mass_only(self, steps: int) -> None:
        self.particles = self.env.propagate_masses_only(self.particles, steps)

    def sync_with_robot_state(self, robot) -> None:
        qpos = np.asarray(robot.data.qpos, dtype=np.float32)
        qvel = np.asarray(robot.data.qvel, dtype=np.float32)
        ctrl = np.asarray(robot.data.ctrl, dtype=np.float32)
        self.env.sync_world_state(qpos=qpos, qvel=qvel, ctrl=ctrl)

    @staticmethod
    def _measurement_is_informative(diagnostics: dict[str, float]) -> bool:
        likelihood_std = float(np.nan_to_num(diagnostics.get("likelihood_std", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
        likelihood_range = float(np.nan_to_num(diagnostics.get("likelihood_range", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
        sim_force_std_x = float(np.nan_to_num(diagnostics.get("sim_force_axis_std_x", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
        sim_force_std_y = float(np.nan_to_num(diagnostics.get("sim_force_axis_std_y", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
        sim_force_std_z = float(np.nan_to_num(diagnostics.get("sim_force_axis_std_z", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
        return (
            likelihood_std > 1e-3
            or likelihood_range > 1e-3
            or sim_force_std_x > 1e-3
            or sim_force_std_y > 1e-3
            or sim_force_std_z > 1e-3
        )

    def update(self, observation) -> None:
        likelihoods = self.env.compute_likelihoods(self.particles, observation)
        self.weights = _normalize_weights(self.weights, likelihoods)
        self._effective_sample_size = _effective_sample_size(self.weights)
        self._step_index += 1

    def resample(self) -> None:
        if self._effective_sample_size >= self.N / 2:
            return

        offset = float(self._rng.uniform())
        self.weights, self.particles, indexes = _systematic_resample(
            self.weights,
            self.particles,
            offset,
        )
        self.env.resample_states(indexes)
        self._effective_sample_size = float(self.N)
        self._resample_count += 1

    def step(self, control_input, observation) -> dict[str, float | bool]:
        with annotate("warp_pf_propagate"):
            self.particles = self.env.propagate(self.particles, control_input)
        with annotate("warp_pf_likelihood"):
            likelihoods = self.env.compute_likelihoods(self.particles, observation)
        diagnostics = self.env.last_measurement_diagnostics()
        measurement_informative = self._measurement_is_informative(diagnostics)
        allow_resample = self._step_index >= self._resample_warmup_steps

        offset = float(self._rng.uniform())
        (
            self.weights,
            self.particles,
            self._effective_sample_size,
            indexes,
            did_resample,
        ) = _update_and_optionally_resample(
            self.weights,
            self.particles,
            likelihoods,
            offset,
            allow_resample=allow_resample,
        )
        if did_resample:
            with annotate("warp_pf_resample_states"):
                self.env.resample_states(indexes)
            self._resample_count += 1
        uniform_weight_l1, uniform_weight_max_dev, collapsed_to_uniform = _uniform_weight_metrics(self.weights)
        current_invalid_measurement = (
            diagnostics.get("sim_force_nonfinite_count", 0.0) > 0.0
            or diagnostics.get("diff_nonfinite_count", 0.0) > 0.0
            or diagnostics.get("likelihood_nonfinite_count", 0.0) > 0.0
            or diagnostics.get("qpos_nonfinite_count", 0.0) > 0.0
            or diagnostics.get("qvel_nonfinite_count", 0.0) > 0.0
            or diagnostics.get("sensordata_nonfinite_count", 0.0) > 0.0
            or diagnostics.get("ctrl_nonfinite_count", 0.0) > 0.0
        )
        if collapsed_to_uniform and (not did_resample) and current_invalid_measurement:
            self.logger.warning(
                {
                    **self.logging_data,
                    "event": "warp_weight_update_uninformative",
                    "msg": (f"Detected an uninformative Warp weight update at step " f"{self._step_index}."),
                    "step": self._step_index,
                }
            )
        self._step_index += 1
        return {
            "effective_sample_size": float(self._effective_sample_size),
            "resampled": did_resample,
            "resample_count": self._resample_count,
            "uniform_weight_l1_distance": uniform_weight_l1,
            "uniform_weight_max_deviation": uniform_weight_max_dev,
            "collapsed_to_uniform": collapsed_to_uniform,
            "diagnostics": diagnostics,
            "skipped_invalid_update": False,
            "skipped_invalid_updates": self._skipped_invalid_updates,
            "bootstrap_attempts": 1,
            "uninformative_update": not measurement_informative,
            "resample_warmup_active": not allow_resample,
        }

    def bootstrap_first_update(
        self,
        control_input,
        observation,
        *,
        max_attempts: int = 3,
    ) -> dict[str, float | bool]:
        del max_attempts
        result = self.step(control_input, observation)
        result["bootstrap_attempts"] = 1
        return result

    def effective_sample_size(self) -> float:
        return float(self._effective_sample_size)

    def estimate(self) -> float:
        return _estimate_particles(self.particles, self.weights)

    def particles_host(self):
        return self.particles.copy()

    def memory_profile(self) -> dict[str, float | int]:
        return {
            "particles": self.N,
            "state_bytes_total": self.state_bytes_total,
            "state_bytes_per_particle": self.state_bytes_per_particle,
            "process_memory_per_particle_estimate_bytes": self.process_memory_per_particle_estimate,
        }
