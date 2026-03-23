import logging

import numpy as np

from src.utils.logging_utils import get_process_memory_bytes

from .warp_particle_filter import FrankaWarpEnv


def _normalize_weights(weights: np.ndarray, likelihoods: np.ndarray) -> np.ndarray:
    tiny = np.finfo(weights.dtype).tiny
    safe_weights = np.maximum(weights, tiny)
    safe_likelihoods = np.nan_to_num(
        likelihoods,
        nan=tiny,
        posinf=1.0,
        neginf=tiny,
    )
    safe_likelihoods = np.maximum(safe_likelihoods, tiny)

    log_updated = np.log(safe_weights) + np.log(safe_likelihoods)
    log_updated = log_updated - np.max(log_updated)
    updated = np.exp(log_updated)
    total = max(float(np.sum(updated)), tiny)
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
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
    updated_weights = _normalize_weights(weights, likelihoods)
    ess = _effective_sample_size(updated_weights)
    should_resample = ess < (weights.shape[0] / 2.0)

    if should_resample:
        next_weights, next_particles, indexes = _systematic_resample(
            updated_weights,
            particles,
            resample_offset,
        )
        reported_ess = float(weights.shape[0])
        return next_weights, next_particles, reported_ess, indexes, True

    indexes = np.arange(weights.shape[0], dtype=np.int32)
    return updated_weights, particles, ess, indexes, False


class WarpParticleFilter:
    """NumPy-backed particle filter for the Warp environment."""

    def __init__(self, env: FrankaWarpEnv):
        self.logger = logging.getLogger("simbay.warp_particle_filter")
        self.env = env

        init_memory_before = get_process_memory_bytes()
        self.particles = self.env.initialize_particles().astype(np.float32, copy=False)
        init_memory_after = get_process_memory_bytes()
        self.N = int(self.particles.shape[0])

        self.weights = np.full((self.N,), 1.0 / self.N, dtype=self.particles.dtype)
        self._rng = np.random.default_rng(7)
        self._ess = float(self.N)
        self._step_index = 0
        self._resample_count = 0
        self._skipped_invalid_updates = 0
        self._last_good_particles = self.particles.copy()
        self._last_good_weights = self.weights.copy()
        self._last_good_ess = self._ess

        state_bytes_total = int(self.particles.nbytes + self.weights.nbytes)
        self.state_bytes_total = state_bytes_total
        self.state_bytes_per_particle = self.state_bytes_total / self.N if self.N else 0.0
        self.process_memory_per_particle_estimate = (
            max(init_memory_after - init_memory_before, 0) / self.N if self.N else 0.0
        )

        self.logger.info(
            "warp_particle_filter_initialized particles=%d state_bytes_total=%d "
            "state_bytes_per_particle=%.2f process_memory_per_particle_estimate_bytes=%.2f",
            self.N,
            self.state_bytes_total,
            self.state_bytes_per_particle,
            self.process_memory_per_particle_estimate,
        )

    def warmup_runtime(self, rollout_lengths: list[int]) -> list[int]:
        warmed_rollout_lengths = self.env.warmup_runtime(rollout_lengths)
        zero_observation = np.zeros((3,), dtype=self.particles.dtype)
        likelihoods = self.env.compute_likelihoods(self.particles, zero_observation)
        weights = _normalize_weights(self.weights, likelihoods)
        ess = _effective_sample_size(weights)
        estimate = _estimate_particles(self.particles, weights)

        offset = float(self._rng.uniform())
        update_resample = _update_and_optionally_resample(
            self.weights,
            self.particles,
            likelihoods,
            offset,
        )
        indexes = np.arange(self.N, dtype=np.int32)
        self.env.resample_states(indexes)

        _ = (weights, ess, estimate, update_resample)
        self.logger.info(
            "warp_filter_runtime_warmup_done particles=%d rollout_lengths=%s",
            self.N,
            warmed_rollout_lengths,
        )
        return warmed_rollout_lengths

    def predict(self, control_input) -> None:
        self.particles = self.env.propagate(self.particles, control_input)

    def predict_trajectory(self, trajectory) -> None:
        self.particles = self.env.predict_trajectory(trajectory)

    @staticmethod
    def _measurement_is_valid(diagnostics: dict[str, float]) -> bool:
        return (
            diagnostics.get("likelihood_finite_ratio", 0.0) >= 1.0
            and diagnostics.get("sim_force_nonfinite_count", 0.0) == 0.0
            and diagnostics.get("diff_nonfinite_count", 0.0) == 0.0
            and diagnostics.get("likelihood_nonfinite_count", 0.0) == 0.0
            and diagnostics.get("qpos_nonfinite_count", 0.0) == 0.0
            and diagnostics.get("qvel_nonfinite_count", 0.0) == 0.0
            and diagnostics.get("sensordata_nonfinite_count", 0.0) == 0.0
            and diagnostics.get("ctrl_nonfinite_count", 0.0) == 0.0
        )

    @staticmethod
    def _measurement_is_informative(diagnostics: dict[str, float]) -> bool:
        return (
            diagnostics.get("likelihood_std", 0.0) > 1e-3
            or diagnostics.get("likelihood_range", 0.0) > 1e-3
            or diagnostics.get("sim_force_axis_std_x", 0.0) > 1e-3
            or diagnostics.get("sim_force_axis_std_y", 0.0) > 1e-3
            or diagnostics.get("sim_force_axis_std_z", 0.0) > 1e-3
        )

    def _save_last_good_snapshot(self) -> None:
        diagnostics = self.env.last_measurement_diagnostics()
        if not self._measurement_is_valid(diagnostics):
            return
        if diagnostics.get("repaired_world_count", 0.0) > 0.0:
            return
        self._last_good_particles = self.particles.copy()
        self._last_good_weights = self.weights.copy()
        self._last_good_ess = float(self._ess)
        self.env.capture_recovery_snapshot()

    def _skip_invalid_update(
        self,
        diagnostics: dict[str, float],
        attempt: int = 1,
    ) -> dict[str, float | bool]:
        restored = self.env.restore_recovery_snapshot()
        if restored:
            self.particles = self._last_good_particles.copy()
            self.weights = self._last_good_weights.copy()
            self._ess = float(self._last_good_ess)
        self._skipped_invalid_updates += 1
        self.logger.warning(
            "warp_invalid_update_skipped step=%d attempt=%d restored=%s ess=%.2f "
            "likelihood_finite_ratio=%.6f sim_force_nonfinite=%d diff_nonfinite=%d "
            "likelihood_nonfinite=%d qpos_nonfinite=%d qvel_nonfinite=%d "
            "sensordata_nonfinite=%d ctrl_nonfinite=%d",
            self._step_index,
            attempt,
            restored,
            float(self._ess),
            diagnostics.get("likelihood_finite_ratio", 0.0),
            int(diagnostics.get("sim_force_nonfinite_count", 0.0)),
            int(diagnostics.get("diff_nonfinite_count", 0.0)),
            int(diagnostics.get("likelihood_nonfinite_count", 0.0)),
            int(diagnostics.get("qpos_nonfinite_count", 0.0)),
            int(diagnostics.get("qvel_nonfinite_count", 0.0)),
            int(diagnostics.get("sensordata_nonfinite_count", 0.0)),
            int(diagnostics.get("ctrl_nonfinite_count", 0.0)),
        )
        uniform_weight_l1, uniform_weight_max_dev, collapsed_to_uniform = _uniform_weight_metrics(
            self.weights
        )
        result = {
            "ess": float(self._ess),
            "resampled": False,
            "resample_count": self._resample_count,
            "uniform_weight_l1_distance": uniform_weight_l1,
            "uniform_weight_max_deviation": uniform_weight_max_dev,
            "collapsed_to_uniform": collapsed_to_uniform,
            "diagnostics": diagnostics,
            "skipped_invalid_update": True,
            "skipped_invalid_updates": self._skipped_invalid_updates,
            "bootstrap_attempts": attempt,
            "uninformative_update": False,
        }
        self._step_index += 1
        return result

    def _skip_uninformative_update(
        self,
        diagnostics: dict[str, float],
    ) -> dict[str, float | bool]:
        restored = self.env.restore_recovery_snapshot()
        if restored:
            self.particles = self._last_good_particles.copy()
            self.weights = self._last_good_weights.copy()
            self._ess = float(self._last_good_ess)
        self.logger.warning(
            "warp_uninformative_update_skipped step=%d restored=%s ess=%.2f "
            "likelihood_std=%.6e likelihood_range=%.6e "
            "sim_force_axis_std=(%.6e,%.6e,%.6e)",
            self._step_index,
            restored,
            float(self._ess),
            diagnostics.get("likelihood_std", 0.0),
            diagnostics.get("likelihood_range", 0.0),
            diagnostics.get("sim_force_axis_std_x", 0.0),
            diagnostics.get("sim_force_axis_std_y", 0.0),
            diagnostics.get("sim_force_axis_std_z", 0.0),
        )
        uniform_weight_l1, uniform_weight_max_dev, collapsed_to_uniform = _uniform_weight_metrics(
            self.weights
        )
        result = {
            "ess": float(self._ess),
            "resampled": False,
            "resample_count": self._resample_count,
            "uniform_weight_l1_distance": uniform_weight_l1,
            "uniform_weight_max_deviation": uniform_weight_max_dev,
            "collapsed_to_uniform": collapsed_to_uniform,
            "diagnostics": diagnostics,
            "skipped_invalid_update": False,
            "skipped_invalid_updates": self._skipped_invalid_updates,
            "bootstrap_attempts": 1,
            "uninformative_update": True,
        }
        self._step_index += 1
        return result

    def update(self, observation) -> None:
        likelihoods = self.env.compute_likelihoods(self.particles, observation)
        diagnostics = self.env.last_measurement_diagnostics()
        if self._measurement_is_valid(diagnostics) and self._measurement_is_informative(diagnostics):
            self.weights = _normalize_weights(self.weights, likelihoods)
            self._save_last_good_snapshot()
        self._ess = _effective_sample_size(self.weights)
        self._step_index += 1

    def resample(self) -> None:
        if self._ess >= self.N / 2:
            return

        offset = float(self._rng.uniform())
        self.weights, self.particles, indexes = _systematic_resample(
            self.weights,
            self.particles,
            offset,
        )
        self.env.resample_states(indexes)
        self._ess = float(self.N)
        self._resample_count += 1

    def step(self, control_input, observation) -> dict[str, float | bool]:
        self.particles = self.env.propagate(self.particles, control_input)
        likelihoods = self.env.compute_likelihoods(self.particles, observation)
        diagnostics = self.env.last_measurement_diagnostics()
        if not self._measurement_is_valid(diagnostics):
            return self._skip_invalid_update(diagnostics)
        if not self._measurement_is_informative(diagnostics):
            return self._skip_uninformative_update(diagnostics)

        offset = float(self._rng.uniform())
        (
            self.weights,
            self.particles,
            self._ess,
            indexes,
            did_resample,
        ) = _update_and_optionally_resample(
            self.weights,
            self.particles,
            likelihoods,
            offset,
        )
        if did_resample:
            self.env.resample_states(indexes)
            self._resample_count += 1
        self._save_last_good_snapshot()
        uniform_weight_l1, uniform_weight_max_dev, collapsed_to_uniform = _uniform_weight_metrics(
            self.weights
        )
        if collapsed_to_uniform and (
            diagnostics.get("invalid_sensor_events", 0.0) > 0.0
            or diagnostics.get("invalid_state_events", 0.0) > 0.0
            or diagnostics.get("likelihood_finite_ratio", 1.0) < 1.0
        ):
            self.logger.warning(
                "warp_weight_update_uninformative step=%d ess=%.2f "
                "likelihood_finite_ratio=%.6f sim_force_nonfinite=%d "
                "diff_nonfinite=%d likelihood_nonfinite=%d "
                "first_invalid_sensor_step=%d first_invalid_state_step=%d",
                self._step_index,
                float(self._ess),
                diagnostics.get("likelihood_finite_ratio", 1.0),
                int(diagnostics.get("sim_force_nonfinite_count", 0.0)),
                int(diagnostics.get("diff_nonfinite_count", 0.0)),
                int(diagnostics.get("likelihood_nonfinite_count", 0.0)),
                int(diagnostics.get("first_invalid_sensor_step", -1.0)),
                int(diagnostics.get("first_invalid_state_step", -1.0)),
            )
        should_log_diag = self._step_index < 5 or self._step_index % 100 == 0
        if should_log_diag:
            self.logger.info(
                "warp_measurement_diagnostics step=%d resampled=%s "
                "mass_min=%.6f mass_max=%.6f mass_mean=%.6f "
                "obs=(%.6f,%.6f,%.6f) obs_norm=%.6f "
                "sim_force_norm_min=%.6f sim_force_norm_max=%.6f sim_force_norm_mean=%.6f "
                "sim_force_axis_std=(%.6f,%.6f,%.6f) "
                "diff_norm_min=%.6f diff_norm_max=%.6f diff_norm_mean=%.6f "
                "likelihood_min=%.6e likelihood_max=%.6e likelihood_mean=%.6e likelihood_std=%.6e",
                self._step_index,
                did_resample,
                float(np.min(self.particles)),
                float(np.max(self.particles)),
                float(np.mean(self.particles)),
                diagnostics.get("obs_fx", 0.0),
                diagnostics.get("obs_fy", 0.0),
                diagnostics.get("obs_fz", 0.0),
                diagnostics.get("obs_norm", 0.0),
                diagnostics.get("sim_force_norm_min", 0.0),
                diagnostics.get("sim_force_norm_max", 0.0),
                diagnostics.get("sim_force_norm_mean", 0.0),
                diagnostics.get("sim_force_axis_std_x", 0.0),
                diagnostics.get("sim_force_axis_std_y", 0.0),
                diagnostics.get("sim_force_axis_std_z", 0.0),
                diagnostics.get("diff_norm_min", 0.0),
                diagnostics.get("diff_norm_max", 0.0),
                diagnostics.get("diff_norm_mean", 0.0),
                diagnostics.get("likelihood_min", 0.0),
                diagnostics.get("likelihood_max", 0.0),
                diagnostics.get("likelihood_mean", 0.0),
                diagnostics.get("likelihood_std", 0.0),
            )
        self._step_index += 1
        return {
            "ess": float(self._ess),
            "resampled": did_resample,
            "resample_count": self._resample_count,
            "uniform_weight_l1_distance": uniform_weight_l1,
            "uniform_weight_max_deviation": uniform_weight_max_dev,
            "collapsed_to_uniform": collapsed_to_uniform,
            "diagnostics": diagnostics,
            "skipped_invalid_update": False,
            "skipped_invalid_updates": self._skipped_invalid_updates,
            "bootstrap_attempts": 1,
            "uninformative_update": False,
        }

    def bootstrap_first_update(
        self,
        control_input,
        observation,
        *,
        max_attempts: int = 3,
    ) -> dict[str, float | bool]:
        last_result: dict[str, float | bool] | None = None
        for attempt in range(1, max_attempts + 1):
            result = self.step(control_input, observation)
            result["bootstrap_attempts"] = attempt
            if not bool(result.get("skipped_invalid_update", False)):
                if attempt > 1:
                    self.logger.info(
                        "warp_first_update_recovered attempts=%d step=%d",
                        attempt,
                        self._step_index - 1,
                    )
                return result
            last_result = result
        self.logger.error(
            "warp_first_update_failed attempts=%d",
            max_attempts,
        )
        return last_result if last_result is not None else self.step(control_input, observation)

    def effective_sample_size(self) -> float:
        return float(self._ess)

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
