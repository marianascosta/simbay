"""
FrankaWarpEnv, a ParticleEnvironment backed by MuJoCo Warp.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

from src.utils.settings import DEFAULT_OBJECT_PROPS
from src.utils.settings import WARP_FORCE_MEASUREMENT_VARIANCE
from src.utils.settings import WARP_LIKELIHOOD_DEBUG_ENABLED
from src.utils.settings import WARP_LIKELIHOOD_SPACE
from src.utils.settings import WARP_LIKELIHOOD_TEMPERATURE
from src.utils.settings import WARP_TORQUE_MEASUREMENT_VARIANCE
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
        njmax: int = 384,
        logging_data: dict[str, object] | None = None,
    ):
        self.logger = logging.getLogger("simbay.warp_env")
        self.logging_data = dict(logging_data or {})
        self.min, self.max = limits
        self._num_particles = num_particles
        self.std_dev = 0.005
        self._mj_model = None
        self._mj_data = None
        self.block_body_id = -1
        self.force_adr = -1
        self.torque_adr = -1
        self._nconmax = nconmax
        self._njmax = njmax
        self._batch: WarpBatch | None = None
        # Keep full-contact mode by default to match the CPU path dynamics.
        self._simplified_contacts_enabled = False
        self._rng = np.random.default_rng()
        self._masses = np.empty((0,), dtype=np.float32)
        self._step_count = 0
        self._last_measurement_diagnostics: dict[str, float] = {}
        self._invalid_sensor_events = 0
        self._invalid_state_events = 0
        self._sensor_invalid_active = False
        self._state_invalid_active = False
        self._first_invalid_sensor_step: int | None = None
        self._first_invalid_state_step: int | None = None
        self._likelihood_space = WARP_LIKELIHOOD_SPACE
        self._measurement_dim = 6 if self._likelihood_space == "wrench" else 3
        self._likelihood_temperature = max(float(WARP_LIKELIHOOD_TEMPERATURE), 1e-6)
        self._force_measurement_variance = max(float(WARP_FORCE_MEASUREMENT_VARIANCE), 1e-6)
        self._torque_measurement_variance = max(float(WARP_TORQUE_MEASUREMENT_VARIANCE), 1e-6)
        if self._measurement_dim == 6:
            self._measurement_variance_diag = np.array(
                [
                    self._force_measurement_variance,
                    self._force_measurement_variance,
                    self._force_measurement_variance,
                    self._torque_measurement_variance,
                    self._torque_measurement_variance,
                    self._torque_measurement_variance,
                ],
                dtype=np.float64,
            )
        else:
            self._measurement_variance_diag = np.array(
                [
                    self._force_measurement_variance,
                    self._force_measurement_variance,
                    self._force_measurement_variance,
                ],
                dtype=np.float64,
            )
        self._measurement_std_diag = np.sqrt(self._measurement_variance_diag)

    @property
    def num_particles(self) -> int:
        return self._num_particles

    @property
    def measurement_dim(self) -> int:
        return self._measurement_dim

    def initialize_particles(self) -> np.ndarray:
        masses = self._rng.uniform(self.min, self.max, size=self._num_particles).astype(np.float32)
        self._batch = self._create_batch(masses, simplified_contacts=self._simplified_contacts_enabled)
        if self._mj_model is None or self._mj_data is None:
            raise RuntimeError("MuJoCo model/data must be initialized before Warp particle warmup.")
        clean_qpos = self._require_finite_array(name="warmup_clean_qpos", values=self._mj_data.qpos).copy()
        clean_qvel = self._require_finite_array(name="warmup_clean_qvel", values=self._mj_data.qvel).copy()
        clean_ctrl = self._require_finite_array(name="warmup_clean_ctrl", values=self._mj_data.ctrl).copy()
        self.logger.info(
            {
                **self.logging_data,
                "event": "warp_runtime_warmup_start",
                "msg": (f"Started warming up the Warp runtime with {self._num_particles} particles."),
            }
        )
        self._batch.warmup()
        # Warmup is compile-only. Discard warmup state and restore a clean,
        # finite MuJoCo state before real particle updates begin.
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        self._mj_data.qpos[:] = clean_qpos
        self._mj_data.qvel[:] = clean_qvel
        self._mj_data.ctrl[:] = clean_ctrl
        if hasattr(self._mj_data, "act"):
            self._mj_data.act[:] = 0.0
        if hasattr(self._mj_data, "qacc_warmstart"):
            self._mj_data.qacc_warmstart[:] = 0.0
        self._mj_data.sensordata[:] = 0.0
        self._require_finite_array(name="post_reset_qpos", values=self._mj_data.qpos)
        self._require_finite_array(name="post_reset_qvel", values=self._mj_data.qvel)
        self._require_finite_array(name="post_reset_ctrl", values=self._mj_data.ctrl)
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self._require_finite_array(name="post_forward_qpos", values=self._mj_data.qpos)
        self._require_finite_array(name="post_forward_qvel", values=self._mj_data.qvel)
        self._require_finite_array(name="post_forward_ctrl", values=self._mj_data.ctrl)
        self._masses = masses.copy()
        self.sync_world_state(
            qpos=np.asarray(self._mj_data.qpos, dtype=np.float32),
            qvel=np.asarray(self._mj_data.qvel, dtype=np.float32),
            ctrl=np.asarray(self._mj_data.ctrl, dtype=np.float32),
        )
        post_sync_counts = self._batch.state_nonfinite_counts() if self._batch is not None else {}
        if post_sync_counts and self._has_nonfinite_state(post_sync_counts):
            raise RuntimeError(f"Non-finite Warp state detected after warmup reset: {post_sync_counts}")
        self.logger.info(
            {
                **self.logging_data,
                "event": "warp_runtime_warmup_done",
                "msg": "Finished warming up the Warp runtime and restored a clean state.",
            }
        )
        self._rng = np.random.default_rng()
        self._step_count = 0
        return masses

    def _build_model(self, *, simplified_contacts: bool) -> tuple[object, object, int, int, int]:
        mj_model, mj_data = load_mujoco_model("models/scene.xml")
        modify_object_properties(mj_model, mj_data, "object", DEFAULT_OBJECT_PROPS)
        if simplified_contacts:
            prepare_model_for_warp(mj_model)
        block_body_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "object",
        )
        force_sensor_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "hand_force",
        )
        force_adr = int(mj_model.sensor_adr[force_sensor_id])
        torque_sensor_id = mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "hand_torque",
        )
        torque_adr = int(mj_model.sensor_adr[torque_sensor_id])
        return mj_model, mj_data, block_body_id, force_adr, torque_adr

    def _create_batch(self, masses: np.ndarray, *, simplified_contacts: bool) -> WarpBatch:
        (
            self._mj_model,
            self._mj_data,
            self.block_body_id,
            self.force_adr,
            self.torque_adr,
        ) = self._build_model(simplified_contacts=simplified_contacts)
        self._simplified_contacts_enabled = simplified_contacts
        return WarpBatch(
            self._mj_model,
            self._mj_data,
            masses,
            self.block_body_id,
            nconmax=self._nconmax,
            njmax=self._njmax,
            logging_data=self.logging_data,
        )

    @staticmethod
    def _has_nonfinite_state(counts: dict[str, int]) -> bool:
        return any(value > 0 for value in counts.values())

    def _require_finite_array(self, *, name: str, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Encountered non-finite values in {name}.")
        return array

    def set_contact_mode(self, *, simplified_contacts: bool) -> None:
        if self._simplified_contacts_enabled == simplified_contacts:
            return

        masses = self._masses.copy()
        previous_snapshot = None
        state_transfer_applied = False
        if self._batch is not None:
            pre_switch_counts = self._batch.state_nonfinite_counts()
            if self._has_nonfinite_state(pre_switch_counts):
                self.logger.warning(
                    {
                        **self.logging_data,
                        "event": "warp_contact_mode_pre_switch_invalid_state",
                        "msg": "Detected non-finite Warp state before contact-mode switch; skipping state transfer.",
                        **pre_switch_counts,
                    }
                )
            else:
                previous_snapshot = self._batch.snapshot_state()

        self._batch = self._create_batch(masses, simplified_contacts=simplified_contacts)
        if previous_snapshot is not None:
            self._batch.restore_state(previous_snapshot)
            state_transfer_applied = True

        post_switch_counts = self._batch.state_nonfinite_counts()
        if self._has_nonfinite_state(post_switch_counts):
            self.logger.warning(
                {
                    **self.logging_data,
                    "event": "warp_contact_mode_post_switch_invalid_state",
                    "msg": "Detected non-finite Warp state after contact-mode switch.",
                    **post_switch_counts,
                }
            )
        self.logger.info(
            {
                **self.logging_data,
                "event": "warp_contact_mode_switched",
                "msg": (
                    "Switched Warp contact mode to simplified replay."
                    if simplified_contacts
                    else "Switched Warp contact mode to full-fidelity measurement."
                ),
                "simplified_contacts": simplified_contacts,
                "state_transfer_applied": state_transfer_applied,
            }
        )

    def propagate(self, particles: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        if self._batch is None:
            raise RuntimeError("Warp batch must be initialized before propagation.")

        noise = self._rng.standard_normal(self._num_particles).astype(np.float32) * self.std_dev
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

        noise = self._rng.standard_normal((controls.shape[0], self._num_particles)).astype(np.float32) * self.std_dev
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

    def propagate_masses_only(self, particles: np.ndarray, steps: int) -> np.ndarray:
        step_count = max(int(steps), 0)
        current = np.asarray(particles, dtype=np.float32)
        if step_count <= 0:
            return current.copy()
        noise = self._rng.standard_normal((step_count, self._num_particles)).astype(np.float32) * self.std_dev
        next_particles = np.clip(
            current + np.sum(noise, axis=0),
            self.min,
            self.max,
        ).astype(np.float32)
        self._masses = next_particles.copy()
        if self._batch is not None:
            self._batch.set_masses(self._masses)
            self._batch.forward()
        self._step_count += step_count
        return next_particles

    def sync_world_state(self, *, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray) -> None:
        if self._batch is None:
            raise RuntimeError("Warp batch must be initialized before state sync.")
        qpos_np = self._require_finite_array(name="sync_qpos", values=qpos)
        qvel_np = self._require_finite_array(name="sync_qvel", values=qvel)
        ctrl_np = self._require_finite_array(name="sync_ctrl", values=ctrl)
        self._batch.set_masses(self._masses)
        self._batch.set_state_all_worlds(qpos=qpos_np, qvel=qvel_np, ctrl=ctrl_np)
        self._batch.forward()

    def compute_likelihoods(self, particles: np.ndarray, observation: np.ndarray) -> np.ndarray:
        del particles
        if self._batch is None:
            raise RuntimeError("Warp batch must be initialized before likelihood evaluation.")
        with annotate("warp_sensor_slice"):
            sim_forces = self._batch.sensor_slice(self.force_adr, 3)
            if self._measurement_dim == 6:
                sim_torques = self._batch.sensor_slice(self.torque_adr, 3)
                sim_observation = np.concatenate([sim_forces, sim_torques], axis=1)
            else:
                sim_torques = np.zeros_like(sim_forces)
                sim_observation = sim_forces
        observation_np = np.asarray(observation, dtype=np.float64).reshape(-1)
        if observation_np.size < self._measurement_dim:
            raise ValueError(
                f"Expected at least {self._measurement_dim} observation values, got {observation_np.size}."
            )
        observation_used = observation_np[: self._measurement_dim]
        sim_observation64 = np.asarray(sim_observation, dtype=np.float64)
        diff = observation_used - sim_observation64
        scaled_diff = diff / self._measurement_std_diag[np.newaxis, :]
        dist_sq = np.sum(scaled_diff**2, axis=1)
        sim_force_finite = np.isfinite(sim_observation64)
        diff_finite = np.isfinite(scaled_diff)
        valid_force_particle = np.all(sim_force_finite, axis=1)
        valid_diff_particle = np.all(diff_finite, axis=1)
        valid_dist_particle = np.isfinite(dist_sq)
        invalid_world_mask = self._batch.invalid_world_mask() if self._batch is not None else np.zeros((self._num_particles,), dtype=bool)
        valid_likelihood_particle = valid_force_particle & valid_diff_particle & valid_dist_particle
        invalid_likelihood_particle_count = int(np.count_nonzero(~valid_likelihood_particle))
        likelihoods = np.full((self._num_particles,), float("nan"), dtype=np.float64)
        if np.any(valid_likelihood_particle):
            valid_dist_sq = dist_sq[valid_likelihood_particle]
            shifted_dist_sq = valid_dist_sq - float(np.min(valid_dist_sq))
            likelihoods[valid_likelihood_particle] = np.exp(
                -0.5 * shifted_dist_sq / self._likelihood_temperature
            )
        likelihood_finite = np.isfinite(likelihoods)
        sim_force_nonfinite_count = int(sim_observation.size - np.count_nonzero(sim_force_finite))
        diff_nonfinite_count = int(diff.size - np.count_nonzero(diff_finite))
        likelihood_nonfinite_count = int(likelihoods.size - np.count_nonzero(likelihood_finite))
        sensor_invalid_now = sim_force_nonfinite_count > 0 or diff_nonfinite_count > 0 or likelihood_nonfinite_count > 0
        sensor_invalid_transition = sensor_invalid_now and not self._sensor_invalid_active
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
            "qacc_warmstart_nonfinite_count": 0,
        }
        if self._batch is not None:
            state_nonfinite_counts = self._batch.state_nonfinite_counts()
            state_invalid_now = any(value > 0 for value in state_nonfinite_counts.values())
            state_invalid_transition = state_invalid_now and not self._state_invalid_active
            if state_invalid_transition:
                self._invalid_state_events += 1
                if self._first_invalid_state_step is None:
                    self._first_invalid_state_step = self._step_count
            self._state_invalid_active = state_invalid_now
        else:
            state_invalid_transition = False
        raw_contact_counts = self._batch.contact_counts() if self._batch is not None else np.zeros((0,), dtype=np.int32)
        contact_counts_shape_valid = raw_contact_counts.size == self._num_particles
        contact_counts = (
            raw_contact_counts
            if contact_counts_shape_valid
            else np.zeros((self._num_particles,), dtype=np.int32)
        )

        force_norms = np.linalg.norm(np.asarray(sim_forces, dtype=np.float64), axis=1)
        diff_norms = np.linalg.norm(diff, axis=1)
        finite_force_norms_all = force_norms[np.isfinite(force_norms)]
        masses64 = np.asarray(self._masses, dtype=np.float64)
        mass_finite_mask = np.isfinite(masses64)
        finite_mass_values = masses64[mass_finite_mask]
        paired_mask = mass_finite_mask & np.isfinite(force_norms)
        mass_force_norm_corr = float("nan")
        if np.count_nonzero(paired_mask) >= 2:
            paired_masses = masses64[paired_mask]
            paired_force_norms = force_norms[paired_mask]
            if np.std(paired_masses) > 1e-9 and np.std(paired_force_norms) > 1e-9:
                mass_force_norm_corr = float(np.corrcoef(paired_masses, paired_force_norms)[0, 1])
        sim_force_signal_particle_ratio = float(np.mean(force_norms > 1e-3)) if force_norms.size else 0.0
        contact_metric_available = float(contact_counts_shape_valid and contact_counts.size > 0)
        contact_force_mismatch = float(
            contact_counts.size > 0 and float(np.max(contact_counts)) == 0.0 and sim_force_signal_particle_ratio > 0.0
        )
        finite_dist_sq = dist_sq[np.isfinite(dist_sq)]
        finite_diff_norms = diff_norms[np.isfinite(diff_norms)]
        finite_force_norms = force_norms[np.isfinite(force_norms)]
        finite_likelihoods = likelihoods[np.isfinite(likelihoods)]
        if sensor_invalid_transition:
            self.logger.warning(
                {
                    **self.logging_data,
                    "event": "warp_invalid_sensor_state",
                    "msg": f"Detected an invalid Warp sensor state at step {self._step_count}.",
                    "step": self._step_count,
                }
            )
        if state_invalid_transition:
            self.logger.warning(
                {
                    **self.logging_data,
                    "event": "warp_invalid_backend_state",
                    "msg": f"Detected an invalid Warp backend state at step {self._step_count}.",
                    "step": self._step_count,
                }
            )

        self._last_measurement_diagnostics = {
            "obs_fx": float(observation_used[0]) if observation_used.size > 0 else 0.0,
            "obs_fy": float(observation_used[1]) if observation_used.size > 1 else 0.0,
            "obs_fz": float(observation_used[2]) if observation_used.size > 2 else 0.0,
            "obs_tx": float(observation_used[3]) if observation_used.size > 3 else 0.0,
            "obs_ty": float(observation_used[4]) if observation_used.size > 4 else 0.0,
            "obs_tz": float(observation_used[5]) if observation_used.size > 5 else 0.0,
            "obs_norm": float(np.linalg.norm(observation_used)),
            "likelihood_temperature": float(self._likelihood_temperature),
            "force_measurement_variance": float(self._force_measurement_variance),
            "torque_measurement_variance": float(self._torque_measurement_variance),
            "sim_force_norm_min": float(np.min(finite_force_norms)) if finite_force_norms.size else float("nan"),
            "sim_force_norm_max": float(np.max(finite_force_norms)) if finite_force_norms.size else float("nan"),
            "sim_force_norm_mean": float(np.mean(finite_force_norms)) if finite_force_norms.size else float("nan"),
            "sim_force_norm_std": float(np.std(finite_force_norms_all)) if finite_force_norms_all.size else float("nan"),
            "sim_force_axis_std_x": float(np.nanstd(sim_forces[:, 0])),
            "sim_force_axis_std_y": float(np.nanstd(sim_forces[:, 1])),
            "sim_force_axis_std_z": float(np.nanstd(sim_forces[:, 2])),
            "sim_torque_axis_std_x": float(np.nanstd(sim_torques[:, 0])),
            "sim_torque_axis_std_y": float(np.nanstd(sim_torques[:, 1])),
            "sim_torque_axis_std_z": float(np.nanstd(sim_torques[:, 2])),
            "particle_mass_min": float(np.min(finite_mass_values)) if finite_mass_values.size else float("nan"),
            "particle_mass_max": float(np.max(finite_mass_values)) if finite_mass_values.size else float("nan"),
            "particle_mass_mean": float(np.mean(finite_mass_values)) if finite_mass_values.size else float("nan"),
            "particle_mass_std": float(np.std(finite_mass_values)) if finite_mass_values.size else float("nan"),
            "mass_force_norm_corr": mass_force_norm_corr,
            "dist_sq_min": float(np.min(finite_dist_sq)) if finite_dist_sq.size else float("nan"),
            "dist_sq_max": float(np.max(finite_dist_sq)) if finite_dist_sq.size else float("nan"),
            "dist_sq_mean": float(np.mean(finite_dist_sq)) if finite_dist_sq.size else float("nan"),
            "dist_sq_std": float(np.std(finite_dist_sq)) if finite_dist_sq.size else float("nan"),
            "diff_norm_min": float(np.min(finite_diff_norms)) if finite_diff_norms.size else float("nan"),
            "diff_norm_max": float(np.max(finite_diff_norms)) if finite_diff_norms.size else float("nan"),
            "diff_norm_mean": float(np.mean(finite_diff_norms)) if finite_diff_norms.size else float("nan"),
            "sim_force_finite_ratio": float(np.mean(sim_force_finite)) if sim_force_finite.size else 0.0,
            "diff_finite_ratio": float(np.mean(diff_finite)) if diff_finite.size else 0.0,
            "likelihood_finite_ratio": (float(np.mean(likelihood_finite)) if likelihood_finite.size else 0.0),
            "valid_force_particle_ratio": (float(np.mean(valid_force_particle)) if valid_force_particle.size else 0.0),
            "valid_likelihood_particle_ratio": (
                float(np.mean(valid_likelihood_particle)) if valid_likelihood_particle.size else 0.0
            ),
            "invalid_likelihood_particle_count": float(invalid_likelihood_particle_count),
            "sim_force_signal_particle_ratio": sim_force_signal_particle_ratio,
            "contact_count_mean": float(np.mean(contact_counts)) if contact_counts.size else 0.0,
            "contact_count_max": float(np.max(contact_counts)) if contact_counts.size else 0.0,
            "active_contact_particle_ratio": (float(np.mean(contact_counts > 0)) if contact_counts.size else 0.0),
            "contact_metric_available": contact_metric_available,
            "contact_count_worlds_reported": float(raw_contact_counts.size),
            "contact_force_mismatch": contact_force_mismatch,
            "sim_force_nonfinite_count": float(sim_force_nonfinite_count),
            "diff_nonfinite_count": float(diff_nonfinite_count),
            "likelihood_nonfinite_count": float(likelihood_nonfinite_count),
            "invalid_sensor_events": float(self._invalid_sensor_events),
            "invalid_state_events": float(self._invalid_state_events),
            "first_invalid_sensor_step": float(
                self._first_invalid_sensor_step if self._first_invalid_sensor_step is not None else -1
            ),
            "first_invalid_state_step": float(
                self._first_invalid_state_step if self._first_invalid_state_step is not None else -1
            ),
            "qpos_nonfinite_count": float(state_nonfinite_counts["qpos_nonfinite_count"]),
            "qvel_nonfinite_count": float(state_nonfinite_counts["qvel_nonfinite_count"]),
            "sensordata_nonfinite_count": float(state_nonfinite_counts["sensordata_nonfinite_count"]),
            "ctrl_nonfinite_count": float(state_nonfinite_counts["ctrl_nonfinite_count"]),
            "qacc_warmstart_nonfinite_count": float(state_nonfinite_counts.get("qacc_warmstart_nonfinite_count", 0)),
            "invalid_world_count": float(np.count_nonzero(invalid_world_mask)),
            "repaired_world_count": 0.0,
            "likelihood_min": float(np.min(finite_likelihoods)) if finite_likelihoods.size else 0.0,
            "likelihood_max": float(np.max(finite_likelihoods)) if finite_likelihoods.size else 0.0,
            "likelihood_mean": float(np.mean(finite_likelihoods)) if finite_likelihoods.size else 0.0,
            "likelihood_std": float(np.std(finite_likelihoods)) if finite_likelihoods.size else 0.0,
            "likelihood_range": (
                float(np.max(finite_likelihoods) - np.min(finite_likelihoods)) if finite_likelihoods.size else 0.0
            ),
        }
        if self._step_count == 0:
            self.logger.info(
                {
                    **self.logging_data,
                    "event": "warp_first_valid_measurement",
                    "msg": "First phase-4 measurement observed after reset.",
                    "step": self._step_count,
                    "likelihood_finite_ratio": float(np.mean(likelihood_finite)) if likelihood_finite.size else 0.0,
                    "sim_force_nonfinite_count": sim_force_nonfinite_count,
                    "diff_nonfinite_count": diff_nonfinite_count,
                    "likelihood_nonfinite_count": likelihood_nonfinite_count,
                }
            )
        if WARP_LIKELIHOOD_DEBUG_ENABLED:
            best_idx = int(np.argmax(likelihoods)) if likelihoods.size else -1
            sim_force_mean = np.mean(sim_observation64, axis=0) if sim_observation64.size else np.zeros((self._measurement_dim,), dtype=np.float64)
            sim_force_std = np.std(sim_observation64, axis=0) if sim_observation64.size else np.zeros((self._measurement_dim,), dtype=np.float64)
            sim_force_best = sim_observation64[best_idx] if best_idx >= 0 else np.zeros((self._measurement_dim,), dtype=np.float64)
            best_mass = float(self._masses[best_idx]) if best_idx >= 0 and self._masses.size else 0.0
            self.logger.debug(
                {
                    **self.logging_data,
                    "event": "warp_likelihood_debug",
                    "step": self._step_count,
                    "likelihood_space": self._likelihood_space,
                    "observation_shape": tuple(observation_used.shape),
                    "sim_shape": tuple(sim_observation.shape),
                    "observation": observation_used.tolist(),
                    "sim_wrench_mean": sim_force_mean.tolist(),
                    "sim_wrench_std": sim_force_std.tolist(),
                    "sim_wrench_best": sim_force_best.tolist(),
                    "best_particle_index": best_idx,
                    "best_particle_mass": best_mass,
                    "dist_sq_min": float(np.min(finite_dist_sq)) if finite_dist_sq.size else float("nan"),
                    "dist_sq_max": float(np.max(finite_dist_sq)) if finite_dist_sq.size else float("nan"),
                    "dist_sq_mean": float(np.mean(finite_dist_sq)) if finite_dist_sq.size else float("nan"),
                    "likelihood_min": float(np.min(likelihoods)),
                    "likelihood_max": float(np.max(likelihoods)),
                    "likelihood_mean": float(np.mean(likelihoods)),
                }
            )

        return likelihoods

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
        normalized_lengths = sorted({int(length) for length in rollout_lengths if int(length) > 0})
        if normalized_lengths:
            min_length = normalized_lengths[0]
            self._batch.warmup_rollout(min_length)
            warmed_lengths = [min_length]
        else:
            warmed_lengths = []
        self.logger.info(
            {
                **self.logging_data,
                "event": "warp_runtime_rollout_warmup_done",
                "msg": "Finished warming up Warp rollout execution.",
            }
        )
        return warmed_lengths

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
