import logging
import os
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from src.utils import DEFAULT_OBJECT_PROPS
from src.utils import load_mujoco_model
from src.utils import modify_object_properties
from src.utils.mjx_utils import prepare_model_for_mjx

from .base import ParticleEnvironment
from .mjx_batch import MJXBatch


class FrankaMJXEnv(ParticleEnvironment):
    """Batched MJX particle environment for mass estimation.

    Runs all N particle simulations in a single ``jax.vmap`` call instead
    of a sequential Python loop.
    """

    def __init__(self, limits: tuple[float, float], num_particles: int):
        self.logger = logging.getLogger("simbay.mjx_env")
        self.min, self.max = limits
        self._num_particles = num_particles
        self.std_dev = 0.005

        xml_path = os.path.join("assets", "franka_fr3_v2", "scene.xml")
        self._mj_model, self._mj_data = load_mujoco_model(xml_path)
        modify_object_properties(self._mj_model, self._mj_data, "object", DEFAULT_OBJECT_PROPS)
        prepare_model_for_mjx(self._mj_model)

        self.block_body_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        force_sensor_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_force"
        )
        self.force_adr = int(self._mj_model.sensor_adr[force_sensor_id])

        self._batch: MJXBatch | None = None
        self._rng_key = jax.random.PRNGKey(42)
        self._propagate_once = jax.jit(self._build_propagate_once())
        self._propagate_chunk = jax.jit(self._build_propagate_chunk())
        self._replay_profile_log_every = 100
        self._shape_signatures: dict[str, tuple[tuple[int, ...], str]] = {}
        self.reset_replay_profile()

    @property
    def num_particles(self) -> int:
        return self._num_particles

    def initialize_particles(self) -> np.ndarray:
        return np.asarray(self.initialize_particles_device())

    def initialize_particles_device(self) -> jax.Array:
        n = self._num_particles
        masses = np.random.uniform(self.min, self.max, size=n)

        self._batch = MJXBatch(self._mj_model, self._mj_data, masses, self.block_body_id)
        self.logger.info("mjx_jit_warmup_start particles=%d", n)
        self._batch.warmup()
        self.logger.info("mjx_jit_warmup_done")

        self._step_count = 0
        self.reset_replay_profile()

        return jax.device_put(masses, self._batch.device)

    def _build_propagate_once(self):
        minimum = self.min
        maximum = self.max

        def propagate(particles: jax.Array, noise: jax.Array) -> jax.Array:
            return jnp.clip(particles + noise, minimum, maximum)

        return propagate

    def _build_propagate_chunk(self):
        minimum = self.min
        maximum = self.max

        def propagate_chunk(
            particles: jax.Array,
            noise_chunk: jax.Array,
            step_mask: jax.Array,
        ) -> jax.Array:
            def scan_step(current_particles, inputs):
                noise, active = inputs
                next_particles = jnp.clip(current_particles + noise, minimum, maximum)
                next_particles = jax.lax.cond(
                    active,
                    lambda _: next_particles,
                    lambda _: current_particles,
                    operand=None,
                )
                return next_particles, next_particles

            _, propagated_particles = jax.lax.scan(
                scan_step,
                particles,
                (noise_chunk, step_mask),
            )
            return propagated_particles

        return propagate_chunk

    def propagate(self, particles: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        return np.asarray(self.propagate_particles_device(particles, control_input))

    def reset_replay_profile(self) -> None:
        self._replay_profile = {
            "propagate_call_count": 0,
            "rng_wall_seconds": 0.0,
            "propagate_wall_seconds": 0.0,
            "batch_step_wall_seconds": 0.0,
            "block_until_ready_seconds": 0.0,
        }

    def _audit_replay_signature(self, name: str, array, phase: str | None) -> None:
        jax_array = jnp.asarray(array)
        signature = (tuple(jax_array.shape), str(jax_array.dtype))
        previous = self._shape_signatures.get(name)
        if previous is None:
            self._shape_signatures[name] = signature
            return
        if previous != signature:
            self.logger.warning(
                "mjx_replay_signature_changed phase=%s name=%s old=%s new=%s",
                phase or "unknown",
                name,
                previous,
                signature,
            )
            self._shape_signatures[name] = signature

    def replay_profile_snapshot(self) -> dict[str, float | int]:
        memory_profile = self.memory_profile()
        batch_step_call_count = self._batch.step_call_count if self._batch is not None else 0
        batch_step_chunk_call_count = (
            self._batch.step_chunk_call_count if self._batch is not None else 0
        )
        return {
            "propagate_call_count": int(self._replay_profile["propagate_call_count"]),
            "batch_step_call_count": int(batch_step_call_count),
            "batch_step_chunk_call_count": int(batch_step_chunk_call_count),
            "rng_wall_seconds": float(self._replay_profile["rng_wall_seconds"]),
            "propagate_wall_seconds": float(self._replay_profile["propagate_wall_seconds"]),
            "batch_step_wall_seconds": float(self._replay_profile["batch_step_wall_seconds"]),
            "block_until_ready_seconds": float(self._replay_profile["block_until_ready_seconds"]),
            "mjx_bytes_in_use": int(memory_profile["bytes_in_use"]),
            "mjx_peak_bytes_in_use": int(memory_profile["peak_bytes_in_use"]),
            "mjx_bytes_limit": int(memory_profile["bytes_limit"]),
        }

    @property
    def replay_profile_log_every(self) -> int:
        return self._replay_profile_log_every

    def warmup_replay_chunk(self, chunk_size: int) -> None:
        if self._batch is None:
            raise RuntimeError("MJX batch must be initialized before replay warmup.")

        zero_particles = jnp.zeros((self._num_particles,), dtype=self._batch._model.body_mass.dtype)
        zero_noise = jnp.zeros((chunk_size, self._num_particles), dtype=zero_particles.dtype)
        step_mask = jnp.ones((chunk_size,), dtype=bool)
        propagated_particles = self._propagate_chunk(zero_particles, zero_noise, step_mask)
        self._batch.warmup_chunk(chunk_size)
        jax.block_until_ready(propagated_particles)

    def propagate_particles_device(
        self,
        particles: np.ndarray | jax.Array,
        control_input: np.ndarray,
        *,
        synchronize: bool = False,
        phase: str | None = None,
    ) -> jax.Array:
        if self._batch is None:
            raise RuntimeError("MJX batch must be initialized before propagation.")

        self._replay_profile["propagate_call_count"] += 1
        self._audit_replay_signature("particles", particles, phase)
        self._audit_replay_signature("control_input", control_input, phase)

        rng_start = time.perf_counter()
        self._rng_key, subkey = jax.random.split(self._rng_key)
        noise = jax.random.normal(subkey, shape=(self._num_particles,)) * self.std_dev
        self._replay_profile["rng_wall_seconds"] += time.perf_counter() - rng_start

        propagate_start = time.perf_counter()
        jax_particles = self._propagate_once(jnp.asarray(particles), noise)
        self._replay_profile["propagate_wall_seconds"] += time.perf_counter() - propagate_start

        batch_step_start = time.perf_counter()
        self._batch.step(control_input, jax_particles, phase=phase)
        self._replay_profile["batch_step_wall_seconds"] += time.perf_counter() - batch_step_start

        if synchronize:
            block_start = time.perf_counter()
            self._batch.block_until_ready()
            jax.block_until_ready(jax_particles)
            self._replay_profile["block_until_ready_seconds"] += time.perf_counter() - block_start

        self._step_count += 1
        if self._step_count % 500 == 0:
            self.logger.info("mjx_propagate step=%d", self._step_count)

        return jax_particles

    def replay_chunked_device(
        self,
        particles: np.ndarray | jax.Array,
        control_inputs: np.ndarray,
        chunk_size: int,
        *,
        phase: str | None = None,
    ) -> jax.Array:
        if self._batch is None:
            raise RuntimeError("MJX batch must be initialized before chunked replay.")

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        jax_particles = jnp.asarray(particles)
        controls = np.asarray(control_inputs)
        total_steps = int(controls.shape[0])
        if total_steps == 0:
            return jax_particles

        self._audit_replay_signature("control_chunk", controls[: min(total_steps, chunk_size)], phase)
        pad_steps = (-total_steps) % chunk_size
        if pad_steps:
            controls = np.pad(controls, ((0, pad_steps), (0, 0)), mode="edge")

        for chunk_start in range(0, controls.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_steps)
            active_steps = chunk_end - chunk_start
            control_chunk = jnp.asarray(controls[chunk_start : chunk_start + chunk_size])
            step_mask = jnp.arange(chunk_size) < active_steps
            self._rng_key, subkey = jax.random.split(self._rng_key)
            noise_chunk = jax.random.normal(
                subkey,
                shape=(chunk_size, self._num_particles),
                dtype=jax_particles.dtype,
            ) * self.std_dev
            noise_chunk = jnp.where(step_mask[:, None], noise_chunk, 0.0)
            masses_chunk = self._propagate_chunk(jax_particles, noise_chunk, step_mask)
            self._batch.step_chunk(control_chunk, masses_chunk, step_mask, phase=phase)
            jax_particles = masses_chunk[-1]

        return jax_particles

    def block_until_ready(self) -> None:
        if self._batch is None:
            raise RuntimeError("MJX batch must be initialized before synchronization.")
        self._batch.block_until_ready()

    def replay_state_snapshot(self) -> dict[str, jax.Array]:
        if self._batch is None:
            raise RuntimeError("MJX batch must be initialized before snapshotting state.")
        return {
            "qpos": self._batch.qpos(),
            "force_sensor": self._batch.sensor_slice(self.force_adr, 3),
        }

    def compute_likelihoods(self, particles: np.ndarray, observation: np.ndarray) -> np.ndarray:
        return np.asarray(self.compute_likelihoods_device(observation))

    def compute_likelihoods_device(self, observation: np.ndarray) -> jax.Array:
        if self._batch is None:
            raise RuntimeError("MJX batch must be initialized before likelihood evaluation.")
        sim_forces = self._batch.sensor_slice(self.force_adr, 3)

        obs = jnp.array(observation)
        diff = obs - sim_forces
        R = 1.0
        dist_sq = jnp.sum(diff ** 2, axis=1)
        likelihoods = jnp.exp(-0.5 * dist_sq / R)

        return likelihoods

    def resample_states(self, indexes: np.ndarray) -> None:
        self.resample_states_device(indexes)

    def resample_states_device(self, indexes: np.ndarray | jax.Array) -> None:
        if self._batch is None:
            raise RuntimeError("MJX batch must be initialized before resampling.")
        self._batch.resample(indexes)

    def memory_profile(self) -> dict[str, int | float | str]:
        """Report the actual MJX execution device and memory usage."""
        default_device = jax.devices()[0]
        if self._batch is not None:
            return self._batch.memory_profile()
        return {
            "execution_platform": "uninitialized",
            "execution_device": "uninitialized",
            "default_jax_platform": str(default_device.platform),
            "default_jax_device": str(default_device.device_kind),
            "device_fallback_applied": False,
            "bytes_in_use": 0,
            "peak_bytes_in_use": 0,
            "bytes_limit": 0,
        }
