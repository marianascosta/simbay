import logging
import os

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

    @property
    def num_particles(self) -> int:
        return self._num_particles

    def initialize_particles(self) -> np.ndarray:
        n = self._num_particles
        masses = np.random.uniform(self.min, self.max, size=n)

        self._batch = MJXBatch(self._mj_model, self._mj_data, masses, self.block_body_id)
        self.logger.info("mjx_jit_warmup_start particles=%d", n)
        self._batch.warmup()
        self.logger.info("mjx_jit_warmup_done")

        self._step_count = 0

        return masses

    def propagate(self, particles: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        n = self._num_particles

        # Apply process noise
        self._rng_key, subkey = jax.random.split(self._rng_key)
        noise = jax.random.normal(subkey, shape=(n,)) * self.std_dev
        jax_particles = jnp.array(particles) + noise
        jax_particles = jnp.clip(jax_particles, self.min, self.max)

        if self._batch is None:
            raise RuntimeError("MJX batch must be initialized before propagation.")
        self._batch.step(control_input, jax_particles)

        self._step_count += 1
        if self._step_count % 500 == 0:
            self.logger.info("mjx_propagate step=%d", self._step_count)

        return np.asarray(jax_particles)

    def compute_likelihoods(self, particles: np.ndarray, observation: np.ndarray) -> np.ndarray:
        if self._batch is None:
            raise RuntimeError("MJX batch must be initialized before likelihood evaluation.")
        sim_forces = self._batch.sensor_slice(self.force_adr, 3)

        obs = jnp.array(observation)
        diff = obs - sim_forces
        R = 1.0
        dist_sq = jnp.sum(diff ** 2, axis=1)
        likelihoods = jnp.exp(-0.5 * dist_sq / R)

        return np.asarray(likelihoods)

    def resample_states(self, indexes: np.ndarray) -> None:
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
