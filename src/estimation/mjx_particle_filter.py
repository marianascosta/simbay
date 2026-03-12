import logging
import os

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from src.utils import DEFAULT_OBJECT_PROPS
from src.utils import FRANKA_HOME_QPOS
from src.utils.mjx_utils import batch_mjx_data
from src.utils.mjx_utils import batch_mjx_model
from src.utils.mjx_utils import prepare_model_for_mjx

from .base import ParticleEnvironment


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

        # Load a single reference MuJoCo model
        xml_path = os.path.join("assets", "franka_fr3_v2", "scene.xml")
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        prepare_model_for_mjx(self._mj_model)

        # Resolve body / sensor IDs from the reference model
        self.block_body_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        force_sensor_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_force"
        )
        self.force_adr = int(self._mj_model.sensor_adr[force_sensor_id])

        # Convert the reference model to MJX (used as template for batching)
        self._mjx_model = mjx.put_model(self._mj_model)

        # JIT-compiled batched step
        self._batched_step = jax.jit(jax.vmap(mjx.step, in_axes=(0, 0)))

        # Will be set during initialize_particles
        self._mjx_model_batch = None
        self._mjx_data_batch = None
        self._rng_key = jax.random.PRNGKey(42)

    @property
    def num_particles(self) -> int:
        return self._num_particles

    def initialize_particles(self) -> np.ndarray:
        n = self._num_particles
        masses = np.random.uniform(self.min, self.max, size=n)

        # Build a single MjData at home configuration with default object props
        mj_data = mujoco.MjData(self._mj_model)
        mj_data.qpos[:7] = FRANKA_HOME_QPOS
        mj_data.ctrl[:7] = FRANKA_HOME_QPOS

        # Apply default object properties (position, size, friction)
        body_id = self.block_body_id
        geom_id = self._mj_model.body_geomadr[body_id]
        props = DEFAULT_OBJECT_PROPS

        type_map = {
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        }
        geom_type = props["type"]
        if geom_type in type_map:
            self._mj_model.geom_type[geom_id] = type_map[geom_type]
        self._mj_model.geom_size[geom_id] = np.array(props["size"])
        self._mj_model.body_mass[body_id] = props["mass"]
        self._mj_model.geom_friction[geom_id] = np.array(props["friction"])

        # Set object position via joint
        jnt_adr = self._mj_model.body_jntadr[body_id]
        if jnt_adr != -1:
            qpos_adr = self._mj_model.jnt_qposadr[jnt_adr]
            mj_data.qpos[qpos_adr : qpos_adr + 3] = np.array(props["pos"])
            dof_adr = self._mj_model.jnt_dofadr[jnt_adr]
            mj_data.qvel[dof_adr : dof_adr + 6] = 0.0

        mujoco.mj_forward(self._mj_model, mj_data)

        # Convert to MJX and batch
        mjx_data = mjx.put_data(self._mj_model, mj_data)

        # Re-convert model after property changes
        self._mjx_model = mjx.put_model(self._mj_model)

        jax_masses = jnp.array(masses)
        self._mjx_model_batch = batch_mjx_model(
            self._mjx_model, jax_masses, self.block_body_id, n
        )
        self._mjx_data_batch = batch_mjx_data(mjx_data, n)

        # Warm up JIT: the first call traces + compiles for this pytree shape
        self.logger.info("mjx_jit_warmup_start particles=%d", n)
        self._mjx_data_batch = self._batched_step(
            self._mjx_model_batch, self._mjx_data_batch
        )
        # Block until the computation actually finishes on device
        jax.block_until_ready(self._mjx_data_batch)
        self.logger.info("mjx_jit_warmup_done")

        self._ctrl_dim = self._mjx_data_batch.ctrl.shape[-1]
        self._step_count = 0

        return masses

    def propagate(self, particles: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        n = self._num_particles

        # Apply process noise
        self._rng_key, subkey = jax.random.split(self._rng_key)
        noise = jax.random.normal(subkey, shape=(n,)) * self.std_dev
        jax_particles = jnp.array(particles) + noise
        jax_particles = jnp.clip(jax_particles, self.min, self.max)

        # Update masses in the batched model
        new_body_mass = self._mjx_model_batch.body_mass.at[:, self.block_body_id].set(
            jax_particles
        )
        self._mjx_model_batch = self._mjx_model_batch.replace(body_mass=new_body_mass)

        # Broadcast control input to all particles
        ctrl = jnp.broadcast_to(
            jnp.array(control_input), (n, self._ctrl_dim)
        )
        self._mjx_data_batch = self._mjx_data_batch.replace(ctrl=ctrl)

        # Single vectorized step
        self._mjx_data_batch = self._batched_step(
            self._mjx_model_batch, self._mjx_data_batch
        )

        self._step_count += 1
        if self._step_count % 500 == 0:
            self.logger.info("mjx_propagate step=%d", self._step_count)

        return np.asarray(jax_particles)

    def compute_likelihoods(self, particles: np.ndarray, observation: np.ndarray) -> np.ndarray:
        # Read force sensor data — shape (N, 3)
        sim_forces = self._mjx_data_batch.sensordata[
            :, self.force_adr : self.force_adr + 3
        ]

        obs = jnp.array(observation)
        diff = obs - sim_forces
        R = 1.0
        dist_sq = jnp.sum(diff ** 2, axis=1)
        likelihoods = jnp.exp(-0.5 * dist_sq / R)

        return np.asarray(likelihoods)

    def resample_states(self, indexes: np.ndarray) -> None:
        jax_idx = jnp.array(indexes)
        self._mjx_data_batch = jax.tree.map(
            lambda x: x[jax_idx], self._mjx_data_batch
        )
        new_body_mass = self._mjx_model_batch.body_mass[jax_idx]
        self._mjx_model_batch = self._mjx_model_batch.replace(body_mass=new_body_mass)

    def memory_profile(self) -> dict[str, int | float | str]:
        """Report JAX device memory usage."""
        try:
            device = jax.local_devices()[0]
            stats = device.memory_stats()
            if stats:
                return {
                    "backend": str(device.platform),
                    "device_name": str(device.device_kind),
                    "bytes_in_use": int(stats.get("bytes_in_use", 0)),
                    "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
                    "bytes_limit": int(stats.get("bytes_limit", 0)),
                }
        except Exception:
            pass
        return {
            "backend": str(jax.local_devices()[0].platform),
            "device_name": str(jax.local_devices()[0].device_kind),
            "bytes_in_use": 0,
            "peak_bytes_in_use": 0,
            "bytes_limit": 0,
        }
