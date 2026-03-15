import logging

import jax
import jax.numpy as jnp

from src.utils.logging_utils import get_process_memory_bytes

from .mjx_particle_filter import FrankaMJXEnv


def _normalize_weights(weights: jax.Array, likelihoods: jax.Array) -> jax.Array:
    updated = weights * likelihoods
    updated = updated + 1.0e-300
    return updated / jnp.sum(updated)


def _effective_sample_size(weights: jax.Array) -> jax.Array:
    return 1.0 / jnp.sum(weights**2)


def _estimate_particles(particles: jax.Array, weights: jax.Array) -> jax.Array:
    return jnp.sum(particles * weights, axis=0)


def _systematic_resample(
    weights: jax.Array,
    particles: jax.Array,
    offset: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    count = weights.shape[0]
    positions = (jnp.arange(count, dtype=weights.dtype) + offset) / count
    cumulative = jnp.cumsum(weights)
    cumulative = cumulative.at[-1].set(1.0)
    indexes = jnp.searchsorted(cumulative, positions, side="right")
    resampled_particles = particles[indexes]
    uniform_weights = jnp.full_like(weights, 1.0 / count)
    return uniform_weights, resampled_particles, indexes


def _update_and_optionally_resample(
    weights: jax.Array,
    particles: jax.Array,
    likelihoods: jax.Array,
    resample_offset: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    updated_weights = _normalize_weights(weights, likelihoods)
    ess = _effective_sample_size(updated_weights)
    should_resample = ess < (weights.shape[0] / 2.0)

    def resample_branch(
        args: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        branch_weights, branch_particles, branch_offset = args
        return _systematic_resample(branch_weights, branch_particles, branch_offset)

    def keep_branch(
        args: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        branch_weights, branch_particles, _ = args
        return branch_weights, branch_particles, jnp.arange(branch_weights.shape[0])

    next_weights, next_particles, indexes = jax.lax.cond(
        should_resample,
        resample_branch,
        keep_branch,
        (updated_weights, particles, resample_offset),
    )
    reported_ess = jax.lax.cond(
        should_resample,
        lambda _: jnp.asarray(weights.shape[0], dtype=weights.dtype),
        lambda _: ess,
        operand=None,
    )
    return next_weights, next_particles, reported_ess, indexes, should_resample


class MJXParticleFilter:
    """JAX-backed particle filter for the MJX environment."""

    def __init__(self, env: FrankaMJXEnv):
        self.logger = logging.getLogger("simbay.mjx_particle_filter")
        self.env = env

        init_memory_before = get_process_memory_bytes()
        self.particles = self.env.initialize_particles_device()
        init_memory_after = get_process_memory_bytes()
        self.N = int(self.particles.shape[0])

        self.weights = jnp.full((self.N,), 1.0 / self.N, dtype=self.particles.dtype)
        self._rng_key = jax.random.PRNGKey(7)
        self._ess = float(self.N)

        state_bytes_total = int(self.particles.size * self.particles.dtype.itemsize)
        state_bytes_total += int(self.weights.size * self.weights.dtype.itemsize)
        self.state_bytes_total = state_bytes_total
        self.state_bytes_per_particle = (
            self.state_bytes_total / self.N if self.N else 0.0
        )
        self.process_memory_per_particle_estimate = (
            max(init_memory_after - init_memory_before, 0) / self.N if self.N else 0.0
        )

        self._estimate_jit = jax.jit(_estimate_particles)
        self._update_jit = jax.jit(_normalize_weights)
        self._ess_jit = jax.jit(_effective_sample_size)
        self._update_resample_jit = jax.jit(_update_and_optionally_resample)

        self.logger.info(
            "mjx_particle_filter_initialized particles=%d state_bytes_total=%d "
            "state_bytes_per_particle=%.2f process_memory_per_particle_estimate_bytes=%.2f",
            self.N,
            self.state_bytes_total,
            self.state_bytes_per_particle,
            self.process_memory_per_particle_estimate,
        )

    def predict(self, control_input) -> None:
        self.particles = self.env.propagate_particles_device(self.particles, control_input)

    def predict_trajectory(self, trajectory) -> None:
        self.particles = self.env.rollout_predict_only_device(self.particles, trajectory)

    def update(self, observation) -> None:
        likelihoods = self.env.compute_likelihoods_device(observation)
        self.weights = self._update_jit(self.weights, likelihoods)
        self._ess = float(jax.device_get(self._ess_jit(self.weights)))

    def resample(self) -> None:
        if self._ess >= self.N / 2:
            return

        self._rng_key, subkey = jax.random.split(self._rng_key)
        offset = jax.random.uniform(subkey, (), dtype=self.weights.dtype)
        self.weights, self.particles, indexes = _systematic_resample(
            self.weights,
            self.particles,
            offset,
        )
        self.env.resample_states_device(indexes)
        self._ess = float(self.N)

    def step(self, control_input, observation) -> dict[str, float | bool]:
        self.particles = self.env.propagate_particles_device(self.particles, control_input)
        likelihoods = self.env.compute_likelihoods_device(observation)

        self._rng_key, subkey = jax.random.split(self._rng_key)
        offset = jax.random.uniform(subkey, (), dtype=self.weights.dtype)
        (
            self.weights,
            self.particles,
            ess,
            indexes,
            did_resample,
        ) = self._update_resample_jit(self.weights, self.particles, likelihoods, offset)
        did_resample_host = bool(jax.device_get(did_resample))
        if did_resample_host:
            self.env.resample_states_device(indexes)
        self._ess = float(jax.device_get(ess))
        return {
            "ess": self._ess,
            "resampled": did_resample_host,
        }

    def effective_sample_size(self) -> float:
        return self._ess

    def estimate(self) -> float:
        estimate = self._estimate_jit(self.particles, self.weights)
        return float(jax.device_get(estimate))

    def particles_host(self):
        return jax.device_get(self.particles)

    def memory_profile(self) -> dict[str, float | int]:
        return {
            "particles": self.N,
            "state_bytes_total": self.state_bytes_total,
            "state_bytes_per_particle": self.state_bytes_per_particle,
            "process_memory_per_particle_estimate_bytes": self.process_memory_per_particle_estimate,
        }
