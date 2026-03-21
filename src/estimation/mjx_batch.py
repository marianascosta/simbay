import logging

import jax
import jax.numpy as jnp
from mujoco import mjx


def _resolve_mjx_device():
    """Use JAX default device when MJX supports it, otherwise fall back to CPU."""
    device = jax.devices()[0]
    if device.platform in {"cpu", "gpu", "tpu"}:
        return device
    return jax.devices("cpu")[0]


def _device_kind(device) -> str:
    return str(getattr(device, "device_kind", device))


class MJXBatch:
    """Manage a batched MJX model/data pair for particle simulation."""

    def __init__(self, mj_model, mj_data, masses, body_id: int):
        self.logger = logging.getLogger("simbay.mjx_batch")
        self._body_id = body_id
        self._size = int(len(masses))
        self._device = _resolve_mjx_device()
        self._default_device = jax.devices()[0]
        self._step_call_count = 0
        self._step_chunk_call_count = 0
        self._step_signature: tuple[tuple[int, ...], str, tuple[int, ...], str] | None = None
        self._step_chunk_signature: tuple[
            tuple[int, ...],
            str,
            tuple[int, ...],
            str,
            tuple[int, ...],
        ] | None = None

        mjx_model = mjx.put_model(mj_model, device=self._device)
        mjx_data = mjx.put_data(mj_model, mj_data, device=self._device)

        self._model = jax.tree.map(lambda x: jnp.stack([x] * self._size), mjx_model)
        self._model = self._model.replace(
            body_mass=self._model.body_mass.at[:, self._body_id].set(jnp.asarray(masses))
        )
        self._data = jax.tree.map(lambda x: jnp.stack([x] * self._size), mjx_data)
        self._ctrl_dim = int(self._data.ctrl.shape[-1])
        self._step = jax.jit(self._build_step_fn())
        self._step_chunks: dict[int, callable] = {}
        self._resample = jax.jit(self._build_resample_fn())

    @property
    def ctrl_dim(self) -> int:
        return self._ctrl_dim

    @property
    def device(self):
        return self._device

    def _build_step_fn(self):
        body_id = self._body_id
        size = self._size
        ctrl_dim = self._ctrl_dim
        step_fn = jax.vmap(mjx.step, in_axes=(0, 0))

        def apply(model, data, control_input, masses):
            next_model = model.replace(
                body_mass=model.body_mass.at[:, body_id].set(jnp.asarray(masses))
            )
            control = jnp.broadcast_to(jnp.asarray(control_input), (size, ctrl_dim))
            next_data = data.replace(ctrl=control)
            next_data = step_fn(next_model, next_data)
            return next_model, next_data

        return apply

    def _build_resample_fn(self):
        def resample(data, body_mass, indexes):
            return (
                jax.tree.map(lambda x: x[indexes], data),
                body_mass[indexes],
            )

        return resample

    def _build_step_chunk_fn(self):
        body_id = self._body_id
        size = self._size
        ctrl_dim = self._ctrl_dim
        step_fn = jax.vmap(mjx.step, in_axes=(0, 0))

        def apply(model, data, control_chunk, masses_chunk, step_mask):
            def scan_step(carry, inputs):
                current_model, current_data = carry
                control_input, masses, active = inputs
                next_model = current_model.replace(
                    body_mass=current_model.body_mass.at[:, body_id].set(jnp.asarray(masses))
                )
                control = jnp.broadcast_to(jnp.asarray(control_input), (size, ctrl_dim))
                next_data = current_data.replace(ctrl=control)
                next_data = step_fn(next_model, next_data)

                next_carry = jax.lax.cond(
                    active,
                    lambda _: (next_model, next_data),
                    lambda _: (current_model, current_data),
                    operand=None,
                )
                return next_carry, None

            (final_model, final_data), _ = jax.lax.scan(
                scan_step,
                (model, data),
                (control_chunk, masses_chunk, step_mask),
            )
            return final_model, final_data

        return apply

    def warmup(self) -> None:
        warm_model, warm_data = self._step(
            self._model,
            self._data,
            jnp.zeros((self._ctrl_dim,)),
            self._model.body_mass[:, self._body_id],
        )
        indexes = jnp.arange(self._size, dtype=jnp.int32)
        warm_data, body_mass = self._resample(
            warm_data,
            warm_model.body_mass,
            indexes,
        )
        warm_model = warm_model.replace(body_mass=body_mass)
        jax.block_until_ready((warm_model, warm_data))

    def warmup_chunk(self, chunk_size: int) -> None:
        step_chunk = self._step_chunks.get(chunk_size)
        if step_chunk is None:
            step_chunk = jax.jit(self._build_step_chunk_fn())
            self._step_chunks[chunk_size] = step_chunk

        control_chunk = jnp.zeros((chunk_size, self._ctrl_dim), dtype=self._data.ctrl.dtype)
        masses_chunk = jnp.broadcast_to(
            self._model.body_mass[:, self._body_id],
            (chunk_size, self._size),
        )
        step_mask = jnp.ones((chunk_size,), dtype=bool)
        warm_model, warm_data = step_chunk(
            self._model,
            self._data,
            control_chunk,
            masses_chunk,
            step_mask,
        )
        jax.block_until_ready((warm_model.body_mass, warm_data.qpos, warm_data.sensordata))

    @property
    def step_call_count(self) -> int:
        return self._step_call_count

    def _audit_step_signature(self, control_input, masses, phase: str | None) -> None:
        control = jnp.asarray(control_input)
        particle_masses = jnp.asarray(masses)
        signature = (
            tuple(control.shape),
            str(control.dtype),
            tuple(particle_masses.shape),
            str(particle_masses.dtype),
        )
        if self._step_signature is None:
            self._step_signature = signature
            return
        if self._step_signature != signature:
            self.logger.warning(
                "mjx_step_signature_changed phase=%s old=%s new=%s",
                phase or "unknown",
                self._step_signature,
                signature,
            )
            self._step_signature = signature

    def _audit_step_chunk_signature(
        self,
        control_chunk,
        masses_chunk,
        step_mask,
        phase: str | None,
    ) -> None:
        controls = jnp.asarray(control_chunk)
        particle_masses = jnp.asarray(masses_chunk)
        mask = jnp.asarray(step_mask)
        signature = (
            tuple(controls.shape),
            str(controls.dtype),
            tuple(particle_masses.shape),
            str(particle_masses.dtype),
            tuple(mask.shape),
        )
        if self._step_chunk_signature is None:
            self._step_chunk_signature = signature
            return
        if self._step_chunk_signature != signature:
            self.logger.warning(
                "mjx_step_chunk_signature_changed phase=%s old=%s new=%s",
                phase or "unknown",
                self._step_chunk_signature,
                signature,
            )
            self._step_chunk_signature = signature

    def step(self, control_input, masses, phase: str | None = None) -> None:
        self._step_call_count += 1
        self._audit_step_signature(control_input, masses, phase)
        self._model, self._data = self._step(
            self._model,
            self._data,
            control_input,
            masses,
        )

    def step_chunk(self, control_chunk, masses_chunk, step_mask, phase: str | None = None) -> None:
        chunk_size = int(jnp.asarray(control_chunk).shape[0])
        step_chunk = self._step_chunks.get(chunk_size)
        if step_chunk is None:
            step_chunk = jax.jit(self._build_step_chunk_fn())
            self._step_chunks[chunk_size] = step_chunk

        self._step_chunk_call_count += 1
        self._audit_step_chunk_signature(control_chunk, masses_chunk, step_mask, phase)
        self._model, self._data = step_chunk(
            self._model,
            self._data,
            control_chunk,
            masses_chunk,
            step_mask,
        )

    def block_until_ready(self) -> None:
        jax.block_until_ready((self._model.body_mass, self._data.qpos, self._data.sensordata))

    def sensor_slice(self, start: int, width: int):
        return self._data.sensordata[:, start : start + width]

    def resample(self, indexes) -> None:
        jax_indexes = jnp.asarray(indexes)
        self._data, body_mass = self._resample(
            self._data,
            self._model.body_mass,
            jax_indexes,
        )
        self._model = self._model.replace(body_mass=body_mass)

    def memory_profile(self) -> dict[str, int | str | bool]:
        """Report the actual MJX execution device and its memory stats."""
        stats = {}
        try:
            stats = self._device.memory_stats() or {}
        except Exception:
            stats = {}

        return {
            "execution_platform": str(self._device.platform),
            "execution_device": _device_kind(self._device),
            "default_jax_platform": str(self._default_device.platform),
            "default_jax_device": _device_kind(self._default_device),
            "device_fallback_applied": self._device != self._default_device,
            "bytes_in_use": int(stats.get("bytes_in_use", 0)),
            "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
            "bytes_limit": int(stats.get("bytes_limit", 0)),
        }
