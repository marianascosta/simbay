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
        self._body_id = body_id
        self._size = int(len(masses))
        self._device = _resolve_mjx_device()
        self._default_device = jax.devices()[0]

        mjx_model = mjx.put_model(mj_model, device=self._device)
        mjx_data = mjx.put_data(mj_model, mj_data, device=self._device)

        self._model = jax.tree.map(lambda x: jnp.stack([x] * self._size), mjx_model)
        self._model = self._model.replace(
            body_mass=self._model.body_mass.at[:, self._body_id].set(jnp.asarray(masses))
        )
        self._data = jax.tree.map(lambda x: jnp.stack([x] * self._size), mjx_data)
        self._ctrl_dim = int(self._data.ctrl.shape[-1])
        self._step = jax.jit(self._build_step_fn())
        self._rollout = jax.jit(self._build_rollout_fn())
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

    def _build_rollout_fn(self):
        step_fn = self._build_step_fn()

        def rollout(model, data, control_inputs, mass_trajectory):
            def scan_step(carry, inputs):
                scan_model, scan_data = carry
                control_input, masses = inputs
                next_model, next_data = step_fn(scan_model, scan_data, control_input, masses)
                return (next_model, next_data), ()

            (final_model, final_data), _ = jax.lax.scan(
                scan_step,
                (model, data),
                (control_inputs, mass_trajectory),
            )
            return final_model, final_data

        return rollout

    def _build_resample_fn(self):
        def resample(data, body_mass, indexes):
            return (
                jax.tree.map(lambda x: x[indexes], data),
                body_mass[indexes],
            )

        return resample

    def warmup(self) -> None:
        self._model, self._data = self._step(
            self._model,
            self._data,
            jnp.zeros((self._ctrl_dim,)),
            self._model.body_mass[:, self._body_id],
        )
        jax.block_until_ready(self._data)

    def step(self, control_input, masses) -> None:
        self._model, self._data = self._step(
            self._model,
            self._data,
            control_input,
            masses,
        )

    def rollout(self, control_inputs, mass_trajectory) -> None:
        self._model, self._data = self._rollout(
            self._model,
            self._data,
            jnp.asarray(control_inputs),
            jnp.asarray(mass_trajectory),
        )

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
