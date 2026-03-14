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
        self._step = jax.jit(jax.vmap(mjx.step, in_axes=(0, 0)))
        self._ctrl_dim = int(self._data.ctrl.shape[-1])

    @property
    def ctrl_dim(self) -> int:
        return self._ctrl_dim

    def warmup(self) -> None:
        self._data = self._step(self._model, self._data)
        jax.block_until_ready(self._data)

    def step(self, control_input, masses) -> None:
        control = jnp.broadcast_to(jnp.asarray(control_input), (self._size, self._ctrl_dim))
        self._model = self._model.replace(
            body_mass=self._model.body_mass.at[:, self._body_id].set(jnp.asarray(masses))
        )
        self._data = self._data.replace(ctrl=control)
        self._data = self._step(self._model, self._data)

    def sensor_slice(self, start: int, width: int):
        return self._data.sensordata[:, start : start + width]

    def resample(self, indexes) -> None:
        jax_indexes = jnp.asarray(indexes)
        self._data = jax.tree.map(lambda x: x[jax_indexes], self._data)
        self._model = self._model.replace(body_mass=self._model.body_mass[jax_indexes])

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
