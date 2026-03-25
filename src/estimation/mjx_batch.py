import jax
import jax.numpy as jnp
from mujoco import mjx

from src.utils.tracing import get_tracer
from src.utils.tracing import span as tracing_span


_TRACER = get_tracer("simbay.mjx_batch")


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
        with tracing_span(_TRACER, "mjx_batch.__init__", {"particles": int(len(masses))}):
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
        with tracing_span(_TRACER, "mjx_batch._build_step_fn", {"particles": self._size}):
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
        with tracing_span(_TRACER, "mjx_batch._build_rollout_fn", {"particles": self._size}):
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
        with tracing_span(_TRACER, "mjx_batch._build_resample_fn", {"particles": self._size}):
            def resample(data, body_mass, indexes):
                return (
                    jax.tree.map(lambda x: x[indexes], data),
                    body_mass[indexes],
                )

            return resample

    def warmup(self) -> None:
        with tracing_span(_TRACER, "mjx_batch.warmup", {"particles": self._size}):
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

    def warmup_rollout(self, steps: int) -> None:
        with tracing_span(_TRACER, "mjx_batch.warmup_rollout", {"particles": self._size, "steps": steps}):
            if steps <= 0:
                return
            controls = jnp.zeros((steps, self._ctrl_dim))
            masses = jnp.broadcast_to(
                self._model.body_mass[:, self._body_id],
                (steps, self._size),
            )
            warm_model, warm_data = self._rollout(
                self._model,
                self._data,
                controls,
                masses,
            )
            jax.block_until_ready((warm_model, warm_data))

    def step(self, control_input, masses) -> None:
        with tracing_span(_TRACER, "mjx_batch.step", {"particles": self._size}):
            self._model, self._data = self._step(
                self._model,
                self._data,
                control_input,
                masses,
            )

    def rollout(self, control_inputs, mass_trajectory) -> None:
        step_count = int(getattr(control_inputs, "shape", [len(control_inputs)])[0]) if control_inputs is not None else 0
        with tracing_span(_TRACER, "mjx_batch.rollout", {"particles": self._size, "steps": step_count}):
            self._model, self._data = self._rollout(
                self._model,
                self._data,
                jnp.asarray(control_inputs),
                jnp.asarray(mass_trajectory),
            )

    def sensor_slice(self, start: int, width: int):
        with tracing_span(_TRACER, "mjx_batch.sensor_slice", {"start": start, "width": width}):
            return self._data.sensordata[:, start : start + width]

    def resample(self, indexes) -> None:
        with tracing_span(_TRACER, "mjx_batch.resample", {"particles": self._size}):
            jax_indexes = jnp.asarray(indexes)
            self._data, body_mass = self._resample(
                self._data,
                self._model.body_mass,
                jax_indexes,
            )
            self._model = self._model.replace(body_mass=body_mass)

    def memory_profile(self) -> dict[str, int | str | bool]:
        """Report the actual MJX execution device and its memory stats."""
        with tracing_span(_TRACER, "mjx_batch.memory_profile", {"particles": self._size}):
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
