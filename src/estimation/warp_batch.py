"""
MuJoCo Warp batch backend.

This module owns the nworld-batched simulation state for all N particles.
It is the batched simulation backend used by the MuJoCo Warp flow.
"""

from __future__ import annotations

import logging

import numpy as np
import mujoco_warp as mjw
import warp as wp

logger = logging.getLogger("simbay.warp_batch")

_RESAMPLE_STATE_FIELDS = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")
_WARP_MEMORY_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc_warmstart",
    "sensordata",
    "ncon",
)
_RECOVERY_STATE_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc_warmstart",
    "sensordata",
)
_STATE_TRANSFER_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
)
_NONFINITE_CHECK_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qacc_warmstart",
    "sensordata",
)


def _assign_warp_array(array, values: np.ndarray, dtype=wp.float32) -> None:
    values = np.asarray(values)
    if hasattr(array, "assign"):
        array.assign(values)
        return
    wp.copy(array, wp.from_numpy(values, dtype=dtype))


class WarpBatch:
    """Manage a batched MJWarp model/data pair for particle simulation."""

    def __init__(
        self,
        mj_model,
        mj_data,
        masses: np.ndarray,
        body_id: int,
        nconmax: int = 128,
        njmax: int = 384,
        logging_data: dict[str, object] | None = None,
    ):
        self.logging_data = dict(logging_data or {})
        self._body_id = body_id
        self._size = int(len(masses))
        self._ctrl_dim = int(mj_model.nu)
        self._ctrl_np = np.zeros((self._size, self._ctrl_dim), dtype=np.float32)

        logger.info(
            {
                **self.logging_data,
                "event": "warp_batch_init",
                "msg": f"Initialised the Warp batch with {self._size} worlds.",
                "nworld": self._size,
            }
        )

        self._model = mjw.put_model(mj_model)
        self._data = mjw.put_data(
            mj_model,
            mj_data,
            nworld=self._size,
            nconmax=nconmax,
            njmax=njmax,
        )

        base_mass_np = np.tile(
            mj_model.body_mass[np.newaxis, :],
            (self._size, 1),
        ).astype(np.float32)
        base_mass_np[:, body_id] = np.asarray(masses, dtype=np.float32)
        self._body_mass_np = base_mass_np.copy()
        self._body_mass_column_np = self._body_mass_np[:, self._body_id]
        self._model.body_mass = wp.from_numpy(base_mass_np, dtype=wp.float32)
        mjw.set_const(self._model, self._data)
        self._peak_bytes_in_use = 0
        self._recovery_snapshot: dict[str, np.ndarray] | None = None

        logger.info(
            {
                **self.logging_data,
                "event": "warp_batch_init_complete",
                "msg": f"Finished setting up the Warp batch with {self._size} worlds.",
                "nworld": self._size,
            }
        )

    @property
    def ctrl_dim(self) -> int:
        return self._ctrl_dim

    def warmup(self) -> None:
        self._ctrl_np.fill(0.0)
        _assign_warp_array(self._data.ctrl, self._ctrl_np)
        mjw.step(self._model, self._data)
        wp.synchronize()
        logger.info(
            {
                **self.logging_data,
                "event": "warp_batch_warmup_done",
                "msg": f"Finished warming up the Warp batch for {self._size} worlds.",
                "nworld": self._size,
            }
        )

    def warmup_rollout(self, steps: int) -> None:
        if steps <= 0:
            return
        self._ctrl_np.fill(0.0)
        _assign_warp_array(self._data.ctrl, self._ctrl_np)
        for _ in range(steps):
            mjw.step(self._model, self._data)
        wp.synchronize()
        logger.info(
            {
                **self.logging_data,
                "event": "warp_batch_rollout_warmup_done",
                "msg": f"Finished Warp rollout warm-up over {steps} steps.",
            }
        )

    def step(self, control_input: np.ndarray, masses: np.ndarray) -> None:
        np.copyto(self._body_mass_column_np, np.asarray(masses, dtype=np.float32))
        _assign_warp_array(self._model.body_mass, self._body_mass_np)
        self._ctrl_np[:] = np.asarray(control_input, dtype=np.float32)
        _assign_warp_array(self._data.ctrl, self._ctrl_np)

        mjw.step(self._model, self._data)

    def rollout(self, control_inputs: np.ndarray, mass_trajectory: np.ndarray) -> None:
        steps = int(control_inputs.shape[0])
        if steps <= 0:
            return
        control_inputs_np = np.asarray(control_inputs, dtype=np.float32)
        mass_trajectory_np = np.asarray(mass_trajectory, dtype=np.float32)
        for step_idx in range(steps):
            np.copyto(self._body_mass_column_np, mass_trajectory_np[step_idx])
            _assign_warp_array(self._model.body_mass, self._body_mass_np)
            self._ctrl_np[:] = control_inputs_np[step_idx]
            _assign_warp_array(self._data.ctrl, self._ctrl_np)
            mjw.step(self._model, self._data)

    def sensor_slice(self, start: int, width: int) -> np.ndarray:
        return self._data.sensordata.numpy()[:, start : start + width]

    def contact_counts(self) -> np.ndarray:
        ncon = getattr(self._data, "ncon", None)
        if ncon is None:
            return np.zeros((self._size,), dtype=np.int32)
        return np.asarray(ncon.numpy(), dtype=np.int32).reshape(-1)

    def state_nonfinite_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for field_name in _NONFINITE_CHECK_FIELDS:
            array = getattr(self._data, field_name, None)
            if array is None:
                counts[f"{field_name}_nonfinite_count"] = 0
                continue
            np_array = np.asarray(array.numpy())
            counts[f"{field_name}_nonfinite_count"] = int(np_array.size - np.count_nonzero(np.isfinite(np_array)))
        return counts

    def invalid_world_mask(self) -> np.ndarray:
        invalid_mask = np.zeros((self._size,), dtype=bool)
        for field_name in _NONFINITE_CHECK_FIELDS:
            array = getattr(self._data, field_name, None)
            if array is None:
                continue
            np_array = np.asarray(array.numpy())
            invalid_mask |= ~np.all(np.isfinite(np_array), axis=1)
        return invalid_mask

    def resample(self, indexes: np.ndarray) -> None:
        indexes_np = np.asarray(indexes, dtype=np.int32)

        for field_name in _RESAMPLE_STATE_FIELDS:
            array = getattr(self._data, field_name, None)
            if array is None:
                continue
            np_array = array.numpy()
            _assign_warp_array(array, np_array[indexes_np])

        self._body_mass_np = self._body_mass_np[indexes_np]
        self._body_mass_column_np = self._body_mass_np[:, self._body_id]
        _assign_warp_array(self._model.body_mass, self._body_mass_np)
        mjw.set_const(self._model, self._data)

    def snapshot_state(self) -> dict[str, np.ndarray]:
        snapshot: dict[str, np.ndarray] = {
            "body_mass": self._body_mass_np.copy(),
        }
        for field_name in _STATE_TRANSFER_FIELDS:
            array = getattr(self._data, field_name, None)
            if array is None:
                continue
            snapshot[field_name] = np.asarray(array.numpy()).copy()
        return snapshot

    def restore_state(self, snapshot: dict[str, np.ndarray]) -> None:
        self._body_mass_np = snapshot["body_mass"].copy()
        self._body_mass_column_np = self._body_mass_np[:, self._body_id]
        _assign_warp_array(self._model.body_mass, self._body_mass_np)
        for field_name in _STATE_TRANSFER_FIELDS:
            array = getattr(self._data, field_name, None)
            values = snapshot.get(field_name)
            if array is None or values is None:
                continue
            _assign_warp_array(array, values)
        qacc_warmstart = getattr(self._data, "qacc_warmstart", None)
        if qacc_warmstart is not None:
            zeros = np.zeros_like(np.asarray(qacc_warmstart.numpy()))
            _assign_warp_array(qacc_warmstart, zeros)
        mjw.set_const(self._model, self._data)

    def capture_recovery_snapshot(self) -> None:
        snapshot: dict[str, np.ndarray] = {
            "body_mass": self._body_mass_np.copy(),
        }
        for field_name in _RECOVERY_STATE_FIELDS:
            array = getattr(self._data, field_name, None)
            if array is None:
                continue
            snapshot[field_name] = np.asarray(array.numpy()).copy()
        self._recovery_snapshot = snapshot

    def restore_recovery_snapshot(self) -> bool:
        if self._recovery_snapshot is None:
            return False
        snapshot = self._recovery_snapshot
        self._body_mass_np = snapshot["body_mass"].copy()
        self._body_mass_column_np = self._body_mass_np[:, self._body_id]
        _assign_warp_array(self._model.body_mass, self._body_mass_np)
        for field_name in _RECOVERY_STATE_FIELDS:
            array = getattr(self._data, field_name, None)
            values = snapshot.get(field_name)
            if array is None or values is None:
                continue
            _assign_warp_array(array, values)
        mjw.set_const(self._model, self._data)
        return True

    def repair_invalid_worlds_from_snapshot(self) -> np.ndarray:
        if self._recovery_snapshot is None:
            return np.zeros((self._size,), dtype=bool)

        invalid_mask = self.invalid_world_mask()
        if not np.any(invalid_mask):
            return invalid_mask

        snapshot = self._recovery_snapshot
        self._body_mass_np[invalid_mask] = snapshot["body_mass"][invalid_mask]
        self._body_mass_column_np = self._body_mass_np[:, self._body_id]
        _assign_warp_array(self._model.body_mass, self._body_mass_np)
        for field_name in _RECOVERY_STATE_FIELDS:
            array = getattr(self._data, field_name, None)
            values = snapshot.get(field_name)
            if array is None or values is None:
                continue
            current = np.asarray(array.numpy()).copy()
            current[invalid_mask] = values[invalid_mask]
            _assign_warp_array(array, current)
        mjw.set_const(self._model, self._data)
        return invalid_mask

    def memory_profile(self) -> dict[str, int | str | bool]:
        device = wp.get_preferred_device()
        execution_platform = "cuda" if getattr(device, "is_cuda", False) else str(device)
        state_bytes_estimate = self._estimate_state_bytes()
        bytes_in_use = state_bytes_estimate
        bytes_limit = 0
        mem_info = getattr(device, "mem_info", None)
        if callable(mem_info):
            try:
                free_bytes, total_bytes = mem_info()
                bytes_in_use = max(int(total_bytes) - int(free_bytes), 0)
                bytes_limit = int(total_bytes)
            except Exception:
                bytes_in_use = state_bytes_estimate
                bytes_limit = 0
        self._peak_bytes_in_use = max(self._peak_bytes_in_use, bytes_in_use)
        return {
            "execution_platform": execution_platform,
            "execution_device": getattr(device, "name", str(device)),
            "default_jax_platform": "n/a",
            "default_jax_device": "n/a",
            "device_fallback_applied": False,
            "bytes_in_use": bytes_in_use,
            "peak_bytes_in_use": self._peak_bytes_in_use,
            "bytes_limit": bytes_limit,
            "state_bytes_estimate": state_bytes_estimate,
        }

    def _estimate_state_bytes(self) -> int:
        total = int(self._body_mass_np.nbytes)
        total += self._estimate_object_bytes(self._model)
        total += self._estimate_object_bytes(self._data)
        if self._recovery_snapshot is not None:
            total += sum(int(values.nbytes) for values in self._recovery_snapshot.values())
        return total

    @staticmethod
    def _estimate_object_bytes(obj) -> int:
        total = 0
        seen_arrays: set[int] = set()
        for field_name in dir(obj):
            if field_name.startswith("_"):
                continue
            try:
                value = getattr(obj, field_name)
            except Exception:
                continue
            if callable(value) or id(value) in seen_arrays:
                continue
            numpy_fn = getattr(value, "numpy", None)
            if not callable(numpy_fn):
                continue
            try:
                total += int(np.asarray(numpy_fn()).nbytes)
                seen_arrays.add(id(value))
            except Exception:
                continue
        return total
