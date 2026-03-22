"""
MuJoCo Warp batch backend.

This module owns the nworld-batched simulation state for all N particles.
It replaces the JAX-based MJXBatch for the warp backend path.
"""

from __future__ import annotations

import logging

import numpy as np
import mujoco_warp as mjw
import warp as wp


logger = logging.getLogger("simbay.warp_batch")

_RESAMPLE_STATE_FIELDS = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")


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
        njmax: int = 256,
    ):
        self._body_id = body_id
        self._size = int(len(masses))
        self._ctrl_dim = int(mj_model.nu)

        logger.info(
            "warp_batch_init nworld=%d body_id=%d nconmax=%d njmax=%d",
            self._size,
            body_id,
            nconmax,
            njmax,
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
        self._model.body_mass = wp.from_numpy(base_mass_np, dtype=wp.float32)
        mjw.set_const(self._model, self._data)

        logger.info("warp_batch_init_complete nworld=%d", self._size)

    @property
    def ctrl_dim(self) -> int:
        return self._ctrl_dim

    def warmup(self) -> None:
        dummy_ctrl = np.zeros((self._size, self._ctrl_dim), dtype=np.float32)
        _assign_warp_array(self._data.ctrl, dummy_ctrl)
        mjw.step(self._model, self._data)
        wp.synchronize()
        logger.info("warp_batch_warmup_done nworld=%d", self._size)

    def warmup_rollout(self, steps: int) -> None:
        if steps <= 0:
            return
        dummy_ctrl = np.zeros((self._size, self._ctrl_dim), dtype=np.float32)
        _assign_warp_array(self._data.ctrl, dummy_ctrl)
        for _ in range(steps):
            mjw.step(self._model, self._data)
        wp.synchronize()
        logger.info("warp_batch_rollout_warmup_done steps=%d", steps)

    def step(self, control_input: np.ndarray, masses: np.ndarray) -> None:
        self._body_mass_np[:, self._body_id] = np.asarray(masses, dtype=np.float32)
        _assign_warp_array(self._model.body_mass, self._body_mass_np)

        ctrl_np = np.broadcast_to(
            np.asarray(control_input, dtype=np.float32),
            (self._size, self._ctrl_dim),
        ).copy()
        _assign_warp_array(self._data.ctrl, ctrl_np)

        mjw.step(self._model, self._data)

    def rollout(self, control_inputs: np.ndarray, mass_trajectory: np.ndarray) -> None:
        steps = int(control_inputs.shape[0])
        if steps <= 0:
            return
        for step_idx in range(steps):
            self.step(control_inputs[step_idx], mass_trajectory[step_idx])

    def sensor_slice(self, start: int, width: int) -> np.ndarray:
        return self._data.sensordata.numpy()[:, start : start + width]

    def resample(self, indexes: np.ndarray) -> None:
        indexes_np = np.asarray(indexes, dtype=np.int32)

        for field_name in _RESAMPLE_STATE_FIELDS:
            array = getattr(self._data, field_name, None)
            if array is None:
                continue
            np_array = array.numpy()
            _assign_warp_array(array, np_array[indexes_np])

        self._body_mass_np = self._body_mass_np[indexes_np]
        _assign_warp_array(self._model.body_mass, self._body_mass_np)
        mjw.set_const(self._model, self._data)

    def memory_profile(self) -> dict[str, int | str | bool]:
        device = wp.get_preferred_device()
        execution_platform = "cuda" if getattr(device, "is_cuda", False) else str(device)
        return {
            "execution_platform": execution_platform,
            "execution_device": getattr(device, "name", str(device)),
            "default_jax_platform": "n/a",
            "default_jax_device": "n/a",
            "device_fallback_applied": False,
            "bytes_in_use": 0,
            "peak_bytes_in_use": 0,
            "bytes_limit": 0,
        }
