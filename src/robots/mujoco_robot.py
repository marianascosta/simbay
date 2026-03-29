import time

import mujoco
import mujoco.viewer
import numpy as np

from src.utils.tracing import set_span_attributes

from .base import BaseRobot


class MujocoRobot(BaseRobot):
    def __init__(self, model, data, viewer=None):
        set_span_attributes(
            {
                "robot.dt": float(model.opt.timestep),
                "robot.nq": int(model.nq),
                "robot.nv": int(model.nv),
            }
        )
        self.model = model
        self.data = data
        self.viewer: mujoco.viewer.Handle | None = viewer
        self.dt = model.opt.timestep

        self.force_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_force")  # type: ignore
        self.force_adress = None
        if self.force_sensor_id != -1:
            self.force_adress = model.sensor_adr[self.force_sensor_id]

        self.torque_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_torque")  # type: ignore
        self.torque_adress = None
        if self.torque_sensor_id != -1:
            self.torque_adress = model.sensor_adr[self.torque_sensor_id]


    def move_joints(self, pos):
        set_span_attributes(
            {
                "robot.command_dim": int(np.asarray(pos).shape[0]),
            }
        )
        # Command control and update model
        self.data.ctrl[:8] = pos
        mujoco.mj_step(self.model, self.data)  # type: ignore

        # Safely sync the viewer if it exists
        if self.viewer is not None:
            self.viewer.sync()

    def move_gripper(self, width):
        """Controls the gripper separately."""
        self.data.ctrl[7] = width * 255 / 0.08
        mujoco.mj_step(self.model, self.data)  # type: ignore
        if self.viewer is not None:
            self.viewer.sync()

    def get_pos(self):
        return self.data.qpos[:7]

    def get_sensor_reads(self):
        if self.force_adress is not None:
            force = self.data.sensordata[self.force_adress : self.force_adress + 3]
        else:
            force = np.zeros(3)

        if self.torque_adress is not None:
            torque = self.data.sensordata[self.torque_adress : self.torque_adress + 3]
        else:
            torque = np.zeros(3)

        set_span_attributes(
            {
                "robot.force_address": int(self.force_adress) if self.force_adress is not None else -1,
                "robot.torque_address": int(self.torque_adress) if self.torque_adress is not None else -1,
            }
        )
        return np.concatenate([force, torque])

    def wait_seconds(self, duration):
        set_span_attributes({"robot.wait_duration": float(duration)})
        steps = int(duration / self.dt)
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)  # type: ignore
            if self.viewer is not None:
                self.viewer.sync()
