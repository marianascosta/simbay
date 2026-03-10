import time

import mujoco
import mujoco.viewer
import numpy as np

from .base import BaseRobot


class MujocoRobot(BaseRobot):
    def __init__(self, model, data, viewer=None):
        self.model = model
        self.data = data
        self.viewer: mujoco.viewer.Handle | None = viewer
        self.dt = model.opt.timestep

        self.force_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_force") # type: ignore
        self.force_adress = model.sensor_adr[self.force_sensor_id]
        
    def move_joints(self, pos):
        # Command control and update model
        self.data.ctrl[:8] = pos              
        mujoco.mj_step(self.model, self.data)  # type: ignore
        
        # Safely sync the viewer if it exists
        if self.viewer is not None:
            self.viewer.sync()                    

        
    def get_pos(self):
        return self.data.qpos[:7]
    
    def get_sensor_reads(self):
        return self.data.sensordata[self.force_adress : self.force_adress + 3]

    
    def wait_seconds(self, duration):
        steps = int(duration / self.dt)  
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)     # type: ignore
            if self.viewer is not None:
                self.viewer.sync() 
