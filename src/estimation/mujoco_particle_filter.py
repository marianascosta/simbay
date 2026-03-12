import mujoco
import numpy as np

from src.robots import MujocoRobot
from src.utils import DEFAULT_OBJECT_PROPS
from src.utils import initialize_mujoco_env

from .base import ParticleEnvironment

""" Mass only for now"""

class FrankaMuJoCoEnv(ParticleEnvironment):
    def __init__(self, limits: tuple[float, float], num_particles: int):
        self.min, self.max = limits
        self.robots: list[MujocoRobot] = []

        self._num_particles = num_particles
        
        self.std_dev = 0.005 # This is our process noise (Q).

    @property
    def num_particles(self) -> int:
        return self._num_particles

    def initialize_particles(self) -> np.ndarray:
        # Generate the initial uniform guesses for the block's mass
        masses = np.random.uniform(self.min, self.max, size=self.num_particles)
        
        for mass in masses:
            # CRITICAL: Use .copy() so each robot gets a unique dictionary!
            object_properties = DEFAULT_OBJECT_PROPS.copy()
            object_properties['mass'] = mass    
            
            robot = initialize_mujoco_env(object_properties)
            self.robots.append(robot)

        # Cache the C++ memory ID for the block (they are all identical models, 
        # so the ID is the same for all 100 robots)
        self.block_body_id = mujoco.mj_name2id( # type: ignore
            self.robots[0].model, 
            mujoco.mjtObj.mjOBJ_BODY,  # type: ignore
            "object"
        )

        return masses

    def memory_profile(self) -> dict[str, int | float]:
        """
        Return the MuJoCo memory owned by the particle robots.
        """
        if not self.robots:
            return {
                "robot_count": 0,
                "model_nbuffer_bytes_per_robot": 0,
                "data_nbuffer_bytes_per_robot": 0,
                "data_narena_bytes_per_robot": 0,
                "native_bytes_per_robot": 0,
                "native_bytes_total": 0,
            }

        model = self.robots[0].model
        data = self.robots[0].data
        model_nbuffer = int(model.nbuffer)
        data_nbuffer = int(data.nbuffer)
        data_narena = int(data.narena)
        native_bytes_per_robot = model_nbuffer + data_nbuffer + data_narena

        return {
            "robot_count": len(self.robots),
            "model_nbuffer_bytes_per_robot": model_nbuffer,
            "data_nbuffer_bytes_per_robot": data_nbuffer,
            "data_narena_bytes_per_robot": data_narena,
            "native_bytes_per_robot": native_bytes_per_robot,
            "native_bytes_total": native_bytes_per_robot * len(self.robots),
        }
    
    def resample_states(self, indexes: np.ndarray) -> None:
        self.robots = [self.robots[i] for i in indexes]

    def propagate(self, particles: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        # 1. Apply process noise to the mathematical state (the Artificial Random Walk)
        particles += np.random.normal(0.0, self.std_dev, self.num_particles)
        np.clip(particles, self.min, self.max, out=particles)

        particles = np.clip(particles, self.min, self.max)

        for i, robot in enumerate(self.robots):
            # 1. Directly inject the new mass using the cached ID
            robot.model.body_mass[self.block_body_id] = particles[i]

            robot.move_joints(control_input)

        return particles

    def compute_likelihoods(self, particles: np.ndarray, observation: np.ndarray) -> np.ndarray:
        # 1. OPTIMIZATION: List comprehension is much faster than a standard for-loop append
        sim_z = np.array([robot.get_sensor_reads() for robot in self.robots])
        
        # 2. Calculate the difference (Error) 
        diff = observation - sim_z
        
        R = 1.0
        
        # 3. OPTIMIZATION: Calculate the squared distance once
        dist_sq = np.sum(diff ** 2, axis=1)
        
        # 4. Calculate likelihood
        likelihoods = np.exp(-0.5 * dist_sq / R)
        
        return likelihoods
        

    

