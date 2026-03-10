import os

import numpy as np

from src.kinematics import MujocoPoseIK
from src.kinematics import StepMethods
from src.kinematics import solve_IKProblem
from src.utils import load_mujoco_model


class FrankaSmartSolver:
    # Class-level variables to store the loaded model (The Cache)
    _model = None
    _data = None
    _solver_instance: MujocoPoseIK


    @classmethod
    def _initialize(cls):
        """Loads the model once and only once."""
        if cls._model is None:
            # Update this path to your specific robot-only XML
            xml_path = os.path.join("assets", "franka_fr3_v2", "fr3v2.xml")
            
            if not os.path.exists(xml_path):
                raise FileNotFoundError(f"Franka XML not found at {xml_path}")
            
            cls._model, cls._data = load_mujoco_model(xml_path)
            # We initialize the IK wrapper once too
            # Assuming 'gripper' or 'hand_tcp' is the site name in your FR3v2 XML
            cls._solver_instance = MujocoPoseIK(cls._model, cls._data, "gripper", StepMethods.SDLS)

    @classmethod
    def solve(cls, current_joints, target_pose, step_method=StepMethods.SDLS, tolerance=1e-6, max_iterations=500):
        """
        The high-level planning function.
        """
        # Ensure model is loaded in memory
        cls._initialize()

        # Update the math strategy in case the user passed something different
        cls._solver_instance.step_method = step_method

        # Solve the problem using the cached 'Brain'
        # We pass the current joints as the 'seed' for the solver
        goal_qpos = solve_IKProblem(
            cls._solver_instance, 
            current_joints,
            target_pose, 
            tol=tolerance, 
            max_iter=max_iterations
        )
        return goal_qpos