import os

import numpy as np

from src.kinematics import MujocoPoseIK
from src.kinematics import StepMethods
from src.kinematics import solve_IKProblem
from src.utils.tracing import set_span_attributes
from src.utils.tracing import trace_call
from src.utils.mujoco_utils import load_mujoco_model


class FrankaSmartSolver:
    # Class-level variables to store the loaded model (The Cache)
    _model = None
    _data = None
    _solver_instance: MujocoPoseIK

    @classmethod
    @trace_call("simbay.franka_solver", "planning.FrankaSmartSolver._initialize")
    def _initialize(cls):
        """Loads the model once and only once."""
        set_span_attributes({"planning.cache_hit": cls._model is not None})
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
    @trace_call("simbay.franka_solver", "planning.FrankaSmartSolver.solve")
    def solve(cls, current_joints, target_pose, step_method=StepMethods.SDLS, tolerance=1e-6, max_iterations=500):
        """
        The high-level planning function.
        """
        set_span_attributes(
            {
                "planning.theta_dim": int(np.asarray(current_joints).shape[0]),
                "planning.target_dim": int(np.asarray(target_pose).shape[0]),
                "planning.tolerance": float(tolerance),
                "planning.max_iterations": int(max_iterations),
                "planning.step_method": getattr(step_method, "__name__", str(step_method)),
            }
        )
        # Ensure model is loaded in memory
        cls._initialize()

        # Update the math strategy in case the user passed something different
        cls._solver_instance.step_method = step_method

        # Solve the problem using the cached 'Brain'
        # We pass the current joints as the 'seed' for the solver
        goal_qpos = solve_IKProblem(
            cls._solver_instance, current_joints, target_pose, tol=tolerance, max_iter=max_iterations
        )
        return goal_qpos
