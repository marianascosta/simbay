from typing import Callable
from typing import Tuple

import mujoco
import numpy as np

from src.utils.tracing import set_span_attributes
from src.utils.tracing import trace_public_methods

# Local imports
from .ik_solver import IKProblem


@trace_public_methods("simbay.mujoco_ik")
class MujocoPoseIK(IKProblem):
    """
    A MuJoCo-specific implementation of the IKProblem interface for 6-DOF tasks.
    
    This class handles the Forward Kinematics and Jacobian calculations required
    to solve for both **Position and Orientation** (Pose) of a single end-effector site.
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, site_name: str, step_method: Callable): # type: ignore
        """
        Initialize the 6D IK wrapper.

        Args:
            model (mujoco.MjModel): The static MuJoCo model definition.
            data (mujoco.MjData): The dynamic MuJoCo simulation state.
            site_name (str): The name of the target site in the XML (e.g., 'grasp_center').
            step_function (Callable): The numerical strategy (e.g., SDLS) to calculate joint deltas.
        """
        set_span_attributes(
            {
                "ik.site_name": site_name,
                "ik.nv": int(model.nv),
                "ik.step_method": getattr(step_method, "__name__", str(step_method)),
            }
        )
        self.model = model
        self.data = data
        self.step_method = step_method

        # Robust ID lookup
        try:
            self.site_id = model.site(site_name).id
        except KeyError:
            raise ValueError(f"Site '{site_name}' not found in model.")

        # Performance Optimization: Pre-allocate 6D Jacobian 
        # Rows 0-2: Translational (Linear) Jacobian
        # Rows 3-5: Rotational (Angular) Jacobian
        self.J = np.zeros((6, self.model.nv))
        
        # Internal buffers (Views) for mj_jacSite
        self.jac_pos = self.J[:3, :] # View of top 3 rows
        self.jac_rot = self.J[3:, :] # View of bottom 3 rows
        

    def get_s(self, theta: np.ndarray) -> np.ndarray:
        """
        Sets the robot joints to 'theta' and returns the current end-effector Pose.
        
        Args:
            theta (np.ndarray): The joint configuration to test.
            
        Returns:
            np.ndarray: A 7D vector [x, y, z, qw, qx, qy, qz] representing 
                        Position (meters) and Orientation (Quaternion).
        """
        set_span_attributes({"ik.theta_dim": int(theta.shape[0])})
        self.data.qpos[:] = theta
        mujoco.mj_forward(self.model, self.data) # type: ignore # Propagate kinematics
        
        # 1. Get Position (3,)
        pos = self.data.site_xpos[self.site_id]
        
        # 2. Get Orientation (9,) -> Convert to Quat (4,)
        # Note: Sites store orientation as a Rotation Matrix (site_xmat), not a Quaternion.
        mat = self.data.site_xmat[self.site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat) # type: ignore
        
        # Return combined 7D vector
        return np.concatenate([pos, quat])
    
    def get_e(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Computes the 6D error vector between current and target pose.
        Position: targ_pos - curr_pos
        Orientation: rotation vector (axis * angle) that rotates current -> target
        """
        set_span_attributes(
            {
                "ik.state_dim": int(s.shape[0]),
                "ik.target_dim": int(t.shape[0]),
            }
        )
        # 1. Unpack and Enforce Contiguity
        curr_pos = s[:3]
        curr_quat = np.ascontiguousarray(s[3:], dtype=np.float64)
        
        targ_pos = t[:3]
        targ_quat = np.ascontiguousarray(t[3:], dtype=np.float64)
        
        # 2. Position Error (Already in World Frame)
        e_pos = targ_pos - curr_pos
        
        # 3. Orientation Error (Axis-Angle)
        # mju_subQuat gives the error in the LOCAL frame of the site
        e_rot_local = np.zeros(3, dtype=np.float64)
        mujoco.mju_subQuat(e_rot_local, targ_quat, curr_quat) # type: ignore
        
        # CRITICAL FIX: Convert Local Error -> World Error
        # We multiply the site's rotation matrix by the local error vector.
        # error_world = rotacional_matrix_object * errol_local
        # This can be viewed as referential change

        orientation = self.data.site_xmat # this returns a (1, 9) matrix
        R = orientation.reshape(3, 3) # so we reshape to (3, 3) (rotational matrix)
        
        e_rot_global = np.zeros(3, dtype=np.float64)
        mujoco.mju_mulMatVec(e_rot_global, R, e_rot_local) # type: ignore

        # Combine into 6D error vector
        return np.concatenate([e_pos, e_rot_global])
    
    def get_jacobian(self, s: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Computes the full 6D Jacobian (6 x nv) for the end-effector site.
        
        Returns:
            np.ndarray: Matrix where top 3 rows are linear velocity derivatives
                        and bottom 3 rows are angular velocity derivatives.
        """
        set_span_attributes(
            {
                "ik.state_dim": int(s.shape[0]),
                "ik.theta_dim": int(theta.shape[0]),
            }
        )
        # Note: mj_jacSite writes directly into our pre-allocated views (jac_pos, jac_rot)
        # which automatically updates self.J.
        
        # We assume robot state is already updated by get_s() call in the loop.
        # But for safety in other contexts, we enforce it:
        self.data.qpos[:] = theta
        mujoco.mj_forward(self.model, self.data) # type: ignore
        
        mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site_id) # type: ignore
        return self.J
    
    def step(self, J: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Delegates the mathematical step calculation to the chosen strategy."""
        set_span_attributes(
            {
                "ik.jacobian_rows": int(J.shape[0]),
                "ik.jacobian_cols": int(J.shape[1]),
                "ik.error_dim": int(e.shape[0]),
            }
        )
        return self.step_method(J, e)

    def clamp_to_limits(self, theta: np.ndarray) -> np.ndarray:
        """Clamps joint angles to the physical limits defined in the XML."""
        set_span_attributes({"ik.theta_dim": int(theta.shape[0])})
        for j in range(self.model.nv):
            if self.model.jnt_limited[j]:
                theta[j] = np.clip(
                    theta[j], 
                    self.model.jnt_range[j, 0], 
                    self.model.jnt_range[j, 1]
                )
        return theta
