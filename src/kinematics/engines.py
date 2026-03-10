from abc import ABC
from abc import abstractmethod

import numpy as np


class StepMethods:
    """
    A static container for different Inverse Kinematics numerical stepping strategies.
    
    These methods calculate the change in joint angles (delta_theta) required to 
    minimize the error between the current end-effector position and the target.
    """

    @staticmethod
    def jacobian_transpose(J: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Calculates the step using the Jacobian Transpose method.
        
        This method is computationally cheap and stable but can be slow to converge.
        It moves the joints in the direction of the gradient of the error.

        Args:
            J (np.ndarray): The Jacobian matrix of shape (3, N) or (6, N).
            e (np.ndarray): The error vector (target - current) of shape (3,) or (6,).

        Returns:
            np.ndarray: The calculated change in joint angles (delta_theta).
        """
        # Calculate the numerator and denominator for the adaptive step size (alpha)
        # alpha = (e^T * J * J^T * e) / ||J * J^T * e||^2
        jjte = J @ J.T @ e
        denom = np.dot(jjte, jjte)
        
        # Avoid division by zero
        if denom < 1e-300:
            return np.zeros(J.shape[1])
            
        alpha = np.dot(e, jjte) / denom
        return alpha * J.T @ e
    
    @staticmethod
    def pseudoinverse(J: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Calculates the step using the Moore-Penrose Pseudoinverse.

        This provides the least-squares solution. It is faster to converge than 
        transpose but suffers from instability near singularities.

        Args:
            J (np.ndarray): The Jacobian matrix.
            e (np.ndarray): The error vector.

        Returns:
            np.ndarray: The calculated change in joint angles.
        """
        return np.linalg.pinv(J) @ e

    @staticmethod
    def DLS(J: np.ndarray, e: np.ndarray, damping_constant: float = 0.1) -> np.ndarray:
        """
        Calculates the step using Damped Least Squares (DLS), also known as Levenberg-Marquardt.

        This method balances between the Pseudoinverse (fast) and Transpose (stable).
        The damping constant prevents unstable velocities near singularities.

        Args:
            J (np.ndarray): The Jacobian matrix.
            e (np.ndarray): The error vector.
            damping_constant (float): The damping factor (lambda). 
                                      Higher = more stable/slower; Lower = faster/unstable.

        Returns:
            np.ndarray: The calculated change in joint angles.
        """
        n = J.shape[0] # Dimension of task space (usually 3 or 6)
        I = np.identity(n)
        
        # Formula: J^T * (J * J^T + lambda^2 * I)^-1 * e
        return J.T @ np.linalg.inv(J @ J.T + damping_constant**2 * I) @ e
    
    @staticmethod
    def SDLS(J: np.ndarray, e: np.ndarray, gamma_max: float = np.pi/4, eps: float = 1e-6) -> np.ndarray:
        """
        Calculates the step using Selectively Damped Least Squares (SDLS).

        This method adjusts damping separately for each singular vector based on the 
        difficulty of reaching the target, offering high stability without manual tuning.
        
        Reference:
            Buss & Kim (2005), "Selectively Damped Least Squares for Inverse Kinematics",
            Equation 3 and 4.

        Args:
            J (np.ndarray): The Jacobian matrix of shape (3*k, n_joints).
            e (np.ndarray): The error vector of shape (3*k,).
            gamma_max (float): Maximum allowed joint change (in radians) per step (paper recommends pi/4).
            eps (float): Threshold for treating singular values as zero (numerical stability).

        Returns:
            np.ndarray: The calculated change in joint angles.
        """
        m, n = J.shape
        num_effectors = m // 3
        
        # 1. Singular Value Decomposition
        # U: (m, m), sigma: (K,), Vt: (K, n) where K = min(m, n)
        U, sigma, Vt = np.linalg.svd(J, full_matrices=False)
        V = Vt.T 
        
        # 2. Vectorized Calculation of 'rho' (Sensitivity of end effectors to joints)
        # Reshape J to (k, 3, n) to compute norms of 3D vectors for each end-effector
        # rho[l, j] = ||d s_l / d theta_j|| 
        # Result rho is shape (num_effectors, n)
        J_reshaped = J.reshape(num_effectors, 3, n)
        rho = np.linalg.norm(J_reshaped, axis=1) 
        
        # 3. Vectorized Calculation of N (Aggregate Joint Velocity)
        # N_i = sum_l ||u_{l, i}|| 
        # Reshape U to (k, 3, m) to sum norms of 3D vectors
        U_reshaped = U.reshape(num_effectors, 3, -1) # -1 handles m or K columns
        N = np.sum(np.linalg.norm(U_reshaped, axis=1), axis=0)
        
        # 4. Vectorized Calculation of M (Aggregate Individual Capability)
        # M_{i,l} = sigma_i^-1 * sum_j (|v_{j,i}| * rho_{l,j}) 
        # We sum rho over all end effectors first -> rho_sum shape (n,)
        rho_sum = np.sum(rho, axis=0)
        
        # M_unscaled[i] = sum_j (|v_ji| * rho_sum[j])
        # This is a matrix-vector product: |V|^T dot rho_sum
        M_unscaled = np.abs(V).T @ rho_sum
        
        # 5. Accumulate Updates
        phi = np.zeros(n)
        
        # Iterate over singular values to calculate selective damping
        for i, s_i in enumerate(sigma):
            # Singularity check: skip directions effectively in the null space
            if s_i < eps:
                continue
                
            inv_sigma = 1.0 / s_i
            
            # M_i = sigma_i^-1 * M_unscaled_i 
            M_i = inv_sigma * M_unscaled[i]
            
            # Calculate damping factor gamma_i 
            # If M_i is tiny (no sensitivity), we default to gamma_max (no damping restriction)
            if M_i < 1e-9:
                gamma_i = gamma_max
            else:
                gamma_i = min(1.0, N[i] / M_i) * gamma_max
                
            # Calculate component update phi_i 
            alpha_i = np.dot(e, U[:, i]) # Component of error along u_i
            w = inv_sigma * alpha_i * V[:, i] # Pure pseudoinverse step for this component
            
            # First clamping stage: Clamp this specific component's contribution
            phi_i = StepMethods._clamp_max_abs(w, gamma_i)
            phi += phi_i

        # 6. Final global clamping
        return StepMethods._clamp_max_abs(phi, gamma_max)
    
    @staticmethod
    def _clamp_max_abs(w: np.ndarray, d: float) -> np.ndarray:
        """
        Implements the 'ClampMaxAbs' function from the SDLS paper.
        
        Scales the vector such that its maximum component magnitude is at most d, 
        preserving the vector's direction.

        NOTE ON TERMINOLOGY:
            The Buss & Kim (2005) paper (Eq. 3 text) refers to this as using the 
            "1-norm", but they explicitly define it as the "maximum of the 
            absolute values of the components". 
            
            Mathematically, the "maximum absolute value" is the Infinity Norm (L_inf),
            NOT the 1-Norm (Sum of absolute values). This implementation uses the 
            Infinity Norm (np.max(np.abs(w))) to match the authors' mathematical 
            definition/intent rather than their confusing naming.

        Args:
            w (np.ndarray): The vector to clamp.
            d (float): The maximum allowed value for any single component.
            
        Returns:
            np.ndarray: The scaled vector.
        """
        # Calculate the Infinity Norm (Maximum Absolute Component)
        norm = np.max(np.abs(w))
        
        # Determine scaling factor.
        # Since 'd' is a magnitude limit (>= 0), the condition 'norm > d' 
        # implies 'norm > 0'. This implicitly prevents division by zero.
        if norm > d:
            scale = d / norm
        else:
            scale = 1.0
            
        return w * scale