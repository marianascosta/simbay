"""
Global constants and default physics properties for the Simbay project.
"""

import numpy as np

# Default properties for objects spawned in simulation (Cube, etc.)
DEFAULT_OBJECT_PROPS = {
    "type":     "box",                      
    "pos":      (0.3, 0.1, 0.022),           
    "size":     (0.02, 0.02, 0.02),      
    "mass":     2.50,                        
    "friction": (5.0, 0.5, 0.001)  # Lateral, Spinning, Rolling
}

FRANKA_HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])