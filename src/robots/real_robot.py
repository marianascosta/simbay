# type: ignore
import logging
import time

import numpy as np

from .base import BaseRobot
from src.utils.logging_utils import extend_logging_data


logger = logging.getLogger("simbay.real_robot")

# --- CONDITIONAL IMPORTS ---
try:
    import rclpy
    from builtin_interfaces.msg import Duration  # Helper for time
    from sensor_msgs.msg import JointState  # Added this!
    from std_msgs.msg import Header
    from trajectory_msgs.msg import JointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.warning(
        {
            "event": "ros_libraries_unavailable",
            "message": "RealRobot will not work, but SimRobot is fine.",
        }
    )



class RealRobot(BaseRobot):
    def __init__(self):
        if not ROS_AVAILABLE:
            raise ImportError("Cannot use RealRobot: ROS libraries are missing!")
        
        # 1. Setup
        self.joint_names = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3", 
            "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"
        ]
        self.current_joints = None
        
        # 2. Connect
        self._setup_ros()

    def _setup_ros(self):
        if not rclpy.ok():
            rclpy.init()
        self.node1 = rclpy.create_node("simbay_pick_test_node")
        
        self.pub1 = self.node1.create_publisher(JointTrajectory, "/fr3_arm_controller/joint_trajectory", 10)
        self.sub1 = self.node1.create_subscription(JointState, "/joint_states", self.jointstate_callback, 10)
        
        logger.info({"event": "robot_connection_waiting"})
        while self.current_joints is None:
            rclpy.spin_once(self.node1, timeout_sec=0.1)
        logger.info({"event": "robot_connected"})

    def jointstate_callback(self, msg):
        if len(msg.name) >= 7:
            self.current_joints = list(msg.position[:7])

    def get_pos(self):
        # Must update the spinner to get the freshest data
        rclpy.spin_once(self.node1, timeout_sec=0.01)
        return np.array(self.current_joints)

    def move_to_pos(self, target_pos, max_velocity=0.5):
        """
        ROS Implementation: Calculates duration and sends a Trajectory Message.
        """
        # 1. Calculate Duration (Same math as SimRobot)
        start_pos = self.get_pos()
        distance = np.abs(target_pos - start_pos)
        duration = np.max(distance / max_velocity)
        
        if duration < 0.1:
            duration = 0.5 # Minimum safety time for real hardware

        # 2. Construct ROS Message
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = list(target_pos) # Convert numpy to list
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        
        msg.points = [point]

        # 3. Send and Wait
        self.pub1.publish(msg)
        logger.info(
            {
                "event": "real_robot_move_to_pos",
                "duration_s": duration,
            }
        )
        
        # Blocking wait (Sleep) to match SimRobot behavior
        time.sleep(duration + 0.1) 

    def open_gripper(self):
        logger.info(
            {
                "event": "mock_gripper_open",
                "hardware_online": False,
            }
        )
        time.sleep(0.5)

    def close_gripper(self):
        logger.info(
            {
                "event": "mock_gripper_close",
                "hardware_online": False,
            }
        )
        time.sleep(0.5)
