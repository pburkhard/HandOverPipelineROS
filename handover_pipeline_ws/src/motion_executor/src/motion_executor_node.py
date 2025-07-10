#!/usr/bin/env python
from actionlib import SimpleActionClient
from math import cos, dist, fabs
import moveit_commander
import numpy as np
import os
import rospy
import sys
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_slerp
from tf import TransformListener
from typing import List
import yaml


from controller_manager_msgs.srv import SwitchController, UnloadController, LoadController
from franka_gripper.msg import MoveAction, MoveGoal
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import PositionIKRequest
from moveit_msgs.srv import GetPositionIK
from visualization_msgs.msg import Marker
from std_msgs.msg import Empty, String

# Absolute path to the root directory of the package
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MotionExecutor:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.is_executing = False  # Whether the robot is currently executing a motion

        moveit_commander.roscpp_initialize(sys.argv)  # Initialize MoveIt! commander
        rospy.init_node(self.cfg['ros']['node_name'], anonymous=True)

        self.target_pose_history = []  # History of target poses received from the subscribed topic
        self.measured_pose = None  # Measured pose from the subscribed topic

        # MoveIt! interface
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("panda_manipulator", wait_for_servers=15)
        self.gripper = moveit_commander.MoveGroupCommander("panda_hand", wait_for_servers=15)

        # Set the output directory
        self.out_dir = None
        if self.cfg['debug']['out_dir_mode'] == "fixed":
            self.out_dir = self.cfg['debug']['out_dir_fixed']
        elif self.cfg['debug']['out_dir_mode'] == "topic":
            self._out_dir_sub = rospy.Subscriber(
                self.cfg['debug']['out_dir_topic'],
                String,
                self._out_dir_callback,
                queue_size=1,
            )
            while self.out_dir is None and not rospy.is_shutdown():
                rospy.loginfo(
                    "Waiting for output directory to be set via topic: "
                    + f"{self.cfg['debug']['out_dir_topic']}"
                )
                rospy.sleep(1.0)
        else:
            rospy.logerr(
                "Invalid out_dir_mode. Supported modes are 'fixed' and 'topic'."
            )

        # Log the config
        if self.cfg['debug']['log_config']:
            config_path = os.path.join(self.out_dir, "(me)_config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(self.cfg, f)

        self.load_scene()
        self.group.set_planning_time(self.cfg['planning_timeout'])
        self.group.set_max_velocity_scaling_factor(0.15)
        self.group.set_max_acceleration_scaling_factor(0.15)
        # self.group.set_planner_id("TRRT")  # RRT
        self.group.set_planning_pipeline_id("ompl")
        self.group.set_goal_position_tolerance(0.01)

        self.tf_listener = TransformListener(100)
        self.marker_publisher = rospy.Publisher(
            self.cfg['ros']['published_topics']['marker'],
            Marker,
            queue_size=1,
        )
        self.pose_pubisher = rospy.Publisher(
            self.cfg['ros']['published_topics']['target_pose'],
            PoseStamped,
            queue_size=1,
        )  # Used in phase 2
        self.target_subscriber = rospy.Subscriber(
            self.cfg['ros']['subscribed_topics']['target_pose'],
            PoseStamped,
            self.target_callback,
            queue_size=1,
        )  # Used in phase 2
        self.pose_subscriber = rospy.Subscriber(
            self.cfg['ros']['subscribed_topics']['measured_pose'],
            PoseStamped,
            self.measured_pose_callback,
            queue_size=1,
        )  # Used in phase 2
        self.trigger_subscriber = rospy.Subscriber(
            self.cfg['ros']['subscribed_topics']['trigger'],
            Empty,
            self.trigger_callback,
            queue_size=1,
        )  # Used to trigger the whole execution
        self.gripper_client = SimpleActionClient(self.cfg['ros']['actions']['gripper_control'], MoveAction) # Used to open the gripper after phase 2
        self.controller_switch_client = rospy.ServiceProxy(
            self.cfg['ros']['services']['switch_controller'], SwitchController
        )  # Used to switch the controllers between phases
        self.controller_unload_client = rospy.ServiceProxy(
            self.cfg['ros']['services']['unload_controller'], UnloadController
        )  # Used to switch the controllers between phases
        self.controller_load_client = rospy.ServiceProxy(
            self.cfg['ros']['services']['load_controller'], LoadController
        )  # Used to switch the controllers between phases

        self.compute_ik = rospy.ServiceProxy("/compute_ik", GetPositionIK)
        self.upper_limit = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.lower_limit = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])

        rospy.loginfo(f"{self.cfg['ros']['node_name']} node started.")

    def phase1(self, target_pose: PoseStamped):
        rospy.loginfo("Executing Phase 1: Moving to target pose.")

        eef_link = self.group.get_end_effector_link()
        rospy.loginfo("============ End effector link: %s" % eef_link)

        joint_vals = self.group.get_current_joint_values()
        rospy.loginfo(joint_vals)

        rospy.loginfo(f"Target pose:\n{target_pose}")

        ik_result = self.computeIK(target_pose)
        rospy.loginfo("Is solvable: %s" % ik_result)

        self._mark_grasp(target_pose, 0, self.cfg['ros']['frames']['camera'])

        # intermediate_pose = target_pose  TODO: Remove this
        intermediate_pose = self.group.get_current_pose()
        intermediate_pose.pose.position.z += 0.2  # Move up to avoid collisions
        self.group.set_pose_target(intermediate_pose)
        success = self.group.go(wait=True)

        self.group.set_pose_target(target_pose)
        success = self.group.go(wait=True)

        if not success:
            rospy.logerr("Failed to move to target pose.")
            return
        
        self.group.stop()
        self.group.clear_pose_targets()
        rospy.loginfo("Phase 1 completed.")

    def phase2(self, target_pose: PoseStamped):
        rospy.loginfo(f"Executing Phase 2: Moving to target pose\n{target_pose}")

        while not rospy.is_shutdown() and self.measured_pose is None:
            rospy.logwarn("No measured pose available. Trying again in 1 second.")
            rospy.sleep(1.0)
        
        rospy.loginfo(f"Starting to move.")
        delta_pose = self.cfg['phase_2_max_vel'] / self.cfg['phase_2_rate']
        delta_angle = self.cfg['phase_2_max_rot_vel'] / self.cfg['phase_2_rate']

        # 1. Move the gripper to the target pose using impedance control
        while not rospy.is_shutdown() and not self.all_close(
            target_pose.pose, self.measured_pose.pose, self.cfg['target_tolerance']
        ):  
            if not self.cfg['fix_target_pose']:
                target_pose = self.average_poses(self.target_pose_history)
            rospy.loginfo(f"Distance to target pose: {self.distance(target_pose, self.measured_pose)}")

            target_equilibrium_pose = self.limit_pose(
                start_pose=self.measured_pose,
                end_pose=target_pose,
                max_distance_to_start_pose=delta_pose,
                max_angle_to_start_pose=delta_angle
            )

            rospy.loginfo(f"Moving distance of {self.distance(target_equilibrium_pose, self.measured_pose)} to target pose.")

            self._mark_grasp(target_equilibrium_pose, 0, self.cfg['ros']['frames']['camera'])

            # Update the target pose
            self.pose_pubisher.publish(target_equilibrium_pose)
            rospy.sleep(0.02)

        # 2. Open the gripper
        rospy.loginfo("Phase 2 completed.")

    def open_gripper(self):
        """Open the gripper using the gripper action client.
        """
        rospy.loginfo("Opening the gripper.")
        goal = MoveGoal(width=0.05 * 2, speed=1.0)
        self.gripper_client.send_goal(goal)
        self.gripper_client.wait_for_result(timeout=rospy.Duration(2.0))
        rospy.loginfo("Gripper opened phase 2 terminated.")


    def switch_controllers(self, start_controller: str, stop_controller: str):
        """Switch the controllers using the controller manager service.
        Args:
            start_controller (str): The type of the controller to start.
            stop_controller (str): The type of the controller to stop.
        """
        # At the moment, we only support switching from the joint trajectory controller to the impedance controller
        # assert start_controller == "cartesian_impedance_example_controller" and stop_controller == "position_joint_trajectory_controller"

        rospy.loginfo(f"Switching controllers: {start_controller} -> {stop_controller}")
        try:
            # Stop the old controller
            self.controller_switch_client(
                start_controllers=[],
                stop_controllers=[stop_controller],
                start_asap=True,
            )
            # Unload the old controller
            self.controller_unload_client(
                name=stop_controller,
            )
            # Load the new controller
            self.controller_load_client(
                name=start_controller,
            )
            # Start the new controller
            self.controller_switch_client(
                start_controllers=[start_controller],
                stop_controllers=[],
                start_asap=True,
            )
            rospy.loginfo("Controllers switched successfully.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to switch controllers: {e}")


    def load_scene(self):
        """Load the scene in the MoveIt! planning scene.
        """
        wall_pose = self.get_stamped_pose([1.7, 0.0, +0.005], [0, 0, 0, 1], "panda_link0")  # was - 0.035 ( cube is 4.2 cm) tried with 0.015 and was working for pan
        self.scene.add_box("floor", wall_pose, size=(3, 3, 0))
        pole_pose = self.get_stamped_pose([0.09, -0.4, 0.2], [0, 0, 0, 1], "panda_link0")
        self.scene.add_box("pole", pole_pose, size=(0.08, 0.11, 0.46))

    ######################
    # Callback Functions #
    ######################

    def measured_pose_callback(self, measured_pose: PoseStamped):
        """ Callback function for the measured pose subscriber.
        """
        self.measured_pose = measured_pose

    def target_callback(self, target_pose: PoseStamped):
        """ Callback function for the target pose subscriber.
        """

        # Update the target pose history if not in fix_target_pose mode and already moving
        if not self.cfg['fix_target_pose'] or not self.is_executing:
            self.target_pose_history.append(target_pose)
            if len(self.target_pose_history) > self.cfg['average_over_n_poses']:
                self.target_pose_history.pop(0)
            rospy.loginfo("Target pose received and added to history.")

    def trigger_callback(self, msg: Empty):
        """ Callback function for the trigger subscriber.
        """
        if self.is_executing:
            rospy.logwarn("Already executing. Ignoring trigger.")
            return

        while not rospy.is_shutdown() and len(self.target_pose_history) < self.cfg['average_over_n_poses']:
            rospy.logwarn(
                f"Expected {self.cfg['average_over_n_poses']} poses, "
                f"but got {len(self.target_pose_history)}. Trying again in 1 second."
            )
            rospy.sleep(1.0)
        
        self.is_executing = True
        target_pose = self.average_poses(self.target_pose_history)

        # Execute Phase 1 (go to a pose close to the target pose)
        target_pose_phase1 = self.limit_pose(
            start_pose=target_pose,
            end_pose=self.group.get_current_pose(),
            max_distance_to_start_pose=self.cfg['phase_1_target_distance']
        )
        self.phase1(target_pose_phase1)

        # Switch controllers to impedance control
        self.switch_controllers(
            start_controller="cartesian_impedance_example_controller",
            stop_controller="position_joint_trajectory_controller",
        )

        # Execute Phase 2 (impedance control to the target pose)
        self.phase2(target_pose)

        # Switch back to joint trajectory controller
        self.switch_controllers(
            start_controller="position_joint_trajectory_controller",
            stop_controller="cartesian_impedance_example_controller",
        )

        # Open the gripper
        self.open_gripper()

        rospy.loginfo("Motion execution completed.")

    def _out_dir_callback(self, msg: String):
        """
        Callback function for the output directory topic subscriber.
        Sets the output directory based on the received message.
        Args:
            msg (String): The message containing the output directory path.
        """
        if self.out_dir != msg.data:
            self.out_dir = msg.data
            rospy.loginfo(f"Output directory set to: {self.out_dir}")


    #####################
    # Utility Functions #
    #####################
    def all_close(self, goal, actual, tolerance):
        """
        Adapted from offical moveit! tutorial: https://github.com/moveit/moveit_tutorials/blob/master/doc/move_group_python_interface/scripts/move_group_python_interface_tutorial.py
        Convenience method for testing if the values in two lists are within a tolerance of each other.
        For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
        between the identical orientations q and -q is calculated correctly).
        @param: goal       A list of floats, a Pose or a PoseStamped
        @param: actual     A list of floats, a Pose or a PoseStamped
        @param: tolerance  A float
        @returns: bool
        """
        if type(goal) is list:
            for index in range(len(goal)):
                if abs(actual[index] - goal[index]) > tolerance:
                    return False

        elif type(goal) is PoseStamped:
            return self.all_close(goal.pose, actual.pose, tolerance)

        elif type(goal) is Pose:
            x0 = actual.position.x
            y0 = actual.position.y
            z0 = actual.position.z
            qx0 = actual.orientation.x
            qy0 = actual.orientation.y
            qz0 = actual.orientation.z
            qw0 = actual.orientation.w
            x1 = goal.position.x
            y1 = goal.position.y
            z1 = goal.position.z
            qx1 = goal.orientation.x
            qy1 = goal.orientation.y
            qz1 = goal.orientation.z
            qw1 = goal.orientation.w
            # Euclidean distance
            d = dist((x1, y1, z1), (x0, y0, z0))
            # phi = angle between orientations
            cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
            return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

        return True
    
    def computeIK(self, pose_stamped, ik_link_name="panda_hand_tcp", move_group="panda_manipulator", is_grasps=True) -> bool:
        """Check if a given pose is reachable for the robot. Return True if it is, False otherwise."""

        # Create a pose to compute IK for
        self.group.set_goal_position_tolerance(0.001)
        ik_request = PositionIKRequest()
        if not is_grasps:
            self.group.set_goal_position_tolerance(0.05)

        ik_request.group_name = move_group
        ik_request.ik_link_name = ik_link_name
        ik_request.pose_stamped = pose_stamped
        ik_request.robot_state = self.robot.get_current_state()
        ik_request.avoid_collisions = True

        request_value = self.compute_ik(ik_request)

        if request_value.error_code.val == -31:
            return False

        if request_value.error_code.val == 1:
            # Check if the IK is at the limits
            joint_positions = np.array(request_value.solution.joint_state.position[:7])
            upper_diff = np.min(np.abs(joint_positions - self.upper_limit))
            lower_diff = np.min(np.abs(joint_positions - self.lower_limit))
            return min(upper_diff, lower_diff) > 0.1
        else:
            return False
    
    def distance(self, pose1: PoseStamped, pose2: PoseStamped):
        """
        Calculate the Euclidean distance between two PoseStamped objects.
        """
        assert pose1.header.frame_id == pose2.header.frame_id
        
        dx = pose1.pose.position.x - pose2.pose.position.x
        dy = pose1.pose.position.y - pose2.pose.position.y
        dz = pose1.pose.position.z - pose2.pose.position.z
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def angle(self, pose1: PoseStamped, pose2: PoseStamped):
        """
        Calculate the angle between the orientations of two PoseStamped objects.
        The angle is calculated as the angle between the quaternions.
        """
        assert pose1.header.frame_id == pose2.header.frame_id
        
        quat1 = (pose1.pose.orientation.x, pose1.pose.orientation.y, pose1.pose.orientation.z, pose1.pose.orientation.w)
        quat2 = (pose2.pose.orientation.x, pose2.pose.orientation.y, pose2.pose.orientation.z, pose2.pose.orientation.w)
        
        dot_product = np.dot(quat1, quat2)
        return 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    def average_poses(self, poses: List[PoseStamped]):
        """
        """
        if not poses:
            rospy.logerr("No poses to average.")
            return None
        
        frame_id = poses[0].header.frame_id
        if not all(p.header.frame_id == frame_id for p in poses):
            rospy.logerr("All poses must have the same frame_id to be averaged.")
            return None

        avg_pose = PoseStamped()
        avg_pose.pose.position.x = np.mean([p.pose.position.x for p in poses])
        avg_pose.pose.position.y = np.mean([p.pose.position.y for p in poses])
        avg_pose.pose.position.z = np.mean([p.pose.position.z for p in poses])

        # Average over the orientations in euler space
        euler_angles = []
        for pose in poses:
            quat = pose.pose.orientation
            euler = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
            euler_angles.append(euler)
        euler_angles_avg = np.array(euler_angles).mean(axis=0)
        quat_avg = quaternion_from_euler(euler_angles_avg[0], euler_angles_avg[1], euler_angles_avg[2])

        avg_pose.pose.orientation.x = quat_avg[0]
        avg_pose.pose.orientation.y = quat_avg[1]
        avg_pose.pose.orientation.z = quat_avg[2]
        avg_pose.pose.orientation.w = quat_avg[3]
        
        avg_pose.header.frame_id = frame_id
        avg_pose.header.stamp = rospy.Time.now()

        return avg_pose
    
    def get_stamped_pose(self, position, orientation, frame):
        stamped_pose = PoseStamped()
        stamped_pose.header.frame_id = frame
        stamped_pose.header.stamp = rospy.Time.now()
        stamped_pose.pose.position.x = position[0]
        stamped_pose.pose.position.y = position[1]
        stamped_pose.pose.position.z = position[2]
        stamped_pose.pose.orientation.x = orientation[0]
        stamped_pose.pose.orientation.y = orientation[1]
        stamped_pose.pose.orientation.z = orientation[2]
        stamped_pose.pose.orientation.w = orientation[3]
        return stamped_pose


    def _mark_grasp(self, pose: PoseStamped, id: int, camera_frame: str) -> None:
        """Mark the target gripper pose in the camera frame using a panda gripper mesh.

        Args:
            pose (PoseStamped): The gripper pose.
            id (int): The ID of the marker.
            camera_frame (str): The frame ID of the camera.
        """
        pose.header.stamp = rospy.Time(0)  # Avoid time issues with the transform
        pose = self.tf_listener.transformPose(camera_frame, pose)
        marker = Marker()
        marker.header = pose.header
        marker.pose = pose.pose
        marker.ns = "trajectory"
        marker.id = int(id)
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = "package://grasp_execution/assets/franka_hand.stl"
        marker.action = Marker.ADD
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.a = 0.8
        marker.color.r = 0.0
        marker.color.b = 1.0
        marker.color.g = 0.0
        self.marker_publisher.publish(marker)

    def limit_pose(self, start_pose: PoseStamped, end_pose: PoseStamped, max_distance_to_start_pose: float = None, max_angle_to_start_pose: float = None) -> PoseStamped:
        """Returns a pose that is as close as possible to the end pose, but within a maximum distance
        and rotation angle from the start pose.
        Args:
            start_pose (PoseStamped): The starting pose.
            end_pose (PoseStamped): The ending pose.
            max_distance_to_start_pose (float): The maximum distance from the start pose.
            max_angle_to_start_pose (float): The maximum rotation angle in radians from the start pose.
        Returns:
            PoseStamped: The interpolated pose.
        """

        assert start_pose.header.frame_id == end_pose.header.frame_id

        # Limitation due to distance
        if max_distance_to_start_pose is not None:
            t_d = max_distance_to_start_pose / self.distance(start_pose, end_pose)
        else:
            t_d = 1.0

        # Limitation due to rotation
        if max_angle_to_start_pose is not None:
            t_r = max_angle_to_start_pose / self.angle(start_pose, end_pose)
        else:
            t_r = 1.0

        # Ensure t is between 0 and 1
        t = min(t_d, t_r, 1.0)  

        if t == 1.0:
            return end_pose
        
        rospy.loginfo(f"Setting interpolated pose with t = {t:.2f} (distance limit: {t_d:.2f}, rotation limit: {t_r:.2f})")

        interpol_pose = PoseStamped()
        interpol_pose.header.frame_id = start_pose.header.frame_id
        interpol_pose.header.stamp = rospy.Time.now()

        # Interpolate position
        start_x = start_pose.pose.position.x
        start_y = start_pose.pose.position.y
        start_z = start_pose.pose.position.z
        end_x = end_pose.pose.position.x
        end_y = end_pose.pose.position.y
        end_z = end_pose.pose.position.z
        interpol_pose.pose.position.x = start_x + t * (end_x - start_x)
        interpol_pose.pose.position.y = start_y + t * (end_y - start_y)
        interpol_pose.pose.position.z = start_z + t * (end_z - start_z)

        # Interpolate orientation using quaternion slerp
        start_quat = [start_pose.pose.orientation.x,
                      start_pose.pose.orientation.y,
                      start_pose.pose.orientation.z,
                      start_pose.pose.orientation.w]
        end_quat = [end_pose.pose.orientation.x,
                    end_pose.pose.orientation.y,
                    end_pose.pose.orientation.z,
                    end_pose.pose.orientation.w]
        interpol_quat = quaternion_slerp(
            start_quat, end_quat, t
        )
        interpol_pose.pose.orientation.x = interpol_quat[0]
        interpol_pose.pose.orientation.y = interpol_quat[1]
        interpol_pose.pose.orientation.z = interpol_quat[2]
        interpol_pose.pose.orientation.w = interpol_quat[3]

        return interpol_pose

if __name__ == "__main__":
    with open(os.path.join(ROOT_DIR, "config", "default.yaml"), 'r') as f:
        cfg = yaml.safe_load(f)
        try:
            MotionExecutor(cfg)
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo("Motion executor node interrupted.")
        except Exception as e:
            rospy.logerr(f"Error: {e}")
