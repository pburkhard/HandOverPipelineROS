#!/usr/bin/env python
from actionlib import SimpleActionClient
from math import cos, dist, fabs
import moveit_commander
import numpy as np
import os
import rospy
import sys
from typing import List
import yaml
from franka_gripper.msg import MoveAction, MoveGoal
from geometry_msgs.msg import PoseStamped, Pose
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

        self.gripper_client = SimpleActionClient(self.cfg['ros']['actions']['gripper_control'], MoveAction)
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
            self.target_callback,
            queue_size=1,
        )  # Used in phase 2
        self.trigger_subscriber = rospy.Subscriber(
            self.cfg['ros']['subscribed_topics']['trigger'],
            Empty,
            self.trigger_callback,
            queue_size=1,
        )

        rospy.loginfo(f"{self.cfg['ros']['node_name']} node started.")

    def phase1(self, target_pose: PoseStamped):
        rospy.loginfo("Executing Phase 1: Moving to target pose.")

        eef_link = self.group.get_end_effector_link()
        rospy.loginfo("============ End effector link: %s" % eef_link)

        joint_vals = self.group.get_current_joint_values()
        rospy.loginfo(joint_vals)

        rospy.loginfo(f"Target pose:\n{target_pose}")

        self.group.set_pose_target(target_pose)
        success = self.group.go(wait=True)

        if not success:
            rospy.logerr("Failed to move to target pose.")
            return
        
        self.group.stop()
        self.group.clear_pose_targets()

    def phase2(self, target_pose: PoseStamped):

        if self.measured_pose is None:
            rospy.logerr("No measured pose available. Cannot execute Phase 2.")
            return
        
        # 1. Move the gripper to the target pose using impedance control
        rospy.loginfo(f"Executing Phase 2: Moving to target pose\n{target_pose}")
        while not rospy.is_shutdown() and not self.all_close(
            target_pose.pose, self.measured_pose.pose, self.cfg['target_tolerance']
        ):  
            rospy.loginfo(f"Target pose:\n{target_pose}")
            rospy.loginfo(f"Distance to target pose: {self.distance(target_pose, self.measured_pose)}")

            # Update the target pose
            if not self.cfg['fix_target_pose']:
                target_pose = self.average_poses(self.target_pose_history)
            self.pose_pubisher.publish(target_pose)
            rospy.sleep(0.1)

        # 2. Open the gripper
        rospy.loginfo("Reached target pose, opening the gripper.")
        goal = MoveGoal(width=0.05 * 2, speed=1.0)
        self.gripper_client.send_goal(goal)
        self.gripper_client.wait_for_result(timeout=rospy.Duration(2.0))
        rospy.loginfo("Gripper opened phase 2 terminated.")


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

        # Get a pose a bit away from the target pose
        # Calculate the direction vector from the base link to the target pose
        direction_vector = np.array([
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z
        ])

        # Normalize the direction vector
        norm = np.linalg.norm(direction_vector)
        direction_vector /= (norm + 1e-9)

        # Calculate the new pose 20cm away from the target pose in the direction of the base link
        distance = self.cfg['phase_1_target_distance']
        target_pose_phase1 = PoseStamped()
        target_pose_phase1.header.frame_id = "panda_link0"
        target_pose_phase1.header.stamp = rospy.Time.now()
        target_pose_phase1.pose.position.x = target_pose.pose.position.x - direction_vector[0] * distance
        target_pose_phase1.pose.position.y = target_pose.pose.position.y - direction_vector[1] * distance
        target_pose_phase1.pose.position.z = target_pose.pose.position.z - direction_vector[2] * distance
        target_pose_phase1.pose.orientation = target_pose.pose.orientation  # Keep the same orientation

        # Execute Phase 1
        self.phase1(target_pose_phase1)

        # Execute Phase 2
        self.phase2(target_pose)

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
            x0, y0, z0, qx0, qy0, qz0, qw0 = moveit_commander.pose_to_list(actual)
            x1, y1, z1, qx1, qy1, qz1, qw1 = moveit_commander.pose_to_list(goal)
            # Euclidean distance
            d = dist((x1, y1, z1), (x0, y0, z0))
            # phi = angle between orientations
            cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
            return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

        return True
    
    def distance(self, pose1: PoseStamped, pose2: PoseStamped):
        """
        Calculate the Euclidean distance between two PoseStamped objects.
        """
        assert pose1.header.frame_id != pose2.header.frame_id
        
        dx = pose1.pose.position.x - pose2.pose.position.x
        dy = pose1.pose.position.y - pose2.pose.position.y
        dz = pose1.pose.position.z - pose2.pose.position.z
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
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
        avg_pose.pose.orientation.x = np.mean([p.pose.orientation.x for p in poses])
        avg_pose.pose.orientation.y = np.mean([p.pose.orientation.y for p in poses])
        avg_pose.pose.orientation.z = np.mean([p.pose.orientation.z for p in poses])
        avg_pose.pose.orientation.w = np.mean([p.pose.orientation.w for p in poses])
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
