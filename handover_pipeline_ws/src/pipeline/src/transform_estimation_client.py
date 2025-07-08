#!/usr/bin/env python3
import actionlib
from omegaconf import DictConfig
import rospy
from typing import Tuple

from geometry_msgs.msg import Transform
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32MultiArray
from transform_estimator.msg import (
    EstimateTransformAction,
    EstimateTransformGoal,
    EstimateTransformHeuristicAction,
    EstimateTransformHeuristicGoal,
)


class TransformEstimationClient:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self._action_client = actionlib.SimpleActionClient(
            self.cfg.actions.estimate_transform, EstimateTransformAction
        )
        self._action_client_heuristic = actionlib.SimpleActionClient(
            self.cfg.actions.estimate_transform_heuristic, EstimateTransformHeuristicAction
        )
        rospy.loginfo(f"Waiting for action server {self.cfg.actions.estimate_transform}...")
        self._action_client.wait_for_server()
        rospy.loginfo(f"Waiting for action server {self.cfg.actions.estimate_transform_heuristic}...")
        self._action_client_heuristic.wait_for_server()
        rospy.loginfo(f"Action servers {self.cfg.actions.estimate_transform} and {self.cfg.actions.estimate_transform_heuristic} are up.")

    def estimate_transform(
        self,
        object_camera_info: CameraInfo,
        grasp_camera_info: CameraInfo,
        object_image_depth: Image,
        corr_points_object: Int32MultiArray,
        corr_points_grasp: Int32MultiArray,
    ) -> Tuple[Transform, CameraInfo, float]:
        """
        Estimate the transform between the object and grasp images based on the provided
        camera information and corresponding points.
        Args:
            object_camera_info: Camera information for the object image.
            grasp_camera_info: Camera information for the grasp image.
            object_image_depth: Depth image of the object.
            corr_points_object: Corresponding points in the object image.
            corr_points_grasp: Corresponding points in the grasp image.
        Returns:
            The estimated transform between the object and grasp image, the optimized camera
            info for the (virtual) grasp camera, and the mean squared error of the
            transformation.
        """
        goal_msg = EstimateTransformGoal()
        goal_msg.object_camera_info = object_camera_info
        goal_msg.grasp_camera_info = grasp_camera_info
        goal_msg.object_image_depth = object_image_depth
        goal_msg.corr_points_object = corr_points_object
        goal_msg.corr_points_grasp = corr_points_grasp

        rospy.loginfo("Sending goal")
        self._action_client.send_goal(
            goal_msg,
            done_cb=self._done_callback,
            active_cb=self._active_callback,
            feedback_cb=self._feedback_callback,
        )

        # Wait for the action to complete
        self._action_client.wait_for_result(rospy.Duration(self.cfg.ros.timeout))
        result = self._action_client.get_result()

        if result is None:
            rospy.logerr("No result received from the action server.")
            return None
        if not result.success:
            rospy.logerr("Transform estimation failed.")
            return None

        rospy.loginfo("Successfully estimated transform.")
        return result.transform_robot_cam_to_gen_cam, result.grasp_camera_info, result.mse.data
    
    def estimate_transform_heuristic(
        self,
        object_camera_info: CameraInfo,
        object_image_depth: Image,
        corr_points_object: Int32MultiArray,
        corr_points_grasp: Int32MultiArray,
        transform_hand_pose_to_gen_cam: Transform,
        hand_keypoints: Int32MultiArray,
    ) -> Tuple[Transform, CameraInfo, float]:
        """
        Estimate the transform between the object and grasp images based on the provided
        camera information and corresponding points.
        Args:
            object_camera_info: Camera information for the object image.
            object_image_depth: Depth image of the object.
            corr_points_object: Corresponding points in the object image.
            corr_points_grasp: Corresponding points in the grasp image.
            transform_camera_to_hand: Transform from the camera to the hand.
            hand_keypoints: Keypoints of the hand in the grasp image (pixel coordinates).
        Returns:
            The estimated transform between the object and grasp image, the optimized camera
            info for the (virtual) grasp camera, and the mean squared error of the
            transformation.
        """
        goal_msg = EstimateTransformHeuristicGoal()
        goal_msg.object_camera_info = object_camera_info
        goal_msg.object_image_depth = object_image_depth
        goal_msg.corr_points_object = corr_points_object
        goal_msg.corr_points_grasp = corr_points_grasp
        goal_msg.transform_hand_pose_to_camera = transform_hand_pose_to_gen_cam
        goal_msg.hand_keypoints = hand_keypoints

        rospy.loginfo("Sending goal")
        self._action_client_heuristic.send_goal(
            goal_msg,
            done_cb=self._done_callback,
            active_cb=self._active_callback,
            feedback_cb=self._feedback_callback,
        )

        # Wait for the action to complete
        self._action_client_heuristic.wait_for_result(rospy.Duration(self.cfg.ros.timeout))
        result = self._action_client_heuristic.get_result()

        if result is None:
            rospy.logerr("No result received from the action server.")
            return None
        if not result.success:
            rospy.logerr("Transform estimation failed.")
            return None

        rospy.loginfo("Successfully estimated transform.")
        return result.transform_hand_pose_to_robot_camera, result.cost.data

    def _active_callback(self):
        rospy.loginfo("Goal just went active.")

    def _feedback_callback(self, feedback):
        rospy.loginfo(f"Received feedback: {feedback.status}")

    def _done_callback(self, status, result):
        rospy.loginfo(f"Action done. Status: {status}, success: {result.success}")
