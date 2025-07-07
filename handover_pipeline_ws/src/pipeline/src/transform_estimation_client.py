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
)


class TransformEstimationClient:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self._action_client = actionlib.SimpleActionClient(
            self.cfg.server_name, EstimateTransformAction
        )
        rospy.loginfo(f"Waiting for action server {self.cfg.server_name}...")
        self._action_client.wait_for_server()
        rospy.loginfo(f"Action server {self.cfg.server_name} is up.")

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
        self._send_goal(
            object_camera_info=object_camera_info,
            grasp_camera_info=grasp_camera_info,
            object_image_depth=object_image_depth,
            corr_points_object=corr_points_object,
            corr_points_grasp=corr_points_grasp,
        )

        # Wait for the action to complete
        self._action_client.wait_for_result(rospy.Duration(self.cfg.ros.timeout))
        result = self._action_client.get_result()

        if result is None:
            rospy.logerr("No result received from the action server.")
            return None

        rospy.loginfo("Grasp image generated successfully.")
        return result.transform_grasp_to_object, result.grasp_camera_info, result.mse.data

    def _send_goal(
        self,
        object_camera_info: CameraInfo,
        grasp_camera_info: CameraInfo,
        object_image_depth: Image,
        corr_points_object: Int32MultiArray,
        corr_points_grasp: Int32MultiArray,
    ):
        """
        Send a goal to the action server to generate a grasp image.
        Args:
            object_image: The image of the object to grasp.
            object_description: Description of the object.
            task_description: Description of the task.
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
        print("Sent goal")

    def _active_callback(self):
        rospy.loginfo("Goal just went active.")

    def _feedback_callback(self, feedback):
        rospy.loginfo(f"Received feedback: {feedback.status}")

    def _done_callback(self, status, result):
        rospy.loginfo(f"Action done. Status: {status}, success: {result.success}")
