#!/usr/bin/env python3
import actionlib
from omegaconf import DictConfig
import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import String

from correspondence_estimator.msg import (
    EstimateCorrespondenceAction,
    EstimateCorrespondenceGoal,
)


class CorrespondenceEstimationClient:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self._action_client = actionlib.SimpleActionClient(
            self.cfg.server_name, EstimateCorrespondenceAction
        )
        rospy.loginfo(f"Waiting for action server {self.cfg.server_name}...")
        self._action_client.wait_for_server()
        rospy.loginfo(f"Action server {self.cfg.server_name} is up.")

    def estimate_correspondence(
        self, object_image: Image, grasp_image: Image, object_description: String
    ):
        """
        Estimate correspondence points between the object image and the grasp image.
        Args:
            object_image: The image of the object to grasp.
            grasp_image: The image of the generated grasp.
            object_description: Description of the object.
        Returns:
            A tuple of two lists containing the correspondence points in the object image
            and the grasp image, respectively (in this order).
        """
        rospy.loginfo("Estimating correspondence...")
        self._send_goal(
            image_1=object_image,
            image_2=grasp_image,
            object_description=object_description,
        )

        self._action_client.wait_for_result(rospy.Duration(self.cfg.ros.timeout))
        result = self._action_client.get_result()

        if result is None:
            rospy.logerr("No result received from the action server.")
            return None

        rospy.loginfo("Correspondence estimated successfully.")
        return result.points_1, result.points_2

    def _send_goal(self, image_1: Image, image_2: Image, object_description: String):
        goal_msg = EstimateCorrespondenceGoal()
        goal_msg.image_1 = image_1
        goal_msg.image_2 = image_2
        goal_msg.object_description = object_description.data

        rospy.loginfo(
            f"Sending goal with object description: '{goal_msg.object_description}'."
        )
        self._action_client.send_goal(
            goal_msg,
            done_cb=self._done_callback,
            active_cb=self._active_callback,
            feedback_cb=self._feedback_callback,
        )

    def _active_callback(self):
        rospy.loginfo("Correspondence goal just went active.")

    def _feedback_callback(self, feedback):
        rospy.loginfo(f"Received feedback: {feedback.status}")

    def _done_callback(self, status, result):
        rospy.loginfo(
            f"Action done. Status: {status}, success: {getattr(result, 'success', None)}"
        )
