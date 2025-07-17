#!/usr/bin/env python3
import actionlib
from omegaconf import DictConfig
import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from grasp_generator.msg import (
    GenerateGraspAction,
    GenerateGraspGoal,
)


class GraspGenerationClient:
    """Client for the grasp generation action server."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self._action_client = actionlib.SimpleActionClient(
            self.cfg.server_name, GenerateGraspAction
        )
        rospy.loginfo(f"Waiting for action server {self.cfg.server_name}...")
        self._action_client.wait_for_server()
        rospy.loginfo(f"Action server {self.cfg.server_name} is up.")

    def generate_grasp_image(
        self, object_image: Image, object_description: String, task_description: String
    ) -> Image:
        """
        Generate a grasp image based on the provided object image and descriptions. Waits
        for the action server to complete the task and returns the result.
        Args:
            object_image: The image of the object to grasp.
            object_description: Description of the object.
            task_description: Description of the task.
        Returns:
            The result of the grasp generation.
        """
        rospy.loginfo("Generating grasp image...")
        self._send_goal(
            object_image=object_image,
            object_description=object_description,
            task_description=task_description,
        )

        # Wait for the action to complete
        self._action_client.wait_for_result(rospy.Duration(self.cfg.ros.timeout))
        result = self._action_client.get_result()

        if result is None:
            rospy.logerr("No result received from the action server.")
            return None

        rospy.loginfo("Grasp image generated successfully.")
        return result.grasp_image

    def _send_goal(
        self, object_image: Image, object_description: String, task_description: String
    ):
        """
        Send a goal to the action server to generate a grasp image.
        Args:
            object_image: The image of the object to grasp.
            object_description: Description of the object.
            task_description: Description of the task.
        """
        goal_msg = GenerateGraspGoal()
        goal_msg.object_image = object_image
        goal_msg.object_description = object_description.data
        goal_msg.task_description = task_description.data

        rospy.loginfo(
            f"Sending goal with object description: '{goal_msg.object_description}'"
            + f" and task description: '{goal_msg.task_description}'."
        )
        self._action_client.send_goal(
            goal_msg,
            done_cb=self._done_callback,
            active_cb=self._active_callback,
            feedback_cb=self._feedback_callback,
        )

    def _active_callback(self):
        rospy.loginfo("Goal just went active.")

    def _feedback_callback(self, feedback):
        rospy.loginfo(f"Received feedback: {feedback.status}")

    def _done_callback(self, status, result):
        rospy.loginfo(f"Action done. Status: {status}, success: {result.success}")
