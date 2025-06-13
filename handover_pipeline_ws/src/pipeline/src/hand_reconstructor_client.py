#!/usr/bin/env python3
from omegaconf import DictConfig
import rospy
from typing import Tuple

from hand_reconstructor.srv import (
    ReconstructHand,
    ReconstructHandRequest,
    ReconstructHandResponse,
    EstimateCamera,
    EstimateCameraRequest,
    EstimateCameraResponse,
)
from geometry_msgs.msg import Transform
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Int32MultiArray


class HandReconstructorClient:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self._reconstr_client = rospy.ServiceProxy(
            self.cfg.services.reconstruct_hand, ReconstructHand
        )
        self._cam_client = rospy.ServiceProxy(
            self.cfg.services.estimate_camera, EstimateCamera
        )

        for srv_name in self.cfg.services.values():
            rospy.loginfo(f"Waiting for service {srv_name}...")
            rospy.wait_for_service(srv_name)
            rospy.loginfo(f"Service {srv_name} is up.")
        rospy.loginfo("All services are ready.")

    def reconstruct_hand(self, image: Image) -> Tuple[Transform, Int32MultiArray]:
        """
        Reconstruct the hand from the provided image. Waits for the service to
        complete and returns the result.
        Args:
            image: The image of the hand to reconstruct.
        Returns:
            A tuple containing the hand transform and 2D-hand-keypoints.
        """
        rospy.loginfo("Reconstructing hand...")
        request = ReconstructHandRequest(image=image)

        try:
            response: ReconstructHandResponse = self._reconstr_client(request)
            rospy.loginfo("Hand reconstruction completed successfully.")
            return (response.transform_camera_to_hand, response.keypoints_2d)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None, None

    def estimate_camera_info(self, image: Image) -> CameraInfo:
        """
        Estimate the camera parameters from the provided image. Waits for the
        service to complete and returns the result.
        Args:
            image: The image to estimate camera parameters from.
        Returns:
            The estimated camera parameters as a CameraInfo message.
        """
        rospy.loginfo("Estimating camera parameters...")
        request = EstimateCameraRequest(image=image)

        try:
            response: EstimateCameraResponse = self._cam_client(request)
            rospy.loginfo("Camera estimation completed successfully.")
            return response.camera_info
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None
