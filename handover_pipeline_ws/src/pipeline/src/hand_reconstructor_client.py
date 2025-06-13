#!/usr/bin/env python3
from omegaconf import DictConfig
import rospy
from typing import Tuple

from hand_reconstructor.srv import (
    ReconstructHand,
    ReconstructHandRequest,
    ReconstructHandResponse,
)
from geometry_msgs.msg import Transform
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Int32MultiArray


class HandReconstructorClient:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self._service_client = rospy.ServiceProxy(self.cfg.server_name, ReconstructHand)

        rospy.loginfo(f"Waiting for service {self.cfg.server_name}...")
        rospy.wait_for_service(self.cfg.server_name)
        rospy.loginfo(f"Service {self.cfg.server_name} is up.")

    def reconstruct_hand(
        self, image: Image
    ) -> Tuple[Transform, Int32MultiArray, CameraInfo]:
        """
        Reconstruct the hand from the provided image. Waits for the service to
        complete and returns the result.
        Args:
            image: The image of the hand to reconstruct.
        Returns:
            A tuple containing the hand transform, 2D-hand-keypoints and
            the estimated camera info.
        """
        rospy.loginfo("Reconstructing hand...")
        request = ReconstructHandRequest(image=image)

        try:
            response: ReconstructHandResponse = self._service_client(request)
            rospy.loginfo("Hand reconstruction completed successfully.")
            return (
                response.transform_camera_to_hand,
                response.keypoints_2d,
                response.camera_info,
            )
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None, None
