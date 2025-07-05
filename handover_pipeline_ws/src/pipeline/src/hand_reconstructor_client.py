#!/usr/bin/env python3
import json
import numpy as np
from omegaconf import DictConfig
import rospy
from typing import Tuple

from hand_reconstructor.srv import (
    ReconstructHand,
    ReconstructHandRequest,
    ReconstructHandResponse,
    ReconstructHandPose,
    ReconstructHandPoseRequest,
    ReconstructHandPoseResponse,
    EstimateCamera,
    EstimateCameraRequest,
    EstimateCameraResponse,
    RenderHand,
    RenderHandRequest,
    RenderHandResponse,
)
from geometry_msgs.msg import Transform
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Int32MultiArray, String


class HandReconstructorClient:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self._reconstr_hand_client = rospy.ServiceProxy(
            self.cfg.services.reconstruct_hand, ReconstructHand
        )
        self._reconstr_hand_pose_client = rospy.ServiceProxy(
            self.cfg.services.reconstruct_hand_pose, ReconstructHandPose
        )
        self._cam_client = rospy.ServiceProxy(
            self.cfg.services.estimate_camera, EstimateCamera
        )
        self._render_hand_client = rospy.ServiceProxy(
            self.cfg.services.render_hand, RenderHand
        )

        for srv_name in self.cfg.services.values():
            rospy.loginfo(f"Waiting for service {srv_name}...")
            rospy.wait_for_service(srv_name)
            rospy.loginfo(f"Service {srv_name} is up.")
        rospy.loginfo("All services are ready.")

    def reconstruct_hand(self, image: Image) -> dict:
        """
        Reconstruct the hand from the provided image. Waits for the service to
        complete and returns the result.
        Args:
            image: The image of the hand to reconstruct.
        Returns:
            A dictionary containing all outputs of the hand reconstructor.
        """
        rospy.loginfo("Reconstructing hand...")
        request = ReconstructHandRequest(image=image)

        try:
            response: ReconstructHandResponse = self._reconstr_hand_client(request)
            rospy.loginfo("Hand reconstruction completed successfully.")
            estimation_dict = json.loads(response.estimation_dict.data)
            # Convert the lists inside the dictionary to numpy arrays
            for key, value in estimation_dict.items():
                if isinstance(value, list):
                    estimation_dict[key] = np.array(value)
            return estimation_dict
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None, None
        
    def reconstruct_hand_pose(self, image: Image, focal_length: float) -> Tuple[Transform, Int32MultiArray]:
        """ Reconstruct the hand pose from the provided image. Waits for the
        service to complete and returns the result.
        Args:
            image: The image of the hand to reconstruct.
            focal_length: The focal length of the camera used to capture the image.
        Returns:
            A tuple containing the hand transform and 2D-hand-keypoints.
        """

        rospy.loginfo("Reconstructing hand pose...")
        request = ReconstructHandPoseRequest(image=image, focal_length=Float32(focal_length))

        try:
            response: ReconstructHandPoseResponse = self._reconstr_hand_pose_client(request)
            if not response.success:
                rospy.logerr("Hand pose reconstruction failed.")
                return None, None
            rospy.loginfo("Hand pose reconstruction completed successfully.")
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
        
    def render_hand(self, image: Image, estimation: dict) -> Image:
        """
        Render the hand on the provided image using the estimation data.
        Args:
            image: The image to render the hand on.
            estimation: The estimation data containing hand parameters. it must have
                the same structure as the output of reconstruct_hand.
        Returns:
            The rendered image with the hand overlay.
        """
        rospy.loginfo("Rendering hand on image...")
        # Convert NumPy arrays to lists for JSON serialization
        for key, value in estimation.items():
            if isinstance(value, np.ndarray):
                estimation[key] = value.tolist()
        estimation_msg = String()
        estimation_msg.data = json.dumps(estimation)
        request = RenderHandRequest(image=image, estimation_dict=estimation_msg)

        try:
            response: RenderHandResponse = self._render_hand_client(request)
            rospy.loginfo("Hand rendering completed successfully.")
            return response.rendered_image
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None
