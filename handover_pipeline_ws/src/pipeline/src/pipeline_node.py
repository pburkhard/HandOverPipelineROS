#!/usr/bin/env python3
from datetime import datetime
from dotenv import load_dotenv
import cv2
from hydra import initialize, compose
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import rospy
import tf
from correspondence_estimation_client import CorrespondenceEstimationClient
from grasp_generation_client import GraspGenerationClient
from hand_reconstructor_client import HandReconstructorClient
from transform_estimation_client import TransformEstimationClient
from msg_utils import (
    cv2_to_imgmsg,
    imgmsg_to_cv2,
    multiarraymsg_to_np,
    np_to_multiarraymsg,
    np_to_transformmsg,
    transformmsg_to_np,
    transformmsg_to_posemsg_stamped,
)

from geometry_msgs.msg import Transform, TransformStamped, PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Int32MultiArray
from tf2_msgs.msg import TFMessage


try:
    from custom_msg.msg import Task
except ImportError:
    from pipeline.msg import Task  # Fallback if custom_msg is not available


import tf2_ros
from visualization_msgs.msg import Marker


class Pipeline:

    # This publisher tells the other nodes where to save the output files
    out_dir_publisher: rospy.Publisher = None

    ###########################################################################
    ### Initialization Step Attributes
    ###########################################################################
    # Camera info for the object image
    object_camera_info: CameraInfo = None  # Only set once
    latest_object_camera_info: CameraInfo = None  # Continuously updated
    # Camera info for the grasp image
    grasp_camera_info: CameraInfo = None
    # Image of the target object
    object_image: Image = None  # Only set once
    latest_object_image: Image = None  # Continuously updated
    # Depth imag of the target object
    object_image_depth: Image = None  # Only set once
    latest_object_image_depth: Image = None  # Continuously updated
    # Description of the target object
    object_description: String = None
    # Description of the task to be performed with the object
    task_description: String = None
    # Image of the generated grasp
    grasp_image: Image = None
    # Correspondence points in the object image (pixel coordinates)
    corr_points_object: Int32MultiArray = None
    # Correspondence points in the grasp image (pixel coordinates)
    corr_points_grasp: Int32MultiArray = None
    # The transform from the generated image camera frame to the hand pose
    transform_hand_pose_to_gen_cam: Transform = None
    # The 3D hand pose from the generated image in the robot camera frame
    transform_robot_cam_to_gen_cam: Transform = None
    # The selected robot gripper pose in the robot camera frame
    transform_selected_grasp_to_robot_cam: Transform = None
    # Overall transform from the hand pose to the selected robot gripper pose
    transform_selected_grasp_to_hand_pose: Transform = (
        None  # This matters for the main loop
    )

    # Subscribers and clients
    task_topic_subscriber: rospy.Subscriber = None
    rgb_camera_topic_subscriber: rospy.Subscriber = None
    depth_camera_topic_subscriber: rospy.Subscriber = None
    camera_info_topic_subscriber: rospy.Subscriber = None
    transform_topic_subscriber: rospy.Subscriber = None
    correspondence_estimation_client: CorrespondenceEstimationClient = None
    hand_reconstructor_client: HandReconstructorClient = None
    grasp_generation_client: GraspGenerationClient = None
    transform_estimation_client: TransformEstimationClient = None

    ###########################################################################
    ### Main Loop Attributes
    ###########################################################################
    transform_publisher: rospy.Publisher = None
    marker_publisher: rospy.Publisher = None

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Create output directory with a timestamp and pass it to other components
        self.out_dir = self._create_output_directory(self.cfg.debug.out_dir)

        rospy.init_node(self.cfg.ros.node_name, anonymous=True)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf.TransformListener(self.tf_buffer)

        # Setup the log directory publisher
        self.out_dir_publisher = rospy.Publisher(
            self.cfg.ros.published_topics.out_dir,
            String,
            queue_size=1,
        )

        # Publish the output directory immediately and then every 10 seconds
        self._publish_out_dir(None)
        rospy.Timer(rospy.Duration(1), self._publish_out_dir)

        # Setup the task topic subscriber
        if not self.cfg.debug.bypass_task_subscriber:
            self.task_topic_subscriber = rospy.Subscriber(
                self.cfg.ros.subscribed_topics.task,
                Task,
                self._task_callback,
                queue_size=1,
            )
            rospy.loginfo(
                f"Subscribed to task topic: {self.cfg.ros.subscribed_topics.task}"
            )

        # Setup the camera topic subscribers
        if not self.cfg.debug.bypass_camera_subscriber:
            self.rgb_camera_topic_subscriber = rospy.Subscriber(
                self.cfg.ros.subscribed_topics.rgb_camera,
                Image,
                self._rgb_camera_callback,
                queue_size=1,
            )
            rospy.loginfo(
                f"Subscribed to rgb camera topic: {self.cfg.ros.subscribed_topics.rgb_camera}"
            )
            self.depth_camera_topic_subscriber = rospy.Subscriber(
                self.cfg.ros.subscribed_topics.depth_camera,
                Image,
                self._depth_camera_callback,
                queue_size=1,
            )
            rospy.loginfo(
                f"Subscribed to depth camera topic: {self.cfg.ros.subscribed_topics.depth_camera}"
            )
            self.camera_info_topic_subscriber = rospy.Subscriber(
                self.cfg.ros.subscribed_topics.camera_info,
                CameraInfo,
                self._camera_info_callback,
                queue_size=1,
            )

        # Setup the transform topic subscriber
        if not self.cfg.debug.bypass_transform_subscriber:
            self.transform_topic_subscriber = rospy.Subscriber(
                self.cfg.ros.subscribed_topics.transform_selected_grasp_to_robot_cam,
                Marker,
                self._transform_callback,
                queue_size=1,
            )
            rospy.loginfo(
                f"Subscribed to transform topic: {self.cfg.ros.subscribed_topics.transform_selected_grasp_to_robot_cam}"
            )

        # Setup the grasp generation client
        if not self.cfg.debug.bypass_grasp_generator:
            self.grasp_generation_client = GraspGenerationClient(
                self.cfg.grasp_generator_client
            )

        # Setup the correspondence estimation client
        if not self.cfg.debug.bypass_correspondence_estimator:
            self.correspondence_estimation_client = CorrespondenceEstimationClient(
                self.cfg.correspondence_estimator_client
            )

        # Setup the hand reconstruction client
        if not self.cfg.debug.bypass_hand_reconstructor:
            self.hand_reconstructor_client = HandReconstructorClient(
                self.cfg.hand_reconstructor_client
            )

        # Setup the transform estimation client
        if not self.cfg.debug.bypass_transform_estimator:
            self.transform_estimation_client = TransformEstimationClient(
                self.cfg.transform_estimator_client
            )

        # Set up the transform publisher
        self.transform_publisher = rospy.Publisher(
            self.cfg.ros.published_topics.target_gripper_pose,
            PoseStamped,
            queue_size=1,
        )

        # Set up the marker publisher
        self.marker_publisher = rospy.Publisher(
            self.cfg.ros.published_topics.marker,
            Marker,
            queue_size=1,
        )

        # Log the config
        if self.cfg.debug.log_config:
            config_path = os.path.join(self.out_dir, "main_config.yaml")
            with open(config_path, "w") as f:
                OmegaConf.save(config=self.cfg, f=f.name)

    def run(self):
        # Run the initialization step to get the transform hand->gripper
        try:
            self.initialization_step()
        except Exception as e:
            rospy.logerr(f"Initialization step failed: {e}")
            # dump all results to the output directory
            if self.cfg.debug.log_init_results:
                self._dump_results()
            return

        # Enter the main loop
        if self.cfg.debug.skip_main_loop:
            rospy.loginfo("Skipping main loop as per configuration.")
            return
        self.main_loop()

    def initialization_step(self):
        """
        Run the initialization step of the pipeline. This step is responsible for
        getting the transform from the grasp camera frame to the gripper frame.
        """
        rospy.loginfo("Starting initialization step...")

        # Get the task description data
        if self.cfg.debug.bypass_task_subscriber:
            rospy.loginfo("Bypassing task subscriber. Using example data.")
            obj_desc_path = os.path.join(self.cfg.debug.example_dir, "task.tool.txt")
            with open(obj_desc_path, "r") as f:
                self.object_description = String(f.read().strip())
            task_desc_path = os.path.join(self.cfg.debug.example_dir, "task.task.txt")
            with open(task_desc_path, "r") as f:
                self.task_description = String(f.read().strip())
        else:
            while not rospy.is_shutdown() and (
                self.object_description is None or self.task_description is None
            ):
                rospy.loginfo("Waiting for task description from topic...")
                rospy.sleep(1)
            rospy.loginfo("Task description received from topic.")

        # Get the selected grasp transform, the object image, object depth image, and the
        # camera info for the object image. IT IS IMPORTANT THAT ALL THESE ARE SYNCHRONIZED
        if (
            self.cfg.debug.bypass_camera_subscriber
            or self.cfg.debug.bypass_transform_subscriber
        ):
            if not (
                self.cfg.debug.bypass_camera_subscriber
                and self.cfg.debug.bypass_transform_subscriber
            ):
                rospy.logwarn(
                    "Camera subscriber and transform subscriber must both be bypassed or both be active. Bypassing both."
                )
            rospy.loginfo(
                "Bypassing camera subscriber. Using example image and camera info."
            )
            # Camera info
            path = os.path.join(self.cfg.debug.example_dir, "K_object.npy")
            K_object = np.load(path)
            self.object_camera_info = CameraInfo()
            self.object_camera_info.K = K_object.flatten().tolist()
            # RGB image
            path = os.path.join(self.cfg.debug.example_dir, "object_image.png")
            image = cv2.imread(path)
            self.object_image = cv2_to_imgmsg(image, encoding="bgr8")
            # Depth image
            path = os.path.join(self.cfg.debug.example_dir, "object_image_depth.png")
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self.object_image_depth = cv2_to_imgmsg(image, encoding="16UC1")
            rospy.loginfo("Bypassing transform subscriber. Using example data.")
            # Load the example transform from a file
            transform_path = os.path.join(
                self.cfg.debug.example_dir, "transform_selected_grasp_to_robot_cam.npy"
            )
            transform_data = np.load(transform_path)
            self.transform_selected_grasp_to_robot_cam = np_to_transformmsg(
                transform_data
            )
        else:
            # The transform callback will set the object image, depth image, camera info and
            # the transform from robot camera to selected grasp frame.
            while (
                not rospy.is_shutdown()
                and self.transform_selected_grasp_to_robot_cam is None
            ):
                rospy.loginfo(
                    "Waiting for transform from robot camera to selected grasp..."
                )
                rospy.sleep(1)
            rospy.loginfo("Transform from robot camera to selected grasp received.")

        # Generate the grasp image
        if self.cfg.debug.bypass_grasp_generator:
            rospy.loginfo("Bypassing grasp generator. Using example image.")
            rgb_path = os.path.join(self.cfg.debug.example_dir, "grasp_image.png")
            grasp_image = cv2.imread(rgb_path)
            self.grasp_image = cv2_to_imgmsg(grasp_image, encoding="bgr8")
        else:
            rospy.loginfo("Generating grasp image...")
            self.grasp_image = self.grasp_generation_client.generate_grasp_image(
                object_image=self.object_image,
                object_description=self.object_description,
                task_description=self.task_description,
            )
            rospy.loginfo("Grasp image generation completed.")

        # Get the camera info
        if (
            self.cfg.camera_info_source == "example_data"
            or (
                self.cfg.camera_info_source == "grasp_image"
                and self.cfg.debug.bypass_hand_reconstructor
            )
            or (
                self.cfg.camera_info_source == "robot_camera"
                and self.cfg.debug.bypass_camera_subscriber
            )
        ):
            if self.cfg.camera_info_source != "example_data":
                rospy.logwarn(
                    f"Using example data for camera intrinsics. Option '{self.cfg.camera_info_source}' is not available."
                )
            rospy.loginfo("Bypassing hand reconstructor. Using example data.")
            path = os.path.join(self.cfg.debug.example_dir, "K_grasp.npy")
            K_grasp = np.load(path)
            self.grasp_camera_info = CameraInfo()
            self.grasp_camera_info.K = K_grasp.flatten().tolist()
        elif (
            self.cfg.camera_info_source == "grasp_image"
            and not self.cfg.debug.bypass_hand_reconstructor
        ):
            rospy.loginfo("Estimating camera intrinsics for the grasp image...")
            self.grasp_camera_info = (
                self.hand_reconstructor_client.estimate_camera_info(
                    image=self.grasp_image
                )
            )
        elif (
            self.cfg.camera_info_source == "robot_camera"
            and not self.cfg.debug.bypass_camera_subscriber
        ):
            rospy.loginfo("Using camera info from the robot camera...")
            # Wait for the latest camera info to be available
            while not rospy.is_shutdown() and self.latest_object_camera_info is None:
                rospy.loginfo("Waiting for latest camera info...")
                rospy.sleep(1)
            self.grasp_camera_info = self.latest_object_camera_info
        else:
            rospy.logerr(
                f"Invalid camera info source: {self.cfg.camera_info_source}. "
                "Please choose from 'image', 'robot_camera', or 'example_data'."
            )
            return

        # Get the hand reconstruction in the grasp image
        if self.cfg.debug.bypass_hand_reconstructor:
            rospy.loginfo("Bypassing hand reconstructor. Using example data.")
            path = os.path.join(
                self.cfg.debug.example_dir, "transform_hand_pose_to_gen_cam.npy"
            )
            self.transform_hand_pose_to_gen_cam = np_to_transformmsg(np.load(path))
            path = os.path.join(self.cfg.debug.example_dir, "hand_keypoints.npy")
            hand_key_points = np.load(path)
        else:
            f = self.grasp_camera_info.K[0]  # Focal length
            rospy.loginfo(
                f"Estimating hand pose in the grasp image with focal length {f}..."
            )
            self.transform_hand_pose_to_gen_cam, hand_key_points = (
                self.hand_reconstructor_client.reconstruct_hand_pose(
                    image=self.grasp_image, focal_length=f
                )
            )

        if self.cfg.try_mirrored_image:
            object_image_np = imgmsg_to_cv2(self.object_image)
            object_image_mirrored_np = cv2.flip(
                object_image_np, 1
            )  # 1 means vertical axis (left-right)
            object_image_mirrored = cv2_to_imgmsg(
                object_image_mirrored_np, encoding="bgr8"
            )
            object_images = [self.object_image, object_image_mirrored]
            rospy.loginfo(
                "Running correspondence estimator twice: With mirrored object images."
            )
        else:
            object_images = [self.object_image]

        transforms = []
        cam_info_list = []
        cost_list = []
        for object_image in object_images:
            # Estimate correspondence
            if self.cfg.debug.bypass_correspondence_estimator:
                rospy.loginfo("Bypassing correspondence estimator. Using example data.")
                # object correspondence points
                path = os.path.join(
                    self.cfg.debug.example_dir, "corr_points_object.npy"
                )
                np_data = np.load(path)
                self.corr_points_object = np_to_multiarraymsg(np_data, Int32MultiArray)
                # grasp correspondence points
                path = os.path.join(self.cfg.debug.example_dir, "corr_points_grasp.npy")
                np_data = np.load(path)
                self.corr_points_grasp = np_to_multiarraymsg(np_data, Int32MultiArray)
            else:
                rospy.loginfo("Estimating correspondence...")
                self.corr_points_object, self.corr_points_grasp = (
                    self.correspondence_estimation_client.estimate_correspondence(
                        object_image=object_image,
                        grasp_image=self.grasp_image,
                        object_description=self.object_description,
                    )
                )
                rospy.loginfo("Correspondence estimation completed.")

            # Estimate the transform from the gen camera frame to the robot camera frame
            if self.cfg.debug.bypass_transform_estimator:
                rospy.loginfo("Bypassing transform estimator. Using example data.")
                # Load the example transform from a file
                if not self.cfg.use_heuristic_transform_estimation:
                    transform_path = os.path.join(
                        self.cfg.debug.example_dir, "transform_robot_cam_to_gen_cam.npy"
                    )
                else:
                    transform_path = os.path.join(
                        self.cfg.debug.example_dir,
                        "transform_hand_pose_to_robot_cam.npy",
                    )
                transform_np = np.load(transform_path)
                transforms.append(np_to_transformmsg(transform_np))
                cam_info_list.append(self.grasp_camera_info)
                cost_list.append(0.0)  # Use a dummy value
            else:
                if not self.cfg.use_heuristic_transform_estimation:
                    rospy.loginfo("Estimating hand pose in robot camera frame...")
                    transform, cam_info, cost = (
                        self.transform_estimation_client.estimate_transform(
                            object_camera_info=self.object_camera_info,
                            grasp_camera_info=self.grasp_camera_info,
                            object_image_depth=self.object_image_depth,
                            corr_points_object=self.corr_points_object,
                            corr_points_grasp=self.corr_points_grasp,
                        )
                    )
                    transforms.append(transform)
                    cam_info_list.append(cam_info)
                    cost_list.append(cost)
                else:
                    rospy.loginfo(
                        "Using heuristic transform estimation for robustness..."
                    )
                    transform, cost = (
                        self.transform_estimation_client.estimate_transform_heuristic(
                            object_camera_info=self.object_camera_info,
                            object_image_depth=self.object_image_depth,
                            corr_points_object=self.corr_points_object,
                            corr_points_grasp=self.corr_points_grasp,
                            transform_hand_pose_to_gen_cam=self.transform_hand_pose_to_gen_cam,
                            hand_keypoints=hand_key_points,
                        )
                    )
                    transforms.append(transform)
                    cost_list.append(cost)

        # Get the best transform based on the mean squared error
        idx = cost_list.index(min(cost_list))
        if self.cfg.debug.log_verbose:
            rospy.loginfo(f"Transform cost: {cost_list[idx]}")
            if self.cfg.try_mirrored_image:
                rospy.loginfo(f"Transform cost for original image: {cost_list[0]}")
                rospy.loginfo(f"Transform cost for mirrored image: {cost_list[1]}")
        if idx == 0:
            rospy.loginfo("Using original object image for transform estimation.")
        else:
            rospy.loginfo("Using mirrored object image for transform estimation.")
        # self.focal_length_optimized = cam_info_list[idx].K[0]

        if not self.cfg.use_heuristic_transform_estimation:
            transform_hand_pose_to_robot_cam = None
            # Finally calculate the overall transform from the hand frame to the gripper frame
            self.transform_robot_cam_to_gen_cam = transforms[idx]
            transform_gen_cam_to_hand_pose = self._invert_transform(
                self.transform_hand_pose_to_gen_cam
            )
            self.transform_selected_grasp_to_hand_pose = self._concat_transforms(
                transform_gen_cam_to_hand_pose,
                self.transform_robot_cam_to_gen_cam,
                self.transform_selected_grasp_to_robot_cam,
            )
            if self.cfg.debug.log_verbose:
                rospy.loginfo(
                    f"Selected grasp pose in robot camera frame:\n{self.transform_selected_grasp_to_robot_cam}"
                )
                transform_selected_grasp_to_robot_cam = self._concat_transforms(
                    self.transform_robot_cam_to_gen_cam,
                    self.transform_selected_grasp_to_robot_cam,
                )
                rospy.loginfo(
                    f"Hand pose in gen camera frame:\n{self.transform_hand_pose_to_gen_cam}"
                )
                rospy.loginfo(
                    f"Selected grasp in gen camera frame:\n{transform_selected_grasp_to_robot_cam}"
                )
                rospy.loginfo(
                    f"Transform from selected grasp to hand pose:\n{self.transform_selected_grasp_to_hand_pose}"
                )
        else:
            transform_hand_pose_to_robot_cam = transforms[idx]
            transform_robot_cam_to_hand_pose = self._invert_transform(
                transform_hand_pose_to_robot_cam
            )
            self.transform_selected_grasp_to_hand_pose = self._concat_transforms(
                transform_robot_cam_to_hand_pose,
                self.transform_selected_grasp_to_robot_cam,
            )

            if self.cfg.debug.log_verbose:
                rospy.loginfo(
                    f"Transform from hand pose to robot camera frame:\n{transform_hand_pose_to_robot_cam}"
                )
                rospy.loginfo(
                    f"Selected grasp pose in robot camera frame:\n{self.transform_selected_grasp_to_robot_cam}"
                )
                rospy.loginfo(
                    f"Hand pose in gen camera frame:\n{self.transform_hand_pose_to_gen_cam}"
                )
                rospy.loginfo(
                    f"Transform from selected grasp to hand pose:\n{self.transform_selected_grasp_to_hand_pose}"
                )

        if self.cfg.debug.visualize_target_hand_pose:
            # Visualize the hand pose in the object image
            if self.cfg.debug.bypass_hand_reconstructor:
                rospy.logwarn(
                    "Cannot visualize target hand pose in object image because the hand reconstructor is bypassed."
                )
            else:
                # We need the full estimation dict to visualize the hand pose
                estimation_dict: dict = self.hand_reconstructor_client.reconstruct_hand(
                    self.grasp_image
                )

                n_hands = estimation_dict["n_hands"]
                # n_hands = 1 # We force single hand for visualization
                # estimation_dict["hand_pose"] = estimation_dict["hand_pose"][:n_hands]
                # estimation_dict["hand_shape"] = estimation_dict["hand_shape"][:n_hands]
                # estimation_dict["is_right"] = estimation_dict["is_right"][:n_hands]

                if transform_hand_pose_to_robot_cam is None:
                    transform_gen_cam_to_robot_cam = self._invert_transform(
                        self.transform_robot_cam_to_gen_cam
                    )

                    # Get the hand global orientation(s) in the object image frame
                    hand_global_orient = estimation_dict["hand_global_orient"]
                    hand_global_orient_in_robot_cam = []
                    for i in range(n_hands):
                        hand_global_orient_4x4 = np.eye(4)
                        hand_global_orient_4x4[:3, :3] = hand_global_orient[i]
                        orient_in_robot_cam = np.dot(
                            transformmsg_to_np(transform_gen_cam_to_robot_cam),
                            hand_global_orient_4x4,
                        )
                        hand_global_orient_in_robot_cam.append(
                            np.expand_dims(orient_in_robot_cam[:3, :3], axis=0)
                        )
                    estimation_dict["hand_global_orient"] = np.stack(
                        hand_global_orient_in_robot_cam, axis=0
                    )

                    # Get the hand translation(s) in the object image frame
                    cam_t = estimation_dict["pred_cam_t_global"]
                    cam_t_in_robot_cam = []
                    for i in range(n_hands):
                        cam_t_4x4 = np.eye(4)
                        cam_t_4x4[:3, 3] = cam_t[i]
                        cam_t_in_robot_cam.append(
                            np.dot(
                                transformmsg_to_np(transform_gen_cam_to_robot_cam),
                                cam_t_4x4,
                            )[:3, 3]
                        )
                    estimation_dict["pred_cam_t_global"] = np.stack(
                        cam_t_in_robot_cam, axis=0
                    )
                else:
                    estimation_dict["hand_global_orient"] = np.tile(
                        transformmsg_to_np(transform_hand_pose_to_robot_cam)[:3, :3],
                        (n_hands, 1, 1, 1),
                    )

                    estimation_dict["pred_cam_t_global"] = np.tile(
                        transformmsg_to_np(transform_hand_pose_to_robot_cam)[:3, 3],
                        (n_hands, 1),
                    )

                # Extract the focal length from the camera info
                # K = np.array(self.object_camera_info.K).reshape(3, 3)
                # focal_length = (K[0, 0] + K[1, 1]) / 2.0  # Average focal length
                # estimation_dict["scaled_focal_length"] = np.tile(
                #     focal_length, n_hands
                # ).reshape(-1, 1)

                # call the renderer
                out_image = self.hand_reconstructor_client.render_hand(
                    self.object_image, estimation_dict
                )

                # Save the result
                out_image_path = os.path.join(
                    self.out_dir, "target_hand_pose_in_object_image.png"
                )
                cv2.imwrite(out_image_path, imgmsg_to_cv2(out_image))
                rospy.loginfo(
                    f"Target hand pose in object image saved to {out_image_path}"
                )
        # dump all results to the output directory
        if self.cfg.debug.log_init_results:
            self._dump_results()

        rospy.loginfo("Initialization step completed successfully.")

    def main_loop(self):
        """Main loop of the pipeline. It repeatedly takes the latest camera image and calculates
        the target robot gripper position with respect to the camera. The transform from camera
        to gripper is then published. This loop will run indefinitely until the node is shut down.
        """

        if (
            not self.object_image
            or not self.object_image_depth
            or not self.transform_selected_grasp_to_hand_pose
        ):
            rospy.logerr(
                "Initialization step did not complete successfully. "
                "Cannot enter main loop."
            )
            return
        rospy.loginfo("Entered main loop")

        hand_pose_history = np.ndarray((self.cfg.hand_pose_history_size, 4, 4))

        i = 0
        try:
            while not rospy.is_shutdown():
                i += 1
                # Store the rbg image and depth image in a variable to make sure they are synchronized
                if self.cfg.debug.bypass_camera_subscriber:
                    rospy.loginfo(
                        f"[it {i:04d}] Bypassing camera subscriber. Using example data."
                    )
                    # Camera info
                    path = os.path.join(self.cfg.debug.example_dir, "K_live.npy")
                    K_live = np.load(path)
                    live_camera_info = CameraInfo()
                    live_camera_info.K = K_live.flatten().tolist()
                    # RGB image
                    path = os.path.join(self.cfg.debug.example_dir, "live_image.png")
                    live_image_np = cv2.imread(path)
                    live_image = cv2_to_imgmsg(live_image_np, encoding="bgr8")
                    # Depth image
                    path = os.path.join(
                        self.cfg.debug.example_dir, "live_image_depth.png"
                    )
                    live_image_depth_np = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    live_image_depth = cv2_to_imgmsg(
                        live_image_depth_np, encoding="16UC1"
                    )
                else:
                    # Wait for the latest object image and depth image to be available
                    while not rospy.is_shutdown() and (
                        self.latest_object_image is None
                        or self.latest_object_image_depth is None
                        or self.latest_object_camera_info is None
                    ):
                        rospy.loginfo(
                            f"[it {i:04d}] Waiting for latest camera image and info..."
                        )
                        rospy.sleep(1)

                    live_image = self.latest_object_image
                    live_image_depth = self.latest_object_image_depth
                    live_camera_info = self.latest_object_camera_info

                if self.cfg.debug.bypass_hand_reconstructor:
                    rospy.loginfo(
                        f"[it {i:04d}] Bypassing hand reconstructor. Using example data."
                    )
                    path = os.path.join(
                        self.cfg.debug.example_dir,
                        "transform_hand_pose_to_robot_cam_latest.npy",
                    )
                    transform_hand_pose_to_robot_cam = np_to_transformmsg(np.load(path))
                else:
                    # Estimate the hand orientation and 2D keypoints in the object image
                    f = live_camera_info.K[0]  # Focal length
                    rospy.loginfo(
                        f"[it {i:04d}] Estimating transform from camera to hand with focal length {f}..."
                    )
                    transform_hand_pose_to_robot_cam, keypoints_2d = (
                        self.hand_reconstructor_client.reconstruct_hand_pose(
                            image=live_image, focal_length=f
                        )
                    )
                    if transform_hand_pose_to_robot_cam is None or keypoints_2d is None:
                        rospy.logwarn(f"[it {i:04d}] Could not reconstruct hand pose")
                        continue  # Restart the loop iteration
                    if keypoints_2d.layout.dim[0].size == 0:
                        rospy.logwarn(f"[it {i:04d}] No keypoints detected.")
                        continue  # Skip this iteration if no keypoints are found

                    # clip the keypoints_2d to the image size
                    keypoints_2d_np = multiarraymsg_to_np(keypoints_2d)
                    keypoints_2d_np[:, 0] = np.clip(
                        keypoints_2d_np[:, 0], 0, live_image_depth.width - 1
                    )
                    keypoints_2d_np[:, 1] = np.clip(
                        keypoints_2d_np[:, 1], 0, live_image_depth.height - 1
                    )

                    if cfg.use_camera_depth_for_hand_pose:
                        # Only take the wrist keypoint for the translation estimation.
                        # It is central in the hand and has a relatively stable depth value
                        keypoints_2d_np = keypoints_2d_np[0:1]

                        # Extract the 3D values using the depth image
                        K = np.array(live_camera_info.K, dtype=np.float64).reshape(3, 3)
                        depth = imgmsg_to_cv2(live_image_depth)
                        keypoints_3d = self._points_2D_to_3D(
                            intrinsic_matrix=K,
                            depth_image=depth,
                            points_2D=keypoints_2d_np,
                        )
                        # Calculate the translation from the camera to the hand
                        translation_np = np.mean(keypoints_3d, axis=0)  # Take the average

                        # Create the transform
                        _transform_hand_pose_to_robot_cam = Transform()
                        _transform_hand_pose_to_robot_cam.translation.x = translation_np[0]
                        _transform_hand_pose_to_robot_cam.translation.y = translation_np[1]
                        _transform_hand_pose_to_robot_cam.translation.z = translation_np[2]
                        _transform_hand_pose_to_robot_cam.rotation = transform_hand_pose_to_robot_cam.rotation

                        transform_hand_pose_to_robot_cam = _transform_hand_pose_to_robot_cam

                # Perform temporal smoothing
                hand_pose_history[i % self.cfg.hand_pose_history_size] = transformmsg_to_np(
                    transform_hand_pose_to_robot_cam
                )

                if i < self.cfg.hand_pose_history_size:
                    continue  # Skip the first few iterations until we have enough history

                # Update the hand pose with the mean of the history
                mean_hand_pose = np.mean(hand_pose_history, axis=0)
                transform_hand_pose_to_robot_cam = np_to_transformmsg(mean_hand_pose)

                # Calculate the transform from gripper to base
                transform_camera_to_base = self._get_transform(
                    target=self.cfg.ros.frame_ids.base,
                    source=self.cfg.ros.frame_ids.camera,
                )
                if cfg.debug.use_hand_pose_as_target:
                    # Use the hand pose as the target gripper pose
                    target_transform = self._concat_transforms(
                        transform_camera_to_base,
                        transform_hand_pose_to_robot_cam,
                    )
                else:
                    target_transform = self._concat_transforms(
                        transform_camera_to_base,
                        transform_hand_pose_to_robot_cam,
                        self.transform_selected_grasp_to_hand_pose,
                    )
                if self.cfg.debug.log_verbose:
                    rospy.loginfo(
                        f"[it {i:04d}] Transform from hand to camera:\n{transform_hand_pose_to_robot_cam}"
                    )
                    rospy.loginfo(f"[it {i:04d}] Target transform:\n{target_transform}")


                # TODO: Remove this
                # p = np.eye(4)
                # trans = np.array([0.5, 0.0, 0.5])
                # rot = np.array([[ -0.4480736,  0.0000000,  0.8939967],
                #     [0.0000000,  1.0000000,  0.0000000],
                #     [-0.8939967,  0.0000000, -0.4480736 ]])
                # p[:3, 3] = trans
                # p[:3, :3] = rot
                # temp = np_to_transformmsg(p)
                # target_transform_stamped = TransformStamped()
                # target_transform_stamped.header.stamp = rospy.Time.now()
                # target_transform_stamped.header.frame_id = "panda_link0"
                # target_transform_stamped.transform = target_pose
                # target_pose = transformmsg_to_posemsg_stamped(target_transform_stamped)

                # Publish the target pose
                target_pose = PoseStamped()
                target_pose.header.seq = i
                target_pose.header.frame_id = self.cfg.ros.frame_ids.base
                target_pose.header.stamp = rospy.Time.now()
                target_pose.pose.position.x = target_transform.translation.x
                target_pose.pose.position.y = target_transform.translation.y
                target_pose.pose.position.z = target_transform.translation.z
                target_pose.pose.orientation = target_transform.rotation

                try:
                    self.transform_publisher.publish(target_pose)
                    rospy.loginfo(f"[it {i:04d}] Published target transform.")
                except Exception as e:
                    rospy.logerr(
                        f"[it {i:04d}] Failed to publish transform from camera to gripper: {e}"
                    )

                # Publish a marker for the gripper pose
                if self.cfg.debug.publish_marker:
                    self._mark_grasp(
                        pose=target_pose,
                        id=i,
                        camera_frame=self.cfg.ros.frame_ids.camera,
                    )
                    rospy.loginfo(f"[it {i:04d}] Published marker for gripper pose.")
        except Exception as e:
            rospy.loginfo(f"Main loop interrupted with exception: {e}")

    #########################
    ### Utility Functions ###
    #########################

    def _create_output_directory(self, base_dir: str) -> str:
        """
        Create an output directory with the current timestamp as name inside the given base dir.
        If the directory already exists, a new one will be created with a different timestamp.

        Args:
            base_dir (str): The base directory where the output directory will be created.
        Returns:
            str: The path to the created output directory.
        """

        while not rospy.is_shutdown():
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = os.path.join(base_dir, timestamp)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                rospy.loginfo(f"Output directory created: {out_dir}")
                return out_dir
            rospy.logwarn(
                f"Output directory {out_dir} already exists. "
                "Creating a new directory with a different timestamp."
            )
            rospy.sleep(1)  # Wait for a second before trying again

    def _concat_transforms(self, *transforms: Transform) -> Transform:
        """
        Concatenate multiple transforms and return the resulting transform.
        Attention: The order matters! The first transform is applied first, 
        then the second, and so on.
        Args:
            *transforms (Transform): The transforms to concatenate.
        Returns:
            Transform: The concatenated transform.
        """
        if not transforms:
            raise ValueError("At least one transform must be provided.")

        # Initialize with the first transform
        t_combined = transformmsg_to_np(transforms[0])

        # Iterate through the remaining transforms and concatenate
        for transform in transforms[1:]:
            t_next = transformmsg_to_np(transform)
            t_combined = np.dot(t_combined, t_next)

        return np_to_transformmsg(t_combined)

    def _invert_transform(self, transform: Transform) -> Transform:
        """
        Invert a transform and return the resulting transform.
        Args:
            transform (Transform): The transform to invert.
        Returns:
            Transform: The inverted transform.
        """
        t = transformmsg_to_np(transform)
        t_inv = np.linalg.inv(t)
        return np_to_transformmsg(t_inv)

    def _get_transform(self, target: str, source: str) -> Transform:
        """Get the transform between two frames using the tf listener.

        Args:
            frame1 (str): The first frame ID.
            frame2 (str): The second frame ID.

        Returns:
            Transform: The transform between the two frames.
        """

        self.tf_listener.waitForTransform(
            target, source, rospy.Time(0), rospy.Duration(4.0)
        )
        trans, rot = self.tf_listener.lookupTransform(target, source, rospy.Time(0))
        transform = Transform()
        transform.translation.x = trans[0]
        transform.translation.y = trans[1]
        transform.translation.z = trans[2]
        transform.rotation.x = rot[0]
        transform.rotation.y = rot[1]
        transform.rotation.z = rot[2]
        transform.rotation.w = rot[3]

        return transform

    def _points_2D_to_3D(
        self,
        intrinsic_matrix: np.ndarray,
        depth_image: np.ndarray,
        points_2D: np.ndarray,
    ) -> np.ndarray:
        """
        Get the 3D points corresponding to the given 2D points using the depth image
        and the intrinsic matrix.

        Args:
            intrinsic_matrix (np.ndarray): 3x3 intrinsic camera matrix.
            depth_image (np.ndarray): Depth image of shape (H,W) where each pixel
                value is the depth in millimeters.
            points_2D (np.ndarray): Nx2 array of 2D points in image coordinates.

        Returns:
            np.ndarray (shape (n,3)): 3D points corresponding to the 2D points.
        """

        if depth_image.ndim != 2:
            raise ValueError("Depth image must be a 2D array (H, W).")

        # Get the intrinsic parameters
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        # Get the 3D points
        points_3D = []
        for point in points_2D:
            x, y = point
            z = depth_image[int(y), int(x)] / 1000.0  # Convert to meters
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            points_3D.append([X, Y, z])

        return np.array(points_3D)

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
        marker.color.r = 1.0
        marker.color.b = 1.0
        marker.color.g = 1.0
        self.marker_publisher.publish(marker)

    def _dump_results(self):
        """
        Save the results of the pipeline to the output directory specified in the configuration.

        """

        rospy.loginfo(f"Dumping results to {self.out_dir}.")

        # Save all string variables to a text file
        with open(os.path.join(self.out_dir, "task.txt"), "w") as f:
            obj_desc = (
                self.object_description.data
                if self.object_description is not None
                else "None"
            )
            task_desc = (
                self.task_description.data
                if self.task_description is not None
                else "None"
            )
            f.write(f"Object Description: {obj_desc}\n")
            f.write(f"Task Description: {task_desc}\n")

        # Save the object camera intrinsic matrix if available
        if self.object_camera_info is not None:
            intrinsic_matrix = np.array(
                self.object_camera_info.K, dtype=np.float64
            ).reshape(3, 3)
            np.save(os.path.join(self.out_dir, "K_object.npy"), intrinsic_matrix)
        else:
            rospy.logwarn("No object camera info to save.")

        # Save the object image if available
        if self.object_image is not None:
            object_image_path = os.path.join(self.out_dir, "object_image.png")
            cv2.imwrite(object_image_path, imgmsg_to_cv2(self.object_image))
        else:
            rospy.logwarn("No object image to save.")

        # Save the object depth image if available
        if self.object_image_depth is not None:
            object_depth_image_path = os.path.join(
                self.out_dir, "object_image_depth.png"
            )
            cv2.imwrite(object_depth_image_path, imgmsg_to_cv2(self.object_image_depth))
        else:
            rospy.logwarn("No object depth image to save.")

        # Save the grasp camera intrinsic matrix if available
        if self.grasp_camera_info is not None:
            intrinsic_matrix = np.array(
                self.grasp_camera_info.K, dtype=np.float64
            ).reshape(3, 3)
            np.save(os.path.join(self.out_dir, "K_grasp.npy"), intrinsic_matrix)
        else:
            rospy.logwarn("No grasp camera info to save.")

        # Save the grasp image if available
        if self.grasp_image is not None:
            grasp_image_path = os.path.join(self.out_dir, "grasp_image.png")
            cv2.imwrite(grasp_image_path, imgmsg_to_cv2(self.grasp_image))
        else:
            rospy.logwarn("No grasp image to save.")

        # Save the correspondence points if available
        if self.corr_points_object is not None and self.corr_points_grasp is not None:
            np.save(
                os.path.join(self.out_dir, "corr_points_object.npy"),
                np.array(self.corr_points_object.data).reshape(-1, 2),
            )
            np.save(
                os.path.join(self.out_dir, "corr_points_grasp.npy"),
                np.array(self.corr_points_grasp.data).reshape(-1, 2),
            )
        else:
            rospy.logwarn("No correspondence points to save.")

        # Save the transform from generated camera to hand pose if available
        if self.transform_hand_pose_to_gen_cam is not None:
            transform_matrix = transformmsg_to_np(self.transform_hand_pose_to_gen_cam)
            np.save(
                os.path.join(self.out_dir, "transform_hand_pose_to_gen_cam.npy"),
                transform_matrix,
            )
        else:
            rospy.logwarn("No transform from grasp to object image to save.")

        # Save the transform from generated camera to robot camera if available
        if self.transform_robot_cam_to_gen_cam is not None:
            transform_matrix = transformmsg_to_np(self.transform_robot_cam_to_gen_cam)
            np.save(
                os.path.join(self.out_dir, "transform_robot_cam_to_gen_cam.npy"),
                transform_matrix,
            )
        else:
            rospy.logwarn("No transform from generated camera to robot camera to save.")

        # Save the transform from the robot camera to the selected grasp if available
        if self.transform_selected_grasp_to_robot_cam is not None:
            transform_matrix = transformmsg_to_np(
                self.transform_selected_grasp_to_robot_cam
            )
            np.save(
                os.path.join(self.out_dir, "transform_selected_grasp_to_robot_cam.npy"),
                transform_matrix,
            )
        else:
            rospy.logwarn("No transform from object to gripper frame to save.")

        # Save the overall transform from hand to gripper frame if available
        if self.transform_selected_grasp_to_hand_pose is not None:
            transform_matrix = transformmsg_to_np(
                self.transform_selected_grasp_to_hand_pose
            )
            np.save(
                os.path.join(self.out_dir, "transform_selected_grasp_to_hand_pose.npy"),
                transform_matrix,
            )
        else:
            rospy.logwarn("No overall transform from hand to gripper frame to save.")

        rospy.loginfo("All results saved")

    #####################
    ### ROS Callbacks ###
    #####################

    def _publish_out_dir(self, event):
        """
        Publish the output directory to the out_dir topic.
        This is used by other nodes to know where to save their output files.
        """
        try:
            self.out_dir_publisher.publish(String(self.out_dir))
            # rospy.loginfo(f"Published output directory: {self.out_dir}")
        except Exception as e:
            rospy.logerr(f"Failed to publish output directory: {e}")

    def _task_callback(self, msg: Task):
        """
        Callback function for the task topic subscriber. Receives a message containing
        the object and task description, and stores them in the respective attributes.
        Args:
            msg (Task): The message received from the task topic containing the object
                and task description.
        """
        self.object_description = String(msg.tool)
        self.task_description = String(msg.action)
        # rospy.loginfo("Received object and task description from task topic.")

    def _rgb_camera_callback(self, msg: Image):
        """
        Callback function for the rgb camera topic subscriber. Receives an image
        from the camera and stores it in the object_image attribute.
        Args:
            msg (Image): The image message received from the camera topic.
        """
        self.latest_object_image = msg
        # rospy.loginfo("Received object image from camera topic.")

    def _depth_camera_callback(self, msg: Image):
        """
        Callback function for the depth camera topic subscriber. Receives a depth image
        from the camera and stores it in the object_image_depth attribute.
        Args:
            msg (Image): The depth image message received from the camera topic.
        """
        self.latest_object_image_depth = msg
        # rospy.loginfo("Received object depth image from camera topic.")

    def _camera_info_callback(self, msg: CameraInfo):
        """
        Callback function for the camera info topic subscriber. Receives camera info
        and stores it in the respective camera info attributes.
        Args:
            msg (CameraInfo): The camera info message received from the camera topic.
        """
        self.latest_object_camera_info = msg
        # rospy.loginfo("Received object camera info from camera topic.")

    def _transform_callback(self, msg: TFMessage):
        """
        Callback function for the transform topic subscriber. Receives a transform message
        and stores it in the transform_selected_grasp_to_robot_cam attribute.
        Args:
            msg (TFMessage): The transform message received from the transform topic.
        """
        # The transform must be synchronized with the object images. The first time this
        # callback is called, the transform and the object images are set. Any subsequent
        # calls to this callback and the image callbacks will be ignored.
        if not self.transform_selected_grasp_to_robot_cam:
            # Set the object image, depth image, and camera info
            self.object_image = self.latest_object_image
            self.object_image_depth = self.latest_object_image_depth
            self.object_camera_info = self.latest_object_camera_info
            # Set the transform
            self.transform_selected_grasp_to_robot_cam = Transform()
            self.transform_selected_grasp_to_robot_cam.translation = msg.pose.position
            self.transform_selected_grasp_to_robot_cam.rotation = msg.pose.orientation
            rospy.loginfo(
                "Received transform from robot camera to selected grasp frame."
            )


if __name__ == "__main__":
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="default.yaml")
        load_dotenv()  # Load environment variables from .env file
        try:
            Pipeline(cfg).run()
        except rospy.ROSInterruptException:
            rospy.loginfo("GraspGenerator node interrupted.")
        except Exception as e:
            rospy.logerr(f"Error: {e}")
