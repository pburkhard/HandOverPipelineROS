#!/usr/bin/env python3
from datetime import datetime
from dotenv import load_dotenv
import cv2
from hydra import initialize, compose
import numpy as np
from omegaconf import DictConfig
import os
import rospy

from correspondence_estimation_client import CorrespondenceEstimationClient
from grasp_generation_client import GraspGenerationClient
from hand_reconstructor_client import HandReconstructorClient
from transform_estimation_client import TransformEstimationClient
from msg_utils import (
    cv2_to_imgmsg,
    imgmsg_to_cv2,
    np_to_transformmsg,
    transformmsg_to_np,
    np_to_multiarraymsg,
)

from geometry_msgs.msg import Transform
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Int32MultiArray


class Pipeline:
    # Camera info for the object image
    object_camera_info: CameraInfo = None
    # Camera info for the grasp image
    grasp_camera_info: CameraInfo = None
    # Image of the target object
    object_image: Image = None
    # Depth imag of the target object
    object_image_depth: Image = None
    # Description of the target object
    object_description: String = None
    # Description of the task to be performed with the object
    task_description: String = None
    # Image of the generated grasp
    grasp_image: Image = None
    # Correspondence points in the object image
    corr_points_object: Int32MultiArray = None
    # Correspondence points in the grasp image
    corr_points_grasp: Int32MultiArray = None
    # Transform from the grasp frame to the object frames
    transform_grasp_to_object: Transform = None

    # This publisher tells the other nodes where to save the output files
    out_dir_publisher: rospy.Publisher = None

    # Subscribers and clients
    task_topic_subscriber: rospy.Subscriber = None
    camera_topic_subscriber: rospy.Subscriber = None
    correspondence_estimation_client: CorrespondenceEstimationClient = None
    hand_reconstructor_client: HandReconstructorClient = None
    grasp_generation_client: GraspGenerationClient = None
    transform_estimation_client: TransformEstimationClient = None

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Create output directory with a timestamp and pass it to other components
        self.out_dir = self._create_output_directory(self.cfg.debug.out_dir)

        rospy.init_node(self.cfg.ros.node_name, anonymous=True)

        # Setup the log directory publisher
        self.out_dir_publisher = rospy.Publisher(
            self.cfg.ros.out_dir_topic,
            String,
            queue_size=1,
        )

        # Publish the output directory immediately and then every 10 seconds
        self._publish_out_dir(None)
        rospy.Timer(rospy.Duration(1), self._publish_out_dir)

        # Setup the task topic subscriber
        if not self.cfg.debug.bypass_task_subscriber:
            self.task_topic_subscriber = rospy.Subscriber(
                self.cfg.ros.task_topic,
                String,
                self._task_callback,
                queue_size=self.cfg.ros.queue_size,
            )
            rospy.loginfo(f"Subscribed to task topic: {self.cfg.ros.task_topic}")

        # Setup the camera topic subscriber
        if not self.cfg.debug.bypass_camera_subscriber:
            self.camera_topic_subscriber = rospy.Subscriber(
                self.cfg.ros.camera_topic,
                Image,
                self._camera_callback,
                queue_size=self.cfg.ros.queue_size,
            )
            rospy.loginfo(f"Subscribed to camera topic: {self.cfg.ros.camera_topic}")

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

    def run(self):
        # Run the initialization step to get the transform hand->gripper
        self.initialization_step()

        # Enter the main loop
        self.main_loop()

    def initialization_step(self):

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
            while self.object_description is None or self.task_description is None:
                rospy.loginfo("Waiting for task description from topic...")
                rospy.sleep(0.1)
            rospy.loginfo("Task description received from topic.")

        # Get the object image data
        if self.cfg.debug.bypass_camera_subscriber:
            rospy.loginfo("Bypassing camera subscriber. Using example image.")
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
            self.object_image_depth = cv2_to_imgmsg(image, encoding="mono8")
        else:
            while self.object_image is None:
                rospy.loginfo("Waiting for object image from camera topic...")
                rospy.sleep(0.1)
            rospy.loginfo("Object image received from camera topic.")

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

        # Estimate correspondence
        if self.cfg.debug.bypass_correspondence_estimator:
            rospy.loginfo("Bypassing correspondence estimator. Using example data.")
            # object correspondence points
            path = os.path.join(self.cfg.debug.example_dir, "corr_points_object.npy")
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
                    object_image=self.object_image,
                    grasp_image=self.grasp_image,
                    object_description=self.object_description,
                )
            )
            rospy.loginfo("Correspondence estimation completed.")

        # Get the camera info for the grasp image
        if self.cfg.debug.bypass_hand_reconstructor:
            rospy.loginfo("Bypassing hand reconstructor. Using example data.")
            # Camera info
            path = os.path.join(self.cfg.debug.example_dir, "K_grasp.npy")
            K_grasp = np.load(path)
            self.grasp_camera_info = CameraInfo()
            self.grasp_camera_info.K = K_grasp.flatten().tolist()
        else:
            rospy.loginfo("Estimating camera intrinsics for the grasp image...")
            self.grasp_camera_info = self.hand_reconstructor_client.reconstruct_hand(
                image=self.grasp_image
            )[1]

        # Estimate the transform between the object and grasp images
        if self.cfg.debug.bypass_transform_estimator:
            rospy.loginfo("Bypassing transform estimator. Using example data.")
            # Load the example transform from a file
            transform_path = os.path.join(
                self.cfg.debug.example_dir, "transform_grasp_to_object.npy"
            )
            transform_data = np.load(transform_path)
            self.transform_grasp_to_object = np_to_transformmsg(transform_data)
        else:
            rospy.loginfo("Estimating transform from grasp to object image...")
            self.transform_grasp_to_object = (
                self.transform_estimation_client.estimate_transform(
                    object_camera_info=self.object_camera_info,
                    grasp_camera_info=self.grasp_camera_info,
                    object_image_depth=self.object_image_depth,
                    corr_points_object=self.corr_points_object,
                    corr_points_grasp=self.corr_points_grasp,
                )
            )

        # dump all results to the output directory
        self.dump_results()

    def main_loop(self):
        pass

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

        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = os.path.join(self.cfg.debug.out_dir, timestamp)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                rospy.loginfo(f"Output directory created: {out_dir}")
                return out_dir
            rospy.logwarn(
                f"Output directory {out_dir} already exists. "
                "Creating a new directory with a different timestamp."
            )
            rospy.sleep(1)  # Wait for a second before trying again

    def dump_results(self):
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
                os.path.join(self.out_dir, "object_points.npy"),
                np.array(self.corr_points_object.data).reshape(-1, 2),
            )
            np.save(
                os.path.join(self.out_dir, "grasp_points.npy"),
                np.array(self.corr_points_grasp.data).reshape(-1, 2),
            )
        else:
            rospy.logwarn("No correspondence points to save.")

        # Save the transform from grasp to object image if available
        if self.transform_grasp_to_object is not None:
            transform_matrix = transformmsg_to_np(self.transform_grasp_to_object)
            np.save(
                os.path.join(self.out_dir, "transform_grasp_to_object.npy"),
                transform_matrix,
            )
        else:
            rospy.logwarn("No transform from grasp to object image to save.")

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

    def _task_callback(self, msg):
        """
        TODO
        """
        raise NotImplementedError("Task callback is not implemented yet.")

    def _camera_callback(self, msg: Image):
        """
        Callback function for the camera topic subscriber. Receives an image from the camera
        and stores it in the object_image attribute.
        Args:
            msg (Image): The image message received from the camera topic.
        """
        self.object_image = msg
        rospy.loginfo("Received object image from camera topic.")


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
