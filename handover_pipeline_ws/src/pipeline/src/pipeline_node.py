#!/usr/bin/env python3
import actionlib
from datetime import datetime
from dotenv import load_dotenv
import cv2
from cv_bridge import CvBridge
from hydra import initialize, compose
import numpy as np
from omegaconf import DictConfig
import os
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32MultiArray

from grasp_generator.msg import (
    GenerateGraspAction,
    GenerateGraspGoal,
)

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
        self._cv_bridge = CvBridge()
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


class GraspGenerationClient:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self._action_client = actionlib.SimpleActionClient(
            self.cfg.server_name, GenerateGraspAction
        )
        self._cv_bridge = CvBridge()
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


class Pipeline:
    # Image of the target object
    object_image: Image = None
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

    # This publisher tells the other nodes where to save the output files
    out_dir_publisher: rospy.Publisher = None

    # Subscribers and clients
    task_topic_subscriber: rospy.Subscriber = None
    camera_topic_subscriber: rospy.Subscriber = None
    grasp_generation_client: GraspGenerationClient = None
    correspondence_estimation_client: CorrespondenceEstimationClient = None

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._cv_bridge = CvBridge()

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

    def run(self):

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
            path = os.path.join(self.cfg.debug.example_dir, "object_image.png")
            image = cv2.imread(path)
            self.object_image = self._cv_bridge.cv2_to_imgmsg(image, encoding="bgr8")
        else:
            while self.object_image is None:
                rospy.loginfo("Waiting for object image from camera topic...")
                rospy.sleep(0.1)
            rospy.loginfo("Object image received from camera topic.")

        # Generate the grasp image
        if self.cfg.debug.bypass_grasp_generator:
            rospy.loginfo("Bypassing grasp generator. Using example image.")
            path = os.path.join(self.cfg.debug.example_dir, "grasp_image.png")
            grasp_image = cv2.imread(path)
            self.grasp_image = self._cv_bridge.cv2_to_imgmsg(
                grasp_image, encoding="bgr8"
            )
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
            object_points_path = os.path.join(
                self.cfg.debug.example_dir, "corr_points_object.npy"
            )
            grasp_points_path = os.path.join(
                self.cfg.debug.example_dir, "corr_points_grasp.npy"
            )
            self.corr_points_object = Int32MultiArray(
                data=np.load(object_points_path).flatten().tolist()
            )
            self.corr_points_grasp = Int32MultiArray(
                data=np.load(grasp_points_path).flatten().tolist()
            )
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
        # TODO: Extend the pipeline
        self.dump_results()

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

        # Save the object image if available
        if self.object_image is not None:
            object_image_path = os.path.join(self.out_dir, "object_image.png")
            cv2.imwrite(
                object_image_path, self._cv_bridge.imgmsg_to_cv2(self.object_image)
            )
        else:
            rospy.logwarn("No object image to save.")

        # Save the grasp image if available
        if self.grasp_image is not None:
            grasp_image_path = os.path.join(self.out_dir, "grasp_image.png")
            cv2.imwrite(
                grasp_image_path, self._cv_bridge.imgmsg_to_cv2(self.grasp_image)
            )
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

        rospy.loginfo(f"All results saved")

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
        pass

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
