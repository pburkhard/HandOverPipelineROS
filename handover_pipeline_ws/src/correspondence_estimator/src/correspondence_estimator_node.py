#!/usr/bin/env python3
import actionlib
import cv2
from dotenv import load_dotenv
import io
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import os
from PIL import Image as PILImage
import torch
from typing import Tuple, List
import rospy
import sys
import yaml

from lang_sam import LangSAM

from correspondence_estimator.msg import (
    EstimateCorrespondenceAction,
    EstimateCorrespondenceResult,
    EstimateCorrespondenceFeedback,
    EstimateCorrespondenceGoal,
)

from std_msgs.msg import String, Int32MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f"{ROOT_DIR}/third-party/dino-vit-features")
from correspondences import find_correspondences, draw_correspondences


class CorrespondenceEstimator:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.object_detector = LangSAM()

        rospy.init_node(cfg.ros.node_name, anonymous=True)

        self._feedback = EstimateCorrespondenceFeedback()
        self._result = EstimateCorrespondenceResult()
        self._server = actionlib.SimpleActionServer(
            cfg.ros.node_name,
            EstimateCorrespondenceAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        self._server.start()
        rospy.loginfo(f"{cfg.ros.node_name} action server started.")

    def _execute(self, goal: EstimateCorrespondenceGoal):
        """Callback function for the action server that estimates correspondences
        between two images based on the input goal.

        Args:
            goal (EstimateCorrespondencesGoal): The goal containing the two images
            and the object type.
        """
        rospy.loginfo(f"Received goal with object {goal.object_description}.")

        # Validate the goal
        if not goal.image_1 or not goal.image_2:
            rospy.logerr("Invalid goal: images cannot be empty.")
            self._result.success = False
            self._server.set_aborted()
            return

        image_1 = goal.image_1
        image_2 = goal.image_2
        # Keep track of the original image dimensions
        w1, h1 = image_1.width, image_1.height
        w2, h2 = image_2.width, image_2.height

        # Save the original images if debug logging is enabled
        if self.cfg.debug.log_unprocessed_images:
            # Convert ROS Image data to numpy array
            img1_data = np.frombuffer(image_1.data, dtype=np.uint8).reshape(
                image_1.height, image_1.width, -1
            )
            img2_data = np.frombuffer(image_2.data, dtype=np.uint8).reshape(
                image_2.height, image_2.width, -1
            )

            # Save the images using OpenCV
            path_1 = os.path.join(self.cfg.debug.out_dir, "image_1_unpreprocessed.png")
            path_2 = os.path.join(self.cfg.debug.out_dir, "image_2_unpreprocessed.png")
            cv2.imwrite(path_1, img1_data)
            cv2.imwrite(path_2, img2_data)

        # Preprocess the images if set in the configuration
        if self.cfg.preprocess_images:
            self._feedback.status = "Preprocessing images..."
            self._feedback.percent_complete = 0.0
            self._server.publish_feedback(self._feedback)
            if not goal.object_description:
                rospy.logerr(
                    "Invalid goal: object_type must be provided for preprocessing."
                )
                self._result.success = False
                self._server.set_aborted()
                return

            images = [image_1, image_2]
            object_descriptions = [goal.object_description] * 2

            images, bboxes = self._preprocess_images(images, object_descriptions)
            image_1 = images[0]
            image_2 = images[1]
            rospy.loginfo("Preprocessed images.")

        # Convert ROS Image data to numpy arrays
        img1_data = np.frombuffer(image_1.data, dtype=np.uint8).reshape(
            image_1.height, image_1.width, -1
        )
        img2_data = np.frombuffer(image_2.data, dtype=np.uint8).reshape(
            image_2.height, image_2.width, -1
        )

        # Save the images as required by the find_correspondences function
        image_path1 = os.path.join(self.cfg.debug.out_dir, "image_1.png")
        image_path2 = os.path.join(self.cfg.debug.out_dir, "image_2.png")
        cv2.imwrite(image_path1, img1_data)
        cv2.imwrite(image_path2, img2_data)

        rospy.loginfo("Estimating correspondences...")
        self._feedback.status = "Estimating correspondences..."
        self._feedback.percent_complete = 50.0
        self._server.publish_feedback(self._feedback)
        with torch.no_grad():
            points1, points2, out_1, out_2 = find_correspondences(
                image_path1,
                image_path2,
                num_pairs=self.cfg.num_correspondences,
                load_size=self.cfg.target_image_size,
                layer=self.cfg.extraction_layer,
                facet=self.cfg.extraction_facet,
                bin=self.cfg.use_binned_descriptor,
                thresh=self.cfg.saliency_threshold,
                model_type=self.cfg.model_type,
                stride=self.cfg.model_stride,
            )
        rospy.loginfo("Correspondence estimation successful.")

        # Convert to numpy arrays
        points1 = np.array(points1, dtype=np.float32)
        points2 = np.array(points2, dtype=np.float32)

        # Scale points back to original image size
        # Due to scaling, the points might become floating point numbers!
        w1_scaled, h1_scaled = out_1.size
        w2_scaled, h2_scaled = out_2.size
        points1[:, 0] = points1[:, 0] * (h1 / h1_scaled)
        points1[:, 1] = points1[:, 1] * (w1 / w1_scaled)
        points2[:, 0] = points2[:, 0] * (h2 / h2_scaled)
        points2[:, 1] = points2[:, 1] * (w2 / w2_scaled)

        # Get the coordinates in the original image
        if self.cfg.preprocess_images:
            x1, y1, _, _ = bboxes[0]
            points1[:, 0] += y1
            points1[:, 1] += x1
            x1, y1, _, _ = bboxes[1]
            points2[:, 0] += y1
            points2[:, 1] += x1

        # Convert the points to ROS Int32MultiArrays
        points1 = points1.astype(np.int32)
        points2 = points2.astype(np.int32)
        n_points = points1.shape[0]  # Both have the same number of points

        # Convert numpy arrays to Int32MultiArray for ROS message compatibility
        points1_msg = Int32MultiArray()
        points1_msg.data = points1.flatten().tolist()
        points1_msg.layout = MultiArrayLayout()
        points1_msg.layout.dim = [
            MultiArrayDimension(label="pixels", size=n_points, stride=n_points * 2),
            MultiArrayDimension(label="coordinates", size=2, stride=2),
        ]

        points2_msg = Int32MultiArray()
        points2_msg.data = points2.flatten().tolist()
        points2_msg.layout.dim = [
            MultiArrayDimension(label="pixels", size=n_points, stride=n_points * 2),
            MultiArrayDimension(label="coordinates", size=2, stride=2),
        ]

        self._feedback.status = "Correspondence estimation successful."
        self._feedback.percent_complete = 100.0
        self._server.publish_feedback(self._feedback)
        self._result.points_1 = points1_msg
        self._result.points_2 = points2_msg
        self._result.success = True
        self._server.set_succeeded(self._result)

    def _preprocess_images(
        self,
        images: List[Image],
        object_types: List[String],
    ) -> Tuple[List[Image], List[np.ndarray]]:
        """Preprocess images to improve downstream correspondence
        point estimation. The images are cropped around the object of
        interest such that the correspondence estimator is not distracted
        by the background.

        Args:
            images (List[Image]): List of sensor_msgs/Image to be preprocessed.
            object_types (List[String]): List of object types to detect in the images.

        Returns:
            Tuple[List[Image], List[np.ndarray]]: A tuple containing the preprocessed
            images as sensor_msgs/Image and the bounding boxes of the detected objects.
        """

        # Convert list of sensor_msgs/Image to list of PIL Images
        pil_images = []
        for ros_img in images:
            img_array = np.frombuffer(ros_img.data, dtype=np.uint8).reshape(
                ros_img.height, ros_img.width, -1
            )
            img = PILImage.fromarray(img_array)
            pil_images.append(img)
        images = pil_images

        # Use lang-sam to detect objects in the images
        results = self.object_detector.predict(images, object_types)

        # For simplicity, we take the first box of each image (that's the one
        # with the highest confidence)
        bboxes: List[np.ndarray] = [res["boxes"][0] for res in results]

        # Crop the images
        cropped_images = []
        for image, bbox in zip(images, bboxes):
            x1, y1, x2, y2 = bbox
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped_image)

        # Convert the cropped images back to sensor_msgs/Image
        ros_cropped_images = []
        for cropped_image in cropped_images:
            ros_img = Image()
            ros_img.header.stamp = rospy.Time.now()
            ros_img.height = cropped_image.height
            ros_img.width = cropped_image.width
            ros_img.encoding = "rgb8"
            ros_img.step = ros_img.width * 3
            ros_img.data = cropped_image.tobytes()
            ros_cropped_images.append(ros_img)
        return ros_cropped_images, bboxes

    def _get_visualisation_object(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        image1_path: str,
        image2_path: str,
        title: str = "",
    ) -> plt.Figure:
        """Create visualization object for the correspondence points.

        Args:
            points1 (np.ndarray): Points in the first image.
                Dimensions should be (N, 2).
            points2 (np.ndarray): Points in the second image.
                Dimensions should be (N, 2).
            image1_path (str): Path to the first image.
            image2_path (str): Path to the second image.
            title (str): Title for the visualization.

        Returns:
            plt.Figure: A matplotlib figure containing the visualization.
        """

        # Convert points to list of tuples
        points1_tuples = [(int(x), int(y)) for x, y in points1]
        points2_tuples = [(int(x), int(y)) for x, y in points2]

        # Convert images to PIL format
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        # Get figures for the images
        fig1, fig2 = draw_correspondences(
            points1_tuples, points2_tuples, image1, image2
        )
        fig1_size = fig1.get_size_inches()
        fig2_size = fig2.get_size_inches()

        # Create a new figure for side-by-side display
        figsize = (fig1_size[0] + fig2_size[0], max(fig1_size[1], fig2_size[1]))
        output_fig = plt.figure(figsize=figsize)

        # Extract the figure's canvas as an array
        for i, fig in enumerate([fig1, fig2]):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)

            # Add subplot and display the image
            ax = output_fig.add_subplot(1, 2, i + 1)
            ax.imshow(np.asarray(img))
            ax.axis("off")

        # Close the original figures to avoid memory leaks
        plt.close(fig1)
        plt.close(fig2)

        # Set the title and adjust layout
        output_fig.suptitle(title)
        output_fig.tight_layout()
        return output_fig

    def visualize(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        image1_path: str,
        image2_path: str,
        title: str = "",
    ) -> None:
        """Visualize the correspondence points between two images.

        Args:
            points1 (np.ndarray): Points in the first image. Dimensions should be (N, 2)
            points2 (np.ndarray): Points in the second image. Dimensions should be (N, 2)
            image1 (np.ndarray): First image.
            image2 (np.ndarray): Second image.
        """

        self._get_visualisation_object(
            points1, points2, image1_path, image2_path, title
        )
        plt.show()

    def save_visualization(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        image1_path: str,
        image2_path: str,
        output_path: str,
        title: str = "",
    ) -> None:
        """Save the visualization of the correspondence points between two images.

        Args:
            points1 (np.ndarray): Points in the first image. Dimensions should be (N, 2)
            points2 (np.ndarray): Points in the second image. Dimensions should be (N, 2)
            image1 (np.ndarray): First image.
            image2 (np.ndarray): Second image.
            output_path (str): Path to save the visualization.
        """

        fig = self._get_visualisation_object(
            points1, points2, image1_path, image2_path, title
        )
        fig.savefig(output_path)
        plt.close(fig)


if __name__ == "__main__":

    load_dotenv()  # Load environment variables from .env file
    try:
        config_path = os.path.join(os.path.dirname(__file__), "../config/default.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        cfg = DictConfig(config)
        CorrespondenceEstimator(cfg)
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("GraspGenerator node interrupted.")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
