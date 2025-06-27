#!/usr/bin/env python3
import actionlib
import cv2
from dotenv import load_dotenv
import io
from lang_sam import LangSAM
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
from PIL import Image as PILImage
import rospy
import sys
import torch
from typing import Tuple, List
import yaml


from correspondence_estimator.msg import (
    EstimateCorrespondenceAction,
    EstimateCorrespondenceResult,
    EstimateCorrespondenceFeedback,
    EstimateCorrespondenceGoal,
)

# TODO: Remove dependency on the pipeline package
sys.path.append(os.path.join(os.path.dirname(__file__), "../../pipeline/src/"))
from msg_utils import imgmsg_to_cv2

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

        # Set the output directory
        self.out_dir = None
        if self.cfg.debug.out_dir_mode == "fixed":
            self.out_dir = self.cfg.debug.out_dir_fixed
        elif self.cfg.debug.out_dir_mode == "topic":
            self._out_dir_sub = rospy.Subscriber(
                self.cfg.debug.out_dir_topic,
                String,
                self._out_dir_callback,
                queue_size=1,
            )
            while self.out_dir is None and not rospy.is_shutdown():
                rospy.loginfo(
                    "Waiting for output directory to be set via topic: "
                    + f"{self.cfg.debug.out_dir_topic}"
                )
                rospy.sleep(1.0)
        else:
            rospy.logerr(
                "Invalid out_dir_mode. Supported modes are 'fixed' and 'topic'."
            )

        # Log the config
        if self.cfg.debug.log_config:
            config_path = os.path.join(self.out_dir, "(ce)_config.yaml")
            with open(config_path, "w") as f:
                OmegaConf.save(config=self.cfg, f=f.name)

        self.n_requests = 0  # Keep track of the number of requests

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

        self.n_requests += 1
        image_1 = goal.image_1
        image_2 = goal.image_2

        # Convert ROS Image data to numpy array
        img1_data_original = imgmsg_to_cv2(image_1)
        img2_data_original = imgmsg_to_cv2(image_2)

        # Save the original images if debug logging is enabled
        if self.cfg.debug.log_unprocessed_images:
            # Save the images using OpenCV
            path_1 = os.path.join(
                self.out_dir, f"(ce)_in1_unprocessed_{self.n_requests:04d}.png"
            )
            path_2 = os.path.join(
                self.out_dir, f"(ce)_in2_unprocessed_{self.n_requests:04d}.png"
            )
            cv2.imwrite(path_1, img1_data_original)
            cv2.imwrite(path_2, img2_data_original)

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

        # Ensure to not exceed the system limits
        image_1, limit_scale_1 = self._limit_image_size(image_1)
        image_1, padding_1 = self._limit_aspect_ratio(image_1)
        image_2, limit_scale_2 = self._limit_image_size(image_2)
        image_2, padding_2 = self._limit_aspect_ratio(image_2)

        # Keep track of the input image sizes for scaling later
        w1, h1 = image_1.width, image_1.height
        w2, h2 = image_2.width, image_2.height

        # Convert ROS Image data to numpy arrays
        img1_data = imgmsg_to_cv2(image_1)
        img2_data = imgmsg_to_cv2(image_2)

        # Save the images as required by the find_correspondences function
        image_path1 = os.path.join(self.out_dir, f"(ce)_in1_{self.n_requests:04d}.png")
        image_path2 = os.path.join(self.out_dir, f"(ce)_in2_{self.n_requests:04d}.png")
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

        if self.cfg.debug.log_visualization:
            self.save_visualization(
                points1,
                points2,
                out_1,
                out_2,
                self.out_dir,
                title=f"Correspondences for {goal.object_description}",
            )

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
            points1[:, 0] += y1 - padding_1[0]
            points1[:, 1] += x1 - padding_1[2]
            x1, y1, _, _ = bboxes[1]
            points2[:, 0] += y1 - padding_2[0]
            points2[:, 1] += x1 - padding_2[2]

        if self.cfg.debug.log_visualization:
            self.save_visualization(
                points1,
                points2,
                PILImage.fromarray(img1_data_original[..., ::-1]),  # Convert BGR to RGB
                PILImage.fromarray(img2_data_original[..., ::-1]),  # Convert BGR to RGB
                self.out_dir,
                postfix="_remapped",
                title=f"Correspondences for {goal.object_description}",
            )

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
            img_array = imgmsg_to_cv2(ros_img)
            img = PILImage.fromarray(img_array[..., ::-1])  # Convert BGR to RGB
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

    def _limit_image_size(self, image: Image) -> Tuple[Image, float]:
        """Limit the size of the image to avoid memory issues.

        Args:
            image (Image): The image to limit.

        Returns:
            Tuple[Image, float]: The limited image and the scale factor used for resizing.
        """
        max_size = self.cfg.max_image_size
        width, height = image.width, image.height
        scale = 1.0  # Default scale factor
        if width * height > max_size:
            rospy.logwarn(
                f"Image size ({width}, {height}) exceeds maximum size {max_size}. Resizing."
            )
            scale = np.sqrt(max_size / (width * height))
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Convert ROS Image to PIL Image, resize, then convert back
            img_array = imgmsg_to_cv2(image)
            pil_img = PILImage.fromarray(img_array[..., ::-1])  # Convert BGR to RGB
            resized_pil_img = pil_img.resize((new_width, new_height), PILImage.LANCZOS)
            # Convert back to ROS Image
            resized_ros_img = Image()
            resized_ros_img.header = image.header
            resized_ros_img.height = new_height
            resized_ros_img.width = new_width
            resized_ros_img.encoding = "rgb8"
            resized_ros_img.step = new_width * 3
            resized_ros_img.data = resized_pil_img.tobytes()
            return resized_ros_img
        return image, scale

    def _limit_aspect_ratio(self, image: Image) -> Tuple[Image, List[int]]:
        """Limit the aspect ratio of the image to avoid memory issues.

        Args:
            image (Image): The image to limit.

        Returns:
            Tuple[Image, List[int]]: The limited image and the padding applied
        """
        max_aspect_ratio = self.cfg.max_aspect_ratio
        width, height = image.width, image.height
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            rospy.logwarn(
                f"Image aspect ratio {aspect_ratio} exceeds maximum {max_aspect_ratio}. Padding image."
            )
            # Pad the image with black pixels to achieve the target aspect ratio
            if width / height > max_aspect_ratio:
                # Too wide: pad height
                target_height = int(width / max_aspect_ratio)
                pad_top = (target_height - height) // 2
                pad_bottom = target_height - height - pad_top
                pad_left = pad_right = 0
            else:
                # Too tall: pad width
                target_width = int(height / max_aspect_ratio)
                pad_left = (target_width - width) // 2
                pad_right = target_width - width - pad_left
                pad_top = pad_bottom = 0

            img_array = imgmsg_to_cv2(image)
            padded_img = cv2.copyMakeBorder(
                img_array,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
            padded_ros_img = Image()
            padded_ros_img.header = image.header
            padded_ros_img.height = padded_img.shape[0]
            padded_ros_img.width = padded_img.shape[1]
            padded_ros_img.encoding = "bgr8"  # OpenCV uses BGR format
            padded_ros_img.step = padded_ros_img.width * 3
            padded_ros_img.data = padded_img.tobytes()
            return padded_ros_img, [pad_top, pad_bottom, pad_left, pad_right]
        return image, [0, 0, 0, 0]  # No padding needed

    def _get_visualisation_objects(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        image1: PILImage.Image,
        image2: PILImage.Image,
        title: str = "",
    ) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
        """Create visualization object for the correspondence points.

        Args:
            points1 (np.ndarray): Points in the first image.
                Dimensions should be (N, 2).
            points2 (np.ndarray): Points in the second image.
                Dimensions should be (N, 2).
            image1 (PILImage.Image): First image as a PIL Image.
            image2 (PILImage.Image): Second image as a PIL Image.
            title (str): Title for the visualization.

        Returns:
            Tuple[plt.Figure, plt.Figure, plt.Figure]: A tuple containing the
            figures for the first image, second image, and the combined visualization.
        """

        # Convert points to list of tuples
        points1_tuples = [(int(x), int(y)) for x, y in points1]
        points2_tuples = [(int(x), int(y)) for x, y in points2]

        # Get figures for the images
        fig1, fig2 = draw_correspondences(
            points1_tuples, points2_tuples, image1, image2
        )

        # Add correspondence point labels
        ax1 = fig1.axes[0]
        ax2 = fig2.axes[0]
        for i in range(len(points1_tuples)):
            y1, x1 = points1_tuples[i]
            y2, x2 = points2_tuples[i]
            ax1.text(x1, y1, f"p{i}", color="red", fontsize=12)
            ax2.text(x2, y2, f"p{i}", color="red", fontsize=12)

        # Create a new figure for side-by-side display
        fig1_size = fig1.get_size_inches()
        fig2_size = fig2.get_size_inches()
        figsize = (fig1_size[0] + fig2_size[0], max(fig1_size[1], fig2_size[1]))
        combined_fig = plt.figure(figsize=figsize)

        # Extract the figure's canvas as an array
        for i, fig in enumerate([fig1, fig2]):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img = PILImage.open(buf)

            # Add subplot and display the image
            ax = combined_fig.add_subplot(1, 2, i + 1)
            ax.imshow(np.asarray(img))
            ax.axis("off")

        # Close the original figures to avoid memory leaks
        plt.close(fig1)
        plt.close(fig2)

        # Set the title and adjust layout
        combined_fig.suptitle(title)
        combined_fig.tight_layout()
        return fig1, fig2, combined_fig

    def visualize(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        image1: PILImage.Image,
        image2: PILImage.Image,
        title: str = "",
    ) -> None:
        """Visualize the correspondence points between two images.

        Args:
            points1 (np.ndarray): Points in the first image. Dimensions should be (N, 2)
            points2 (np.ndarray): Points in the second image. Dimensions should be (N, 2)
            image1 (PILImage.Image): First image as a PIL Image.
            image2 (PILImage.Image): Second image as a PIL Image.
        """

        self._get_visualisation_objects(points1, points2, image1, image2, title)
        plt.show()

    def save_visualization(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        image1: PILImage.Image,
        image2: PILImage.Image,
        output_dir: str,
        postfix: str = "",
        title: str = "",
    ) -> None:
        """Save the visualization of the correspondence points between two images.

        Args:
            points1 (np.ndarray): Points in the first image. Dimensions should be (N, 2)
            points2 (np.ndarray): Points in the second image. Dimensions should be (N, 2)
            image1 (PILImage.Image): First image as a PIL Image.
            image2 (PILImage.Image): Second image as a PIL Image.
            output_dir (str): Folder to save the visualization in.
            postfix (str): Postfix to append to the output filenames before the extension.
            title (str): Title for the visualization.
        """

        fig1, fig2, combined_fig = self._get_visualisation_objects(
            points1, points2, image1, image2, title
        )
        fig1.savefig(
            os.path.join(output_dir, f"(ce)_out1{postfix}_{self.n_requests:04d}.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        fig2.savefig(
            os.path.join(output_dir, f"(ce)_out2{postfix}_{self.n_requests:04d}.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        combined_fig.savefig(
            os.path.join(
                output_dir, f"(ce)_out_combined{postfix}_{self.n_requests:04d}.png"
            ),
            bbox_inches="tight",
        )
        plt.close("all")

    def _out_dir_callback(self, msg: String):
        """
        Callback function for the output directory topic subscriber.
        Sets the output directory based on the received message.
        Args:
            msg (String): The message containing the output directory path.
        """
        if self.out_dir != msg.data:
            self.out_dir = msg.data
            rospy.loginfo(f"Output directory set to: {self.out_dir}")


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
