#!/usr/bin/env python
import actionlib
import base64
import cv2
from dotenv import load_dotenv
from hydra import initialize, compose
import numpy as np
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI
from openai.types.images_response import ImagesResponse
import os
import requests
import rospy

from grasp_generator.msg import (
    GenerateGraspAction,
    GenerateGraspResult,
    GenerateGraspFeedback,
    GenerateGraspGoal,
)
from std_msgs.msg import String
from sensor_msgs.msg import Image

# Absolute path to the root directory of the package
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GraspGenerator:

    def __init__(self, cfg: DictConfig, api_key: str):

        self._api_client = OpenAI(api_key=api_key)
        self.cfg = cfg

        rospy.init_node(self.cfg.ros.node_name, anonymous=True)

        self._feedback = GenerateGraspFeedback()
        self._result = GenerateGraspResult()
        self._server = actionlib.SimpleActionServer(
            self.cfg.ros.node_name,
            GenerateGraspAction,
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
            config_path = os.path.join(self.out_dir, "(gg)_config.yaml")
            with open(config_path, "w") as f:
                OmegaConf.save(config=self.cfg, f=f.name)

        self._server.start()
        rospy.loginfo(f"{self.cfg.ros.node_name} action server started.")

    def _execute(self, goal: GenerateGraspGoal):
        """
        Callback function for the action server that generates a grasp image based on the input goal.
        Args:
            goal (GenerateGraspGoal): The goal containing the image path, object type, and task description.
        """
        rospy.loginfo(
            f"Received goal with object description: '{goal.object_description}'"
            + f" and task description: '{goal.task_description}'."
        )

        # Validate the goal
        if (
            not goal.object_image
            or not goal.object_description
            or not goal.task_description
        ):
            rospy.logerr(
                "Invalid goal: image_path, object_type, and task_description are required."
            )
            self._result.success = False
            self._server.set_aborted(self._result)
            return
        object_image = goal.object_image
        object_type = goal.object_description
        task_description = goal.task_description

        # Describe the object in the image
        self._feedback.status = "Describing original image..."
        self._feedback.percent_complete = 0.0
        self._server.publish_feedback(self._feedback)
        prompt = self.cfg.descriptor.prompt.replace("object", object_type)
        object_description = self._describe_image(object_image, prompt)
        rospy.loginfo(f"Object description: {object_description}")

        # Generate the grasp image
        self._feedback.status = "Generating grasp image..."
        self._feedback.percent_complete = 50.0
        self._server.publish_feedback(self._feedback)
        prompt = (
            f"A hand grasping a {object_type} ready to {task_description}. "
            + f"The {object_type} is described as folllows: {object_description} "
            + "The hand and the object are completely contained in the image. "
            + self.cfg.generator.grasp_prompt.replace("{task}", task_description)
            + f" {self.cfg.generator.background_prompt}"
        )
        rospy.loginfo(f"Image generation prompt: '{prompt}'")
        grasp_image = self._generate_image(prompt)

        if grasp_image is None:
            rospy.logerr("Failed to generate grasp image.")
            self._result.success = False
            self._result.grasp_image = None
            self._server.set_aborted(self._result)
            return
        rospy.loginfo("Grasp image generated successfully.")
        self._feedback.status = "Grasp image generated successfully."
        self._feedback.percent_complete = 100.0
        self._server.publish_feedback(self._feedback)
        self._result.grasp_image = grasp_image
        self._result.success = True
        self._server.set_succeeded(self._result)

    def _download_image(self, response: ImagesResponse) -> Image:
        """
        Download image from the response and return it as a list of PIL Images. This function
        only supports a single image in the response.
        Args:
            response (ImagesResponse): The response object containing image URL.
        Returns:
            sensor_msgs.Image: The downloaded image as a ROS sensor_msgs/Image.
        """
        if not response.data or len(response.data) == 0:
            rospy.logerr("No images found in the response.")
            return None
        if len(response.data) > 1:
            rospy.logwarn(
                "Multiple images found in the response. Only the first image will be processed."
            )

        # Get a OpenCV image from the response
        image_bytes = requests.get(response.data[0].url).content
        np_arr = np.frombuffer(image_bytes, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert OpenCV image to ROS Image message
        image = Image()
        image.height, image.width = cv_image.shape[:2]
        image.encoding = "bgr8"  # OpenCV uses BGR format
        image.data = cv_image.tobytes()
        image.step = cv_image.shape[1] * cv_image.shape[2]  # width * channels
        image.is_bigendian = False  # ROS Image is little-endian
        image.header.seq = 0  # Sequence number, can be set to 0 for simplicity
        image.header.frame_id = "grasp_generator_frame"
        image.header.stamp = rospy.Time.now()

        return image

    def _generate_image(self, prompt) -> Image:
        """
        Generate image based on the provided prompt and configuration.
        Returns:
            sensor_msgs.Image: The generated image as a ROS sensor_msgs/Image.
        """

        if self.cfg.debug.log_generation_prompt:
            path = os.path.join(self.out_dir, "(gg)_generation_prompt.txt")
            with open(path, "w") as f:
                f.write(prompt)

        # Make API request
        response = self._api_client.images.generate(
            prompt=prompt,
            model=self.cfg.generator.model,
            n=1,
            quality=self.cfg.generator.quality,
            size=self.cfg.generator.size,
            style=self.cfg.generator.style,
        )

        # Download and return the generated images
        return self._download_image(response)

    def _describe_image(self, image: Image, prompt: str) -> str:
        """
        Describe the given image using the OpenAI API.
        Args:
            image (Image): The ROS sensor_msgs/Image to be described.
            prompt (str): The prompt to use for the description.
        Returns:
            str: The description of the image.
        """
        # Convert ROS sensor_msgs/Image to OpenCV image
        cv_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            (image.height, image.width, -1)
        )
        # Encode as PNG
        _, buffer = cv2.imencode(".png", cv_image)
        # Convert to base64 string
        image = base64.b64encode(buffer).decode("utf-8")

        # Log the description prompt if enabled
        if self.cfg.debug.log_description_prompt:
            path = os.path.join(self.out_dir, "(gg)_description_prompt.txt")
            with open(path, "w") as f:
                f.write(self.cfg.descriptor.prompt)

        # Make API request
        response = self._api_client.chat.completions.create(
            model=self.cfg.descriptor.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                            },
                        },
                    ],
                }
            ],
        )

        # Return the description
        return response.choices[0].message.content

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

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="default.yaml")
        load_dotenv()  # Load environment variables from .env file
        api_key = os.getenv("OPENAI_API_KEY")
        try:
            GraspGenerator(cfg, api_key)
            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.loginfo("GraspGenerator node interrupted.")
        except Exception as e:
            rospy.logerr(f"Error: {e}")
