#!/usr/bin/env python3
# Publishes tf's for debugging purposes.
from omegaconf import DictConfig
import os
import rospy
import tf
import tf.transformations
import yaml

from geometry_msgs.msg import TransformStamped

TRANSLATION = (0.1, 0.0, 0.2)  # (x, y, z) in meters
ROTATION = (0.0, 0.0, 0.0)  # (roll, pitch, yaw) in radians


def publish_static_tf(cfg: DictConfig):

    rospy.init_node("camera_static_tf_publisher")

    # Create a tf broadcaster
    br = tf.TransformBroadcaster()

    # Set the rate for publishing (10 Hz is usually sufficient for static transforms)
    rate = rospy.Rate(10.0)

    # Define the transform parameters
    translation = TRANSLATION  # (x, y, z) in meters

    # Define rotation as quaternion [x, y, z, w]
    # Here we're using no rotation (identity quaternion)
    rotation = tf.transformations.quaternion_from_euler(
        ROTATION[0], ROTATION[1], ROTATION[2]
    )  # (roll, pitch, yaw) in radians

    # Parent and child frame IDs
    parent_frame = cfg.ros.frame_ids.base
    child_frame = cfg.ros.frame_ids.camera

    rospy.loginfo(
        "Publishing static transform from '{}' to '{}'".format(
            parent_frame, child_frame
        )
    )

    while not rospy.is_shutdown():
        # Get current timestamp
        current_time = rospy.Time.now()

        # Publish the transform
        br.sendTransform(translation, rotation, current_time, child_frame, parent_frame)

        # Sleep to maintain the publishing rate
        rate.sleep()


if __name__ == "__main__":
    try:
        config_path = os.path.join(os.path.dirname(__file__), "../config/default.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        cfg = DictConfig(config)
        publish_static_tf(cfg)
    except rospy.ROSInterruptException:
        pass
