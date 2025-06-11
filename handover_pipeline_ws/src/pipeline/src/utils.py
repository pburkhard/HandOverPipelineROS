import numpy as np
from sensor_msgs.msg import Image


def cv2_to_imgmsg(image, encoding="bgr8"):
    """
    Convert a cv2 image to a ROS Image message.
    Args:
        image: The cv2 image to convert.
        encoding: The encoding of the image (default is "bgr8").
    Returns:
        A ROS Image message.
    """

    msg = Image()
    msg.height = image.shape[0]
    msg.width = image.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = image.dtype.byteorder == ">" or (
        image.dtype.byteorder == "=" and np.byteorder == "big"
    )
    msg.step = image.strides[0]
    msg.data = image.tobytes()
    return msg


def imgmsg_to_cv2(image_msg):
    """
    Convert a ROS Image message to a cv2 image.
    Args:
        image_msg: The ROS Image message to convert.
    Returns:
        A cv2 image.
    """
    if image_msg.encoding == "bgr8":
        dtype = np.uint8
        channels = 3
        img = np.frombuffer(image_msg.data, dtype=dtype).reshape(
            (image_msg.height, image_msg.width, channels)
        )
        return img
    elif image_msg.encoding == "rgb8":
        dtype = np.uint8
        channels = 3
        img = np.frombuffer(image_msg.data, dtype=dtype).reshape(
            (image_msg.height, image_msg.width, channels)
        )
        # Convert RGB to BGR for OpenCV compatibility
        return img[..., ::-1]
    else:
        raise ValueError(f"Unsupported encoding: {image_msg.encoding}")
