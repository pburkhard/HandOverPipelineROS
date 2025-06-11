import numpy as np
from std_msgs.msg import Int32MultiArray, MultiArrayDimension
from geometry_msgs.msg import Transform
import tf.transformations


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
    elif image_msg.encoding == "mono8":
        dtype = np.uint8
        img = np.frombuffer(image_msg.data, dtype=dtype).reshape(
            (image_msg.height, image_msg.width)
        )
        return img
    else:
        raise ValueError(f"Unsupported encoding: {image_msg.encoding}")


def np_to_multiarraymsg(
    np_array: np.ndarray, array_type, dimension_labels: list = None
):
    """
    Convert a numpy array to a ROS multiarray message.

    Args:
        np_array (np.ndarray): The numpy array to convert.
        array_type (type): The type of the multiarray message (e.g., Float32MultiArray).
        dimension_labels (list, optional): Labels for the dimensions of the multiarray.

    Returns:
        A ROS multiarray message containing the data from the numpy array.
    """

    if array_type == Int32MultiArray:
        msg = Int32MultiArray()
    else:
        raise ValueError("Unsupported array type")

    if dimension_labels is not None and len(dimension_labels) != np_array.ndim:
        raise ValueError(
            "Number of dimension labels must match the number of dimensions in the numpy array"
        )
    n = np_array.ndim
    sizes = np_array.shape

    # Use empty strings as default labels for dimensions when no labels are provided.
    dimension_labels = dimension_labels or [""] * n

    msg.data = np_array.flatten().tolist()
    for i in range(np_array.ndim):
        msg.layout.dim.append(
            MultiArrayDimension(
                size=sizes[i],
                stride=int(np.prod(sizes[i:n])),
                label=dimension_labels[i],
            )
        )

    return msg


def multiarraymsg_to_np(multiarray_msg):
    """
    Convert a ROS multiarray message to a numpy array.

    Args:
        multiarray_msg: The ROS multiarray message to convert.

    Returns:
        np.ndarray: The numpy array containing the data from the multiarray message.
    """
    return np.array(multiarray_msg.data).reshape(
        [dim.size for dim in multiarray_msg.layout.dim]
    )


def np_to_transformmsg(np_array: np.ndarray):
    """
    Convert a numpy array to a transformation matrix.

    Args:
        np_array (np.ndarray): The numpy array to convert, expected to be of shape (4, 4).

    Returns:
        Transform: A ROS Transform message containing the translation and rotation.
    """
    if np_array.shape != (4, 4):
        raise ValueError("Input numpy array must be of shape (4, 4)")

    # Extract translation
    translation = np_array[:3, 3]
    # Extract rotation as quaternion
    quaternion = tf.transformations.quaternion_from_matrix(np_array)

    transform_msg = Transform()
    transform_msg.translation.x = translation[0]
    transform_msg.translation.y = translation[1]
    transform_msg.translation.z = translation[2]
    transform_msg.rotation.x = quaternion[0]
    transform_msg.rotation.y = quaternion[1]
    transform_msg.rotation.z = quaternion[2]
    transform_msg.rotation.w = quaternion[3]

    return transform_msg
