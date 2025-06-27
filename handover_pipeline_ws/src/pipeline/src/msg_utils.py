import numpy as np
import tf.transformations

from geometry_msgs.msg import Transform, Pose, TransformStamped, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, MultiArrayDimension


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
    msg.is_bigendian = (
        1
        if (
            image.dtype.byteorder == ">"
            or (image.dtype.byteorder == "=" and image.dtype.name == "big")
        )
        else 0
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
    elif image_msg.encoding == "mono8":
        dtype = np.uint8
        img = np.frombuffer(image_msg.data, dtype=dtype).reshape(
            (image_msg.height, image_msg.width)
        )
        return img
    elif image_msg.encoding == "16UC1":
        dtype = np.uint16
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
        msg.data = np_array.astype(np.int32).flatten().tolist()
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


def transformmsg_to_np(transform_msg: Transform):
    """
    Convert a ROS Transform message to a numpy array.

    Args:
        transform_msg (Transform): The ROS Transform message to convert.

    Returns:
        np.ndarray: A numpy array of shape (4, 4) representing the transformation matrix.
    """
    translation = np.array(
        [
            transform_msg.translation.x,
            transform_msg.translation.y,
            transform_msg.translation.z,
        ]
    )

    quaternion = np.array(
        [
            transform_msg.rotation.x,
            transform_msg.rotation.y,
            transform_msg.rotation.z,
            transform_msg.rotation.w,
        ]
    )

    rotation_matrix = tf.transformations.quaternion_matrix(quaternion)
    rotation_matrix[:3, 3] = translation

    return rotation_matrix


def transformmsg_to_posemsg(transform_msg: Transform):
    """
    Convert a ROS Transform message to a Pose message.

    Args:
        transform_msg (Transform): The ROS Transform message to convert.

    Returns:
        geometry_msgs.msg.Pose: A Pose message representing the transformation.
    """

    pose_msg = Pose()
    pose_msg.position.x = transform_msg.translation.x
    pose_msg.position.y = transform_msg.translation.y
    pose_msg.position.z = transform_msg.translation.z
    pose_msg.orientation = transform_msg.rotation

    return pose_msg


def transformmsg_to_posemsg_stamped(transform_stamped: TransformStamped):
    """
    Convert a ROS TransformStamped message to a PoseStamped message.

    Args:
        transform_stamped (TransformStamped): The ROS TransformStamped message to convert.

    Returns:
        geometry_msgs.msg.PoseStamped: A PoseStamped message representing the transformation.
    """
    pose_stamped = PoseStamped()
    pose_stamped.header = transform_stamped.header
    pose_stamped.pose.position.x = transform_stamped.transform.translation.x
    pose_stamped.pose.position.y = transform_stamped.transform.translation.y
    pose_stamped.pose.position.z = transform_stamped.transform.translation.z
    pose_stamped.pose.orientation = transform_stamped.transform.rotation

    return pose_stamped
