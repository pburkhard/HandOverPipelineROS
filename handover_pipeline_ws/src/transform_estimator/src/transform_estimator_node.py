#!/usr/bin/env python3
import actionlib
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from omegaconf import DictConfig, OmegaConf
import open3d
import os
from scipy.optimize import minimize
import sys
import rospy
import yaml

# TODO: Remove dependency on the pipeline package
sys.path.append(os.path.join(os.path.dirname(__file__), "../../pipeline/src/"))
from msg_utils import (
    imgmsg_to_cv2,
    multiarraymsg_to_np,
    np_to_transformmsg,
)

from std_msgs.msg import String
from transform_estimator.msg import (
    EstimateTransformAction,
    EstimateTransformResult,
    EstimateTransformFeedback,
    EstimateTransformGoal,
)


class TransformEstimator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        rospy.init_node(cfg.ros.node_name, anonymous=True)

        self._feedback = EstimateTransformFeedback()
        self._result = EstimateTransformResult()
        self._server = actionlib.SimpleActionServer(
            cfg.ros.node_name,
            EstimateTransformAction,
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
            config_path = os.path.join(self.out_dir, "(te)_config.yaml")
            with open(config_path, "w") as f:
                OmegaConf.save(config=self.cfg, f=f.name)

        self._server.start()
        rospy.loginfo(f"{cfg.ros.node_name} action server started.")

    def _execute(self, goal: EstimateTransformGoal):
        """
        Callback function for the action server. It estimates the transformation
        between the object image and grasp image frames based on the provided 2D points
        and camera parameters.
        """
        rospy.loginfo(f"Received goal.")

        # Validate the goal
        if (
            not goal.object_camera_info
            or not goal.grasp_camera_info
            or not goal.object_image_depth
            or not goal.corr_points_object
            or not goal.corr_points_grasp
        ):
            rospy.logerr("Invalid goal: Missing required fields.")
            self._result.success = False
            self._server.set_aborted()
            return

        K_object = np.array(goal.object_camera_info.K, dtype=np.float64).reshape(3, 3)
        K_grasp = np.array(goal.grasp_camera_info.K, dtype=np.float64).reshape(3, 3)
        object_image_depth = imgmsg_to_cv2(goal.object_image_depth)
        corr_points_object = multiarraymsg_to_np(goal.corr_points_object)
        corr_points_grasp = multiarraymsg_to_np(goal.corr_points_grasp)

        self._feedback.status = "Estimating 3D points..."
        self._feedback.percent_complete = 0.0
        self._server.publish_feedback(self._feedback)

        # Lift the 2D points from the object camera frame into the 3D space
        corr_points_object_3D = self.get_3D_points(
            K_object, object_image_depth, corr_points_object
        )  # Values are in meters
        if self.cfg.debug.log_verbose:
            rospy.loginfo(
                f"Extracted 3D points of object (in meters):\n{corr_points_object_3D}"
            )

        # The depth image can be corrupted, cause some 3D points being zero, filter those out
        valid_idx = np.any(corr_points_object_3D != 0, axis=1)
        corr_points_grasp = corr_points_grasp[valid_idx]
        corr_points_object = corr_points_object[valid_idx]
        corr_points_object_3D = corr_points_object_3D[valid_idx]
        if valid_idx.shape[0] != corr_points_object.shape[0]:
            rospy.logwarn(
                f"Filtered out correspondence points due to corrupted depth image. {corr_points_object.shape[0]} points remaining."
            )

        rospy.loginfo("Extracted object points in 3D")
        if self.cfg.debug.log_3d_points:
            path = os.path.join(self.out_dir, "(te)_corr_points_object_3D.npy")
            np.save(path, corr_points_object_3D)
        if self.cfg.debug.log_visualization:
            path = os.path.join(self.out_dir, "(te)_3d_reconstruction_object.png")
            self.save_3d_reconstruction_visualization(
                intrinsic_matrix=K_object,
                points_3D=corr_points_object_3D,
                points_2D=corr_points_object,
                output_path=path,
            )

        # Get the relative distances between the points in the target frame
        # Those will be used to reconstruct the 3D points in the source frame
        relative_distances = self._get_relative_distances(corr_points_object_3D)

        # Reconstruct the 3D points in the grasp frame
        corr_points_grasp_3D, cost = self.reconstruct_3d_points(
            K_grasp, corr_points_grasp, relative_distances, corr_points_object_3D
        )
        rospy.loginfo(f"Reconstructed grasp points in 3D space with cost: {cost:.4f}")
        if self.cfg.debug.log_3d_points:
            path = os.path.join(self.out_dir, "(te)_corr_points_grasp_3D.npy")
            np.save(path, corr_points_grasp_3D)
        if self.cfg.debug.log_visualization:
            path = os.path.join(self.out_dir, "(te)_3d_reconstruction_grasp.png")
            self.save_3d_reconstruction_visualization(
                intrinsic_matrix=K_grasp,
                points_3D=corr_points_grasp_3D,
                points_2D=corr_points_grasp,
                output_path=path,
            )
        if self.cfg.debug.log_verbose:
            rospy.loginfo(
                f"Reconstructed 3D points in grasp frame (in meters):\n{corr_points_grasp_3D}"
            )

        # Estimate the transformation matrix
        self._feedback.status = "Estimating transformation matrix..."
        self._feedback.percent_complete = 50
        self._server.publish_feedback(self._feedback)
        transformation = self.reconstruct_tranformation(
            corr_points_grasp_3D,
            corr_points_object_3D,
        )
        if self.cfg.debug.log_verbose:
            rospy.loginfo(
                f"Estimated transformation matrix (grasp to object frame):\n{transformation}"
            )
        if self.cfg.debug.log_visualization:
            path = os.path.join(self.out_dir, "(te)_transformation_visualization.png")
            self.save_transform_visualization(
                source_points=corr_points_grasp_3D,
                target_points=corr_points_object_3D,
                transformation_matrix=transformation,
                output_path=path,
                source_label="Reconstructed Grasp Points",
                target_label="Object Points",
            )

        # Publish feedback
        self._feedback.status = "Transformation estimation completed."
        self._feedback.percent_complete = 100
        self._server.publish_feedback(self._feedback)
        rospy.loginfo("Transformation estimation completed.")

        # Set the result and mark the action as succeeded
        self._result.transform_grasp_to_object = np_to_transformmsg(transformation)
        self._server.set_succeeded(self._result)

    def reconstruct_tranformation(
        self, source_points: np.ndarray, target_points: np.ndarray
    ):
        """
        Estimates the transformation matrix that aligns the source points with
        the target points using open3d's registration method.

        Args:
            source_points (np.ndarray): Nx3 array of 3D points in the source frame.
            target_points (np.ndarray): Nx3 array of 3D points in the target frame.
        Returns:
            np.ndarray: 4x4 transformation matrix.
        """

        source = open3d.geometry.PointCloud()
        target = open3d.geometry.PointCloud()
        source.points = open3d.utility.Vector3dVector(source_points)
        target.points = open3d.utility.Vector3dVector(target_points)

        # Create correspondences explicitly (point i in source corresponds to point i in target)
        correspondences = np.array([(i, i) for i in range(len(source_points))])

        # TODO: Implement RANSAC if necessary
        transformation = open3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(
            source, target, open3d.utility.Vector2iVector(correspondences)
        )

        return transformation

    def reconstruct_3d_points(
        self,
        intrinsic_matrix: np.ndarray,
        image_points: np.ndarray,
        relative_distances: dict,
        initial_points_3D: np.ndarray = None,
    ) -> tuple[np.ndarray, float]:
        """
        Reconstructs 3D points from 2D image points using a pinhole camera model.
        The reconstruction is done by minimizing the reprojection error and
        enforcing distance constraints between points.

        Args:
            intrinsic_matrix (np.ndarray): 3x3 intrinsic camera matrix.
            image_points (np.ndarray): Nx2 array of 2D image points.
            relative_distances (dict): Dictionary with keys as tuples of point
                indices and values as distances.
                Example: {(0, 1): 2.0, (1, 2): 3.0}
            initial_points_3D (np.ndarray): Nx3 array containing the inital
                3D coordinates for the optimization.
        Returns:
            np.ndarray: Nx3 array of reconstructed 3D points.
            float: Final cost of the optimization.
        """
        n_points = len(image_points)
        normalized_points = self._pixel_to_normalized(intrinsic_matrix, image_points)

        # Initial guess
        if initial_points_3D is None:
            initial_depths = np.linspace(1, 10, n_points)
            initial_points_3D = np.empty((n_points, 3))
            for i in range(n_points):
                initial_points_3D[i] = (
                    np.array([normalized_points[i, 0], normalized_points[i, 1], 1])
                    * initial_depths[i]
                )

        # Flatten to 1D array for optimization: (x1, y1, z1, x2, y2, z2, ...)
        initial_params = initial_points_3D.flatten()

        def _reprojection_cost(points_3d):
            # Cost for reprojection error -> Project 3d points back to a fictional
            # image plane at z = 1 and compare with the original normalized 2D points
            reprojection_cost = 0
            for i in range(n_points):
                if points_3d[i, 2] <= 0:  # Ensure points are in front of camera
                    return 1e10
                projected = points_3d[i, :2] / points_3d[i, 2]
                reprojection_cost += np.sum((projected - normalized_points[i]) ** 2)

            return reprojection_cost

        def _distance_cost(points_3d):
            # Cost for distance constraints in 3D
            distance_cost = 0
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    if (i, j) in relative_distances or (j, i) in relative_distances:
                        key = (i, j) if (i, j) in relative_distances else (j, i)
                        target_dist = relative_distances[key]
                        actual_dist = np.linalg.norm(points_3d[i] - points_3d[j])
                        distance_cost += (actual_dist - target_dist) ** 2

            return distance_cost

        def objective_function(params):
            # Reshape flattened parameters back to 3D points
            points_3d = params.reshape(-1, 3)

            if points_3d.shape[0] != n_points:
                raise ValueError(
                    f"Expected {n_points} points, but got {points_3d.shape[0]}"
                )

            # Cost for reprojection error
            reprojection_cost = _reprojection_cost(points_3d)

            # Cost for distance constraints in 3D
            distance_cost = _distance_cost(points_3d)

            return (
                self.cfg.lambda_proj * reprojection_cost
                + self.cfg.lambda_dist * distance_cost
            )

        # Run optimization
        result = minimize(objective_function, initial_params, method="Powell")
        reconstructed_points = result.x.reshape(-1, 3)
        if self.cfg.debug.log_optimization_results:
            path = os.path.join(self.out_dir, "(te)_transform_estimation_results.yaml")
            with open(path, "w") as f:
                yaml.dump(
                    {
                        "lambda_proj": float(self.cfg.lambda_proj),
                        "lambda_dist": float(self.cfg.lambda_dist),
                        "reprojection_cost": float(
                            _reprojection_cost(reconstructed_points)
                        ),
                        "distance_cost": float(_distance_cost(reconstructed_points)),
                        "objective_function_value": float(result.fun),
                        "message": str(result.message),
                        "n_function_evaluations": int(result.nfev),
                        "n_iterations": int(result.nit),
                        "status": int(result.status),
                        "success": bool(result.success),
                    },
                    f,
                )

        return reconstructed_points, result.fun

    def get_3D_points(
        self,
        intrinsic_matrix: np.ndarray,
        depth_image: np.ndarray,
        points_2D: np.ndarray,
        average_around_patch=False,
    ) -> np.ndarray:
        """
        Get the 3D points corresponding to the given 2D points using the depth image
        and the intrinsic matrix.

        Args:
            intrinsic_matrix (np.ndarray): 3x3 intrinsic camera matrix.
            depth_image (np.ndarray): Depth image of shape (H,W) where each pixel
                value is the depth in millimeters.
            points_2D (np.ndarray): Nx2 array of 2D points in pixel coordinates.
            average_around_patch (bool): Whether to average the depth value around a
                patch of 3x3 pixels in the depth image.

        Returns:
            np.ndarray (shape (n,3)): 3D points corresponding to the 2D points. All
            values are in meters.
        """

        if depth_image.ndim != 2:
            raise ValueError("Depth image must be a 2D array (H, W).")

        # Get the intrinsic parameters
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        h, w = depth_image.shape

        # Get the 3D points
        points_3D = []
        for point in points_2D:
            y, x = point
            if average_around_patch:
                # Average around 3x3 grid
                Z_values = np.array(
                    [
                        depth_image[max(int(y) - 1, 0), max(int(x) - 1, 0)],
                        depth_image[max(int(y) - 1, 0), int(x)],
                        depth_image[max(int(y) - 1, 0), min(int(x) + 1, w - 1)],
                        depth_image[int(y), max(int(x) - 1, 0)],
                        depth_image[int(y), int(x)],
                        depth_image[int(y), min(int(x) + 1, w - 1)],
                        depth_image[min(int(y) + 1, h - 1), max(int(x) - 1, 0)],
                        depth_image[min(int(y) + 1, h - 1), int(x)],
                        depth_image[min(int(y) + 1, h - 1), min(int(x) + 1, w - 1)],
                    ]
                )
                # Filter out invalid values
                Z_values = Z_values[Z_values != 0].astype(np.float32)
                Z = np.mean(Z_values) * 0.001  # Convert to meters
                Z = 0.0 if np.isnan(Z) else Z
            else:
                Z = depth_image[int(y), int(x)] * 0.001
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy
            points_3D.append([X, Y, Z])

        return np.array(points_3D)

    #########################
    ### Utility functions ###
    #########################

    def _pixel_to_normalized(
        self, intrinsic_matrix: np.ndarray, image_points: np.ndarray
    ):
        """
        Normalizes 2D pixel coordinates (u, v) using the intrinsic matrix.
        The resulting coordinates are correspoding to a focal length of 1.

        Args:
            intrinsic_matrix (np.ndarray): 3x3 intrinsic camera matrix.
            image_points (np.ndarray): Nx2 array of pixel coordinates.

        Returns:
            np.ndarray: Nx2 array of normalized coordinates.
        """
        # We want float coordinates in the end
        image_points = image_points.astype(np.float32)
        normalized_points = np.empty_like(image_points)
        for i in range(len(image_points)):
            y, x = image_points[i]
            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]
            cx = intrinsic_matrix[0, 2]
            cy = intrinsic_matrix[1, 2]
            normalized_points[i, 0] = (x - cx) / fx
            normalized_points[i, 1] = (y - cy) / fy
        return normalized_points

    def _get_relative_distances(self, points_3D: np.ndarray):
        """
        Computes the relative distances between points in 3D space.
        Args:
            points_3D (np.ndarray): Nx3 array of 3D points.
        Returns:
            dict: Dictionary with keys as tuples of point indices and values as distances.
        """
        n_points = len(points_3D)
        relative_distances = {}
        for i in range(n_points):
            for j in range(i + 1, n_points):
                distance = np.linalg.norm(points_3D[i] - points_3D[j])
                relative_distances[(i, j)] = distance
        return relative_distances

    def _apply_transformation(
        self, points_3D: np.ndarray, transformation: np.ndarray
    ) -> np.ndarray:
        """
        Applies a transformation matrix to a set of 3D points.

        Args:
            points_3D (np.ndarray): Nx3 array of 3D points.
            transformation (np.ndarray): 4x4 transformation matrix.

        Returns:
            np.ndarray: Nx3 array of transformed 3D points.
        """
        # Convert points to homogeneous coordinates
        points_3D_homogeneous = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))
        # Apply transformation
        transformed_points = transformation @ points_3D_homogeneous.T
        return transformed_points[:3].T

    def _plot_camera_model(
        self,
        intrinsic_matrix: np.ndarray,
        points_2D: np.ndarray,
        points_3D: np.ndarray,
    ):
        """Plots a pinhole camera model with the camera center at (0, 0, 0)
        and the image plane at z = z_img_plane. The 2D image points are
        projected onto the image plane, and the 3D points are plotted in 3D space.
        The plot can be displayed by calling plt.show().

        Args:
            intrinsic_matrix (np.ndarray): 3x3 intrinsic camera matrix.
            points_2D (np.ndarray): Nx2 array of points in pixel coordinates.
              The origin is at the top left corner of the image, the first entry
              describes the heigh coordinate and the second one the width
            points_3D (np.ndarray): Nx3 array of the corresponding 3D coordinates.

        Returns:
            fig (plt.Figure): The figure object containing the plot.
            ax (plt.Axes3D): The 3D axes object for further customization.
        """

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Ensure consistent data types
        intrinsic_matrix = intrinsic_matrix.astype(np.float64)
        points_2D = points_2D.astype(np.float64)
        points_3D = points_3D.astype(np.float64)

        # Extract camera intrinsic parameters
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        # Compute image plane z position: half the minimum z of the 3D points
        min_z = np.min(points_3D[:, 2])
        z_img_plane = min_z / 2.0 if min_z > 0 else 1.0  # fallback to 1.0 if min_z <= 0

        if points_2D.min() < 0:
            raise Warning("Got negative pixel coordinates in image points")

        # Get the image points in 3D coordinates
        # Convert pixel coordinates to normalized image coordinates (focal length = z_image_plane)
        points_2D_projected = np.zeros_like(points_2D)
        for i in range(len(points_2D)):
            y, x = points_2D[i]
            points_2D_projected[i, 0] = (x - cx) / fx * z_img_plane
            points_2D_projected[i, 1] = (y - cy) / fy * z_img_plane

        # Plot camera center
        ax.scatter(
            [0],
            [0],
            [0],
            c="black",
            s=80,
            marker="*",
            label="Camera Center",
            zorder=10,
        )
        ax.text(0, 0, 0, "Camera", color="black", fontsize=12)

        # Plot the image plane at z = z_img_plane
        xx, yy = np.meshgrid(
            np.linspace(-cx, max(points_2D_projected[:, 0].max() * 1.2, 1), 10),
            np.linspace(-cy, max(points_2D_projected[:, 1].max() * 1.2, 1), 10),
        )
        zz = np.ones_like(xx) * z_img_plane
        ax.plot_surface(xx, yy, zz, alpha=0.1, color="gray")

        # Plot 2D image points on the image plane
        ax.scatter(
            points_2D_projected[:, 0],
            points_2D_projected[:, 1],
            z_img_plane,
            c="blue",
            s=50,
            label="2D Image Points",
            marker="o",
        )
        for i, pt in enumerate(points_2D_projected):
            ax.text(pt[0], pt[1], z_img_plane, f"P{i}", color="blue", fontsize=10)

        # Plot 3D points
        ax.scatter(
            points_3D[:, 0],
            points_3D[:, 1],
            points_3D[:, 2],
            c="red",
            s=50,
            label="Reconstructed 3D Points",
            marker="o",
        )
        for i, pt in enumerate(points_3D):
            ax.text(pt[0], pt[1], pt[2], f"P{i}", color="red", fontsize=10)

        # Draw lines from camera center (0,0,0) through image points to 3D points
        for i, point_2D in enumerate(points_2D_projected):
            # Draw line from camera center to image plane
            ax.plot(
                [0, point_2D[0]],
                [0, point_2D[1]],
                [0, z_img_plane],
                "g--",
                alpha=0.5,
            )
            # Draw line from image plane to reconstructed 3D point
            ax.plot(
                [point_2D[0], points_3D[i, 0]],
                [point_2D[1], points_3D[i, 1]],
                [z_img_plane, points_3D[i, 2]],
                "k-",
                alpha=0.7,
            )

        # Rescale axes to fit all content but keep the aspect ratio
        # Compute axis limits to fit all points and camera center, keeping aspect ratio
        xyz = np.vstack((points_3D, [[0, 0, 0]]))
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        center = (xyz_min + xyz_max) / 2
        size = (xyz_max - xyz_min).max() * 0.6 + 1e-3  # Add small epsilon to avoid zero
        ax.set_xlim(center[0] - size, center[0] + size)
        ax.set_ylim(center[1] - size, center[1] + size)
        ax.set_zlim(center[2] - size, center[2] + size)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=45)
        ax.legend()
        plt.tight_layout()

        return fig, ax

    def _plot_3D_points(
        self,
        points_3D: np.ndarray,
        color: str = "green",
        label: str = None,
        ax: Axes3D = None,
    ):
        """
        Plots 3D points in a 3D space. If an axes object is provided, it uses that
        for plotting.

        Args:
            points_3D (np.ndarray): Nx3 array of 3D points.
            color (str, optional): Color of the points. Defaults to "green".
            label (str, optional): Label for the points. Defaults to None.
            ax (Axes3D, optional): Axes object for plotting. Defaults to None.
        Returns:
            ax (Axes3D): The 3D axes object for further customization.
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            points_3D[:, 0],
            points_3D[:, 1],
            points_3D[:, 2],
            c=color,
            s=50,
            label=label,
            marker="o",
        )
        for i, pt in enumerate(points_3D):
            ax.text(pt[0], pt[1], pt[2], f"P{i}", color=color, fontsize=10)
        if label:
            ax.legend()

        return ax

    def visualize_3d_reconstruction(
        self,
        intrinsic_matrix: np.ndarray,
        points_3D: np.ndarray,
        points_2D: np.ndarray,
    ):
        """
        Visualizes the reconstruction process by plotting the original 3D points,
        the reconstructed 3D points, and the 2D image points.

        Args:
            intrinstic_matrix (np.ndarray): 3x3 intrinsic camera matrix.
            points_3D (np.ndarray): Nx3 array of original 3D points.
            reconstructed_points_3D (np.ndarray): Nx3 array of reconstructed 3D points.
            points_2D (np.ndarray): Nx2 array of 2D image points.
        """

        _, ax = self._plot_camera_model(intrinsic_matrix, points_2D, points_3D)
        plt.show()

    def save_3d_reconstruction_visualization(
        self,
        intrinsic_matrix: np.ndarray,
        points_3D: np.ndarray,
        points_2D: np.ndarray,
        output_path: str,
    ):
        """
        Saves the visualization of the reconstruction process by plotting the
        3D points, the reconstructed 3D points, and the 2D image points.

        Args:
            intrinstic_matrix (np.ndarray): 3x3 intrinsic camera matrix.
            points_3D (np.ndarray): Nx3 array of 3D points.
            points_2D (np.ndarray): Nx2 array of 2D image points.
            output_path (str): Path to save the visualization.
        """

        base, ext = os.path.splitext(output_path)
        _, ax = self._plot_camera_model(intrinsic_matrix, points_2D, points_3D)
        plt.savefig(output_path)
        ax.view_init(elev=0, azim=-90)
        plt.savefig(f"{base}_top_view{ext}")
        ax.view_init(elev=-90, azim=90, roll=180)
        plt.savefig(f"{base}_camera_view{ext}")
        plt.close()

    def visualize_transform(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        transformation_matrix: np.ndarray,
        source_label: str = None,
        target_label: str = None,
    ):
        """
        Visualizes the transformed source point alongside the actual target points
        for comparison.

        Args:
            source_points (np.ndarray): Nx3 array of 3D points in the source frame.
            target_points (np.ndarray): Nx3 array of 3D points in the target frame.
            transformation_matrix (np.ndarray): 4x4 transformation matrix.
            source_label (str): Label of the source points.
            target_label (str): Label of the target points.
        """
        if not source_label:
            source_label = "Transformed Source Points"
        if not target_label:
            target_label = "Transformed Target Points"
        transformed_points = self._apply_transformation(
            source_points, transformation_matrix
        )
        ax = self._plot_3D_points(transformed_points, color="red", label=source_label)
        self._plot_3D_points(target_points, color="blue", label=target_label, ax=ax)
        plt.title("Transformation Estimation")
        plt.show()

    def save_transform_visualization(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        transformation_matrix: np.ndarray,
        output_path: str,
        source_label: str = None,
        target_label: str = None,
    ):
        """
        Saves the visualization of the transformed source point alongside the actual
        target points for comparison.

        Args:
            source_points (np.ndarray): Nx3 array of 3D points in the source frame.
            target_points (np.ndarray): Nx3 array of 3D points in the target frame.
            transformation_matrix (np.ndarray): 4x4 transformation matrix.
            output_path (str): Path to save the visualization.
            source_label (str): Label of the source points.
            target_label (str): Label of the target points.
        """

        if not source_label:
            source_label = "Transformed Source Points"
        if not target_label:
            target_label = "Transformed Target Points"
        transformed_points = self._apply_transformation(
            source_points, transformation_matrix
        )
        ax = self._plot_3D_points(transformed_points, color="red", label=source_label)
        self._plot_3D_points(target_points, color="blue", label=target_label, ax=ax)
        plt.title("Transformation Estimation")
        plt.savefig(output_path)
        plt.close()

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
        TransformEstimator(cfg)
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("TransformEstimator node interrupted.")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
