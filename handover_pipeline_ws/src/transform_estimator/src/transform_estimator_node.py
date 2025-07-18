#!/usr/bin/env python3
import actionlib
import cv2
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from omegaconf import DictConfig, OmegaConf
import open3d
import os
from scipy.optimize import least_squares
from sklearn.decomposition import PCA
import sys
from typing import Tuple
import rospy
import yaml

# Import custom message utilities from the pipeline package
sys.path.append(os.path.join(os.path.dirname(__file__), "../../pipeline/src/"))
from msg_utils import (
    imgmsg_to_cv2,
    multiarraymsg_to_np,
    np_to_transformmsg,
    transformmsg_to_np,
)

from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String, Float32
from transform_estimator.msg import (
    EstimateTransformAction,
    EstimateTransformResult,
    EstimateTransformFeedback,
    EstimateTransformGoal,
    EstimateTransformHeuristicAction,
    EstimateTransformHeuristicResult,
    EstimateTransformHeuristicFeedback,
    EstimateTransformHeuristicGoal,
)


class TransformEstimator:
    """Class for estimating the transform between the robot camera frame and the gen camera frame."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        rospy.init_node(cfg.ros.node_name, anonymous=True)

        self._feedback = EstimateTransformFeedback()
        self._result = EstimateTransformResult()
        self._server = actionlib.SimpleActionServer(
            cfg.ros.actions.estimate_transform,
            EstimateTransformAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        self._feedback_heuristic = EstimateTransformHeuristicFeedback()
        self._result_heuristic = EstimateTransformHeuristicResult()
        self._server_heuristic = actionlib.SimpleActionServer(
            cfg.ros.actions.estimate_transform_heuristic,
            EstimateTransformHeuristicAction,
            execute_cb=self._execute_heuristic,
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

        self.n_requests = 0  # Keep track of the number of requests

        self._server.start()
        self._server_heuristic.start()
        rospy.loginfo(f"action servers started.")

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

        self.n_requests += 1

        # print(
        #     f"Got intrinsic matrices:\n{goal.object_camera_info.K}\n{goal.grasp_camera_info.K}"
        # )

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
            path = os.path.join(
                self.out_dir, f"(te)_corr_points_object_3D_{self.n_requests:04d}.npy"
            )
            np.save(path, corr_points_object_3D)
        if self.cfg.debug.log_visualization:
            path = os.path.join(
                self.out_dir, f"(te)_3d_reconstruction_object_{self.n_requests:04d}.png"
            )
            self.save_3d_reconstruction_visualization(
                intrinsic_matrix=K_object,
                points_3D=corr_points_object_3D,
                points_2D=corr_points_object,
                output_path=path,
            )

        costs = []
        transforms = []
        focal_lengths = [
            float(K_grasp[0, 0] + K_grasp[1, 1]) / 2.0
        ]  # We only try one value for the focal length. Feel free to add more values here.

        for f in focal_lengths:
            K_grasp[0, 0] = f
            K_grasp[1, 1] = f

            rospy.loginfo(f"Estimating transformation with focal length: {f:.4f}")

            try:
                transform_robot_cam_to_gen_cam, cost = self.ransac_pnp_robust(
                    intrinsic_matrix=K_grasp,
                    points_image=corr_points_grasp[
                        :, [1, 0]
                    ],  # Use (width, height) for OpenCV
                    points_3D=corr_points_object_3D,
                )
                costs.append(cost)
                transforms.append(transform_robot_cam_to_gen_cam)
            except Exception as e:
                rospy.logerr(f"RANSAC PnP failed: {e}")
                continue

        if not costs:
            rospy.logerr("RANSAC PnP failed for all focal lengths.")
            self._result.success = False
            self._server.set_aborted()
            return

        # Find the focal length with the lowest cost
        best_idx = np.argmin(costs)
        cost = costs[best_idx]
        f = focal_lengths[best_idx]
        K_grasp[0, 0] = f
        K_grasp[1, 1] = f
        transform_robot_cam_to_gen_cam = transforms[best_idx]
        rospy.loginfo(f"Best focal length: {f:.4f} with cost: {costs[best_idx]:.4f}")

        corr_points_grasp_3D = self._apply_transformation(
            corr_points_object_3D, transform_robot_cam_to_gen_cam
        )
        rospy.loginfo(f"Estimated transform with cost: {cost:.4f}")
        if self.cfg.debug.log_3d_points:
            path = os.path.join(
                self.out_dir, f"(te)_corr_points_grasp_3D_{self.n_requests:04d}.npy"
            )
            np.save(path, corr_points_grasp_3D)
        if self.cfg.debug.log_verbose:
            rospy.loginfo(
                f"Reconstructed 3D points in grasp frame (in meters):\n{corr_points_grasp_3D}"
            )
            rospy.loginfo(
                f"Estimated transformation matrix (robot camera to gen camera frame):\n{transform_robot_cam_to_gen_cam}"
            )

        if self.cfg.debug.log_visualization:
            path = os.path.join(
                self.out_dir, f"(te)_3d_reconstruction_grasp_{self.n_requests:04d}.png"
            )
            self.save_3d_reconstruction_visualization(
                intrinsic_matrix=K_grasp,
                points_3D=corr_points_grasp_3D,
                points_2D=corr_points_grasp,
                output_path=path,
            )

        # Publish feedback
        self._feedback.status = "Transformation estimation completed."
        self._feedback.percent_complete = 100
        self._server.publish_feedback(self._feedback)
        rospy.loginfo("Transformation estimation completed.")

        # Set the result and mark the action as succeeded
        self._result.success = True
        self._result.mse = Float32(cost)
        self._result.transform_robot_cam_to_gen_cam = np_to_transformmsg(
            transform_robot_cam_to_gen_cam
        )
        self._result.grasp_camera_info = CameraInfo()
        self._result.grasp_camera_info.K = K_grasp.flatten().tolist()
        self._server.set_succeeded(self._result)

    def _execute_heuristic(self, goal: EstimateTransformHeuristicGoal):
        """
        Callback function for the heuristic action server. It estimates the transformation
        from the hand pose to the robot camera frame based on the provided data. This method
        uses a heuristic approach based on the wrist keypoint and hand orientation.
        """
        rospy.loginfo(f"Received heuristic goal.")

        # Validate the goal
        if (
            not goal.object_camera_info
            or not goal.object_image_depth
            or not goal.corr_points_object
            or not goal.corr_points_grasp
            or not goal.transform_hand_pose_to_camera
            or not goal.hand_keypoints
        ):
            rospy.logerr("Invalid heuristic goal: Missing required fields.")
            self._result_heuristic.success = False
            self._server_heuristic.set_aborted()
            return

        self.n_requests += 1

        K_object = np.array(goal.object_camera_info.K, dtype=np.float64).reshape(3, 3)
        object_image_depth = imgmsg_to_cv2(goal.object_image_depth)
        corr_points_object = multiarraymsg_to_np(goal.corr_points_object)
        corr_points_grasp = multiarraymsg_to_np(goal.corr_points_grasp)
        wrist_keypoint = multiarraymsg_to_np(goal.hand_keypoints).reshape(-1, 2)[
            0, [1, 0]
        ]  # Extract wrist (first keypoint)
        hand_orient = transformmsg_to_np(goal.transform_hand_pose_to_camera)[
            :3, :3
        ]  # Extract rotation part

        self._feedback_heuristic.status = "Estimating 3D points..."
        self._feedback_heuristic.percent_complete = 0.0
        self._server_heuristic.publish_feedback(self._feedback_heuristic)

        # Lift the 2D points from the object camera frame into the 3D space
        corr_points_object_3D = self.get_3D_points(
            K_object, object_image_depth, corr_points_object
        )
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
            path = os.path.join(
                self.out_dir, f"(te)_corr_points_object_3D_{self.n_requests:04d}.npy"
            )
            np.save(path, corr_points_object_3D)
        if self.cfg.debug.log_visualization:
            path = os.path.join(
                self.out_dir, f"(te)_3d_reconstruction_object_{self.n_requests:04d}.png"
            )
            self.save_3d_reconstruction_visualization(
                intrinsic_matrix=K_object,
                points_3D=corr_points_object_3D,
                points_2D=corr_points_object,
                output_path=path,
            )

        self._feedback_heuristic.status = "Estimating transformation..."
        self._feedback_heuristic.percent_complete = 50.0
        self._server_heuristic.publish_feedback(self._feedback_heuristic)

        # Estimate the transformation using the heuristic method
        try:
            transform_hand_pose_to_robot_camera, cost = (
                self.estimate_transformation_heuristic(
                    points_image=corr_points_grasp,
                    wrist_keypoint=wrist_keypoint,
                    hand_global_orient=hand_orient,
                    points_3D=corr_points_object_3D,
                )
            )
        except Exception as e:
            rospy.logerr(f"Heuristic transformation estimation failed: {e}")
            self._result_heuristic.success = False
            self._server_heuristic.set_aborted()
            return

        rospy.loginfo(f"Estimated transform with cost: {cost:.4f}")
        if self.cfg.debug.log_verbose:
            rospy.loginfo(
                f"Estimated transformation matrix (hand pose to robot camera):\n{transform_hand_pose_to_robot_camera}"
            )

        # Publish feedback
        self._feedback_heuristic.status = "Transformation estimation completed."
        self._feedback_heuristic.percent_complete = 100
        self._server_heuristic.publish_feedback(self._feedback_heuristic)
        rospy.loginfo("Transformation estimation completed.")

        # Set the result and mark the action as succeeded
        self._result_heuristic.success = True
        self._result_heuristic.cost = Float32(cost)
        self._result_heuristic.transform_hand_pose_to_robot_camera = np_to_transformmsg(
            transform_hand_pose_to_robot_camera
        )
        self._server_heuristic.set_succeeded(self._result_heuristic)

    def ransac_pnp_robust(
        self,
        intrinsic_matrix: np.ndarray,
        points_image: np.ndarray,
        points_3D: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        # 1. Adaptive threshold
        ransac_thresh = max(2 * self.cfg.noise_std, 8)
        # 2. MAGSAC++ for robust outlier handling
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points_3D.astype(np.float32),
            imagePoints=points_image.astype(np.float32),
            cameraMatrix=intrinsic_matrix.astype(np.float32),
            distCoeffs=None,
            flags=cv2.USAC_MAGSAC,
            reprojectionError=ransac_thresh,
            iterationsCount=10000,
            confidence=0.99,
        )
        if not success or inliers is None or len(inliers) < 4:
            raise ValueError(
                "RANSAC PnP failed to estimate the transformation. "
                "Check the input points and camera parameters."
            )
        inliers = inliers.flatten()
        points_3D_inliers = points_3D[inliers]
        image_points_inliers = points_image[inliers]
        # 3. Refine using OpenCV VVS refinement
        rvec, tvec = cv2.solvePnPRefineVVS(
            objectPoints=points_3D_inliers.astype(np.float32),
            imagePoints=image_points_inliers.astype(np.float32),
            cameraMatrix=intrinsic_matrix.astype(np.float32),
            distCoeffs=None,
            rvec=rvec,
            tvec=tvec,
        )
        # 4. Further reject inliers with large post-refinement errors
        projected_points, _ = cv2.projectPoints(
            objectPoints=points_3D_inliers.astype(np.float32),
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=intrinsic_matrix.astype(np.float32),
            distCoeffs=None,
        )
        reprojection_errors = np.linalg.norm(
            projected_points[:, 0, :] - image_points_inliers, axis=1
        )
        # 2.5x the normal noise as a cutoff
        mask = reprojection_errors < (2.5 * self.cfg.noise_std)
        points_3D_final = points_3D_inliers[mask]
        image_points_final = image_points_inliers[mask]
        print(
            f"Post-refinement: using {len(points_3D_final)}/{len(points_3D_inliers)} inliers for robust optimization"
        )
        if len(points_3D_final) < 4:
            points_3D_final = points_3D_inliers
            image_points_final = image_points_inliers
        # 5. Robust nonlinear refinement with Huber loss
        rvec, tvec = self._refine_pose_robust(
            points_3D_final, image_points_final, intrinsic_matrix, rvec, tvec
        )
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = tvec.flatten()
        # Final reprojection error (Huber-refined inliers only)
        projected_points, _ = cv2.projectPoints(
            objectPoints=points_3D_final.astype(np.float32),
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=intrinsic_matrix.astype(np.float32),
            distCoeffs=None,
        )
        reprojection_error = np.mean(
            np.linalg.norm(projected_points[:, 0, :] - image_points_final, axis=1)
        )
        rospy.loginfo(
            f"RANSAC PnP inliers: {len(inliers)} from {len(points_image)} total points"
        )

        projected_points_total, _ = cv2.projectPoints(
            objectPoints=points_3D.astype(np.float32),
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=intrinsic_matrix.astype(np.float32),
            distCoeffs=None,
        )
        reprojection_error_total = np.mean(
            np.linalg.norm(projected_points_total[:, 0, :] - points_image, axis=1)
        )

        if self.cfg.debug.log_optimization_results:
            results_path = os.path.join(
                self.out_dir,
                f"(te)_transform_estimation_results_{self.n_requests:04d}.yaml",
            )
            with open(results_path, "w") as f:
                yaml.dump(
                    {
                        "transformation_matrix": transformation_matrix.tolist(),
                        "translation": tvec.flatten().tolist(),
                        "rotation_vector": rvec.flatten().tolist(),
                        "reprojection_error (inliers only)": reprojection_error.tolist(),
                        "reprojection_error (all points)": reprojection_error_total.tolist(),
                        "inliers": inliers.flatten().tolist(),
                        "number of_inliers": len(inliers),
                        "number of_points": len(points_image),
                    },
                    f,
                )
            rospy.loginfo(f"RANSAC PnP results saved to {results_path}")

        if self.cfg.debug.log_visualization:
            points_image = points_image[:, [1, 0]]  # Swap x,y for visualization
            normalized_points = self._pixel_to_normalized(
                intrinsic_matrix, points_image
            )

            points_3D_transformed = self._apply_transformation(
                points_3D, transformation_matrix
            )
            projected_points = (
                points_3D_transformed[:, :2] / points_3D_transformed[:, 2, np.newaxis]
            )

            path = os.path.join(
                self.out_dir,
                f"(te)_point_projection_final_{self.n_requests:04d}.png",
            )
            self.save_2d_visualization(
                normalized_points,
                projected_points,
                path,
                "Keypoints of grasp image",
                "Projected 3D points",
            )

        return transformation_matrix, reprojection_error_total

    def _refine_pose_robust(
        self,
        points_3D,
        image_points,
        intrinsic_matrix,
        rvec_init,
        tvec_init,
        loss="huber",
        f_scale=15.0,
        max_nfev=500,
    ):
        def residuals(params, object_points, image_points, K):
            rvec = params[:3].reshape(3, 1)
            tvec = params[3:].reshape(3, 1)
            projected, _ = cv2.projectPoints(
                objectPoints=object_points.astype(np.float32),
                rvec=rvec,
                tvec=tvec,
                cameraMatrix=K.astype(np.float32),
                distCoeffs=None,
            )
            residual = projected[:, 0, :] - image_points
            return residual.ravel()

        params0 = np.hstack([rvec_init.flatten(), tvec_init.flatten()])
        result = least_squares(
            residuals,
            params0,
            args=(points_3D, image_points, intrinsic_matrix),
            method="trf",
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
        )
        rvec_refined = result.x[:3].reshape(3, 1)
        tvec_refined = result.x[3:].reshape(3, 1)
        return rvec_refined, tvec_refined

    def estimate_transformation_heuristic(
        self,
        points_image: np.ndarray,
        wrist_keypoint: np.ndarray,
        hand_global_orient: np.ndarray,
        points_3D: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Estimates the transformation matrix using a heuristic approach based on the
        centroid of the points and the principal axes of the point clouds.

        Args:
            points_image (np.ndarray): Nx2 array of 2D image points in pixel coordinates
                The first column is the height and the second column is the width.
            wrist_keypoint (np.ndarray): 2D coordinates of the wrist keypoint in pixel
                The first column is the height and the second column is the width.
            hand_global_orient (np.ndarray): 3D orientation as a (3,3) rotation matrix,
                specified in the (virtual) camera frame that has taken the image with the
                points_image and the wrist_keypoint.
            points_3D (np.ndarray): Nx3 array of 3D points in meters, the frame in which
                the coordinates are expressed is arbitrary.

        Returns:
            Tuple[np.ndarray, float]: The estimated transformation from the center of the
                image point cloud (interpreted as 3D point) to the origin of the 3D points,
                which coincides with the object camera frame. Additionally, a sanity value
                is returned (lower is better).
        """
        # frame 1: Coordinate frame in which the 3D points are expressed
        # frame 2: Coordinate frame defined by the principal axes of the 3D points, originating in the centroid of the 3D points
        # frame 3: Coordinate frame defined by the principal axes of the image points, originating in the centroid of the image points (pixel coordinates)
        # frame 4: Coordinate frame in which the image points are expressed (pixel coordinates),
        # frame 5: Coordinate frame defined by the hand pose, originating in the wrist keypoint
        # We want the transformation from frame 5 to frame 1
        # Between frame 2 and 3, there is a scaling factor

        points_image = points_image.astype(np.float32)[
            :, [1, 0]
        ]  # flip to (width, height) format
        wrist_keypoint = wrist_keypoint.astype(np.float32)[
            [1, 0]
        ]  # flip to (width, height) format
        hand_global_orient = hand_global_orient
        points_3D = points_3D.astype(np.float32)

        # Calculate centroids
        centroid_image = np.mean(points_image, axis=0)  # in pixel coords (2D)
        centroid_3D = np.mean(points_3D, axis=0)  # in meters (3D)

        # Perform PcA to find the principal axes
        pca_points_3D = PCA()
        pca_points_3D.fit(points_3D)
        points_3D_axes = pca_points_3D.components_
        pca_image_points = PCA()
        pca_image_points.fit(points_image)
        image_point_axes = pca_image_points.components_

        # Get the wrist keypoint in the principal axes frame (still in pixel coordinates)
        wrist_keypoint_frame3 = (wrist_keypoint - centroid_image) @ image_point_axes.T

        # Get the hand orientation in the principal axes frame (in 3D -> we leave the z-axis as is)
        image_point_axes_3D = np.eye(3, dtype=np.float32)
        image_point_axes_3D[:2, :2] = image_point_axes
        hand_global_orient_frame2 = hand_global_orient @ image_point_axes_3D.T

        # Align the orientation of the principal axes (they might point in opposite directions)
        points_image_frame3 = (points_image - centroid_image) @ image_point_axes.T
        points_3D_frame2 = (points_3D - centroid_3D) @ points_3D_axes.T
        unique_pairs = [(i, j) for i in range(len(points_3D)) for j in range(i)]
        n_consensus_x = 0
        n_consensus_y = 0
        for i, j in unique_pairs:
            if (
                points_3D_frame2[i, 0] - points_3D_frame2[j, 0] < 0
                and points_image_frame3[i, 0] - points_image_frame3[j, 0] < 0
            ) or (
                points_3D_frame2[i, 0] - points_3D_frame2[j, 0] > 0
                and points_image_frame3[i, 0] - points_image_frame3[j, 0] > 0
            ):
                n_consensus_x += 1
            if (
                points_3D_frame2[i, 1] - points_3D_frame2[j, 1] < 0
                and points_image_frame3[i, 1] - points_image_frame3[j, 1] < 0
            ) or (
                points_3D_frame2[i, 1] - points_3D_frame2[j, 1] > 0
                and points_image_frame3[i, 1] - points_image_frame3[j, 1] > 0
            ):
                n_consensus_y += 1

        flip_x = n_consensus_x < len(unique_pairs) / 2
        flip_y = n_consensus_y < len(unique_pairs) / 2
        if flip_x:
            points_image_frame3[:, 0] *= -1
            wrist_keypoint_frame3[0] *= -1
            hand_global_orient_frame2[
                :, 0
            ] *= -1  # Flip the x-axis of the hand orientation
            rospy.logwarn("Flipping x-axis of image points and wrist keypoint")
        if flip_y:
            points_image_frame3[:, 1] *= -1
            wrist_keypoint_frame3[1] *= -1
            hand_global_orient_frame2[
                :, 1
            ] *= -1  # Flip the y-axis of the hand orientation
            rospy.logwarn("Flipping y-axis of image points and wrist keypoint")

        if self.cfg.debug.log_verbose:
            rospy.loginfo(
                f" x consensus: {float(n_consensus_x) / len(unique_pairs)}, y consensus: {float(n_consensus_y) / len(unique_pairs)}"
            )

        if self.cfg.debug.log_visualization:
            # Visualize the principal axes of the 3D points and image points
            path = os.path.join(
                self.out_dir,
                f"(te)_points_3D_in_principal_coords{self.n_requests:04d}.png",
            )
            self.save_2d_visualization(
                points1=points_3D_frame2,
                points2=np.array([[0, 0]]),  # Origin for the principal axes
                output_path=path,
                label1="3D points in principal axes frame",
                label2="Origin of principal axes",
            )
            self.save_2d_visualization(
                points1=points_image_frame3,
                points2=np.array([[0, 0]]),  # Origin for the principal axes
                output_path=path.replace("points_3D", "image_points"),
                label1="Image points in principal axes frame",
                label2="Origin of principal axes",
            )

        # Scale the wrist keypoint and lift it to the 3D space
        scale_factor = np.sqrt(pca_points_3D.explained_variance_[0]) / np.sqrt(
            pca_image_points.explained_variance_[0]
        )
        wrist_keypoint_frame2 = np.zeros(3, dtype=np.float32)
        wrist_keypoint_frame2[:2] = wrist_keypoint_frame3 * scale_factor

        # Formulate the hand pose in the frame 2
        hand_pose_frame2 = np.eye(4)
        hand_pose_frame2[:3, :3] = hand_global_orient_frame2  # Set rotation part
        hand_pose_frame2[:3, 3] = wrist_keypoint_frame2

        if self.cfg.debug.log_visualization:
            points_image_frame4_3D = np.zeros(
                (points_image.shape[0], 3), dtype=np.float32
            )
            points_image_frame4_3D[:, :2] = points_image
            wrist_keypoint_frame4_3D = np.zeros(3, dtype=np.float32)
            wrist_keypoint_frame4_3D[:2] = wrist_keypoint
            hand_pose_matrix = np.eye(4, dtype=np.float32)
            hand_pose_matrix[:3, :3] = hand_global_orient
            hand_pose_matrix[:3, 3] = wrist_keypoint_frame4_3D
            path = os.path.join(
                self.out_dir, f"(te)_hand_pose_image_coords_{self.n_requests:04d}.png"
            )
            self.save_heuristic_transform_visualization(
                points_3D=points_image_frame4_3D,
                points_image_3D=points_image_frame4_3D,
                hand_pose=hand_pose_matrix,
                output_path=path,
            )

            # Scale the points and lift them to the 3D space
            points_image_frame2 = np.zeros(
                (points_image_frame3.shape[0], 3), dtype=np.float32
            )
            points_image_frame2[:, :2] = points_image_frame3 * scale_factor
            path = os.path.join(
                self.out_dir,
                f"(te)_hand_pose_in_principal_cords_{self.n_requests:04d}.png",
            )
            self.save_heuristic_transform_visualization(
                points_3D=points_3D_frame2,
                points_image_3D=points_image_frame2,
                hand_pose=hand_pose_frame2,
                output_path=path,
            )

        # Get the hand pose in frame 1
        tf_frame2_to_frame1 = np.eye(4)
        tf_frame2_to_frame1[:3, :3] = points_3D_axes.T
        tf_frame2_to_frame1[:3, 3] = centroid_3D

        # Finally, get the hand pose in frame 1
        hand_pose = tf_frame2_to_frame1 @ hand_pose_frame2

        if self.cfg.debug.log_optimization_results:
            path = os.path.join(
                self.out_dir,
                f"(te)_transformation_heuristic_{self.n_requests:04d}.yaml",
            )
            with open(path, "w") as f:
                yaml.dump(
                    {
                        "transformation_matrix": hand_pose.tolist(),
                        "centroid_image": centroid_image.tolist(),
                        "centroid_3D": centroid_3D.tolist(),
                        "scale_factor": scale_factor.tolist(),
                    },
                    f,
                )

        if self.cfg.debug.log_visualization:
            path = os.path.join(
                self.out_dir,
                f"(te)_hand_pose_in_principal_axes_{self.n_requests:04d}.png",
            )
            self.save_2d_visualization(
                points1=points_3D_frame2[:, :2],
                points2=points_image_frame2[:, :2],
                output_path=path,
                label1="3D points in principal axes frame",
                label2="Image points in principal axes frame",
            )

            # Get the points in frame 1
            points_image_frame1 = self._apply_transformation(
                points_image_frame2, tf_frame2_to_frame1
            )

            path = os.path.join(
                self.out_dir, f"(te)_heuristic_transformation_{self.n_requests:04d}.png"
            )
            self.save_heuristic_transform_visualization(
                points_3D=points_3D,
                points_image_3D=points_image_frame1,
                hand_pose=hand_pose,
                output_path=path,
            )

            path = os.path.join(
                self.out_dir, f"(te)_wrist_keypoint_{self.n_requests:04d}.png"
            )
            self.save_2d_visualization(
                points1=points_image,
                points2=wrist_keypoint.reshape(1, 2),
                output_path=path,
                label1="Image points",
                label2="Wrist keypoint",
            )

        return hand_pose, 0

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
    @staticmethod
    def _get_transform(
        translation: np.ndarray, rotation_quat: np.ndarray
    ) -> np.ndarray:
        """
        Constructs a 4x4 transformation matrix from translation and rotation quaternion.

        Args:
            translation (np.ndarray): 3-element array representing the translation vector.
            rotation_quat (np.ndarray): 4-element array representing the rotation quaternion.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        assert translation.shape == (3,)
        assert rotation_quat.shape == (4,)

        # Convert quaternion to rotation matrix
        R = open3d.geometry.get_rotation_matrix_from_quaternion(rotation_quat)

        # Create the transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = translation

        return transform

    @staticmethod
    def _get_transform_euler(
        translation: np.ndarray, rotation_euler: np.ndarray
    ) -> np.ndarray:
        """
        Constructs a 4x4 transformation matrix from translation and rotation quaternion.

        Args:
            translation (np.ndarray): 3-element array representing the translation vector.
            rotation_euler (np.ndarray): 3-element array representing the rotation.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        assert translation.shape == (3,)
        assert rotation_euler.shape == (3,)

        # Convert quaternion to rotation matrix
        R = open3d.geometry.get_rotation_matrix_from_xyz(rotation_euler)

        # Create the transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = translation

        return transform

    def _pixel_to_normalized(
        self, intrinsic_matrix: np.ndarray, image_points: np.ndarray
    ):
        """
        Normalizes 2D pixel coordinates (u, v) using the intrinsic matrix.
        The resulting coordinates are correspoding to a focal length of 1.

        Args:
            intrinsic_matrix (np.ndarray): 3x3 intrinsic camera matrix.
            image_points (np.ndarray): Nx2 array of pixel coordinates (height, width)

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
        points_3D_reference: np.ndarray = None,
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
        if points_3D_reference is not None:
            points_3D_reference = points_3D_reference.astype(np.float64)

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

        # Plot reference 3D points if provided
        if points_3D_reference is not None:
            ax.scatter(
                points_3D_reference[:, 0],
                points_3D_reference[:, 1],
                points_3D_reference[:, 2],
                c="green",
                s=50,
                label="Reference 3D Points",
                marker="^",
            )
            for i, pt in enumerate(points_3D_reference):
                ax.text(pt[0], pt[1], pt[2], f"R{i}", color="green", fontsize=10)

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

    def plot_2d_points(
        self, points1: np.ndarray, points2: np.ndarray, label1=None, label2=None
    ):
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(points1[:, 0], points1[:, 1], c="blue", label=label1)
        ax.scatter(points2[:, 0], points2[:, 1], c="red", label=label2)
        labels = [f"P{i}" for i in range(len(points1))]
        for i in range(len(points1)):
            ax.text(points1[i, 0], points1[i, 1], f"P{i}", color="blue", fontsize=10)
        for i in range(len(points2)):
            ax.text(points2[i, 0], points2[i, 1], f"P{i}", color="red", fontsize=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # plt.title('2D Points Plot')
        ax.grid(True)
        ax.axis("equal")
        ax.legend()

    def save_3d_reconstruction_visualization(
        self,
        intrinsic_matrix: np.ndarray,
        points_3D: np.ndarray,
        points_2D: np.ndarray,
        output_path: str,
        points_3D_reference: np.ndarray = None,
    ):
        """
        Saves the visualization of the reconstruction process by plotting the
        3D points, the reconstructed 3D points, and the 2D image points.

        Args:
            intrinsic_matrix (np.ndarray): 3x3 intrinsic camera matrix.
            points_3D (np.ndarray): Nx3 array of 3D points.
            points_2D (np.ndarray): Nx2 array of 2D image points.
            output_path (str): Path to save the visualization.
        """

        base, ext = os.path.splitext(output_path)
        _, ax = self._plot_camera_model(
            intrinsic_matrix, points_2D, points_3D, points_3D_reference
        )
        plt.savefig(output_path)
        ax.view_init(elev=0, azim=-90)
        plt.savefig(f"{base}_top_view{ext}")
        ax.view_init(elev=-90, azim=90, roll=180)
        plt.savefig(f"{base}_camera_view{ext}")
        plt.close()

    def save_2d_visualization(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
        output_path: str,
        label1=None,
        label2=None,
    ):
        ax = self.plot_2d_points(
            points1=points1, points2=points2, label1=label1, label2=label2
        )
        plt.savefig(output_path)
        plt.close()

    def save_heuristic_transform_visualization(
        self,
        points_3D: np.ndarray,
        points_image_3D: np.ndarray,
        hand_pose: np.ndarray,
        output_path: str,
    ):
        """
        Saves a visualization of the heuristic transformation estimation process.

        Args:
            points_3D (np.ndarray): Nx3 array of 3D points in their original frame.
            points_image_3D (np.ndarray): Nx3 array of the image points in the same frame as points_3D.
            hand_pose (np.ndarray): 4x4 transformation matrix representing the hand pose.
            principal_axes_3D (np.ndarray): Nx3 array of the principal axes of the 3D points
            output_path (str): Path to save the visualization.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot original 3D points
        ax.scatter(
            points_3D[:, 0],
            points_3D[:, 1],
            points_3D[:, 2],
            c="blue",
            label="3D Points",
            s=50,
        )
        for i, pt in enumerate(points_3D):
            ax.text(pt[0], pt[1], pt[2], f"P{i}", color="blue", fontsize=10)

        # Plot image points transformed to 3D
        ax.scatter(
            points_image_3D[:, 0],
            points_image_3D[:, 1],
            points_image_3D[:, 2],
            c="red",
            label="Image Points (3D)",
            s=50,
            marker="^",
        )
        for i, pt in enumerate(points_image_3D):
            ax.text(pt[0], pt[1], pt[2], f"I{i}", color="red", fontsize=10)

        # Plot principal axes
        pca = PCA()
        pca.fit(points_3D)
        mean_point = np.mean(points_3D, axis=0)
        bbox_size = np.linalg.norm(points_3D.max(axis=0) - points_3D.min(axis=0))
        axis_length = 0.2 * bbox_size  # scale for visibility

        for i, axis in enumerate(pca.components_[:2]):
            ax.quiver(
                mean_point[0],
                mean_point[1],
                mean_point[2],
                axis[0],
                axis[1],
                axis[2],
                color="orange",
                length=axis_length,
                normalize=True,
                label=f"Principal Axis {i+1}",
            )
            # Place label at the tip of the axis
            tip = mean_point + axis * axis_length
            ax.text(tip[0], tip[1], tip[2], f"A{i+1}", color="orange", fontsize=10)

        # Plot hand pose axes
        origin = hand_pose[:3, 3]
        axes = hand_pose[:3, :3]
        bbox_size = np.linalg.norm(points_3D.max(axis=0) - points_3D.min(axis=0))
        length = 0.05 * bbox_size  # scale for visibility

        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axes[0, 0],
            axes[1, 0],
            axes[2, 0],
            color="r",
            length=length,
            normalize=True,
            label="Hand X",
        )
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axes[0, 1],
            axes[1, 1],
            axes[2, 1],
            color="g",
            length=length,
            normalize=True,
            label="Hand Y",
        )
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axes[0, 2],
            axes[1, 2],
            axes[2, 2],
            color="b",
            length=length,
            normalize=True,
            label="Hand Z",
        )

        # Set equal scaling for all axes
        xyz = np.vstack((points_3D, points_image_3D, origin.reshape(1, 3)))
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        max_range = (xyz_max - xyz_min).max() / 2.0
        mid = (xyz_max + xyz_min) / 2.0
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.title("Heuristic Transform Visualization")
        plt.tight_layout()
        plt.savefig(output_path)
        # Save additional perspectives
        base, ext = os.path.splitext(output_path)
        ax.view_init(elev=0, azim=-90)
        plt.savefig(f"{base}_top_view{ext}")
        ax.view_init(elev=-90, azim=90, roll=180)
        plt.savefig(f"{base}_camera_view{ext}")
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
