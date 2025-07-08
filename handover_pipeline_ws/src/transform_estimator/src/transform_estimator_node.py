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
from scipy.optimize import minimize, NonlinearConstraint, least_squares
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import sys
from typing import Tuple
import rospy
import yaml

# TODO: Remove dependency on the pipeline package
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

        # transform_robot_cam_to_gen_cam, cost = self.estimate_transformation(
        #     intrinsic_matrix=K_grasp,
        #     points_image=corr_points_grasp,
        #     points_3D=corr_points_object_3D,
        #     n_points=None,
        # )

        costs = []
        transforms = []
        focal_lengths = [float(K_grasp[0, 0] + K_grasp[1, 1])/2.0]  # TODO: Use a range of focal lengths
        
        print(f"Initial focal lengths: {focal_lengths}")

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
        rospy.loginfo(
            f"Best focal length: {f:.4f} with cost: {costs[best_idx]:.4f}"
        )

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
        self._result.transform_robot_cam_to_gen_cam = np_to_transformmsg(transform_robot_cam_to_gen_cam)
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
        wrist_keypoint = multiarraymsg_to_np(goal.hand_keypoints).reshape(-1, 2)[0,[1,0]]  # Extract wrist (first keypoint)
        hand_orient = transformmsg_to_np(goal.transform_hand_pose_to_camera)[:3, :3] # Extract rotation part

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

        #####################################################################################
        # We fake data to test the heuristic method. TODO: Remove this

        # N_points = 15
        # corr_points_object_3D = corr_points_object_3D[:N_points]  # Use only the first N points
        # corr_points_grasp = corr_points_grasp[:N_points]  # Use only the first N points

        # points_3d = np.array([[1,0,10], [0, 5, 10], [-1, 0, 10], [0, -5, 10], [0, 17, 10]], dtype=np.float32)
        # corr_points_object_3D = points_3d[:-1]  # Use all but the last point
        # tvec = np.array([-3, 2, 0], dtype=np.float32)  # Fake translation vector
        # rvec = np.array([0, 0, 0], dtype=np.float32) * np.pi / 180 # Fake rotation vector (identity)
        # transform = self._get_transform_euler(translation=tvec, rotation_euler=rvec)
        # transformed_points = self._apply_transformation(
        #     points_3d, transform
        # )
        # K= np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]], dtype=np.float32)
        # corr_points_grasp = self.project_points_to_pixels(transformed_points[:-1], K)
        # wrist_keypoint = self.project_points_to_pixels(transformed_points[-1:], K)[0]  # Project a point at (0, 0, 10) to pixel coordinates
        # # hand_orient = np.eye(3, dtype=np.float32)  # Identity rotation for the hand orientation
        # rvec = np.array([0, 0, 0], dtype=np.float32)  # Identity rotation vector
        # hand_orient = Rotation.from_euler('xyz', rvec, degrees=True).as_matrix().astype(np.float32)

        # print(f"Fake 3D points in object frame:\n{corr_points_object_3D}")
        # print(f"Fake 2D points in grasp frame:\n{corr_points_grasp}")
        # print(f"Fake wrist keypoint in pixel coordinates: {wrist_keypoint}")
        # print(f"Fake hand orientation:\n{hand_orient}")
        #####################################################################################


        # Estimate the transformation using the heuristic method
        try:
            transform_hand_pose_to_robot_camera, cost = self.estimate_transformation_heuristic(
                points_image=corr_points_grasp,
                wrist_keypoint=wrist_keypoint,
                hand_global_orient=hand_orient,
                points_3D=corr_points_object_3D,
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
        self._result_heuristic.transform_hand_pose_to_robot_camera = np_to_transformmsg(transform_hand_pose_to_robot_camera)
        self._server_heuristic.set_succeeded(self._result_heuristic)

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
            float: Mean squared error between the transformed source points and target points.
        """

        n = len(source_points)
        source = open3d.geometry.PointCloud()
        target = open3d.geometry.PointCloud()
        source.points = open3d.utility.Vector3dVector(source_points)
        target.points = open3d.utility.Vector3dVector(target_points)

        # Create correspondences explicitly (point i in source corresponds to point i in target)
        correspondences = np.array([(i, i) for i in range(n)])

        # TODO: Implement RANSAC if necessary
        transformation = open3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(
            source, target, open3d.utility.Vector2iVector(correspondences)
        )

        # Mean squared distance between transformed source and target points
        source_homogeneous = np.hstack((source_points, np.ones((n, 1))))
        source_transformed = (transformation @ source_homogeneous.T).T[:, :3]

        mse = np.mean(np.linalg.norm(source_transformed - target_points, axis=1) ** 2)

        if self.cfg.debug.log_transform_mse:
            mse_path = os.path.join(
                self.out_dir, f"(te)_mse_transformation_{self.n_requests:04d}.txt"
            )
            with open(mse_path, "w") as f:
                f.write(f"Mean Squared Error: {mse:.4f}\n")
            rospy.loginfo(f"Mean Squared Error saved to {mse_path}")

        return transformation, mse

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
        print(f"RANSAC PnP inliers: {len(inliers)} from {len(points_image)} total points")
        return transformation_matrix, reprojection_error


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


    def ransac_pnp(
        self,
        intrinsic_matrix: np.ndarray,
        points_image: np.ndarray,
        points_3D: np.ndarray,
        n_points: int = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Estimates the transformation matrix using RANSAC and PnP.

        Args:
            intrinsic_matrix (np.ndarray): 3x3 intrinsic camera matrix.
            points_image (np.ndarray): Nx2 array of 2D image points in pixel coordinates.
                The first column is the width and the second column is the height.
            points_3D (np.ndarray): Nx3 array of 3D points in meters, the frame in which
                the coordinates are expressed is arbitrary.
            n_points (int, optional): Number of points to use for estimation. If None,
                all points will be used.

        Returns:
            Tuple[np.ndarray, float]: The estimated transformation matrix and the final
                value of the optimization function.
        """

        print(f"Points image shape: {points_image.shape}")
        print(f"Points 3D shape: {points_3D.shape}")

        # Get an initial guess using an iterative PnP method
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points_3D.astype(np.float32),
            imagePoints=points_image.astype(np.float32),
            cameraMatrix=intrinsic_matrix.astype(np.float32),
            distCoeffs=None,  # Assuming no lens distortion
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
            iterationsCount=10000,
        )

        print(f"RANSAC PnP success: {success}")
        print(f"RANSAC PnP inliers: {inliers}")
        print(f"RANSAC PnP rvec: {rvec}")
        print(f"RANSAC PnP tvec: {tvec}")

        # Refine by using the (nonlinear) VVS method
        rvec, tvec = cv2.solvePnPRefineVVS(
            objectPoints=points_3D.astype(np.float32),
            imagePoints=points_image.astype(np.float32),
            cameraMatrix=intrinsic_matrix.astype(np.float32),
            distCoeffs=None,  # Assuming no lens distortion
            rvec=rvec,
            tvec=tvec,
        )

        print(f"RANSAC PnP rvec: {rvec}")
        print(f"RANSAC PnP tvec: {tvec}")

        if not success:
            rospy.logerr("RANSAC PnP failed to estimate the transformation.")
            raise ValueError(
                "RANSAC PnP failed to estimate the transformation. "
                "Check the input points and camera parameters."
            )

        # Convert rotation vector and translation vector to a transformation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = tvec.flatten()

        rospy.loginfo(
            f"Solved PnP with {len(inliers)} inliers out of {len(points_3D)} points."
        )

        # Calculate the reprojection error
        projected_points, _ = cv2.projectPoints(
            objectPoints=points_3D[inliers].astype(np.float32),
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=intrinsic_matrix.astype(np.float32),
            distCoeffs=None,  # Assuming no lens distortion
        )
        reprojection_error = np.mean(
            np.linalg.norm(projected_points[:, 0, :] - points_image[inliers], axis=1)
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
                        "reprojection_error": reprojection_error.tolist(),
                        "inliers": inliers.flatten().tolist(),
                    },
                    f,
                )
            rospy.loginfo(f"RANSAC PnP results saved to {results_path}")

        if self.cfg.debug.log_verbose:
            rospy.loginfo(f"Estimated translation vector:\n{tvec.flatten()}")
            rospy.loginfo(f"Estimated rotation vector:\n{rvec.flatten()}")

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
        return transformation_matrix, reprojection_error

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

        points_image = points_image.astype(np.float32)[:,[1,0]] # flip to (width, height) format
        wrist_keypoint = wrist_keypoint.astype(np.float32)[[1,0]] # flip to (width, height) format
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

        # Align the orientation of the principal axes (they might point in opposite directions)
        points_image_frame3 = (points_image - centroid_image) @ image_point_axes.T
        points_3D_frame2 = (points_3D - centroid_3D) @ points_3D_axes.T
        unique_pairs = [(i, j) for i in range(len(points_3D)) for j in range(i)]
        n_consensus_x = 0
        n_consensus_y = 0
        for i, j in unique_pairs:
            if (points_3D_frame2[i, 0] - points_3D_frame2[j, 0] < 0 and \
               points_image_frame3[i, 0] - points_image_frame3[j, 0] < 0) or \
                (points_3D_frame2[i, 0] - points_3D_frame2[j, 0] > 0 and \
                points_image_frame3[i, 0] - points_image_frame3[j, 0] > 0):
                n_consensus_x += 1
            if (points_3D_frame2[i, 1] - points_3D_frame2[j, 1] < 0 and \
               points_image_frame3[i, 1] - points_image_frame3[j, 1] < 0) or \
                (points_3D_frame2[i, 1] - points_3D_frame2[j, 1] > 0 and \
                points_image_frame3[i, 1] - points_image_frame3[j, 1] > 0):
                n_consensus_y += 1
        
        flip_x = n_consensus_x < len(unique_pairs) / 2
        flip_y = n_consensus_y < len(unique_pairs) / 2
        if flip_x:
            points_image_frame3[:, 0] *= -1
            wrist_keypoint_frame3[0] *= -1
            rospy.logwarn("Flipping x-axis of image points and wrist keypoint")
        if flip_y:
            points_image_frame3[:, 1] *= -1
            wrist_keypoint_frame3[1] *= -1
            rospy.logwarn("Flipping y-axis of image points and wrist keypoint")

        if self.cfg.debug.log_verbose:
            rospy.loginfo(f" x consensus: {float(n_consensus_x) / len(unique_pairs)}, y consensus: {float(n_consensus_y) / len(unique_pairs)}")

        if self.cfg.debug.log_visualization:
            # Visualize the principal axes of the 3D points and image points
            path = os.path.join(
                self.out_dir, f"(te)_points_3D_in_principal_coords{self.n_requests:04d}.png"
            )
            self.save_2d_visualization(
                points1=points_3D_frame2,
                points2=np.array([[0,0]]),  # Origin for the principal axes
                output_path=path,
                label1="3D points in principal axes frame",
                label2="Origin of principal axes",
            )
            self.save_2d_visualization(
                points1=points_image_frame3,
                points2=np.array([[0,0]]),  # Origin for the principal axes
                output_path=path.replace("points_3D", "image_points"),
                label1="Image points in principal axes frame",
                label2="Origin of principal axes",
            )

        # Scale the wrist keypoint and lift it to the 3D space
        scale_factor = np.sqrt(pca_points_3D.explained_variance_[0]) / np.sqrt(pca_image_points.explained_variance_[0])
        wrist_keypoint_frame2 = np.zeros(3, dtype=np.float32)
        wrist_keypoint_frame2[:2] = wrist_keypoint_frame3 * scale_factor

        # Formulate the hand pose in the frame 2
        hand_pose_frame2 = np.eye(4)
        hand_pose_frame2[:3, :3] = hand_global_orient  # Set rotation part
        hand_pose_frame2[:3, 3] = wrist_keypoint_frame2

        # Get the hand pose in frame 1
        tf_frame2_to_frame1 = np.eye(4)
        tf_frame2_to_frame1[:3, :3] = points_3D_axes.T
        tf_frame2_to_frame1[:3, 3] = centroid_3D

        # Finally, get the hand pose in frame 1
        hand_pose = tf_frame2_to_frame1 @ hand_pose_frame2

        if self.cfg.debug.log_optimization_results:
            path = os.path.join(
                self.out_dir, f"(te)_transformation_heuristic_{self.n_requests:04d}.yaml"
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
            # Scale the points and lift them to the 3D space
            points_image_frame2 = np.zeros((points_image_frame3.shape[0], 3), dtype=np.float32)
            points_image_frame2[:, :2] = points_image_frame3 * scale_factor

            path = os.path.join(
                self.out_dir, f"(te)_hand_pose_in_principal_axes_{self.n_requests:04d}.png"
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

        # TODO: Return a sensible cost value
        cost = 0

        return hand_pose, cost

    def estimate_all(
        self,
        intrinsic_matrix: np.ndarray,
        points_image: np.ndarray,
        points_3D: np.ndarray,
        n_points: int = None,
        f_init: float = None,
    ) -> Tuple[float, np.ndarray, float]:
        """
        Estimates the focal length of the camera by minimizing the reprojection error.

        Args:
            intrinsic_matrix (np.ndarray): 3x3 intrinsic camera matrix. The focal length
                entry is taken as the initial guess if f_init is None.
            points_image (np.ndarray): Nx2 array of 2D image points in pixel coordinates.
            points_3D (np.ndarray): Nx3 array of 3D points in meters, the frame in which
                the coordinates are expressed is arbitrary.
            f_init (float, optional): Initial guess for the focal length. If None,
                it will be set to the focal length from the intrinsic matrix.

        Returns:
            Tuple[float, np.ndarray, float]: The optimal focal length, the optimal
                transform from the (real) robot camera frame to the (virtual)
                gen camera frame and the final value of the optimization function.
        """

        points_image = points_image.astype(np.float32)
        points_3D = points_3D.astype(np.float32)

        # Constrain the number of points
        if n_points:
            if n_points > len(points_3D):
                rospy.logerr(f"Too large number of points given: {n_points}")
                return None
            points_image = points_image[:n_points, :]
            points_3D = points_3D[:n_points, :]
        else:
            n_points = len(points_image)

        if len(points_3D) != len(points_image):
            rospy.logerr(
                f"Number of 3D points ({len(points_3D)}) and 2D points ({len(points_image)}) is not equal"
            )
        rospy.loginfo(f"Using {n_points} points for estimation.")

        # Express the points w.r.t. to an anchor point (e.g., the first point)
        # anchor_point = np.array([0,0,0])
        anchor_point = points_3D[0]
        points_3D_optim = points_3D - anchor_point

        # Initial guess
        f = f_init if f_init else intrinsic_matrix[0, 0]
        translation = anchor_point
        rotation_euler = np.array([0, 0, 0])
        initial_params = np.hstack([f, translation, rotation_euler])

        n_it = 0  # Keep track of number of function call

        def _reprojection_cost(intrinsic_matrix, points_3d, log):
            # Cost for reprojection error -> Project 3d points back to a fictional
            # image plane at z = 1 and compare with the original normalized 2D points
            normalized_points = self._pixel_to_normalized(
                intrinsic_matrix, points_image
            )

            reprojection_cost = 0
            projected_points = np.ndarray([n_points, 2])
            for i in range(n_points):
                if points_3d[i, 2] <= 0:  # Ensure points are in front of camera
                    rospy.logwarn(f"[optim]: point {i} had negative z coordinate.")
                    return -1e10 * points_3d[i, 2]
                projected = points_3d[i, :2] / points_3d[i, 2]
                reprojection_cost += np.sum((projected - normalized_points[i]) ** 2)
                projected_points[i] = projected

            nonlocal n_it
            if n_it % 500 == 0 or log:
                path = os.path.join(
                    self.out_dir,
                    f"(te)_point_projection_k{n_it:04d}.png",
                )
                self.save_2d_visualization(
                    normalized_points,
                    projected_points,
                    path,
                    "Keypoints of grasp image",
                    "Projected 3D points",
                )
            n_it += 1

            return reprojection_cost

        def _objective_function(params, log=False):
            f = params[0]
            translation = params[1:4]
            rotation_euler = params[4:7]

            # Transform points from the 3D points in the (real) object camera frame
            # to the (virtual) gen camera frame
            tf = self._get_transform_euler(
                translation=translation, rotation_euler=rotation_euler
            )
            if log:
                rospy.loginfo(f"(obj fun) transform:\n{tf}")
            points_3D_transformed = self._apply_transformation(points_3D_optim, tf)

            # Update the intrinsic matrix with the current focal length
            intrinsic_matrix[0, 0] = f
            intrinsic_matrix[1, 1] = f
            # print(f"Using focal length: {f:.4f}")

            # Get cost
            return self.cfg.lambda_proj * _reprojection_cost(
                intrinsic_matrix, points_3D_transformed, log
            )

        # Run optimization
        result = minimize(_objective_function, initial_params, method="Newton-CG")

        # Extract the results
        cost = float(result.fun)
        focal_length = float(result.x[0])
        intrinsic_matrix[0, 0] = focal_length
        intrinsic_matrix[1, 1] = focal_length

        # Calculate the transform
        translation = result.x[1:4]
        rotation_euler = result.x[4:7]
        transform_anchor_to_gen_cam = self._get_transform_euler(
            translation=translation, rotation_euler=rotation_euler
        )
        transform_robot_cam_to_anchor = np.eye(4)
        transform_robot_cam_to_anchor[:3, 3] = -anchor_point
        transform_robot_cam_to_gen_cam = (
            transform_anchor_to_gen_cam @ transform_robot_cam_to_anchor
        )

        points_3D_gen_frame = self._apply_transformation(
            points_3D, transform_robot_cam_to_gen_cam
        )
        translation_final = transform_robot_cam_to_gen_cam[:3, 3]
        rotation_final = Rotation.from_matrix(
            transform_robot_cam_to_gen_cam[:3, :3]
        ).as_euler("xyz")

        # Log the results
        if self.cfg.debug.log_optimization_results:
            path = os.path.join(
                self.out_dir,
                f"(te)_transform_estimation_results_{self.n_requests:04d}.yaml",
            )
            with open(path, "w") as f:
                yaml.dump(
                    {
                        "lambda_proj": float(self.cfg.lambda_proj),
                        "reprojection_cost": float(
                            _reprojection_cost(
                                intrinsic_matrix, points_3D_gen_frame, False
                            )
                        ),
                        "objective_function_value": cost,
                        "optimal focal_length": focal_length,
                        "optimal translation (optimization problem)": translation.tolist(),
                        "optimal rotation (optimization problem)": rotation_euler.tolist(),
                        "optimal translation (robot cam frame to gen cam frame)": translation_final.tolist(),
                        "optimal rotation (robot cam frame to gen cam frame)": rotation_final.tolist(),
                        # "optimal transformation from real camera to gen camera": transform_robot_cam_to_gen_cam.tolist(),
                        # "intrinsic_matrix": intrinsic_matrix.tolist(),
                        "message": str(result.message),
                        "n_function_evaluations": int(result.nfev),
                        "n_iterations": int(result.nit),
                        "status": int(result.status),
                        "success": bool(result.success),
                    },
                    f,
                )
        if self.cfg.debug.log_visualization:
            normalized_points = self._pixel_to_normalized(
                intrinsic_matrix, points_image
            )
            # points_3D = points_3D + anchor_point
            # points_3D_transformed = self._apply_transformation(points_3D_optim, transform_robot_cam_to_gen_cam)
            points_3D_transformed = self._apply_transformation(
                points_3D, transform_robot_cam_to_gen_cam
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

        return focal_length, transform_robot_cam_to_gen_cam, cost

    def estimate_transformation(
        self,
        intrinsic_matrix: np.ndarray,
        points_image: np.ndarray,
        points_3D: np.ndarray,
        n_points: int = None,
    ) -> Tuple[float, np.ndarray, float]:
        """
        TODO

        Args:
            intrinsic_matrix (np.ndarray): 3x3 intrinsic camera matrix. The focal length
                entry is taken as the initial guess if f_init is None.
            points_image (np.ndarray): Nx2 array of 2D image points in pixel coordinates.
            points_3D (np.ndarray): Nx3 array of 3D points in meters, the frame in which
                the coordinates are expressed is arbitrary.

        Returns:
            Tuple[np.ndarray, float]: The optimal transform from the (real) robot camera
            frame to the (virtual) gen camera frame and the final value of the
            optimization function.
        """

        points_image = points_image.astype(np.float32)
        points_3D = points_3D.astype(np.float32)

        # Constrain the number of points
        if n_points:
            if n_points > len(points_3D):
                rospy.logerr(f"Too large number of points given: {n_points}")
                return None
            points_image = points_image[:n_points, :]
            points_3D = points_3D[:n_points, :]
        else:
            n_points = len(points_image)

        if len(points_3D) != len(points_image):
            rospy.logerr(
                f"Number of 3D points ({len(points_3D)}) and 2D points ({len(points_image)}) is not equal"
            )
        rospy.loginfo(f"Using {n_points} points for estimation.")

        # Express the points w.r.t. to an anchor point (e.g., the first point)
        # anchor_point = np.array([0,0,0])
        anchor_point = points_3D[0]
        points_3D_optim = points_3D - anchor_point

        # Initial guess
        translation = anchor_point
        rotation_euler = np.array([0, 0, 0])
        initial_params = np.hstack([translation, rotation_euler])

        # project the keypoints on a plane at z = 1
        normalized_points = self._pixel_to_normalized(intrinsic_matrix, points_image)
        n_it = 0  # Keep track of number of function call

        def _reprojection_cost(points_3d, log):
            # Cost for reprojection error -> Project 3d points back to a fictional
            # image plane at z = 1 and compare with the original normalized 2D points

            reprojection_cost = 0
            projected_points = np.ndarray([n_points, 2])
            for i in range(n_points):
                if points_3d[i, 2] <= 0:  # Ensure points are in front of camera
                    rospy.logwarn(f"[optim]: point {i} had negative z coordinate.")
                    return -1e10 * points_3d[i, 2]
                projected = points_3d[i, :2] / points_3d[i, 2]
                reprojection_cost += np.sum((projected - normalized_points[i]) ** 2)
                projected_points[i] = projected

            nonlocal n_it
            if n_it % 500 == 0 or log:
                path = os.path.join(
                    self.out_dir,
                    f"(te)_point_projection_k{n_it:04d}.png",
                )
                self.save_2d_visualization(
                    normalized_points,
                    projected_points,
                    path,
                    "Keypoints of grasp image",
                    "Projected 3D points",
                )
            n_it += 1

            return reprojection_cost

        def _objective_function(params, log=False):
            translation = params[:3]
            rotation_euler = params[3:6]

            # Transform points from the 3D points in the (real) object camera frame
            # to the (virtual) gen camera frame
            tf = self._get_transform_euler(
                translation=translation, rotation_euler=rotation_euler
            )
            if log:
                rospy.loginfo(f"(obj fun) transform:\n{tf}")
            points_3D_transformed = self._apply_transformation(points_3D_optim, tf)

            # Get cost
            return self.cfg.lambda_proj * _reprojection_cost(points_3D_transformed, log)

        # Run optimization
        # result = minimize(_objective_function, initial_params, method="BFGS")
        result = minimize(_objective_function, initial_params, method="Powell")

        # Extract the results
        cost = float(result.fun)

        # Calculate the transform
        translation = result.x[:3]
        rotation_euler = result.x[3:6]
        transform_anchor_to_gen_cam = self._get_transform_euler(
            translation=translation, rotation_euler=rotation_euler
        )
        transform_robot_cam_to_anchor = np.eye(4)
        transform_robot_cam_to_anchor[:3, 3] = -anchor_point
        transform_robot_cam_to_gen_cam = (
            transform_anchor_to_gen_cam @ transform_robot_cam_to_anchor
        )

        points_3D_gen_frame = self._apply_transformation(
            points_3D, transform_robot_cam_to_gen_cam
        )
        translation_final = transform_robot_cam_to_gen_cam[:3, 3]
        rotation_final = Rotation.from_matrix(
            transform_robot_cam_to_gen_cam[:3, :3]
        ).as_euler("xyz")

        # Log the results
        if self.cfg.debug.log_optimization_results:
            path = os.path.join(
                self.out_dir,
                f"(te)_transform_estimation_results_{self.n_requests:04d}.yaml",
            )
            with open(path, "w") as f:
                yaml.dump(
                    {
                        "lambda_proj": float(self.cfg.lambda_proj),
                        "reprojection_cost": float(
                            _reprojection_cost(points_3D_gen_frame, False)
                        ),
                        "objective_function_value": cost,
                        "optimal translation (optimization problem)": translation.tolist(),
                        "optimal rotation (optimization problem)": rotation_euler.tolist(),
                        "optimal translation (robot cam frame to gen cam frame)": translation_final.tolist(),
                        "optimal rotation (robot cam frame to gen cam frame)": rotation_final.tolist(),
                        "message": str(result.message),
                        "n_function_evaluations": int(result.nfev),
                        "n_iterations": int(result.nit),
                        "status": int(result.status),
                        "success": bool(result.success),
                    },
                    f,
                )
        if self.cfg.debug.log_visualization:
            normalized_points = self._pixel_to_normalized(
                intrinsic_matrix, points_image
            )
            # points_3D = points_3D + anchor_point
            # points_3D_transformed = self._apply_transformation(points_3D_optim, transform_robot_cam_to_gen_cam)
            points_3D_transformed = self._apply_transformation(
                points_3D, transform_robot_cam_to_gen_cam
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

        return transform_robot_cam_to_gen_cam, cost

    def estimate_focal_length(
        self,
        intrinsic_matrix: np.ndarray,
        points_image: np.ndarray,
        points_3D: np.ndarray,
        n_points: int = None,
        f_init: float = None,
    ) -> Tuple[float, np.ndarray, float]:
        """
        Estimates the focal length of the camera by minimizing the reprojection error.

        Args:
            intrinsic_matrix (np.ndarray): 3x3 intrinsic camera matrix. The focal length
                entry is taken as the initial guess if f_init is None.
            points_image (np.ndarray): Nx2 array of 2D image points in pixel coordinates.
            points_3D (np.ndarray): Nx3 array of 3D points in meters, the frame in which
                the coordinates are expressed is arbitrary.
            f_init (float, optional): Initial guess for the focal length. If None,
                it will be set to the focal length from the intrinsic matrix.

        Returns:
            Tuple[float, np.ndarray, float]: The optimal focal length, the optimal
                transform from the (real) robot camera frame to the (virtual)
                gen camera frame and the final value of the optimization function.
        """

        points_image = points_image.astype(np.float32)
        points_3D = points_3D.astype(np.float32)

        # Constrain the number of points
        if n_points:
            if n_points > len(points_3D):
                rospy.logerr(f"Too large number of points given: {n_points}")
                return None
            points_image = points_image[:n_points, :]
            points_3D = points_3D[:n_points, :]
        else:
            n_points = len(points_image)

        if len(points_3D) != len(points_image):
            rospy.logerr(
                f"Number of 3D points ({len(points_3D)}) and 2D points ({len(points_image)}) is not equal"
            )
        rospy.loginfo(f"Using {n_points} points for estimation.")

        # Initial guess
        f = f_init if f_init else intrinsic_matrix[0, 0]
        initial_params = np.hstack([f])

        n_it = 0  # Keep track of number of function call

        def _reprojection_cost(intrinsic_matrix, points_3d, log):
            # Cost for reprojection error -> Project 3d points back to a fictional
            # image plane at z = 1 and compare with the original normalized 2D points
            normalized_points = self._pixel_to_normalized(
                intrinsic_matrix, points_image
            )

            reprojection_cost = 0
            projected_points = np.ndarray([n_points, 2])
            for i in range(n_points):
                if points_3d[i, 2] <= 0:  # Ensure points are in front of camera
                    rospy.logwarn(f"[optim]: point {i} had negative z coordinate.")
                    return -1e10 * points_3d[i, 2]
                projected = points_3d[i, :2] / points_3d[i, 2]
                reprojection_cost += np.sum((projected - normalized_points[i]) ** 2)
                projected_points[i] = projected

            nonlocal n_it
            if n_it % 500 == 0 or log:
                path = os.path.join(
                    self.out_dir,
                    f"(te)_point_projection_(f)_k{n_it:04d}.png",
                )
                self.save_2d_visualization(
                    normalized_points,
                    projected_points,
                    path,
                    "Keypoints of grasp image",
                    "Projected 3D points",
                )
            n_it += 1

            return reprojection_cost

        def _objective_function(params, log=False):
            f = params[0]

            # Update the intrinsic matrix with the current focal length
            intrinsic_matrix[0, 0] = f
            intrinsic_matrix[1, 1] = f

            # Get cost
            return self.cfg.lambda_proj * _reprojection_cost(
                intrinsic_matrix, points_3D, log
            )

        # Run optimization
        result = minimize(_objective_function, initial_params, method="Powell")

        # Extract the results
        cost = float(result.fun)
        focal_length = float(result.x[0])
        intrinsic_matrix[0, 0] = focal_length
        intrinsic_matrix[1, 1] = focal_length

        # Log the results
        if self.cfg.debug.log_optimization_results:
            path = os.path.join(
                self.out_dir,
                f"(te)_focal_length_estimation_result_{self.n_requests:04d}.yaml",
            )
            with open(path, "w") as f:
                yaml.dump(
                    {
                        "lambda_proj": float(self.cfg.lambda_proj),
                        "reprojection_cost": float(
                            _reprojection_cost(intrinsic_matrix, points_3D, False)
                        ),
                        "objective_function_value": cost,
                        "optimal focal_length": focal_length,
                        "message": str(result.message),
                        "n_function_evaluations": int(result.nfev),
                        "n_iterations": int(result.nit),
                        "status": int(result.status),
                        "success": bool(result.success),
                    },
                    f,
                )
        if self.cfg.debug.log_visualization:
            normalized_points = self._pixel_to_normalized(
                intrinsic_matrix, points_image
            )
            projected_points = points_3D[:, :2] / points_3D[:, 2, np.newaxis]

            path = os.path.join(
                self.out_dir,
                f"(te)_point_projection_final_(f)_{self.n_requests:04d}.png",
            )
            self.save_2d_visualization(
                normalized_points,
                projected_points,
                path,
                "Keypoints of grasp image",
                "Projected 3D points",
            )

        return focal_length, cost

    def reconstruct_3d_points(
        self,
        intrinsic_matrix: np.ndarray,
        image_points: np.ndarray,
        relative_distances: dict,
        initial_points_3D: np.ndarray = None,
        optimize_focal_length: bool = True,
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
            optimize_focal_length (bool): Whether to treat the focal length
                as an optimization parameter. If False, the focal length is
                fixed to the value given in the intrinsic matrix.
        Returns:
            np.ndarray: Nx3 array of reconstructed 3D points.
            float: Final cost of the optimization.
        """
        # image_points = image_points[:3,:]
        # initial_points_3D = initial_points_3D[:3,:]
        n_points = len(image_points)

        # Initial guess
        initial_focal_length = (
            intrinsic_matrix[0, 0] if optimize_focal_length else np.nan
        )
        if initial_points_3D is None:
            normalized_points = self._pixel_to_normalized(
                intrinsic_matrix, image_points
            )
            initial_depths = np.linspace(1, 10, n_points)
            initial_points_3D = np.empty((n_points, 3))
            for i in range(n_points):
                initial_points_3D[i] = (
                    np.array([normalized_points[i, 0], normalized_points[i, 1], 1])
                    * initial_depths[i]
                )

        # Flatten to 1D array for optimization: (f, x1, y1, z1, x2, y2, z2, ...)
        initial_params = np.hstack((initial_focal_length, initial_points_3D.flatten()))

        def _reprojection_cost(normalized_points, points_3d):
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
                        distance_cost += (1.0 - float(actual_dist) / target_dist) ** 2

            return distance_cost

        def distance_constraint(params, i, j):
            x = params[1:]
            key = (i, j) if (i, j) in relative_distances else (j, i)
            return (
                np.linalg.norm(x[i * 3 : i * 3 + 3] - x[j * 3 : j * 3 + 3])
                - relative_distances[key]
            )

        constraints = []
        for i in range(n_points):
            j = 0  # constrain on a single point
            if (i, j) in relative_distances or (j, i) in relative_distances:
                constr = NonlinearConstraint(
                    fun=lambda params, i=i, j=j: distance_constraint(params, i, j),
                    lb=0.0,
                    ub=0.0,
                )
                constraints.append(constr)

        def objective_function(params):
            # Extract and reshape parameters
            f = params[0]
            points_3d = params[1:].reshape(-1, 3)

            if points_3d.shape[0] != n_points:
                raise ValueError(
                    f"Expected {n_points} points, but got {points_3d.shape[0]}"
                )

            # Cost for reprojection error
            if optimize_focal_length:
                # Recalculate the normalized image points based on the focal length
                intrinsic_matrix[0, 0] = f
                intrinsic_matrix[1, 1] = f
                print(f"Using focal length: {f:.4f}")
            normalized_points = self._pixel_to_normalized(
                intrinsic_matrix, image_points
            )
            reprojection_cost = _reprojection_cost(normalized_points, points_3d)

            # Cost for distance constraints in 3D
            # distance_cost = _distance_cost(points_3d)

            return (
                self.cfg.lambda_proj
                * reprojection_cost
                # + self.cfg.lambda_dist * distance_cost
            )

        # Run optimization
        result = minimize(
            objective_function, initial_params, constraints=constraints, method="SLSQP"
        )
        focal_length = result.x[0]
        reconstructed_points = result.x[1:].reshape(-1, 3)
        normalized_points = self._pixel_to_normalized(intrinsic_matrix, image_points)
        if self.cfg.debug.log_optimization_results:
            path = os.path.join(
                self.out_dir,
                f"(te)_transform_estimation_results_{self.n_requests:04d}.yaml",
            )
            with open(path, "w") as f:
                yaml.dump(
                    {
                        "lambda_proj": float(self.cfg.lambda_proj),
                        "lambda_dist": float(self.cfg.lambda_dist),
                        "reprojection_cost": float(
                            _reprojection_cost(normalized_points, reconstructed_points)
                        ),
                        "distance_cost": float(_distance_cost(reconstructed_points)),
                        "objective_function_value": float(result.fun),
                        "optimal focal_length": float(result.x[0]),
                        "message": str(result.message),
                        "n_function_evaluations": int(result.nfev),
                        "n_iterations": int(result.nit),
                        "status": int(result.status),
                        "success": bool(result.success),
                    },
                    f,
                )
        if optimize_focal_length:
            return focal_length, reconstructed_points, result.fun
        else:
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

    @staticmethod
    def project_points_to_pixels(points_3D, intrinsic_matrix):
        """
        Projects 3D points to 2D pixel coordinates using the intrinsic matrix.

        Parameters:
        - points_3D: (N, 3) numpy array of 3D points in camera coordinates
        - intrinsic_matrix: (3, 3) intrinsic matrix

        Returns:
        - pixels: (N, 2) pixel coordinates (height, width)
        """
        # Split XYZ
        X = points_3D[:, 0]
        Y = points_3D[:, 1]
        Z = points_3D[:, 2]

        # Avoid division by zero
        Z = np.where(Z == 0, 1e-8, Z)

        # Normalize to get [x, y] in camera image plane
        x_norm = X / Z
        y_norm = Y / Z

        # Apply intrinsics
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

        u = fx * x_norm + cx
        v = fy * y_norm + cy

        return np.stack((v, u), axis=1)

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
        ax.set_xlabel("Y")
        # plt.title('2D Points Plot')
        ax.grid(True)
        ax.axis("equal")
        ax.legend()

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
        points_3D_reference: np.ndarray = None,
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
        _, ax = self._plot_camera_model(
            intrinsic_matrix, points_2D, points_3D, points_3D_reference
        )
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
        ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], c="blue", label="3D Points", s=50)
        for i, pt in enumerate(points_3D):
            ax.text(pt[0], pt[1], pt[2], f"P{i}", color="blue", fontsize=10)

        # Plot image points transformed to 3D
        ax.scatter(points_image_3D[:, 0], points_image_3D[:, 1], points_image_3D[:, 2], c="red", label="Image Points (3D)", s=50, marker="^")
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
            mean_point[0], mean_point[1], mean_point[2],
            axis[0], axis[1], axis[2],
            color="orange", length=axis_length, normalize=True, label=f"Principal Axis {i+1}"
            )
            # Place label at the tip of the axis
            tip = mean_point + axis * axis_length
            ax.text(tip[0], tip[1], tip[2], f"A{i+1}", color="orange", fontsize=10)

        # Plot principal axes of the image points
        # pca_image = PCA()
        # pca_image.fit(points_image_3D)
        # mean_image_point = np.mean(points_image_3D, axis=0)
        # for i, axis in enumerate(pca_image.components_[:2]):
        #     ax.quiver(
        #         mean_image_point[0], mean_image_point[1], mean_image_point[2],
        #         axis[0], axis[1], axis[2],
        #         color="purple", length=axis_length, normalize=True, label=f"Image Axis {i+1}"
        #     )
        #     # Place label at the tip of the axis
        #     tip = mean_image_point + axis * axis_length
        #     ax.text(tip[0], tip[1], tip[2], f"IA{i+1}", color="purple", fontsize=10)

        # Plot hand pose axes
        origin = hand_pose[:3, 3]
        axes = hand_pose[:3, :3]
        bbox_size = np.linalg.norm(points_3D.max(axis=0) - points_3D.min(axis=0))
        length = 0.05 * bbox_size  # scale for visibility

        ax.quiver(
            origin[0], origin[1], origin[2],
            axes[0, 0], axes[1, 0], axes[2, 0],
            color="r", length=length, normalize=True, label="Hand X"
        )
        ax.quiver(
            origin[0], origin[1], origin[2],
            axes[0, 1], axes[1, 1], axes[2, 1],
            color="g", length=length, normalize=True, label="Hand Y"
        )
        ax.quiver(
            origin[0], origin[1], origin[2],
            axes[0, 2], axes[1, 2], axes[2, 2],
            color="b", length=length, normalize=True, label="Hand Z"
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
