#!/usr/bin/env python3
import cv2
from dotenv import load_dotenv
from detectron2.config import LazyConfig
from detectron2 import model_zoo
import json
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
from pathlib import Path
import sys
import torch
import rospy
import yaml

import vitpose_model
from hamer.models import load_hamer
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
import hamer

# TODO: Remove dependency on the pipeline package
sys.path.append(str(Path(__file__).parent.parent.parent / "pipeline/src/"))
from msg_utils import np_to_transformmsg, imgmsg_to_cv2, np_to_multiarraymsg, cv2_to_imgmsg

from hand_reconstructor.srv import (
    ReconstructHand,
    ReconstructHandRequest,
    ReconstructHandResponse,
    ReconstructHandPose,
    ReconstructHandPoseRequest,
    ReconstructHandPoseResponse,
    EstimateCamera,
    EstimateCameraRequest,
    EstimateCameraResponse,
    RenderHand,
    RenderHandRequest,
    RenderHandResponse,
)
from std_msgs.msg import String, Int32MultiArray
from sensor_msgs.msg import CameraInfo


class HandReconstructor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rospy.init_node(cfg.ros.node_name, anonymous=True)

        # Unfortunately, some paths are hardcoded in the hamer package.
        # So we have to overwrite some of them here.
        DATA_ROOT_DIR = Path(cfg.hamer_data_dir)
        PACKAGE_ROOT_DIR = Path(__file__).parent.parent
        hamer.configs.CACHE_DIR_HAMER = DATA_ROOT_DIR / "_DATA/"
        vitpose_model.ViTPoseModel.MODEL_DICT = {
            "ViTPose+-G (multi-task train, COCO)": {
                "config": str(
                    PACKAGE_ROOT_DIR
                    / "third-party/hamer/third-party/ViTPose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py"
                ),
                "model": str(
                    DATA_ROOT_DIR / "_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth"
                ),
            }
        }

        # Initialize the hand pose estimator
        path = DATA_ROOT_DIR / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
        self.hamer, self.hamer_cfg = load_hamer(path)
        self.hamer = self.hamer.to(self.device)
        self.hamer.eval()
        self.hamer_cfg.EXTRA.FOCAL_LENGTH = cfg.focal_length

        # Initialize the body detector (Needed to crop the hand region)
        if cfg.body_detector == "vitdet":

            cfg_path = (
                Path(hamer.__file__).parent
                / "configs"
                / "cascade_mask_rcnn_vitdet_h_75ep.py"
            )
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = (
                    0.25
                )
            self.body_detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif cfg.body_detector == "regnety":

            detectron2_cfg = model_zoo.get_config(
                "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
            )
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.body_detector = DefaultPredictor_Lazy(detectron2_cfg)
        else:
            raise ValueError(
                "Invalid body detector. Choose either 'vitdet' or 'regnety'."
            )

        # Load keypoint detector (needed to get the hand type)
        self.keypoint_detector = vitpose_model.ViTPoseModel(self.device)

        # Setup the renderer to visualize the hand pose
        if self.cfg.debug.log_visualization:
            self.hand_renderer = Renderer(self.hamer_cfg, faces=self.hamer.mano.faces)

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
            if not os.path.exists(self.out_dir):
                rospy.logwarn(
                    f"Output directory {self.out_dir} does not exist. Creating it."
                )
                os.makedirs(self.out_dir, exist_ok=True)
        else:
            rospy.logerr(
                "Invalid out_dir_mode. Supported modes are 'fixed' and 'topic'."
            )

        # Log the config
        if self.cfg.debug.log_config:
            config_path = os.path.join(self.out_dir, "(hr)_config.yaml")
            with open(config_path, "w") as f:
                OmegaConf.save(config=self.cfg, f=f.name)

        # Once set up, we can start the services
        self._reconstr_hand_srv = rospy.Service(
            cfg.ros.provided_services.reconstruct_hand,
            ReconstructHand,
            self._reconstr_hand_callback,
        )
        self._reconstr_hand_pose_srv = rospy.Service(
            cfg.ros.provided_services.reconstruct_hand_pose,
            ReconstructHandPose,
            self._reconstr_hand_pose_callback,
        )
        self._estim_cam_srv = rospy.Service(
            cfg.ros.provided_services.estimate_camera,
            EstimateCamera,
            self._estim_cam_callback,
        )
        self._render_hand_srv = rospy.Service(
            cfg.ros.provided_services.render_hand,
            RenderHand,
            self._render_hand_callback,
        )
        self.n_requests = 0  # Keep track of the number of requests

        rospy.loginfo(f"{cfg.ros.node_name} service initialized.")


    def _reconstr_hand_callback(
        self, request: ReconstructHandRequest
    ) -> ReconstructHandResponse:
        """Service callback to reconstruct the hand pose from an image.

        Args:
            request: ReconstructHandRequest containing the image to process.

        Returns:
            ReconstructHandResponse containing the complete output of the hand
            pose estimator.
        """

        # Validate request
        if not request.image:
            rospy.logerr("Invalid request: No image given.")
            return ReconstructHandResponse(success=False)
        self.n_requests += 1

        # run the image through the hand pose estimator
        image = imgmsg_to_cv2(request.image)
        estimation = self.estimate_hand_poses(image)

        if self.cfg.debug.log_visualization.reconstruct_hand:
            path = Path(self.out_dir) / f"(hr)_hand_mesh_{self.n_requests:04d}.png"
            self.save_mesh_visualization(
                estimation=estimation,
                image=image,
                output_path=path,
            )
            path = Path(self.out_dir) / f"(hr)_hand_keypoints_{self.n_requests:04d}.png"
            self.save_keypoint_visualization(
                estimation=estimation,
                image=image,
                output_path=path,
            )

        # Convert NumPy arrays to lists for JSON serialization
        serializable_estimation = {}
        for key, value in estimation.items():
            if isinstance(value, np.ndarray):
                serializable_estimation[key] = value.tolist()
            else:
                serializable_estimation[key] = value

        # Prepare the response
        response = ReconstructHandResponse()
        response.success = True
        response.estimation_dict.data = json.dumps(serializable_estimation)

        return response

    def _reconstr_hand_pose_callback(
        self, request: ReconstructHandPoseRequest
    ) -> ReconstructHandPoseResponse:
        """Service callback to reconstruct the hand pose from an image.

        Args:
            request: ReconstructHandPoseRequest containing the image to process.

        Returns:
            ReconstructHandPoseResponse containing the transform from camera to
            hand and 2D keypoints.
        """

        # Validate request
        if not request.image:
            rospy.logerr("Invalid request: No image given.")
            return ReconstructHandPoseResponse(success=False)
        self.n_requests += 1

        # run the image through the hand pose estimator
        image = imgmsg_to_cv2(request.image)
        estimation = self.estimate_hand_poses(image)

        if self.cfg.debug.log_visualization.reconstruct_hand_pose:
            path = Path(self.out_dir) / f"(hr)_hand_mesh_{self.n_requests:04d}.png"
            self.save_mesh_visualization(
                estimation=estimation,
                image=image,
                output_path=path,
            )
            path = Path(self.out_dir) / f"(hr)_hand_keypoints_{self.n_requests:04d}.png"
            self.save_keypoint_visualization(
                estimation=estimation,
                image=image,
                output_path=path,
            )

        # Check how many hands were detected
        if estimation["n_hands"] == 0:
            rospy.logerr("No hands detected in the image.")
            return ReconstructHandPoseResponse(success=False)
        elif estimation["n_hands"] > 1:
            rospy.logwarn(
                f"Expected exactly one hand, but detected {estimation['n_hands']} hands. Taking the first one."
            )

        # Extract the transform
        rotation = estimation["hand_global_orient"][0, 0, ...]  # Shape (3, 3)
        translation = estimation["pred_keypoints_3d"][0, 0, :] + estimation["pred_cam_t_global"][0, :] # Shape (3,)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation
        transform_matrix[:3, 3] = translation
        transform_camera_to_hand = np_to_transformmsg(transform_matrix)

        # Extract the 2D keypoints
        keypoints_2d_np = estimation["pred_keypoints_2d_full"][0, ...]  # Shape (21, 2)
        labels = ["Point", "Pixel Coordinate"]
        keypoints_2d = np_to_multiarraymsg(keypoints_2d_np, Int32MultiArray, labels)

        # Prepare the response
        response = ReconstructHandPoseResponse()
        response.success = True
        response.transform_camera_to_hand = transform_camera_to_hand
        response.keypoints_2d = keypoints_2d

        return response

    def _estim_cam_callback(
        self, request: EstimateCameraRequest
    ) -> EstimateCameraResponse:
        """Service callback to estimate the camera parameters from an image.

        Args:
            request: EstimateCameraRequest containing the image to process.
        Returns:
            EstimateCameraResponse containing the estimated camera parameters.
        """

        # Validate request
        if not request.image:
            rospy.logerr("Invalid request: No image given.")
            return EstimateCameraResponse(success=False)
        self.n_requests += 1

        # run the image through the hand pose estimator
        image = imgmsg_to_cv2(request.image)
        estimation = self.estimate_hand_poses(image)

        if self.cfg.debug.log_visualization.estimate_camera:
            path = Path(self.out_dir) / f"(hr)_hand_mesh_{self.n_requests:04d}.png"
            self.save_mesh_visualization(
                estimation=estimation,
                image=image,
                output_path=path,
            )
            path = Path(self.out_dir) / f"(hr)_hand_keypoints_{self.n_requests:04d}.png"
            self.save_keypoint_visualization(
                estimation=estimation,
                image=image,
                output_path=path,
            )

        # Check how many hands were detected
        if estimation["n_hands"] == 0:
            rospy.logerr("No hands detected in the image.")
            return EstimateCameraResponse(success=False)
        elif estimation["n_hands"] > 1:
            rospy.logwarn(
                f"Expected exactly one hand, but detected {estimation['n_hands']} hands. Taking the first one."
            )

        # Extract the intrinsic matrix
        K = estimation["intrinsic_matrix"][0, ...]  # Shape (3, 3)
        camera_info = CameraInfo()
        camera_info.width = request.image.width
        camera_info.height = request.image.height
        camera_info.K = K.flatten().tolist()  # Flatten the matrix to a list

        # Prepare the response
        response = EstimateCameraResponse()
        response.success = True
        response.camera_info = camera_info

        return response
    
    def _render_hand_callback(
        self, request: RenderHandRequest
    ) -> RenderHandResponse:
        """Service callback to render the hand on the provided image.

        Args:
            request: RenderHandRequest containing the image and estimation data.

        Returns:
            RenderHandResponse containing the rendered image.
        """

        # Validate request
        if not request.image or not request.estimation_dict:
            rospy.logerr("Invalid request: No image or estimation data given.")
            return RenderHandResponse(success=False)

        self.n_requests += 1

        # Convert the image from ROS message to OpenCV format
        image = imgmsg_to_cv2(request.image)

        # Load the estimation data
        estimation = json.loads(request.estimation_dict.data)
        # Convert the estimation data to numpy arrays
        for key, value in estimation.items():
            if isinstance(value, list):
                estimation[key] = np.array(value)
            elif isinstance(value, dict):
                # Convert nested lists in dict to numpy arrays
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        estimation[key][sub_key] = np.array(sub_value)

        # We overwrite the predicted vertices by reconstructing them from the
        # hand pose parameters.
        if "pred_vertices" in estimation:
            rospy.logwarn("Ignoring the predicted vertices in the estimation data.")
        mano_out = self.hamer.mano(
            global_orient=torch.from_numpy(estimation["hand_global_orient"]).float().to(self.device),  # Shape: (1, 1, 3, 3)
            hand_pose=torch.from_numpy(estimation["hand_pose"]).float().to(self.device),  # Shape: (1, 15, 3, 3)
            betas=torch.from_numpy(estimation["hand_shape"]).float().to(self.device)  # Shape: (1, 10)
        )
        estimation["pred_vertices"] = mano_out.vertices.detach().cpu().numpy()

        # Render the hand on the image
        rendered_image = self._get_mesh_visualization(estimation, image)

        # Convert the rendered image back to ROS message format
        rendered_image_bgr = rendered_image.astype(np.uint8)
        rendered_image_msg = cv2_to_imgmsg(rendered_image_bgr, encoding="bgr8")
        
        response = RenderHandResponse()
        response.success = True
        response.rendered_image = rendered_image_msg

        return response

    def _detect_hands(self, img: np.ndarray) -> np.ndarray | None:
        """Detects the hands of all humans in an image using Detectron2 and ViTPose.

        Args:
            img: Image in BGR format. Can be loaded using cv2.imread.

        Returns:
            np.ndarray: Array of boundary boxes for the detected hands of each person.
                        The left hand boundary box for person i is at [i][0], the right
                        hand is at [i][1]. The boundary boxes themselfs are an array of
                        the form [x_min, y_min, x_max, y_max]. Undetected hands are
                        represented by an array of NaNs. Output shape is (n_persons, 2, 4).
                        If no hand is detected, the return value is None
        """

        # Detect humans in image
        det_out = self.body_detector(img)
        img = img.copy()[:, :, ::-1]  # Convert to RGB format

        if self.cfg.debug.log_detections:
            path = Path(self.out_dir) / f"(hr)_human_detection_{self.n_requests:04d}.txt"
            with open(path, "w") as f:
                instances = det_out["instances"]
                for i in range(len(instances)):
                    bbox = instances.pred_boxes.tensor[i].cpu().numpy()
                    score = instances.scores[i].cpu().numpy()
                    class_id = instances.pred_classes[i].cpu().numpy()
                    if class_id == 0:  # Only log human detections
                        f.write(f"Person {i}: bbox={bbox}, confidence={score}\n")

        # Extract predicted bounding boxes (around human) and scores
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (
            det_instances.scores > self.cfg.body_detector_confidence_threshold
        )
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = self.keypoint_detector.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        if self.cfg.debug.log_detections:
            path = Path(self.out_dir) / f"(hr)_vitpose_detection_{self.n_requests:04d}.txt"
            with open(path, "w") as f:
                for i, vitposes in enumerate(vitposes_out):
                    f.write(f"Person {i}:\n")
                    for j, point in enumerate(vitposes["keypoints"]):
                        f.write(f"  Keypoint {j}: coordinates={point[:2]}, confidence={point[2]}\n")

        bboxes = []

        # Create boundary box for each person
        for i, vitposes in enumerate(vitposes_out):
            left_hand_keyp = vitposes["keypoints"][-42:-21]
            right_hand_keyp = vitposes["keypoints"][-21:]

            # Rejecting not confident detections
            bboxes.append(
                np.array(
                    [
                        self._get_bbox_from_keypoints(
                            left_hand_keyp, self.cfg.hand_detector_confidence_threshold
                        ),
                        self._get_bbox_from_keypoints(
                            right_hand_keyp, self.cfg.hand_detector_confidence_threshold
                        ),
                    ]
                )
            )

        # Return None if nothing is detected
        return np.stack(bboxes) if bboxes else None

    @staticmethod
    def _get_bbox_from_keypoints(keypoints: np.ndarray, threshold: float) -> np.ndarray:
        """Get hand bounding box from hand keypoints.

        Args:
            keypoints: Keypoints of the hand, shape (21, 3).
            threshold: Threshold for keypoint confidence.

        Returns:
            np.ndarray: Bounding box of the hand. If not enough keypoints are detected,
                        returns [Nan, NaN, NaN, NaN].
        """
        valid = keypoints[:, 2] > threshold
        if sum(valid) > 3:
            bbox = [
                keypoints[valid, 0].min(),
                keypoints[valid, 1].min(),
                keypoints[valid, 0].max(),
                keypoints[valid, 1].max(),
            ]
            return bbox
        else:
            return np.array([np.nan, np.nan, np.nan, np.nan])

    @staticmethod
    def _get_intrinsic_matrix(
        fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor
    ) -> torch.Tensor:
        """Get camera intrinsic matrix from focal lengths and principal point.

        Args:
            fx (torch.Tensor): Focal length in x direction, shape (n,).
            fy (torch.Tensor): Focal length in y direction, shape (n,).
            cx (torch.Tensor): Principal point x coordinate, shape (n,).
            cy (torch.Tensor): Principal point y coordinate, shape (n,).

        Returns:
            torch.Tensor: Camera intrinsic matrices, shape (n, 3, 3).
        """
        # Ensure all inputs are tensors and on the same device
        device = fx.device
        dtype = fx.dtype

        K = torch.zeros((fx.shape[0], 3, 3), device=device, dtype=dtype)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        return K

    def estimate_hand_poses(self, image: np.ndarray | str) -> dict:
        """Estimates the hand poses present in the given image.
            The image should contain a single person.

        Args:
            image: Path to an image or an image in BGR format (as a numpy array).

        Returns:
            out_dict: Dictionary containing the estimated hand pose parameters.
                      The dictionary contains the following keys:
                      - "hand_pose": Hand pose parameters (shape: (n, 15, 3, 3)).
                      - "hand_shape": Hand shape parameters (shape: (n, 10)).
                      - "intrinsic_matrix": Camera intrinsic matrix (shape: (n, 3, 3)).
                      - "pred_keypoints_2d": Predicted 2D keypoints (shape: (n, 21, 3)).
                      where n is the number of detected hands in the image.
        """

        BATCH_SIZE = 16

        # Check if the input is a path or an image
        if isinstance(image, str):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image path {img_path} does not exist.")
            pose_img_cv2 = cv2.imread(str(img_path))
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Input image must be a BGR image with 3 channels.")
            pose_img_cv2 = image
        else:
            raise TypeError(
                "Input must be a path to an image or a BGR image as a numpy array."
            )

        # Get the bounding boxes of all hands in the image
        bboxes = self._detect_hands(pose_img_cv2)

        if bboxes is None:
            raise Exception("No hand detected. Please check the image.")

        # Keep track, which hand is left and which is right
        is_right = np.zeros([bboxes.shape[0], bboxes.shape[1], 1])
        is_right[:, 1, :] = 1

        # Remove bounding boxes for undetected hands
        valid_idxs = ~np.isnan(bboxes).any(axis=2)
        bboxes = bboxes[valid_idxs]
        is_right = is_right[valid_idxs]

        n_hands = bboxes.shape[0]  # Number of detected hands in the image

        # Create a dataset from the single image
        dataset_pose = ViTDetDataset(
            self.hamer_cfg,
            pose_img_cv2,
            bboxes,
            is_right,
            rescale_factor=self.cfg.rescale_factor,
        )
        dataloader_pose = torch.utils.data.DataLoader(
            dataset_pose, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        out_dict = {
            # --- Direct outputs of hamer -------------------------------------
            "focal_length": np.empty(
                [n_hands, 2]
            ),  # focal length in x and y direction in the full image frame
            "hand_global_orient": np.empty(
                [n_hands, 1, 3, 3]
            ),  # 3x3 rotation matrix of the wrist in the full image frame
            "hand_pose": np.empty(
                [n_hands, 15, 3, 3]
            ),  # 3x3 rotation matrix per joint per hand
            "hand_shape": np.empty([n_hands, 10]),  # 10 shape parameters
            "is_right": np.empty(
                [n_hands, 1]
            ),  # indicates whether it's the right or the left hand
            "pred_cam": np.empty(
                [n_hands, 3]
            ),  # Camera params f, cx, cy in the bbox frame
            "pred_cam_t": np.empty(
                [n_hands, 3]
            ),  # predicted camera translation tx, ty, tz in the bbox frame
            "pred_keypoints_2d": np.empty(
                [n_hands, 21, 2]
            ),  # predicted 2d keypoints in the cropped bbox 
            "pred_keypoints_3d": np.empty(
                [n_hands, 21, 3]
            ),  # predicted 3d keypoints. Add pred_cam_t_global to get the camera frame coordinates.
            "pred_vertices": np.empty(
                [n_hands, 778, 3]
            ),  # predicted 3d vertices. Add pred_cam_t_global to get the camera frame coordinates.
            # --- Processed output --------------------------------------------
            "pred_cam_t_global": np.empty(
                [n_hands, 3]
            ),  # predicted camera translation tx, ty, tz in the full image frame
            "scaled_focal_length": np.empty(
                [n_hands, 1]
            ),  # scaled focal length adjusted to the image resolution in the full image frame
            "intrinsic_matrix": np.empty(
                [n_hands, 3, 3]
            ),  # intrinsic matrix in the full image frame
            "pred_keypoints_2d_full": np.empty(
                [n_hands, 21, 2]
            ),  # predicted 2d keypoints in the full image
            "n_hands": n_hands,  # Number of detected hands in the image
        }

        # Run the model
        for i, batch in enumerate(dataloader_pose):
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.hamer(batch)

            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, n_hands)

            # Get the hamer outputs
            out_dict["focal_length"][start_idx:end_idx] = (
                out["focal_length"].detach().cpu().float().numpy()
            )
            out_dict["hand_global_orient"][start_idx:end_idx] = (
                out["pred_mano_params"]["global_orient"].detach().cpu().float().numpy()
            )
            out_dict["pred_cam"][start_idx:end_idx] = (
                out["pred_cam"].detach().cpu().float().numpy()
            )
            out_dict["hand_pose"][start_idx:end_idx] = (
                out["pred_mano_params"]["hand_pose"].detach().cpu().float().numpy()
            )
            out_dict["hand_shape"][start_idx:end_idx] = (
                out["pred_mano_params"]["betas"].detach().cpu().float().numpy()
            )
            out_dict["is_right"][start_idx:end_idx] = (
                batch["right"].detach().cpu().float().numpy()
            )
            out_dict["pred_vertices"][start_idx:end_idx] = (
                out["pred_vertices"].detach().cpu().float().numpy()
            )
            out_dict["pred_keypoints_3d"][start_idx:end_idx] = (
                out["pred_keypoints_3d"].detach().cpu().float().numpy()
            )
            out_dict["pred_cam_t"][start_idx:end_idx] = (
                out["pred_cam_t"].detach().cpu().float().numpy()
            )
            out_dict["pred_keypoints_2d"][start_idx:end_idx] = (
                out["pred_keypoints_2d"].detach().cpu().float().numpy()
            )
            out_dict["pred_keypoints_3d"][start_idx:end_idx] = (
                out["pred_keypoints_3d"].detach().cpu().float().numpy()
            )
            out_dict["pred_vertices"][start_idx:end_idx] = (
                out["pred_vertices"].detach().cpu().float().numpy()
            )

            # Calculate the global camera parameters
            multiplier = 2 * batch["right"] - 1  # 1 for right hand, -1 for left hand
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier[:, 0] * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                self.hamer_cfg.EXTRA.FOCAL_LENGTH
                / self.hamer_cfg.MODEL.IMAGE_SIZE
                * img_size.max()
            )
            out_dict["scaled_focal_length"][start_idx:end_idx] = (
                scaled_focal_length.detach().cpu().float().numpy()
            )
            pred_cam_t_global = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            )
            out_dict["pred_cam_t_global"][start_idx:end_idx] = (
                pred_cam_t_global.detach().cpu().float().numpy()
            )
            out_dict["intrinsic_matrix"][start_idx:end_idx] = (
                self._get_intrinsic_matrix(
                    scaled_focal_length.repeat(end_idx - start_idx),
                    scaled_focal_length.repeat(end_idx - start_idx),
                    pred_cam_t_global[:, 0],
                    pred_cam_t_global[:, 1],
                )
                .detach()
                .cpu()
                .numpy()
            )
            multiplier = torch.stack([multiplier, torch.ones_like(multiplier)], dim=-1)
            out_dict["pred_keypoints_2d_full"][start_idx:end_idx] = (
                (out["pred_keypoints_2d"][start_idx:end_idx] * box_size * multiplier + box_center).detach().cpu().float().numpy()
            )

        return out_dict

    def _get_mesh_visualization(self, estimation: dict, image: np.ndarray) -> np.ndarray:
        """Renders the estimated hand mesh into the original image

        Args:
            estimation (dict): Dictionary containing hand pose estimation
                outputs according to the function "estimate_hand_poses"
            image (np.ndarray): Image in BGR format to visualize the hand poses on.

        Returns:
            np.ndarry: the original image with all predicted hands rendered
                in it.
        """

        # Open the original image and get the size
        img_size = torch.tensor(
            [image.shape[1], image.shape[0]], device=self.device
        ).float()  # width x height

        # Return the original image if no hands are detected
        if estimation["pred_vertices"].shape[0] == 0:
            return image

        # We expect the same focal length for all hands -> Take the first one
        focal_length = estimation["scaled_focal_length"][0]

        # Prepare the arguments for the renderer
        misc_args = dict(
            mesh_base_color=(0.8, 0.8, 0.8),
            scene_bg_color=(1, 1, 1),
            focal_length=focal_length,
        )
        multiplier = (
            2 * estimation["is_right"] - 1
        )  # 1 for right hand, -1 for left hand
        vertices = estimation["pred_vertices"]
        vertices[:, :, 0] = multiplier * vertices[:, :, 0]
        vertices = [v for v in vertices]
        cam_t = [t for t in estimation["pred_cam_t_global"]]
        is_right = [r for r in estimation["is_right"]]

        # Render the hands
        cam_view = self.hand_renderer.render_rgba_multiple(
            vertices,
            cam_t=cam_t,
            render_res=img_size,
            is_right=is_right,
            **misc_args,
        )

        # Bring the image into the right form
        image = image.astype(np.float32)[:, :, ::-1] / 255.0
        image = np.concatenate(
            [image, np.ones_like(image[:, :, :1])], axis=2
        )  # Add alpha channel

        # Blend the rendered hands with the original image
        image = (
            image[:, :, :3] * (1 - cam_view[:, :, 3:])
            + cam_view[:, :, :3] * cam_view[:, :, 3:]
        )

        # Scale the values back to [0, 255]
        image = 255 * image[:, :, ::-1]

        return image
    
    def _get_keypoint_visualization(self, estimation: dict, image: np.ndarray) -> np.ndarray:
        """Renders the estimated hand keypoints into the original image

        Args:
            estimation (dict): Dictionary containing hand pose estimation
                outputs according to the function "estimate_hand_poses"
            image (np.ndarray): Image in BGR format to visualize the hand poses on.

        Returns:
            np.ndarry: the original image with all predicted hands rendered
                in it.
        """
        # Ensure the correct data type
        image = image.astype(np.uint8)

        # Return the original image if no hands are detected
        keypoints_2d = estimation["pred_keypoints_2d_full"]
        if keypoints_2d.shape[0] == 0:
            return image
        
        # Define the hand skeleton connections (using standard 21 keypoints)
        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (0, 9), (9,10), (10,11), (11,12),      # Middle
            (0,13), (13,14), (14,15), (15,16),     # Ring
            (0,17), (17,18), (18,19), (19,20)      # Pinky
        ]

        # Draw keypoints and skeleton for each detected hand
        for hand in keypoints_2d:
            # Draw keypoints
            for x, y in hand:
                cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
                # Draw skeleton lines
                for start, end in hand_connections:
                    x1, y1 = hand[start]
                    x2, y2 = hand[end]
                    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        return image

    def visualize_mesh(self, estimation: dict, img_path: str):
        """Renders the estimated hand into the original image

        Args:
            estimation (dict): Dictionary containing hand pose estimation
                outputs according to the function "estimate_hand_poses
            img_path (str): path to the original image
        """

        img = self._get_mesh_visualization(self, estimation, img_path)
        cv2.imshow("Hand poses", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_mesh_visualization(self, estimation: dict, image: np.ndarray, output_path: str):
        """Renders the estimated hand into the original image and saves the
            result at the specified path

        Args:
            estimation (dict): Dictionary containing hand pose estimation
                outputs according to the function "estimate_hand_poses"
            image (np.ndarray): Image in BGR format to visualize the hand poses on
            output_path (str): path where to store the resulting image
        """

        out = self._get_mesh_visualization(estimation, image)
        cv2.imwrite(output_path, out)

    def save_keypoint_visualization(self, estimation: dict, image: np.ndarray, output_path: str):
        """Renders the estimated hand keypoints into the original image and saves the
            result at the specified path

        Args:
            estimation (dict): Dictionary containing hand pose estimation
                outputs according to the function "estimate_hand_poses"
            image (np.ndarray): Image in BGR format to visualize the hand poses on
            output_path (str): path where to store the resulting image
        """

        out = self._get_keypoint_visualization(estimation, image)
        cv2.imwrite(output_path, out)

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
        HandReconstructor(cfg)
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("GraspGenerator node interrupted.")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
