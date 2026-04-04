import ast
import json
import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm
from scipy.optimize import least_squares
from dataclasses import dataclass
from typing import Literal
from urllib.request import urlopen

from loma.geometry import pose_auc
from loma.random import set_seed
from loma.loma import LoMa


# Reference code: https://github.com/thibautloiseau/RUBIK/blob/main/eval.py
class RubikBenchmark:
    @dataclass(frozen=True)
    class Cfg:
        # Preprocess the data by following the instructions at https://github.com/thibautloiseau/RUBIK/tree/main
        data_path: str = "data/nuScenes"
        seed: int | None = None
        estimate_pose_method: Literal["essential", "fundamental"] = "essential"
        num_keypoints: int = 4096

    def __init__(
        self,
        cfg: Cfg | None = None,
    ) -> None:
        if cfg is None:
            cfg = RubikBenchmark.Cfg()
        self.cfg = cfg
        if cfg.seed is not None:
            set_seed(cfg.seed)
        self.prefix = f"{type(self).__name__}"

    def benchmark(self, model: LoMa):
        estimate_pose = self.cfg.estimate_pose_method
        with urlopen(
            "https://raw.githubusercontent.com/thibautloiseau/RUBIK/refs/heads/main/rubik.json"
        ) as f:
            data = json.load(f)

        # Get all scenes to get intrinsics to be able to recover pose
        all_scenes = list(set([scene for box in data for scene in data[box]]))

        # Collect errors for AUC computation
        tot_R_err = []
        tot_t_err_angle = []
        tot_t_err_metric = []
        tot_pose_err = []  # max of R_err and t_err_angle

        # Success is defined in the paper as rotation error less than 5 degrees and translation error less than 2m
        success = 0
        total = 0

        for scene in tqdm(all_scenes):
            # We iterate over scenes whether than boxes to avoid loading all images from every scene at once
            for box in (pbar := tqdm(data, leave=False)):
                if data[box].get(scene) is None:
                    continue

                pairs = [ast.literal_eval(el) for el in list(data[box][scene].keys())]
                paths = [
                    [
                        osp.join(
                            self.cfg.data_path,
                            "sweeps",
                            el[0].split("__")[1].split("__")[0],
                            el[0],
                        ),
                        osp.join(
                            self.cfg.data_path,
                            "sweeps",
                            el[1].split("__")[1].split("__")[0],
                            el[1],
                        ),
                    ]
                    for el in pairs
                ]

                for i, pair in enumerate(paths):
                    # Get gt pose
                    gt_pose = np.array(data[box][scene][str(pairs[i])]["rel_pose"])

                    mkpts1, mkpts2 = model.match(
                        pair[0], pair[1], num_keypoints=self.cfg.num_keypoints
                    )

                    # Estimate pose
                    K1 = np.array(data[box][scene][str(pairs[i])]["K1"])
                    K2 = np.array(data[box][scene][str(pairs[i])]["K2"])
                    ret = None
                    if estimate_pose == "essential":
                        ret = self.estimate_pose_essential(mkpts1, mkpts2, K1, K2, 0.5)

                    elif estimate_pose == "fundamental":
                        ret = self.estimate_pose_fundamental(
                            mkpts1, mkpts2, K1, K2, 0.5
                        )

                    if ret is None:
                        R_err = 90  # Failed pose estimation
                        t_err_angle = 90
                        t_err_metric = np.inf
                    else:
                        R_est, t_est, _ = ret

                        # Normalize t_est
                        t_est = t_est / np.linalg.norm(t_est)

                        # Get scale factor using unidepths by minimizing distance to 3D points after applying transformation
                        depth1 = np.load(
                            f"{self.cfg.data_path}/unidepths/{osp.basename(pair[0]).replace('.jpg', '.npy')}"
                        )
                        depth2 = np.load(
                            f"{self.cfg.data_path}/unidepths/{osp.basename(pair[1]).replace('.jpg', '.npy')}"
                        )

                        # Get 3D points
                        pts3D_1 = self.backproject_to_3D(mkpts1, depth1, K1)
                        pts3D_2 = self.backproject_to_3D(mkpts2, depth2, K2)

                        # Get scale factor
                        scale = self.get_scale(
                            self.scale_cost_function, 1, R_est, t_est, pts3D_1, pts3D_2
                        )
                        t_est *= scale

                        # Compute error
                        t_err_angle, t_err_metric, R_err = self.relative_pose_error(
                            gt_pose, R_est, t_est
                        )

                    tot_R_err.append(R_err)
                    tot_t_err_angle.append(t_err_angle)
                    tot_t_err_metric.append(t_err_metric)
                    tot_pose_err.append(max(R_err, t_err_angle))

                    total += 1
                    if (R_err < 5.0) and (t_err_metric < 2.0):
                        success += 1
                    pbar.set_postfix(
                        auc=f"{[f'{a.item() * 100:.1f}' for a in pose_auc(tot_pose_err, [5, 10, 20])]}",
                        success_ratio=f"{success / total:.1f}",
                    )

        # Compute summary statistics
        tot_R_err = np.array(tot_R_err)
        tot_t_err_metric = np.array(tot_t_err_metric)
        tot_pose_err = np.array(tot_pose_err)

        # AUC for rotation error (thresholds in degrees)
        R_thresholds = [5, 10, 20]
        R_auc = pose_auc(tot_R_err, R_thresholds)

        # AUC for translation error (thresholds in meters)
        t_thresholds = [0.5, 1.0, 2.0]
        t_auc = pose_auc(tot_t_err_metric, t_thresholds)

        # AUC for combined pose error (max of R and t angular errors, thresholds in degrees)
        pose_thresholds = [5, 10, 20]
        pose_auc_vals = pose_auc(tot_pose_err, pose_thresholds)

        # Success ratio
        success_ratio = success / total if total > 0 else 0.0

        return {
            f"{self.prefix}_auc_5": pose_auc_vals[0] * 100,
            f"{self.prefix}_auc_10": pose_auc_vals[1] * 100,
            f"{self.prefix}_auc_20": pose_auc_vals[2] * 100,
            f"{self.prefix}_R_auc_5": R_auc[0] * 100,
            f"{self.prefix}_R_auc_10": R_auc[1] * 100,
            f"{self.prefix}_R_auc_20": R_auc[2] * 100,
            f"{self.prefix}_t_auc_0.5": t_auc[0] * 100,
            f"{self.prefix}_t_auc_1.0": t_auc[1] * 100,
            f"{self.prefix}_t_auc_2.0": t_auc[2] * 100,
            f"{self.prefix}_success_ratio": success_ratio * 100,
        }

    @staticmethod
    def estimate_pose_essential(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
        # From https://github.com/zju3dv/LoFTR/blob/master/src/utils/metrics.py#L72
        if len(kpts0) < 5:
            return None

        # normalize keypoints
        kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

        # normalize ransac threshold
        ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

        # compute pose with cv2
        E, mask = cv2.findEssentialMat(
            kpts0,
            kpts1,
            np.eye(3),
            threshold=ransac_thr,
            prob=conf,
            method=cv2.USAC_MAGSAC,
        )

        if E is None:
            return None

        # recover pose from E
        best_num_inliers = 0
        ret = None
        for _E in np.split(E, len(E) // 3):
            pose = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            n = int(pose[0])
            R = pose[1]
            t = pose[2]
            if n > best_num_inliers:
                ret = (R, t[:, 0], mask.ravel() > 0)
                best_num_inliers = n

        return ret

    @staticmethod
    def estimate_pose_fundamental(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
        if len(kpts0) < 7:
            return None

        # Find fundamental matrix using RANSAC
        try:
            F, mask = cv2.findFundamentalMat(
                kpts0,
                kpts1,
                ransacReprojThreshold=thresh,
                confidence=conf,
                method=cv2.USAC_MAGSAC,
            )
        except Exception as _:
            return None

        if F is None:
            return None

        # Convert fundamental matrix to essential matrix using intrinsics
        E = K1.T @ F @ K0

        # Recover pose from essential matrix
        best_num_inliers = 0
        ret = None
        for _E in np.split(E, len(E) // 3):
            pose = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            n = int(pose[0])
            R = pose[1]
            t = pose[2]
            if n > best_num_inliers:
                ret = (R, t[:, 0], mask.ravel() > 0)
                best_num_inliers = n

        return ret

    @staticmethod
    def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
        # Angle error between 2 vectors
        t_gt = T_0to1[:3, 3]
        n = np.linalg.norm(t) * np.linalg.norm(t_gt)
        t_err_angle = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
        t_err_angle = np.minimum(t_err_angle, 180 - t_err_angle)  # handle E ambiguity
        if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
            t_err_angle = 0

        # Metric translation error
        t_err_metric = np.linalg.norm(t - t_gt)

        # Angle error between 2 rotation matrices
        R_gt = T_0to1[:3, :3]
        cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
        cos = np.clip(cos, -1.0, 1.0)  # handle numerical errors
        R_err = np.rad2deg(np.abs(np.arccos(cos)))

        return t_err_angle, t_err_metric, R_err

    @staticmethod
    def backproject_to_3D(uv_points, depth, K):
        # Convert 2D points to homogeneous for and clip points outside image
        uv_homogeneous = (
            np.hstack((uv_points, np.ones((uv_points.shape[0], 1)))).round().astype(int)
        )
        uv_homogeneous = np.clip(uv_homogeneous, a_min=(0, 0, 1), a_max=(1599, 899, 1))

        selected_depths = depth[uv_homogeneous[:, 1], uv_homogeneous[:, 0]]

        # Scale by depth
        points_3D = selected_depths[:, None] * (np.linalg.inv(K) @ (uv_homogeneous.T)).T
        return points_3D

    @staticmethod
    def scale_cost_function(scale, R, t, pts3D_1, pts3D_2, delta=1.0):
        # Apply transformation
        pts3D_1 = (R @ pts3D_1.T).T + scale * t
        res = np.linalg.norm(pts3D_1 - pts3D_2, axis=1)

        # Huber loss with 1m threshold
        cost = np.where(res <= delta, 0.5 * res**2, delta * (res - 0.5 * delta))
        return cost

    @staticmethod
    def get_scale(func, scale_ini, R, t, pts3D_1, pts3D_2):
        res = least_squares(
            func, scale_ini, args=(R, t, pts3D_1, pts3D_2), bounds=(0, np.inf)
        )
        scale = res.x[0]
        return scale
