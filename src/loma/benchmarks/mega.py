import numpy as np
from tqdm import tqdm
import logging

from loma.loma import LoMa
from loma.geometry import (
    compute_pose_error,
    compute_relative_pose,
    estimate_pose_cv2_ransac,
    pose_auc,
)

logger = logging.getLogger(__name__)


class MegaDepthPoseEstimationBenchmark:
    def __init__(
        self,
        data_root="data/megadepth",
        num_keypoints: int = 4096,
    ) -> None:
        self.data_root = data_root
        self.num_keypoints = num_keypoints
        self._post_init()

    def _post_init(self):
        raise NotImplementedError(
            "Add scene names and benchmark name in derived class _post_init"
        )

    def benchmark(
        self,
        model: LoMa,
        num_ransac_runs: int = 5,
        sample_every: int = 1,
    ):
        self.scenes = [
            np.load(f"{self.data_root}/{scene}", allow_pickle=True)
            for scene in self.scene_names
        ]

        data_root = self.data_root
        thresholds = [5, 10, 20]
        tot_e_t, tot_e_R, tot_e_pose = [], [], []
        for scene_ind in range(len(self.scenes)):
            scene = self.scenes[scene_ind]
            pairs = scene["pair_infos"]
            intrinsics = scene["intrinsics"]
            poses = scene["poses"]
            im_paths = scene["image_paths"]
            pair_inds = range(len(pairs))
            for pairind in (
                pbar := tqdm(
                    pair_inds[::sample_every],
                    desc="Current AUC: ?",
                    mininterval=10,
                )
            ):
                idx1, idx2 = pairs[pairind][0]
                K1 = intrinsics[idx1].copy()
                T1 = poses[idx1].copy()
                R1, t1 = T1[:3, :3], T1[:3, 3]
                K2 = intrinsics[idx2].copy()
                T2 = poses[idx2].copy()
                R2, t2 = T2[:3, :3], T2[:3, 3]
                R, t = compute_relative_pose(R1, t1, R2, t2)
                im_A_path = f"{data_root}/{im_paths[idx1]}"
                im_B_path = f"{data_root}/{im_paths[idx2]}"

                kpts1, kpts2 = model.match(
                    im_A_path, im_B_path, num_keypoints=self.num_keypoints
                )

                for _ in range(num_ransac_runs):
                    shuffling = np.random.permutation(np.arange(len(kpts1)))
                    kpts1 = kpts1[shuffling]
                    kpts2 = kpts2[shuffling]

                    try:
                        threshold = 0.5
                        norm_threshold = threshold / (
                            np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2]))
                        )
                        R_est, t_est, _ = estimate_pose_cv2_ransac(
                            kpts1,
                            kpts2,
                            K1,
                            K2,
                            norm_threshold,
                            conf=0.99999,
                        )
                        e_t, e_R = compute_pose_error(R_est, t_est[:, 0], R, t)
                        e_pose = max(e_t, e_R)
                    except Exception as e:
                        logger.debug(f"Pose estimation error: {e}")
                        e_t, e_R = 90, 90
                        e_pose = max(e_t, e_R)
                    tot_e_t.append(e_t)
                    tot_e_R.append(e_R)
                    tot_e_pose.append(e_pose)

                pbar.set_postfix(
                    auc=f"{[f'{a.item():.3f}' for a in pose_auc(tot_e_pose, thresholds)]}"
                )
        tot_e_pose = np.array(tot_e_pose)
        auc = pose_auc(tot_e_pose, thresholds)
        acc_5 = (tot_e_pose < 5).mean()
        acc_10 = (tot_e_pose < 10).mean()
        acc_15 = (tot_e_pose < 15).mean()
        acc_20 = (tot_e_pose < 20).mean()
        map_5 = acc_5
        map_10 = np.mean([acc_5, acc_10])
        map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
        logger.info("%s auc: %s", model.name, auc)
        return {
            "auc_5": auc[0],
            "auc_10": auc[1],
            "auc_20": auc[2],
            "map_5": map_5,
            "map_10": map_10,
            "map_20": map_20,
        }


class Mega1500(MegaDepthPoseEstimationBenchmark):
    def _post_init(self):
        self.scene_names = [
            "0015_0.1_0.3.npz",
            "0015_0.3_0.5.npz",
            "0022_0.1_0.3.npz",
            "0022_0.3_0.5.npz",
            "0022_0.5_0.7.npz",
        ]
        self.benchmark_name = "Mega1500"
        self.model = "essential"


class Mega1500_F(MegaDepthPoseEstimationBenchmark):
    def _post_init(self):
        self.scene_names = [
            "0015_0.1_0.3.npz",
            "0015_0.3_0.5.npz",
            "0022_0.1_0.3.npz",
            "0022_0.3_0.5.npz",
            "0022_0.5_0.7.npz",
        ]
        # self.benchmark_name = "Mega1500_F"
        self.model = "fundamental"


class MegaIMCPT(MegaDepthPoseEstimationBenchmark):
    def _post_init(self):
        self.scene_names = [
            "mega_8_scenes_0008_0.1_0.3.npz",
            "mega_8_scenes_0008_0.3_0.5.npz",
            "mega_8_scenes_0019_0.1_0.3.npz",
            "mega_8_scenes_0019_0.3_0.5.npz",
            "mega_8_scenes_0021_0.1_0.3.npz",
            "mega_8_scenes_0021_0.3_0.5.npz",
            "mega_8_scenes_0024_0.1_0.3.npz",
            "mega_8_scenes_0024_0.3_0.5.npz",
            "mega_8_scenes_0025_0.1_0.3.npz",
            "mega_8_scenes_0025_0.3_0.5.npz",
            "mega_8_scenes_0032_0.1_0.3.npz",
            "mega_8_scenes_0032_0.3_0.5.npz",
            "mega_8_scenes_0063_0.1_0.3.npz",
            "mega_8_scenes_0063_0.3_0.5.npz",
            "mega_8_scenes_1589_0.1_0.3.npz",
            "mega_8_scenes_1589_0.3_0.5.npz",
        ]
        # self.benchmark_name = "MegaIMCPT"
        self.model = "essential"


class MegaIMCPT_F(MegaDepthPoseEstimationBenchmark):
    def _post_init(self):
        self.scene_names = [
            "mega_8_scenes_0008_0.1_0.3.npz",
            "mega_8_scenes_0008_0.3_0.5.npz",
            "mega_8_scenes_0019_0.1_0.3.npz",
            "mega_8_scenes_0019_0.3_0.5.npz",
            "mega_8_scenes_0021_0.1_0.3.npz",
            "mega_8_scenes_0021_0.3_0.5.npz",
            "mega_8_scenes_0024_0.1_0.3.npz",
            "mega_8_scenes_0024_0.3_0.5.npz",
            "mega_8_scenes_0025_0.1_0.3.npz",
            "mega_8_scenes_0025_0.3_0.5.npz",
            "mega_8_scenes_0032_0.1_0.3.npz",
            "mega_8_scenes_0032_0.3_0.5.npz",
            "mega_8_scenes_0063_0.1_0.3.npz",
            "mega_8_scenes_0063_0.3_0.5.npz",
            "mega_8_scenes_1589_0.1_0.3.npz",
            "mega_8_scenes_1589_0.3_0.5.npz",
        ]
        # self.benchmark_name = "MegaIMCPT_F"
        self.model = "fundamental"
