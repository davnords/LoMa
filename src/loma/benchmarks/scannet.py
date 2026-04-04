import os.path as osp
import logging
import numpy as np
from tqdm import tqdm

from loma.geometry import compute_pose_error, pose_auc, estimate_pose_cv2_ransac
from loma.loma import LoMa

logger = logging.getLogger(__name__)


class ScanNet1500:
    def __init__(
        self, data_root="data/scannet/scans", num_keypoints: int = 4096
    ) -> None:
        self.data_root = data_root
        self.num_keypoints = num_keypoints

    def benchmark(
        self,
        model: LoMa,
        num_ransac_runs: int = 5,
    ):
        thresholds = [5, 10, 20]
        data_root = self.data_root
        tmp = np.load(osp.join(data_root, "test.npz"))
        pairs, rel_pose = tmp["name"], tmp["rel_pose"]
        tot_e_t, tot_e_R, tot_e_pose = [], [], []
        pair_inds = range(len(pairs))
        for pairind in (pbar := tqdm(pair_inds, smoothing=0.9)):
            scene = pairs[pairind]
            scene_name = f"scene0{scene[0]}_00"
            im_A_path = osp.join(
                self.data_root,
                "scans_test",
                scene_name,
                "color",
                f"{scene[2]}.jpg",
            )
            im_B_path = osp.join(
                self.data_root,
                "scans_test",
                scene_name,
                "color",
                f"{scene[3]}.jpg",
            )
            T_gt = rel_pose[pairind].reshape(3, 4)
            R, t = T_gt[:3, :3], T_gt[:3, 3]
            intrinsic_path = osp.join(
                self.data_root,
                "scans_test",
                scene_name,
                "intrinsic",
                "intrinsic_color.txt",
            )
            with open(intrinsic_path, "r") as f:
                K = np.stack(
                    [
                        np.array([float(i) for i in r.split()])
                        for r in f.read().split("\n")
                        if r
                    ]
                )

            kpts1, kpts2 = model.match(im_A_path, im_B_path)

            # scale1 = 480 / min(w1, h1)
            # scale2 = 480 / min(w2, h2)
            # w1, h1 = scale1 * w1, scale1 * h1
            # w2, h2 = scale2 * w2, scale2 * h2
            K1 = K.copy()
            K2 = K.copy()
            # K1[:2] = K1[:2] * scale1
            # K2[:2] = K2[:2] * scale2

            for _ in range(num_ransac_runs):
                shuffling = np.random.permutation(np.arange(len(kpts1)))
                kpts1 = kpts1[shuffling]
                kpts2 = kpts2[shuffling]
                try:
                    norm_threshold = 0.5 / (
                        np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2]))
                    )
                    R_est, t_est, mask = estimate_pose_cv2_ransac(
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
