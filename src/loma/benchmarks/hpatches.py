import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from loma.loma import LoMa

# Viewpoint sequences only, without the sequences LoFTR and RoMa drop.
IGNORE = {"v_artisans", "v_astronautis", "v_talent"}


class HPatchesAffine:
    """Affine accuracy on HPatches, where the GT affine is the Jacobian of the homography.

    Only matches within `reproj_thresh` pixels (at a canonical 480-px scale) of the GT warp are scored, so the metrics isolate the affine.
    Errors are in the normalized frame [-1, 1]^2 of each image: median Frobenius error (x100) and residual rotation.
    """

    def __init__(
        self,
        data_root="data/hpatches/hpatches-sequences-release",
        num_keypoints: int = 2048,
        reproj_thresh: float = 3.0,
    ) -> None:
        self.data_root = data_root
        self.num_keypoints = num_keypoints
        self.reproj_thresh = reproj_thresh

    def benchmark(self, model: LoMa):
        frob, rot = [], []
        seqs = [s for s in sorted(os.listdir(self.data_root)) if s.startswith("v_") and s not in IGNORE]
        for seq in tqdm(seqs):
            w1, h1 = Image.open(f"{self.data_root}/{seq}/1.ppm").size
            for i in range(2, 7):
                w2, h2 = Image.open(f"{self.data_root}/{seq}/{i}.ppm").size
                H = np.loadtxt(f"{self.data_root}/{seq}/H_1_{i}")
                kptsA, kptsB, A = model.match(
                    f"{self.data_root}/{seq}/1.ppm",
                    f"{self.data_root}/{seq}/{i}.ppm",
                    num_keypoints=self.num_keypoints,
                    return_affine=True,
                )
                # GT warp and Jacobian at the pixel centers of the keypoints in A.
                p = kptsA - 0.5
                q = np.concatenate([p, np.ones((len(p), 1))], 1) @ H.T
                warped, d = q[:, :2] / q[:, 2:], q[:, 2]
                J = (H[:2, :2][None] - warped[:, :, None] * H[2, :2][None, None]) / d[:, None, None]
                # To the normalized frames: x_norm = 2 x / (w, h), A_norm = S_B^-1 A S_A with S = diag(w, h) / 2.
                S_A, S_B = np.diag([w1, h1]) / 2, np.diag([w2, h2]) / 2
                A_gt = np.linalg.inv(S_B) @ J @ S_A
                A_pred = np.linalg.inv(S_B) @ A @ S_A
                err = 2 * (kptsB - 0.5 - warped) / [w2, h2]
                keep = (np.linalg.norm(err, axis=1) * 240 < self.reproj_thresh) & (np.linalg.det(A_gt) > 0)
                A_gt, A_pred = A_gt[keep], A_pred[keep]
                frob.append(np.linalg.norm(A_pred - A_gt, axis=(1, 2)))
                R = A_pred @ np.linalg.inv(A_gt)
                rot.append(np.degrees(np.abs(np.arctan2(R[:, 1, 0] - R[:, 0, 1], R[:, 0, 0] + R[:, 1, 1]))))
        frob, rot = np.concatenate(frob), np.concatenate(rot)
        return {"frobenius_x100": 100 * np.median(frob), "rotation_deg": np.median(rot), "n_matches": len(frob)}
