from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Generic, TypeVar
from PIL import Image
import h5py
import numpy as np
import torch

from ..logging import logger
from ..types import Batch

from .transforms import Transform


class COLMAPScene(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: Path,
        scene_info: dict[str, np.ndarray[Any, Any]],
        *,
        scene_name: str,
        min_overlap: float,
        max_overlap: float,
        max_num_pairs: int,
        use_sky_mask: bool,
        transform_cfg: Transform.Cfg,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.scene_name = (
            os.path.splitext(scene_name)[0] + f"_{min_overlap}_{max_overlap}"
        )
        self.use_sky_mask = use_sky_mask
        self.image_rel_paths = scene_info["image_paths"]
        self.depth_rel_paths = scene_info["depth_paths"]
        # assert isinstance(self.depth_rel_paths, dict), "depth_rel_paths must be a dictionary"
        # try to load some arbitrary depth to check if they work (just for debugging)
        # for i in self.depth_rel_paths:
        #     self.load_depth(self.data_root / self.depth_rel_paths[i])
        #     break

        self.intrinsics = scene_info["intrinsics"]
        self.poses = scene_info["poses"]
        self.pairs = scene_info["pairs"]
        self.overlaps = scene_info["overlaps"]
        threshold = (self.overlaps > min_overlap) & (self.overlaps < max_overlap)
        self.pairs = self.pairs[threshold]
        self.overlaps = self.overlaps[threshold]
        if len(self.pairs) > max_num_pairs:
            pairinds = np.random.choice(
                np.arange(0, len(self.pairs)), max_num_pairs, replace=False
            )
            self.pairs = self.pairs[pairinds]
            self.overlaps = self.overlaps[pairinds]
        self.transform = Transform(transform_cfg)

    def load_im(self, im_path: str) -> Image.Image:
        return Image.open(im_path)

    def load_depth(self, depth_path: str) -> torch.Tensor:
        with h5py.File(depth_path, "r") as f:
            depth_np = np.array(f["depth"])
        return torch.from_numpy(depth_np)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, pair_idx: int) -> Batch:
        try:
            return self.load_item(pair_idx)
        except Exception as e:
            print(f"Error loading item {pair_idx}: {e}")
            return self.load_item(np.random.choice(len(self)))

    def load_item(self, pair_idx: int):
        try:
            # read intrinsics of original size
            idx_A, idx_B = self.pairs[pair_idx]
            K_A = torch.tensor(
                self.intrinsics[idx_A].copy(), dtype=torch.float
            ).reshape(3, 3)
            K_B = torch.tensor(
                self.intrinsics[idx_B].copy(), dtype=torch.float
            ).reshape(3, 3)

            # read and compute relative poses
            T_A = torch.tensor(self.poses[idx_A]).float()
            T_B = torch.tensor(self.poses[idx_B]).float()
            T_AB = T_B @ torch.linalg.inv(T_A)

            # Load positive pair data
            im_A_rel_path, im_B_rel_path = (
                self.image_rel_paths[idx_A],
                self.image_rel_paths[idx_B],
            )
            depth_A_rel_path, depth_B_rel_path = (
                self.depth_rel_paths[idx_A],
                self.depth_rel_paths[idx_B],
            )
            im_A_path = self.data_root / im_A_rel_path
            im_B_path = self.data_root / im_B_rel_path
            depth_A_path = self.data_root / depth_A_rel_path
            depth_B_path = self.data_root / depth_B_rel_path
            im_A = self.load_im(im_A_path)
            im_B = self.load_im(im_B_path)
            depth_A = self.load_depth(depth_A_path)
            depth_B = self.load_depth(depth_B_path)
            if self.use_sky_mask:
                sky_mask_A_path = Path(
                    im_A_path.as_posix().replace("images", "sky_masks")
                ).with_suffix(".png")

                sky_mask_B_path = Path(
                    im_B_path.as_posix().replace("images", "sky_masks")
                ).with_suffix(".png")
                if not sky_mask_A_path.exists():
                    sky_mask_A = torch.ones_like(depth_A)
                    logger.warning(f"Sky mask not found for image {im_A_path}")
                else:
                    sky_mask_A = torch.from_numpy(
                        np.array(self.load_im(sky_mask_A_path))
                    )
                if not sky_mask_B_path.exists():
                    sky_mask_B = torch.ones_like(depth_B)
                    logger.warning(f"Sky mask not found for image {im_B_path}")
                else:
                    sky_mask_B = torch.from_numpy(
                        np.array(self.load_im(sky_mask_B_path))
                    )
                depth_A = depth_A * (sky_mask_A > 0)
                depth_B = depth_B * (sky_mask_B > 0)
            else:
                sky_mask_A = None
                sky_mask_B = None
            # Recompute camera intrinsic matrix due to the resize
            ret = self.transform(
                pil_img_A=im_A,
                pil_img_B=im_B,
                depth_A=depth_A,
                depth_B=depth_B,
                K_A=K_A,
                K_B=K_B,
            )
        except Exception as e:
            logger.warning(e)
            logger.warning(f"Failed to load image pair {self.pairs[pair_idx]}")
            logger.warning("Loading a random pair in scene instead")
            rand_ind = np.random.choice(range(len(self)))
            return self[rand_ind]
        assert (
            depth_A is not None
            and depth_B is not None
            and K_A is not None
            and K_B is not None
        )
        H, W = depth_A.shape[:2]
        return Batch(
            **ret,
            pose_A=T_A,
            pose_B=T_B,
            T_AB=T_AB,
            img_A_path=im_A_path,
            img_B_path=im_B_path,
            source="depth",
            quality="low",
        )


SceneType = TypeVar("SceneType", bound=COLMAPScene)


class COLMAPDataset(torch.utils.data.Dataset, Generic[SceneType]):
    @dataclass(frozen=True)
    class Cfg:
        split: str
        data_root: str | None = None
        alpha: float | None = None
        use_sky_mask: bool = False
        # Dataset-specific transform overrides (merged with global transform at setup time)
        transform_overrides: dict[str, Any] = field(default_factory=dict)
        weight: int = 1_000_000
        fraction: float = 1.0

    def __init__(self, cfg: Cfg, transform_cfg: Transform.Cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.transform_cfg = transform_cfg
        assert cfg.data_root is not None, "data_root must be provided"
        self.data_root = Path(cfg.data_root)
        self.scene_info_root = self.data_root / "prep_scene_info"
        self.all_scenes = sorted(os.listdir(self.scene_info_root))
        self.cum_weights: torch.Tensor | None = None
        self.scenes: list[SceneType] | None = None

    def __len__(self):
        return self.cfg.weight

    def __getitem__(self, idx: int) -> Batch:
        assert self.cum_weights is not None
        assert self.scenes is not None
        scene_index = torch.searchsorted(self.cum_weights, (idx + 0.5) / len(self))
        scene = self.scenes[scene_index]
        pair_idx = np.random.choice(len(scene))
        return scene[pair_idx]
