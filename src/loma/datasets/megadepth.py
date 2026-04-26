from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Literal
import numpy as np
import torch
from tqdm import tqdm

from ..types import Batch
from ..logging import logger

from .transforms import Transform
from .colmap_dataset import COLMAPScene


class MegaDepthScene(COLMAPScene):
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
        super().__init__(
            data_root=data_root,
            scene_info=scene_info,
            scene_name=scene_name,
            min_overlap=min_overlap,
            max_overlap=max_overlap,
            max_num_pairs=max_num_pairs,
            use_sky_mask=use_sky_mask,
            transform_cfg=transform_cfg,
        )


class MegaDepth(torch.utils.data.Dataset):
    @dataclass(frozen=True)
    class Cfg:
        split: Literal["dedode_train", "dedode_test"]
        alpha: float | None = None
        loftr_ignore: bool = True
        imc21_ignore: bool = True
        # Dataset-specific transform overrides (merged with global transform at setup time)
        # MegaDepth defaults to shake_t=32 for augmentation
        transform_overrides: dict[str, Any] = field(
            default_factory=lambda: {"shake_t": 32}
        )
        weight: int = 1_000_000
        data_root: str = "data/megadepth"
        use_sky_mask: bool = False
        fraction: float = 1.0

    def __init__(self, cfg: Cfg, transform_cfg: Transform.Cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.transform_cfg = transform_cfg
        self.data_root = Path(cfg.data_root)
        self.scene_info_root = self.data_root / "prep_scene_info"
        self.all_scenes = sorted(os.listdir(self.scene_info_root))
        self.test_scenes = ["0017.npy", "0004.npy", "0048.npy", "0013.npy"]
        # LoFTR did the D2-net preprocessing differently than we did and got more ignore scenes, can optionially ignore those
        self.loftr_ignore_scenes = set(
            [
                "0121.npy",
                "0133.npy",
                "0168.npy",
                "0178.npy",
                "0229.npy",
                "0349.npy",
                "0412.npy",
                "0430.npy",
                "0443.npy",
                "1001.npy",
                "5014.npy",
                "5015.npy",
                "5016.npy",
            ]
        )
        self.imc21_scenes = set(
            [
                "0008.npy",
                "0019.npy",
                "0021.npy",
                "0024.npy",
                "0025.npy",
                "0032.npy",
                "0063.npy",
                "1589.npy",
            ]
        )
        self.test_scenes_loftr = ["0015.npy", "0022.npy"]
        logger.info(f"Loading {type(self).__name__} {cfg.split=} scenes.")
        if cfg.split == "dedode_train":
            scene_split = "train_loftr"
            scenes = self.build_scenes(
                scene_split,
                min_overlap=0.01,
                max_overlap=1.0,
                max_num_pairs=100_000,
            ) + self.build_scenes(
                scene_split,
                min_overlap=0.35,
                max_overlap=1.0,
                max_num_pairs=100_000,
            )
            self.scenes = scenes
        elif cfg.split == "dedode_test":
            scene_split = "test_loftr"
            scenes = self.build_scenes(
                scene_split,
                min_overlap=0.01,
                max_overlap=1.0,
                max_num_pairs=100_000,
            )
            self.scenes = scenes
        else:
            raise ValueError(f"Split {cfg.split} not available")
        logger.info(
            f"Built {len(scenes)} scenes with a total of {sum([len(scene) for scene in self.scenes])} pairs."
        )
        if cfg.split == "dedode_train":
            if cfg.alpha is None:
                alpha = 0.75
            else:
                logger.warning(
                    f"Using alpha {cfg.alpha} for DeDoDe train which overrides default 0.75."
                )
                alpha = cfg.alpha
        elif cfg.split == "dedode_test":
            if cfg.alpha is None:
                alpha = 0.0
            else:
                logger.warning(
                    f"Using alpha {cfg.alpha} for DeDoDe test which overrides default 0.0."
                )
                alpha = cfg.alpha
        else:
            assert cfg.alpha is not None, "Alpha must be set for other splits."
            alpha = cfg.alpha
        if cfg.fraction < 1.0:
            self.scenes = self.scenes[:max(1, round(len(self.scenes) * cfg.fraction))]
        self.cum_weights = self.weight_scenes(alpha)

    def __len__(self):
        return self.cfg.weight

    def __getitem__(self, idx: int) -> Batch:
        scene_index = torch.searchsorted(self.cum_weights, (idx + 0.5) / len(self))
        scene = self.scenes[scene_index]
        pair_idx = np.random.choice(len(scene))
        return scene[pair_idx]

    def build_scenes(
        self,
        scene_split: str,
        min_overlap: float,
        max_overlap: float,
        max_num_pairs: int,
    ) -> list[MegaDepthScene]:
        if scene_split == "train":
            scene_names = set(self.all_scenes) - set(self.test_scenes)
        elif scene_split == "train_loftr":
            scene_names = set(self.all_scenes) - set(self.test_scenes_loftr)
        elif scene_split == "test":
            scene_names = self.test_scenes
        elif scene_split == "test_loftr":
            scene_names = self.test_scenes_loftr
        elif scene_split == "all_scenes":
            scene_names = self.all_scenes
        else:
            raise ValueError(f"Split {scene_split} not available")
        scene_names = sorted(list(scene_names))
        scenes = []

        for scene_name in tqdm(
            scene_names, desc=f"Loading {type(self).__name__} {scene_split} scenes..."
        ):
            if self.cfg.loftr_ignore and scene_name in self.loftr_ignore_scenes:
                continue
            if self.cfg.imc21_ignore and scene_name in self.imc21_scenes:
                continue
            if ".npy" not in scene_name:
                continue
            scene_info = np.load(
                self.scene_info_root / scene_name, allow_pickle=True
            ).item()

            scenes.append(
                MegaDepthScene(
                    self.data_root,
                    scene_info,
                    scene_name=scene_name,
                    min_overlap=min_overlap,
                    max_overlap=max_overlap,
                    max_num_pairs=max_num_pairs,
                    transform_cfg=self.transform_cfg,
                    use_sky_mask=self.cfg.use_sky_mask,
                )
            )
        return scenes

    def weight_scenes(self, alpha: float) -> torch.Tensor:
        assert alpha is not None
        ws = torch.tensor([len(scene) ** (1 - alpha) for scene in self.scenes])
        cum_ws = torch.cumsum(ws, dim=0)
        cum_ws = cum_ws / cum_ws[-1]
        return cum_ws
