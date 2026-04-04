from __future__ import annotations
import torch.nn as nn
from dataclasses import dataclass, field
from pathlib import Path
import torch
from typing import Literal, Protocol


@dataclass
class Warp:
    warp: torch.Tensor
    covis: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    error: torch.Tensor | None = None


@dataclass
class Batch:
    img_A: torch.Tensor
    img_B: torch.Tensor
    depth_A: torch.Tensor
    depth_B: torch.Tensor
    img_A_path: Path
    img_B_path: Path
    source: GTSource | list[GTSource]
    flow_AB: torch.Tensor
    flow_BA: torch.Tensor
    mask_AB: torch.Tensor
    mask_BA: torch.Tensor
    quality: Quality | list[Quality]
    # Optional sparse pixel correspondences: (N, 4) = (x_A, y_A, x_B, y_B) in pixels.
    correspondences_AB: torch.Tensor = field(
        default_factory=lambda: torch.zeros((0, 4))
    )
    K_A: torch.Tensor = field(default_factory=lambda: torch.eye(3))
    K_B: torch.Tensor = field(default_factory=lambda: torch.eye(3))
    pose_A: torch.Tensor = field(default_factory=lambda: torch.eye(4))
    pose_B: torch.Tensor = field(default_factory=lambda: torch.eye(4))
    T_AB: torch.Tensor = field(default_factory=lambda: torch.eye(4))
    num_corresp: int | list[int] = 0

    def to(self, device: torch.device) -> Batch:
        # Preserve dynamic batch type (subclasses may add more tensor fields).
        data = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
            else:
                data[k] = v
        return type(self)(**data)

    def swap_AB(self) -> Batch:
        """Return a "reversed" batch (B->A) useful for symmetric GT computation."""
        return Batch(
            img_A=self.img_B,
            img_B=self.img_A,
            depth_A=self.depth_B,
            depth_B=self.depth_A,
            K_A=self.K_B,
            K_B=self.K_A,
            pose_A=self.pose_B,
            pose_B=self.pose_A,
            T_AB=torch.linalg.inv(self.T_AB),
            img_A_path=self.img_B_path,
            img_B_path=self.img_A_path,
            source=self.source,
            flow_AB=self.flow_BA,
            flow_BA=self.flow_AB,
            mask_AB=self.mask_BA,
            mask_BA=self.mask_AB,
            quality=self.quality,
            correspondences_AB=self.correspondences_AB[..., [2, 3, 0, 1]],
            num_corresp=self.num_corresp,
        )

    @classmethod
    def collate(cls, samples: list[Batch]) -> Batch:
        keys = samples[0].__dict__.keys()
        batch = {}
        for k in keys:
            if k == "correspondences_AB":
                corresps = 2 * torch.rand((len(samples), MAX_NUM_SPARSE_CORRESP, 4)) - 1
                for i, s in enumerate(samples):
                    corresps[i, : s.num_corresp] = s.correspondences_AB
                batch[k] = corresps
            elif isinstance(samples[0].__dict__[k], torch.Tensor):
                batch[k] = torch.stack([s.__dict__[k] for s in samples])
            else:
                batch[k] = [s.__dict__[k] for s in samples]
        return Batch(**batch)


class Model(nn.Module):
    @property
    def name(self) -> str:
        return self.__class__.__name__


Quality = Literal["high", "low"]
GTSource = Literal["depth", "flow", "sparse"]
DescriptorName = Literal["dinov2_vitl14"]
Normalizer = Literal["imagenet", "inception"]
FineFeaturesType = Literal["vgg19bn"]
MAX_NUM_SPARSE_CORRESP = 64


class Detector(Protocol):
    @property
    def topleft(self) -> float: ...

    def detect_from_path(
        self,
        im_path: str | Path,
        *,
        num_keypoints: int,
        return_dense_probs: bool = False,
    ) -> dict[str, torch.Tensor]: ...

    def detect(
        self, batch: Batch, *, num_keypoints: int, return_dense_probs: bool = False
    ) -> dict[str, torch.Tensor]: ...

    def to_pixel_coords(
        self, normalized_coords: torch.Tensor, h: int, w: int
    ) -> torch.Tensor: ...
