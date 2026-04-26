from dataclasses import dataclass
from typing import Literal
import logging

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from loma.detector.dad import ConvRefiner, VGG
from loma.detector.utils import sample_keypoints
from loma.types import Batch, Model
from loma.device import device

logger = logging.getLogger(__name__)

class TurboDetector(Model):
    """
    Joint detection + description network using the DaD (small VGG11) architecture.
    Trained by distilling detection probabilities from DaD and descriptors from the main LoMa descriptor.
    """
    @dataclass(frozen=True)
    class Cfg:
        compile: bool = True
        arch: Literal["dedode_s"] = "dedode_s"
        descriptor_dim: int = 256
        hidden_blocks: int = 3

    def __init__(self, cfg: Cfg | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = TurboDetector.Cfg()
        self.cfg = cfg
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if cfg.arch == "dedode_s":
            encoder, decoder = distdesc_S(
                descriptor_dim=cfg.descriptor_dim,
                hidden_blocks=cfg.hidden_blocks,
            )
        else:
            raise ValueError(f"Architecture {cfg.arch} not supported")
        self.encoder = encoder
        self.decoder = decoder
        if cfg.compile:
            logger.info("Compiling DistDesc...")
            self.compile()
        self.to(device)

    def forward_impl(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (scoremap, description_map) both at full resolution."""
        images = self.normalizer(images)
        features, sizes = self.encoder(images)
        det_logits = 0
        desc = 0
        context = None
        scales = ["8", "4", "2", "1"]
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_det, delta_desc, context = self.decoder(
                feature_map, context=context, scale=scale
            )
            det_logits = det_logits + delta_det.float()  # float32: needed for bicubic interpolate
            desc = desc + delta_desc  # stay in bfloat16
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                det_logits = F.interpolate(
                    det_logits, size=size, mode="bicubic", align_corners=False
                )
                desc = F.interpolate(
                    desc.float(), size=size, mode="bilinear", align_corners=False
                ).to(delta_desc.dtype)
                context = F.interpolate(
                    context.float(), size=size, mode="bilinear", align_corners=False
                )
        return det_logits.float(), desc

    @torch.inference_mode()
    def detect(self, batch: Batch | dict[str, torch.Tensor], num_keypoints: int) -> dict[str, torch.Tensor]:
        self.train(False)
        scoremap, _ = self.forward_impl(self._to_images(batch))
        B, K, H, W = scoremap.shape
        dense_probs = (
            scoremap.reshape(B, K * H * W)
            .softmax(dim=-1)
            .reshape(B, K, H * W)
            .sum(dim=1)
            .reshape(B, H, W)
        )
        keypoints, confidence = sample_keypoints(
            dense_probs,
            use_nms=True,
            nms_size=3,
            sample_topk=True,
            num_samples=num_keypoints,
            return_probs=True,
            subpixel=True,
            subpixel_temp=0.5,
            scoremap=scoremap.reshape(B, H, W),
        )
        return {"keypoints": keypoints, "keypoint_probs": confidence}

    @torch.inference_mode()
    def describe_keypoints(self, batch: Batch | dict[str, torch.Tensor], keypoints: torch.Tensor) -> dict[str, torch.Tensor]:
        self.train(False)
        _, desc_grid = self.forward_impl(self._to_images(batch))
        described = F.grid_sample(
            desc_grid.float(),
            keypoints[:, None],
            mode="bilinear",
            align_corners=False,
        )[:, :, 0].mT
        return {"descriptions": described}

    def to_pixel_coords(self, normalized_coords: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return torch.stack(
            (w * (normalized_coords[..., 0] + 1) / 2, h * (normalized_coords[..., 1] + 1) / 2),
            dim=-1,
        )

    def load_image(self, im_path: str | Path, resize: int = 1024) -> dict[str, torch.Tensor]:
        pil_im = Image.open(im_path).convert("RGB")
        W, H = pil_im.size
        scale = resize / max(W, H)
        W = int((scale * W) // 8 * 8)
        H = int((scale * H) // 8 * 8)
        pil_im = pil_im.resize((W, H))
        return {
            "image": torch.from_numpy(np.array(pil_im) / 255.0)
            .permute(2, 0, 1)
            .float()
            .to(device)[None]
        }

    @torch.inference_mode()
    def detect_and_describe(self, batch: Batch | dict[str, torch.Tensor], num_keypoints: int) -> dict[str, torch.Tensor]:
        """Joint detection and description in a single forward pass."""
        self.train(False)
        scoremap, desc_grid = self.forward_impl(self._to_images(batch))
        B, K, H, W = scoremap.shape
        dense_probs = (
            scoremap.reshape(B, K * H * W)
            .softmax(dim=-1)
            .reshape(B, K, H * W)
            .sum(dim=1)
            .reshape(B, H, W)
        )
        keypoints, confidence = sample_keypoints(
            dense_probs,
            use_nms=True,
            nms_size=3,
            sample_topk=True,
            num_samples=num_keypoints,
            return_probs=True,
            subpixel=True,
            subpixel_temp=0.5,
            scoremap=scoremap.reshape(B, H, W),
        )
        descriptions = F.grid_sample(
            desc_grid.float(),
            keypoints[:, None],
            mode="bilinear",
            align_corners=False,
        )[:, :, 0].mT
        return {"keypoints": keypoints, "keypoint_probs": confidence, "descriptions": descriptions}

    @torch.inference_mode()
    def detect_from_path(self, im_path: str | Path, *, num_keypoints: int) -> dict[str, torch.Tensor]:
        return self.detect(self.load_image(im_path), num_keypoints=num_keypoints)

    @torch.inference_mode()
    def detect_and_describe_from_path(self, im_path: str | Path, *, num_keypoints: int) -> dict[str, torch.Tensor]:
        return self.detect_and_describe(self.load_image(im_path), num_keypoints=num_keypoints)

    @torch.inference_mode()
    def describe_keypoints_from_path(self, im_path: str | Path, keypoints: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.describe_keypoints(self.load_image(im_path), keypoints)

    def _to_images(self, batch: Batch | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(batch, Batch):
            return torch.cat((batch.img_A, batch.img_B), dim=0)
        elif isinstance(batch, dict) and "im_A" in batch:
            return torch.cat((batch["im_A"], batch["im_B"]), dim=0)
        elif isinstance(batch, dict) and "image" in batch:
            return batch["image"]
        raise TypeError(f"Expected Batch or dict with keys ('im_A','im_B') or ('image',), got {type(batch)}")

    def forward(
        self, batch: Batch | dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        scoremap, descriptions = self.forward_impl(self._to_images(batch))
        return {"scoremap": scoremap, "descriptions": descriptions}


class JointDecoder(nn.Module):
    """
    Decoder that splits ConvRefiner output into (detection logits, descriptors, context).
    """

    def __init__(
        self,
        layers: nn.ModuleDict,
        num_prototypes: int = 1,
        descriptor_dim: int = 256,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.num_prototypes = num_prototypes
        self.descriptor_dim = descriptor_dim

    def forward(
        self, features: torch.Tensor, context: torch.Tensor | None, scale: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context is not None:
            features = torch.cat((features, context), dim=1)
        out = self.layers[scale](features)
        det = out[:, : self.num_prototypes]
        desc = out[:, self.num_prototypes : self.num_prototypes + self.descriptor_dim]
        context = out[:, self.num_prototypes + self.descriptor_dim :]
        return det, desc, context


def distdesc_S(descriptor_dim: int = 256, hidden_blocks: int = 3):
    """
    Small joint detector+descriptor network matching DaD's VGG11 encoder.
    Each scale outputs: 1 (det) + descriptor_dim (desc) + context_dim.
    """
    amp = True
    amp_dtype = torch.bfloat16
    residual = True
    NUM_PROTOTYPES = 1
    # context dims mirror DaD: 256, 128, 32, 1
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512, 512,
                NUM_PROTOTYPES + descriptor_dim + 256,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256, 256,
                NUM_PROTOTYPES + descriptor_dim + 128,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128, 64,
                NUM_PROTOTYPES + descriptor_dim + 32,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 32, 32,
                NUM_PROTOTYPES + descriptor_dim + 1,
                hidden_blocks=hidden_blocks,
                residual=residual,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    encoder = VGG(size="11", amp=amp, amp_dtype=amp_dtype)
    decoder = JointDecoder(
        conv_refiner,
        num_prototypes=NUM_PROTOTYPES,
        descriptor_dim=descriptor_dim,
    )
    return encoder, decoder
