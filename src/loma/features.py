from dataclasses import dataclass, field
import math
from typing import Any, Callable
import torch
from einops import rearrange
from torch import nn
import torchvision.models as models
from torch.nn import functional as F
from torch import Tensor

from loma.types import Normalizer, FineFeaturesType
from loma.device import device
from loma.normalizers import imagenet

def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def wrap_with_normalize(
    forward: Callable[[torch.Tensor], list[torch.Tensor]],
    *,
    normalizer: Normalizer,
    patch_size: int,
    enable_amp: bool,
    frozen: bool,
    normalize_feats: bool,
):
    def wrapped_forward(self, img: torch.Tensor) -> list[torch.Tensor]:
        with (
            torch.autocast(device.type, torch.bfloat16, enabled=enable_amp),
            torch.set_grad_enabled(not frozen),
        ):
            if self.training and frozen:
                self.eval()
            B, C, H, W = img.shape
            assert C == 3, f"Image must have 3 channels, but got shape {img.shape=}"
            img_n = normalizer(img)
            H = H // patch_size
            W = W // patch_size
            raw_outs = forward(img_n)
            maybe_feat_normalizer = (
                F.normalize if normalize_feats else lambda x, dim=-1: x
            )
            return [
                maybe_feat_normalizer(
                    rearrange(x, "B (H W) D -> B H W D", H=H, W=W), dim=-1
                )
                for x in raw_outs
            ]

    return wrapped_forward


def wrap_model(
    model: nn.Module,
    *,
    normalizer: Normalizer,
    patch_size: int,
    enable_amp: bool,
    frozen: bool,
    normalize_feats: bool,
    func: Any,
):
    if enable_amp and frozen:  # if training we want params in fp32
        model = model.to(torch.bfloat16)
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.frozen = frozen
    type(model).forward = wrap_with_normalize(
        func,
        normalizer=normalizer,
        patch_size=patch_size,
        enable_amp=enable_amp,
        frozen=frozen,
        normalize_feats=normalize_feats,
    )
    return model

class VGG(nn.Module):
    def forward(self, x):
        x = imagenet(x)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x.permute(0, 2, 3, 1)
                    scale = scale * 2
                x = layer(x)
            return feats


class VGG19(VGG):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        if patch_size not in [8]:
            raise NotImplementedError(
                f"VGG19 is not supported for patch size {patch_size}"
            )
        last_layer = {8: 28}[patch_size]
        self.layers = nn.ModuleList(
            models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[
                :last_layer
            ]
        )


class VGG19BN(VGG):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        last_layer = {1: 7, 2: 14, 4: 27, 8: 40, 16: 52}[patch_size]
        self.layers = nn.ModuleList(
            models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1).features[
                :last_layer
            ]
        )

class FineFeatures(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        type: FineFeaturesType = "vgg19bn"
        patch_size: int = 4

    def __new__(cls, cfg: Cfg):
        match cfg.type:
            case "vgg19":
                return VGG19(cfg.patch_size)
            case "vgg19bn":
                return VGG19BN(cfg.patch_size)
            case "flux2":
                raise NotImplementedError("Flux2 is not supported for fine features")
                return Flux2(cfg.patch_size)
            case _:
                raise ValueError(f"Unknown refiner features type: {cfg.type}")
