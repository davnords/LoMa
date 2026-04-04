from dataclasses import dataclass
import torch
from torch import nn
import torchvision.models as models
from torch import Tensor

from loma.types import FineFeaturesType
from loma.device import device, amp_dtype
from loma.normalizers import imagenet


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class VGG(nn.Module):
    def forward(self, x):
        x = imagenet(x)
        with torch.autocast(device_type=device.type, enabled=True, dtype=amp_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x.permute(0, 2, 3, 1)
                    scale = scale * 2
                x = layer(x)
            return feats


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
            case "vgg19bn":
                return VGG19BN(cfg.patch_size)
            case _:
                raise ValueError(f"Unknown refiner features type: {cfg.type}")
