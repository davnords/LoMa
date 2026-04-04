import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image

from loma.device import device, amp_dtype
from loma.types import Model

logger = logging.getLogger(__name__)


class DeDoDeDescriptor(Model):
    @dataclass(frozen=True)
    class Cfg:
        arch: Literal["dedode_b", "dedode_g"] = "dedode_b"
        compile: bool = True
        descriptor_dim: int = 256
        hidden_blocks: int = 5

    def __init__(self, cfg: Cfg) -> None:
        super().__init__()

        if cfg.arch == "dedode_b":
            encoder, decoder = dedode_descriptor_B(
                descriptor_dim=cfg.descriptor_dim, hidden_blocks=cfg.hidden_blocks
            )
        elif cfg.arch == "dedode_g":
            encoder, decoder = dedode_descriptor_G(
                descriptor_dim=cfg.descriptor_dim, hidden_blocks=cfg.hidden_blocks
            )
        else:
            raise ValueError(f"Architecture {cfg.arch} not supported")
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder

        self.to(device)
        if cfg.compile:
            logger.info("Compiling DeDoDeDescriptor...")
            self.compile()
        logger.info(f"{self.name} initialized.")

    @staticmethod
    def load_pretrained(arch: Literal["dedode_b", "dedode_g"]):
        cfg = DeDoDeDescriptor.Cfg(arch=arch)
        if arch == "dedode_b":
            weights = torch.hub.load_state_dict_from_url(
                "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth"
            )
        elif arch == "dedode_g":
            weights = torch.hub.load_state_dict_from_url(
                "https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth"
            )
        else:
            raise ValueError(f"Architecture {arch} not supported")
        model = DeDoDeDescriptor(cfg)
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        assert len(unexpected_keys) == 0, (
            f"Unexpected keys when loading pretrained weights: {unexpected_keys}"
        )
        return model

    def forward(
        self,
        images: torch.Tensor,
    ):
        features, sizes = self.encoder(images)
        descriptions = 0
        context = None
        scales = self.decoder.scales
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_description, context = self.decoder(
                feature_map, scale=scale, context=context
            )
            descriptions = descriptions + delta_description
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                descriptions = F.interpolate(
                    descriptions, size=size, mode="bilinear", align_corners=False
                )
                context = F.interpolate(
                    context, size=size, mode="bilinear", align_corners=False
                )
        return descriptions

    @torch.inference_mode()
    def describe_keypoints(
        self,
        images: torch.Tensor,
        keypoints: torch.Tensor,
    ):
        self.train(False)
        images = images.to(device)
        keypoints = keypoints.to(device)
        # TODO is this compile error //je ?
        description_grid = self(images)
        described_keypoints = F.grid_sample(
            description_grid.float(),
            keypoints[:, None],
            mode="bilinear",
            align_corners=False,
        )[:, :, 0].mT
        return {"descriptions": described_keypoints}

    def read_image(self, im_path, H=784, W=784):
        return (
            torch.from_numpy(
                np.array(Image.open(im_path).convert("RGB").resize((W, H))) / 255.0
            )
            .permute(2, 0, 1)
            .float()
            .to(device)[None]
        )

    def describe_keypoints_from_path(self, im_path, keypoints, H=784, W=784):
        images = self.read_image(im_path, H=H, W=W)
        return self.describe_keypoints(images, keypoints)


class Decoder(nn.Module):
    def __init__(
        self, layers, *args, super_resolution=False, descriptor_dim=1, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.scales = sorted(list(layers.keys()), key=lambda x: int(x), reverse=True)
        self.super_resolution = super_resolution
        self.descriptor_dim = descriptor_dim

    def forward(self, features, context=None, scale=None):
        if context is not None:
            features = torch.cat((features, context), dim=1)
        stuff = self.layers[scale](features)
        logits, context = (
            stuff[:, : self.descriptor_dim],
            stuff[:, self.descriptor_dim :],
        )
        return logits, context


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        hidden_blocks: int,
        amp: bool,
        amp_dtype: torch.dtype,
        dw: bool = True,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim,
            hidden_dim,
            dw=False,
            kernel_size=1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.amp = amp
        self.amp_dtype = amp_dtype

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=True,
        kernel_size=5,
        bias=True,
        norm_type=nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert out_dim % in_dim == 0, (
                "outdim must be divisible by indim for depthwise"
            )
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = (
            norm_type(out_dim)
            if norm_type is nn.BatchNorm2d
            else norm_type(num_channels=out_dim)
        )
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, feats):
        b, c, hs, ws = feats.shape
        with torch.autocast(
            device_type=feats.device.type, enabled=self.amp, dtype=self.amp_dtype
        ):
            x0 = self.block1(feats)
            x = self.hidden_blocks(x0)
            x = (x + x0) / 1.4
            x = self.out_conv(x)
            return x


class VGG(nn.Module):
    def __init__(self, size="19", amp=False, amp_dtype=amp_dtype) -> None:
        super().__init__()
        if size == "11":
            self.layers = nn.ModuleList(tvm.vgg11_bn().features[:22])
        elif size == "13":
            self.layers = nn.ModuleList(tvm.vgg13_bn().features[:28])
        elif size == "19":
            self.layers = nn.ModuleList(tvm.vgg19_bn().features[:40])
        # Maxpool layers: 6, 13, 26, 39
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        with torch.autocast(
            device_type=x.device.type, enabled=self.amp, dtype=self.amp_dtype
        ):
            feats = []
            sizes = []
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats.append(x)
                    sizes.append(x.shape[-2:])
                x = layer(x)
            return feats, sizes


class FrozenDINOv2(nn.Module):
    def __init__(self, amp=True, amp_dtype=amp_dtype, dinov2_weights=None):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
                map_location="cpu",
            )
        from .transformer import vit_large

        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
        )
        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
        for param in dinov2_vitl14.parameters():
            param.requires_grad = False
        self.dinov2_vitl14 = dinov2_vitl14

    @torch.compiler.disable
    def forward(self, x):
        B, C, H, W = x.shape
        with torch.inference_mode():
            dinov2_features_16 = self.dinov2_vitl14.forward_features(
                x.to(self.amp_dtype)
            )
            features_16 = (
                dinov2_features_16["x_norm_patchtokens"]
                .permute(0, 2, 1)
                .reshape(B, 1024, H // 14, W // 14)
            )
        return [features_16.clone()], [
            (H // 14, W // 14)
        ]  # clone from inference mode to use in autograd


class VGG_DINOv2(nn.Module):
    def __init__(self, vgg_kwargs=None, dinov2_kwargs=None):
        assert vgg_kwargs is not None and dinov2_kwargs is not None, "Input kwargs pls"
        super().__init__()
        self.vgg = VGG(**vgg_kwargs)
        self.frozen_dinov2 = FrozenDINOv2(**dinov2_kwargs)

    def forward(self, x):
        feats_vgg, sizes_vgg = self.vgg(x)
        feat_dinov2, size_dinov2 = self.frozen_dinov2(x)
        return feats_vgg + feat_dinov2, sizes_vgg + size_dinov2


def dedode_descriptor_B(descriptor_dim: int, hidden_blocks: int = 5):
    amp = True
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                512,
                512,
                256 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    encoder = VGG(size="19", amp=amp, amp_dtype=amp_dtype)
    decoder = Decoder(conv_refiner, descriptor_dim=descriptor_dim)
    return encoder, decoder


def dedode_descriptor_G(
    descriptor_dim: int, dinov2_weights: str | None = None, hidden_blocks: int = 5
):
    amp = True

    conv_refiner = nn.ModuleDict(
        {
            "14": ConvRefiner(
                1024,
                768,
                512 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "8": ConvRefiner(
                512 + 512,
                512,
                256 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "4": ConvRefiner(
                256 + 256,
                256,
                128 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "2": ConvRefiner(
                128 + 128,
                64,
                32 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
            "1": ConvRefiner(
                64 + 32,
                32,
                1 + descriptor_dim,
                hidden_blocks=hidden_blocks,
                amp=amp,
                amp_dtype=amp_dtype,
            ),
        }
    )
    vgg_kwargs = dict(size="19", amp=amp, amp_dtype=amp_dtype)
    dinov2_kwargs = dict(amp=amp, amp_dtype=amp_dtype, dinov2_weights=dinov2_weights)
    encoder = VGG_DINOv2(vgg_kwargs=vgg_kwargs, dinov2_kwargs=dinov2_kwargs)
    decoder = Decoder(conv_refiner, descriptor_dim=descriptor_dim)
    return encoder, decoder
