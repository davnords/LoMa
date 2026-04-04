from dataclasses import dataclass
import re
from typing import Annotated, Callable, Literal, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import tyro
from torch import nn

from loma.types import Model, Batch
from loma.device import device, amp_dtype
from loma.descriptor.dedode import DeDoDeDescriptor
from loma.detector.dad import DaD


# Reference code from LightGlue: https://github.com/cvg/LightGlue/blob/main/lightglue/lightglue.py
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


def to_pixel_coords(flow, h1, w1):
    flow = torch.stack(
        (
            w1 * (flow[..., 0] + 1) / 2,
            h1 * (flow[..., 1] + 1) / 2,
        ),
        dim=-1,
    )
    return flow


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(
        self,
        M: int,
        dim: int,
        F_dim: int | None = None,
        *,
        gamma: float,
    ) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class FixedPosEnc(nn.Module):
    def __init__(
        self,
        M: int,
        dim: int,
        F_dim: int | None = None,
        *,
        gamma: float,
    ) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        freqs = torch.randn(F_dim // 2, M).to(device) * self.gamma**-2
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        with torch.no_grad():
            self.Wr.weight.data = freqs
        self.Wr.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        # if encoding is not None:
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = F.scaled_dot_product_attention(q, k, v)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        m0 = F.scaled_dot_product_attention(qk0, qk1, v1)
        m1 = F.scaled_dot_product_attention(qk1, qk0, v0)
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(self, desc0, desc1, encoding0, encoding1):
        desc0 = self.self_attn(desc0, encoding0)
        desc1 = self.self_attn(desc1, encoding1)
        return self.cross_attn(desc0, desc1)


def log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    b, m, n = sim.shape
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1
    scores[:, :-1, -1] = z0.squeeze(-1)
    scores[:, -1, :-1] = z1.squeeze(-1)
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        # For training we supervise matchability but for inference we don't use it
        # NOTE: This operation, double softmax, takes a lot of current inference time.
        if self.training:
            z0 = self.matchability(desc0)
            z1 = self.matchability(desc1)
            scores = log_double_softmax(sim, z0, z1)
        else:
            scores = F.softmax(sim, dim=2) * F.softmax(sim, dim=1)
        return scores, sim


def filter_matches(scores: torch.Tensor, th: float):
    max0, max1 = scores.max(2), scores.max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    mscores0 = torch.where(mutual0, max0.values, max0.values.new_tensor(0))
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), mscores0.new_tensor(0))
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class LoMa(Model):
    @dataclass(frozen=True, kw_only=True)
    class Cfg:
        input_dim: int = 256
        embed_dim: int = 256
        n_layers: int = 9
        num_heads: int = 4
        filter_threshold: float = 0.1
        mp: bool = True
        compile: bool = False
        normalize_descriptions: bool = False
        descriptor: Literal["dedode_b", "dedode_g"] = "dedode_b"
        num_keypoints: int = 2048
        # Positional encoding config
        posenc_type: Literal["learnable", "fixed"] = "learnable"
        # basically the wavelength of the positional encoding
        posenc_gamma: float = 1.0
        # Optional pretrained checkpoint URL.
        weights_url: str | None = None

    def __init__(self, cfg: Cfg | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = LoMa.Cfg()
        self.cfg = cfg

        if cfg.input_dim != cfg.embed_dim:
            self.input_proj = nn.Linear(cfg.input_dim, cfg.embed_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = cfg.embed_dim // cfg.num_heads
        if cfg.posenc_type == "learnable":
            self.posenc = LearnableFourierPositionalEncoding(
                2,
                head_dim,
                head_dim,
                gamma=cfg.posenc_gamma,
            )
        elif cfg.posenc_type == "fixed":
            self.posenc = FixedPosEnc(
                2,
                head_dim,
                head_dim,
                gamma=cfg.posenc_gamma,
            )
        else:
            raise ValueError(
                f"Positional encoding type {cfg.posenc_type} not supported"
            )

        self.transformers = nn.ModuleList(
            [
                TransformerLayer(cfg.embed_dim, cfg.num_heads)
                for _ in range(cfg.n_layers)
            ]
        )
        self.log_assignment = nn.ModuleList(
            [MatchAssignment(cfg.embed_dim) for _ in range(cfg.n_layers)]
        )

        self._detector = DaD().eval()
        for p in self._detector.parameters():
            p.requires_grad = False

        self._descriptor = DeDoDeDescriptor(
            DeDoDeDescriptor.Cfg(arch=cfg.descriptor, descriptor_dim=cfg.input_dim)
        ).eval()
        for p in self._descriptor.parameters():
            p.requires_grad = False

        self.to(device)

        if cfg.weights_url is not None:
            weights = torch.hub.load_state_dict_from_url(
                cfg.weights_url,
                map_location=device,
            )
            missing_keys, unexpected_keys = self.load_state_dict(weights, strict=False)
            if len(unexpected_keys) > 0:
                allowed_extra_layer_keys = {
                    key
                    for key in unexpected_keys
                    if self._is_unexpected_extra_layer_key(key, cfg.n_layers)
                }
                disallowed_keys = sorted(
                    set(unexpected_keys) - allowed_extra_layer_keys
                )
                assert len(disallowed_keys) == 0, (
                    "Unexpected keys when loading pretrained weights "
                    f"(not extra layers beyond n_layers={cfg.n_layers}): {disallowed_keys}"
                )
            assert len(missing_keys) == 0, (
                f"Missing keys when loading pretrained weights: {missing_keys}"
            )

        self.eval()
        if cfg.compile:
            # self.compile()
            self._detector.compile()
            self._descriptor.compile()
            pass

    @staticmethod
    def _is_unexpected_extra_layer_key(key: str, n_layers: int) -> bool:
        match = re.match(r"^(transformers|log_assignment)\.(\d+)\.", key)
        if match is None:
            return False
        layer_idx = int(match.group(2))
        return layer_idx >= n_layers

    @torch.inference_mode()
    def detect(self, batch: Batch, num_keypoints: int | None = None) -> dict:
        """Detect keypoints using the frozen detector."""
        if num_keypoints is None:
            num_keypoints = self.cfg.num_keypoints
        return self._detector.detect(batch, num_keypoints=num_keypoints)

    @torch.inference_mode()
    def describe(self, batch: Batch, keypoints: torch.Tensor) -> dict:
        """Describe keypoints using the frozen descriptor."""
        images = torch.cat([batch.img_A, batch.img_B], dim=0)
        return self._descriptor.describe_keypoints(images, keypoints)

    def forward(
        self,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
    ) -> dict:

        with torch.autocast(
            enabled=self.cfg.mp, dtype=amp_dtype, device_type=device.type
        ):
            kpts0 = kpts0.to(device)
            kpts1 = kpts1.to(device)
            desc0 = desc0.to(device).detach().contiguous()
            desc1 = desc1.to(device).detach().contiguous()
            desc0 = self.input_proj(desc0)
            desc1 = self.input_proj(desc1)
            encoding0 = self.posenc(kpts0)
            encoding1 = self.posenc(kpts1)
            scores = None
            for i in range(self.cfg.n_layers):
                desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)
            scores, _ = self.log_assignment[i](desc0, desc1)
            assert scores is not None

        return {
            "scores": scores,
            # "all_scores": all_scores,
        }

    @torch.inference_mode()
    def detect_and_describe(
        self, image: str | torch.Tensor, num_keypoints: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Returns (keypoints, descriptions, H, W) where H, W are the original image dimensions."""
        if num_keypoints is None:
            num_keypoints = self.cfg.num_keypoints
        if isinstance(image, str):
            keypoints = self._detector.detect_from_path(
                image, num_keypoints=num_keypoints
            )["keypoints"]
            descriptions = self._descriptor.describe_keypoints_from_path(
                image, keypoints
            )["descriptions"]
            w, h = Image.open(image).size
        else:
            image = image.to(device)
            batch = {"image": image}
            keypoints = self._detector.detect(batch, num_keypoints=num_keypoints)[
                "keypoints"
            ]
            descriptions = self._descriptor.describe_keypoints(
                batch["image"], keypoints
            )["descriptions"]
            h, w = image.shape[-2:]
        return keypoints, descriptions, h, w

    @torch.inference_mode()
    def match(
        self,
        image_A: str | torch.Tensor,
        image_B: str | torch.Tensor,
        filter_threshold: float | None = None,
        num_keypoints: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Match keypoints between two images, returning pixel-coordinate matches.

        Args:
            image_A: Path to image A (str) or image tensor (B, C, H, W).
            image_B: Path to image B (str) or image tensor (B, C, H, W).
        Returns:
            Tuple of (kptsA, kptsB) as numpy arrays of pixel coordinates.
        """
        keypoints_A, descriptors_A, h1, w1 = self.detect_and_describe(
            image_A, num_keypoints
        )
        keypoints_B, descriptors_B, h2, w2 = self.detect_and_describe(
            image_B, num_keypoints
        )
        # return None
        if filter_threshold is None:
            filter_threshold = self.cfg.filter_threshold

        scores = self(keypoints_A, keypoints_B, descriptors_A, descriptors_B)["scores"]
        m0, _, _, _ = filter_matches(scores, filter_threshold)

        valid = m0[0] > -1
        matched_A = keypoints_A[0][torch.where(valid)[0]]
        matched_B = keypoints_B[0][m0[0][valid]]

        return to_pixel_coords(matched_A, h1, w1).cpu().numpy(), to_pixel_coords(
            matched_B, h2, w2
        ).cpu().numpy()


@dataclass(frozen=True, kw_only=True)
class LoMaB128(LoMa.Cfg):
    name: Literal["loma_B128"] = "loma_B128"
    input_dim: Literal[128] = 128
    embed_dim: Literal[256] = 256
    num_heads: Literal[4] = 4
    descriptor: Literal["dedode_b"] = "dedode_b"
    weights_url: str = (
        "https://github.com/davnords/storage/releases/download/loma/loma_B128.pth"
    )


@dataclass(frozen=True, kw_only=True)
class LoMaB(LoMa.Cfg):
    name: Literal["loma_B"] = "loma_B"
    input_dim: Literal[256] = 256
    embed_dim: Literal[256] = 256
    num_heads: Literal[4] = 4
    descriptor: Literal["dedode_g"] = "dedode_g"
    weights_url: str = (
        "https://github.com/davnords/storage/releases/download/loma/loma_B.pt"
    )


@dataclass(frozen=True, kw_only=True)
class LoMaL(LoMa.Cfg):
    name: Literal["loma_L"] = "loma_L"
    input_dim: Literal[256] = 256
    embed_dim: Literal[512] = 512
    num_heads: Literal[8] = 8
    descriptor: Literal["dedode_g"] = "dedode_g"
    weights_url: str = (
        "https://github.com/davnords/storage/releases/download/loma/loma_L.pth"
    )


@dataclass(frozen=True, kw_only=True)
class LoMaG(LoMa.Cfg):
    name: Literal["loma_G"] = "loma_G"
    input_dim: Literal[256] = 256
    embed_dim: Literal[1024] = 1024
    num_heads: Literal[16] = 16
    descriptor: Literal["dedode_g"] = "dedode_g"
    weights_url: str = (
        "https://github.com/davnords/storage/releases/download/loma/loma_G.pth"
    )


LoMaName = Literal["loma_B128", "loma_B", "loma_L", "loma_G"]
# Accept either a raw LoMa.Cfg instance or a named preset.
LoMaConfig = (
    Annotated[LoMaB128, tyro.conf.subcommand("loma_b128")]
    | Annotated[LoMaB, tyro.conf.subcommand("loma_b")]
    | Annotated[LoMaL, tyro.conf.subcommand("loma_l")]
    | Annotated[LoMaG, tyro.conf.subcommand("loma_g")]
    | Annotated[LoMa.Cfg, tyro.conf.subcommand("loma_custom")]
)
