import logging
from dataclasses import dataclass
import math
from typing import Callable, List, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..device import device
from ..types import Batch, Model
from ..detector.dad import DaD

logger = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = True

def _load_descriptor(descriptor: Literal["dedode_b", "dedode_g"], run_path: str | None):
    """Load descriptor model (imported here to avoid circular imports)."""
    from ..descriptor.dedode import DeDoDeDescriptor
    if run_path is not None:
        return DeDoDeDescriptor.from_run(run_path)
    else:
        return DeDoDeDescriptor.load_pretrained(descriptor)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, *, gamma: float, posenc_dist: Literal["normal", "reciprocal"]) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        assert posenc_dist == "normal"
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)

class FixedPosEnc(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, *, gamma: float, posenc_dist: Literal["normal", "reciprocal"]) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        if posenc_dist == "normal":
            freqs = torch.randn(F_dim // 2, M).to(device) * self.gamma**-2
        elif posenc_dist == "reciprocal":
            min_freq = 1e-0 # about ~1 Hz
            assert 1/gamma > min_freq
            max_freq = 1/gamma 
            freqs = (torch.rand(F_dim // 2, M).to(device) * (math.log(max_freq)-math.log(min_freq)) + math.log(min_freq)).exp()
        else:
            raise ValueError(f"Positional encoding distribution {posenc_dist} not supported")
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
        if encoding is not None:
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

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> List[torch.Tensor]:
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

class AlternatingTransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.frame_block = SelfBlock(*args, **kwargs)
        self.global_block = SelfBlock(*args, **kwargs)

    def forward(self, desc0, desc1, encoding0, encoding1):
        desc0 = self.frame_block(desc0, encoding0)
        desc1 = self.frame_block(desc1, encoding1)

        desc = torch.cat([desc0, desc1], dim=1)

        if self.training:
            desc = checkpoint(self.global_block, desc, None, use_reentrant=False)
        else:
            desc = self.global_block(desc, None)
        desc0, desc1 = torch.chunk(desc, 2, dim=1)
        return desc0, desc1

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
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = log_double_softmax(sim, z0, z1)
        return scores, sim


def filter_matches(scores: torch.Tensor, th: float):
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class LoMa(Model):
    @dataclass(frozen=True)
    class Cfg:
        input_dim: int = 256
        descriptor_dim: int = 256
        n_layers: int = 9
        num_heads: int = 4
        filter_threshold: float = 0.1
        mp: bool = True
        compile: bool = True
        normalize_descriptions: bool = False
        # Descriptor config (frozen, used for detect/describe)
        descriptor: Literal["dedode_b", "dedode_g"] = "dedode_g"
        descriptor_run_path: str | None = None
        num_keypoints: int = 2048
        # Positional encoding config
        posenc_type: Literal["learnable", "fixed", "none"] = "learnable"
        # basically the wavelength of the positional encoding
        posenc_gamma: float = 1.0
        # posenc
        posenc_dist: Literal["normal", "reciprocal"] = "normal"

    def __init__(self, cfg: Cfg | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = LoMa.Cfg()
        self.cfg = cfg

        if cfg.input_dim != cfg.descriptor_dim:
            self.input_proj = nn.Linear(cfg.input_dim, cfg.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = cfg.descriptor_dim // cfg.num_heads
        if cfg.posenc_type == "learnable":
            self.posenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim, gamma=cfg.posenc_gamma, posenc_dist=cfg.posenc_dist)
        elif cfg.posenc_type == "fixed":
            self.posenc = FixedPosEnc(2, head_dim, head_dim, gamma=cfg.posenc_gamma, posenc_dist=cfg.posenc_dist)
        elif cfg.posenc_type == "none":
            self.posenc = None
        else:
            raise ValueError(f"Positional encoding type {cfg.posenc_type} not supported")

        self.transformers = nn.ModuleList([TransformerLayer(cfg.descriptor_dim, cfg.num_heads) for _ in range(cfg.n_layers)])
        self.log_assignment = nn.ModuleList(
            [MatchAssignment(cfg.descriptor_dim) for _ in range(cfg.n_layers)]
        )

        self._detector = DaD()
        self._detector.eval()
        for p in self._detector.parameters():
            p.requires_grad = False

        self._descriptor = _load_descriptor(cfg.descriptor, cfg.descriptor_run_path)
        self._descriptor.eval()
        for p in self._descriptor.parameters():
            p.requires_grad = False

        self.to(device)
        if cfg.compile:
            logger.info(f"Compiling {self.name}...")
            self.compile()
        logger.info(f"{self.name} initialized.")


        self.num_layers_inference = cfg.n_layers

    @torch.inference_mode()
    def detect(self, batch: Batch, num_keypoints: int | None = None) -> dict:
        """Detect keypoints using the frozen detector."""
        if num_keypoints is None:
            num_keypoints = self.cfg.num_keypoints
        return self._detector.detect(batch, num_keypoints=num_keypoints)

    @torch.inference_mode()
    def describe(self, batch: Batch, keypoints: torch.Tensor) -> dict:
        """Describe keypoints using the frozen descriptor."""
        return self._descriptor.describe_keypoints(batch, keypoints)

    def forward(
        self,
        batch: Batch | dict[str, torch.Tensor],
        keypoints_A: torch.Tensor | None = None,
        keypoints_B: torch.Tensor | None = None,
        descriptors_A: torch.Tensor | None = None,
        descriptors_B: torch.Tensor | None = None,
    ) -> dict:
        with torch.autocast(
            enabled=self.cfg.mp, dtype=torch.bfloat16, device_type="cuda"
        ):
            if isinstance(batch, Batch):
                assert keypoints_A is not None and keypoints_B is not None
                assert descriptors_A is not None and descriptors_B is not None
                return self._forward(
                    keypoints_A, keypoints_B, descriptors_A, descriptors_B
                )
            else:
                data0, data1 = batch["image0"], batch["image1"]
                return self._forward(
                    data0["keypoints"],
                    data1["keypoints"],
                    data0["descriptors"],
                    data1["descriptors"],
                )

    def _forward(
        self,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
    ) -> dict:
        desc0 = desc0.detach().contiguous()
        desc1 = desc1.detach().contiguous()
        if self.cfg.normalize_descriptions:
            desc0 = F.normalize(desc0, dim=-1)
            desc1 = F.normalize(desc1, dim=-1)

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)

        if self.posenc is not None:
            encoding0 = self.posenc(kpts0)
            encoding1 = self.posenc(kpts1)
        else:
            encoding0 = encoding1 = None

        all_scores = []
        # for i in range(self.cfg.n_layers):
        for i in range(self.num_layers_inference):
            desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)
            scores, _ = self.log_assignment[i](desc0, desc1)
            all_scores.append(scores)

        return {
            "scores": all_scores[-1],
            "all_scores": all_scores,
        }

    @torch.inference_mode()
    def match(
        self,
        batch: Batch | dict[str, torch.Tensor],
        keypoints_A: torch.Tensor | None = None,
        keypoints_B: torch.Tensor | None = None,
        descriptors_A: torch.Tensor | None = None,
        descriptors_B: torch.Tensor | None = None,
        filter_threshold: float | None = None,
    ) -> dict:
        """Run matching with filtering (for inference/evaluation)."""
        if filter_threshold is None:
            filter_threshold = self.cfg.filter_threshold

        # Get keypoints for batch size
        if isinstance(batch, Batch):
            assert keypoints_A is not None
            kpts0 = keypoints_A
        else:
            kpts0 = batch["image0"]["keypoints"]

        # Run forward pass
        output = self(
            batch,
            keypoints_A=keypoints_A,
            keypoints_B=keypoints_B,
            descriptors_A=descriptors_A,
            descriptors_B=descriptors_B,
        )
        scores = output["scores"]

        # Apply filtering
        b = kpts0.shape[0]
        m0, m1, mscores0, mscores1 = filter_matches(scores, filter_threshold)
        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])

        return {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "matches": matches,
            "match_scores": mscores,
            "scores": scores,
            "all_scores": output["all_scores"],
        }