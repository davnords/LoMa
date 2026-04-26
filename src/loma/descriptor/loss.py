from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from typing import Literal

from ..types import Batch
from ..detector import DaD
from ..geometry import compute_sparse_mnn_matches
from .dedode import DeDoDeDescriptor


class DescriptorLoss(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        num_keypoints: int = 2048
        normalize_descriptions: bool = True
        inv_temp: float = 20
        detector: Literal["dad"] = "dad"
        depth_error_threshold: float = 0.05
        flow_error_threshold: float = 5e-3
        error_threshold: float = 0.01
        local_neighbourhood_size: int = 1
        matchability_loss: bool = False

    def __init__(self, cfg: Cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.detector = DaD()

    def forward(self, batch: Batch, model: DeDoDeDescriptor, step: int):
        kpts_A, kpts_B = (
            self.detector.detect(batch, num_keypoints=self.cfg.num_keypoints)["keypoints"]
            .clone()
            .chunk(2)
        )
        
        description_grid = model(batch)
        desc_grid_A, desc_grid_B = description_grid.chunk(2)
        desc_A = F.grid_sample(
            desc_grid_A.float(), kpts_A[:, None], mode="bilinear", align_corners=False
        )[:, :, 0].mT
        desc_B = F.grid_sample(
            desc_grid_B.float(), kpts_B[:, None], mode="bilinear", align_corners=False
        )[:, :, 0].mT

        mnn = compute_sparse_mnn_matches(kpts_A, kpts_B, batch, depth_error_threshold=self.cfg.depth_error_threshold, flow_error_threshold=self.cfg.flow_error_threshold, local_neighbourhood_size=self.cfg.local_neighbourhood_size, error_threshold=self.cfg.error_threshold)
        logP_A_B = dual_log_softmax_matcher(
            desc_A,
            desc_B,
            normalize=self.cfg.normalize_descriptions,
            inv_temperature=self.cfg.inv_temp,
        )

        neg_log_likelihood = compute_descriptor_matching_loss(logP_A_B, mnn)
        if self.cfg.matchability_loss:
            matchability_loss = compute_descriptor_matchability_loss(model, desc_A, desc_B, mnn)
            neg_log_likelihood += matchability_loss

        num_matches = mnn.shape[0] if mnn.numel() > 0 else 0
        wandb.log(
            {"neg_log_likelihood": neg_log_likelihood.item(), "num_matches": num_matches}, step=step
        )
        return neg_log_likelihood, None


def dual_log_softmax_matcher(
    desc_A: torch.Tensor,
    desc_B: torch.Tensor,
    *,
    inv_temperature: float,
    normalize: bool,
    eps: float = 1e-8,
):
    B, N, C = desc_A.shape
    if normalize:
        desc_A = desc_A / (desc_A.norm(dim=-1, keepdim=True) + eps)
        desc_B = desc_B / (desc_B.norm(dim=-1, keepdim=True) + eps)
    corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    logP = corr.log_softmax(dim=-2) + corr.log_softmax(dim=-1)
    return logP


def compute_descriptor_matching_loss(logP: torch.Tensor, mnn: torch.Tensor):
    if mnn.numel() == 0:
        return (0 * logP).mean()
    return -logP[mnn[:, 0], mnn[:, 1], mnn[:, 2]].mean()

def compute_descriptor_matchability_loss(model: DeDoDeDescriptor, desc_A: torch.Tensor, desc_B: torch.Tensor, mnn: torch.Tensor):
    z0 = model.module.matchability(desc_A).squeeze(-1)
    z1 = model.module.matchability(desc_B).squeeze(-1)

    B = z0.shape[0]
    M = z0.shape[-1]
    N = z1.shape[-1]

    matchable_A = torch.zeros((B, M), dtype=torch.float32, device=z0.device)
    matchable_A[mnn[:, 0], mnn[:, 1]] = 1.

    matchable_B = torch.zeros((B, N), dtype=torch.float32, device=z1.device)
    matchable_B[mnn[:, 0], mnn[:, 2]] = 1.

    loss_matchability_A = F.binary_cross_entropy_with_logits(z0, matchable_A)
    loss_matchability_B = F.binary_cross_entropy_with_logits(z1, matchable_B)
    return loss_matchability_A + loss_matchability_B
