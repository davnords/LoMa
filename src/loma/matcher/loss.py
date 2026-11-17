from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F

from loma.geometry import (
    compute_gt_warp_from_batch,
    compute_sparse_mnn_matches,
    gt_affine_at_matches,
)
from loma.loma import LoMa
from loma.types import Batch


class GlueLoss(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        num_keypoints: int = 2048
        depth_error_threshold: float = 0.05
        flow_error_threshold: float = 5e-3
        error_threshold: float = 0.01
        local_neighbourhood_size: int = 1
        # Loss weights for intermediate layers
        layer_loss_weight: float = 1.0
        # Affine LoMa: weight of the Charbonnier loss on the relative affine of each match (0 = no affine loss)
        affine_loss_weight: float = 0.0
        charbonnier_eps: float = 1e-3

    def __init__(self, cfg: Cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, batch: Batch, model: LoMa | DistributedDataParallel, step: int):
        module: LoMa = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        # Detect keypoints using model's frozen detector
        detector_kpts_A, detector_kpts_B = (
            module.detect(batch, num_keypoints=self.cfg.num_keypoints)["keypoints"]
            .clone()
            .chunk(2)
        )

        # Get descriptors at keypoints using model's frozen descriptor
        desc_A, desc_B = module.describe(
            batch, torch.cat((detector_kpts_A, detector_kpts_B), dim=0)
        )["descriptions"].chunk(2)

        # Compute ground truth matches
        gt_AB = None
        if self.cfg.affine_loss_weight > 0:
            gt_AB = compute_gt_warp_from_batch(
                batch,
                depth_error_threshold=self.cfg.depth_error_threshold,
                flow_error_threshold=self.cfg.flow_error_threshold,
                local_neighbourhood_size=self.cfg.local_neighbourhood_size,
            )
        mnn = compute_sparse_mnn_matches(
            detector_kpts_A,
            detector_kpts_B,
            batch,
            depth_error_threshold=self.cfg.depth_error_threshold,
            flow_error_threshold=self.cfg.flow_error_threshold,
            local_neighbourhood_size=self.cfg.local_neighbourhood_size,
            error_threshold=self.cfg.error_threshold,
            gt_AB=gt_AB,
        )

        # Run LoMa (trainable)
        result = model(batch, detector_kpts_A, detector_kpts_B, desc_A, desc_B)
        all_scores = result["all_scores"]  # List of log assignment matrices per layer

        if mnn.numel() == 0:
            # No valid matches found - return zero loss
            dummy_loss = sum((0 * s).mean() for s in all_scores)
            if gt_AB is not None:
                dummy_loss = dummy_loss + 0 * (result["affine_A"].sum() + result["affine_B"].sum())
            return dummy_loss, None

        # Compute loss over all layers (intermediate supervision)
        total_loss = 0.0
        n_layers = len(all_scores)
        for scores in all_scores:
            # scores shape: [B, M+1, N+1] - log assignment matrix
            # mnn shape: [num_matches, 3] - (batch_idx, idx_A, idx_B)
            M, N = scores.shape[1] - 1, scores.shape[2] - 1
            matchable_A = torch.zeros(
                (scores.shape[0], M), dtype=torch.float32, device=scores.device
            )
            matchable_A[mnn[:, 0], mnn[:, 1]] = 1.0
            matchable_B = torch.zeros(
                (scores.shape[0], N), dtype=torch.float32, device=scores.device
            )
            matchable_B[mnn[:, 0], mnn[:, 2]] = 1.0
            layer_loss_conditional = -scores[mnn[:, 0], mnn[:, 1], mnn[:, 2]].mean()

            layer_loss_matchability_A = F.binary_cross_entropy_with_logits(
                scores[:, :-1, -1], matchable_A
            )
            layer_loss_matchability_B = F.binary_cross_entropy_with_logits(
                scores[:, -1, :-1], matchable_B
            )

            layer_loss_matchability = (
                layer_loss_matchability_A + layer_loss_matchability_B
            )
            total_loss = (
                total_loss
                + (layer_loss_conditional + layer_loss_matchability)
                * self.cfg.layer_loss_weight
            )

        # Average over layers
        total_loss = total_loss / n_layers

        if gt_AB is not None:
            gt_affine, valid = gt_affine_at_matches(gt_AB, detector_kpts_A, mnn)
            A_A = result["affine_A"][mnn[:, 0], mnn[:, 1]]
            A_B = result["affine_B"][mnn[:, 0], mnn[:, 2]]
            r2 = ((A_B @ torch.linalg.inv(A_A) - gt_affine) ** 2).sum((-2, -1))
            eps = self.cfg.charbonnier_eps
            affine_loss = (torch.sqrt(r2 + eps**2) - eps)[valid].sum() / valid.sum().clamp(min=1)
            total_loss = total_loss + self.cfg.affine_loss_weight * affine_loss

        return total_loss, None
