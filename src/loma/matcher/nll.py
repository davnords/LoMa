from dataclasses import dataclass

import torch
import wandb
from tqdm import tqdm

from ..device import device
from ..geometry import compute_sparse_mnn_matches
from ..types import Batch

from .loma import LoMa

class NLLBenchmark:
    @dataclass(frozen=True)
    class Cfg:
        num_samples: int = 1000
        batch_size: int = 8
        num_keypoints: int = 2048
        depth_error_threshold: float = 0.05
        flow_error_threshold: float = 5e-3
        error_threshold: float = 0.01
        local_neighbourhood_size: int = 1

    def __init__(
        self,
        dataset: torch.utils.data.Dataset[Batch],
        cfg: Cfg,
    ) -> None:
        self.cfg = cfg
        sampler = (
            torch.utils.data.WeightedRandomSampler(
                torch.ones(len(dataset)), replacement=False, num_samples=cfg.num_samples
            )
            if cfg.num_samples != -1
            else None
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.batch_size,
            sampler=sampler,
            collate_fn=Batch.collate,
        )
        self.dataloader = dataloader
        self.tracked_metrics = {}
        self.batch_size = cfg.batch_size
        self.N = len(dataloader)
        self.num_keypoints = cfg.num_keypoints

    @torch.inference_mode()
    def benchmark(
        self,
        glue: LoMa,
        step: int,
    ) -> dict[str, float]:
        self.tracked_metrics = {}
        glue.eval()

        print("Evaluating Glue NLL...")
        neg_log_likelihood = 0.0
        total_matches = 0
        num_batches_with_matches = 0

        for batch in tqdm(self.dataloader, mininterval=1.0):
            batch: Batch = batch.to(device)

            # Detect keypoints using glue's frozen detector
            detections = glue.detect(batch, num_keypoints=self.num_keypoints)
            detector_keypoints_A, detector_keypoints_B = detections["keypoints"].chunk(2)

            # Prepend correspondences to keypoints (compute_mnn_matches handles sparse samples)
            corresp_A, corresp_B = batch.correspondences_AB.chunk(2, dim=-1)
            keypoints_A = torch.cat([corresp_A, detector_keypoints_A], dim=1)
            keypoints_B = torch.cat([corresp_B, detector_keypoints_B], dim=1)

            # Get descriptors at keypoints using glue's frozen descriptor
            desc_A, desc_B = glue.describe(
                batch, torch.cat((keypoints_A, keypoints_B), dim=0)
            )["descriptions"].chunk(2)

            # Compute GT matches
            mnn = compute_sparse_mnn_matches(
                keypoints_A,
                keypoints_B,
                batch,
                depth_error_threshold=self.cfg.depth_error_threshold,
                flow_error_threshold=self.cfg.flow_error_threshold,
                local_neighbourhood_size=self.cfg.local_neighbourhood_size,
                error_threshold=self.cfg.error_threshold,
            )

            if mnn.numel() == 0:
                continue

            # Run matcher
            result = glue(batch, keypoints_A, keypoints_B, desc_A, desc_B)
            scores = result["scores"]

            # Compute NLL on GT matches
            # mnn shape: [num_matches, 3] - (batch_idx, idx_A, idx_B)
            batch_nll = -scores[mnn[:, 0], mnn[:, 1], mnn[:, 2]].mean()
            neg_log_likelihood += batch_nll
            total_matches += mnn.shape[0]
            num_batches_with_matches += 1

        if num_batches_with_matches > 0:
            mean_neg_log_likelihood = neg_log_likelihood / num_batches_with_matches
        else:
            mean_neg_log_likelihood = torch.tensor(0.0)

        dataset_name = type(self.dataloader.dataset).__name__
        wandb.log(
            {
                f"{dataset_name}_glue_nll": mean_neg_log_likelihood.item(),
                f"{dataset_name}_glue_total_matches": total_matches,
            },
            step=step,
        )

        return {"nll": mean_neg_log_likelihood.item(), "total_matches": total_matches}

    def __call__(
        self,
        glue: LoMa,
        step: int,
    ) -> dict[str, float]:
        return self.benchmark(glue, step)
