import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
import wandb
from .loss import dual_log_softmax_matcher
from ..geometry import compute_sparse_mnn_matches
from ..types import Batch, Detector
from ..device import device
from .dedode import DeDoDeDescriptor


class NLLBenchmark:
    @dataclass(frozen=True)
    class Cfg:
        rel_depth_error_threshold: float = 0.05
        gt_warp_fwd_bwd_error_threshold: float = 5e-3
        local_neighbourhood_size: int = 1
        num_samples: int = 1000
        batch_size: int = 8
        num_keypoints: int = 2048
        normalize_descriptions: bool = True
        inv_temp: float = 20
        depth_error_threshold: float = 0.05
        flow_error_threshold: float = 5e-3
        error_threshold: float = 0.01

    def __init__(
        self,
        dataset: torch.utils.data.Dataset[Batch],
        cfg: Cfg,
    ) -> None:
        self.cfg = cfg
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.ones(len(dataset)), replacement=False, num_samples=cfg.num_samples
        ) if cfg.num_samples != -1 else None
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
        detector: Detector,
        descriptor: DeDoDeDescriptor,
        step: int,
    ) -> dict[str, float]:
        self.tracked_metrics = {}
        descriptor.eval()
        if detector is not None:
            detector.eval()

        print("Evaluating NLL...")
        neg_log_likelihood = 0.0
        for batch in tqdm(self.dataloader, mininterval=1.0):
            batch: Batch = batch.to(device)
            detections = detector.detect(batch, num_keypoints=self.num_keypoints)
            detector_keypoints_A, detector_keypoints_B = detections["keypoints"].chunk(2)
            
            # Prepend correspondences to keypoints (compute_mnn_matches handles sparse samples)
            corresp_A, corresp_B = batch.correspondences_AB.chunk(2, dim=-1)
            keypoints_A = torch.cat([corresp_A, detector_keypoints_A], dim=1)
            keypoints_B = torch.cat([corresp_B, detector_keypoints_B], dim=1)
            
            mnn = compute_sparse_mnn_matches(
                keypoints_A,
                keypoints_B,
                batch,
                depth_error_threshold=self.cfg.depth_error_threshold,
                flow_error_threshold=self.cfg.flow_error_threshold,
                local_neighbourhood_size=self.cfg.local_neighbourhood_size,
                error_threshold=self.cfg.error_threshold,
            )
            
            desc_grid_A, desc_grid_B = descriptor(batch).chunk(2)
            desc_A = F.grid_sample(
                desc_grid_A.float(),
                keypoints_A[:, None],
                mode="bilinear",
                align_corners=False,
            )[:, :, 0].mT
            desc_B = F.grid_sample(
                desc_grid_B.float(),
                keypoints_B[:, None],
                mode="bilinear",
                align_corners=False,
            )[:, :, 0].mT
            
            if mnn.numel() != 0:
                logP_A_B = dual_log_softmax_matcher(
                    desc_A,
                    desc_B,
                    normalize=self.cfg.normalize_descriptions,
                    inv_temperature=self.cfg.inv_temp,
                )
                neg_log_likelihood += -logP_A_B[mnn[:, 0], mnn[:, 1], mnn[:, 2]].mean()
        mean_neg_log_likelihood = neg_log_likelihood / self.N
        wandb.log(
            {f"{type(self.dataloader.dataset).__name__}_neg_log_likelihood": mean_neg_log_likelihood.item()},
            step=step,
        )
        return {"nll": mean_neg_log_likelihood.item()}

    def __call__(
        self,
        detector: Detector | None,
        descriptor: DeDoDeDescriptor,
        step: int,
    ) -> dict[str, float]:
        return self.benchmark(detector, descriptor, step)
