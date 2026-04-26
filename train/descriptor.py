from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch.nn as nn
import tyro

from loma.datasets.megadepth import MegaDepth
from loma.datasets.transforms import Transform
from loma.descriptor.nll import NLLBenchmark
from loma.descriptor.dedode import DeDoDeDescriptor
from loma.descriptor.loss import DescriptorLoss
from loma.detector.dad import DaD
from loma.device import device
from loma.distrib import init_distributed, is_main_process
from loma.logging import configure_logger, logger
from loma.run import (
    LoMaCfg,
    setup_data,
    setup_run,
    create_grad_scaler,
    create_averaged_model,
    init_wandb,
    create_run_dir,
    train_loop,
    set_torch_misc,
    maybe_load_run,
    maybe_load_weights,
    maybe_load_averaged_model,
    maybe_wrap_model_ddp,
    create_optimizer_and_scheduler,
)

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@dataclass(frozen=True)
class DescriptorCfg(LoMaCfg):
    transform: Transform.Cfg = Transform.Cfg(heights=[560], widths=[560])
    wandb_project: str = "LoMa-descriptor"

    eval_transform: Transform.BenchmarkCfg = Transform.BenchmarkCfg(heights = [560], widths = [560])
    loss: DescriptorLoss.Cfg = DescriptorLoss.Cfg()
    model: DeDoDeDescriptor.Cfg = DeDoDeDescriptor.Cfg()

    mega_benchmark: NLLBenchmark.Cfg = NLLBenchmark.Cfg(num_samples=500)
    mega_eval_data: MegaDepth.Cfg = MegaDepth.Cfg(split="dedode_test", weight=10_000)


def main(_cfg: DescriptorCfg) -> None:
    # Handle run resumption or fresh start
    run_state = maybe_load_run(_cfg)
    cfg = run_state.cfg
    step = run_state.step

    # init distributed
    init_distributed()
    set_torch_misc(cfg)

    # run directory and logging
    run_dir = create_run_dir("train/descriptor/runs", cfg.name)
    if is_main_process() and not cfg.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        configure_logger(file_path=str(run_dir / "train.log"))
    logger.info(f"Run directory: {run_dir}")

    # model
    model = DeDoDeDescriptor(cfg.model).to(device)
    maybe_load_weights(model, run_state)
    d_model = maybe_wrap_model_ddp(model, run_state)

    # optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, run_state)

    # grad scaler and EMA
    grad_scaler = create_grad_scaler(enabled=(device.type == "cuda"))
    averaged_model = create_averaged_model(model, ema_decay=cfg.ema_decay)
    maybe_load_averaged_model(averaged_model, run_state)

    # loss
    loss_fn = DescriptorLoss(cfg.loss)

    # Initialize detector for evaluation benchmarks
    detector = DaD().to(device)
    detector.eval()

    # Setup evaluation benchmarks
    mega_dataset = MegaDepth(
        cfg.mega_eval_data,
        transform_cfg=cfg.eval_transform,
    )
    mega_benchmark = NLLBenchmark(mega_dataset, cfg.mega_benchmark)

    # Evaluation callback for descriptor training
    def descriptor_eval_callback(
        eval_model: nn.Module, step: int, run_dir: Path
    ) -> None:
        eval_model.eval()
        logger.info("Running MegaDepth NLL benchmark...")
        mega_benchmark(
            detector=detector,
            descriptor=eval_model,
            step=step,
        )
        logger.info("MegaDepth benchmark complete.")

    # wandb
    wandb_run, wandb_url = init_wandb(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        config=asdict(cfg),
        name=cfg.name,
        enabled=cfg.wandb,
        dry_run=cfg.dry_run,
    )

    # setup run directory and save initial checkpoint
    if is_main_process() and not cfg.dry_run:
        setup_run(
            run_dir=run_dir,
            cfg=cfg,
            step=step,
            weights=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            wandb_url=wandb_url,
        )
        if run_state.is_resumed:
            with open(run_dir / "resumed_from.txt", "w") as f:
                f.write(str(_cfg.resume_run))

    # data
    sampler, loader = setup_data(cfg=cfg)

    # training loop
    train_loop(
        cfg=cfg,
        model=model,
        d_model=d_model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss_fn,
        grad_scaler=grad_scaler,
        averaged_model=averaged_model,
        sampler=sampler,
        loader=loader,
        run_dir=run_dir,
        step=step,
        epoch_size=cfg.epoch_size,
        eval_callback=descriptor_eval_callback,
    )


if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    cfg = tyro.cli(DescriptorCfg)
    main(cfg)
