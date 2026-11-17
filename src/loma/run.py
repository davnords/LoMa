from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import time
import os
from pathlib import Path
from typing import Callable, Generic, Protocol, runtime_checkable, TypeVar

import git
import torch
import torch.amp.grad_scaler
import torch.nn as nn
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from .distrib import is_main_process, is_distributed
from .logging import logger
from .types import Batch

from .datasets.transforms import Transform
from .datasets.megadepth import MegaDepth


@runtime_checkable
class LossProtocol(Protocol):
    """Protocol for loss functions compatible with train_step."""

    def __call__(
        self, *, batch: Batch, model: nn.Module, step: int
    ) -> tuple[torch.Tensor, dict | None]: ...


def _update_ema_grads(self, grad_norm: float, step: int, gamma: float = 0.99):
    self._ema_grad_raw = (
        gamma * getattr(self, "_ema_grad_raw", 0.0) + (1 - gamma) * grad_norm
    )
    bias_correction = 1 - gamma ** (step + 1)
    self.ema_grad = self._ema_grad_raw / bias_correction


def create_grad_scaler(enabled: bool = True) -> torch.amp.grad_scaler.GradScaler:
    torch.amp.grad_scaler.GradScaler.update_ema_grads = _update_ema_grads
    return torch.amp.grad_scaler.GradScaler(
        enabled=enabled,
        init_scale=65536,
        growth_interval=2000,
        backoff_factor=1 - 1e-8,
        growth_factor=1 + 1e-8,
    )


def create_averaged_model(model: nn.Module, ema_decay: float = 0.999) -> AveragedModel:
    return AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True
    )


def init_wandb(
    *,
    entity: str,
    project: str,
    config: dict,
    name: str | None,
    enabled: bool = True,
    dry_run: bool = False,
) -> tuple[wandb.sdk.wandb_run.Run | None, str | None]:
    wandb_mode = (
        "online" if enabled and not dry_run and is_main_process() else "disabled"
    )
    wandb_run = wandb.init(
        entity=entity,
        project=project,
        config=config,
        name=name,
        mode=wandb_mode,
    )
    wandb_url = wandb_run.url if wandb_run and wandb_mode != "disabled" else None
    return wandb_run, wandb_url


def create_run_dir(base_dir: str, name: str | None) -> Path:
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(f"{base_dir}/{name}/{now}")
    return run_dir


@dataclass(frozen=True)
class TrainData:
    # Dataset configs with transform_overrides (merged with global transform at setup time)
    # MegaDepth family uses shake_t=32 for data augmentation
    mega: MegaDepth.Cfg = MegaDepth.Cfg(split="dedode_train", weight=10_000, transform_overrides={"shake_t": 32})

@dataclass(frozen=True)
class LoMaCfg:
    """Base config for detector/descriptor/glue training experiments."""
    # Data config
    train_data: TrainData = TrainData()
    transform: Transform.Cfg = Transform.Cfg()
    batch_size: int = 8
    num_workers: int = 16

    # Run / logging
    name: str | None = None
    wandb: bool = True
    wandb_entity: str = "<insert_wandb_entity>"
    wandb_project: str = "LoMa"  # Override in subclass
    resume_run: str | None = None

    # Data scaling
    data_fraction: float = 1.0

    # Optimization
    num_steps: int = 1_000_000
    lr: float = 2e-4
    weight_decay: float = 0
    eval_interval: int = 2000
    ckpt_interval: int = 2000
    step_dir_interval: int = 10_000
    epoch_size: int = 2000
    grad_clip_norm: float = 0.01
    ema_decay: float = 0.999

    # Run control
    omp_num_threads: int = 16
    cudnn_benchmark: bool = True
    dry_run: bool = False
    no_test: bool = False
    git: bool = False
    allow_dirty: bool = True

# ---------------------------------------------------------------------------
# Run state management helpers
# ---------------------------------------------------------------------------

# TypeVar for config types - covariant to preserve specific config type
CfgT = TypeVar("CfgT", bound=LoMaCfg)


@dataclass
class RunState(Generic[CfgT]):
    """Container for resumed run state, generic over config type."""

    cfg: CfgT
    step: int
    weights: dict[str, torch.Tensor] | None
    optimizer_state: dict | None
    scheduler_state: dict | None
    averaged_model_state: dict[str, torch.Tensor] | None = None

    @property
    def is_resumed(self) -> bool:
        return self.weights is not None


def maybe_load_run(cfg: CfgT) -> RunState[CfgT]:
    """
    Handle run resumption or fresh start.

    If cfg.resume_run is set, loads the run state from disk.
    Otherwise, initializes a fresh run state.

    Args:
        cfg: The config passed to the training script

    Returns:
        RunState containing the config, step, and optional state dicts.
        The config type is preserved (assumes resumed config is same type).
    """
    is_resumed = cfg.resume_run is not None
    if not is_resumed and not cfg.dry_run:
        assert cfg.name is not None, "name is required"

    if cfg.resume_run is not None:
        loaded_cfg, step, weights, optimizer_state, scheduler_state, averaged_model_state = load_run(
            Path(cfg.resume_run)
        )
        # Cast to CfgT - assumes resumed config is same type as input
        return RunState[CfgT](
            cfg=loaded_cfg,  # type: ignore[arg-type]
            step=step,
            weights=weights,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            averaged_model_state=averaged_model_state,
        )
    else:
        return RunState[CfgT](
            cfg=cfg,
            step=0,
            weights=None,
            optimizer_state=None,
            scheduler_state=None,
            averaged_model_state=None,
        )


def maybe_load_weights(model: nn.Module, run_state: RunState[CfgT]) -> None:
    """
    Load weights into model if resuming from a checkpoint.

    Args:
        model: The model to load weights into
        run_state: The run state containing optional weights
    """
    if run_state.weights is not None:
        model.load_state_dict(run_state.weights)


def maybe_load_averaged_model(
    averaged_model: AveragedModel, run_state: RunState[CfgT]
) -> None:
    """
    Load averaged model state if resuming from a checkpoint.

    Args:
        averaged_model: The averaged (EMA) model to load state into
        run_state: The run state containing optional averaged model state
    """
    if run_state.averaged_model_state is not None:
        averaged_model.load_state_dict(run_state.averaged_model_state)


def maybe_wrap_model_ddp(model: nn.Module, run_state: RunState[CfgT]) -> nn.Module:
    """
    Wrap model in DDP if distributed training is enabled.

    Args:
        model: The model to wrap
        run_state: The run state (unused, but kept for consistency)

    Returns:
        The model wrapped in DDP if distributed, otherwise the model itself
    """
    if is_distributed():
        return torch.nn.parallel.DistributedDataParallel(
            model, gradient_as_bucket_view=False, find_unused_parameters=True
        )
    return model


def create_muon(model, lr, wd):
    requires_grad_params = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad
    ]
    # Muon version from KimiK2 from https://arxiv.org/pdf/2502.16982
    # this allows the same learning rate as for Adamw to be used
    from torch.optim import Muon
    from .optimizer import DualOptimizer
    # We want to filter out the embedding layer and the final projection
    projector_keywords = ["patch_embed", "head"]
    # Weights to ignore in muon:
    # This checks if the parameter name contains any of the keywords and if so excludes it
    projector_params = set([p for name, p in requires_grad_params if any(k in name for k in projector_keywords)])
    hidden_gains_biases = [p for _, p in requires_grad_params if p.ndim != 2 and p not in projector_params]
    # Weights to use muon for:
    hidden_weights = [p for _, p in requires_grad_params if p.ndim == 2 and p not in projector_params]

    adamw = AdamW(
        hidden_gains_biases+list(projector_params),
        lr=lr,
        weight_decay=wd,
    )
    muon = Muon(hidden_weights, lr=lr, weight_decay=wd, adjust_lr_fn="match_rms_adamw")
    return DualOptimizer(adamw, muon)

def create_optimizer_and_scheduler(
    model: nn.Module,
    run_state: RunState[CfgT],
) -> tuple[AdamW, CosineAnnealingLR]:
    """
    Create optimizer and scheduler, loading state if resuming.

    Args:
        model: The model to optimize
        run_state: The run state containing config and optional optimizer/scheduler state

    Returns:
        Tuple of (optimizer, scheduler)
    """
    cfg = run_state.cfg
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # optimizer = create_muon(model, lr=cfg.lr, wd=cfg.weight_decay)
    if run_state.optimizer_state is not None:
        optimizer.load_state_dict(run_state.optimizer_state)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_steps,
        last_epoch=run_state.step - 1 if run_state.is_resumed else -1,
    )
    if run_state.scheduler_state is not None:
        scheduler.load_state_dict(run_state.scheduler_state)

    return optimizer, scheduler


def dump_run(
    *,
    run_dir: Path,
    cfg,
    step: int,
    weights: dict[str, torch.Tensor],
    optimizer_state: dict[str, torch.Tensor],
    make_step_dir: bool,
    avg_weights: dict[str, torch.Tensor] | None = None,
    scheduler_state: dict | None = None,
):
    logger.info(f"Dumping run at step {step}")
    if make_step_dir:
        run_dir = run_dir / f"step_{step}"
        run_dir.mkdir(parents=True, exist_ok=True)
    serialized_cfg = asdict(cfg)
    json.dump(serialized_cfg, open(run_dir / "cfg.json", "w"), indent=2)
    pickle.dump(cfg, open(run_dir / "cfg.pkl", "wb"))
    open(run_dir / "step.txt", "w").write(str(step))
    torch.save(weights, run_dir / "weights.pth")
    torch.save(optimizer_state, run_dir / "optimizer_state.pth")
    if avg_weights is not None:
        torch.save(avg_weights, run_dir / "avg_weights.pth")
    if scheduler_state is not None:
        torch.save(scheduler_state, run_dir / "scheduler_state.pth")


def load_run(run_dir: Path):
    cfg = pickle.load(open(run_dir / "cfg.pkl", "rb"))
    step = int(open(run_dir / "step.txt", "r").read())
    weights = torch.load(run_dir / "weights.pth")
    optimizer_state = torch.load(run_dir / "optimizer_state.pth")
    scheduler_state = None
    scheduler_state_path = run_dir / "scheduler_state.pth"
    if scheduler_state_path.exists():
        scheduler_state = torch.load(scheduler_state_path)
    averaged_model_state = None
    avg_weights_path = run_dir / "avg_weights.pth"
    if avg_weights_path.exists():
        averaged_model_state = torch.load(avg_weights_path)
    logger.info(f"Loaded run at step {step}")
    return cfg, step, weights, optimizer_state, scheduler_state, averaged_model_state


def train_step(
    *,
    batch: Batch,
    model: torch.nn.Module,
    step: int,
    loss: LossProtocol,
    grad_scaler: torch.amp.grad_scaler.GradScaler,
    optimizer: torch.optim.Optimizer,
    cfg,  # Any config with grad_clip_norm attribute
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run_name: str | None,
):
    t0 = time.perf_counter()
    loss_value, stats = loss(batch=batch, model=model, step=step)
    grad_scaler.scale(loss_value).backward()
    grad_scaler.unscale_(optimizer)
    named_params_with_grads = [
        (n, p) for n, p in model.named_parameters() if p.grad is not None
    ]
    named_grad_norms = {n: p.grad.norm() for n, p in named_params_with_grads}
    tot_grad_norm = sum([v**2 for v in named_grad_norms.values()]) ** 0.5
    if tot_grad_norm.isnan():
        logger.warning(f"NaN grad norm at step {step}")
    if is_main_process():
        wandb.log({"total-grad-norm": tot_grad_norm}, step=step)
        wandb.log({"scale": grad_scaler.get_scale()}, step=step)
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
    if tot_grad_norm > 2 * getattr(grad_scaler, "ema_grad", 0.0) and step > 100:
        pass
        # logger.warning(f"Grad norm {tot_grad_norm=} is too large.")
    else:
        if hasattr(grad_scaler, "update_ema_grads"):
            grad_scaler.update_ema_grads(tot_grad_norm, step)
        # if not outlier, update ema grad
    grad_scaler.step(optimizer)
    grad_scaler.update()
    scheduler.step()
    optimizer.zero_grad()
    t1 = time.perf_counter()
    if is_main_process():
        wandb.log({"train-step-time": t1 - t0}, step=step)
    return loss_value.item()


def _get_class(cfg):
    import sys

    return getattr(
        sys.modules[cfg.__module__],
        type(cfg).__qualname__.rsplit(".", 1)[0],
    )


def _init_train_datasets(data_cfgs, transform_cfg, data_fraction: float = 1.0):
    from dataclasses import fields as dc_fields
    datasets = []
    for dataset in data_cfgs:
        if dataset.weight > 0:
            if data_fraction < 1.0 and any(f.name == "fraction" for f in dc_fields(dataset)):
                dataset = replace(dataset, fraction=data_fraction)
            data_class = _get_class(dataset)
            datasets.append(
                data_class(
                    dataset, replace(transform_cfg, **dataset.transform_overrides)
                )
            )
    return datasets


def init_benchmarks(benchmarks_cfg):
    """Initialize benchmarks from a benchmarks config dataclass.
    
    Args:
        benchmarks_cfg: A dataclass containing benchmark configs as fields
        
    Returns:
        List of initialized benchmark instances
    """
    benchmarks = []
    for benchmark_cfg in benchmarks_cfg.__dict__.values():
        benchmark_class = _get_class(benchmark_cfg)
        benchmarks.append(benchmark_class(benchmark_cfg))
    return benchmarks


def setup_data(
    cfg,
):  # Any config with train_data, transform, batch_size, num_workers
    train_data = torch.utils.data.ConcatDataset(
        _init_train_datasets(cfg.train_data.__dict__.values(), cfg.transform, getattr(cfg, "data_fraction", 1.0))
    )

    sampler = DistributedSampler(train_data) if is_distributed() else None
    loader = torch.utils.data.DataLoader[Batch](
        train_data,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=Batch.collate,
    )
    return sampler, loader


EvalCallback = Callable[[nn.Module, int, Path], None]
"""Signature: (model, step, run_dir) -> None. Called during evaluation."""


def set_torch_misc(cfg):
    torch.set_num_threads(cfg.omp_num_threads)
    logger.info(f"Using {torch.get_num_threads()} OpenMP threads")
    os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        logger.info("CUDNN benchmark enabled")
    torch.set_float32_matmul_precision("highest")

def setup_run(
    *,
    run_dir: Path,
    cfg,  # Any config dataclass
    step: int,
    weights: dict[str, torch.Tensor],
    optimizer_state: dict[str, torch.Tensor],
    wandb_url: str | None = None,
):
    assert run_dir.exists(), f"Run directory {run_dir} does not exist"
    logger.info(
        f"Setting up experiment tracking for {cfg.name=} at {run_dir=} (and on wandb)."
    )

    # check git not dirty
    if cfg.git:
        repo = git.Repo(search_parent_directories=True)
        is_dirty = False
        logger.info(f"On branch: {repo.active_branch}")
        logger.info(f"Git commit: {repo.head.commit.hexsha}")
        is_dirty = repo.is_dirty()
        logger.warning(f"Git is dirty with diff:\n{repo.git.diff('HEAD')}")
        if is_dirty and not cfg.allow_dirty:
            raise RuntimeError(f"Git is dirty with diff:\n{repo.git.diff('HEAD')}")
        # all ok

    dump_run(
        run_dir=run_dir,
        cfg=cfg,
        step=step,
        weights=weights,
        optimizer_state=optimizer_state,
        make_step_dir=False,
    )
    if wandb_url:
        (run_dir / "wandb_url.txt").write_text(wandb_url or "")

    commit_msg = f"Run {cfg.name} at {run_dir}"
    if wandb_url:
        commit_msg += f"\n\nWandB: {wandb_url}"

    not_committed = True
    while not_committed and cfg.git:
        try:
            # stage and commit
            repo.git.add(".")
            # repo.git.add(run_dir / "cfg.json")
            repo.git.commit(m=commit_msg)
            not_committed = False
        except git.exc.GitCommandError as e:  # type: ignore
            logger.warning(f"Failed to commit: {e}")
            time.sleep(1)
    if cfg.git:
        logger.info(f"Staged and committed config to {run_dir / 'cfg.json'}")


def train_loop(
    *,
    cfg,  # config with num_steps, ckpt_interval, eval_interval, step_dir_interval, no_test, dry_run
    model: nn.Module,
    d_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss: LossProtocol,
    grad_scaler: torch.amp.grad_scaler.GradScaler,
    averaged_model: AveragedModel,
    sampler: DistributedSampler | None,
    loader: torch.utils.data.DataLoader,
    run_dir: Path,
    step: int,
    epoch_size: int | None = None,
    eval_callback: EvalCallback | None = None,
    eval_model_attr: str = "module",  # attribute of averaged_model to use for evaluation
) -> int:
    from tqdm import tqdm
    from .device import device

    if epoch_size is None:
        epoch_size = len(loader)

    num_epochs = cfg.num_steps // epoch_size + 1
    start_epoch = step // epoch_size

    for epoch in range(start_epoch, num_epochs):
        torch.cuda.empty_cache() 
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in (pbar := tqdm(loader)):
            if step >= cfg.num_steps:
                break

            model.train(True)
            batch = batch.to(device)
            loss_value = train_step(
                batch=batch,
                model=d_model,
                step=step,
                loss=loss,
                grad_scaler=grad_scaler,
                optimizer=optimizer,
                cfg=cfg,
                scheduler=scheduler,
                run_name=cfg.name,
            )
            averaged_model.update_parameters(model)
            pbar.set_description(f"Loss: {loss_value:.4f}")

            # Checkpoint
            if step % cfg.ckpt_interval == 0 and is_main_process() and not cfg.dry_run:
                dump_run(
                    run_dir=run_dir,
                    cfg=cfg,
                    step=step,
                    weights=model.state_dict(),
                    avg_weights=averaged_model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    make_step_dir=(step % cfg.step_dir_interval == 0),
                )

            # Evaluation
            if (
                step % cfg.eval_interval == 0
                and is_main_process()
                and not cfg.no_test
            ):
                eval_model = getattr(averaged_model, eval_model_attr)
                if eval_callback is not None:
                    eval_callback(eval_model, step, run_dir)

            step += 1
            if step % epoch_size == 0:
                break

        if step >= cfg.num_steps:
            break

    # Final checkpoint
    dump_run(
        run_dir=run_dir,
        cfg=cfg,
        step=step,
        weights=model.state_dict(),
        avg_weights=averaged_model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        make_step_dir=False,
    )
    if is_main_process():
        wandb.finish()

    return step
