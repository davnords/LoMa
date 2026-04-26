import os
import torch
import logging

logger = logging.getLogger(__name__)


__all__ = ["init_distributed", "is_main_process", "is_distributed", "get_rank"]

_is_distributed: bool = False
_is_initialized: bool = False


def init_distributed():
    global _is_distributed
    global _is_initialized
    if _is_initialized:
        raise RuntimeError("Distributed training already initialized.")
    local_rank = int(os.environ.get("LOCAL_RANK", default=-1))
    if local_rank != -1:
        logger.info(f"Initializing distributed training on local rank {local_rank}")

        torch.distributed.init_process_group(backend="nccl")  # ty: ignore[possibly-unbound-attribute]
        torch.cuda.set_device(local_rank)
    _set_is_distributed(local_rank != -1)
    _is_initialized = True


def get_rank() -> int:
    if not is_distributed():
        return 0
    return torch.distributed.get_rank()  # type: ignore[possibly-unbound-attribute]


def is_main_process():
    return get_rank() == 0
    # if not is_distributed():
    #     return True
    # return torch.distributed.get_rank() == 0  # type: ignore[possibly-unbound-attribute]


def is_distributed():
    global _is_distributed
    return _is_distributed


def _set_is_distributed(value: bool):
    global _is_distributed
    _is_distributed = value
