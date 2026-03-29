import random
import numpy as np

import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    logger.info(f"Setting seed to {seed}")
    # raise NotImplementedError("Not implemented yet.")
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
