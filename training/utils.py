import random

import numpy as np
import torch
from lifelines.utils import concordance_index as lifelines_ci


def set_seed(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_beta(epoch: int, kl_warmup_epochs: int, beta_max: float = 1.0) -> float:
    """Linear KL annealing: 0 -> beta_max over kl_warmup_epochs epochs."""
    if epoch >= kl_warmup_epochs:
        return beta_max
    return beta_max * (epoch / kl_warmup_epochs)


def get_dropout_p(epoch: int, cfg: dict) -> float:
    """
    Curriculum modality dropout probability for the current epoch.

    Linearly ramps from 0 to cfg['dropout_p_max'] over cfg['dropout_ramp_epochs'],
    starting at cfg['dropout_start_epoch']. Returns 0 if dropout is disabled.
    """
    if not cfg.get("use_modality_dropout", False):
        return 0.0
    start    = cfg["dropout_start_epoch"]
    ramp     = cfg["dropout_ramp_epochs"]
    p_max    = cfg["dropout_p_max"]
    if epoch < start:
        return 0.0
    progress = min(1.0, (epoch - start) / ramp)
    return progress * p_max


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensor values in a batch dict to the target device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def concordance_index(times, risk_scores, events) -> float:
    """Harrell's C-index via lifelines. Returns 0.5 on failure."""
    try:
        return lifelines_ci(times, -risk_scores, events)
    except Exception:
        return 0.5
