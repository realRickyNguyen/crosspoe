import torch.nn as nn


def fc_block(in_dim: int, out_dim: int, dropout: float = 0.2) -> nn.Sequential:
    """Fully-connected block: Linear -> LayerNorm -> GELU -> Dropout."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


def count_params(model: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
