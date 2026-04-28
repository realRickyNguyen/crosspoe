import torch
import torch.nn as nn


class SurvivalHead(nn.Module):
    """Linear Cox PH head: maps a latent vector to a scalar risk score."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.risk = nn.Linear(latent_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns risk scores of shape [batch, 1]. Higher score = higher risk."""
        return self.risk(z)


def cox_partial_likelihood_loss(
    risk_scores: torch.Tensor,
    os_time: torch.Tensor,
    os_event: torch.Tensor,
) -> torch.Tensor:
    """
    Negative partial log-likelihood for the Cox proportional hazards model.

    Args:
        risk_scores : [batch, 1]  raw risk scores (higher = more risk)
        os_time     : [batch]     follow-up / survival time
        os_event    : [batch]     event indicator (1 = event, 0 = censored, -1 = missing)

    Returns:
        Scalar loss averaged over uncensored samples.
        Returns 0 (with grad) if fewer than 2 valid samples or no events.
    """
    valid = os_event >= 0
    if valid.sum() < 2 or os_event[valid].sum() < 1:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

    risk  = risk_scores[valid].squeeze(-1)
    time  = os_time[valid]
    event = os_event[valid].float()

    sorted_idx     = torch.argsort(time, descending=True)
    risk           = risk[sorted_idx]
    event          = event[sorted_idx]
    log_cumsum_exp = torch.logcumsumexp(risk, dim=0)
    partial_ll     = event * (risk - log_cumsum_exp)

    n_events = event.sum()
    if n_events > 0:
        return -partial_ll.sum() / n_events
    return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)
