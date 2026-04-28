import torch
import torch.nn as nn


class ProductOfExperts(nn.Module):
    """
    Product of Experts posterior fusion with optional gated cross-modal translations.

    Observed modalities contribute their full precision to the PoE aggregate.
    Translated pseudo-posteriors (for missing modalities) contribute gated precision:
        precision_translated = gate * precision_pseudo
    where gate in (0, 1) is a learned per-sample trust score.
    """

    def __init__(self, latent_dim: int, eps: float = 1e-8):
        super().__init__()
        self.latent_dim = latent_dim
        self.eps = eps

    def forward(
        self,
        mu_list,
        logvar_list,
        translated_mu_list=None,
        translated_logvar_list=None,
        translation_gates=None,
    ):
        """
        Args:
            mu_list:                list of mu tensors for observed modalities
            logvar_list:            list of logvar tensors for observed modalities
            translated_mu_list:     list of pseudo-posterior mus (optional)
            translated_logvar_list: list of pseudo-posterior logvars (optional)
            translation_gates:      list of gate tensors in (0, 1) (optional)

        Returns:
            mu_poe, logvar_poe — aggregated posterior parameters
        """
        device     = mu_list[0].device
        batch_size = mu_list[0].size(0)

        mu_prior     = torch.zeros(batch_size, self.latent_dim, device=device)
        logvar_prior = torch.zeros(batch_size, self.latent_dim, device=device)
        precision    = torch.exp(-logvar_prior)
        precision_mu = mu_prior * precision

        for mu, logvar in zip(mu_list, logvar_list):
            prec         = torch.exp(-logvar)
            precision    = precision + prec
            precision_mu = precision_mu + mu * prec

        if translated_mu_list is not None:
            for mu_t, logvar_t, gate in zip(
                translated_mu_list, translated_logvar_list, translation_gates
            ):
                prec_t       = torch.exp(-logvar_t)
                gated_prec   = gate * prec_t
                precision    = precision + gated_prec
                precision_mu = precision_mu + mu_t * gated_prec

        mu_poe     = precision_mu / (precision + self.eps)
        logvar_poe = -torch.log(precision + self.eps)
        return mu_poe, logvar_poe


def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterisation trick: z = mu + eps * std,  eps ~ N(0, I).
    Returns mu directly when grad is disabled (i.e. during inference).
    """
    if not torch.is_grad_enabled():
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
