import torch
import torch.nn as nn

from .decoders import MethylDecoder, MIRNADecoder, RNADecoder
from .encoders import MethylEncoder, MIRNAEncoder, RNAEncoder
from .poe import reparameterise

# Default feature dimensions — adjust for other datasets
_N_RNA    = 4652
_N_MIRNA  = 524
_N_METHYL = 37482
_N_LATENT = 48


class CrossPoE(nn.Module):
    """
    Multi-omics Product of Experts VAE for survival prediction.

    Fuses mRNA, miRNA, and methylation posteriors via precision-weighted PoE.
    Supports optional cross-modal latent translation for partially-observed samples.

    Architecture:
        - Three modality-specific VAE encoders -> (mu, logvar)
        - Precision-weighted PoE fusion -> shared latent z
        - Three symmetric decoders for reconstruction
        - Optionally enhanced by CrossModalTranslator (passed at forward time)

    Args:
        latent_dim : dimension of the shared latent space
        n_rna      : input feature dimension for the RNA modality
        n_mirna    : input feature dimension for the miRNA modality
        n_methyl   : input feature dimension for the methylation modality
    """

    def __init__(
        self,
        latent_dim: int = _N_LATENT,
        n_rna:      int = _N_RNA,
        n_mirna:    int = _N_MIRNA,
        n_methyl:   int = _N_METHYL,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_rna      = n_rna
        self.n_mirna    = n_mirna
        self.n_methyl   = n_methyl

        self.rna_enc    = RNAEncoder(in_dim=n_rna,    latent_dim=latent_dim)
        self.mirna_enc  = MIRNAEncoder(in_dim=n_mirna,  latent_dim=latent_dim)
        self.methyl_enc = MethylEncoder(in_dim=n_methyl, latent_dim=latent_dim)

        self.rna_dec    = RNADecoder(latent_dim=latent_dim, out_dim=n_rna)
        self.mirna_dec  = MIRNADecoder(latent_dim=latent_dim, out_dim=n_mirna)
        self.methyl_dec = MethylDecoder(latent_dim=latent_dim, out_dim=n_methyl)

    def forward(
        self,
        batch,
        translator=None,
        epoch: int = 0,
        translation_warmup_epochs: int = 10,
    ) -> dict:
        """
        Args:
            batch                     : dict from collate_fn
            translator                : CrossModalTranslator or None
            epoch                     : current training epoch
            translation_warmup_epochs : epoch at which translations are activated

        Returns dict with keys:
            mu_poe, logvar_poe, z, z_surv,
            recons  (dict: rna/mirna/methyl -> tensor),
            mus, logvars  (per-modality encoder outputs),
            z_rna, z_mirna, z_methyl  (per-modality samples, or None if absent),
            masks  (list of per-modality bool masks)
        """
        batch_size = batch["mask"].shape[0]
        device     = batch["mask"].device
        mask       = batch["mask"]

        rna_mask    = mask[:, 0]
        mirna_mask  = mask[:, 1]
        methyl_mask = mask[:, 2]

        # --- Encode observed modalities ---
        mu_rna    = torch.zeros(batch_size, self.latent_dim, device=device)
        lv_rna    = torch.zeros(batch_size, self.latent_dim, device=device)
        mu_mirna  = torch.zeros(batch_size, self.latent_dim, device=device)
        lv_mirna  = torch.zeros(batch_size, self.latent_dim, device=device)
        mu_methyl = torch.zeros(batch_size, self.latent_dim, device=device)
        lv_methyl = torch.zeros(batch_size, self.latent_dim, device=device)

        if rna_mask.any() and batch["rna"] is not None:
            mu_r, lv_r             = self.rna_enc(batch["rna"][rna_mask])
            mu_rna[rna_mask]       = mu_r
            lv_rna[rna_mask]       = lv_r
        if mirna_mask.any() and batch["mirna"] is not None:
            mu_m, lv_m             = self.mirna_enc(batch["mirna"][mirna_mask])
            mu_mirna[mirna_mask]   = mu_m
            lv_mirna[mirna_mask]   = lv_m
        if methyl_mask.any() and batch["methyl"] is not None:
            mu_me, lv_me           = self.methyl_enc(batch["methyl"][methyl_mask])
            mu_methyl[methyl_mask] = mu_me
            lv_methyl[methyl_mask] = lv_me

        mus            = [mu_rna, mu_mirna, mu_methyl]
        logvars        = [lv_rna, lv_mirna, lv_methyl]
        modality_masks = [rna_mask, mirna_mask, methyl_mask]

        # --- Precision-weighted PoE fusion ---
        precision_sum = torch.ones(batch_size, self.latent_dim, device=device)
        weighted_mu   = torch.zeros(batch_size, self.latent_dim, device=device)

        for m_idx in range(3):
            obs = modality_masks[m_idx]
            if not obs.any():
                continue
            prec = torch.exp(-logvars[m_idx])
            precision_sum[obs] += prec[obs]
            weighted_mu[obs]   += (prec * mus[m_idx])[obs]

        # --- Optional cross-modal translation (activated after warmup) ---
        if translator is not None and epoch >= translation_warmup_epochs:
            trans_mus, trans_logvars, trans_gates = translator(mus, logvars, mask)
            for mu_t, lv_t, gate in zip(trans_mus, trans_logvars, trans_gates):
                prec_t        = torch.exp(-lv_t)
                gated_prec    = gate * prec_t
                precision_sum = precision_sum + gated_prec
                weighted_mu   = weighted_mu   + mu_t * gated_prec

        var_poe    = 1.0 / (precision_sum + 1e-8)
        logvar_poe = torch.log(var_poe + 1e-8)
        mu_poe     = var_poe * weighted_mu
        z          = reparameterise(mu_poe, logvar_poe)

        # --- Decode from PoE latent ---
        feat_dims = [self.n_rna, self.n_mirna, self.n_methyl]
        decoders  = [self.rna_dec, self.mirna_dec, self.methyl_dec]
        names     = ["rna", "mirna", "methyl"]
        recons    = {}

        for m_idx, (dec, name, feat_dim) in enumerate(zip(decoders, names, feat_dims)):
            obs = modality_masks[m_idx]
            if not obs.any():
                continue
            recon_full      = torch.zeros(batch_size, feat_dim, device=device)
            recon_full[obs] = dec(z[obs])
            recons[name]    = recon_full

        z_rna    = reparameterise(mu_rna,    lv_rna)    if rna_mask.any()    else None
        z_mirna  = reparameterise(mu_mirna,  lv_mirna)  if mirna_mask.any()  else None
        z_methyl = reparameterise(mu_methyl, lv_methyl) if methyl_mask.any() else None

        return {
            "mu_poe":     mu_poe,
            "logvar_poe": logvar_poe,
            "z":          z,
            "z_surv":     z,
            "recons":     recons,
            "mus":        mus,
            "logvars":    logvars,
            "z_rna":      z_rna,
            "z_mirna":    z_mirna,
            "z_methyl":   z_methyl,
            "masks":      modality_masks,
        }


def kl_divergence_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL[ N(mu, sigma^2) || N(0, I) ] averaged over batch and latent dims."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def kl_divergence_two_gaussians(
    mu1: torch.Tensor, logvar1: torch.Tensor,
    mu2: torch.Tensor, logvar2: torch.Tensor,
) -> torch.Tensor:
    """KL[ N(mu1, sigma1^2) || N(mu2, sigma2^2) ] averaged over batch and latent dims."""
    return 0.5 * torch.mean(
        logvar2 - logvar1 - 1
        + logvar1.exp() / logvar2.exp()
        + (mu1 - mu2).pow(2) / logvar2.exp()
    )
