import torch
import torch.nn.functional as F

from models.poe import reparameterise
from models.survival import cox_partial_likelihood_loss


def translation_consistency_loss(mus, logvars, translator, mask) -> torch.Tensor:
    """
    Wasserstein-2 distance between translated and real posteriors for
    fully-observed pairs.

    Using W2 instead of asymmetric KL avoids mode-seeking pathologies:
        W2^2 = ||mu_pseudo - mu_real||^2 + ||sigma_pseudo - sigma_real||^2
    """
    total_loss = 0.0
    n_pairs    = 0

    for src in range(3):
        for tgt in range(3):
            if src == tgt:
                continue
            both_observed = mask[:, src] & mask[:, tgt]
            if not both_observed.any():
                continue

            key = f"{src}_to_{tgt}"
            mu_pseudo, logvar_pseudo = translator.translation_heads[key](
                mus[src], logvars[src]
            )
            mu_real     = mus[tgt].detach()
            logvar_real = logvars[tgt].detach()

            sigma_pseudo = torch.exp(0.5 * logvar_pseudo)
            sigma_real   = torch.exp(0.5 * logvar_real)

            mean_term  = (mu_pseudo - mu_real).pow(2)
            var_term   = (sigma_pseudo - sigma_real).pow(2)
            w2         = (mean_term + var_term)[both_observed].mean()
            total_loss = total_loss + w2
            n_pairs   += 1

    if n_pairs > 0:
        return total_loss / n_pairs
    return torch.tensor(0.0, device=mus[0].device, requires_grad=True)


def cycle_consistency_loss(mus, logvars, translator, mask) -> torch.Tensor:
    """
    Cycle consistency: A -> pseudo_B -> pseudo_A ≈ A.
    Applied to all 6 directions using fully-observed pairs.
    Prevents translators from collapsing to mean-seeking solutions.
    """
    total_loss = 0.0
    n_cycles   = 0

    for src in range(3):
        for tgt in range(3):
            if src == tgt:
                continue
            both_observed = mask[:, src] & mask[:, tgt]
            if not both_observed.any():
                continue

            key_fwd = f"{src}_to_{tgt}"
            key_bwd = f"{tgt}_to_{src}"

            mu_pseudo_tgt, logvar_pseudo_tgt = translator.translation_heads[key_fwd](
                mus[src][both_observed], logvars[src][both_observed]
            )
            mu_recon_src, _ = translator.translation_heads[key_bwd](
                mu_pseudo_tgt, logvar_pseudo_tgt
            )
            mu_src_real = mus[src][both_observed].detach()
            total_loss  = total_loss + F.mse_loss(mu_recon_src, mu_src_real)
            n_cycles   += 1

    if n_cycles > 0:
        return total_loss / n_cycles
    return torch.tensor(0.0, device=mus[0].device, requires_grad=True)


def compute_loss(
    outputs,
    batch,
    beta: float,
    lambda_recon: float,
    lambda_consist: float,
    lambda_survival: float,
    lambda_translation: float,
    lambda_cycle: float,
    survival_head,
    translator,
    epoch: int,
    translation_warmup_epochs: int,
    model=None,
    lambda_surv_trans: float = 0.0,
):
    """
    Composite CrossPoE loss:
        survival + recon + kl + consist + translation + cycle + surv_trans

    Args:
        outputs                   : dict from CrossPoE.forward()
        batch                     : collated batch dict
        beta                      : KL annealing coefficient
        lambda_*                  : per-term loss weights
        survival_head             : SurvivalHead module
        translator                : CrossModalTranslator or None
        epoch                     : current epoch (for translation warmup guard)
        translation_warmup_epochs : epoch at which translation losses activate
        model                     : CrossPoE instance (used for feature dims)
        lambda_surv_trans         : weight for translation survival-preservation loss

    Returns:
        (total_loss, loss_dict)
    """
    device = batch["mask"].device
    mask   = batch["mask"]

    n_rna    = model.n_rna    if model is not None else 4652
    n_mirna  = model.n_mirna  if model is not None else 524
    n_methyl = model.n_methyl if model is not None else 37482

    RECON_WEIGHTS = {
        "rna":    1.0,
        "mirna":  n_rna / n_mirna,
        "methyl": n_rna / n_methyl,
    }

    # -- Reconstruction loss --------------------------------------------------
    loss_recon    = torch.tensor(0.0, device=device, requires_grad=True)
    recon_targets = {"rna": batch["rna"], "mirna": batch["mirna"], "methyl": batch["methyl"]}

    for m_idx, m_name in enumerate(["rna", "mirna", "methyl"]):
        if m_name not in outputs["recons"]:
            continue
        obs = mask[:, m_idx]
        if not obs.any() or recon_targets[m_name] is None:
            continue
        loss_recon = loss_recon + RECON_WEIGHTS[m_name] * F.mse_loss(
            outputs["recons"][m_name][obs], recon_targets[m_name][obs]
        )

    if model is not None:
        _z_indiv  = [outputs.get("z_rna"), outputs.get("z_mirna"), outputs.get("z_methyl")]
        _decoders = [model.rna_dec, model.mirna_dec, model.methyl_dec]
        for m_idx, (z_i, dec) in enumerate(zip(_z_indiv, _decoders)):
            m_name = ["rna", "mirna", "methyl"][m_idx]
            if z_i is None or recon_targets[m_name] is None:
                continue
            if not mask[:, m_idx].any():
                continue
            _ = dec(z_i)
            
    # -- KL divergence --------------------------------------------------------
    mu_poe     = outputs["mu_poe"]
    logvar_poe = outputs["logvar_poe"]
    loss_kl    = -0.5 * torch.mean(1 + logvar_poe - mu_poe.pow(2) - logvar_poe.exp())

    # -- Consistency: per-modality KL to PoE posterior ------------------------
    loss_consist = torch.tensor(0.0, device=device, requires_grad=True)
    mus          = outputs["mus"]
    logvars      = outputs["logvars"]
    mu_poe_d     = mu_poe.detach()
    lv_poe_d     = logvar_poe.detach()
    n_consist    = 0

    for m_idx in range(3):
        obs = mask[:, m_idx]
        if obs.sum() < 2:
            continue
        kl = 0.5 * (
            lv_poe_d[obs] - logvars[m_idx][obs]
            + (logvars[m_idx][obs].exp() + (mus[m_idx][obs] - mu_poe_d[obs]).pow(2))
            / (lv_poe_d[obs].exp() + 1e-8) - 1.0
        )
        loss_consist = loss_consist + kl.mean()
        n_consist   += 1

    if n_consist > 0:
        loss_consist = loss_consist / n_consist

    # -- Survival (Cox PH) ----------------------------------------------------
    risk_scores   = survival_head(outputs["z_surv"])
    loss_survival = cox_partial_likelihood_loss(
        risk_scores, batch["pfi_time"], batch["pfi_event"]
    )

    # -- Translation losses (only after warmup) --------------------------------
    use_trans = (
        translator is not None
        and hasattr(translator, "translation_heads")
        and epoch >= translation_warmup_epochs
    )

    if use_trans:
        loss_translation = translation_consistency_loss(mus, logvars, translator, mask)
        loss_cycle       = cycle_consistency_loss(mus, logvars, translator, mask)

        loss_surv_trans = torch.tensor(0.0, device=device)
        n_surv_trans    = 0
        for src in range(3):
            for tgt in range(3):
                if src == tgt:
                    continue
                both_observed = mask[:, src] & mask[:, tgt]
                if both_observed.sum() < 2:
                    continue
                key = f"{src}_to_{tgt}"
                mu_pseudo, logvar_pseudo = translator.translation_heads[key](
                    mus[src][both_observed], logvars[src][both_observed]
                )
                z_pseudo = reparameterise(mu_pseudo, logvar_pseudo)
                z_real   = reparameterise(
                    mus[tgt][both_observed].detach(),
                    logvars[tgt][both_observed].detach(),
                )
                with torch.no_grad():
                    risk_real = survival_head(z_real)
                risk_pseudo = F.linear(
                    z_pseudo,
                    survival_head.risk.weight.detach(),
                    survival_head.risk.bias.detach(),
                )
                loss_surv_trans = loss_surv_trans + F.mse_loss(risk_pseudo, risk_real)
                n_surv_trans   += 1
        if n_surv_trans > 0:
            loss_surv_trans = loss_surv_trans / n_surv_trans
    else:
        loss_translation = torch.tensor(0.0, device=device)
        loss_cycle       = torch.tensor(0.0, device=device)
        loss_surv_trans  = torch.tensor(0.0, device=device)

    # -- Combine --------------------------------------------------------------
    total = (
          lambda_survival    * loss_survival
        + lambda_recon       * loss_recon
        + beta               * loss_kl
        + lambda_consist     * loss_consist
        + lambda_translation * loss_translation
        + lambda_cycle       * loss_cycle
        + lambda_surv_trans  * loss_surv_trans
    )

    loss_dict = {
        "total":       total.item(),
        "survival":    loss_survival.item(),
        "recon":       loss_recon.item(),
        "kl":          loss_kl.item(),
        "consist":     loss_consist.item(),
        "translation": loss_translation.item(),
        "cycle":       loss_cycle.item(),
        "surv_trans":  loss_surv_trans.item(),
    }
    return total, loss_dict
