# MVAE Baseline — Mixture of Experts fusion
# Shi et al (2019)
# Same encoders, decoders, survival head as CrossPoE.
# Only difference: MixtureOfExperts fusion instead of ProductOfExperts.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset import MultiOmicsDataset, collate_fn
from data.mcar import MCARDataset
from data.preprocessing import compute_scaler
from models.encoders import MethylEncoder, MIRNAEncoder, RNAEncoder
from models.decoders import MethylDecoder, MIRNADecoder, RNADecoder
from models.poe import reparameterise
from models.survival import SurvivalHead, cox_partial_likelihood_loss
from training.utils import concordance_index, move_batch_to_device, set_seed


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts posterior (Shi et al 2019).
    Joint posterior = uniform mixture of individual modality posteriors.
    Var[mixture] approximated via law of total variance.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, mu_list, logvar_list):
        mus    = torch.stack(mu_list,    dim=0)  # (K, B, D)
        logvars = torch.stack(logvar_list, dim=0)

        mu_moe = mus.mean(dim=0)

        vars_      = logvars.exp()
        e_var      = vars_.mean(dim=0)
        var_e      = mus.var(dim=0, unbiased=False)
        var_moe    = e_var + var_e
        logvar_moe = torch.log(var_moe + 1e-8)

        return mu_moe, logvar_moe


class MVAE(nn.Module):
    """
    Multimodal VAE with Mixture-of-Experts fusion.
    Architecturally identical to CrossPoE except MoE replaces PoE.
    No translation heads, consistency, or cycle losses.
    """

    def __init__(self, latent_dim: int = 48, n_rna: int = None, n_mirna: int = None,
                 n_methyl: int = None):
        super().__init__()
        self.latent_dim = latent_dim

        n_rna    = n_rna    or MultiOmicsDataset._rna_data.shape[1]
        n_mirna  = n_mirna  or MultiOmicsDataset._mirna_data.shape[1]
        n_methyl = n_methyl or MultiOmicsDataset._methyl_data.shape[1]

        self.rna_enc    = RNAEncoder(in_dim=n_rna,    latent_dim=latent_dim)
        self.mirna_enc  = MIRNAEncoder(in_dim=n_mirna, latent_dim=latent_dim)
        self.methyl_enc = MethylEncoder(in_dim=n_methyl, latent_dim=latent_dim)

        self.rna_dec    = RNADecoder(latent_dim=latent_dim, out_dim=n_rna)
        self.mirna_dec  = MIRNADecoder(latent_dim=latent_dim, out_dim=n_mirna)
        self.methyl_dec = MethylDecoder(latent_dim=latent_dim, out_dim=n_methyl)

        self.moe = MixtureOfExperts(latent_dim=latent_dim)

    def forward(self, batch):
        batch_size = batch["mask"].shape[0]
        device     = batch["mask"].device
        mask       = batch["mask"]

        rna_mask    = mask[:, 0]
        mirna_mask  = mask[:, 1]
        methyl_mask = mask[:, 2]

        mu_rna    = torch.zeros(batch_size, self.latent_dim, device=device)
        lv_rna    = torch.zeros(batch_size, self.latent_dim, device=device)
        mu_mirna  = torch.zeros(batch_size, self.latent_dim, device=device)
        lv_mirna  = torch.zeros(batch_size, self.latent_dim, device=device)
        mu_methyl = torch.zeros(batch_size, self.latent_dim, device=device)
        lv_methyl = torch.zeros(batch_size, self.latent_dim, device=device)

        if rna_mask.any() and batch["rna"] is not None:
            mu_rna[rna_mask],    lv_rna[rna_mask]    = self.rna_enc(batch["rna"][rna_mask])
        if mirna_mask.any() and batch["mirna"] is not None:
            mu_mirna[mirna_mask], lv_mirna[mirna_mask] = self.mirna_enc(batch["mirna"][mirna_mask])
        if methyl_mask.any() and batch["methyl"] is not None:
            mu_methyl[methyl_mask], lv_methyl[methyl_mask] = self.methyl_enc(batch["methyl"][methyl_mask])

        modality_masks = [rna_mask, mirna_mask, methyl_mask]
        all_mus        = [mu_rna, mu_mirna, mu_methyl]
        all_logvars    = [lv_rna, lv_mirna, lv_methyl]

        mu_moe     = torch.zeros(batch_size, self.latent_dim, device=device)
        logvar_moe = torch.zeros(batch_size, self.latent_dim, device=device)

        # Group by unique observed pattern for efficiency
        patterns        = mask.cpu().numpy()
        unique_patterns = np.unique(patterns, axis=0)

        for pat in unique_patterns:
            pat_tensor  = torch.tensor(pat, dtype=torch.bool, device=device)
            sample_mask = (mask == pat_tensor).all(dim=1)
            if not sample_mask.any():
                continue
            obs_indices = [i for i in range(3) if pat[i]]
            if not obs_indices:
                continue
            mu_obs = [all_mus[i][sample_mask]    for i in obs_indices]
            lv_obs = [all_logvars[i][sample_mask] for i in obs_indices]
            if len(obs_indices) == 1:
                mu_moe[sample_mask]     = mu_obs[0]
                logvar_moe[sample_mask] = lv_obs[0]
            else:
                mu_p, lv_p = self.moe(mu_obs, lv_obs)
                mu_moe[sample_mask]     = mu_p
                logvar_moe[sample_mask] = lv_p

        z = reparameterise(mu_moe, logvar_moe)

        recons   = {}
        decoders = [self.rna_dec, self.mirna_dec, self.methyl_dec]
        names    = ["rna", "mirna", "methyl"]
        for m_idx, (dec, name) in enumerate(zip(decoders, names)):
            obs = modality_masks[m_idx]
            if not obs.any():
                continue
            feat_dim       = dec.net[-1].out_features
            recon_full     = torch.zeros(batch_size, feat_dim, device=device)
            recon_full[obs] = dec(z[obs])
            recons[name]   = recon_full

        return {
            "mu_moe":     mu_moe,
            "logvar_moe": logvar_moe,
            "z_surv":     z,
            "recons":     recons,
            "mus":        all_mus,
            "logvars":    all_logvars,
            "mask":       mask,
        }


def compute_loss_mvae(outputs, batch, beta, lambda_recon, lambda_survival, survival_head):
    """MVAE loss: reconstruction + KL + survival. No translation or consistency."""
    device = batch["mask"].device
    mask   = batch["mask"]

    n_rna    = MultiOmicsDataset._rna_data.shape[1]
    n_mirna  = MultiOmicsDataset._mirna_data.shape[1]
    n_methyl = MultiOmicsDataset._methyl_data.shape[1]
    recon_weights = {
        "rna":    1.0,
        "mirna":  n_rna / n_mirna,
        "methyl": n_rna / n_methyl,
    }

    loss_recon = torch.tensor(0.0, device=device, requires_grad=True)
    for m_idx, m_name in enumerate(["rna", "mirna", "methyl"]):
        if m_name not in outputs["recons"]:
            continue
        obs = mask[:, m_idx]
        if not obs.any() or batch[m_name] is None:
            continue
        loss_recon = loss_recon + recon_weights[m_name] * F.mse_loss(
            outputs["recons"][m_name][obs], batch[m_name][obs]
        )

    mu     = outputs["mu_moe"]
    logvar = outputs["logvar_moe"]
    loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss_surv  = torch.tensor(0.0, device=device, requires_grad=True)
    valid_surv = batch["pfi_event"] >= 0
    if valid_surv.sum() >= 2 and batch["pfi_event"][valid_surv].sum() >= 1:
        risk = survival_head(outputs["z_surv"])
        loss_surv = cox_partial_likelihood_loss(
            risk[valid_surv], batch["pfi_time"][valid_surv], batch["pfi_event"][valid_surv]
        )

    total = (lambda_recon * loss_recon + beta * loss_kl + lambda_survival * loss_surv)

    return total, {
        "total":    total.item(),
        "recon":    loss_recon.item(),
        "kl":       loss_kl.item(),
        "survival": loss_surv.item(),
    }


def run_mvae(cfg, device):
    """5-fold CV for MVAE baseline. Same splits and scalers as CrossPoE."""
    set_seed(cfg["seed"])

    n_latent = cfg.get("n_latent", 48)
    n_rna    = MultiOmicsDataset._rna_data.shape[1]
    n_mirna  = MultiOmicsDataset._mirna_data.shape[1]
    n_methyl = MultiOmicsDataset._methyl_data.shape[1]

    pfi_event_all = MultiOmicsDataset._pfi_event.numpy()
    strat_labels  = np.clip(pfi_event_all, 0, 1)
    skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True,
                          random_state=cfg["seed"])
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.arange(len(strat_labels)), strat_labels)):

        print(f"\n{'='*60}")
        print(f"MVAE FOLD {fold_idx+1} / {cfg['n_folds']}")
        print(f"{'='*60}")

        rna_scaler    = compute_scaler(MultiOmicsDataset._rna_data[train_idx][MultiOmicsDataset._rna_mask[train_idx]])
        mirna_scaler  = compute_scaler(MultiOmicsDataset._mirna_data[train_idx][MultiOmicsDataset._mirna_mask[train_idx]])
        methyl_scaler = compute_scaler(MultiOmicsDataset._methyl_data[train_idx][MultiOmicsDataset._methyl_mask[train_idx]])

        train_dataset = MultiOmicsDataset(
            indices=train_idx, rna_scaler=rna_scaler,
            mirna_scaler=mirna_scaler, methyl_scaler=methyl_scaler,
            dropout_probs={"rna": 0.0, "mirna": 0.0, "methyl": 0.0},
        )
        val_dataset = MultiOmicsDataset(
            indices=val_idx, rna_scaler=rna_scaler,
            mirna_scaler=mirna_scaler, methyl_scaler=methyl_scaler,
            dropout_probs=None,
        )
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                                  shuffle=True, collate_fn=collate_fn,
                                  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_dataset, batch_size=cfg["batch_size"] * 2,
                                  shuffle=False, collate_fn=collate_fn,
                                  num_workers=0, pin_memory=True)

        set_seed(cfg["seed"] + fold_idx)
        model         = MVAE(latent_dim=n_latent, n_rna=n_rna,
                             n_mirna=n_mirna, n_methyl=n_methyl).to(device)
        survival_head = SurvivalHead(n_latent).to(device)
        optimizer     = torch.optim.AdamW(
            list(model.parameters()) + list(survival_head.parameters()),
            lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["n_epochs"], eta_min=1e-5)

        best_score   = -np.inf
        best_c_index = float("nan")
        best_epoch   = 0
        patience_ctr = 0
        best_state   = None

        for epoch in range(1, cfg["n_epochs"] + 1):
            model.train()
            survival_head.train()
            beta = min(1.0, epoch / cfg["kl_warmup_epochs"])

            for batch in train_loader:
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad()
                outputs = model(batch)
                loss, _ = compute_loss_mvae(
                    outputs, batch, beta,
                    lambda_recon=cfg["lambda_recon"],
                    lambda_survival=cfg["lambda_survival"],
                    survival_head=survival_head,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(survival_head.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
            scheduler.step()

            model.eval()
            survival_head.eval()
            all_risk, all_time, all_event = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch   = move_batch_to_device(batch, device)
                    outputs = model(batch)
                    risk    = survival_head(outputs["z_surv"])
                    valid   = batch["pfi_event"] >= 0
                    if valid.any():
                        all_risk.append(risk[valid].cpu())
                        all_time.append(batch["pfi_time"][valid].cpu())
                        all_event.append(batch["pfi_event"][valid].cpu())

            c_index = float("nan")
            if all_risk:
                risk_np  = torch.cat(all_risk).squeeze().numpy()
                time_np  = torch.cat(all_time).numpy()
                event_np = torch.cat(all_event).numpy()
                if event_np.sum() > 0:
                    c_index = concordance_index(time_np, risk_np, event_np)

            print(f"  Epoch {epoch:3d} | beta={beta:.2f} | Val C-index: {c_index:.4f}")

            if epoch >= cfg.get("min_epochs", 1):
                score = c_index if not np.isnan(c_index) else -np.inf
                if score > best_score:
                    best_score   = score
                    best_c_index = c_index
                    best_epoch   = epoch
                    patience_ctr = 0
                    best_state   = {
                        "model":    {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        "survival": {k: v.cpu().clone() for k, v in survival_head.state_dict().items()},
                    }
                else:
                    patience_ctr += 1
                    if patience_ctr >= cfg["patience"]:
                        print(f"  Early stopping at epoch {epoch} "
                              f"(best {best_epoch}, C-index {best_score:.4f})")
                        break

        fold_results.append({
            "fold":         fold_idx + 1,
            "best_epoch":   best_epoch,
            "best_c_index": best_c_index,
            "val_metrics":  {"c_index": best_c_index},
            "model_state":  best_state,
            "scalers":      {"rna": rna_scaler, "mirna": mirna_scaler, "methyl": methyl_scaler},
        })
        print(f"\n  Fold {fold_idx+1} best epoch: {best_epoch}, C-index: {best_c_index:.4f}")

    cidxs = [fr["best_c_index"] for fr in fold_results]
    print(f"\n{'='*60}")
    print("MVAE CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"C-index: {np.nanmean(cidxs):.4f} ± {np.nanstd(cidxs):.4f}")
    for fr in fold_results:
        print(f"  Fold {fr['fold']}: {fr['best_c_index']:.4f} (best epoch {fr['best_epoch']})")

    return fold_results


def run_mcar_mvae(fold_results, cfg, device, missing_rates=None, modalities=None):
    """MCAR evaluation for MVAE baseline (no translator)."""
    if missing_rates is None:
        missing_rates = [0.0, 0.3, 0.5, 0.7, 0.9]
    if modalities is None:
        modalities = ["rna", "mirna", "methyl"]

    n_latent = cfg.get("n_latent", 48)
    n_rna    = MultiOmicsDataset._rna_data.shape[1]
    n_mirna  = MultiOmicsDataset._mirna_data.shape[1]
    n_methyl = MultiOmicsDataset._methyl_data.shape[1]

    pfi_event_all = MultiOmicsDataset._pfi_event.numpy()
    strat_labels  = np.clip(pfi_event_all, 0, 1)
    skf    = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True,
                             random_state=cfg["seed"])
    splits = list(skf.split(np.arange(len(strat_labels)), strat_labels))

    results = {mod: {r: [] for r in missing_rates} for mod in modalities}

    for fold_idx, fr in enumerate(fold_results):
        _, val_idx = splits[fold_idx]
        print(f"\nFold {fold_idx + 1} / {len(fold_results)}")

        model         = MVAE(latent_dim=n_latent, n_rna=n_rna,
                             n_mirna=n_mirna, n_methyl=n_methyl).to(device)
        survival_head = SurvivalHead(n_latent).to(device)
        model.load_state_dict({k: v.to(device) for k, v in fr["model_state"]["model"].items()})
        survival_head.load_state_dict({k: v.to(device) for k, v in fr["model_state"]["survival"].items()})
        model.eval()
        survival_head.eval()

        val_dataset = MultiOmicsDataset(
            indices=val_idx,
            rna_scaler=fr["scalers"]["rna"],
            mirna_scaler=fr["scalers"]["mirna"],
            methyl_scaler=fr["scalers"]["methyl"],
            dropout_probs=None,
        )

        for mod in modalities:
            for rate in missing_rates:
                mcar_ds = MCARDataset(val_dataset, mod, rate)
                loader  = DataLoader(mcar_ds, batch_size=cfg["batch_size"] * 2,
                                     shuffle=False, collate_fn=collate_fn, num_workers=0)

                all_risk, all_time, all_event = [], [], []
                with torch.no_grad():
                    for batch in loader:
                        batch   = move_batch_to_device(batch, device)
                        outputs = model(batch)
                        risk    = survival_head(outputs["z_surv"])
                        valid   = batch["pfi_event"] >= 0
                        if valid.any():
                            all_risk.append(risk[valid].cpu())
                            all_time.append(batch["pfi_time"][valid].cpu())
                            all_event.append(batch["pfi_event"][valid].cpu())

                c_index = float("nan")
                if all_risk:
                    risk_np  = torch.cat(all_risk).squeeze().numpy()
                    time_np  = torch.cat(all_time).numpy()
                    event_np = torch.cat(all_event).numpy()
                    if event_np.sum() > 0:
                        c_index = concordance_index(time_np, risk_np, event_np)

                results[mod][rate].append(c_index)
                print(f"  {mod:>6} @{rate:.1f} | C-index={c_index:.4f}")

    print(f"\n{'='*60}")
    print("MCAR ROBUSTNESS — MVAE")
    print(f"{'='*60}")
    for mod in modalities:
        print(f"\n{mod}:")
        for rate in missing_rates:
            vals = [v for v in results[mod][rate] if not np.isnan(v)]
            mean = np.mean(vals) if vals else float("nan")
            std  = np.std(vals)  if vals else float("nan")
            print(f"  rate={rate:.1f}: C-index = {mean:.4f} ± {std:.4f}")

    return results
