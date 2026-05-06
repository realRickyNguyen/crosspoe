# PoE-VAE Baseline — adapted from Wu & Goodman (2018)
# Pure Product-of-Experts VAE with Cox survival head.
# No translation heads, no gate networks, no consistency losses.
# Only: recon + KL + survival.
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
from training.utils import concordance_index, get_beta, move_batch_to_device, set_seed


class VanillaPoE(nn.Module):
    """
    Clean PoE-VAE baseline (Wu & Goodman 2018).
    Shares encoder/decoder architectures with CrossPoE for controlled
    comparison but contains none of the translation infrastructure.
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

        # PoE fusion — precision-weighted product; prior N(0,I) contributes precision=1
        precision_sum = torch.ones(batch_size, self.latent_dim, device=device)
        weighted_mu   = torch.zeros(batch_size, self.latent_dim, device=device)

        for mu, lv, obs in [
            (mu_rna,    lv_rna,    rna_mask),
            (mu_mirna,  lv_mirna,  mirna_mask),
            (mu_methyl, lv_methyl, methyl_mask),
        ]:
            if not obs.any():
                continue
            prec = torch.exp(-lv)
            precision_sum[obs] += prec[obs]
            weighted_mu[obs]   += (prec * mu)[obs]

        var_poe    = 1.0 / (precision_sum + 1e-8)
        logvar_poe = torch.log(var_poe + 1e-8)
        mu_poe     = var_poe * weighted_mu

        z = reparameterise(mu_poe, logvar_poe)

        # Decode observed modalities
        modality_masks = [rna_mask, mirna_mask, methyl_mask]
        decoders       = [self.rna_dec, self.mirna_dec, self.methyl_dec]
        names          = ["rna", "mirna", "methyl"]
        recons         = {}

        for obs, dec, name in zip(modality_masks, decoders, names):
            if not obs.any():
                continue
            feat_dim       = dec.net[-1].out_features
            recon_full     = torch.zeros(batch_size, feat_dim, device=device)
            recon_full[obs] = dec(z[obs])
            recons[name]   = recon_full

        return {
            "mu_poe":     mu_poe,
            "logvar_poe": logvar_poe,
            "z":          z,
            "z_surv":     z,
            "recons":     recons,
            "masks":      modality_masks,
        }


def run_vanilla_poe(cfg, device):
    """
    5-fold CV for PoE-VAE baseline (Wu & Goodman 2018).
    Loss: survival + recon + KL only. No translation or consistency.
    """
    set_seed(cfg["seed"])

    n_latent = cfg.get("n_latent", 48)
    n_rna    = MultiOmicsDataset._rna_data.shape[1]
    n_mirna  = MultiOmicsDataset._mirna_data.shape[1]
    n_methyl = MultiOmicsDataset._methyl_data.shape[1]

    pfi_event_all = MultiOmicsDataset._pfi_event.numpy()
    strat_labels  = np.clip(pfi_event_all, 0, 1)
    skf           = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True,
                                    random_state=cfg["seed"])
    fold_results  = []

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.arange(len(strat_labels)), strat_labels)):

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1} / {cfg['n_folds']}  [VanillaPoE]")
        print(f"  Train: {len(train_idx)}  |  Val: {len(val_idx)}")
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
        train_loader = DataLoader(
            train_dataset, batch_size=cfg["batch_size"],
            shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg["batch_size"] * 2,
            shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True,
        )

        set_seed(cfg["seed"] + fold_idx)
        model         = VanillaPoE(latent_dim=n_latent, n_rna=n_rna,
                                   n_mirna=n_mirna, n_methyl=n_methyl).to(device)
        survival_head = SurvivalHead(n_latent).to(device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(survival_head.parameters()),
            lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
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
            loss_accum = {"total": 0.0, "survival": 0.0, "recon": 0.0, "kl": 0.0}
            n_batches  = 0
            beta       = get_beta(epoch, cfg["kl_warmup_epochs"])

            for batch in train_loader:
                batch   = move_batch_to_device(batch, device)
                optimizer.zero_grad()
                outputs = model(batch)

                risk_scores   = survival_head(outputs["z_surv"])
                loss_survival = cox_partial_likelihood_loss(
                    risk_scores, batch["pfi_time"], batch["pfi_event"]
                )

                loss_recon = torch.tensor(0.0, device=device, requires_grad=True)
                for m_idx, m_name in enumerate(["rna", "mirna", "methyl"]):
                    if m_name not in outputs["recons"]:
                        continue
                    obs = batch["mask"][:, m_idx]
                    if not obs.any() or batch[m_name] is None:
                        continue
                    loss_recon = loss_recon + F.mse_loss(
                        outputs["recons"][m_name][obs], batch[m_name][obs],
                    )

                loss_kl = -0.5 * torch.mean(
                    1 + outputs["logvar_poe"] - outputs["mu_poe"].pow(2) - outputs["logvar_poe"].exp()
                )

                loss = (
                      cfg["lambda_survival"] * loss_survival
                    + cfg["lambda_recon"]    * loss_recon
                    + beta                   * loss_kl
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(survival_head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

                loss_accum["total"]    += loss.item()
                loss_accum["survival"] += loss_survival.item()
                loss_accum["recon"]    += loss_recon.item()
                loss_accum["kl"]       += loss_kl.item()
                n_batches += 1

            scheduler.step()
            train_metrics = {k: v / n_batches for k, v in loss_accum.items()}

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

            print(
                f"  Epoch {epoch:3d}/{cfg['n_epochs']} | beta={beta:.2f} | "
                f"Train: {train_metrics['total']:.4f} "
                f"(surv={train_metrics['survival']:.3f} "
                f"kl={train_metrics['kl']:.3f} "
                f"recon={train_metrics['recon']:.3f}) | "
                f"Val C-index: {c_index:.4f}"
            )

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
                              f"(best epoch {best_epoch}, C-index {best_score:.4f})")
                        break

        model.load_state_dict({k: v.to(device) for k, v in best_state["model"].items()})
        survival_head.load_state_dict(
            {k: v.to(device) for k, v in best_state["survival"].items()})

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

        final_val_ci = float("nan")
        if all_risk:
            risk_np  = torch.cat(all_risk).squeeze().numpy()
            time_np  = torch.cat(all_time).numpy()
            event_np = torch.cat(all_event).numpy()
            if event_np.sum() > 0:
                final_val_ci = concordance_index(time_np, risk_np, event_np)

        print(f"\n  Fold {fold_idx+1} best epoch : {best_epoch}")
        print(f"  Final val C-index          : {final_val_ci:.4f}")

        fold_results.append({
            "fold":         fold_idx + 1,
            "best_epoch":   best_epoch,
            "best_c_index": best_c_index,
            "val_metrics":  {"c_index": final_val_ci},
            "model_state":  best_state,
            "scalers":      {"rna": rna_scaler, "mirna": mirna_scaler, "methyl": methyl_scaler},
        })

    cidxs = [fr["val_metrics"]["c_index"] for fr in fold_results]
    print(f"\n{'='*60}")
    print("VANILLA PoE CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"C-index: {np.nanmean(cidxs):.4f} ± {np.nanstd(cidxs):.4f}")
    for fr in fold_results:
        print(f"  Fold {fr['fold']}: {fr['best_c_index']:.4f} (best epoch {fr['best_epoch']})")

    return fold_results


def run_mcar_vanilla_poe(fold_results, cfg, device,
                         missing_rates=None, modalities=None):
    """MCAR evaluation for VanillaPoE baseline (no translator)."""
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

        model         = VanillaPoE(latent_dim=n_latent, n_rna=n_rna,
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
    print("MCAR ROBUSTNESS — VanillaPoE")
    print(f"{'='*60}")
    for mod in modalities:
        print(f"\n{mod}:")
        for rate in missing_rates:
            vals = [v for v in results[mod][rate] if not np.isnan(v)]
            mean = np.mean(vals) if vals else float("nan")
            std  = np.std(vals)  if vals else float("nan")
            print(f"  rate={rate:.1f}: C-index = {mean:.4f} ± {std:.4f}")

    return results
