# HEALNet-OmicsLite Baseline
# PyTorch-only HEALNet-style molecular adaptation for CrossPoE benchmarking.
#
# This is NOT the original WSI + omics HEALNet experiment.
# It is a molecular-only HEALNet-style fusion baseline that avoids
# dependency conflicts with the official HEALNet repo.
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset import MultiOmicsDataset, collate_fn
from data.mcar import MCARDataset
from data.preprocessing import compute_scaler
from models.survival import SurvivalHead, cox_partial_likelihood_loss
from training.utils import concordance_index, move_batch_to_device, set_seed


class HEALNetLiteCrossAttentionBlock(nn.Module):
    """
    Latent-query cross-attention block.
    Learned latent tokens attend into one observed omics modality.
    An optional compress_dim reduces high-dimensional inputs (e.g. methylation).
    """

    def __init__(self, latent_dim: int, input_dim: int, n_heads: int = 4,
                 dropout: float = 0.1, ff_mult: int = 4, compress_dim: int = None):
        super().__init__()

        if compress_dim is not None:
            self.compress = nn.Sequential(
                nn.Linear(input_dim, compress_dim),
                nn.LayerNorm(compress_dim),
                nn.GELU(),
            )
            proj_in = compress_dim
        else:
            self.compress = None
            proj_in = input_dim

        self.input_proj   = nn.Linear(proj_in, latent_dim)
        self.norm_latent  = nn.LayerNorm(latent_dim)
        self.norm_context = nn.LayerNorm(latent_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * ff_mult, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, latents, x):
        """Args: latents [B, L, D], x [B, F]. Returns updated latents [B, L, D]."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if self.compress is not None:
            x = self.compress(x)
        context = self.input_proj(x)

        q  = self.norm_latent(latents)
        kv = self.norm_context(context)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)

        latents = latents + attn_out
        latents = latents + self.ff(latents)
        return latents


class HEALNetLiteLatentSelfAttentionBlock(nn.Module):
    """Self-attention over the learned latent bottleneck tokens."""

    def __init__(self, latent_dim: int, n_heads: int = 4, dropout: float = 0.1,
                 ff_mult: int = 4):
        super().__init__()
        self.norm     = nn.LayerNorm(latent_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * ff_mult, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, latents):
        q = self.norm(latents)
        attn_out, _ = self.self_attn(query=q, key=q, value=q, need_weights=False)
        latents = latents + attn_out
        latents = latents + self.ff(latents)
        return latents


class HEALNetOmicsLite(nn.Module):
    """
    Molecular-only HEALNet-style adaptation.

    Missing modalities are handled by skipping their cross-attention update.
    Samples are grouped by unique observed-modality pattern to avoid treating
    zero-filled collate_fn tensors as real observations.
    """

    def __init__(self, latent_dim: int = 48, n_latents: int = 16, depth: int = 2,
                 n_heads: int = 4, dropout: float = 0.1, ff_mult: int = 4,
                 n_rna: int = None, n_mirna: int = None, n_methyl: int = None):
        super().__init__()

        self.latent_dim     = latent_dim
        self.n_latents      = n_latents
        self.modality_order = ["rna", "mirna", "methyl"]

        n_rna    = n_rna    or MultiOmicsDataset._rna_data.shape[1]
        n_mirna  = n_mirna  or MultiOmicsDataset._mirna_data.shape[1]
        n_methyl = n_methyl or MultiOmicsDataset._methyl_data.shape[1]

        self.latents = nn.Parameter(torch.randn(n_latents, latent_dim) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            cross_blocks = nn.ModuleDict({
                "rna": HEALNetLiteCrossAttentionBlock(
                    latent_dim=latent_dim, input_dim=n_rna,
                    n_heads=n_heads, dropout=dropout, ff_mult=ff_mult,
                    compress_dim=None,
                ),
                "mirna": HEALNetLiteCrossAttentionBlock(
                    latent_dim=latent_dim, input_dim=n_mirna,
                    n_heads=n_heads, dropout=dropout, ff_mult=ff_mult,
                    compress_dim=None,
                ),
                "methyl": HEALNetLiteCrossAttentionBlock(
                    latent_dim=latent_dim, input_dim=n_methyl,
                    n_heads=n_heads, dropout=dropout, ff_mult=ff_mult,
                    compress_dim=256,
                ),
            })
            latent_block = HEALNetLiteLatentSelfAttentionBlock(
                latent_dim=latent_dim, n_heads=n_heads, dropout=dropout, ff_mult=ff_mult,
            )
            self.layers.append(nn.ModuleDict({"cross": cross_blocks, "latent": latent_block}))

        self.out_norm = nn.LayerNorm(latent_dim)
        self.no_modality_embedding = nn.Parameter(torch.zeros(latent_dim))

    def _forward_one_pattern(self, batch, sample_idx, pattern):
        b = int(sample_idx.sum().item())
        latents = self.latents.unsqueeze(0).expand(b, -1, -1).clone()

        for layer in self.layers:
            for m_idx, mod in enumerate(self.modality_order):
                if bool(pattern[m_idx].item()):
                    x = batch[mod][sample_idx]
                    latents = layer["cross"][mod](latents, x)
            latents = layer["latent"](latents)

        z = latents.mean(dim=1)
        z = self.out_norm(z)
        return z

    def forward(self, batch):
        mask   = batch["mask"]
        device = mask.device
        b      = mask.shape[0]

        z_all = torch.zeros(b, self.latent_dim, device=device, dtype=self.latents.dtype)
        unique_patterns = torch.unique(mask, dim=0)

        for pattern in unique_patterns:
            sample_idx = (mask == pattern.unsqueeze(0)).all(dim=1)
            if not sample_idx.any():
                continue
            if not pattern.any():
                n_empty = int(sample_idx.sum().item())
                z_all[sample_idx] = self.no_modality_embedding.unsqueeze(0).expand(n_empty, -1)
                continue
            z_all[sample_idx] = self._forward_one_pattern(batch, sample_idx, pattern)

        return {"z_surv": z_all}


def compute_loss_healnet_omics_lite(outputs, batch, survival_head):
    """HEALNet-OmicsLite uses only Cox partial likelihood (no recon, KL, or consistency)."""
    risk = survival_head(outputs["z_surv"])
    loss_surv = cox_partial_likelihood_loss(risk, batch["pfi_time"], batch["pfi_event"])
    val = float(loss_surv.detach().cpu())
    return loss_surv, {"total": val, "survival": val}


def run_healnet_omics_lite(cfg, device):
    """
    Train HEALNet-OmicsLite under the same CrossPoE CV protocol.
    Deterministic HEALNet-style attention fusion; no VAE, PoE, or translation.
    Config keys: healnet_n_latents, healnet_depth, healnet_n_heads,
                 healnet_dropout, healnet_ff_mult (all optional).
    """
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
        print(f"HEALNet-OmicsLite FOLD {fold_idx + 1} / {cfg['n_folds']}")
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
        model = HEALNetOmicsLite(
            latent_dim=n_latent,
            n_latents=cfg.get("healnet_n_latents", 16),
            depth=cfg.get("healnet_depth", 2),
            n_heads=cfg.get("healnet_n_heads", 4),
            dropout=cfg.get("healnet_dropout", 0.1),
            ff_mult=cfg.get("healnet_ff_mult", 4),
            n_rna=n_rna, n_mirna=n_mirna, n_methyl=n_methyl,
        ).to(device)

        survival_head = SurvivalHead(n_latent).to(device)
        optimizer     = torch.optim.AdamW(
            list(model.parameters()) + list(survival_head.parameters()),
            lr=cfg.get("healnet_learning_rate", cfg["learning_rate"]),
            weight_decay=cfg.get("healnet_weight_decay", cfg["weight_decay"]),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["n_epochs"], eta_min=1e-5)

        best_score   = -np.inf
        best_c_index = float("nan")
        best_epoch   = 0
        best_state   = None
        patience_ctr = 0

        for epoch in range(1, cfg["n_epochs"] + 1):
            model.train()
            survival_head.train()
            train_loss_sum = 0.0
            n_batches      = 0

            for batch in train_loader:
                batch = move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch)
                loss, loss_dict = compute_loss_healnet_omics_lite(outputs, batch, survival_head)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(survival_head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                train_loss_sum += loss_dict["total"]
                n_batches      += 1

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
                        all_risk.append(risk[valid].detach().cpu())
                        all_time.append(batch["pfi_time"][valid].detach().cpu())
                        all_event.append(batch["pfi_event"][valid].detach().cpu())

            c_index = float("nan")
            if all_risk:
                risk_np  = torch.cat(all_risk).squeeze().numpy()
                time_np  = torch.cat(all_time).numpy()
                event_np = torch.cat(all_event).numpy()
                if event_np.sum() > 0:
                    c_index = concordance_index(time_np, risk_np, event_np)

            print(f"  Epoch {epoch:3d}/{cfg['n_epochs']} | "
                  f"Train loss: {train_loss_sum / max(n_batches, 1):.4f} | "
                  f"Val C-index: {c_index:.4f}")

            score = c_index if not np.isnan(c_index) else -np.inf
            if epoch >= cfg.get("min_epochs", 1):
                if best_state is None or score > best_score:
                    best_score   = score
                    best_c_index = c_index
                    best_epoch   = epoch
                    patience_ctr = 0
                    best_state   = {
                        "model":    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "survival": {k: v.detach().cpu().clone() for k, v in survival_head.state_dict().items()},
                    }
                else:
                    patience_ctr += 1
                    if patience_ctr >= cfg["patience"]:
                        print(f"  Early stopping at epoch {epoch} "
                              f"(best epoch {best_epoch}, C-index {best_c_index:.4f})")
                        break

        fold_results.append({
            "fold":         fold_idx + 1,
            "best_epoch":   best_epoch,
            "best_c_index": best_c_index,
            "val_metrics":  {"c_index": best_c_index},
            "model_state":  best_state,
            "scalers":      {"rna": rna_scaler, "mirna": mirna_scaler, "methyl": methyl_scaler},
            "model_name":   "HEALNet-OmicsLite",
        })
        print(f"\n  Fold {fold_idx + 1} best epoch:   {best_epoch}")
        print(f"  Fold {fold_idx + 1} best C-index: {best_c_index:.4f}")

    cidxs = [fr["val_metrics"]["c_index"] for fr in fold_results]
    print(f"\n{'='*60}")
    print("HEALNet-OmicsLite CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"C-index: {np.nanmean(cidxs):.4f} ± {np.nanstd(cidxs):.4f}")
    for fr in fold_results:
        print(f"  Fold {fr['fold']}: {fr['val_metrics']['c_index']:.4f} "
              f"(best epoch {fr['best_epoch']})")

    return fold_results


def run_mcar_healnet_omics_lite(fold_results, cfg, device,
                                missing_rates=None, modalities=None):
    """MCAR evaluation for HEALNet-OmicsLite. Same protocol as CrossPoE."""
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
    splits  = list(skf.split(np.arange(len(strat_labels)), strat_labels))
    results = {mod: {r: [] for r in missing_rates} for mod in modalities}

    for fold_idx, fr in enumerate(fold_results):
        _, val_idx = splits[fold_idx]
        print(f"\nEvaluating HEALNet-OmicsLite MCAR | Fold {fold_idx + 1}")

        model = HEALNetOmicsLite(
            latent_dim=n_latent,
            n_latents=cfg.get("healnet_n_latents", 16),
            depth=cfg.get("healnet_depth", 2),
            n_heads=cfg.get("healnet_n_heads", 4),
            dropout=cfg.get("healnet_dropout", 0.1),
            ff_mult=cfg.get("healnet_ff_mult", 4),
            n_rna=n_rna, n_mirna=n_mirna, n_methyl=n_methyl,
        ).to(device)

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
                                     shuffle=False, collate_fn=collate_fn,
                                     num_workers=0, pin_memory=True)

                all_risk, all_time, all_event = [], [], []
                with torch.no_grad():
                    for batch in loader:
                        batch   = move_batch_to_device(batch, device)
                        outputs = model(batch)
                        risk    = survival_head(outputs["z_surv"])
                        valid   = batch["pfi_event"] >= 0
                        if valid.any():
                            all_risk.append(risk[valid].detach().cpu())
                            all_time.append(batch["pfi_time"][valid].detach().cpu())
                            all_event.append(batch["pfi_event"][valid].detach().cpu())

                c_index = float("nan")
                if all_risk:
                    risk_np  = torch.cat(all_risk).squeeze().numpy()
                    time_np  = torch.cat(all_time).numpy()
                    event_np = torch.cat(all_event).numpy()
                    if event_np.sum() > 0:
                        c_index = concordance_index(time_np, risk_np, event_np)

                results[mod][rate].append(c_index)
                print(f"  Fold {fold_idx+1} | missing={mod:<6} | "
                      f"rate={rate:.1f} | C-index={c_index:.4f}")

    print(f"\n{'='*60}")
    print("MCAR ROBUSTNESS — HEALNet-OmicsLite")
    print(f"{'='*60}")
    for mod in modalities:
        print(f"\n{mod}:")
        for rate in missing_rates:
            vals = [v for v in results[mod][rate] if not np.isnan(v)]
            mean = np.mean(vals) if vals else float("nan")
            std  = np.std(vals)  if vals else float("nan")
            print(f"  rate={rate:.1f}: C-index = {mean:.4f} ± {std:.4f}")

    return results
