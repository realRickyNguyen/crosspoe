import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset import MultiOmicsDataset, collate_fn
from data.preprocessing import compute_scaler
from models.crosspoe import CrossPoE
from models.survival import SurvivalHead
from models.translation import CrossModalTranslator

from .losses import compute_loss
from .utils import concordance_index, get_beta, get_dropout_p, move_batch_to_device, set_seed


def train_one_epoch(model, translator, survival_head, loader, optimizer, device, epoch, cfg):
    """Run one full training epoch and return averaged loss components."""
    model.train()
    translator.train()
    survival_head.train()

    loss_keys  = [
        "total", "survival", "recon", "kl", "consist",
        "translation", "gate", "cycle", "unimodal", "surv_trans",
    ]
    loss_accum = {k: 0.0 for k in loss_keys}
    n_batches  = 0
    beta       = get_beta(epoch, cfg["kl_warmup_epochs"])

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        outputs = model(
            batch,
            translator=translator,
            epoch=epoch,
            translation_warmup_epochs=cfg["translation_warmup_epochs"],
        )
        loss, loss_dict = compute_loss(
            outputs=outputs,
            batch=batch,
            beta=beta,
            lambda_recon=cfg["lambda_recon"],
            lambda_consist=cfg["lambda_consist"],
            lambda_survival=cfg["lambda_survival"],
            lambda_translation=cfg["lambda_translation"],
            lambda_cycle=cfg["lambda_cycle"],
            lambda_gate=cfg["lambda_gate"],
            survival_head=survival_head,
            translator=translator,
            epoch=epoch,
            translation_warmup_epochs=cfg["translation_warmup_epochs"],
            model=model,
            lambda_unimodal=cfg["lambda_unimodal"],
            lambda_surv_trans=cfg.get("lambda_surv_trans", 0.0),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters())
            + list(translator.parameters())
            + list(survival_head.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        for k in loss_keys:
            loss_accum[k] += loss_dict[k]
        n_batches += 1

    return {k: v / n_batches for k, v in loss_accum.items()}


@torch.no_grad()
def evaluate(model, translator, survival_head, loader, device, cfg):
    """Evaluate model on a data loader; returns validation loss and C-index."""
    model.eval()
    if translator is not None:
        translator.eval()
    survival_head.eval()

    all_risk, all_pfi_time, all_pfi_event = [], [], []
    total_loss_sum = 0.0
    n_batches      = 0

    for batch in loader:
        batch   = move_batch_to_device(batch, device)
        outputs = model(
            batch,
            translator=translator,
            epoch=cfg["n_epochs"],
            translation_warmup_epochs=cfg["translation_warmup_epochs"],
        )
        loss, _ = compute_loss(
            outputs=outputs,
            batch=batch,
            beta=1.0,
            lambda_recon=cfg["lambda_recon"],
            lambda_consist=cfg["lambda_consist"],
            lambda_survival=cfg["lambda_survival"],
            lambda_translation=cfg["lambda_translation"],
            lambda_cycle=cfg["lambda_cycle"],
            lambda_gate=cfg["lambda_gate"],
            survival_head=survival_head,
            translator=translator,
            epoch=cfg["n_epochs"],
            translation_warmup_epochs=cfg["translation_warmup_epochs"],
            model=model,
            lambda_unimodal=cfg["lambda_unimodal"],
            lambda_surv_trans=cfg.get("lambda_surv_trans", 0.0),
        )
        total_loss_sum += loss.item()

        risk       = survival_head(outputs["z_surv"])
        valid_surv = batch["pfi_event"] >= 0
        if valid_surv.any():
            all_risk.append(risk[valid_surv].cpu())
            all_pfi_time.append(batch["pfi_time"][valid_surv].cpu())
            all_pfi_event.append(batch["pfi_event"][valid_surv].cpu())
        n_batches += 1

    c_index = float("nan")
    if all_risk:
        risk_np  = torch.cat(all_risk).squeeze().numpy()
        time_np  = torch.cat(all_pfi_time).numpy()
        event_np = torch.cat(all_pfi_event).numpy()
        if event_np.sum() > 0:
            c_index = concordance_index(time_np, risk_np, event_np)

    return {
        "loss":    total_loss_sum / max(n_batches, 1),
        "c_index": c_index,
    }


def _run_fold_diagnostics(model, translator, survival_head, val_loader, device, cfg):
    """
    Post-training diagnostics printed for fold 0:
      - Per-modality encoder logvar means (confidence check)
      - Per-direction cosine similarity and gate values
      - Gate-quality (cosine) Pearson correlation
      - PoE fusion dominance check
    """
    model.eval()
    translator.eval()
    survival_head.eval()

    modality_names = ["rna", "mirna", "methyl"]
    logvar_means   = {name: [] for name in modality_names}
    gate_vals      = []
    trans_cosine   = {f"{s}_to_{t}": [] for s in range(3) for t in range(3) if s != t}

    with torch.no_grad():
        for batch in val_loader:
            batch   = move_batch_to_device(batch, device)
            mask    = batch["mask"]
            outputs = model(
                batch, translator=translator, epoch=999,
                translation_warmup_epochs=cfg["translation_warmup_epochs"],
            )
            mus     = outputs["mus"]
            logvars = outputs["logvars"]

            for i, name in enumerate(modality_names):
                obs = mask[:, i]
                if obs.any():
                    logvar_means[name].append(logvars[i][obs].mean().item())

            for src in range(3):
                for tgt in range(3):
                    if src == tgt:
                        continue
                    both = mask[:, src] & mask[:, tgt]
                    if not both.any():
                        continue
                    key          = f"{src}_to_{tgt}"
                    mu_pseudo, _ = translator.translation_heads[key](
                        mus[src][both], logvars[src][both]
                    )
                    gate = translator.gate_networks[key](
                        mus[src][both], logvars[src][both]
                    )
                    cos = F.cosine_similarity(mu_pseudo, mus[tgt][both].detach(), dim=1)
                    trans_cosine[key].append(cos.mean().item())
                    gate_vals.append((key, gate.mean().item(), cos.mean().item()))

    print("=== Encoder logvar means (higher = less confident) ===")
    for name in modality_names:
        vals = logvar_means[name]
        if vals:
            print(f"  {name:8s}: {np.mean(vals):.4f}")

    dir_names = ["RNA", "miRNA", "Methyl"]
    print("\n=== Per-direction cosine similarity and gate values ===")
    for src in range(3):
        for tgt in range(3):
            if src == tgt:
                continue
            key          = f"{src}_to_{tgt}"
            cos_vals     = trans_cosine[key]
            gate_entries = [(g, c) for k, g, c in gate_vals if k == key]
            if cos_vals and gate_entries:
                avg_cos  = np.mean(cos_vals)
                avg_gate = np.mean([g for g, c in gate_entries])
                print(f"  {dir_names[src]:6s} -> {dir_names[tgt]:6s}: "
                      f"cosine={avg_cos:.4f}  gate={avg_gate:.4f}")

    all_gates   = [g for _, g, _ in gate_vals]
    all_cosines = [c for _, _, c in gate_vals]
    if len(all_gates) > 2:
        corr = np.corrcoef(all_gates, all_cosines)[0, 1]
        print(f"\n=== Gate-quality correlation ===")
        print(f"  Pearson r = {corr:.4f}  (target: > 0.0)")

    print("\n=== PoE fusion dominance check ===")
    with torch.no_grad():
        batch   = next(iter(val_loader))
        batch   = move_batch_to_device(batch, device)
        mask    = batch["mask"]
        outputs = model(
            batch, translator=translator, epoch=999,
            translation_warmup_epochs=cfg["translation_warmup_epochs"],
        )
        mus     = outputs["mus"]
        logvars = outputs["logvars"]
        mu_poe  = outputs["mu_poe"]

        for i, name in enumerate(modality_names):
            obs = mask[:, i]
            if obs.any():
                cos_to_poe = F.cosine_similarity(
                    mus[i][obs], mu_poe[obs], dim=1
                ).mean().item()
                prec = torch.exp(-logvars[i][obs]).mean().item()
                print(f"  {name:8s}: precision={prec:.4f}  cosine_to_PoE={cos_to_poe:.4f}")


def run_cross_validation(cfg, device):
    """
    5-fold stratified cross-validation for CrossPoE PFI survival prediction.

    Stratification is on the PFI event label (missing labels clipped to 0).
    Early stopping is based on validation C-index with a patience threshold.
    Fold-0 diagnostics (gate, cosine, PoE dominance) are printed after training.

    Args:
        cfg    : configuration dict (see config.py)
        device : torch.device

    Returns:
        List of per-fold result dicts with keys:
            fold, best_epoch, best_score, best_c_index,
            val_metrics, model_state, scalers
    """
    set_seed(cfg["seed"])

    pfi_event_all = MultiOmicsDataset._pfi_event.numpy()
    strat_labels  = np.clip(pfi_event_all, 0, 1)
    skf           = StratifiedKFold(
        n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"]
    )
    fold_results = []

    n_rna    = MultiOmicsDataset._rna_data.shape[1]
    n_mirna  = MultiOmicsDataset._mirna_data.shape[1]
    n_methyl = MultiOmicsDataset._methyl_data.shape[1]
    n_latent = cfg.get("n_latent", 48)

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.arange(len(strat_labels)), strat_labels)
    ):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1} / {cfg['n_folds']}")
        print(f"  Train: {len(train_idx)}  |  Val: {len(val_idx)}")
        print(f"{'='*60}")

        # Compute scalers from training observations only
        rna_scaler    = compute_scaler(
            MultiOmicsDataset._rna_data[train_idx][MultiOmicsDataset._rna_mask[train_idx]]
        )
        mirna_scaler  = compute_scaler(
            MultiOmicsDataset._mirna_data[train_idx][MultiOmicsDataset._mirna_mask[train_idx]]
        )
        methyl_scaler = compute_scaler(
            MultiOmicsDataset._methyl_data[train_idx][MultiOmicsDataset._methyl_mask[train_idx]]
        )

        train_dataset = MultiOmicsDataset(
            indices=train_idx,
            rna_scaler=rna_scaler,
            mirna_scaler=mirna_scaler,
            methyl_scaler=methyl_scaler,
            dropout_probs={"rna": 0.0, "mirna": 0.0, "methyl": 0.0},
        )
        val_dataset = MultiOmicsDataset(
            indices=val_idx,
            rna_scaler=rna_scaler,
            mirna_scaler=mirna_scaler,
            methyl_scaler=methyl_scaler,
            dropout_probs=None,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["batch_size"] * 2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        set_seed(cfg["seed"] + fold_idx)
        model = CrossPoE(
            latent_dim=n_latent, n_rna=n_rna, n_mirna=n_mirna, n_methyl=n_methyl
        ).to(device)
        translator    = CrossModalTranslator(
            n_latent, hidden_dim=cfg["translation_hidden_dim"]
        ).to(device)
        survival_head = SurvivalHead(n_latent).to(device)

        all_params = (
            list(model.parameters())
            + list(translator.parameters())
            + list(survival_head.parameters())
        )
        optimizer = torch.optim.AdamW(
            all_params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["n_epochs"], eta_min=1e-5)

        best_score   = -np.inf
        best_c_index = float("nan")
        best_epoch   = 0
        patience_ctr = 0
        best_state   = None

        for epoch in range(1, cfg["n_epochs"] + 1):
            p = get_dropout_p(epoch, cfg)
            train_dataset.dropout_probs = {"rna": p, "mirna": p, "methyl": p}

            train_metrics = train_one_epoch(
                model, translator, survival_head,
                train_loader, optimizer, device, epoch, cfg,
            )
            scheduler.step()
            val_metrics = evaluate(
                model, translator, survival_head, val_loader, device, cfg
            )

            print(
                f"  Epoch {epoch:3d}/{cfg['n_epochs']} | "
                f"beta={get_beta(epoch, cfg['kl_warmup_epochs']):.2f} | "
                f"Train: {train_metrics['total']:.4f} "
                f"(surv={train_metrics['survival']:.3f} "
                f"kl={train_metrics['kl']:.3f} "
                f"recon={train_metrics['recon']:.3f} "
                f"trans={train_metrics['translation']:.3f} "
                f"cycle={train_metrics['cycle']:.3f} "
                f"surv_trans={train_metrics['surv_trans']:.3f}) | "
                f"Val C-index: {val_metrics['c_index']:.4f}"
            )

            if epoch >= cfg.get("min_epochs", 1):
                c     = val_metrics["c_index"]
                score = c if not np.isnan(c) else -np.inf
                if score > best_score:
                    best_score   = score
                    best_c_index = c
                    best_epoch   = epoch
                    patience_ctr = 0
                    best_state   = {
                        "model": {
                            k: v.cpu().clone() for k, v in model.state_dict().items()
                        },
                        "translator": {
                            k: v.cpu().clone() for k, v in translator.state_dict().items()
                        },
                        "survival": {
                            k: v.cpu().clone() for k, v in survival_head.state_dict().items()
                        },
                    }
                else:
                    patience_ctr += 1
                    if patience_ctr >= cfg["patience"]:
                        print(
                            f"  Early stopping at epoch {epoch} "
                            f"(best epoch {best_epoch}, C-index {best_score:.4f})"
                        )
                        break

        # Restore best checkpoint
        model.load_state_dict({k: v.to(device) for k, v in best_state["model"].items()})
        translator.load_state_dict(
            {k: v.to(device) for k, v in best_state["translator"].items()}
        )
        survival_head.load_state_dict(
            {k: v.to(device) for k, v in best_state["survival"].items()}
        )

        final_val = evaluate(model, translator, survival_head, val_loader, device, cfg)
        print(f"\n  Fold {fold_idx + 1} best epoch : {best_epoch}")
        print(f"  Final val C-index: {final_val['c_index']:.4f}")

        if fold_idx == 0:
            _run_fold_diagnostics(model, translator, survival_head, val_loader, device, cfg)

        fold_results.append({
            "fold":         fold_idx + 1,
            "best_epoch":   best_epoch,
            "best_score":   best_score,
            "best_c_index": best_c_index,
            "val_metrics":  final_val,
            "model_state":  best_state,
            "scalers":      {
                "rna":    rna_scaler,
                "mirna":  mirna_scaler,
                "methyl": methyl_scaler,
            },
        })

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    cidxs = [fr["val_metrics"]["c_index"] for fr in fold_results]
    print(f"C-index: {np.nanmean(cidxs):.4f} ± {np.nanstd(cidxs):.4f}")
    print("\nPer-fold C-index:")
    for fr in fold_results:
        print(f"  Fold {fr['fold']}: {fr['best_c_index']:.4f}  (best epoch {fr['best_epoch']})")

    return fold_results
