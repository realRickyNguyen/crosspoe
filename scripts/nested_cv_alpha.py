"""
Nested CV for alpha selection in CrossPoE.

Outer loop: 5 folds for reporting performance (same splits as run_cross_validation).
Inner loop: 4-fold CV on the outer training split to select alpha.
Alpha selected per outer fold independently by minimising MCAR RNA decline.
"""
import copy

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset import MultiOmicsDataset, collate_fn
from data.mcar import MCARDataset
from data.preprocessing import compute_scaler
from models.crosspoe import CrossPoE
from models.survival import SurvivalHead
from models.translation import CrossModalTranslator
from training.losses import compute_loss
from training.trainer import evaluate, train_one_epoch
from training.utils import concordance_index, get_beta, move_batch_to_device, set_seed


@torch.no_grad()
def _eval_mcar_single(model, translator, survival_head, val_dataset, mod, rate, cfg, device):
    """
    Returns C-index for a single modality/rate combination.
    Used inside the nested CV inner loop for alpha selection.
    """
    mcar_dataset = MCARDataset(val_dataset, forced_missing_modality=mod,
                               forced_missing_rate=rate)
    loader = DataLoader(mcar_dataset, batch_size=cfg["batch_size"] * 2,
                        shuffle=False, collate_fn=collate_fn, num_workers=0)
    all_risk, all_time, all_event = [], [], []

    for batch in loader:
        batch   = move_batch_to_device(batch, device)
        outputs = model(batch, translator=translator,
                        epoch=cfg["n_epochs"],
                        translation_warmup_epochs=cfg["translation_warmup_epochs"])
        risk  = survival_head(outputs["z_surv"])
        valid = batch["pfi_event"] >= 0
        if valid.any():
            all_risk.append(risk[valid].cpu())
            all_time.append(batch["pfi_time"][valid].cpu())
            all_event.append(batch["pfi_event"][valid].cpu())

    if not all_risk:
        return float("nan")
    risk_np  = torch.cat(all_risk).squeeze().numpy()
    time_np  = torch.cat(all_time).numpy()
    event_np = torch.cat(all_event).numpy()
    if event_np.sum() == 0:
        return float("nan")
    return concordance_index(time_np, risk_np, event_np)


def run_cross_validation_nested_alpha(cfg, device, alphas=None):
    """
    Nested CV for alpha selection in CrossPoE.

    For each outer fold, an inner 4-fold CV on the outer training split
    selects the alpha that minimises MCAR RNA C-index decline (rate 0.9 - 0.0).
    The outer fold is then trained with the selected alpha.

    Args:
        cfg:    config dict (must include standard CrossPoE keys)
        device: torch.device
        alphas: list of alpha values to search (default [0.1, 0.25, 0.5, 0.75, 1.0])

    Returns:
        (fold_results, selected_alphas)
          fold_results:     same format as run_cross_validation()
          selected_alphas:  list of per-fold selected alpha values
    """
    if alphas is None:
        alphas = [0.1, 0.25, 0.5, 0.75, 1.0]

    set_seed(cfg["seed"])

    n_latent = cfg.get("n_latent", 48)
    n_rna    = MultiOmicsDataset._rna_data.shape[1]
    n_mirna  = MultiOmicsDataset._mirna_data.shape[1]
    n_methyl = MultiOmicsDataset._methyl_data.shape[1]

    pfi_event_all = MultiOmicsDataset._pfi_event.numpy()
    strat_labels  = np.clip(pfi_event_all, 0, 1)

    outer_skf    = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True,
                                   random_state=cfg["seed"])
    outer_splits = list(outer_skf.split(np.arange(len(strat_labels)), strat_labels))

    fold_results    = []
    selected_alphas = []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_splits):

        print(f"\n{'='*60}")
        print(f"OUTER FOLD {fold_idx + 1} / {cfg['n_folds']}")
        print(f"{'='*60}")

        # ── Inner CV: select alpha on training split only ─────────────────────
        inner_skf   = StratifiedKFold(n_splits=4, shuffle=True,
                                      random_state=cfg["seed"] + fold_idx)
        inner_strat = np.clip(MultiOmicsDataset._pfi_event.numpy()[train_idx], 0, 1)
        inner_splits = list(inner_skf.split(np.arange(len(train_idx)), inner_strat))

        alpha_scores = {alpha: [] for alpha in alphas}

        for alpha in alphas:
            print(f"\n  Inner CV for alpha={alpha}")
            cfg_inner = copy.deepcopy(cfg)
            cfg_inner["alpha"] = alpha

            for inner_fold_idx, (inner_train_rel, inner_val_rel) in enumerate(inner_splits):
                inner_train_abs = train_idx[inner_train_rel]
                inner_val_abs   = train_idx[inner_val_rel]

                rna_scaler    = compute_scaler(MultiOmicsDataset._rna_data[inner_train_abs][MultiOmicsDataset._rna_mask[inner_train_abs]])
                mirna_scaler  = compute_scaler(MultiOmicsDataset._mirna_data[inner_train_abs][MultiOmicsDataset._mirna_mask[inner_train_abs]])
                methyl_scaler = compute_scaler(MultiOmicsDataset._methyl_data[inner_train_abs][MultiOmicsDataset._methyl_mask[inner_train_abs]])

                train_dataset = MultiOmicsDataset(
                    indices=inner_train_abs, rna_scaler=rna_scaler,
                    mirna_scaler=mirna_scaler, methyl_scaler=methyl_scaler,
                    dropout_probs={"rna": 0.0, "mirna": 0.0, "methyl": 0.0},
                )
                inner_val_dataset = MultiOmicsDataset(
                    indices=inner_val_abs, rna_scaler=rna_scaler,
                    mirna_scaler=mirna_scaler, methyl_scaler=methyl_scaler,
                    dropout_probs=None,
                )
                train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                                          shuffle=True, collate_fn=collate_fn, num_workers=0)
                val_loader   = DataLoader(inner_val_dataset, batch_size=cfg["batch_size"] * 2,
                                          shuffle=False, collate_fn=collate_fn, num_workers=0)

                set_seed(cfg["seed"] + fold_idx * 10 + inner_fold_idx)
                model         = CrossPoE(latent_dim=n_latent, n_rna=n_rna,
                                         n_mirna=n_mirna, n_methyl=n_methyl).to(device)
                translator    = CrossModalTranslator(
                    n_latent, hidden_dim=cfg["translation_hidden_dim"], alpha=alpha
                ).to(device)
                survival_head = SurvivalHead(n_latent).to(device)

                all_params = (list(model.parameters())
                              + list(translator.parameters())
                              + list(survival_head.parameters()))
                optimizer  = torch.optim.AdamW(all_params, lr=cfg["learning_rate"],
                                               weight_decay=cfg["weight_decay"])
                scheduler  = CosineAnnealingLR(optimizer, T_max=cfg["n_epochs"], eta_min=1e-5)

                best_score   = -np.inf
                best_state   = None
                patience_ctr = 0

                for epoch in range(1, cfg["n_epochs"] + 1):
                    train_one_epoch(model, translator, survival_head,
                                    train_loader, optimizer, device, epoch, cfg_inner)
                    scheduler.step()
                    val_metrics = evaluate(model, translator, survival_head,
                                           val_loader, device, cfg_inner)
                    score = val_metrics["c_index"] if not np.isnan(val_metrics["c_index"]) else -np.inf
                    if score > best_score:
                        best_score   = score
                        patience_ctr = 0
                        best_state   = {
                            "model":      {k: v.cpu().clone() for k, v in model.state_dict().items()},
                            "translator": {k: v.cpu().clone() for k, v in translator.state_dict().items()},
                            "survival":   {k: v.cpu().clone() for k, v in survival_head.state_dict().items()},
                        }
                    else:
                        patience_ctr += 1
                        if patience_ctr >= cfg["patience"]:
                            break

                model.load_state_dict({k: v.to(device) for k, v in best_state["model"].items()})
                translator.load_state_dict({k: v.to(device) for k, v in best_state["translator"].items()})
                survival_head.load_state_dict({k: v.to(device) for k, v in best_state["survival"].items()})
                model.eval()
                translator.eval()
                survival_head.eval()

                ci_r0  = _eval_mcar_single(model, translator, survival_head,
                                           inner_val_dataset, "rna", 0.0, cfg, device)
                ci_r09 = _eval_mcar_single(model, translator, survival_head,
                                           inner_val_dataset, "rna", 0.9, cfg, device)
                decline = ci_r09 - ci_r0
                alpha_scores[alpha].append(decline)
                print(f"    Inner fold {inner_fold_idx+1}: decline={decline:+.4f}")

        mean_declines = {a: np.mean(alpha_scores[a]) for a in alphas}
        best_alpha    = max(mean_declines, key=mean_declines.get)
        selected_alphas.append(best_alpha)
        print(f"\n  Outer fold {fold_idx+1} selected alpha={best_alpha}")
        print(f"  Inner mean declines: {mean_declines}")

        # ── Train outer fold with selected alpha ──────────────────────────────
        cfg_outer = copy.deepcopy(cfg)
        cfg_outer["alpha"] = best_alpha

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
        model         = CrossPoE(latent_dim=n_latent, n_rna=n_rna,
                                 n_mirna=n_mirna, n_methyl=n_methyl).to(device)
        translator    = CrossModalTranslator(
            n_latent, hidden_dim=cfg["translation_hidden_dim"], alpha=best_alpha
        ).to(device)
        survival_head = SurvivalHead(n_latent).to(device)

        all_params = (list(model.parameters())
                      + list(translator.parameters())
                      + list(survival_head.parameters()))
        optimizer  = torch.optim.AdamW(all_params, lr=cfg["learning_rate"],
                                       weight_decay=cfg["weight_decay"])
        scheduler  = CosineAnnealingLR(optimizer, T_max=cfg["n_epochs"], eta_min=1e-5)

        best_score   = -np.inf
        best_c_index = float("nan")
        best_epoch   = 0
        patience_ctr = 0
        best_state   = None

        for epoch in range(1, cfg["n_epochs"] + 1):
            train_one_epoch(model, translator, survival_head,
                            train_loader, optimizer, device, epoch, cfg_outer)
            scheduler.step()
            val_metrics = evaluate(model, translator, survival_head,
                                   val_loader, device, cfg_outer)
            score = val_metrics["c_index"] if not np.isnan(val_metrics["c_index"]) else -np.inf
            if epoch >= cfg.get("min_epochs", 1):
                if score > best_score:
                    best_score   = score
                    best_c_index = val_metrics["c_index"]
                    best_epoch   = epoch
                    patience_ctr = 0
                    best_state   = {
                        "model":      {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        "translator": {k: v.cpu().clone() for k, v in translator.state_dict().items()},
                        "survival":   {k: v.cpu().clone() for k, v in survival_head.state_dict().items()},
                    }
                else:
                    patience_ctr += 1
                    if patience_ctr >= cfg["patience"]:
                        print(f"  Early stopping at epoch {epoch} "
                              f"(best {best_epoch}, C-index {best_score:.4f})")
                        break

        model.load_state_dict({k: v.to(device) for k, v in best_state["model"].items()})
        translator.load_state_dict({k: v.to(device) for k, v in best_state["translator"].items()})
        survival_head.load_state_dict({k: v.to(device) for k, v in best_state["survival"].items()})

        final_val = evaluate(model, translator, survival_head, val_loader, device, cfg_outer)

        fold_results.append({
            "fold":         fold_idx + 1,
            "best_epoch":   best_epoch,
            "best_c_index": best_c_index,
            "val_metrics":  final_val,
            "model_state":  best_state,
            "scalers":      {"rna": rna_scaler, "mirna": mirna_scaler, "methyl": methyl_scaler},
            "alpha":        best_alpha,
        })

    cidxs = [fr["val_metrics"]["c_index"] for fr in fold_results]
    print(f"\n{'='*60}")
    print("NESTED CV SUMMARY")
    print(f"{'='*60}")
    print(f"C-index: {np.nanmean(cidxs):.4f} ± {np.nanstd(cidxs):.4f}")
    print(f"Selected alphas per fold: {selected_alphas}")
    for fr in fold_results:
        print(f"  Fold {fr['fold']}: C-index={fr['val_metrics']['c_index']:.4f}  alpha={fr['alpha']}")

    return fold_results, selected_alphas
