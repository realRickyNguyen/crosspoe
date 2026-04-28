import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from data.dataset import MultiOmicsDataset, collate_fn
from data.mcar import MCARDataset
from models.crosspoe import CrossPoE
from models.survival import SurvivalHead
from models.translation import CrossModalTranslator

from .utils import concordance_index, move_batch_to_device


@torch.no_grad()
def evaluate_mcar(model, translator, survival_head, dataset, device, cfg,
                  use_translation=True):
    """
    Evaluate a CrossPoE model on an MCARDataset. Returns C-index only.

    Args:
        model          : CrossPoE instance
        translator     : CrossModalTranslator or None
        survival_head  : SurvivalHead instance
        dataset        : MCARDataset instance
        device         : torch.device
        cfg            : config dict
        use_translation: if False, disables translation even when translator
                         is provided (useful for ablation)

    Returns:
        dict with key "c_index"
    """
    model.eval()
    if translator is not None:
        translator.eval()
    survival_head.eval()

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    trans_warmup              = cfg["translation_warmup_epochs"] if use_translation else 9999
    all_risk, all_pfi_time, all_pfi_event = [], [], []

    for batch in loader:
        batch   = move_batch_to_device(batch, device)
        outputs = model(
            batch,
            translator=translator if use_translation else None,
            epoch=cfg["n_epochs"],
            translation_warmup_epochs=trans_warmup,
        )
        risk       = survival_head(outputs["z_surv"])
        valid_surv = batch["pfi_event"] >= 0
        if valid_surv.any():
            all_risk.append(risk[valid_surv].cpu())
            all_pfi_time.append(batch["pfi_time"][valid_surv].cpu())
            all_pfi_event.append(batch["pfi_event"][valid_surv].cpu())

    c_index = float("nan")
    if all_risk:
        risk_np  = torch.cat(all_risk).squeeze().numpy()
        time_np  = torch.cat(all_pfi_time).numpy()
        event_np = torch.cat(all_pfi_event).numpy()
        if event_np.sum() > 0:
            c_index = concordance_index(time_np, risk_np, event_np)

    return {"c_index": c_index}


def run_mcar_evaluation(fold_results, cfg, device,
                        missing_rates=None, modalities=None):
    """
    Run a full MCAR grid over all folds of a trained CrossPoE model.

    For each fold's best checkpoint, for each modality, for each missing rate,
    the val set is evaluated with that modality force-dropped at the given rate
    on top of natural missingness.

    Fold val splits are reproduced deterministically using the same seed and
    StratifiedKFold settings as run_cross_validation().

    Args:
        fold_results  : list of fold dicts returned by run_cross_validation()
        cfg           : config dict (must include n_folds, seed, n_latent,
                        translation_hidden_dim, batch_size, n_epochs,
                        translation_warmup_epochs)
        device        : torch.device
        missing_rates : floats in [0, 1]; default [0.0, 0.3, 0.5, 0.7, 0.9]
        modalities    : subset of ["rna", "mirna", "methyl"]; default all three

    Returns:
        results[modality][rate] — list of C-index values, one per fold
    """
    if missing_rates is None:
        missing_rates = [0.0, 0.3, 0.5, 0.7, 0.9]
    if modalities is None:
        modalities = ["rna", "mirna", "methyl"]

    results = {
        mod: {rate: [] for rate in missing_rates}
        for mod in modalities
    }

    # Reproduce val splits identically to run_cross_validation()
    pfi_event_all = MultiOmicsDataset._pfi_event.numpy()
    strat_labels  = np.clip(pfi_event_all, 0, 1)
    skf    = StratifiedKFold(
        n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"]
    )
    splits = list(skf.split(np.arange(len(strat_labels)), strat_labels))

    n_rna    = MultiOmicsDataset._rna_data.shape[1]
    n_mirna  = MultiOmicsDataset._mirna_data.shape[1]
    n_methyl = MultiOmicsDataset._methyl_data.shape[1]
    n_latent = cfg.get("n_latent", 48)

    for fold_idx, fr in enumerate(fold_results):
        print(f"\nFold {fold_idx + 1} / {len(fold_results)}")

        # Restore best checkpoint for this fold
        model = CrossPoE(
            latent_dim=n_latent, n_rna=n_rna, n_mirna=n_mirna, n_methyl=n_methyl
        ).to(device)
        model.load_state_dict(
            {k: v.to(device) for k, v in fr["model_state"]["model"].items()}
        )

        translator = CrossModalTranslator(
            n_latent, hidden_dim=cfg["translation_hidden_dim"]
        ).to(device)
        translator.load_state_dict(
            {k: v.to(device) for k, v in fr["model_state"]["translator"].items()}
        )

        survival_head = SurvivalHead(n_latent).to(device)
        survival_head.load_state_dict(
            {k: v.to(device) for k, v in fr["model_state"]["survival"].items()}
        )

        model.eval()
        translator.eval()
        survival_head.eval()

        # Rebuild val dataset with the scalers stored in fold_results
        _, val_idx = splits[fold_idx]
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
                res     = evaluate_mcar(
                    model, translator, survival_head,
                    mcar_ds, device, cfg, use_translation=True,
                )
                results[mod][rate].append(res["c_index"])
                print(f"  {mod:>6} @{rate:.1f} | C-index={res['c_index']:.4f}")

    # Summary table
    print(f"\n{'='*50}")
    print("MCAR EVALUATION SUMMARY")
    print(f"{'='*50}")
    for mod in modalities:
        print(f"\n{mod}:")
        for rate in missing_rates:
            vals = [v for v in results[mod][rate] if not np.isnan(v)]
            mean = np.mean(vals) if vals else float("nan")
            std  = np.std(vals)  if vals else float("nan")
            print(f"  rate={rate:.1f}: C-index = {mean:.4f} ± {std:.4f}")

    return results
