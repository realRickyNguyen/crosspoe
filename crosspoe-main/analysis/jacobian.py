"""
Jacobian analysis for CrossPoE cross-modal translation heads.

Computes ∂μ_pseudo_tgt / ∂μ_src (48×48 matrices) across all 6 pairwise
translation directions, identifies hub latent dimensions via majority vote,
and produces a publication-ready heatmap figure.
"""

import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from data.dataset import MultiOmicsDataset, collate_fn
from models.crosspoe import CrossPoE
from models.translation import CrossModalTranslator
from training.utils import move_batch_to_device

_MODALITY_NAMES = ["RNA", "miRNA", "Methyl"]


def _rebuild_val_dataset(fold_result: dict, cfg: dict) -> MultiOmicsDataset:
    """Reproduce the validation split for a given fold deterministically."""
    pfi_event_all = MultiOmicsDataset._pfi_event.numpy()
    strat_labels  = np.clip(pfi_event_all, 0, 1)
    skf = StratifiedKFold(
        n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"]
    )
    target_fold = fold_result["fold"] - 1
    for i, (_, val_idx) in enumerate(
            skf.split(np.arange(len(strat_labels)), strat_labels)):
        if i == target_fold:
            break

    return MultiOmicsDataset(
        indices=val_idx,
        rna_scaler=fold_result["scalers"]["rna"],
        mirna_scaler=fold_result["scalers"]["mirna"],
        methyl_scaler=fold_result["scalers"]["methyl"],
        dropout_probs=None,
    )


def _load_model_and_translator(fold_result: dict, cfg: dict, device: torch.device):
    """Restore CrossPoE and CrossModalTranslator from a fold checkpoint."""
    n_rna    = MultiOmicsDataset._rna_data.shape[1]
    n_mirna  = MultiOmicsDataset._mirna_data.shape[1]
    n_methyl = MultiOmicsDataset._methyl_data.shape[1]
    n_latent = cfg.get("n_latent", 48)

    model = CrossPoE(
        latent_dim=n_latent, n_rna=n_rna, n_mirna=n_mirna, n_methyl=n_methyl
    ).to(device)
    model.load_state_dict(
        {k: v.to(device) for k, v in fold_result["model_state"]["model"].items()}
    )

    translator = CrossModalTranslator(
        n_latent, hidden_dim=cfg["translation_hidden_dim"]
    ).to(device)
    translator.load_state_dict(
        {k: v.to(device) for k, v in fold_result["model_state"]["translator"].items()}
    )

    model.eval()
    translator.eval()
    return model, translator


@torch.no_grad()
def compute_translation_jacobians(
    fold_results: list,
    cfg:          dict,
    device:       torch.device,
    fold_idx:     int = 0,
) -> tuple:
    """
    Compute mean Jacobian ∂μ_pseudo / ∂μ_src for all 6 translation directions
    on the validation set of the specified fold.

    Args:
        fold_results : list of fold dicts returned by run_cross_validation()
        cfg          : config dict
        device       : torch.device
        fold_idx     : which fold to use (0-based)

    Returns:
        jacobians     : dict direction_key -> mean Jacobian (n_latent × n_latent)
        jacobian_stds : dict direction_key -> std  Jacobian (n_latent × n_latent)
    """
    fr         = fold_results[fold_idx]
    model, translator = _load_model_and_translator(fr, cfg, device)
    val_dataset = _rebuild_val_dataset(fr, cfg)
    val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    n_latent   = cfg.get("n_latent", 48)
    directions = [(s, t) for s in range(3) for t in range(3) if s != t]
    jac_samples = {f"{s}_to_{t}": [] for s, t in directions}

    # Grad computation requires temporarily lifting no_grad
    for batch in val_loader:
        batch  = move_batch_to_device(batch, device)
        mask   = batch["mask"]

        with torch.no_grad():
            outputs = model(
                batch, translator=translator,
                epoch=cfg["n_epochs"],
                translation_warmup_epochs=cfg["translation_warmup_epochs"],
            )
        mus     = outputs["mus"]
        logvars = outputs["logvars"]

        for src, tgt in directions:
            key      = f"{src}_to_{tgt}"
            both_obs = mask[:, src] & mask[:, tgt]
            if not both_obs.any():
                continue

            mu_src_batch = mus[src][both_obs].detach()
            lv_src_batch = logvars[src][both_obs].detach()
            head         = translator.translation_heads[key]

            for i in range(mu_src_batch.shape[0]):
                mu_i = mu_src_batch[i:i+1].requires_grad_(True)
                lv_i = lv_src_batch[i:i+1]

                def mu_pseudo_fn(mu_in, lv=lv_i, h=head, d=n_latent):
                    out = h.net(torch.cat([mu_in, lv], dim=-1))
                    return out[:, :d]

                J = torch.autograd.functional.jacobian(
                    mu_pseudo_fn, mu_i, create_graph=False, strict=False
                )
                jac_samples[key].append(J.squeeze().cpu().numpy())

    jacobians     = {}
    jacobian_stds = {}
    for key, jlist in jac_samples.items():
        if not jlist:
            continue
        stack = np.stack(jlist, axis=0)
        jacobians[key]     = stack.mean(axis=0)
        jacobian_stds[key] = stack.std(axis=0)

    return jacobians, jacobian_stds


def compute_translation_jacobians_all_folds(
    fold_results: list,
    cfg:          dict,
    device:       torch.device,
) -> tuple:
    """
    Run Jacobian analysis across all folds.

    Returns:
        jacobians_mean : dict key -> (n_latent, n_latent) mean J across folds
        jacobians_std  : dict key -> (n_latent, n_latent) std  across folds
        fold_jacs      : list of per-fold jacobian dicts (for majority vote)
    """
    fold_jacs = []
    for fold_idx in range(len(fold_results)):
        print(f"  Computing Jacobians — fold {fold_idx + 1}/{len(fold_results)}")
        jacs, _ = compute_translation_jacobians(fold_results, cfg, device, fold_idx)
        fold_jacs.append(jacs)

    directions     = list(fold_jacs[0].keys())
    jacobians_mean = {}
    jacobians_std  = {}
    for key in directions:
        stack = np.stack([fj[key] for fj in fold_jacs if key in fj], axis=0)
        jacobians_mean[key] = stack.mean(axis=0)
        jacobians_std[key]  = stack.std(axis=0)

    return jacobians_mean, jacobians_std, fold_jacs


def get_majority_vote_hub_dims(
    fold_jacs:              list,
    jacobians_mean:         dict,
    top_k:                  int = 8,
    min_folds:              int = 4,
    min_global_appearances: int = 4,
) -> tuple:
    """
    Identify hub latent dimensions via two-condition majority vote.

    Condition 1 — fold consistency: dimension appears in the top-k source OR
        target dims in at least min_folds folds.
    Condition 2 — global presence: dimension accumulates at least
        min_global_appearances in the mean Jacobian heatmap (top-5 source +
        target across all 6 directions).

    Args:
        fold_jacs              : per-fold jacobian dicts
        jacobians_mean         : mean jacobian dict
        top_k                  : top-k per direction for condition 1
        min_folds              : minimum folds for condition 1
        min_global_appearances : minimum score for condition 2

    Returns:
        hub_dims     : sorted list of dims passing both conditions
        fold_counts  : dict dim -> number of folds it appeared in
        global_scores: dict dim -> global heatmap score
    """
    dim_fold_counts = Counter()
    for fold_jac in fold_jacs:
        dims_this_fold = set()
        for key, J in fold_jac.items():
            abs_J = np.abs(J)
            top_src = set(np.argsort(abs_J.sum(axis=0))[::-1][:top_k].tolist())
            top_tgt = set(np.argsort(abs_J.sum(axis=1))[::-1][:top_k].tolist())
            dims_this_fold |= top_src | top_tgt
        for d in dims_this_fold:
            dim_fold_counts[d] += 1

    passes_consistency = {d for d, cnt in dim_fold_counts.items() if cnt >= min_folds}

    global_scores = Counter()
    for key, J in jacobians_mean.items():
        abs_J   = np.abs(J)
        top_src = set(np.argsort(abs_J.sum(axis=0))[::-1][:5].tolist())
        top_tgt = set(np.argsort(abs_J.sum(axis=1))[::-1][:5].tolist())
        for d in top_src | top_tgt:
            global_scores[d] += 1

    passes_global    = {d for d, score in global_scores.items()
                        if score >= min_global_appearances}
    hub_dims_voted   = sorted(passes_consistency & passes_global)

    return hub_dims_voted, dict(dim_fold_counts), dict(global_scores)


def print_majority_vote_summary(
    fold_jacs:              list,
    jacobians_mean:         dict,
    top_k:                  int = 8,
    min_folds:              int = 4,
    min_global_appearances: int = 4,
) -> list:
    """
    Print the majority-vote hub dim table and return the hub dims list.

    Returns:
        hub_dims : list of ints — dimensions passing both conditions
    """
    hub_dims, fold_counts, global_scores = get_majority_vote_hub_dims(
        fold_jacs, jacobians_mean,
        top_k=top_k, min_folds=min_folds,
        min_global_appearances=min_global_appearances,
    )
    n_folds      = len(fold_jacs)
    n_directions = len(jacobians_mean)

    direction_coverage = {}
    for d in fold_counts:
        count = 0
        for key, J in jacobians_mean.items():
            abs_J   = np.abs(J)
            top_src = set(np.argsort(abs_J.sum(axis=0))[::-1][:5].tolist())
            top_tgt = set(np.argsort(abs_J.sum(axis=1))[::-1][:5].tolist())
            if d in top_src or d in top_tgt:
                count += 1
        direction_coverage[d] = count

    print("=" * 80)
    print("JACOBIAN HUB DIMS — two-condition criterion")
    print(f"  Condition 1: top-{top_k} source/target in >= {min_folds}/{n_folds} folds")
    print(f"  Condition 2: >= {min_global_appearances} global top-5 appearances in mean Jacobian")
    print("=" * 80)
    print(f"  {'Dim':>4}  {'Folds':>7}  {'Global':>9}  {'Dirs':>6}  {'C2':>4}  {'':>8}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*9}  {'-'*6}  {'-'*4}  {'-'*8}")

    cond1_dims = sorted(
        [d for d, cnt in fold_counts.items() if cnt >= min_folds],
        key=lambda d: (-direction_coverage[d], -global_scores.get(d, 0), -fold_counts[d]),
    )

    for d in cond1_dims:
        fc  = fold_counts[d]
        gs  = global_scores.get(d, 0)
        dc  = direction_coverage[d]
        c2  = "+" if gs >= min_global_appearances else "-"
        tag = " <- HUB" if d in hub_dims else ""
        print(
            f"  {d:>4}  {fc:>3}/{n_folds} folds  "
            f"{gs:>3}/12 score  "
            f"{dc:>2}/{n_directions} dirs  "
            f"C2:{c2}{tag}"
        )

    print(f"\nFinal hub dims (both conditions): {hub_dims}")
    return hub_dims


def print_jacobian_summary(jacobians: dict) -> None:
    """
    Print per-direction summary: top source dims (by outgoing influence)
    and top target dims (by incoming sensitivity).
    """
    print("=" * 70)
    print("JACOBIAN SUMMARY — d(mu_pseudo_tgt) / d(mu_src)")
    print("Mean absolute Jacobian aggregated across validation samples")
    print("=" * 70)

    for key, J in jacobians.items():
        src_idx, tgt_idx = int(key[0]), int(key[5])
        abs_J = np.abs(J)

        top_src = np.argsort(abs_J.sum(axis=0))[::-1][:5]
        top_tgt = np.argsort(abs_J.sum(axis=1))[::-1][:5]

        print(f"\n{_MODALITY_NAMES[src_idx]} -> {_MODALITY_NAMES[tgt_idx]}  (key: {key})")
        print(f"  Mean |J|:        {abs_J.mean():.4f}")
        print(f"  Max  |J|:        {abs_J.max():.4f}")
        print(f"  Frobenius norm:  {np.linalg.norm(J, 'fro'):.4f}")
        print(f"  Top source dims: {top_src.tolist()}")
        print(f"  Top target dims: {top_tgt.tolist()}")


def plot_jacobian_paper(
    jacobians:  dict,
    hub_dims:   list,
    fold_jacs:  list = None,
    save_path:  str  = None,
) -> None:
    """
    Publication heatmap: top-5 source + target appearances per direction,
    with vertical dashed lines marking hub dims from majority vote.

    Args:
        jacobians : dict key -> (n_latent, n_latent) mean Jacobian
        hub_dims  : list of hub dimension indices (from print_majority_vote_summary)
        fold_jacs : unused — kept for API compatibility
        save_path : file path for saving (PDF/PNG); None = display only
    """
    import matplotlib.pyplot as plt

    n_latent   = next(iter(jacobians.values())).shape[0]
    directions = ["0_to_1", "0_to_2", "1_to_0", "1_to_2", "2_to_0", "2_to_1"]
    dir_labels = ["RNA->miRNA", "RNA->Methyl", "miRNA->RNA",
                  "miRNA->Methyl", "Methyl->RNA", "Methyl->miRNA"]

    top_k = 5
    matrix = np.zeros((6, n_latent))
    for i, d in enumerate(directions):
        if d not in jacobians:
            continue
        J = jacobians[d]
        abs_J = np.abs(J)
        for dim in np.argsort(abs_J.sum(axis=0))[::-1][:top_k]:
            matrix[i, dim] += 1
        for dim in np.argsort(abs_J.sum(axis=1))[::-1][:top_k]:
            matrix[i, dim] += 1

    fig, ax = plt.subplots(figsize=(12, 5))
    plt.subplots_adjust(top=0.78, bottom=0.15, left=0.13, right=0.88)

    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=2,
                   interpolation="nearest")

    for hd in hub_dims:
        ax.axvspan(hd - 0.5, hd + 0.5, color="#C0392B", alpha=0.07, zorder=0)
        ax.axvline(x=hd, color="#C0392B", linewidth=1.4,
                   linestyle="--", alpha=0.85, zorder=3)

    base_ticks  = [0, 10, 20, 30, 40, n_latent - 1]
    clean_base  = [t for t in base_ticks if all(abs(t - hd) >= 2 for hd in hub_dims)]
    all_ticks   = sorted(set(clean_base + list(hub_dims)))
    ax.set_xticks(all_ticks)
    ax.set_xticklabels([str(t) for t in all_ticks], fontsize=9)
    for tick_label, t in zip(ax.get_xticklabels(), all_ticks):
        if t in hub_dims:
            tick_label.set_color("#C0392B")
            tick_label.set_fontweight("bold")

    ax.set_yticks(range(6))
    ax.set_yticklabels(dir_labels, fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("Latent dimension index", fontsize=11, labelpad=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=20)
    cbar.set_label("Top-5 appearances\n(source + target)", fontsize=9, labelpad=8)
    cbar.set_ticks([0, 1, 2])
    cbar.ax.tick_params(labelsize=8.5, length=0)
    cbar.outline.set_visible(False)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
