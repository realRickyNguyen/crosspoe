"""
Integrated Gradients (Captum) attribution for CrossPoE hub latent dimensions.

For each modality encoder, computes IG attributions targeting the hub latent
dimensions identified from Jacobian analysis. Baseline = zero vector (appropriate
for z-scored inputs). Supports fold-wise aggregation and majority-vote stable
feature identification.
"""

import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from data.dataset import MultiOmicsDataset, collate_fn
from models.crosspoe import CrossPoE
from training.utils import move_batch_to_device
from .features import resolve_feature_name, get_feature_names
from .jacobian import _rebuild_val_dataset, _load_model_and_translator

_MODALITY_KEYS = ["rna", "mirna", "methyl"]


def compute_hub_ig(
    fold_results: list,
    cfg:          dict,
    device:       torch.device,
    hub_dims:     list,
    fold_idx:     int  = 0,
    n_steps:      int  = 50,
) -> dict:
    """
    Compute mean Integrated Gradients attributions over the validation set
    for each modality encoder × hub dimension combination.

    Args:
        fold_results : list of fold dicts from run_cross_validation()
        cfg          : config dict
        device       : torch.device
        hub_dims     : list of hub latent dimension indices (from Jacobian analysis)
        fold_idx     : which fold to use (0-based)
        n_steps      : Captum IG interpolation steps

    Returns:
        attrs : dict mod -> list of (n_features,) mean attribution arrays,
                one array per hub dim (same order as hub_dims); None if no
                observed samples for that modality in this fold
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError as e:
        raise ImportError(
            "captum is required for IG analysis: pip install captum"
        ) from e

    fr          = fold_results[fold_idx]
    model, _    = _load_model_and_translator(fr, cfg, device)
    val_dataset = _rebuild_val_dataset(fr, cfg)
    val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    encoders = {
        "rna":    model.rna_enc,
        "mirna":  model.mirna_enc,
        "methyl": model.methyl_enc,
    }
    mask_indices = {"rna": 0, "mirna": 1, "methyl": 2}

    # attrs_accum[mod][hub_dim] = list of (n_obs, n_features) arrays
    attrs_accum = {mod: {d: [] for d in hub_dims} for mod in _MODALITY_KEYS}

    def make_encoder_hub_fn(encoder, hub_dim):
        def fn(x):
            mu, _ = encoder(x)
            return mu[:, hub_dim].unsqueeze(1)
        return fn

    for batch in val_loader:
        batch = move_batch_to_device(batch, device)
        mask  = batch["mask"]

        for mod in _MODALITY_KEYS:
            m_idx = mask_indices[mod]
            obs   = mask[:, m_idx]
            if not obs.any() or batch[mod] is None:
                continue

            x_obs    = batch[mod][obs].detach()
            baseline = torch.zeros_like(x_obs)
            encoder  = encoders[mod]

            for hub_dim in hub_dims:
                fn = make_encoder_hub_fn(encoder, hub_dim)
                ig = IntegratedGradients(fn)
                attr = ig.attribute(
                    x_obs,
                    baselines=baseline,
                    n_steps=n_steps,
                    return_convergence_delta=False,
                )
                attrs_accum[mod][hub_dim].append(attr.detach().cpu().numpy())

    attrs = {}
    for mod in _MODALITY_KEYS:
        hub_arrs = []
        for hub_dim in hub_dims:
            chunks = attrs_accum[mod][hub_dim]
            if not chunks:
                hub_arrs.append(None)
                continue
            all_attrs = np.concatenate(chunks, axis=0)
            hub_arrs.append(all_attrs.mean(axis=0))
        attrs[mod] = hub_arrs

    return attrs


def compute_hub_ig_all_folds(
    fold_results: list,
    cfg:          dict,
    device:       torch.device,
    hub_dims:     list,
    n_steps:      int = 50,
    top_k:        int = 50,
    min_folds:    int = 3,
) -> tuple:
    """
    Run IG attribution across all folds and apply majority vote per
    modality × hub dim to identify stable features.

    Args:
        fold_results : list of fold dicts
        cfg          : config dict
        device       : torch.device
        hub_dims     : hub dimension indices (from Jacobian analysis)
        n_steps      : IG interpolation steps
        top_k        : top-k features to track per fold (per modality/hub dim)
        min_folds    : minimum folds a feature must appear in to be "stable"

    Returns:
        voted_attrs   : dict mod -> list of mean (n_features,) attr arrays
        voted_features: dict (mod, hub_dim) -> sorted list of stable feature names
        fold_top_sets : dict (mod, hub_dim) -> list of per-fold top-feature sets
    """
    from data.dataset import MultiOmicsDataset
    feature_names = get_feature_names(MultiOmicsDataset)

    top_k_per_mod = {"rna": top_k, "mirna": top_k, "methyl": top_k * 4}
    n_folds       = len(fold_results)

    # fold_attrs[(mod, hub_dim_pos)] = list of (n_features,) arrays
    fold_attrs    = {(mod, i): [] for mod in _MODALITY_KEYS for i in range(len(hub_dims))}
    fold_top_sets = {(mod, hd): [] for mod in _MODALITY_KEYS for hd in hub_dims}

    for fold_idx in range(n_folds):
        print(f"  Computing IG — fold {fold_idx + 1}/{n_folds}")
        attrs = compute_hub_ig(fold_results, cfg, device,
                               hub_dims=hub_dims, fold_idx=fold_idx, n_steps=n_steps)
        for mod in _MODALITY_KEYS:
            for i, hub_dim in enumerate(hub_dims):
                arr = attrs[mod][i]
                if arr is None:
                    continue
                fold_attrs[(mod, i)].append(arr)
                top_idx   = np.argsort(np.abs(arr))[::-1][:top_k_per_mod[mod]]
                names     = feature_names[mod]
                top_names = set(names[j] for j in top_idx if j < len(names))
                fold_top_sets[(mod, hub_dim)].append(top_names)

    voted_features = {}
    for mod in _MODALITY_KEYS:
        names = feature_names[mod]
        feat_idx = {name: j for j, name in enumerate(names)}
        for i, hub_dim in enumerate(hub_dims):
            sets = fold_top_sets[(mod, hub_dim)]
            if not sets:
                voted_features[(mod, hub_dim)] = []
                continue
            counts  = Counter(feat for s in sets for feat in s)
            passing = [f for f, cnt in counts.items() if cnt >= min_folds]
            arrs    = fold_attrs[(mod, i)]
            if arrs:
                mean_arr = np.stack(arrs).mean(axis=0)
                passing.sort(key=lambda f: -abs(mean_arr[feat_idx[f]])
                             if f in feat_idx else 0)
            voted_features[(mod, hub_dim)] = passing

    voted_attrs = {}
    for mod in _MODALITY_KEYS:
        hub_arrs = []
        for i in range(len(hub_dims)):
            arrs = fold_attrs[(mod, i)]
            hub_arrs.append(np.stack(arrs).mean(axis=0) if arrs else None)
        voted_attrs[mod] = hub_arrs

    return voted_attrs, voted_features, fold_top_sets


def print_top_features(
    attrs:         dict,
    hub_dims:      list,
    feature_names: dict,
    top_k:         int = 20,
) -> None:
    """
    Print top-k features by mean absolute IG attribution per modality × hub dim.

    Args:
        attrs         : output of compute_hub_ig() or compute_hub_ig_all_folds()
        hub_dims      : hub dimension indices
        feature_names : dict mod -> list of feature name strings
        top_k         : number of top features to display
    """
    print("=" * 70)
    print(f"TOP-{top_k} FEATURES BY IG ATTRIBUTION — HUB DIMENSIONS {hub_dims}")
    print("Baseline = zeros (z-scored inputs)")
    print("=" * 70)

    for mod in _MODALITY_KEYS:
        for i, hub_dim in enumerate(hub_dims):
            attr = attrs[mod][i]
            if attr is None:
                continue
            abs_attr  = np.abs(attr)
            names     = feature_names.get(mod, [])
            top_idx   = np.argsort(abs_attr)[::-1][:top_k]
            top_names = [names[j] if j < len(names) else str(j) for j in top_idx]
            top_vals  = abs_attr[top_idx]

            print(f"\n{mod.upper()} -> Hub dim {hub_dim}")
            print(f"  {'Feature':<25s}  {'|IG attr|':>10s}")
            print(f"  {'-'*37}")
            for name, val in zip(top_names, top_vals):
                print(f"  {str(name):<25s}  {val:>10.4f}")


def plot_hub_attributions_paper(
    attrs:          dict,
    voted_features: dict,
    hub_dims:       list,
    feature_names:  dict,
    hub_labels:     dict = None,
    hub_colors:     dict = None,
    ensembl_to_symbol: dict = None,
    probe_to_gene:     dict = None,
    top_k:          int  = 15,
    save_path:      str  = None,
) -> None:
    """
    Publication IG attribution figure.

    Layout: rows = modalities (rna, mirna, methyl), columns = hub dims.
    Stable (majority-voted) features are highlighted with a bold border.

    Args:
        attrs             : output of compute_hub_ig_all_folds() voted_attrs
        voted_features    : dict (mod, hub_dim) -> stable feature names
        hub_dims          : hub dimension indices (from Jacobian analysis)
        feature_names     : dict mod -> list of raw feature name strings
        hub_labels        : dict hub_dim -> display label; defaults to "Hub dim N"
        hub_colors        : dict hub_dim -> hex colour; auto-assigned if None
        ensembl_to_symbol : output of build_rna_symbol_map(); used for RNA display names
        probe_to_gene     : output of build_probe_gene_map(); used for methylation display
        top_k             : max features to show per subplot
        save_path         : file path to save figure; None = display only
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    default_colors = ["#534AB7", "#185FA5", "#C0392B", "#3B6D11"]
    if hub_colors is None:
        hub_colors = {hd: default_colors[i % len(default_colors)]
                      for i, hd in enumerate(hub_dims)}
    if hub_labels is None:
        hub_labels = {hd: f"Hub dim {hd}" for hd in hub_dims}
    else:
        for hd in hub_dims:
            if hd not in hub_labels:
                hub_labels[hd] = f"Hub dim {hd}"

    modality_display = {"rna": "mRNA", "mirna": "miRNA", "methyl": "Methylation"}

    n_rows      = len(_MODALITY_KEYS)
    n_cols      = len(hub_dims)
    row_heights = [3.5, 4.0, 5.5]
    fig, axes   = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, sum(row_heights)),
        gridspec_kw={"height_ratios": row_heights},
        constrained_layout=True,
    )
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, mod in enumerate(_MODALITY_KEYS):
        feat_list = feature_names.get(mod, [])
        for col, hub_dim in enumerate(hub_dims):
            ax    = axes[row, col]
            color = hub_colors[hub_dim]
            attr  = attrs[mod][col]

            if attr is None:
                ax.set_visible(False)
                continue

            abs_attr   = np.abs(attr)
            stable_set = set(voted_features.get((mod, hub_dim), []))

            stable_idx = sorted(
                [j for j, name in enumerate(feat_list) if name in stable_set],
                key=lambda j: -abs_attr[j],
            )
            all_top_idx   = np.argsort(abs_attr)[::-1]
            nonstable_idx = [j for j in all_top_idx if feat_list[j] not in stable_set]

            n_fill      = max(0, top_k - len(stable_idx))
            display_idx = stable_idx + nonstable_idx[:n_fill]
            n_bars      = len(display_idx)

            top_vals       = abs_attr[display_idx][::-1]
            raw_names_asc  = [feat_list[j] for j in display_idx][::-1]
            display_names  = [
                resolve_feature_name(mod, n, ensembl_to_symbol, probe_to_gene)
                for n in raw_names_asc
            ]

            bars = ax.barh(
                range(n_bars), top_vals,
                color=color, alpha=0.65, height=0.72, edgecolor="none",
            )

            for i, (bar, raw) in enumerate(zip(bars, raw_names_asc)):
                if raw in stable_set:
                    bar.set_edgecolor(color)
                    bar.set_linewidth(1.8)
                    bar.set_alpha(0.92)
                else:
                    bar.set_alpha(0.45 if i % 2 == 0 else 0.60)

            ax.set_yticks(range(n_bars))
            ax.set_yticklabels(display_names, fontsize=7.5)
            for tick, raw in zip(ax.get_yticklabels(), raw_names_asc):
                if raw in stable_set:
                    tick.set_fontweight("bold")
                    tick.set_color(color)

            ax.tick_params(axis="x", labelsize=7.5)
            ax.set_xlabel("|IG attribution|", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.grid(axis="x", alpha=0.2, linestyle="--")
            ax.tick_params(axis="y", length=0)

            if row == 0:
                ax.set_title(hub_labels[hub_dim], fontsize=9, fontweight="bold",
                             color=color, pad=10)
            if col == 0:
                ax.set_ylabel(modality_display[mod], fontsize=10,
                              fontweight="bold", labelpad=10)

    legend_elements = [
        mpatches.Patch(facecolor="#555555", alpha=0.90, linewidth=1.8,
                       edgecolor="#555555", label="Stable feature (>= min_folds)"),
        mpatches.Patch(facecolor="#555555", alpha=0.45, edgecolor="none",
                       label="Top by mean attribution"),
    ]
    fig.legend(handles=legend_elements, fontsize=8, loc="lower center", ncol=2,
               framealpha=0.9, edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "Integrated Gradients: stable features activating cross-modal hub dimensions\n"
        "(bold labels and solid borders = majority-vote stable across folds)",
        fontsize=10, fontweight="bold",
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
