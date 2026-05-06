import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from data.dataset import MultiOmicsDataset, collate_fn
from models.crosspoe import CrossPoE
from models.translation import CrossModalTranslator
from models.survival import SurvivalHead
from training.utils import move_batch_to_device


def plot_kaplan_meier(fold_results, cfg, device,
                      save_path="kaplan_meier.svg"):
    """
    Plot Kaplan-Meier survival curves for CrossPoE risk stratification.
    Uses all validation patients across all folds (no patient seen in training).
    High- vs low-risk split at median risk score.

    Args:
        fold_results: list of fold dicts from run_cross_validation()
        cfg:          config dict
        device:       torch.device
        save_path:    output file path

    Returns:
        dict with risk, time, event, high_risk_mask, p_value
    """
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts
    from lifelines.statistics import logrank_test

    n_latent = cfg.get("n_latent", 48)
    n_rna    = MultiOmicsDataset._rna_data.shape[1]
    n_mirna  = MultiOmicsDataset._mirna_data.shape[1]
    n_methyl = MultiOmicsDataset._methyl_data.shape[1]
    alpha    = cfg.get("alpha", 0.75)

    pfi_event_all = MultiOmicsDataset._pfi_event.numpy()
    pfi_time_all  = MultiOmicsDataset._pfi_time.numpy()
    strat_labels  = np.clip(pfi_event_all, 0, 1)

    skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True,
                          random_state=cfg["seed"])
    splits = list(skf.split(np.arange(len(strat_labels)), strat_labels))

    all_risk    = []
    all_val_idx = []

    for fold_idx, fr in enumerate(fold_results):
        _, val_idx = splits[fold_idx]

        model         = CrossPoE(latent_dim=n_latent, n_rna=n_rna,
                                 n_mirna=n_mirna, n_methyl=n_methyl).to(device)
        translator    = CrossModalTranslator(
            n_latent, hidden_dim=cfg["translation_hidden_dim"], alpha=alpha
        ).to(device)
        survival_head = SurvivalHead(n_latent).to(device)

        model.load_state_dict({k: v.to(device) for k, v in fr["model_state"]["model"].items()})
        translator.load_state_dict({k: v.to(device) for k, v in fr["model_state"]["translator"].items()})
        survival_head.load_state_dict({k: v.to(device) for k, v in fr["model_state"]["survival"].items()})
        model.eval()
        translator.eval()
        survival_head.eval()

        val_dataset = MultiOmicsDataset(
            indices=val_idx,
            rna_scaler=fr["scalers"]["rna"],
            mirna_scaler=fr["scalers"]["mirna"],
            methyl_scaler=fr["scalers"]["methyl"],
            dropout_probs=None,
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"] * 2,
                                shuffle=False, collate_fn=collate_fn, num_workers=0)

        fold_risk = []
        with torch.no_grad():
            for batch in val_loader:
                batch   = move_batch_to_device(batch, device)
                outputs = model(batch, translator=translator,
                                epoch=cfg["n_epochs"],
                                translation_warmup_epochs=cfg["translation_warmup_epochs"])
                risk = survival_head(outputs["z_surv"]).squeeze(-1).cpu().numpy()
                fold_risk.append(risk)

        all_risk.append(np.concatenate(fold_risk))
        all_val_idx.append(val_idx)

    all_risk    = np.concatenate(all_risk)
    all_val_idx = np.concatenate(all_val_idx)
    times       = pfi_time_all[all_val_idx]
    events      = pfi_event_all[all_val_idx]

    valid   = events >= 0
    risk_v  = all_risk[valid]
    time_v  = times[valid]
    event_v = events[valid].astype(bool)

    median_risk = np.median(risk_v)
    high_risk   = risk_v >= median_risk
    low_risk    = ~high_risk

    lr    = logrank_test(time_v[high_risk], time_v[low_risk],
                         event_v[high_risk], event_v[low_risk])
    p_val = lr.p_value
    p_str = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"

    fig, ax = plt.subplots(figsize=(7, 5))

    kmf_high = KaplanMeierFitter()
    kmf_low  = KaplanMeierFitter()
    kmf_high.fit(time_v[high_risk],  event_observed=event_v[high_risk],
                 label=f"High risk (n={high_risk.sum()})")
    kmf_low.fit(time_v[low_risk],   event_observed=event_v[low_risk],
                label=f"Low risk (n={low_risk.sum()})")

    kmf_high.plot_survival_function(ax=ax, color="#C62828", linewidth=2.0,
                                    ci_show=True, ci_alpha=0.12)
    kmf_low.plot_survival_function(ax=ax, color="#1565C0", linewidth=2.0,
                                   ci_show=True, ci_alpha=0.12)
    add_at_risk_counts(kmf_high, kmf_low, ax=ax, fontsize=9)

    ax.text(0.97, 0.97, p_str,
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))
    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("PFI Probability", fontsize=12)
    ax.set_title("CrossPoE Risk Stratification — TCGA-BRCA PFI\n"
                 "(Validation patients across all 5 folds)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="lower left")
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")
    print(f"Log-rank test: {p_str}")
    print(f"High-risk events: {event_v[high_risk].sum()}/{high_risk.sum()}")
    print(f"Low-risk events:  {event_v[low_risk].sum()}/{low_risk.sum()}")

    return {
        "risk": risk_v, "time": time_v, "event": event_v,
        "high_risk_mask": high_risk, "p_value": p_val,
    }


def plot_forest_survival(results, save_path="survival_forest.pdf"):
    """
    Forest plot for univariate Cox regression results.

    Args:
        results:   list of dicts, each with keys:
                   "feature" (str matching feature_meta keys),
                   "hr" (float), "hr_lo" (float), "hr_hi" (float), "pval" (float)
        save_path: output path (PDF; also saves SVG at same path)
    """
    from matplotlib import rcParams

    rcParams.update({
        "font.family":    "serif",
        "font.serif":     ["Times New Roman", "DejaVu Serif"],
        "axes.linewidth": 0.7,
        "pdf.fonttype":   42,
        "ps.fonttype":    42,
        "font.size":      12,
    })

    feature_meta = {
        "CCDC9B (RNA) — hubs 6, 17, 37": {
            "label":    "CCDC9B",
            "sublabel": "mRNA expression · hubs 6, 17, 37",
            "modality": "mRNA",
            "hubs":     "6, 17, 37",
            "color":    "#2166AC",
        },
        "cg18149657 → HAMP (methylation) — hubs 17, 37": {
            "label":    "HAMP",
            "sublabel": "cg18149657 · TSS methylation · hubs 17, 37",
            "modality": "Methylation",
            "hubs":     "17, 37",
            "color":    "#B2182B",
        },
        "cg13782615 → STIM1 (methylation) — hubs 12, 17": {
            "label":    "STIM1",
            "sublabel": "cg13782615 · TSS methylation · hubs 12, 17",
            "modality": "Methylation",
            "hubs":     "12, 17",
            "color":    "#B2182B",
        },
        "EIF4EBP1 (RNA) — hub 37": {
            "label":    "EIF4EBP1",
            "sublabel": "mRNA expression · hub 37",
            "modality": "mRNA",
            "hubs":     "37",
            "color":    "#2166AC",
        },
        "cg02732941 → TSHZ2 (methylation) — hubs 6, 37": {
            "label":    "TSHZ2",
            "sublabel": "cg02732941 · TSS methylation · hubs 6, 37",
            "modality": "Methylation",
            "hubs":     "6, 37",
            "color":    "#B2182B",
        },
    }

    plot_results = [r for r in results if r["feature"] in feature_meta]
    n = len(plot_results)
    if n == 0:
        print("No matching features found in results — check feature name strings.")
        return

    fig = plt.figure(figsize=(15, 1.8 * n + 2.5))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1.1],
                           left=0.01, right=0.99, top=0.86, bottom=0.13, wspace=0.03)
    ax_forest = fig.add_subplot(gs[0])
    ax_table  = fig.add_subplot(gs[1])
    ax_forest.set_facecolor("white")
    ax_table.set_facecolor("white")

    y_positions = list(range(n - 1, -1, -1))

    for ax in [ax_forest, ax_table]:
        for j, y in enumerate(y_positions):
            if j % 2 == 1:
                ax.axhspan(y - 0.55, y + 0.55, color="#F4F6F9", zorder=0, alpha=1.0)

    all_lo = [r.get("hr_lo", r["hr"] * 0.8) for r in plot_results]
    all_hi = [r.get("hr_hi", r["hr"] * 1.2) for r in plot_results]
    x_min  = min(all_lo) * 0.92
    x_max  = max(all_hi) * 1.08

    for r, y in zip(plot_results, y_positions):
        meta  = feature_meta[r["feature"]]
        hr    = r["hr"]
        lo    = r.get("hr_lo", hr * 0.8)
        hi    = r.get("hr_hi", hr * 1.2)
        color = meta["color"]
        sig   = ("***" if r["pval"] < 0.001 else
                 "**"  if r["pval"] < 0.01  else
                 "*"   if r["pval"] < 0.05  else "ns")

        ax_forest.barh(y, hi - lo, left=lo, height=0.22, color=color, alpha=0.18, zorder=2)
        ax_forest.plot([lo, hi], [y, y], color=color, linewidth=2.0,
                       solid_capstyle="round", zorder=3)
        for cap_x in [lo, hi]:
            ax_forest.plot([cap_x, cap_x], [y - 0.18, y + 0.18],
                           color=color, linewidth=1.8, zorder=3)
        ax_forest.scatter([hr], [y], color=color, s=90, zorder=4,
                          marker="D", edgecolors="white", linewidths=1.0)
        ax_forest.text(-0.02, y + 0.24, meta["label"],
                       transform=ax_forest.get_yaxis_transform(),
                       ha="right", va="bottom", fontsize=11, fontweight="bold", color=color)
        ax_forest.text(-0.02, y - 0.24, meta["sublabel"],
                       transform=ax_forest.get_yaxis_transform(),
                       ha="right", va="top", fontsize=7.5, color="#777777", style="italic")
        ax_forest.text(hi + (x_max - x_min) * 0.015, y, sig,
                       ha="left", va="center", fontsize=9, color=color, fontweight="bold")

    ax_forest.axvline(x=1.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.55, zorder=1)
    ax_forest.text(1.0, -0.65, "HR = 1", ha="center", va="top", fontsize=8, color="#555555")
    ax_forest.set_xlabel("Hazard Ratio (univariate Cox PH, 95% CI)", fontsize=9.5, labelpad=7)
    ax_forest.set_xlim(x_min, x_max)
    ax_forest.set_ylim(-0.85, n - 0.15)
    ax_forest.set_yticks([])
    ax_forest.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax_forest.tick_params(axis="x", labelsize=9)
    for spine in ["left", "right", "top"]:
        ax_forest.spines[spine].set_visible(False)
    ax_forest.spines["bottom"].set_linewidth(0.7)

    col_headers = ["Hubs", "HR (95% CI)", "p-value"]
    col_x       = [0.04, 0.30, 0.75]
    for cx, ch in zip(col_x, col_headers):
        ax_table.text(cx, n - 0.15, ch, ha="left", va="bottom",
                      fontsize=9.5, fontweight="bold", color="#222222")

    for r, y in zip(plot_results, y_positions):
        meta  = feature_meta[r["feature"]]
        hr    = r["hr"]
        lo    = r.get("hr_lo", hr * 0.8)
        hi    = r.get("hr_hi", hr * 1.2)
        pval  = r["pval"]
        sig   = ("***" if pval < 0.001 else "**" if pval < 0.01 else
                 "*"   if pval < 0.05  else "ns")
        color = meta["color"]
        ax_table.text(col_x[0], y, meta["hubs"], ha="left", va="center",
                      fontsize=9, color="#444444")
        ax_table.text(col_x[1], y, f"{hr:.2f} ({lo:.2f}–{hi:.2f})",
                      ha="left", va="center", fontsize=8.5, color="#222222",
                      family="monospace")
        pval_str = "p<0.001" if pval < 0.001 else f"p={pval:.3f}"
        ax_table.text(col_x[2], y, f"{pval_str} {sig}",
                      ha="left", va="center", fontsize=9, color=color, fontweight="bold")

    ax_table.set_xlim(0, 1.25)
    ax_table.set_ylim(-0.85, n - 0.15)
    ax_table.set_yticks([])
    ax_table.set_xticks([])
    for spine in ax_table.spines.values():
        spine.set_visible(False)
    ax_table.axvline(x=0.0, color="#CCCCCC", linewidth=0.8)

    fig.text(0.35, 0.04,
             "Methylation probes are promoter/TSS-associated: higher M-value = gene silencing. "
             "HR > 1 = worse PFI. HR < 1 = better PFI.",
             ha="center", va="bottom", fontsize=7.5, color="#666666", style="italic")
    legend_elements = [
        mpatches.Patch(color="#2166AC", label="mRNA expression"),
        mpatches.Patch(color="#B2182B", label="DNA methylation (M-value)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.35, 0.07))
    fig.suptitle(
        "Univariate Cox regression — hub-anchoring features independently validated (TCGA-BRCA PFI)",
        fontsize=10.5, fontweight="bold", x=0.35, y=0.93, ha="center", color="#222222",
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    svg_path = save_path.replace(".pdf", ".svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {save_path}")
    print(f"Saved: {svg_path}")
    plt.show()


def plot_mcar_comparison(mcar_crossPoe, mcar_crossae, mcar_moevae,
                         mcar_vanilla, mcar_healnet_lite,
                         save_path="mcar_comparison.png"):
    """
    Plot MCAR robustness comparison across models and modalities.

    Args:
        mcar_crossPoe:     dict from run_mcar_crossPoe()
        mcar_crossae:      dict from run_mcar_clue()
        mcar_moevae:       dict from run_mcar_mvae()
        mcar_vanilla:      dict from run_mcar_vanilla_poe()
        mcar_healnet_lite: dict from run_mcar_healnet_omics_lite()
        save_path:         output file path
    """
    rates      = [0.0, 0.3, 0.5, 0.7, 0.9]
    modalities = ["rna", "mirna", "methyl"]
    mod_labels = {
        "rna":    "mRNA Missing",
        "mirna":  "miRNA Missing",
        "methyl": "Methylation Missing",
    }
    models = [
        (mcar_crossPoe,      "CrossPoE",      "#1565C0", "-",  2.5, "o", 0.15),
        (mcar_crossae,       "CrossAE",       "#2E7D32", "--", 2.0, "s", 0.12),
        (mcar_moevae,        "MoE-VAE",       "#E65100", "-.", 2.0, "^", 0.10),
        (mcar_vanilla,       "PoE-VAE",       "#757575", "--", 1.5, "x", 0.10),
        (mcar_healnet_lite,  "HEALNet-Omics", "#b7950b", ":",  1.5, "D", 0.10),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax_idx, mod in enumerate(modalities):
        ax = axes[ax_idx]
        for results, name, color, ls, lw, marker, alpha in models:
            means = np.array([np.nanmean(results[mod][r]) for r in rates])
            stds  = np.array([
                np.nanstd(results[mod][r], ddof=1)
                if len([x for x in results[mod][r] if not np.isnan(x)]) > 1
                else 0.0
                for r in rates
            ])
            ax.fill_between(rates, means - stds, means + stds,
                            color=color, alpha=alpha, linewidth=0)
            ax.plot(rates, means, label=name, color=color, linestyle=ls,
                    linewidth=lw, marker=marker, markersize=6,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=3)

        ax.set_title(mod_labels[mod], fontsize=12, fontweight="bold")
        ax.set_xlabel("Forced Missingness Rate", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Mean C-index (5-fold CV)", fontsize=11)
        ax.set_xticks(rates)
        ax.set_xticklabels([f"{r:.1f}" for r in rates])
        ax.set_xlim(-0.05, 0.95)
        ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
        ax.tick_params(labelsize=10)

        crosspoe_vals = [np.nanmean(mcar_crossPoe[mod][r]) for r in rates]
        ax.annotate(f"{crosspoe_vals[-1]:.3f}",
                    xy=(0.9, crosspoe_vals[-1]),
                    xytext=(0.78, crosspoe_vals[-1] + 0.008),
                    fontsize=8, color="#1565C0", fontweight="bold")

    fig.text(0.5, 1.01,
             "Shaded regions show ±1 SD across 5 cross-validation folds",
             ha="center", fontsize=9, color="#555555", style="italic")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.08), fontsize=10,
               framealpha=0.9, edgecolor="#CCCCCC")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")
