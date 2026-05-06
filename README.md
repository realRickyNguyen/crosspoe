# CrossPoE

**CrossPoE: Task-Signal-Preserving Latent Translation for Robust Multimodal Learning Under Modality Missingness.**

We present CrossPoE, a Product-of-Experts variational autoencoder with six directional latent cross-modal translation heads for cancer progression-free interval survival prediction under block-wise missingness. When a modality is absent, CrossPoE synthesises a survival-calibrated pseudo-posterior from observed modalities and injects it into the PoE fusion. Unlike standard latent translation, the translation heads are trained with a survival-preserving objective so that pseudo-posteriors retain prognostic signal rather than only matching latent geometry.

---

## Table of Contents

1. [Installation](#installation)
2. [Data Format Requirements](#data-format-requirements)
3. [Quick Start](#quick-start)
4. [Reproducing TCGA Results](#reproducing-tcga-results)
5. [Adapting to New Datasets](#adapting-to-new-datasets)
6. [Configuration Reference](#configuration-reference)
7. [Training Output](#training-output)
8. [MCAR Robustness Evaluation](#mcar-robustness-evaluation)
9. [Baseline Comparisons](#baseline-comparisons)
10. [Statistical Testing](#statistical-testing)
11. [Visualisation](#visualisation)
12. [Alpha Selection (Nested CV)](#alpha-selection-nested-cv)
13. [Post-Training Analysis](#post-training-analysis)
14. [Repository Structure](#repository-structure)
15. [Model Overview](#model-overview)
16. [Loss Function](#loss-function)

---

## Installation

**Python 3.10.4 (recommended).**

```bash
git clone https://github.com/realRickyNguyen/crosspoe.git
cd crosspoe
pip install -r requirements.txt
```

Core dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.7.1 | Model training |
| `numpy` | 2.2.6 | Numerical arrays |
| `pandas` | 2.3.0 | CSV loading and alignment |
| `scikit-learn` | 1.6.1 | Stratified CV splits |
| `lifelines` | 0.30.0 | Harrell's C-index |
| `matplotlib` | 3.10.3 | Analysis plots *(optional)* |
| `captum` | 0.8.0 | Integrated Gradients *(optional)* |
| `mygene` | 3.2.2 | Ensembl → gene symbol lookup *(optional)* |

---

## Data Format Requirements

CrossPoE requires four CSV files. This section documents every assumption the code makes about their format.

### Omics CSVs (mRNA, miRNA, methylation)

All three follow the same structure:

```
sampleID,feature_1,feature_2,...,feature_N
TCGA-XX-XXXX-01,0.341,1.203,...
TCGA-YY-YYYY-01,0.892,0.014,...
```

| Requirement | Detail |
|-------------|--------|
| **Orientation** | Rows = samples, columns = features |
| **Index column** | The **first column** is the sample identifier. It can have any column name — the code reads it as `pd.read_csv(..., index_col=0)`. |
| **Values** | Must be finite numbers. No NaN or Inf within a sample's row. |
| **Missing modality** | A sample that does not appear in a CSV at all is treated as **missing** for that modality — this is the intended way to represent partial observations. Do not add placeholder rows for missing samples. |
| **Minimum coverage** | Each sample must have at least one modality observed. The code takes the **union** of samples across the three omics files (intersected with clinical). |

**Methylation-specific:** values must be **beta values in [0, 1]**. The code applies a logit (M-value) transform automatically before training:

```
M = log2( beta / (1 - beta) )
```

**mRNA and miRNA:** any numeric scale is accepted (TPM, FPKM, read counts, log-counts). The code z-scores each feature per fold using training-set mean and standard deviation — no pre-normalisation required.

---

### Clinical CSV

```
sampleID,PFI,PFI_time,...
TCGA-XX-XXXX-01,1,365,...
TCGA-YY-YYYY-01,0,1820,...
```

| Column | Type | Description |
|--------|------|-------------|
| `sampleID` | string | Must match the sample IDs used in the omics files. **Column name is case-sensitive.** |
| `PFI` | int (0 or 1) | Progression-free interval event indicator: `1` = event (progression/death), `0` = censored. |
| `PFI_time` | float | Time to event or censoring in any consistent unit (days, months). |

- All other columns are ignored.
- Samples absent from the clinical file, or with `NaN` in `PFI`/`PFI_time`, are still used for the unsupervised objectives (reconstruction, KL, translation) but are **excluded from C-index computation** (`pfi_event = -1` internally).
- There is no minimum number of events required to run, but C-index is only meaningful with ≥ 2 distinct event times.

---

### Sample ID alignment

The code aligns samples across files as follows:

```
usable_samples = (rna_samples ∪ mirna_samples ∪ methyl_samples) ∩ clinical_samples
```

- A sample only needs to appear in **one** omics file to be included.
- The alignment is done by exact string matching on sample IDs. IDs must be consistent across files.
- For TCGA data: IDs are typically full barcodes (`TCGA-BH-A0BZ-01A`). Make sure all files use the same barcode format.

---

### Checklist before running

- [ ] First column of each omics CSV contains sample IDs (same format across all files)
- [ ] Methylation values are in [0, 1] (raw beta values, not M-values)
- [ ] Clinical CSV has columns named exactly `sampleID`, `PFI`, `PFI_time`
- [ ] No NaN/Inf within any sample row that is present in a CSV
- [ ] No duplicate sample IDs within a single file

---

## Quick Start

### 1. Train

```bash
python train.py \
  --rna-path    /path/to/rna.csv \
  --mirna-path  /path/to/mirna.csv \
  --methyl-path /path/to/methylation.csv \
  --clin-path   /path/to/clinical.csv \
  --out-dir     results/
```

Optional flags:

```
  --seed INT      Global random seed (default: 0)
  --out-dir PATH  Directory to save outputs (default: results/)
```

Training saves two files to `--out-dir`:

| File | Contents |
|------|----------|
| `fold_results.pt` | List of 5 fold dicts — model weights, scalers, best C-index |
| `config.pt` | Config dict used for training |

The model automatically infers feature dimensions from the CSV column counts.

### 2. Load results for downstream tasks

All downstream tasks (MCAR, Jacobian, plots, baselines) take `fold_results` and `cfg` as inputs. Load them once:

```python
import torch
from data.dataset import MultiOmicsDataset, load_data
import pandas as pd

# Reload data so MultiOmicsDataset class attributes are populated
load_data(
    rna_path="/path/to/rna.csv",
    mirna_path="/path/to/mirna.csv",
    methyl_path="/path/to/methylation.csv",
    clin_path="/path/to/clinical.csv",
)
clinical_df = pd.read_csv("/path/to/clinical.csv", low_memory=False)
MultiOmicsDataset._prepare_survival_labels(clinical_df, MultiOmicsDataset._sample_ids)

# Load saved results
fold_results = torch.load("results/fold_results.pt", weights_only=False)
cfg          = torch.load("results/config.pt",       weights_only=False)
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

> **Why reload data?** `MultiOmicsDataset` stores omics matrices as class-level attributes. These are in memory during training but must be repopulated from disk in a new Python session before running any downstream code that references them.

### 3. Run downstream tasks

With `fold_results`, `cfg`, and `device` in hand, any downstream task is a single function call — see the sections below for examples.

---

## Reproducing TCGA Results

The default configuration was developed on TCGA BRCA multi-omics data from UCSC Xena.

### Downloading TCGA data from UCSC Xena

1. Go to [xenabrowser.net/datapages](https://xenabrowser.net/datapages/)
2. Select the **GDC TCGA Breast Cancer (BRCA)** cohort (or any other TCGA cohort)
3. Download the following datasets:

| File | Xena dataset name | Notes |
|------|-------------------|-------|
| mRNA | `TCGA-BRCA.star_tpm.tsv` | TPM values; transpose so rows = samples |
| miRNA | `TCGA-BRCA.mirna.tsv` | reads_per_million_miRNA_mapped |
| Methylation | `TCGA-BRCA.methylation450.tsv` | beta values |
| Clinical | `TCGA-BRCA.survival.txt` | contains `PFI` and `PFI.time` columns |

**Note:** Xena files are often feature × sample (transposed). Transpose them so rows = samples. Also note that Xena's clinical file uses `PFI.time` (with a dot) — rename it to `PFI_time` and rename `sample` to `sampleID` before running.

4. Optionally filter features (low-variance genes, etc.) before passing to the model. The default architecture expects the dimensions in `config.py` but **adapts automatically** to any input size — see [Adapting to New Datasets](#adapting-to-new-datasets).

### Exact training command

```bash
python train.py \
  --rna-path    data/TCGA_BRCA_rna.csv \
  --mirna-path  data/TCGA_BRCA_mirna.csv \
  --methyl-path data/TCGA_BRCA_methylation.csv \
  --clin-path   data/TCGA_BRCA_clinical.csv \
  --out-dir     results/tcga_brca \
  --seed 0
```

Training runs 5-fold stratified cross-validation (stratified on PFI event). At default settings with a GPU this takes approximately 3 minutes (TCGA-BRCA). Results are saved to `results/tcga_brca/fold_results.pt` and `results/tcga_brca/config.pt`.

### Full workflow example

```python
import torch
import pandas as pd
from data.dataset import MultiOmicsDataset, load_data
from training.mcar import run_mcar_crossPoe
from analysis import compute_translation_jacobians_all_folds, print_majority_vote_summary
from analysis.plots import plot_kaplan_meier, plot_mcar_comparison

# --- Load data (required in every new Python session) ---
load_data(
    rna_path="data/TCGA_BRCA_rna.csv",
    mirna_path="data/TCGA_BRCA_mirna.csv",
    methyl_path="data/TCGA_BRCA_methylation.csv",
    clin_path="data/TCGA_BRCA_clinical.csv",
)
clinical_df = pd.read_csv("data/TCGA_BRCA_clinical.csv", low_memory=False)
MultiOmicsDataset._prepare_survival_labels(clinical_df, MultiOmicsDataset._sample_ids)

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fold_results = torch.load("results/tcga_brca/fold_results.pt", weights_only=False)
cfg          = torch.load("results/tcga_brca/config.pt",       weights_only=False)

# --- MCAR robustness ---
mcar_results = run_mcar_crossPoe(fold_results, cfg, device)

# --- Jacobian hub analysis ---
jacobians_mean, jacobians_std, fold_jacs = compute_translation_jacobians_all_folds(
    fold_results, cfg, device
)
hub_dims = print_majority_vote_summary(fold_jacs, jacobians_mean, top_k=8, min_folds=4)

# --- Kaplan-Meier plot ---
plot_kaplan_meier(fold_results, cfg, device, save_path="results/tcga_brca/kaplan_meier.svg")
```


---

## Adapting to New Datasets

CrossPoE works with any three-modality multi-omics dataset with survival labels. No architecture changes are needed — the model is instantiated with `n_rna`, `n_mirna`, `n_methyl` derived directly from the number of columns in your CSVs.

**Be aware** that we do use reconstruction weights `RECON_WEIGHTS` in `compute_loss()` for each modality, and assume that miRNA will have the lowest features, and methylation having the highest, so weights will be adjusted to that. Additionally the encoder/decoder architectures are different for miRNA, and methylation has a higher dropout 0.4. So adjust accordingly to the specific dataset and number of features per modalitity. 

**Steps:**

1. Prepare your four CSV files following the [Data Format Requirements](#data-format-requirements).
2. Run `train.py` pointing to your files. That's it.

The encoder/decoder architecture automatically scales to your feature dimensions. If your dataset is much larger or smaller than TCGA BRCA, you may want to tune the hidden layer sizes in `models/encoders.py` and `models/decoders.py`, or adjust the loss weights in `config.py`.

**To use a different survival endpoint:** replace the `PFI`/`PFI_time` columns in your clinical file with your endpoint (OS, DFS, etc.) using the same 0/1 event and numeric time format.

---

## Configuration Reference

All hyperparameters are in `config.py`. Edit the file directly before training.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_latent` | 48 | Shared latent space dimension |
| `n_folds` | 5 | Number of cross-validation folds |
| `seed` | 0 | Global random seed for reproducibility |
| `n_epochs` | 100 | Maximum epochs per fold |
| `min_epochs` | 30 | Earliest epoch at which early stopping can trigger |
| `patience` | 15 | Early-stopping patience (val C-index epochs without improvement) |
| `batch_size` | 128 | Training batch size |
| `learning_rate` | 1e-3 | AdamW learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `kl_warmup_epochs` | 30 | Epochs to linearly anneal KL weight from 0 → 1 |
| `translation_warmup_epochs` | 10 | Epoch at which translation losses activate |
| `translation_hidden_dim` | 96 | Hidden dim of the translation head MLPs |
| `lambda_survival` | 0.1 | Cox PH loss weight |
| `lambda_recon` | 0.1 | Reconstruction loss weight |
| `lambda_consist` | 0.1 | Modality-to-PoE KL consistency weight |
| `lambda_translation` | 0.2 | W2 translation consistency loss weight |
| `lambda_cycle` | 0.1 | Cycle consistency loss weight |
| `alpha` | 0.75 | Fixed gate weight applied to pseudo-posterior contributions in PoE fusion |
| `lambda_surv_trans` | 0.05 | Translation survival-preservation weight |

---

## Training Output

Training prints per-epoch metrics for each fold:

```
Epoch  42/100 | beta=1.00 | Train: 0.4821 (surv=0.041 kl=0.183 recon=0.089 trans=0.031 cycle=0.012 surv_trans=0.004) | Val C-index: 0.6213
```

After all folds, a cross-validation summary is printed:

```
==============================
CROSS-VALIDATION SUMMARY
==============================
C-index: 0.6180 ± 0.0241

Per-fold C-index:
  Fold 1: 0.6312  (best epoch 78)
  Fold 2: 0.5994  (best epoch 65)
  ...
```

**`fold_results`** is the return value of `run_cross_validation()` — a list of dicts, one per fold. Each dict contains:

| Key | Content |
|-----|---------|
| `fold` | Fold number (1-based) |
| `best_epoch` | Epoch at which the best C-index was achieved |
| `best_c_index` | Best validation C-index for this fold |
| `val_metrics` | Final `{"loss": ..., "c_index": ...}` at best checkpoint |
| `model_state` | `{"model": ..., "translator": ..., "survival": ...}` state dicts (CPU tensors) |
| `scalers` | `{"rna": (mean, std), "mirna": ..., "methyl": ...}` — training-set scalers per fold |

`train.py` saves `fold_results` automatically to `--out-dir/fold_results.pt`. Load it in any downstream script with `torch.load("results/fold_results.pt", weights_only=False)`. See [Quick Start](#quick-start) for the complete load pattern.

---

## MCAR Robustness Evaluation

After training, evaluate how C-index degrades as modalities are artificially forced missing at varying rates.

**For CrossPoE** (with `CrossModalTranslator`):

```python
from training.mcar import run_mcar_crossPoe

results = run_mcar_crossPoe(
    fold_results,
    cfg=CONFIG,
    device=device,
    missing_rates=[0.0, 0.3, 0.5, 0.7, 0.9],  # default
    modalities=["rna", "mirna", "methyl"],       # default
)
# results[modality][rate] -> list of C-index values, one per fold
```

**For baselines** (CrossAE, PoE-VAE, MoE-VAE, etc. — no translator):

```python
from training.mcar import run_mcar_single

results = run_mcar_single(
    fold_results,
    cfg=CONFIG,
    device=device,
)
```

Both functions force-drop each modality independently at each rate and report mean ± std C-index across folds.

---

## Baseline Comparisons

Four baseline models are implemented in `baselines/` for comparison with CrossPoE. All baselines use the same encoder/decoder architectures, survival head, CV splits, and scalers as CrossPoE.

| Baseline | Reference | Key difference |
|----------|-----------|---------------|
| `VanillaPoE` | Wu & Goodman (2018) | Pure PoE fusion; recon + KL + Cox only |
| `MVAE` | Shi et al. (2019) | Mixture-of-Experts fusion (law of total variance) |
| `CLUE` | Tu et al., NeurIPS 2022 | Cross-encoders for latent consistency; no PoE |
| `HEALNetOmicsLite` | HEALNet-style | Cross-attention fusion; deterministic; Cox only |

```python
from baselines import run_vanilla_poe, run_mcar_vanilla_poe
from baselines import run_mvae, run_mcar_mvae
from baselines import run_clue, run_mcar_clue
from baselines import run_healnet_omics_lite, run_mcar_healnet_omics_lite

# Train
fold_results_vanilla  = run_vanilla_poe(CONFIG, device)
fold_results_mvae     = run_mvae(CONFIG, device)
fold_results_clue     = run_clue(CONFIG, device)       # uses cfg.get("lambda_cross", 0.1)
fold_results_healnet  = run_healnet_omics_lite(CONFIG, device)

# MCAR evaluation
mcar_vanilla  = run_mcar_vanilla_poe(fold_results_vanilla, CONFIG, device)
mcar_mvae     = run_mcar_mvae(fold_results_mvae,    CONFIG, device)
mcar_clue     = run_mcar_clue(fold_results_clue,    CONFIG, device)
mcar_healnet  = run_mcar_healnet_omics_lite(fold_results_healnet, CONFIG, device)
```

**HEALNet-OmicsLite notes:** This is a molecular-only HEALNet-style adaptation; it is not the original WSI + omics experiment. Additional config keys (all optional, with defaults): `healnet_n_latents` (16), `healnet_depth` (2), `healnet_n_heads` (4), `healnet_dropout` (0.1), `healnet_ff_mult` (4).

**CLUE notes:** Add `"lambda_cross": 0.1` to your config dict (it defaults to 0.1 via `cfg.get`).

---

## Statistical Testing

Bootstrap confidence intervals for comparing models on MCAR robustness and full-data C-index.

```python
import numpy as np
from analysis.statistics import (
    bootstrap_ci_decline_diff,
    bootstrap_ci_cindex_diff,
    get_fold_declines_single,
)

# --- MCAR robustness comparison ---
# Extract per-fold declines (C-index at rate=0.9 minus rate=0.0)
crosspoe_declines = get_fold_declines_single(mcar_crosspoe, mod="rna")
baseline_declines = get_fold_declines_single(mcar_vanilla,  mod="rna")

obs, lo, hi = bootstrap_ci_decline_diff(crosspoe_declines, baseline_declines)
excludes_zero = "excludes 0 *" if lo > 0 else "includes 0"
print(f"CrossPoE vs PoE-VAE: obs={obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {excludes_zero}")
# Positive = CrossPoE declined less (better missingness robustness)

# --- Full-data C-index comparison ---
crosspoe_cidxs = np.array([fr["val_metrics"]["c_index"] for fr in fold_results])
vanilla_cidxs  = np.array([fr["val_metrics"]["c_index"] for fr in fold_results_vanilla])

obs, lo, hi = bootstrap_ci_cindex_diff(crosspoe_cidxs, vanilla_cidxs)
excludes_zero = "excludes 0 *" if lo > 0 else "includes 0"
print(f"CrossPoE vs PoE-VAE: obs={obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {excludes_zero}")
# Positive = CrossPoE has higher C-index
```

All bootstrap tests use 10,000 resamples with seed=42 by default.

---

## Visualisation

Three publication-ready plots are in `analysis/plots.py`.

### Kaplan-Meier risk stratification

```python
from analysis.plots import plot_kaplan_meier

km_results = plot_kaplan_meier(
    fold_results, CONFIG, device,
    save_path="kaplan_meier.svg",
)
# Returns dict with risk scores, time, event, high_risk_mask, p_value
```

Splits all validation patients (across all folds) at median risk score. Reports log-rank test p-value and at-risk counts. Requires `lifelines` (`pip install lifelines`).

### MCAR comparison plot

```python
from analysis.plots import plot_mcar_comparison

plot_mcar_comparison(
    mcar_crossPoe=mcar_crosspoe,
    mcar_crossae=mcar_clue,
    mcar_moevae=mcar_mvae,
    mcar_vanilla=mcar_vanilla,
    mcar_healnet_lite=mcar_healnet,
    save_path="mcar_comparison.png",
)
```

Three-panel figure showing mean C-index ± 1 SD across 5 folds for each model and modality at each missingness rate.

### Forest plot (univariate Cox results)

```python
from analysis.plots import plot_forest_survival

results_surv = [
    {"feature": "CCDC9B (RNA) — hubs 6, 17, 37",                       "hr": 1.45, "hr_lo": 1.12, "hr_hi": 1.87, "pval": 0.004},
    {"feature": "cg18149657 → HAMP (methylation) — hubs 17, 37",       "hr": 0.71, "hr_lo": 0.55, "hr_hi": 0.92, "pval": 0.009},
    {"feature": "cg13782615 → STIM1 (methylation) — hubs 12, 17",      "hr": 1.38, "hr_lo": 1.08, "hr_hi": 1.76, "pval": 0.010},
    {"feature": "EIF4EBP1 (RNA) — hub 37",                             "hr": 1.52, "hr_lo": 1.18, "hr_hi": 1.96, "pval": 0.001},
    {"feature": "cg02732941 → TSHZ2 (methylation) — hubs 6, 37",      "hr": 0.68, "hr_lo": 0.52, "hr_hi": 0.89, "pval": 0.005},
]
plot_forest_survival(results_surv, save_path="survival_forest.pdf")
```

Feature keys must match the hardcoded `feature_meta` dict in `analysis/plots.py` exactly; edit that dict to display different features.

---

## Alpha Selection (Nested CV)

To empirically select the `alpha` hyperparameter (rather than using the default 0.75), use nested 5-fold CV with an inner 4-fold CV that minimises MCAR RNA decline.

```python
from scripts.nested_cv_alpha import run_cross_validation_nested_alpha

fold_results_nested, selected_alphas = run_cross_validation_nested_alpha(
    cfg=CONFIG,
    device=device,
    alphas=[0.1, 0.25, 0.5, 0.75, 1.0],  # default grid
)
print("Selected alphas per fold:", selected_alphas)
```

This is computationally expensive (5 outer folds × 5 alpha candidates × 4 inner folds × full training each). It is provided for reproducibility of ablation experiments.

---

## Post-Training Analysis

### Jacobian Analysis

Identifies *hub* latent dimensions — those with high cross-modal influence in the translation heads — via a two-condition majority vote across folds.

```python
from analysis import (
    compute_translation_jacobians_all_folds,
    print_majority_vote_summary,
    plot_jacobian_paper,
)

# 1. Compute mean Jacobians across all folds
jacobians_mean, jacobians_std, fold_jacs = compute_translation_jacobians_all_folds(
    fold_results, CONFIG, device
)

# 2. Identify hub dims — HUB_DIMS flows from this call, never hardcoded
HUB_DIMS = print_majority_vote_summary(
    fold_jacs, jacobians_mean, top_k=8, min_folds=4, min_global_appearances=4
)

# 3. Visualise
plot_jacobian_paper(jacobians_mean, hub_dims=HUB_DIMS)
```

### Integrated Gradients

Computes per-modality input-feature attributions for each hub dimension using Captum, with majority-vote stable feature identification across folds.

```python
from analysis import (
    build_rna_symbol_map, build_probe_gene_map,
    get_feature_names,
    compute_hub_ig_all_folds,
    plot_hub_attributions_paper,
)
from data.dataset import MultiOmicsDataset

feature_names = get_feature_names(MultiOmicsDataset)

# Optional: resolve human-readable display names
ensembl_to_symbol = build_rna_symbol_map(feature_names["rna"])
probe_to_gene     = build_probe_gene_map(feature_names["methyl"], "/path/to/HM450_manifest.csv")

# hub_dims always comes from the Jacobian step — never hardcoded
hub_attrs, voted_features, _ = compute_hub_ig_all_folds(
    fold_results, CONFIG, device,
    hub_dims=HUB_DIMS, n_steps=50, top_k=50, min_folds=3,
)

plot_hub_attributions_paper(
    attrs=hub_attrs,
    voted_features=voted_features,
    hub_dims=HUB_DIMS,
    feature_names=feature_names,
    ensembl_to_symbol=ensembl_to_symbol,
    probe_to_gene=probe_to_gene,
    hub_labels={hd: f"Hub dim {hd}" for hd in HUB_DIMS},
    top_k=15,
)
```

The HM450 manifest CSV (for methylation probe → gene mapping) is available from Illumina or UCSC Xena. The file should be in standard Illumina format with a 7-row header and a `UCSC_RefGene_Name` column.

---

## Repository Structure

```
crosspoe/
│
├── train.py                        # Entry point: 5-fold cross-validation training
├── config.py                       # Hyperparameters (CONFIG) and N_LATENT
├── requirements.txt
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                  # MultiOmicsDataset, load_data(), collate_fn()
│   ├── preprocessing.py            # beta_to_mvalue(), compute_scaler(), apply_scaler()
│   └── mcar.py                     # MCARDataset (forced modality missingness)
│
├── models/
│   ├── __init__.py
│   ├── blocks.py                   # fc_block(), count_params()
│   ├── encoders.py                 # RNAEncoder, MIRNAEncoder, MethylEncoder
│   ├── decoders.py                 # RNADecoder, MIRNADecoder, MethylDecoder
│   ├── poe.py                      # ProductOfExperts, reparameterise()
│   ├── survival.py                 # SurvivalHead, cox_partial_likelihood_loss()
│   ├── translation.py              # LatentTranslationHead, CrossModalTranslator
│   └── crosspoe.py                 # CrossPoE (main model), kl_divergence helpers
│
├── training/
│   ├── __init__.py
│   ├── losses.py                   # compute_loss(), translation_consistency_loss(),
│   │                               #   cycle_consistency_loss()
│   ├── utils.py                    # set_seed(), get_beta(), get_dropout_p(),
│   │                               #   move_batch_to_device(), concordance_index()
│   ├── trainer.py                  # train_one_epoch(), evaluate(), run_cross_validation()
│   └── mcar.py                     # evaluate_mcar(), run_mcar_crossPoe(), run_mcar_single()
│
├── baselines/
│   ├── __init__.py
│   ├── vanilla_poe.py              # VanillaPoE, run_vanilla_poe(), run_mcar_vanilla_poe()
│   │                               #   (Wu & Goodman 2018)
│   ├── moe_vae.py                  # MixtureOfExperts, MVAE, compute_loss_mvae(),
│   │                               #   run_mvae(), run_mcar_mvae()  (Shi et al 2019)
│   ├── cross_ae.py                 # CrossEncoder, CLUE, compute_loss_clue(),
│   │                               #   run_clue(), run_mcar_clue()  (Tu et al NeurIPS 2022)
│   └── healnet_lite.py             # HEALNetLiteCrossAttentionBlock,
│                                   #   HEALNetLiteLatentSelfAttentionBlock,
│                                   #   HEALNetOmicsLite, compute_loss_healnet_omics_lite(),
│                                   #   run_healnet_omics_lite(), run_mcar_healnet_omics_lite()
│
├── analysis/
│   ├── __init__.py
│   ├── features.py                 # build_rna_symbol_map(), build_probe_gene_map(),
│   │                               #   clean_mirna_name(), resolve_feature_name(),
│   │                               #   get_feature_names()
│   ├── jacobian.py                 # compute_translation_jacobians(),
│   │                               #   compute_translation_jacobians_all_folds(),
│   │                               #   get_majority_vote_hub_dims(),
│   │                               #   print_majority_vote_summary(),
│   │                               #   print_jacobian_summary(), plot_jacobian_paper()
│   ├── integrated_gradients.py     # compute_hub_ig(), compute_hub_ig_all_folds(),
│   │                               #   print_top_features(), plot_hub_attributions_paper()
│   ├── statistics.py               # bootstrap_ci_decline_diff(),
│   │                               #   bootstrap_ci_cindex_diff(),
│   │                               #   get_fold_declines_single()
│   └── plots.py                    # plot_kaplan_meier(), plot_forest_survival(),
│                                   #   plot_mcar_comparison()
│
└── scripts/
    ├── __init__.py
    └── nested_cv_alpha.py          # run_cross_validation_nested_alpha(),
                                    #   _eval_mcar_single()
```

---

## Model Overview

### Encoders

Each omics modality has a dedicated VAE encoder that maps high-dimensional features to a low-dimensional Gaussian posterior `(mu, logvar)`. These defaults are for TCGA BRCA; the model adapts to any input size automatically.

| Modality | Input dim | Architecture |
|----------|-----------|--------------|
| mRNA | 4,652 | Linear(in, 512) → 256 → 96 → (mu, logvar) |
| miRNA | 524 | Linear(in, 128) → 96 → (mu, logvar) |
| Methylation | 37,482 | Linear(in, 512, dropout=0.4) → 256 → 96 → (mu, logvar) |

All layers use LayerNorm + GELU + Dropout. Methylation uses higher input dropout (0.4 vs 0.2–0.3) to handle the high-dimensional input.

### Product of Experts Fusion

Observed modality posteriors are fused via precision-weighted PoE:

```
precision_sum = sum_m( exp(-logvar_m) )                    for observed m
mu_poe        = sum_m( mu_m * exp(-logvar_m) ) / precision_sum
```

Modalities with lower variance (higher confidence) contribute more to the fused posterior. A prior N(0, I) is always included, providing a regularising anchor when few modalities are observed.

### Cross-Modal Translation

A `CrossModalTranslator` manages all 6 pairwise translation directions (RNA↔miRNA, RNA↔Methyl, miRNA↔Methyl). For a sample missing modality B but observed in A, a `LatentTranslationHead` maps `(mu_A, logvar_A) → (mu_pseudo_B, logvar_pseudo_B)`. A fixed scalar `alpha` (default 0.75) controls how much each pseudo-posterior influences the PoE fusion. Translation activates after `translation_warmup_epochs`.

### Survival Head

A linear `SurvivalHead` maps the PoE latent `z` to a scalar Cox PH risk score, trained with the negative partial log-likelihood. The model is evaluated on Harrell's C-index (via lifelines).

---

## Loss Function

```
L = λ_survival   · L_cox
  + λ_recon      · L_recon
  + β            · L_kl
  + λ_consist    · L_consist
  + λ_trans      · L_translation   (after warmup)
  + λ_cycle      · L_cycle         (after warmup)
  + λ_surv_trans · L_surv_trans    (after warmup)
```

| Term | Description |
|------|-------------|
| `L_cox` | Negative Cox partial log-likelihood on PFI |
| `L_recon` | MSE reconstruction (weighted by inverse feature-dim ratio) |
| `L_kl` | KL divergence of PoE posterior from N(0, I); annealed linearly during warmup |
| `L_consist` | KL from each modality posterior to the PoE posterior |
| `L_translation` | Wasserstein-2 between translated and real posteriors |
| `L_cycle` | Forward-backward cycle consistency in latent space |
| `L_surv_trans` | Translation survival-preservation (risk score matching) |
