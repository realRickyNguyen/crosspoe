# CrossPoE

**CrossPoE: Survival-Calibrated Latent Cross-Modal Translation for Multi-Omics Prognosis Under Block-Wise Missingness.**

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
9. [Post-Training Analysis](#post-training-analysis)
10. [Repository Structure](#repository-structure)
11. [Model Overview](#model-overview)
12. [Loss Function](#loss-function)

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

```bash
python train.py \
  --rna-path    /path/to/rna.csv \
  --mirna-path  /path/to/mirna.csv \
  --methyl-path /path/to/methylation.csv \
  --clin-path   /path/to/clinical.csv
```

Optional flags:

```bash
  --seed INT    Global random seed (default: 0)
```

The model automatically infers feature dimensions from the CSV column counts — no configuration changes are needed when switching datasets.

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
  --seed 0
```

Training runs 5-fold stratified cross-validation (stratified on PFI event). At default settings with a GPU this takes approximately 2–4 hours.

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
| `lambda_gate` | 0.0 | Gate supervision loss weight (0 = disabled) |
| `lambda_unimodal` | 0.05 | Per-modality reconstruction auxiliary weight |
| `lambda_surv_trans` | 0.05 | Translation survival-preservation weight |
| `use_modality_dropout` | False | Enable stochastic modality masking during training |
| `dropout_p_max` | 0.15 | Max per-modality dropout probability (when enabled) |
| `dropout_start_epoch` | 30 | Epoch to begin modality dropout curriculum |
| `dropout_ramp_epochs` | 30 | Epochs over which dropout ramps to `dropout_p_max` |

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

Save `fold_results` to disk with `torch.save(fold_results, "fold_results.pt")` for post-training analysis.

---

## MCAR Robustness Evaluation

After training, evaluate how C-index degrades as modalities are artificially forced missing at varying rates:

```python
from training.mcar import run_mcar_evaluation

results = run_mcar_evaluation(
    fold_results,
    cfg=CONFIG,
    device=device,
    missing_rates=[0.0, 0.3, 0.5, 0.7, 0.9],  # default
    modalities=["rna", "mirna", "methyl"],       # default
)
# results[modality][rate] -> list of C-index values, one per fold
```

This reproduces the standard MCAR benchmark: force-drop each modality independently at each rate and report mean ± std C-index across folds. Useful for comparing against baselines.

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
│   ├── translation.py              # LatentTranslationHead, TranslationGateNetwork,
│   │                               #   CrossModalTranslator
│   └── crosspoe.py                 # CrossPoE (main model), kl_divergence helpers
│
├── training/
│   ├── __init__.py
│   ├── losses.py                   # compute_loss(), translation_consistency_loss(),
│   │                               #   cycle_consistency_loss(), gate_supervision_loss()
│   ├── utils.py                    # set_seed(), get_beta(), get_dropout_p(),
│   │                               #   move_batch_to_device(), concordance_index()
│   ├── trainer.py                  # train_one_epoch(), evaluate(), run_cross_validation()
│   └── mcar.py                     # evaluate_mcar(), run_mcar_evaluation()
│
└── analysis/
    ├── __init__.py
    ├── features.py                 # build_rna_symbol_map(), build_probe_gene_map(),
    │                               #   clean_mirna_name(), resolve_feature_name(),
    │                               #   get_feature_names()
    ├── jacobian.py                 # compute_translation_jacobians(),
    │                               #   compute_translation_jacobians_all_folds(),
    │                               #   get_majority_vote_hub_dims(),
    │                               #   print_majority_vote_summary(),
    │                               #   print_jacobian_summary(), plot_jacobian_paper()
    └── integrated_gradients.py     # compute_hub_ig(), compute_hub_ig_all_folds(),
                                    #   print_top_features(), plot_hub_attributions_paper()
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

A `CrossModalTranslator` manages all 6 pairwise translation directions (RNA↔miRNA, RNA↔Methyl, miRNA↔Methyl). For a sample missing modality B but observed in A, a `LatentTranslationHead` maps `(mu_A, logvar_A) → (mu_pseudo_B, logvar_pseudo_B)`. A `TranslationGateNetwork` predicts a per-sample trust score (sigmoid output, initialised near zero) controlling how much the pseudo-posterior influences the PoE fusion. Translation activates after `translation_warmup_epochs`.

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
  + λ_gate       · L_gate          (after warmup)
  + λ_unimodal   · L_unimodal
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
| `L_gate` | Gate supervision against survival-risk discrepancy |
| `L_unimodal` | Per-modality reconstruction auxiliary loss |
| `L_surv_trans` | Translation survival-preservation (risk score matching) |
