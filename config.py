"""
Training configuration for CrossPoE.

N_LATENT : shared latent space dimensionality (architecture hyperparameter)
CONFIG   : hyperparameters consumed by run_cross_validation()
"""

N_LATENT = 48

CONFIG = {
    # ── Cross-validation ──────────────────────────────────────────────────────
    "n_folds":    5,
    "seed":       0,
    "min_epochs": 30,   # do not allow early stopping before this epoch
    "patience":   15,   # early-stopping patience (epochs without improvement)

    # ── Optimisation ─────────────────────────────────────────────────────────
    "n_epochs":      100,
    "batch_size":    128,
    "learning_rate": 1e-3,
    "weight_decay":  1e-4,
    "n_latent":      N_LATENT,

    # ── KL annealing ─────────────────────────────────────────────────────────
    "kl_warmup_epochs": 30,   # linearly anneal beta: 0 -> 1 over this many epochs

    # ── Cross-modal translation ───────────────────────────────────────────────
    "translation_warmup_epochs": 10,   # activate translation losses after this epoch
    "translation_hidden_dim":    96,

    # ── Loss weights ─────────────────────────────────────────────────────────
    "lambda_recon":       0.1,
    "lambda_consist":     0.1,
    "lambda_survival":    0.1,
    "lambda_translation": 0.2,
    "lambda_cycle":       0.1,
    "lambda_gate":        0.0,   # set > 0 to supervise gate networks
    "lambda_unimodal":    0.05,  # per-modality reconstruction auxiliary
    "lambda_surv_trans":  0.05,  # translation survival-preservation

    # ── Modality dropout curriculum ───────────────────────────────────────────
    "use_modality_dropout": False,  # enable stochastic modality masking during training
    "dropout_p_max":        0.15,
    "dropout_start_epoch":  30,
    "dropout_ramp_epochs":  30,
}
