import numpy as np


def bootstrap_ci_decline_diff(arr1, arr2, n_boot: int = 10000, seed: int = 42,
                               ci: int = 95):
    """
    Bootstrap CI for mean(arr1) - mean(arr2) on fold-level MCAR declines.
    Positive = arr1 declined less than arr2 (better missingness robustness).

    Args:
        arr1, arr2: numpy arrays of fold-level decline values (ci_0.9 - ci_0.0)
        n_boot:     number of bootstrap resamples
        seed:       RNG seed
        ci:         confidence interval percentage (default 95)

    Returns:
        (observed_diff, lower_bound, upper_bound)
    """
    rng   = np.random.RandomState(seed)
    diffs = []
    n     = len(arr1)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs.append(arr1[idx].mean() - arr2[idx].mean())
    diffs = np.array(diffs)
    lo  = np.percentile(diffs, (100 - ci) / 2)
    hi  = np.percentile(diffs, 100 - (100 - ci) / 2)
    obs = arr1.mean() - arr2.mean()
    return obs, lo, hi


def bootstrap_ci_cindex_diff(arr1, arr2, n_boot: int = 10000, seed: int = 42,
                              ci: int = 95):
    """
    Bootstrap CI for mean(arr1) - mean(arr2) on fold-level C-indices.
    Positive = arr1 has higher C-index than arr2.

    Args:
        arr1, arr2: numpy arrays of fold-level C-index values
        n_boot:     number of bootstrap resamples
        seed:       RNG seed
        ci:         confidence interval percentage (default 95)

    Returns:
        (observed_diff, lower_bound, upper_bound)
    """
    rng   = np.random.RandomState(seed)
    diffs = []
    n     = len(arr1)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs.append(arr1[idx].mean() - arr2[idx].mean())
    diffs = np.array(diffs)
    lo  = np.percentile(diffs, (100 - ci) / 2)
    hi  = np.percentile(diffs, 100 - (100 - ci) / 2)
    obs = arr1.mean() - arr2.mean()
    return obs, lo, hi


def get_fold_declines_single(mcar_dict, mod: str = "rna"):
    """
    Extract per-fold C-index decline from an MCAR results dict.
    Decline = C-index at rate=0.9 minus C-index at rate=0.0 (negative = drop).

    Args:
        mcar_dict: dict returned by run_mcar_crossPoe / run_mcar_single etc.
        mod:       modality key ("rna", "mirna", or "methyl")

    Returns:
        numpy array of shape (n_folds,) with per-fold declines
    """
    ci_r0  = np.array(mcar_dict[mod][0.0])
    ci_r09 = np.array(mcar_dict[mod][0.9])
    return ci_r09 - ci_r0
