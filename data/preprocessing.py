import numpy as np


def beta_to_mvalue(X: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Convert methylation beta values to M-values via logit transform."""
    X = np.clip(X, epsilon, 1 - epsilon)
    return np.log2(X / (1 - X))


def compute_scaler(X: np.ndarray):
    """Compute (mean, std) from observed data for z-score normalisation. Std is floored at 1."""
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply z-score normalisation."""
    return (X - mean) / std
