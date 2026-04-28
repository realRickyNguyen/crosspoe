import numpy as np
import torch
from torch.utils.data import Dataset


class MCARDataset(Dataset):
    """
    Wraps a MultiOmicsDataset and forces a single modality to be missing
    at a given rate on top of natural missingness.

    Natural missingness is preserved — forced missingness is applied
    additionally using a fixed RNG seed for reproducibility.

    Args:
        base_dataset            : MultiOmicsDataset instance
        forced_missing_modality : one of "rna", "mirna", "methyl"
        forced_missing_rate     : fraction of samples to force-drop [0, 1]
        seed                    : RNG seed (default 42)
    """

    _MOD_IDX = {"rna": 0, "mirna": 1, "methyl": 2}

    def __init__(self, base_dataset, forced_missing_modality, forced_missing_rate, seed=42):
        if forced_missing_modality not in self._MOD_IDX:
            raise ValueError(
                f"modality must be one of {list(self._MOD_IDX)}, "
                f"got {forced_missing_modality!r}"
            )
        self.base           = base_dataset
        self.modality       = forced_missing_modality
        self.rate           = forced_missing_rate
        rng                 = np.random.RandomState(seed)
        self.forced_missing = rng.rand(len(base_dataset)) < forced_missing_rate

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if self.forced_missing[idx]:
            item[self.modality] = None
            mod_idx         = self._MOD_IDX[self.modality]
            mask            = item["mask"].clone()
            mask[mod_idx]   = False
            item["mask"]    = mask
        return item
