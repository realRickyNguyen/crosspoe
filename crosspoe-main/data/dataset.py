import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import beta_to_mvalue, compute_scaler, apply_scaler


class MultiOmicsDataset(Dataset):
    """
    Multi-omics dataset with PFI survival labels.

    Class-level arrays (populated by load_data()) hold the full aligned
    dataset so that train/val splits share memory without duplication.

    Each sample returns a dict:
        mask      : bool [3]    — modality availability (rna, mirna, methyl)
        rna       : float32 tensor | None
        mirna     : float32 tensor | None
        methyl    : float32 tensor | None
        sample_id : str
        pfi_time  : float32 scalar
        pfi_event : int64 scalar  (-1 = missing)
    """

    # Class-level storage — populated by load_data()
    _rna_data    = None
    _mirna_data  = None
    _methyl_data = None
    _rna_mask    = None
    _mirna_mask  = None
    _methyl_mask = None
    _sample_ids  = None
    _pfi_time    = None
    _pfi_event   = None
    _rna_feature_names    = None
    _mirna_feature_names  = None
    _methyl_feature_names = None

    def __init__(
        self,
        indices,
        rna_scaler=None,
        mirna_scaler=None,
        methyl_scaler=None,
        dropout_probs=None,
    ):
        self.indices       = np.array(indices)
        self.rna_scaler    = rna_scaler
        self.mirna_scaler  = mirna_scaler
        self.methyl_scaler = methyl_scaler
        self.dropout_probs = dropout_probs

        self.rna_data    = MultiOmicsDataset._rna_data
        self.mirna_data  = MultiOmicsDataset._mirna_data
        self.methyl_data = MultiOmicsDataset._methyl_data
        self.rna_mask    = MultiOmicsDataset._rna_mask
        self.mirna_mask  = MultiOmicsDataset._mirna_mask
        self.methyl_mask = MultiOmicsDataset._methyl_mask
        self.sample_ids  = MultiOmicsDataset._sample_ids

    def __len__(self):
        return len(self.indices)

    @classmethod
    def _prepare_survival_labels(cls, clinical_df, sample_ids):
        """Load PFI survival labels aligned to sample_ids order."""
        clin      = clinical_df.set_index("sampleID")
        n_samples = len(sample_ids)
        pfi_time  = torch.full((n_samples,), -1.0, dtype=torch.float32)
        pfi_event = torch.full((n_samples,), -1,   dtype=torch.long)

        for i, sid in enumerate(sample_ids):
            if sid not in clin.index:
                continue
            row   = clin.loc[sid]
            pfi_t = row.get("PFI_time", None)
            pfi_e = row.get("PFI",      None)
            if pd.notna(pfi_t) and pd.notna(pfi_e):
                pfi_time[i]  = float(pfi_t)
                pfi_event[i] = int(pfi_e)

        cls._pfi_time  = pfi_time
        cls._pfi_event = pfi_event

        valid  = (pfi_event >= 0).sum().item()
        events = (pfi_event == 1).sum().item()
        print(f"  PFI: {valid}/{n_samples} valid, {events} events")

    def __getitem__(self, idx):
        i = self.indices[idx]

        rna_obs    = bool(self.rna_mask[i])
        mirna_obs  = bool(self.mirna_mask[i])
        methyl_obs = bool(self.methyl_mask[i])

        if self.dropout_probs is not None:
            for _ in range(10):
                rna_drop    = rna_obs    and (np.random.rand() < self.dropout_probs["rna"])
                mirna_drop  = mirna_obs  and (np.random.rand() < self.dropout_probs["mirna"])
                methyl_drop = methyl_obs and (np.random.rand() < self.dropout_probs["methyl"])
                if not (rna_drop and mirna_drop and methyl_drop):
                    break
            else:
                rna_drop = mirna_drop = methyl_drop = False

            rna_obs    = rna_obs    and not rna_drop
            mirna_obs  = mirna_obs  and not mirna_drop
            methyl_obs = methyl_obs and not methyl_drop

        rna_tensor = mirna_tensor = methyl_tensor = None

        if rna_obs:
            x = self.rna_data[i].copy()
            if self.rna_scaler is not None:
                x = apply_scaler(x, *self.rna_scaler)
            rna_tensor = torch.tensor(x, dtype=torch.float32)

        if mirna_obs:
            x = self.mirna_data[i].copy()
            if self.mirna_scaler is not None:
                x = apply_scaler(x, *self.mirna_scaler)
            mirna_tensor = torch.tensor(x, dtype=torch.float32)

        if methyl_obs:
            x = self.methyl_data[i].copy()
            if self.methyl_scaler is not None:
                x = apply_scaler(x, *self.methyl_scaler)
            methyl_tensor = torch.tensor(x, dtype=torch.float32)

        mask = torch.tensor([rna_obs, mirna_obs, methyl_obs], dtype=torch.bool)

        return {
            "rna":       rna_tensor,
            "mirna":     mirna_tensor,
            "methyl":    methyl_tensor,
            "mask":      mask,
            "sample_id": self.sample_ids[i],
            "pfi_time":  MultiOmicsDataset._pfi_time[i],
            "pfi_event": MultiOmicsDataset._pfi_event[i],
        }


def load_data(
    rna_path:    str,
    mirna_path:  str,
    methyl_path: str,
    clin_path:   str,
) -> np.ndarray:
    """
    Load, align, and store multi-omics data into MultiOmicsDataset class arrays.

    Expects CSV files where rows are samples and columns are features. The
    clinical file must have a 'sampleID' column plus 'PFI' and 'PFI_time'
    columns for survival labels. Methylation beta values are converted to
    M-values automatically.

    Samples are aligned to the union of those present in at least one omics
    modality and in the clinical file.

    Args:
        rna_path    : path to mRNA expression CSV (samples x genes)
        mirna_path  : path to miRNA expression CSV (samples x miRNAs)
        methyl_path : path to methylation beta-value CSV (samples x CpG sites)
        clin_path   : path to clinical CSV with sampleID, PFI, PFI_time columns

    Returns:
        master_ids: sorted np.ndarray of sample IDs in the aligned dataset
    """
    print("Loading clinical data ...")
    clin = pd.read_csv(clin_path, low_memory=False)
    clin = clin.drop_duplicates(subset="sampleID").set_index("sampleID")
    print(f"  Clinical samples: {len(clin)}")

    print("\nLoading mRNA ...")
    rna = pd.read_csv(rna_path, index_col=0)
    print(f"  Shape: {rna.shape}")
    MultiOmicsDataset._rna_feature_names = list(rna.columns)

    print("Loading miRNA ...")
    mirna = pd.read_csv(mirna_path, index_col=0)
    print(f"  Shape: {mirna.shape}")
    MultiOmicsDataset._mirna_feature_names = list(mirna.columns)

    print("Loading methylation (large file, may take a moment) ...")
    methyl = pd.read_csv(methyl_path, index_col=0)
    print(f"  Shape: {methyl.shape}")
    MultiOmicsDataset._methyl_feature_names = list(methyl.columns)

    print("\nApplying M-value transform to methylation ...")
    methyl_mvals = beta_to_mvalue(methyl.values.astype(np.float32))
    methyl = pd.DataFrame(methyl_mvals, index=methyl.index, columns=methyl.columns)

    all_clin    = set(clin.index)
    rna_sids    = set(rna.index)    & all_clin
    mirna_sids  = set(mirna.index)  & all_clin
    methyl_sids = set(methyl.index) & all_clin
    union_sids  = rna_sids | mirna_sids | methyl_sids

    print(f"\nClinical samples:     {len(all_clin)}")
    print(f"mRNA observed:        {len(rna_sids)}")
    print(f"miRNA observed:       {len(mirna_sids)}")
    print(f"Methylation observed: {len(methyl_sids)}")
    print(f"Total usable (union): {len(union_sids)}")

    master_ids = sorted(union_sids)
    n          = len(master_ids)
    n_rna      = rna.shape[1]
    n_mirna    = mirna.shape[1]
    n_methyl   = methyl.shape[1]

    rna_arr    = np.full((n, n_rna),    np.nan, dtype=np.float32)
    mirna_arr  = np.full((n, n_mirna),  np.nan, dtype=np.float32)
    methyl_arr = np.full((n, n_methyl), np.nan, dtype=np.float32)
    rna_mask    = np.zeros(n, dtype=bool)
    mirna_mask  = np.zeros(n, dtype=bool)
    methyl_mask = np.zeros(n, dtype=bool)

    print("\nFilling aligned arrays ...")
    for i, sid in enumerate(master_ids):
        if sid in rna_sids:
            rna_arr[i]    = rna.loc[sid].values
            rna_mask[i]   = True
        if sid in mirna_sids:
            mirna_arr[i]  = mirna.loc[sid].values
            mirna_mask[i] = True
        if sid in methyl_sids:
            methyl_arr[i]  = methyl.loc[sid].values
            methyl_mask[i] = True

    MultiOmicsDataset._rna_data    = rna_arr
    MultiOmicsDataset._mirna_data  = mirna_arr
    MultiOmicsDataset._methyl_data = methyl_arr
    MultiOmicsDataset._rna_mask    = rna_mask
    MultiOmicsDataset._mirna_mask  = mirna_mask
    MultiOmicsDataset._methyl_mask = methyl_mask
    MultiOmicsDataset._sample_ids  = np.array(master_ids)

    print("\nMissingness pattern:")
    print(f"  All three:         {(rna_mask & mirna_mask & methyl_mask).sum()}")
    print(f"  mRNA + miRNA only: {(rna_mask & mirna_mask & ~methyl_mask).sum()}")
    print(f"  mRNA + methyl:     {(rna_mask & ~mirna_mask & methyl_mask).sum()}")
    print(f"  mRNA only:         {(rna_mask & ~mirna_mask & ~methyl_mask).sum()}")
    print(f"\nFeature dimensions:")
    print(f"  mRNA:        {n_rna}")
    print(f"  miRNA:       {n_mirna}")
    print(f"  Methylation: {n_methyl}")

    return np.array(master_ids)


def collate_fn(batch: list) -> dict:
    """
    Custom collation that pads missing modality tensors with zeros and
    stacks them into a single batch tensor.
    """
    keys   = ["rna", "mirna", "methyl"]
    mask   = torch.stack([b["mask"]      for b in batch])
    ids    = [b["sample_id"] for b in batch]
    result = {"mask": mask, "sample_id": ids}

    for k in keys:
        tensors = [b[k] for b in batch]
        if all(t is None for t in tensors):
            result[k] = None
        else:
            ref       = next(t for t in tensors if t is not None)
            result[k] = torch.stack([
                t if t is not None else torch.zeros_like(ref) for t in tensors
            ])

    result["pfi_time"]  = torch.stack([b["pfi_time"]  for b in batch])
    result["pfi_event"] = torch.stack([b["pfi_event"] for b in batch])
    return result
