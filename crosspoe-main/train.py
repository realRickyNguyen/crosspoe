"""
train.py — Entry point for CrossPoE cross-validation training.

Usage:
    python train.py \
        --rna-path    /path/to/rna.csv \
        --mirna-path  /path/to/mirna.csv \
        --methyl-path /path/to/methylation.csv \
        --clin-path   /path/to/clinical.csv
"""

import argparse
import warnings

import pandas as pd
import torch

from config import CONFIG
from data.dataset import MultiOmicsDataset, load_data
from training.trainer import run_cross_validation

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="CrossPoE: multi-omics PoE VAE for survival prediction"
    )
    parser.add_argument("--rna-path",    required=True, help="Path to mRNA expression CSV (samples x genes)")
    parser.add_argument("--mirna-path",  required=True, help="Path to miRNA expression CSV (samples x miRNAs)")
    parser.add_argument("--methyl-path", required=True, help="Path to methylation beta-value CSV (samples x CpGs)")
    parser.add_argument("--clin-path",   required=True, help="Path to clinical CSV with sampleID, PFI, PFI_time columns")
    parser.add_argument("--seed",        type=int, default=CONFIG["seed"], help="Global random seed")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg         = dict(CONFIG)
    cfg["seed"] = args.seed

    print("\n=== Loading Data ===")
    load_data(
        rna_path=args.rna_path,
        mirna_path=args.mirna_path,
        methyl_path=args.methyl_path,
        clin_path=args.clin_path,
    )
    clinical_df = pd.read_csv(args.clin_path, low_memory=False)
    MultiOmicsDataset._prepare_survival_labels(clinical_df, MultiOmicsDataset._sample_ids)

    print("\n=== Starting Cross-Validation ===")
    fold_results = run_cross_validation(cfg, device)
    return fold_results


if __name__ == "__main__":
    main()
