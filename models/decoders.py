import torch
import torch.nn as nn

from .blocks import fc_block

# Default feature dimensions — adjust for other datasets
_N_RNA    = 4652
_N_MIRNA  = 524
_N_METHYL = 37482
_N_LATENT = 48


class RNADecoder(nn.Module):
    """mRNA decoder (mirror of RNAEncoder): latent_dim -> 96 -> 256 -> 512 -> out_dim."""

    def __init__(self, latent_dim: int = _N_LATENT, out_dim: int = _N_RNA):
        super().__init__()
        self.net = nn.Sequential(
            fc_block(latent_dim, 96,  dropout=0.2),
            fc_block(96,         256, dropout=0.2),
            fc_block(256,        512, dropout=0.2),
            nn.Linear(512, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MIRNADecoder(nn.Module):
    """miRNA decoder (mirror of MIRNAEncoder): latent_dim -> 96 -> 128 -> out_dim."""

    def __init__(self, latent_dim: int = _N_LATENT, out_dim: int = _N_MIRNA):
        super().__init__()
        self.net = nn.Sequential(
            fc_block(latent_dim, 96,  dropout=0.2),
            fc_block(96,         128, dropout=0.2),
            nn.Linear(128, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MethylDecoder(nn.Module):
    """Methylation decoder (mirror of MethylEncoder): latent_dim -> 96 -> 256 -> 512 -> out_dim."""

    def __init__(self, latent_dim: int = _N_LATENT, out_dim: int = _N_METHYL):
        super().__init__()
        self.net = nn.Sequential(
            fc_block(latent_dim, 96,  dropout=0.2),
            fc_block(96,         256, dropout=0.2),
            fc_block(256,        512, dropout=0.2),
            nn.Linear(512, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
