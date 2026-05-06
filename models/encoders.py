import torch.nn as nn

from .blocks import fc_block

# Default feature dimensions — adjust for other datasets
_N_RNA    = 4652
_N_MIRNA  = 524
_N_METHYL = 37482
_N_LATENT = 48


class RNAEncoder(nn.Module):
    """mRNA encoder: in_dim -> 512 -> 256 -> 96 -> (mu, logvar) of size latent_dim."""

    def __init__(self, in_dim: int = _N_RNA, latent_dim: int = _N_LATENT):
        super().__init__()
        self.net = nn.Sequential(
            fc_block(in_dim, 512, dropout=0.3),
            fc_block(512,    256, dropout=0.2),
            fc_block(256,     96, dropout=0.2),
        )
        self.fc_mu     = nn.Linear(96, latent_dim)
        self.fc_logvar = nn.Linear(96, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class MIRNAEncoder(nn.Module):
    """miRNA encoder: in_dim -> 128 -> 96 -> (mu, logvar) of size latent_dim."""

    def __init__(self, in_dim: int = _N_MIRNA, latent_dim: int = _N_LATENT):
        super().__init__()
        self.net = nn.Sequential(
            fc_block(in_dim, 128, dropout=0.2),
            fc_block(128,     96, dropout=0.2),
        )
        self.fc_mu     = nn.Linear(96, latent_dim)
        self.fc_logvar = nn.Linear(96, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class MethylEncoder(nn.Module):
    """
    Methylation encoder: in_dim -> 512 -> 256 -> 96 -> (mu, logvar) of size latent_dim.

    Uses a separate first layer with higher dropout (0.4) to handle the high
    dimensionality of the methylation input (37k+ CpG sites).
    """

    def __init__(self, in_dim: int = _N_METHYL, latent_dim: int = _N_LATENT):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, 512)
        self.input_norm  = nn.LayerNorm(512)
        self.input_act   = nn.GELU()
        self.input_drop  = nn.Dropout(0.4)
        self.net = nn.Sequential(
            fc_block(512, 256, dropout=0.2),
            fc_block(256,  96, dropout=0.2),
        )
        self.fc_mu     = nn.Linear(96, latent_dim)
        self.fc_logvar = nn.Linear(96, latent_dim)

    def forward(self, x):
        h = self.input_layer(x)
        h = self.input_drop(self.input_act(self.input_norm(h)))
        h = self.net(h)
        return self.fc_mu(h), self.fc_logvar(h)
