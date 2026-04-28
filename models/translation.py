import torch
import torch.nn as nn


class LatentTranslationHead(nn.Module):
    """
    Translates a source posterior (mu_src, logvar_src) into a pseudo-posterior
    (mu_pseudo, logvar_pseudo) for the target modality.

    Initialised to the identity (zero weight, zero bias on the output layer)
    so the model starts from no translation and learns incrementally.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mu_src: torch.Tensor, logvar_src: torch.Tensor):
        h             = torch.cat([mu_src, logvar_src], dim=-1)
        out           = self.net(h)
        mu_pseudo     = out[:, :self.latent_dim]
        logvar_pseudo = torch.clamp(out[:, self.latent_dim:], min=-6.0, max=2.0)
        return mu_pseudo, logvar_pseudo


class TranslationGateNetwork(nn.Module):
    """
    Per-sample confidence gate in (0, 1).

    Predicts how much to trust a translated pseudo-posterior based solely on
    the source posterior. Initialised to output ~0.27 (sigmoid(-1)) so
    translations start with low weight.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -1.0)

    def forward(self, mu_src: torch.Tensor, logvar_src: torch.Tensor) -> torch.Tensor:
        h = torch.cat([mu_src, logvar_src], dim=-1)
        return torch.sigmoid(self.net(h))


class CrossModalTranslator(nn.Module):
    """
    Manages all 6 pairwise translation directions among (rna, mirna, methyl)
    with per-sample gated trust scores.

    Only applies translations where the source modality is observed and the
    target modality is missing.
    """

    MODALITY_NAMES = ["rna", "mirna", "methyl"]

    def __init__(self, latent_dim: int, hidden_dim: int = 96, gate_hidden_dim: int = 32):
        super().__init__()
        self.latent_dim   = latent_dim
        self.n_modalities = 3
        self.translation_heads = nn.ModuleDict()
        self.gate_networks     = nn.ModuleDict()

        for src in range(self.n_modalities):
            for tgt in range(self.n_modalities):
                if src == tgt:
                    continue
                key = f"{src}_to_{tgt}"
                self.translation_heads[key] = LatentTranslationHead(latent_dim, hidden_dim)
                self.gate_networks[key]     = TranslationGateNetwork(latent_dim, gate_hidden_dim)

    def forward(self, mus, logvars, mask):
        """
        Compute pseudo-posteriors for each sample's missing modalities.

        Args:
            mus     : list of [batch, latent_dim] tensors, one per modality
            logvars : list of [batch, latent_dim] tensors, one per modality
            mask    : [batch, 3] bool — which modalities are observed

        Returns:
            translated_mus    : list of [batch, latent_dim] tensors
            translated_logvars: list of [batch, latent_dim] tensors
            gates             : list of [batch, 1] gate tensors
        """
        batch_size         = mus[0].shape[0]
        translated_mus     = []
        translated_logvars = []
        gates              = []

        for src in range(self.n_modalities):
            for tgt in range(self.n_modalities):
                if src == tgt:
                    continue
                key               = f"{src}_to_{tgt}"
                needs_translation = mask[:, src] & (~mask[:, tgt])
                if not needs_translation.any():
                    continue

                mu_sub, lv_sub = self.translation_heads[key](
                    mus[src][needs_translation], logvars[src][needs_translation]
                )
                gate_sub = self.gate_networks[key](
                    mus[src][needs_translation], logvars[src][needs_translation]
                )

                mu_full   = torch.zeros(batch_size, self.latent_dim, device=mus[src].device)
                lv_full   = torch.zeros(batch_size, self.latent_dim, device=mus[src].device)
                gate_full = torch.zeros(batch_size, 1,               device=mus[src].device)
                mu_full[needs_translation]   = mu_sub
                lv_full[needs_translation]   = lv_sub
                gate_full[needs_translation] = gate_sub

                translated_mus.append(mu_full)
                translated_logvars.append(lv_full)
                gates.append(gate_full)

        return translated_mus, translated_logvars, gates
