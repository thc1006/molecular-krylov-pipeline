"""Transformer Backflow for molecular NQS wavefunctions.

Implements the Neural Network Backflow (NNBF) / Transformer Backflow
architecture where the wavefunction is represented as a Slater determinant
with configuration-dependent orbital coefficients:

    ψ(x) = det(U_θ(x))

where U_θ(x) is an (n_electrons × n_electrons) matrix whose columns are
orbital coefficients produced by a Transformer conditioned on the occupation
vector x.  The sign of the wavefunction comes naturally from det(),
eliminating the need for a separate SignNetwork or PhaseNetwork.

Key advantage over ψ = √p × s(x):
- Correct sign structure: det() automatically respects antisymmetry
- Universal approximator: any fermionic wavefunction can be written as
  a sum of Slater determinants; single-determinant backflow has been
  shown to approximate multi-determinant wavefunctions
- Smooth gradients: det() is analytic, no sign discontinuities

Architecture:
    occupation x (n_sites,) → Embedding → Transformer → MLP head
        → orbital coefficients U (n_electrons, n_electrons)
        → log|ψ| = 0.5 * log|det(U)|²  [for sampling / log_prob]
        → sign(ψ) = sign(det(U))         [for E_loc computation]

The Transformer processes the occupied orbital indices, producing a
representation that captures inter-electron correlations.  The MLP
head maps each electron's representation to orbital coefficients.

References:
    - Luo & Clark, PRB 2024 (arXiv:2403.03286): NNBF for molecules
    - Ma et al., arXiv:2509.25720: Transformer Backflow for [2Fe-2S]
    - Kim et al., JCTC 2024: Transformer-based NQS for fermions
    - Pfau et al., PRR 2020: FermiNet (continuous backflow, different)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BackflowConfig:
    """Configuration for Transformer Backflow.

    Parameters
    ----------
    d_model : int
        Transformer embedding dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    d_ff : int
        Feedforward dimension (default: 4 * d_model).
    dropout : float
        Dropout rate.
    n_determinants : int
        Number of Slater determinants (multi-determinant expansion).
        1 = single determinant (default, simplest).
        >1 = sum of determinants with learnable weights.
    """

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 0  # 0 = auto (4 * d_model)
    dropout: float = 0.0
    n_determinants: int = 1


class BackflowTransformer(nn.Module):
    """Transformer Backflow wavefunction ansatz.

    Given an occupation vector x ∈ {0,1}^{n_sites}, produces:
        - log_amplitude: 0.5 * log |det(U(x))|²
        - sign: sign(det(U(x)))

    The orbital coefficient matrix U(x) is constructed by:
    1. Extract occupied orbital indices from x
    2. Embed each occupied orbital with a learnable embedding
    3. Process through Transformer encoder (inter-electron correlations)
    4. MLP head maps each electron's output to n_electrons coefficients
    5. Assemble into U matrix and compute determinant

    Parameters
    ----------
    num_sites : int
        Number of spin-orbitals (= 2 * n_orbitals).
    n_alpha : int
        Number of alpha electrons.
    n_beta : int
        Number of beta electrons.
    config : BackflowConfig, optional
        Architecture configuration.
    """

    def __init__(
        self,
        num_sites: int,
        n_alpha: int,
        n_beta: int,
        config: Optional[BackflowConfig] = None,
    ):
        super().__init__()
        self.num_sites = num_sites
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.n_electrons = n_alpha + n_beta
        self.config = config or BackflowConfig()
        cfg = self.config

        d_model = cfg.d_model
        d_ff = cfg.d_ff if cfg.d_ff > 0 else 4 * d_model
        n_det = cfg.n_determinants

        # Orbital embedding: each spin-orbital gets a learnable vector
        # Plus a "spin type" embedding (alpha vs beta)
        self.orbital_embedding = nn.Embedding(num_sites, d_model)
        self.spin_type_embedding = nn.Embedding(2, d_model)  # 0=alpha, 1=beta

        # Positional encoding for electron ordering
        self.position_embedding = nn.Embedding(self.n_electrons, d_model)

        # Transformer encoder for inter-electron correlations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.n_heads,
            dim_feedforward=d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,
        )

        # MLP head: each electron's transformer output → orbital coefficients
        # Output: n_electrons coefficients per electron per determinant
        self.orbital_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, self.n_electrons * n_det),
        )

        # Multi-determinant weights (learnable, log-space for stability)
        if n_det > 1:
            self.det_log_weights = nn.Parameter(torch.zeros(n_det))
        else:
            self.det_log_weights = None

        # Layer norm for input
        self.input_norm = nn.LayerNorm(d_model)

    def _extract_occupied_indices(
        self, configs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract occupied orbital indices from occupation vectors.

        Parameters
        ----------
        configs : torch.Tensor
            (batch, n_sites) binary occupation vectors.

        Returns
        -------
        occupied_indices : torch.Tensor
            (batch, n_electrons) indices of occupied orbitals.
        spin_types : torch.Tensor
            (batch, n_electrons) spin type: 0=alpha (even index), 1=beta (odd).
        """
        batch_size = configs.shape[0]
        device = configs.device

        # For each config, find the indices where configs[b, i] == 1
        # We know there are exactly n_electrons occupied orbitals
        occupied = torch.zeros(
            batch_size, self.n_electrons, dtype=torch.long, device=device
        )
        spin_types = torch.zeros(
            batch_size, self.n_electrons, dtype=torch.long, device=device
        )

        for b in range(batch_size):
            occ_idx = configs[b].nonzero(as_tuple=True)[0]
            n_occ = min(len(occ_idx), self.n_electrons)
            occupied[b, :n_occ] = occ_idx[:n_occ]
            spin_types[b, :n_occ] = occ_idx[:n_occ] % 2  # even=alpha, odd=beta

        return occupied, spin_types

    def _extract_occupied_indices_batched(
        self, configs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized extraction of occupied orbital indices.

        Faster than the loop version for large batches.
        Assumes each config has exactly n_electrons occupied orbitals.
        """
        batch_size = configs.shape[0]
        device = configs.device

        # Get all occupied positions: shape (total_occupied,)
        # configs is (batch, n_sites) with exactly n_electrons 1s per row
        # Use topk to get the positions of 1s (they are either 0 or 1)
        # topk on binary tensor returns indices of 1s first
        _, occupied = configs.topk(self.n_electrons, dim=1, sorted=True)
        # Sort within each row for consistent ordering
        occupied, _ = occupied.sort(dim=1)

        spin_types = occupied % 2  # even=alpha(0), odd=beta(1)

        return occupied, spin_types

    def _build_orbital_matrix(
        self, configs: torch.Tensor
    ) -> torch.Tensor:
        """Build the orbital coefficient matrix U(x) for each config.

        Parameters
        ----------
        configs : torch.Tensor
            (batch, n_sites) binary occupation vectors.

        Returns
        -------
        U : torch.Tensor
            (batch, [n_det,] n_electrons, n_electrons) orbital matrices.
        """
        batch_size = configs.shape[0]
        device = configs.device
        n_det = self.config.n_determinants

        # Extract occupied orbital indices
        occupied, spin_types = self._extract_occupied_indices_batched(configs)

        # Embed: orbital embedding + spin type embedding + position embedding
        orb_emb = self.orbital_embedding(occupied)  # (batch, n_elec, d_model)
        spin_emb = self.spin_type_embedding(spin_types)  # (batch, n_elec, d_model)
        pos_idx = torch.arange(self.n_electrons, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_idx)  # (batch, n_elec, d_model)

        x = orb_emb + spin_emb + pos_emb  # (batch, n_elec, d_model)
        x = self.input_norm(x)

        # Transformer encoder
        x = self.transformer(x)  # (batch, n_elec, d_model)

        # MLP head: produce orbital coefficients
        coeffs = self.orbital_head(x)  # (batch, n_elec, n_elec * n_det)

        if n_det > 1:
            # Reshape to (batch, n_det, n_elec, n_elec)
            coeffs = coeffs.view(batch_size, self.n_electrons, n_det, self.n_electrons)
            coeffs = coeffs.permute(0, 2, 1, 3)  # (batch, n_det, n_elec, n_elec)
        else:
            # (batch, n_elec, n_elec) — single determinant
            coeffs = coeffs.view(batch_size, self.n_electrons, self.n_electrons)

        return coeffs

    def forward(
        self, configs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log|ψ(x)| and sign(ψ(x)) for a batch of configurations.

        Parameters
        ----------
        configs : torch.Tensor
            (batch, n_sites) binary occupation vectors.

        Returns
        -------
        log_amp : torch.Tensor
            (batch,) log |ψ(x)| = 0.5 * log |det(U(x))|² for single det,
            or log |Σ_k w_k det(U_k(x))| for multi-det.
        sign : torch.Tensor
            (batch,) sign of ψ(x), ∈ {-1, +1}.
        """
        U = self._build_orbital_matrix(configs)  # (batch, [n_det,] n_elec, n_elec)

        if self.config.n_determinants == 1:
            # Single determinant: ψ = det(U)
            # Use slogdet for numerical stability
            sign, logabsdet = torch.linalg.slogdet(U)
            log_amp = logabsdet  # log|det(U)|
            return log_amp, sign
        else:
            # Multi-determinant: ψ = Σ_k w_k det(U_k)
            # U shape: (batch, n_det, n_elec, n_elec)
            sign_k, logabsdet_k = torch.linalg.slogdet(U)  # (batch, n_det)

            # Weighted sum in log-space: log|Σ w_k s_k exp(logdet_k)|
            log_weights = torch.log_softmax(self.det_log_weights, dim=0)  # (n_det,)
            # signed_logdet_k = log(w_k) + logabsdet_k, with sign
            weighted = sign_k * torch.exp(log_weights.unsqueeze(0) + logabsdet_k)
            psi = weighted.sum(dim=1)  # (batch,)

            sign = torch.sign(psi)
            log_amp = torch.log(torch.abs(psi) + 1e-30)
            return log_amp, sign

    def log_amplitude(self, configs: torch.Tensor) -> torch.Tensor:
        """Compute log|ψ(x)| for compatibility with NQS interface.

        Parameters
        ----------
        configs : torch.Tensor
            (batch, n_sites) configurations (float).

        Returns
        -------
        log_amp : torch.Tensor
            (batch,) log amplitudes.
        """
        log_amp, _ = self.forward(configs.long())
        return log_amp

    def forward_sign(self, configs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only sign — compatible with SignNetwork interface.

        VMCTrainer calls sign_network(configs) which invokes forward().
        This method is the actual forward() when used as a sign_network drop-in.

        Parameters
        ----------
        configs : torch.Tensor
            (batch, n_sites) configurations (float or long).

        Returns
        -------
        sign : torch.Tensor
            (batch,) signs ∈ {-1, +1} with gradient.
        """
        _, sign = self.forward(configs.long())
        return sign

    def sign_factor(self, configs: torch.Tensor) -> torch.Tensor:
        """Compute sign(ψ(x)) for E_loc computation.

        Parameters
        ----------
        configs : torch.Tensor
            (batch, n_sites) configurations.

        Returns
        -------
        sign : torch.Tensor
            (batch,) signs ∈ {-1, +1}.
        """
        _, sign = self.forward(configs.long())
        return sign

    def phase_factor(self, configs: torch.Tensor) -> torch.Tensor:
        """Complex phase factor e^{iφ} compatible with PhaseNetwork interface.

        For real wavefunctions: sign=+1 → phase=0, sign=-1 → phase=π.

        Parameters
        ----------
        configs : torch.Tensor
            (batch, n_sites) configurations.

        Returns
        -------
        phase : torch.Tensor
            (batch,) complex128 phase factors.
        """
        _, sign = self.forward(configs.long())
        # sign ∈ {-1, +1} → phase ∈ {π, 0}
        phase_angle = (1.0 - sign) * (np.pi / 2.0)  # +1→0, -1→π
        return torch.exp(1j * phase_angle.double())


class BackflowSignAdapter(nn.Module):
    """Adapter that wraps BackflowTransformer to act as a SignNetwork.

    VMCTrainer calls ``sign_network(configs)`` → expects (batch,) sign values.
    This wrapper delegates to BackflowTransformer and returns only the sign.

    Usage::

        backflow = BackflowTransformer(num_sites=20, n_alpha=5, n_beta=5)
        adapter = BackflowSignAdapter(backflow)
        trainer = VMCTrainer(flow, hamiltonian, sign_network=adapter)

    Parameters
    ----------
    backflow : BackflowTransformer
        The underlying backflow model.
    """

    def __init__(self, backflow: BackflowTransformer):
        super().__init__()
        self.backflow = backflow

    def forward(self, configs: torch.Tensor) -> torch.Tensor:
        """Return sign(det(U(x))) — compatible with SignNetwork.forward().

        Parameters
        ----------
        configs : torch.Tensor
            (batch, n_sites) configurations (float).

        Returns
        -------
        torch.Tensor
            (batch,) sign values ∈ {-1, +1}.
        """
        _, sign = self.backflow(configs.long())
        return sign

    def phase_factor(self, configs: torch.Tensor) -> torch.Tensor:
        """Complex phase factor — compatible with PhaseNetwork interface."""
        return self.backflow.phase_factor(configs)
