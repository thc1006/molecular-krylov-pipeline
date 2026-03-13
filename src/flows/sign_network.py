"""Sign and phase networks for molecular wavefunction sign/phase structure.

The autoregressive flow defines |ψ(x)|² = p_θ(x), giving amplitudes
√p(x) that are always positive.  However, molecular ground-state
wavefunctions have CI coefficients with both positive and negative signs.

Two approaches are provided:

1. **SignNetwork** (legacy): ψ(x) = √p(x) × s_φ(x), where s ∈ (-1, 1) via tanh.
   Simple but has gradient discontinuity at sign boundaries and cannot represent
   complex-valued wavefunctions.

2. **PhaseNetwork** (P2.1): ψ(x) = √p(x) × e^{iφ(x)}, where φ ∈ [0, 2π) via
   2π·sigmoid.  Continuous phase: gradient is always smooth.  Subsumes sign network
   (φ=0 → +1, φ=π → -1).  Required for complex-valued Hamiltonians and better
   optimization landscape for real-valued cases.

**Training**: Both are trained jointly with the flow in VMC.
The flow gradient uses MinSR/REINFORCE, while sign/phase gradients use
direct backpropagation through E_loc:

    ∇_θ <H> = E_p[(E_loc - b) × ∇_θ log p_θ(x)]    (REINFORCE/MinSR)
    ∇_φ <H> = E_p[∇_φ E_loc(x)]                       (direct backprop)

References:
    - QiankunNet: Li et al., Nature Comms 2025 — continuous phase MLP
    - Hibat-Allah et al., PRR 2 (2020) — separate amplitude and sign networks
    - Choo et al., PRB 100 (2019) — sign network for NQS
"""

import torch
import torch.nn as nn
from typing import Optional


class SignNetwork(nn.Module):
    """Feedforward network that predicts the sign of the wavefunction.

    Architecture: input → [FC → GELU]×N → FC → tanh → scalar sign.

    The GELU activation (smooth, non-monotonic) works better than ReLU for
    sign prediction because the sign structure often has smooth boundaries
    in configuration space.

    Parameters
    ----------
    num_sites : int
        Number of spin-orbitals (= 2 × n_orbitals).  Input dimension.
    hidden_dims : list of int, optional
        Hidden layer dimensions.  If None, auto-scaled based on num_sites.
    """

    def __init__(
        self,
        num_sites: int,
        hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        self.num_sites = num_sites

        if hidden_dims is None:
            hidden_dims = self._auto_hidden_dims(num_sites)

        layers = []
        in_dim = num_sites
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _auto_hidden_dims(num_sites: int) -> list:
        """Auto-scale hidden dimensions based on system size."""
        if num_sites <= 10:
            return [64, 32]
        elif num_sites <= 20:
            return [128, 64]
        elif num_sites <= 40:
            return [256, 128, 64]
        else:
            return [512, 256, 128]

    def forward(self, configs: torch.Tensor) -> torch.Tensor:
        """Predict sign values for a batch of configurations.

        Parameters
        ----------
        configs : torch.Tensor
            (batch, num_sites) binary configuration vectors (float).

        Returns
        -------
        torch.Tensor
            (batch,) sign values in (-1, 1) via tanh.
        """
        raw = self.net(configs)  # (batch, 1)
        return torch.tanh(raw.squeeze(-1))  # (batch,)


class PhaseNetwork(nn.Module):
    """Network that predicts the complex phase of the wavefunction.

    Outputs a continuous phase φ ∈ [0, 2π) so that

        ψ(x) = √p_θ(x) × e^{iφ_ψ(x)}

    This subsumes the SignNetwork: φ=0 → e^{i0}=+1, φ=π → e^{iπ}=-1.
    For real-valued ground states the network learns φ ∈ {0, π}, while for
    complex-valued cases it learns arbitrary phases.

    Advantages over SignNetwork:
    - Smooth gradient everywhere (sigmoid has no flat regions like tanh at ±∞)
    - Can represent complex-valued wavefunctions (needed for e.g. magnetic systems)
    - Phase ratio e^{i(φ(y)-φ(x))} is always well-defined (no division by zero)

    Parameters
    ----------
    num_sites : int
        Number of spin-orbitals.  Input dimension.
    hidden_dims : list of int, optional
        Hidden layer dimensions.  If None, auto-scaled based on num_sites.

    References
    ----------
    QiankunNet: Li et al., Nature Comms 2025 — continuous phase MLP.
    """

    def __init__(
        self,
        num_sites: int,
        hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        self.num_sites = num_sites

        if hidden_dims is None:
            hidden_dims = SignNetwork._auto_hidden_dims(num_sites)

        layers = []
        in_dim = num_sites
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, configs: torch.Tensor) -> torch.Tensor:
        """Predict phase values for a batch of configurations.

        Parameters
        ----------
        configs : torch.Tensor
            (batch, num_sites) binary configuration vectors (float).

        Returns
        -------
        torch.Tensor
            (batch,) phase values in [0, 2π).
        """
        raw = self.net(configs)  # (batch, 1)
        return 2.0 * torch.pi * torch.sigmoid(raw.squeeze(-1))  # (batch,)

    def phase_factor(self, configs: torch.Tensor) -> torch.Tensor:
        """Compute e^{iφ(x)} for a batch of configurations.

        Parameters
        ----------
        configs : torch.Tensor
            (batch, num_sites) binary configuration vectors (float).

        Returns
        -------
        torch.Tensor
            (batch,) complex unit values e^{iφ} on the unit circle.
        """
        phi = self.forward(configs)  # (batch,)
        return torch.exp(1j * phi.to(torch.float64))  # (batch,) complex128
