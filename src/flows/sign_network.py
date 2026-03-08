"""Sign network for molecular wavefunction sign structure.

The autoregressive flow defines |ψ(x)|² = p_θ(x), giving amplitudes
√p(x) that are always positive.  However, molecular ground-state
wavefunctions have CI coefficients with both positive and negative signs.
The sign network learns this sign structure:

    ψ(x) = √p_θ(x) × s_φ(x)

where s_φ(x) = tanh(f_φ(x)) ∈ (-1, 1) is a continuous relaxation of
the discrete sign ∈ {-1, +1}.

**Training**: The sign network is trained jointly with the flow in VMC.
The flow gradient uses REINFORCE (since changing p changes the sampling
distribution), while the sign gradient uses direct backpropagation
through E_loc (since the sampling distribution p doesn't depend on φ):

    ∇_θ <H> = E_p[(E_loc - b) × ∇_θ log p_θ(x)]    (REINFORCE)
    ∇_φ <H> = E_p[∇_φ E_loc(x)]                       (direct backprop)

where E_loc(x) = H_xx + Σ_{y~conn(x)} H_xy × exp(0.5(log p(y) - log p(x))) × s(y)/s(x).

References:
    - Hibat-Allah et al., "Recurrent Neural Network Wave Functions",
      PRR 2 (2020) — separate amplitude and sign networks
    - Choo et al., "Two-dimensional frustrated J1–J2 model studied with
      neural network quantum states", PRB 100 (2019) — sign network for NQS
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
