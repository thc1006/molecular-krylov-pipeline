"""Base class for Neural Quantum States."""

from abc import ABC, abstractmethod
from typing import Union

import torch
import torch.nn as nn
import numpy as np


class NeuralQuantumState(nn.Module, ABC):
    """
    Abstract base class for Neural Quantum States (NQS).

    A Neural Quantum State represents a quantum wavefunction using a neural network:
        ψ_θ(x) = |ψ_θ(x)| * exp(i * φ_θ(x))

    where |ψ_θ(x)| is the amplitude and φ_θ(x) is the phase.

    For real-valued wavefunctions, the phase is restricted to {0, π}.

    Attributes:
        num_sites: Number of sites/qubits in the system
        local_dim: Local Hilbert space dimension (2 for qubits)
        complex_output: Whether to output complex amplitudes
    """

    def __init__(
        self,
        num_sites: int,
        local_dim: int = 2,
        complex_output: bool = False,
    ):
        super().__init__()
        self.num_sites = num_sites
        self.local_dim = local_dim
        self.complex_output = complex_output

    @abstractmethod
    def log_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log|ψ_θ(x)| for batch of configurations.

        Args:
            x: Batch of configurations, shape (batch_size, num_sites)
               Each site is encoded as integer in [0, local_dim-1]

        Returns:
            Log amplitudes, shape (batch_size,)
        """
        pass

    @abstractmethod
    def phase(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute phase φ_θ(x) for batch of configurations.

        Args:
            x: Batch of configurations, shape (batch_size, num_sites)

        Returns:
            Phases in radians, shape (batch_size,)
        """
        pass

    def log_psi(self, x: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute log(ψ_θ(x)) for batch of configurations.

        For real NQS: returns log|ψ| and sign
        For complex NQS: returns (log|ψ|, phase)

        Args:
            x: Batch of configurations, shape (batch_size, num_sites)

        Returns:
            If complex_output: (log_amplitude, phase)
            Otherwise: (log_amplitude, sign) where sign ∈ {-1, +1}
        """
        log_amp = self.log_amplitude(x)
        phi = self.phase(x)

        if self.complex_output:
            return log_amp, phi
        else:
            # For real NQS, phase should be 0 or π
            sign = torch.cos(phi)
            return log_amp, sign

    def psi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ψ_θ(x) for batch of configurations.

        Args:
            x: Batch of configurations, shape (batch_size, num_sites)

        Returns:
            Complex wavefunction values, shape (batch_size,)
        """
        log_amp = self.log_amplitude(x)
        phi = self.phase(x)

        # ψ = exp(log|ψ| + iφ)
        if self.complex_output:
            return torch.exp(log_amp) * torch.exp(1j * phi)
        else:
            return torch.exp(log_amp) * torch.cos(phi)

    def probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute |ψ_θ(x)|² for batch of configurations.

        Args:
            x: Batch of configurations, shape (batch_size, num_sites)

        Returns:
            Probabilities (unnormalized), shape (batch_size,)
        """
        log_amp = self.log_amplitude(x)
        return torch.exp(2 * log_amp)

    def normalized_probability(
        self, x: torch.Tensor, basis_set: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normalized |ψ_θ(x)|² / Z over a basis set.

        Args:
            x: Configurations to evaluate, shape (batch_size, num_sites)
            basis_set: Full set of basis states for normalization

        Returns:
            Normalized probabilities, shape (batch_size,)
        """
        log_amp_x = self.log_amplitude(x)
        log_amp_basis = self.log_amplitude(basis_set)

        # Compute log(Z) = log(sum(exp(2*log_amp)))
        log_Z = torch.logsumexp(2 * log_amp_basis, dim=0)

        # p(x) = exp(2*log_amp - log_Z)
        return torch.exp(2 * log_amp_x - log_Z)

    def to_numpy(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert numpy array to tensor on the correct device."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(next(self.parameters()).device)

    def encode_configuration(self, config: Union[list, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Encode a configuration into the format expected by the network.

        For spin-1/2 systems: {0, 1} → {-1, +1} or {0, 1}

        Args:
            config: Configuration as list/array of site values

        Returns:
            Encoded tensor on the correct device
        """
        config = self.to_numpy(config)
        if config.dim() == 1:
            config = config.unsqueeze(0)
        return config.float()
