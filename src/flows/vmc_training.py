"""Variational Monte Carlo (VMC) trainer for autoregressive normalizing flows.

This module implements VMC training that directly minimizes the variational energy
<psi|H|psi> using the REINFORCE gradient estimator.  Unlike the cross-entropy
teacher loss in ``PhysicsGuidedFlowTrainer``, VMC targets the ground-state energy
directly without requiring an auxiliary NQS network.

**Positive-real wavefunction ansatz**:
    psi(x) = sqrt(p_theta(x))

where p_theta is the probability distribution defined by the autoregressive flow.
This restricts the ansatz to wavefunctions with non-negative amplitudes (sign
structure is encoded in the flow probabilities).

**Local energy**:
    E_loc(x) = sum_y H_{xy} * psi(y) / psi(x)
             = H_{xx} + sum_{y connected to x} H_{xy} * sqrt(p(y) / p(x))
             = H_{xx} + sum_{y connected} H_{xy} * exp(0.5 * (log p(y) - log p(x)))

**REINFORCE gradient** (with baseline subtraction for variance reduction):
    nabla <H> = 2 * E_{x~p}[(E_loc(x) - baseline) * nabla log p_theta(x)]

The factor of 2 arises because psi = sqrt(p), so nabla log |psi|^2 = nabla log p.
In practice, autograd handles the derivative of log p_theta(x) with respect to
theta, and the factor of 2 cancels with the 1/2 in the variational energy
derivative.

References:
    - Carleo & Troyer, "Solving the Quantum Many-Body Problem with Artificial
      Neural Networks", Science 355 (2017)
    - QiankunNet: Li et al., "QiankunNet: Transformer-Based Autoregressive Neural
      Quantum States for Electronic Structure" (Nature Comms, 2025)
    - Barrett et al., "Autoregressive neural-network wavefunctions for ab initio
      quantum chemistry" (Nature Machine Intelligence, 2022)
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

try:
    from .autoregressive_flow import states_to_configs
except ImportError:
    from flows.autoregressive_flow import states_to_configs

try:
    from ..hamiltonians.base import Hamiltonian
except ImportError:
    try:
        from hamiltonians.base import Hamiltonian
    except ImportError:
        Hamiltonian = None  # type: ignore


@dataclass
class VMCConfig:
    """Configuration for VMC training.

    Parameters
    ----------
    n_samples : int
        Number of configurations sampled per optimization step.  More samples
        reduce variance but increase cost per step linearly.
    n_steps : int
        Maximum number of optimization steps.
    lr : float
        Initial learning rate for Adam optimizer.
    lr_decay : float
        Per-step multiplicative learning rate decay (ExponentialLR gamma).
    clip_grad : float
        Maximum gradient norm for gradient clipping.
    baseline_decay : float
        Exponential moving average decay for the energy baseline.  Values close
        to 1.0 (e.g. 0.99) give a stable baseline; smaller values track
        the current energy more tightly.
    min_steps : int
        Minimum number of steps before early stopping is allowed.
    convergence_window : int
        Number of recent steps to consider for convergence check.
    convergence_threshold : float
        Energy range within the convergence window (in Hartree) below which
        training is considered converged.
    log_prob_batch_size : int
        Batch size for computing log_prob of connected configurations.
        Avoids OOM when there are many off-diagonal connections.  Set to 0
        to process all at once.
    """

    n_samples: int = 2000
    n_steps: int = 1000
    lr: float = 1e-3
    lr_decay: float = 0.999
    clip_grad: float = 1.0
    baseline_decay: float = 0.99
    min_steps: int = 200
    convergence_window: int = 50
    convergence_threshold: float = 1e-4
    log_prob_batch_size: int = 5000


class VMCTrainer:
    """Variational Monte Carlo trainer for autoregressive flow.

    Directly minimizes the variational energy <psi|H|psi> using REINFORCE.
    The flow defines psi(x) = sqrt(p_theta(x)) (positive real wavefunction).

    Parameters
    ----------
    flow : AutoregressiveFlowSampler
        Autoregressive flow whose parameters define the wavefunction.
    hamiltonian : Hamiltonian
        System Hamiltonian (must implement ``diagonal_elements_batch`` and
        ``get_connections``).
    config : VMCConfig, optional
        Training hyperparameters.  Defaults to ``VMCConfig()``.
    device : str
        Device for computation ('cpu' or 'cuda').

    Examples
    --------
    >>> from flows.autoregressive_flow import AutoregressiveFlowSampler
    >>> from hamiltonians.molecular import create_h2_hamiltonian
    >>> H = create_h2_hamiltonian()
    >>> flow = AutoregressiveFlowSampler(num_sites=4, n_alpha=1, n_beta=1)
    >>> trainer = VMCTrainer(flow, H, device="cpu")
    >>> results = trainer.train(verbose=True)
    >>> print(f"Best energy: {results['best_energy']:.6f} Ha")
    """

    def __init__(
        self,
        flow: nn.Module,
        hamiltonian: Any,
        config: VMCConfig | None = None,
        device: str = "cpu",
    ):
        self.flow = flow
        self.hamiltonian = hamiltonian
        self.config = config or VMCConfig()
        self.device = device

        self.optimizer = torch.optim.Adam(flow.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config.lr_decay
        )
        self.energy_baseline: float | None = None

    def compute_local_energies(
        self,
        configs: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute local energy E_loc(x) for each sampled configuration.

        E_loc(x) = H_{xx} + sum_{y connected} H_{xy} * exp(0.5 * (log p(y) - log p(x)))

        Parameters
        ----------
        configs : torch.Tensor
            (n_samples, 2 * n_orbitals) binary configurations.
        log_probs : torch.Tensor
            (n_samples,) log probabilities from sampling (no gradients needed).

        Returns
        -------
        torch.Tensor
            (n_samples,) local energies (float64, real-valued).
        """
        n_samples = configs.shape[0]
        device = self.device

        # Move configs to Hamiltonian's device for matrix element computation
        # Ensure long dtype for bitwise operations in get_connections (Slater-Condon)
        h_device = getattr(self.hamiltonian, "device", device)
        configs_h = configs.long().to(h_device)

        # --- Diagonal contributions: H_{xx} ---
        diag = self.hamiltonian.diagonal_elements_batch(configs_h)  # (n_samples,) float64

        # --- Off-diagonal contributions ---
        # For each config x, get connected configs y and matrix elements H_{xy}
        off_diag = torch.zeros(n_samples, dtype=torch.float64, device=h_device)

        # Collect all connections serially (vectorized batch requires GPU tensors)
        all_connected = []
        all_elements = []
        all_orig_indices = []

        for i in range(n_samples):
            connected, elements = self.hamiltonian.get_connections(configs_h[i])
            if len(connected) > 0:
                all_connected.append(connected)
                all_elements.append(elements)
                all_orig_indices.append(
                    torch.full((len(connected),), i, dtype=torch.long, device=h_device)
                )

        if all_connected:
            all_connected_t = torch.cat(all_connected, dim=0)  # (total_conn, num_sites)
            all_elements_t = torch.cat(all_elements, dim=0)  # (total_conn,)
            all_orig_idx_t = torch.cat(all_orig_indices, dim=0)  # (total_conn,)

            # Compute log_prob of connected configs via teacher forcing
            # Move to flow's device (may differ from Hamiltonian device)
            flow_device = next(self.flow.parameters()).device
            all_connected_flow = all_connected_t.float().to(flow_device)

            # Batch log_prob computation to avoid OOM
            batch_size = self.config.log_prob_batch_size
            n_conn = all_connected_flow.shape[0]

            if batch_size <= 0 or n_conn <= batch_size:
                with torch.no_grad():
                    log_probs_connected = self.flow.log_prob(all_connected_flow)
            else:
                lp_parts = []
                for start in range(0, n_conn, batch_size):
                    end = min(start + batch_size, n_conn)
                    with torch.no_grad():
                        lp_batch = self.flow.log_prob(all_connected_flow[start:end])
                    lp_parts.append(lp_batch)
                log_probs_connected = torch.cat(lp_parts, dim=0)

            log_probs_connected = log_probs_connected.to(h_device).double()

            # log_probs of original configs at each connection's source index
            log_probs_orig = log_probs.to(h_device).double()[all_orig_idx_t]

            # Amplitude ratio: psi(y)/psi(x) = sqrt(p(y)/p(x))
            #                                = exp(0.5 * (log p(y) - log p(x)))
            log_ratio = 0.5 * (log_probs_connected - log_probs_orig)
            # Clamp for numerical stability (avoid exp overflow)
            log_ratio = torch.clamp(log_ratio, min=-30.0, max=30.0)
            amplitude_ratios = torch.exp(log_ratio)

            # Weighted off-diagonal contributions
            all_elements_real = all_elements_t.double()
            if all_elements_real.is_complex():
                all_elements_real = all_elements_real.real
            weighted = all_elements_real * amplitude_ratios

            # Scatter-add to source config indices
            off_diag.scatter_add_(0, all_orig_idx_t, weighted)

        local_energies = diag + off_diag
        if local_energies.is_complex():
            local_energies = local_energies.real

        return local_energies.double()

    def train_step(self) -> dict[str, float]:
        """Execute a single VMC optimization step.

        1. Sample configs from the flow (no gradient).
        2. Recompute log_prob with gradient (teacher forcing).
        3. Compute local energies (no gradient).
        4. REINFORCE loss with baseline subtraction.
        5. Backward, clip, step.

        Returns
        -------
        dict
            Metrics: 'energy', 'energy_std', 'grad_norm', 'loss'.
        """
        self.flow.train()
        cfg = self.config

        # Step 1: Sample configs from flow (no gradient for sampling)
        with torch.no_grad():
            states, sample_log_probs = self.flow._sample_autoregressive(cfg.n_samples)
            configs = states_to_configs(states, self.flow.n_orbitals)

        # Step 2: Recompute log_prob WITH gradient tracking (teacher forcing)
        flow_device = next(self.flow.parameters()).device
        configs_for_logprob = configs.float().to(flow_device)
        log_probs_grad = self.flow.log_prob(configs_for_logprob)  # has grad_fn

        # Step 3: Compute local energies (no gradient needed)
        with torch.no_grad():
            local_energies = self.compute_local_energies(configs, sample_log_probs)

        # Step 4: REINFORCE loss with EMA baseline
        energy_mean = local_energies.mean().item()
        energy_std = local_energies.std(correction=0).item()

        # Update baseline (EMA)
        if self.energy_baseline is None:
            self.energy_baseline = energy_mean
        else:
            self.energy_baseline = (
                cfg.baseline_decay * self.energy_baseline + (1.0 - cfg.baseline_decay) * energy_mean
            )

        # Advantage = E_loc(x) - baseline (detached, float64)
        advantage = local_energies - self.energy_baseline

        # REINFORCE loss: E[(E_loc - b) * log p(x)]
        # The factor of 2 from psi = sqrt(p) is absorbed:
        #   nabla <H> = 2 * E[(E_loc - b) * 0.5 * nabla log p]
        #             = E[(E_loc - b) * nabla log p]
        # So the loss gradient gives the correct energy gradient.
        advantage_f = advantage.detach().float().to(flow_device)
        loss = (advantage_f * log_probs_grad).mean()

        # Step 5: Backward + clip + step
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.flow.parameters(), cfg.clip_grad)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "energy": energy_mean,
            "energy_std": energy_std,
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            "loss": loss.item(),
        }

    def train(self, verbose: bool = True) -> dict[str, Any]:
        """Run the full VMC training loop.

        Parameters
        ----------
        verbose : bool
            Print progress every 50 steps.

        Returns
        -------
        dict
            Results with keys:
            - 'energies': list of mean energies per step
            - 'best_energy': lowest energy observed
            - 'n_steps': total number of steps taken
            - 'converged': whether convergence criterion was met
        """
        energies: list[float] = []
        best_energy = float("inf")
        converged = False
        energy_change = float("inf")

        for step in range(self.config.n_steps):
            metrics = self.train_step()
            energies.append(metrics["energy"])
            best_energy = min(best_energy, metrics["energy"])

            # Convergence check
            if step >= self.config.min_steps and len(energies) >= self.config.convergence_window:
                window = energies[-self.config.convergence_window :]
                energy_change = max(window) - min(window)
                if energy_change < self.config.convergence_threshold:
                    converged = True
                    if verbose:
                        print(f"VMC converged at step {step}: " f"E = {best_energy:.6f} Ha")
                    break

            if verbose and step % 50 == 0:
                print(
                    f"VMC step {step}: E = {metrics['energy']:.6f} "
                    f"+/- {metrics['energy_std']:.4f} Ha  "
                    f"(grad_norm={metrics['grad_norm']:.4f})"
                )

        return {
            "energies": energies,
            "best_energy": best_energy,
            "n_steps": len(energies),
            "converged": converged,
        }
