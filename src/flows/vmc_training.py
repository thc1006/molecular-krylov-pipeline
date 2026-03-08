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
        sign_network: nn.Module | None = None,
    ):
        self.flow = flow
        self.hamiltonian = hamiltonian
        self.config = config or VMCConfig()
        self.device = device
        self.sign_network = sign_network

        # Collect all trainable parameters (flow + sign network)
        params = list(flow.parameters())
        if sign_network is not None:
            params += list(sign_network.parameters())

        self.optimizer = torch.optim.Adam(params, lr=self.config.lr)
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

        Without sign network (positive-real ansatz):
            E_loc(x) = H_{xx} + sum_{y connected} H_{xy} * exp(0.5 * (log p(y) - log p(x)))

        With sign network (signed ansatz):
            E_loc(x) = H_{xx} + sum_{y connected} H_{xy} * exp(0.5 * (log p(y) - log p(x))) * s(y)/s(x)

        When a sign network is present, the returned tensor carries gradients
        through the sign network for direct backpropagation of ∇_φ <H>.

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

        # --- Sign values for sampled configs ---
        if self.sign_network is not None:
            flow_device = next(self.flow.parameters()).device
            sign_x = self.sign_network(configs.float().to(flow_device))  # (n_samples,) WITH grad
        else:
            sign_x = None

        # --- Off-diagonal contributions ---
        # For each config x, get connected configs y and matrix elements H_{xy}
        off_diag = torch.zeros(n_samples, dtype=torch.float64, device=h_device)
        # Track sign contribution separately so we can backprop through it
        sign_off_diag = None

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

            # Amplitude ratio: |psi(y)|/|psi(x)| = sqrt(p(y)/p(x))
            #                                    = exp(0.5 * (log p(y) - log p(x)))
            log_ratio = 0.5 * (log_probs_connected - log_probs_orig)
            # Clamp for numerical stability (avoid exp overflow)
            log_ratio = torch.clamp(log_ratio, min=-30.0, max=30.0)
            amplitude_ratios = torch.exp(log_ratio)

            # Weighted off-diagonal contributions (no sign)
            all_elements_real = all_elements_t.double()
            if all_elements_real.is_complex():
                all_elements_real = all_elements_real.real
            weighted = all_elements_real * amplitude_ratios

            if sign_x is not None:
                # Compute sign ratios s(y)/s(x) WITH gradient tracking.
                # Batch the sign network to avoid OOM for large connection counts.
                batch_size = self.config.log_prob_batch_size
                n_conn = all_connected_flow.shape[0]
                if batch_size <= 0 or n_conn <= batch_size:
                    sign_y = self.sign_network(all_connected_flow)
                else:
                    sign_parts = []
                    for start in range(0, n_conn, batch_size):
                        end = min(start + batch_size, n_conn)
                        sign_parts.append(self.sign_network(all_connected_flow[start:end]))
                    sign_y = torch.cat(sign_parts, dim=0)

                sign_x_expanded = sign_x[all_orig_idx_t]  # (total_conn,) WITH grad

                # Safe sign ratio: s(y) / s(x).  Guard against s(x)=0
                # (tanh(0)=0, sign(0)=0, so `eps*sign(x)` fails at exactly 0).
                # Use directional eps: positive when s(x)>=0, negative otherwise.
                eps = 1e-8
                safe_denom = sign_x_expanded + eps * (
                    2.0 * (sign_x_expanded >= 0).float() - 1.0
                )
                sign_ratio = sign_y / safe_denom

                # Keep FP64 precision: promote sign_ratio to double for
                # multiplication with weighted (float64 Hamiltonian elements).
                sign_ratio_f64 = sign_ratio.double()
                sign_weighted = weighted.to(flow_device) * sign_ratio_f64

                # Scatter-add in FP64 for precision
                sign_off_diag = torch.zeros(
                    n_samples, dtype=torch.float64, device=flow_device
                )
                sign_off_diag.scatter_add_(
                    0, all_orig_idx_t.to(flow_device), sign_weighted
                )
            else:
                # No sign network: scatter-add directly
                off_diag.scatter_add_(0, all_orig_idx_t, weighted)

        if sign_off_diag is not None:
            # Return in FP64 with gradient through sign network
            local_energies = diag.double().to(sign_off_diag.device) + sign_off_diag
            if local_energies.is_complex():
                local_energies = local_energies.real
            return local_energies
        else:
            local_energies = diag + off_diag
            if local_energies.is_complex():
                local_energies = local_energies.real
            return local_energies.double()

    def train_step(self) -> dict[str, float]:
        """Execute a single VMC optimization step.

        Without sign network:
            1. Sample configs (no grad).
            2. Recompute log_prob with grad (teacher forcing).
            3. Compute local energies (no grad).
            4. REINFORCE loss with baseline.
            5. Backward, clip, step.

        With sign network:
            1. Sample configs (no grad).
            2. Recompute log_prob with grad (for REINFORCE on flow).
            3. Compute local energies WITH grad through sign network.
            4. REINFORCE loss for flow (detached E_loc × log_prob).
            5. Sign loss = E[E_loc] (direct backprop through sign network).
            6. Combined backward, clip, step.

        Returns
        -------
        dict
            Metrics: 'energy', 'energy_std', 'grad_norm', 'loss'.
        """
        self.flow.train()
        if self.sign_network is not None:
            self.sign_network.train()
        cfg = self.config

        # Step 1: Sample configs from flow (no gradient for sampling)
        with torch.no_grad():
            states, sample_log_probs = self.flow._sample_autoregressive(cfg.n_samples)
            configs = states_to_configs(states, self.flow.n_orbitals)

        # Step 2: Recompute log_prob WITH gradient tracking (teacher forcing)
        flow_device = next(self.flow.parameters()).device
        configs_for_logprob = configs.float().to(flow_device)
        log_probs_grad = self.flow.log_prob(configs_for_logprob)  # has grad_fn

        # Step 3: Compute local energies
        if self.sign_network is not None:
            # WITH gradient through sign network
            local_energies = self.compute_local_energies(configs, sample_log_probs)
        else:
            with torch.no_grad():
                local_energies = self.compute_local_energies(configs, sample_log_probs)

        # Step 4: REINFORCE loss with EMA baseline
        energy_mean = local_energies.detach().mean().item()
        energy_std = local_energies.detach().std(correction=0).item()

        # Update baseline (EMA)
        if self.energy_baseline is None:
            self.energy_baseline = energy_mean
        else:
            self.energy_baseline = (
                cfg.baseline_decay * self.energy_baseline + (1.0 - cfg.baseline_decay) * energy_mean
            )

        # REINFORCE loss for flow: E[(E_loc - baseline) * log p(x)]
        # E_loc is always detached for REINFORCE — flow gradient comes from log_prob only.
        advantage_f = (local_energies.detach() - self.energy_baseline).float().to(flow_device)
        reinforce_loss = (advantage_f * log_probs_grad).mean()

        # Sign loss: E[E_loc] — direct backprop through sign network.
        # This is correct because ∇_φ <H> = E_p[∇_φ E_loc(x)] (p doesn't depend on φ).
        if self.sign_network is not None and local_energies.requires_grad:
            sign_loss = local_energies.mean()
            total_loss = reinforce_loss + sign_loss
        else:
            total_loss = reinforce_loss

        # Step 5: Backward + clip + step
        self.optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients for all parameters
        all_params = list(self.flow.parameters())
        if self.sign_network is not None:
            all_params += list(self.sign_network.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, cfg.clip_grad)

        self.optimizer.step()
        self.scheduler.step()

        return {
            "energy": energy_mean,
            "energy_std": energy_std,
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            "loss": total_loss.item(),
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
