"""Variational Monte Carlo (VMC) trainer for autoregressive normalizing flows.

This module implements VMC training that directly minimizes the variational energy
<psi|H|psi> using either REINFORCE or MinSR (Minimum Stochastic Reconfiguration)
gradient estimators.  Unlike the cross-entropy teacher loss in
``PhysicsGuidedFlowTrainer``, VMC targets the ground-state energy directly without
requiring an auxiliary NQS network.

**Positive-real wavefunction ansatz**:
    psi(x) = sqrt(p_theta(x))

where p_theta is the probability distribution defined by the autoregressive flow.
This restricts the ansatz to wavefunctions with non-negative amplitudes (sign
structure is encoded in the flow probabilities).

**Local energy**:
    E_loc(x) = sum_y H_{xy} * psi(y) / psi(x)
             = H_{xx} + sum_{y connected to x} H_{xy} * sqrt(p(y) / p(x))
             = H_{xx} + sum_{y connected} H_{xy} * exp(0.5 * (log p(y) - log p(x)))

**Optimizer comparison**:

REINFORCE (baseline subtracted):
    nabla <H> = E_{x~p}[(E_loc(x) - baseline) * nabla log p_theta(x)]
    - Simple score function estimator, no curvature information.
    - Variance grows as |E_loc - E|^2, which makes it diverge at scale (>16Q).
    - Baseline subtraction reduces variance but does not address curvature.

MinSR (Minimum Stochastic Reconfiguration, Chen & Heyl, Nature Physics 2024):
    Solves S * delta_theta = -f  where  S = Fisher information matrix.
    - S_kl = <O_k^dagger O_l> - <O_k^dagger><O_l>,  O_k = d log psi / d theta_k
    - f_k  = <O_k^dagger E_loc> - <O_k^dagger><E_loc>  (energy gradient)
    - Avoids building the full (N_params x N_params) S matrix by using the
      Woodbury identity: cost is O(N_params * N_samples^2) instead of O(N_params^3).
    - Regularization: (S + lambda*I) with decaying lambda for numerical stability.
    - The entire NQS/VMC field uses SR or its variants (KFAC, SPRING).
      MinSR is the O(N_params)-cost variant that enables > 10^6 parameters.

References:
    - Carleo & Troyer, "Solving the Quantum Many-Body Problem with Artificial
      Neural Networks", Science 355 (2017)
    - Chen & Heyl, "Minimum Stochastic Reconfiguration", Nature Physics 20 (2024)
    - Rende et al., "A simple linear algebra identity to optimize large-scale
      neural network quantum states", Commun. Phys. 7, 260 (2024)
    - QiankunNet: Li et al., "QiankunNet: Transformer-Based Autoregressive Neural
      Quantum States for Electronic Structure" (Nature Comms, 2025)
    - Barrett et al., "Autoregressive neural-network wavefunctions for ab initio
      quantum chemistry" (Nature Machine Intelligence, 2022)
"""

import warnings
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
        Initial learning rate.  For ``optimizer_type="reinforce"`` this is the
        Adam learning rate; for ``"minsr"`` it scales the SR parameter update.
    lr_decay : float
        Per-step multiplicative learning rate decay (ExponentialLR gamma).
    clip_grad : float
        Maximum gradient norm for gradient clipping (REINFORCE) or maximum
        parameter update norm (MinSR).
    baseline_decay : float
        Exponential moving average decay for the energy baseline.  Used by
        REINFORCE for variance reduction.  MinSR ignores this (it has curvature
        information from the Fisher matrix).
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
    optimizer_type : str
        Gradient estimator / optimizer.  Options:

        - ``"minsr"``: Minimum Stochastic Reconfiguration (default).  Uses the
          Fisher information matrix via the Woodbury identity for O(N_params)
          cost.  This is the field standard for NQS/VMC (Chen & Heyl 2024).
        - ``"reinforce"``: REINFORCE with baseline subtraction.  Simple but
          high variance; does not converge beyond ~16 qubits.

    sr_regularization : float
        Initial diagonal regularization lambda for S + lambda*I in MinSR.
        Larger values make the update more like plain gradient descent;
        smaller values trust the curvature more.  Typical range: 1e-4 to 1e-2.
    sr_reg_decay : float
        Per-step multiplicative decay for ``sr_regularization``.  Set to 1.0
        to keep lambda constant.
    sr_reg_min : float
        Minimum value for ``sr_regularization`` after decay.  Prevents lambda
        from decaying to dangerously small values that cause numerical
        instability in the linear solve.
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
    optimizer_type: str = "minsr"
    sr_regularization: float = 1e-3
    sr_reg_decay: float = 0.99
    sr_reg_min: float = 1e-5


class VMCTrainer:
    """Variational Monte Carlo trainer for autoregressive flow.

    Directly minimizes the variational energy <psi|H|psi> using either
    REINFORCE or MinSR (Minimum Stochastic Reconfiguration).

    The flow defines psi(x) = sqrt(p_theta(x)) (positive real wavefunction).
    With a sign network, psi(x) = sqrt(p_theta(x)) * s_phi(x).

    MinSR (default) solves the SR equation S * delta = -f using the Woodbury
    identity, giving O(N_params * N_samples^2) cost instead of O(N_params^3).
    This provides curvature-aware updates that converge at 24Q+ scale where
    REINFORCE diverges.

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
    sign_network : nn.Module, optional
        Sign network for signed wavefunction ansatz.  When present, MinSR
        applies to flow parameters only; sign network uses direct backprop.

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

        # Detect if sign_network is actually a PhaseNetwork (P2.1)
        try:
            from .sign_network import PhaseNetwork
        except ImportError:
            try:
                from flows.sign_network import PhaseNetwork
            except ImportError:
                PhaseNetwork = None
        self._is_phase_network = (
            PhaseNetwork is not None and isinstance(sign_network, PhaseNetwork)
        )

        # Validate optimizer_type
        valid_optimizers = ("reinforce", "minsr")
        if self.config.optimizer_type not in valid_optimizers:
            raise ValueError(
                f"Unknown optimizer_type '{self.config.optimizer_type}'. "
                f"Valid options: {valid_optimizers}"
            )

        # Current SR regularization (decays over training)
        self._sr_lambda = self.config.sr_regularization

        if self.config.optimizer_type == "reinforce":
            # REINFORCE: use Adam optimizer for all parameters
            params = list(flow.parameters())
            if sign_network is not None:
                params += list(sign_network.parameters())

            self.optimizer = torch.optim.Adam(params, lr=self.config.lr)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=self.config.lr_decay
            )
        else:
            # MinSR: manual parameter updates for flow, Adam for sign network.
            # We still use a "dummy" optimizer for sign network if present,
            # and track learning rate decay manually for the MinSR flow update.
            self._minsr_lr = self.config.lr

            if sign_network is not None:
                # Sign network keeps its own Adam optimizer (not SR-updated)
                self._sign_optimizer = torch.optim.Adam(
                    sign_network.parameters(), lr=self.config.lr
                )
                self._sign_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self._sign_optimizer, gamma=self.config.lr_decay
                )
            else:
                self._sign_optimizer = None
                self._sign_scheduler = None

            # Expose a dummy optimizer for compatibility with existing code
            # that checks trainer.optimizer.param_groups[0]["lr"] (e.g., LR
            # decay tests).  We manually sync its lr from _minsr_lr each step.
            # Using a sentinel param so Adam doesn't complain about empty params.
            self._dummy_param = torch.nn.Parameter(torch.zeros(1))
            self.optimizer = torch.optim.Adam([self._dummy_param], lr=self.config.lr)
            self.scheduler = None  # Not used for MinSR; LR tracked via _minsr_lr

        self.energy_baseline: float | None = None

        # Count flow parameters once for Jacobian pre-allocation
        self._flow_n_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)

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
        through the sign network for direct backpropagation of nabla_phi <H>.

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

        # --- Sign/phase values for sampled configs ---
        if self.sign_network is not None:
            flow_device = next(self.flow.parameters()).device
            if self._is_phase_network:
                # PhaseNetwork: get phase values φ(x) ∈ [0, 2π)
                phase_x = self.sign_network(configs.float().to(flow_device))  # (n_samples,) WITH grad
                sign_x = phase_x  # Store phase values (used differently below)
            else:
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
                # Batch the sign/phase network to avoid OOM for large connection counts.
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

                sign_x_expanded = sign_x[all_orig_idx_t.to(sign_x.device)]

                if self._is_phase_network:
                    # PhaseNetwork: phase ratio e^{i(φ(y)-φ(x))}
                    # Always well-defined (no division by zero).
                    # Use full complex phase factor during training — the imaginary
                    # parts carry gradient information even for real Hamiltonians.
                    # At convergence (φ ∈ {0,π}), Im→0 naturally.
                    phase_diff = sign_y - sign_x_expanded  # φ(y) - φ(x)
                    phase_factor = torch.exp(1j * phase_diff.double())  # complex128
                    sign_weighted = weighted.to(flow_device).to(torch.complex128) * phase_factor
                else:
                    # SignNetwork: sign ratio s(y)/s(x)
                    # Guard against s(x)=0 (tanh(0)=0).
                    eps = 1e-8
                    safe_denom = sign_x_expanded + eps * (2.0 * (sign_x_expanded >= 0).float() - 1.0)
                    sign_ratio = sign_y / safe_denom
                    sign_ratio_f64 = sign_ratio.double()
                    sign_weighted = weighted.to(flow_device) * sign_ratio_f64

                # Scatter-add: match dtype (complex128 for phase, float64 for sign)
                scatter_dtype = sign_weighted.dtype
                sign_off_diag = torch.zeros(n_samples, dtype=scatter_dtype, device=flow_device)
                sign_off_diag.scatter_add_(0, all_orig_idx_t.to(flow_device), sign_weighted)
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

    def _compute_per_sample_jacobian(
        self,
        configs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Jacobian of log_prob w.r.t. flow parameters for each sample.

        For each sample x_i, computes O_ki = d log p(x_i) / d theta_k, giving a
        matrix of shape (N_samples, N_params).

        This is the "O-matrix" in the SR literature.  We use a simple loop over
        samples with retain_graph=True.  This is O(N_samples * backward_cost) but
        avoids the complexity of torch.func.vmap which has compatibility issues
        with some modules (nn.Embedding, KV cache, etc.).

        Parameters
        ----------
        configs : torch.Tensor
            (N_samples, 2 * n_orbitals) binary configurations.

        Returns
        -------
        torch.Tensor
            (N_samples, N_params) Jacobian in float64 for numerical stability.
        """
        flow_device = next(self.flow.parameters()).device
        configs_f = configs.float().to(flow_device)
        n_samples = configs.shape[0]

        # Compute log_prob with gradient graph
        log_probs = self.flow.log_prob(configs_f)  # (N_samples,)

        # Collect parameter references (only requires_grad params)
        params = [p for p in self.flow.parameters() if p.requires_grad]

        # Pre-allocate Jacobian in FP64
        jacobian = torch.zeros(
            n_samples, self._flow_n_params, dtype=torch.float64, device=flow_device
        )

        for i in range(n_samples):
            # Zero existing gradients
            self.flow.zero_grad()

            # Backward for single sample; retain graph for remaining samples
            retain = i < n_samples - 1
            log_probs[i].backward(retain_graph=retain)

            # Collect flattened gradients into Jacobian row
            offset = 0
            for p in params:
                if p.grad is not None:
                    numel = p.grad.numel()
                    jacobian[i, offset : offset + numel] = p.grad.flatten().double()
                    offset += numel
                else:
                    offset += p.numel()

        # Clear residual gradients
        self.flow.zero_grad()

        return jacobian

    def _minsr_update(
        self,
        jacobian: torch.Tensor,
        local_energies: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Compute the MinSR parameter update direction.

        Solves the regularized SR equation using the push-through identity:
            delta_theta = -lr * J_c^T @ (J_c @ J_c^T + lambda * N * I)^{-1} @ e_c

        where:
            J_c = centered Jacobian (N_samples, N_params)
            e_c = centered local energies (N_samples,)
            lambda = regularization parameter

        The inner linear system is (N_samples x N_samples), which is much smaller
        than (N_params x N_params) when N_samples << N_params.

        Parameters
        ----------
        jacobian : torch.Tensor
            (N_samples, N_params) O-matrix, d log p(x_i) / d theta_k.
        local_energies : torch.Tensor
            (N_samples,) real-valued local energies E_loc(x_i).

        Returns
        -------
        tuple[torch.Tensor, float]
            - (N_params,) parameter update delta_theta
            - Effective gradient norm (for logging)
        """
        n_samples = jacobian.shape[0]
        device = jacobian.device

        # Work in FP64 for numerical stability of the linear solve
        J = jacobian.double()  # (N_samples, N_params)
        e_loc = local_energies.detach().double().to(device)  # (N_samples,)

        # Center the Jacobian: J_c = J - mean(J, dim=0)
        J_mean = J.mean(dim=0, keepdim=True)  # (1, N_params)
        J_c = J - J_mean  # (N_samples, N_params)

        # Center the local energies: e_c = e_loc - mean(e_loc)
        e_mean = e_loc.mean()
        e_c = e_loc - e_mean  # (N_samples,)

        # Compute the energy gradient f = J_c^T @ e_c / N_samples
        # This is equivalent to the REINFORCE gradient with optimal baseline.
        # f shape: (N_params,)
        f = J_c.T @ e_c / n_samples

        # --- Woodbury / push-through identity solve ---
        # We want: delta = (S + lambda*I)^{-1} @ f
        # where S = J_c^T @ J_c / N (Fisher matrix), f = J_c^T @ e_c / N.
        #
        # Push-through identity: (A^T A + λI)^{-1} A^T = A^T (A A^T + λI)^{-1}
        # With A = J_c/√N, λ → λ:
        #   (S + λI)^{-1} f = J_c^T (J_c J_c^T + λN I)^{-1} e_c  [N's cancel]
        #
        # So we solve:
        #   (J_c @ J_c^T + lambda * N * I) @ z = e_c
        #   delta = J_c^T @ z   (NO division by N — the N's cancel in the derivation)
        #
        # The matrix (J_c @ J_c^T + lambda * N * I) is (N_samples x N_samples).

        lam = self._sr_lambda
        G = J_c @ J_c.T  # (N_samples, N_samples) — the Gram matrix
        G += lam * n_samples * torch.eye(n_samples, dtype=torch.float64, device=device)

        # Solve G @ z = e_c
        try:
            z = torch.linalg.solve(G, e_c)  # (N_samples,)
        except torch.linalg.LinAlgError:
            # Singular matrix despite regularization — increase lambda and retry
            warnings.warn(
                f"MinSR: singular Gram matrix at lambda={lam:.2e}. "
                f"Increasing lambda to {lam * 10:.2e} and retrying.",
                RuntimeWarning,
                stacklevel=2,
            )
            G += lam * 9 * n_samples * torch.eye(n_samples, dtype=torch.float64, device=device)
            try:
                z = torch.linalg.solve(G, e_c)
            except torch.linalg.LinAlgError:
                # Still singular — fall back to gradient descent direction
                warnings.warn(
                    "MinSR: Gram matrix still singular after lambda increase. "
                    "Falling back to plain gradient direction.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                delta = -self._minsr_lr * f
                grad_norm = f.norm().item()
                return delta.float(), grad_norm

        # Parameter update (no /N — the N's cancel in the push-through derivation)
        delta = J_c.T @ z  # (N_params,)
        delta = -self._minsr_lr * delta

        grad_norm = delta.norm().item()

        return delta.float(), grad_norm

    def _apply_minsr_update(self, delta: torch.Tensor) -> None:
        """Apply the MinSR parameter update to flow parameters.

        Parameters
        ----------
        delta : torch.Tensor
            (N_params,) parameter update, same ordering as flow.parameters().
        """
        offset = 0
        for p in self.flow.parameters():
            if not p.requires_grad:
                continue
            numel = p.numel()
            p.data.add_(delta[offset : offset + numel].reshape(p.shape).to(p.device))
            offset += numel

    def _reinforce_step(self) -> dict[str, float]:
        """Execute a single REINFORCE VMC optimization step.

        This is the original train_step implementation, preserved for backward
        compatibility when ``optimizer_type="reinforce"``.

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
        # This is correct because nabla_phi <H> = E_p[nabla_phi E_loc(x)] (p doesn't depend on phi).
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

    def _minsr_step(self) -> dict[str, float]:
        """Execute a single MinSR VMC optimization step.

        Algorithm:
            1. Sample configs from flow (no grad).
            2. Compute local energies (no grad through flow).
            3. Compute per-sample Jacobian d log p(x_i) / d theta_k.
            4. Solve the MinSR linear system for delta_theta.
            5. Clip and apply the parameter update.
            6. If sign network present, backprop E[E_loc] through sign network.
            7. Decay SR regularization and learning rate.

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

        # Step 2: Compute local energies (no grad through flow)
        if self.sign_network is not None:
            # WITH gradient through sign network (for sign loss below)
            local_energies = self.compute_local_energies(configs, sample_log_probs)
        else:
            with torch.no_grad():
                local_energies = self.compute_local_energies(configs, sample_log_probs)

        energy_mean = local_energies.detach().mean().item()
        energy_std = local_energies.detach().std(correction=0).item()

        # Step 3: Compute per-sample Jacobian of log_prob w.r.t. flow params
        jacobian = self._compute_per_sample_jacobian(configs)

        # Check for NaN in Jacobian or local energies
        if torch.isnan(jacobian).any() or torch.isnan(local_energies.detach()).any():
            warnings.warn(
                "MinSR: NaN detected in Jacobian or local energies. Skipping step.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Decay LR and SR lambda even on skip, to maintain schedule
            self._minsr_lr *= cfg.lr_decay
            self._sr_lambda = max(self._sr_lambda * cfg.sr_reg_decay, cfg.sr_reg_min)
            if self._sign_optimizer is not None:
                self._sign_scheduler.step()
            self.optimizer.param_groups[0]["lr"] = self._minsr_lr
            return {
                "energy": energy_mean,
                "energy_std": energy_std,
                "grad_norm": 0.0,
                "loss": energy_mean,
            }

        # Step 4: Solve the MinSR linear system
        delta, grad_norm_raw = self._minsr_update(jacobian, local_energies)

        # Step 5: Clip the parameter update norm
        update_norm = delta.norm().item()
        if update_norm > cfg.clip_grad:
            delta = delta * (cfg.clip_grad / update_norm)
            grad_norm = cfg.clip_grad
        else:
            grad_norm = update_norm

        # Step 6: Apply the update to flow parameters
        self._apply_minsr_update(delta)

        # Step 6b: Sign network update via direct backprop (separate from MinSR)
        if self.sign_network is not None and local_energies.requires_grad:
            if self._sign_optimizer is not None:
                self._sign_optimizer.zero_grad()
            sign_loss = local_energies.mean()
            sign_loss.backward()
            if self._sign_optimizer is not None:
                # Clip sign network gradients
                torch.nn.utils.clip_grad_norm_(self.sign_network.parameters(), cfg.clip_grad)
                self._sign_optimizer.step()
                self._sign_scheduler.step()

        # Step 7: Decay SR regularization and learning rate
        self._sr_lambda = max(self._sr_lambda * cfg.sr_reg_decay, cfg.sr_reg_min)
        self._minsr_lr *= cfg.lr_decay

        # Sync monitoring optimizer's LR for code that reads trainer.optimizer.param_groups
        self.optimizer.param_groups[0]["lr"] = self._minsr_lr

        return {
            "energy": energy_mean,
            "energy_std": energy_std,
            "grad_norm": grad_norm,
            "loss": energy_mean,  # For MinSR, "loss" is the energy (no REINFORCE loss)
        }

    def train_step(self) -> dict[str, float]:
        """Execute a single VMC optimization step.

        Dispatches to ``_reinforce_step()`` or ``_minsr_step()`` based on
        ``self.config.optimizer_type``.

        Returns
        -------
        dict
            Metrics: 'energy', 'energy_std', 'grad_norm', 'loss'.
        """
        if self.config.optimizer_type == "reinforce":
            return self._reinforce_step()
        else:
            return self._minsr_step()

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

        optimizer_label = self.config.optimizer_type.upper()

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
                        print(
                            f"VMC ({optimizer_label}) converged at step {step}: "
                            f"E = {best_energy:.6f} Ha"
                        )
                    break

            if verbose and step % 50 == 0:
                extra = ""
                if self.config.optimizer_type == "minsr":
                    extra = f", sr_lambda={self._sr_lambda:.2e}"
                print(
                    f"VMC ({optimizer_label}) step {step}: "
                    f"E = {metrics['energy']:.6f} "
                    f"+/- {metrics['energy_std']:.4f} Ha  "
                    f"(grad_norm={metrics['grad_norm']:.4f}{extra})"
                )

        return {
            "energies": energies,
            "best_energy": best_energy,
            "n_steps": len(energies),
            "converged": converged,
        }
