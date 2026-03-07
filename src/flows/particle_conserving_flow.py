"""
Particle-Conserving Normalizing Flow for Molecular Systems.

This module implements a normalizing flow that respects physical constraints:
1. Fixed electron number N_e (n_alpha + n_beta)
2. Fixed spin projection Sz (n_alpha - n_beta)

The key insight is to sample orbital indices (which orbitals are occupied)
rather than binary occupation strings, which naturally enforces particle
number conservation.

References:
- Differentiable top-k: Cordonnier et al., "Differentiable Subset Selection"
- Gumbel-top-k: Kool et al., "Stochastic Beams and Where to Find Them" (2019)
- Sigmoid top-k: Thomas Ahle, "A Differentiable Top-k Layer for PyTorch" (2022)
- Plackett-Luce: probability model for without-replacement subset selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import permutations as _permutations
from typing import Tuple, Optional


class GumbelTopK(nn.Module):
    """
    Differentiable top-k selection using Gumbel-Softmax trick.

    Samples k indices from n options while maintaining differentiability.
    Forward pass: hard selection (argmax)
    Backward pass: soft gradients via Gumbel-Softmax
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        k: int,
        hard: bool = True
    ) -> torch.Tensor:
        """
        Select k indices from logits using Gumbel-top-k.

        Args:
            logits: (batch_size, n_options) unnormalized log-probabilities
            k: number of items to select
            hard: if True, use hard selection in forward, soft gradients in backward

        Returns:
            (batch_size, n_options) binary mask with exactly k ones per row
        """
        batch_size, n_options = logits.shape
        device = logits.device

        # Add Gumbel noise for stochastic selection
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(logits).clamp(min=1e-10)
        ))
        perturbed_logits = (logits + gumbel_noise) / self.temperature

        # Get top-k indices
        _, top_indices = torch.topk(perturbed_logits, k, dim=-1)

        # Create one-hot encoding
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, top_indices, 1.0)

        # Soft relaxation: full softmax over all positions (preserves gradients everywhere)
        soft = F.softmax(perturbed_logits, dim=-1)

        if hard:
            # Straight-through estimator: hard selection in forward, soft gradients in backward.
            # Use full softmax (not masked) so gradients flow to ALL positions, enabling
            # the network to learn to increase/decrease any orbital's probability.
            return one_hot - soft.detach() + soft
        else:
            # Fully soft (for exploration/temperature annealing)
            return soft


class _SigmoidTopKFn(torch.autograd.Function):
    """Custom autograd for sigmoid top-k with implicit differentiation.

    Forward: binary search for threshold t such that Σσ(x_i + t) = k.
    Backward: exact Jacobian J = diag(v) - vv^T / ||v||_1 where v = σ'(x+t).

    Reference: Thomas Ahle, "A Differentiable Top-k Layer for PyTorch" (2022)
    """

    @staticmethod
    def forward(ctx, xs, k):
        ts = _find_threshold(xs, k)
        ps = torch.sigmoid(xs + ts)
        ctx.save_for_backward(xs, ts)
        return ps

    @staticmethod
    def backward(ctx, grad_output):
        xs, ts = ctx.saved_tensors
        p = torch.sigmoid(xs + ts)
        v = p * (1 - p)  # σ'(x) = σ(x)(1-σ(x))
        s = v.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        uv = grad_output * v
        correction = -uv.sum(dim=-1, keepdim=True) * v / s
        return uv + correction, None


@torch.no_grad()
def _find_threshold(xs: torch.Tensor, k: int) -> torch.Tensor:
    """Binary search for threshold t such that Σσ(x_i + t) = k."""
    lo = -xs.max(dim=-1, keepdim=True).values - 10
    hi = -xs.min(dim=-1, keepdim=True).values + 10
    for _ in range(48):  # 48 iterations: 2^-48 ≈ 3.6e-15 (sufficient for float64 gradcheck)
        mid = (lo + hi) / 2
        count = torch.sigmoid(xs + mid).sum(dim=-1, keepdim=True)
        lo = torch.where(count < k, mid, lo)
        hi = torch.where(count < k, hi, mid)
    return (lo + hi) / 2


class SigmoidTopK(nn.Module):
    """
    Differentiable top-k via sigmoid threshold + implicit differentiation.

    Replaces GumbelTopK with a deterministic, gradient-exact alternative.

    Forward: find threshold t such that Σσ(x_i/τ + t) = k, return σ(x/τ + t).
    Backward: exact Jacobian via implicit differentiation of the constraint.

    Advantages over GumbelTopK:
    - Deterministic (no Gumbel noise)
    - Exact marginal probabilities
    - Better gradient quality at low temperature
    - O(n) forward (binary search is O(n × 64 iterations))

    Reference: Thomas Ahle, "A Differentiable Top-k Layer for PyTorch" (2022)
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        k: int,
        hard: bool = True,
    ) -> torch.Tensor:
        """
        Select k indices from logits using sigmoid threshold.

        Args:
            logits: (batch_size, n_options) unnormalized log-probabilities
            k: number of items to select
            hard: if True, return discrete {0,1} with STE gradients

        Returns:
            (batch_size, n_options) selection (binary if hard, [0,1] if soft)
        """
        scaled = logits / self.temperature
        soft = _SigmoidTopKFn.apply(scaled, k)

        if hard:
            _, top_idx = torch.topk(soft, k, dim=-1)
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, top_idx, 1.0)
            return one_hot - soft.detach() + soft
        return soft


class OrbitalScoringNetwork(nn.Module):
    """
    Neural network that scores orbital occupations.

    Outputs logits for each orbital indicating preference for occupation.
    Uses autoregressive structure to capture correlations between orbitals.
    """

    def __init__(
        self,
        n_orbitals: int,
        hidden_dims: list = [256, 256],
        context_dim: int = 64,
    ):
        super().__init__()
        self.n_orbitals = n_orbitals
        self.context_dim = context_dim

        # Context encoder: encodes partial occupation pattern
        self.context_encoder = nn.Sequential(
            nn.Linear(n_orbitals, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], context_dim),
        )

        # Scoring network: outputs logits for each orbital
        layers = []
        in_dim = context_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.SiLU(),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_orbitals))
        self.scorer = nn.Sequential(*layers)

        # Learnable prior for empty context
        self.prior_logits = nn.Parameter(torch.zeros(n_orbitals))

    def forward(
        self,
        context: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Compute orbital occupation logits.

        Args:
            context: (batch_size, n_orbitals) partial occupation (0/1 or soft)
                    If None, use prior logits
            batch_size: used when context is None

        Returns:
            (batch_size, n_orbitals) logits for each orbital
        """
        if context is None:
            # Use learnable prior
            return self.prior_logits.unsqueeze(0).expand(batch_size, -1)

        # Encode context and compute logits
        ctx = self.context_encoder(context.float())
        logits = self.scorer(ctx)

        # Mask already occupied orbitals (set to -inf)
        # This ensures we don't select the same orbital twice
        mask = context > 0.5  # Already occupied
        logits = logits.masked_fill(mask, float('-inf'))

        return logits


class ParticleConservingFlow(nn.Module):
    """
    Normalizing flow that generates valid molecular configurations.

    Key properties:
    - Always generates configurations with exactly n_alpha alpha electrons
    - Always generates configurations with exactly n_beta beta electrons
    - Uses separate networks for alpha and beta spin channels
    - Differentiable via Gumbel-top-k

    The output is a (batch_size, 2*n_orbitals) tensor where:
    - First n_orbitals: alpha spin occupations (0 or 1)
    - Last n_orbitals: beta spin occupations (0 or 1)
    """

    def __init__(
        self,
        n_orbitals: int,
        n_alpha: int,
        n_beta: int,
        hidden_dims: list = [256, 256],
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_orbitals = n_orbitals
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.n_qubits = 2 * n_orbitals

        # Alpha spin scoring: learnable logits (no context available for alpha)
        # Using a simple parameter instead of OrbitalScoringNetwork avoids dead weights,
        # since alpha is always sampled first without context.
        self.alpha_logits = nn.Parameter(torch.zeros(n_orbitals))

        # Alpha-beta correlation network
        # Beta scorer can see alpha configuration for correlation
        self.alpha_to_beta = nn.Sequential(
            nn.Linear(n_orbitals, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
        )
        self.beta_conditioned_scorer = nn.Sequential(
            nn.Linear(n_orbitals + 64, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[-1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[-1], n_orbitals),
        )

        # Gumbel-top-k selector
        self.gumbel_topk = GumbelTopK(temperature)
        self.temperature = temperature

    def set_temperature(self, temperature: float):
        """Update temperature for Gumbel sampling."""
        self.temperature = temperature
        self.gumbel_topk.temperature = temperature

    def sample(
        self,
        batch_size: int,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample valid molecular configurations.

        Args:
            batch_size: number of configurations to sample
            hard: if True, return discrete configurations

        Returns:
            configs: (batch_size, 2*n_orbitals) occupation configurations
            log_probs: (batch_size,) log probabilities of samples
        """
        device = next(self.parameters()).device

        # Sample alpha spin channel (learnable logits, no context)
        alpha_logits = self.alpha_logits.unsqueeze(0).expand(batch_size, -1)
        alpha_config = self.gumbel_topk(alpha_logits, self.n_alpha, hard=hard)

        # Sample beta spin channel conditioned on alpha
        alpha_context = self.alpha_to_beta(alpha_config)
        beta_input = torch.cat([
            torch.zeros(batch_size, self.n_orbitals, device=device),
            alpha_context
        ], dim=-1)
        beta_logits = self.beta_conditioned_scorer(beta_input)
        beta_config = self.gumbel_topk(beta_logits, self.n_beta, hard=hard)

        # Combine into full configuration
        configs = torch.cat([alpha_config, beta_config], dim=-1)

        # Compute log probabilities
        log_probs = self._compute_log_probs(alpha_logits, alpha_config,
                                            beta_logits, beta_config)

        return configs, log_probs

    def _compute_log_probs(
        self,
        alpha_logits: torch.Tensor,
        alpha_config: torch.Tensor,
        beta_logits: torch.Tensor,
        beta_config: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of configurations.

        Uses log-sum-exp for numerical stability.
        """
        # For top-k sampling, probability is product of selected softmax values
        # normalized by remaining options at each step

        alpha_log_probs = self._topk_log_prob(alpha_logits, alpha_config, self.n_alpha)
        beta_log_probs = self._topk_log_prob(beta_logits, beta_config, self.n_beta)

        return alpha_log_probs + beta_log_probs

    def _topk_log_prob(
        self,
        logits: torch.Tensor,
        selection: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Compute log P(unordered set S) under Plackett-Luce model.

        P(S) = Σ_{π ∈ Perm(S)} Π_{j=1}^k softmax(logit_{π(j)} | remaining)

        For k ≤ 5: exact enumeration over all k! orderings via logsumexp.
        For k > 5: single-ordering approximation (divides by k!).
        """
        batch_size, n = logits.shape

        # Get indices of selected orbitals
        _, selected_idx = torch.topk(selection, k, dim=-1)
        selected_idx, _ = selected_idx.sort(dim=-1)

        if k <= 5:
            return self._exact_pl_log_prob(logits, selected_idx, k)
        else:
            return self._approx_pl_log_prob(logits, selected_idx, k)

    def _exact_pl_log_prob(
        self,
        logits: torch.Tensor,
        selected_idx: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Exact Plackett-Luce: sum over all k! orderings via logsumexp."""
        all_perms = list(_permutations(range(k)))
        log_probs_per_perm = []

        for perm in all_perms:
            perm_idx = selected_idx[:, list(perm)]
            log_p = self._sequential_conditional(logits, perm_idx)
            log_probs_per_perm.append(log_p)

        stacked = torch.stack(log_probs_per_perm, dim=-1)
        return torch.logsumexp(stacked, dim=-1)

    def _approx_pl_log_prob(
        self,
        logits: torch.Tensor,
        selected_idx: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Approximate: single ordering + multiply by k! (for k > 5)."""
        log_p = self._sequential_conditional(logits, selected_idx)
        # Use math.lgamma (CPU, no tensor allocation) — k is a Python int
        import math
        log_k_factorial = math.lgamma(k + 1.0)
        return log_p + log_k_factorial

    def _sequential_conditional(
        self,
        logits: torch.Tensor,
        ordered_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log P(s1, s2, ..., sk | ordering) via sequential conditionals.

        P = Π_{j=1}^k softmax(logit_{s_j} | remaining orbitals)

        At each step, previously selected orbitals are masked to -inf.
        """
        batch_size, n = logits.shape
        k = ordered_idx.shape[1]
        log_prob = torch.zeros(batch_size, device=logits.device, dtype=logits.dtype)
        remaining = torch.ones(batch_size, n, dtype=torch.bool, device=logits.device)

        for step in range(k):
            masked_logits = logits.masked_fill(~remaining, float('-inf'))
            log_probs_step = F.log_softmax(masked_logits, dim=-1)
            idx = ordered_idx[:, step]
            log_prob = log_prob + log_probs_step.gather(1, idx.unsqueeze(1)).squeeze(1)
            remaining.scatter_(1, idx.unsqueeze(1), False)

        return log_prob

    def log_prob(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of given configurations.

        Args:
            configs: (batch_size, 2*n_orbitals) configurations

        Returns:
            (batch_size,) log probabilities
        """
        batch_size = configs.shape[0]
        device = configs.device

        alpha_config = configs[:, :self.n_orbitals]
        beta_config = configs[:, self.n_orbitals:]

        # Alpha logits (unconditional)
        alpha_logits = self.alpha_logits.unsqueeze(0).expand(batch_size, -1)

        # Beta logits (conditioned on alpha)
        alpha_context = self.alpha_to_beta(alpha_config.float())
        beta_input = torch.cat([
            torch.zeros(batch_size, self.n_orbitals, device=device),
            alpha_context
        ], dim=-1)
        beta_logits = self.beta_conditioned_scorer(beta_input)

        return self._compute_log_probs(alpha_logits, alpha_config,
                                       beta_logits, beta_config)

    def sample_with_unique(
        self,
        n_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample configurations and return unique ones.

        Args:
            n_samples: number of samples to draw

        Returns:
            unique_configs: (n_unique, 2*n_orbitals) unique configurations
            counts: (n_unique,) number of times each config was sampled
        """
        configs, _ = self.sample(n_samples, hard=True)

        # Convert to integer for hashing
        configs_int = configs.long()

        # Find unique configurations
        unique_configs, inverse_indices = torch.unique(
            configs_int, dim=0, return_inverse=True
        )

        # Count occurrences
        counts = torch.zeros(len(unique_configs), device=configs.device)
        counts.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float))

        return unique_configs, counts

    def estimate_discrete_prob(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Estimate probability of discrete configurations.

        This provides a differentiable approximation for training.

        Args:
            configs: (batch_size, 2*n_orbitals) configurations

        Returns:
            (batch_size,) estimated probabilities
        """
        log_probs = self.log_prob(configs)
        return torch.exp(log_probs)


class ParticleConservingFlowSampler(nn.Module):
    """
    Drop-in replacement for DiscreteFlowSampler with particle conservation.

    Provides the same interface as DiscreteFlowSampler for easy integration
    with the existing training pipeline.
    """

    def __init__(
        self,
        num_sites: int,
        n_alpha: int,
        n_beta: int,
        num_coupling_layers: int = 4,  # Ignored, kept for API compatibility
        hidden_dims: list = None,
        temperature: float = 1.0,
    ):
        super().__init__()

        # num_sites should be 2 * n_orbitals
        assert num_sites % 2 == 0, "num_sites must be even (2 * n_orbitals)"
        self.n_orbitals = num_sites // 2
        self.num_sites = num_sites
        self.n_alpha = n_alpha
        self.n_beta = n_beta

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.flow = ParticleConservingFlow(
            n_orbitals=self.n_orbitals,
            n_alpha=n_alpha,
            n_beta=n_beta,
            hidden_dims=hidden_dims,
            temperature=temperature,
        )

    def sample(
        self,
        n_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample configurations from the flow.

        Args:
            n_samples: number of samples

        Returns:
            log_probs: (n_samples,) log probabilities
            unique_configs: (n_unique, num_sites) unique configurations
        """
        configs, log_probs = self.flow.sample(n_samples, hard=True)

        # Get unique configurations
        unique_configs = torch.unique(configs.long(), dim=0)

        return log_probs, unique_configs

    def sample_with_probs(
        self,
        n_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample and return configs with their probabilities.

        Returns:
            configs: (n_samples, num_sites) all sampled configurations
            log_probs: (n_samples,) log probabilities
            unique_configs: (n_unique, num_sites) unique configurations
        """
        configs, log_probs = self.flow.sample(n_samples, hard=True)
        unique_configs = torch.unique(configs.long(), dim=0)

        return configs.long(), log_probs, unique_configs

    def log_prob(self, configs: torch.Tensor) -> torch.Tensor:
        """Compute log probability of configurations."""
        return self.flow.log_prob(configs)

    def estimate_discrete_prob(self, configs: torch.Tensor) -> torch.Tensor:
        """Estimate probability for discrete configurations."""
        return self.flow.estimate_discrete_prob(configs)

    def set_temperature(self, temperature: float):
        """Set sampling temperature."""
        self.flow.set_temperature(temperature)

    def forward(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for compatibility."""
        return self.sample(n_samples)


def verify_particle_conservation(
    configs: torch.Tensor,
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
) -> Tuple[bool, dict]:
    """
    Verify that configurations satisfy particle number constraints.

    Args:
        configs: (batch_size, 2*n_orbitals) configurations
        n_orbitals: number of spatial orbitals
        n_alpha: expected number of alpha electrons
        n_beta: expected number of beta electrons

    Returns:
        valid: True if all configurations are valid
        stats: dictionary with violation statistics
    """
    alpha_counts = configs[:, :n_orbitals].sum(dim=1)
    beta_counts = configs[:, n_orbitals:].sum(dim=1)

    alpha_violations = (alpha_counts != n_alpha).sum().item()
    beta_violations = (beta_counts != n_beta).sum().item()

    valid = (alpha_violations == 0) and (beta_violations == 0)

    stats = {
        'n_configs': len(configs),
        'alpha_violations': alpha_violations,
        'beta_violations': beta_violations,
        'alpha_counts_mean': alpha_counts.float().mean().item(),
        'alpha_counts_std': alpha_counts.float().std().item(),
        'beta_counts_mean': beta_counts.float().mean().item(),
        'beta_counts_std': beta_counts.float().std().item(),
    }

    return valid, stats
