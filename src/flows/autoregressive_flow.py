"""
Autoregressive Transformer-Based Normalizing Flow for Molecular Configuration Sampling.

This module implements a decoder-only transformer that autoregressively predicts orbital
occupations, replacing the non-autoregressive product-of-marginals architecture in
``ParticleConservingFlow``. The autoregressive factorization

    P(config) = P(o_1) * P(o_2 | o_1) * ... * P(o_n | o_1, ..., o_{n-1})

captures inter-orbital correlations that the non-autoregressive model cannot, which is
critical for strongly correlated systems (stretched bonds, multi-reference wavefunctions)
and for scaling to 40+ qubits where CISD is insufficient.

**State representation**: Each spatial orbital can be in one of 4 states:
    - 0: unoccupied (alpha=0, beta=0)
    - 1: beta-only (alpha=0, beta=1)
    - 2: alpha-only (alpha=1, beta=0)
    - 3: doubly occupied (alpha=1, beta=1)

A BOS (beginning-of-sequence) token with state_id=4 initiates the sequence.

**Particle conservation**: At each autoregressive step, states that would make it
impossible to place all remaining electrons in the remaining orbitals are masked to
-inf, guaranteeing exact particle number conservation without rejection sampling.

**Log probability**: Computed exactly via teacher forcing in a single forward pass,
since the autoregressive factorization gives P(config) = product of conditionals.

References:
    - QiankunNet: Li et al., "QiankunNet: Transformer-Based Autoregressive Neural
      Quantum States for Electronic Structure" (Nature Comms, 2025)
    - Attention Is All You Need: Vaswani et al. (2017), decoder-only variant
    - AB-SND: Autoregressive basis selection for nuclear structure (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# Number of possible orbital states: empty, beta-only, alpha-only, doubly-occupied
NUM_ORBITAL_STATES = 4
# BOS token id (used as start token for autoregressive generation)
BOS_TOKEN_ID = 4
# Total vocabulary size (4 orbital states + BOS)
VOCAB_SIZE = 5


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------


def configs_to_states(configs: torch.Tensor, n_orbitals: int) -> torch.Tensor:
    """Convert binary spin-orbital configs to quaternary spatial-orbital states.

    Parameters
    ----------
    configs : torch.Tensor
        (batch, 2 * n_orbitals) binary tensor.  First ``n_orbitals`` entries are
        alpha occupations, last ``n_orbitals`` are beta occupations.
    n_orbitals : int
        Number of spatial orbitals.

    Returns
    -------
    torch.Tensor
        (batch, n_orbitals) long tensor with values in {0, 1, 2, 3}.
    """
    alpha = configs[:, :n_orbitals]
    beta = configs[:, n_orbitals:]
    return (2 * alpha + beta).long()


def states_to_configs(states: torch.Tensor, n_orbitals: int) -> torch.Tensor:
    """Convert quaternary spatial-orbital states to binary spin-orbital configs.

    Parameters
    ----------
    states : torch.Tensor
        (batch, n_orbitals) tensor with values in {0, 1, 2, 3}.
    n_orbitals : int
        Number of spatial orbitals.

    Returns
    -------
    torch.Tensor
        (batch, 2 * n_orbitals) float tensor (binary 0/1).
    """
    alpha = (states >> 1).float()
    beta = (states & 1).float()
    return torch.cat([alpha, beta], dim=-1)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------


@dataclass
class AutoregressiveConfig:
    """Hyperparameters for the autoregressive transformer.

    Parameters
    ----------
    n_layers : int
        Number of transformer decoder layers.
    n_heads : int
        Number of attention heads.  ``d_model`` must be divisible by ``n_heads``.
    d_model : int
        Hidden dimension throughout the transformer.
    d_ff : int
        Feed-forward network inner dimension.
    dropout : float
        Dropout rate (0.0 for deterministic inference).
    """

    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.0


class TransformerDecoderLayer(nn.Module):
    """Pre-LN transformer decoder layer with causal self-attention.

    Architecture:
        x -> LayerNorm -> MultiHeadAttention (causal) -> + residual
          -> LayerNorm -> FFN -> + residual

    Pre-LN (as opposed to Post-LN) gives more stable gradients and avoids
    the need for a learning-rate warm-up schedule.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        FFN inner dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with causal masked self-attention.

        Parameters
        ----------
        x : torch.Tensor
            (batch, seq_len, d_model)
        attn_mask : torch.Tensor, optional
            (seq_len, seq_len) additive attention mask (``-inf`` for blocked positions).

        Returns
        -------
        torch.Tensor
            (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_mask,
            is_causal=False,
        )
        x = residual + attn_out

        # FFN with pre-norm
        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x


class AutoregressiveTransformer(nn.Module):
    """Decoder-only transformer for autoregressive orbital state prediction.

    The model takes a sequence of orbital states (with BOS prepended) and predicts
    the next orbital state at each position.  A causal mask ensures that position
    *i* only attends to positions <= *i* (autoregressive property).

    At inference time, states are generated one at a time (autoregressive sampling).
    At training time / log_prob evaluation, the entire sequence is processed in
    parallel via teacher forcing.

    Parameters
    ----------
    n_orbitals : int
        Number of spatial orbitals.
    config : AutoregressiveConfig
        Transformer hyperparameters.
    """

    def __init__(self, n_orbitals: int, config: Optional[AutoregressiveConfig] = None):
        super().__init__()
        if config is None:
            config = AutoregressiveConfig()

        self.n_orbitals = n_orbitals
        self.config = config
        d_model = config.d_model

        # Token embedding: 5 tokens (4 orbital states + BOS)
        self.state_embedding = nn.Embedding(VOCAB_SIZE, d_model)

        # Positional embedding: n_orbitals + 1 positions (BOS at position 0)
        self.position_embedding = nn.Embedding(n_orbitals + 1, d_model)

        # Transformer decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # Final layer norm (Pre-LN convention: add a final norm before output head)
        self.final_norm = nn.LayerNorm(d_model)

        # Output head: predict 4-way logits for the next orbital state
        self.output_head = nn.Linear(d_model, NUM_ORBITAL_STATES)

        # Register causal mask as buffer (upper-triangular = -inf)
        seq_len = n_orbitals + 1  # +1 for BOS
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for linear layers, normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: compute 4-way logits at each position.

        Parameters
        ----------
        input_states : torch.Tensor
            (batch, seq_len) long tensor of state ids (0-4).  Typically
            ``[BOS, s_0, s_1, ..., s_{n-2}]`` (length n_orbitals + 1) for teacher
            forcing, or a partial prefix during autoregressive generation.
        mask : torch.Tensor, optional
            (seq_len, seq_len) additive attention mask.  If None, native causal
            masking is used (enables Flash Attention on supported hardware).

        Returns
        -------
        torch.Tensor
            (batch, seq_len, 4) logits for next orbital state.
        """
        batch_size, seq_len = input_states.shape
        device = input_states.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        x = self.state_embedding(input_states) + self.position_embedding(positions)

        # Causal mask (crop if seq_len < full length)
        if mask is None:
            mask = self.causal_mask[:seq_len, :seq_len]

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=mask)

        # Output projection
        x = self.final_norm(x)
        logits = self.output_head(x)  # (batch, seq_len, 4)

        return logits


# ---------------------------------------------------------------------------
# Particle conservation masking
# ---------------------------------------------------------------------------


def compute_validity_mask(
    alpha_remaining: torch.Tensor,
    beta_remaining: torch.Tensor,
    orbitals_after: int,
) -> torch.Tensor:
    """Compute which of the 4 orbital states are valid given remaining electron budgets.

    At autoregressive step *i* (deciding the state for orbital *i*), we must ensure
    that the remaining electrons can still be placed in the remaining orbitals.

    Parameters
    ----------
    alpha_remaining : torch.Tensor
        (batch,) number of alpha electrons still to place.
    beta_remaining : torch.Tensor
        (batch,) number of beta electrons still to place.
    orbitals_after : int
        Number of orbitals that come *after* the current one (``n_orbitals - i - 1``).

    Returns
    -------
    torch.Tensor
        (batch, 4) boolean mask.  True = state is valid.
    """
    batch_size = alpha_remaining.shape[0]
    device = alpha_remaining.device

    mask = torch.zeros(batch_size, NUM_ORBITAL_STATES, dtype=torch.bool, device=device)

    # State 0 (empty): need alpha_rem <= after AND beta_rem <= after
    mask[:, 0] = (alpha_remaining <= orbitals_after) & (beta_remaining <= orbitals_after)

    # State 1 (beta only): need beta_rem > 0,
    #   alpha_rem <= after, (beta_rem - 1) <= after
    mask[:, 1] = (
        (beta_remaining > 0)
        & (alpha_remaining <= orbitals_after)
        & ((beta_remaining - 1) <= orbitals_after)
    )

    # State 2 (alpha only): need alpha_rem > 0,
    #   (alpha_rem - 1) <= after, beta_rem <= after
    mask[:, 2] = (
        (alpha_remaining > 0)
        & ((alpha_remaining - 1) <= orbitals_after)
        & (beta_remaining <= orbitals_after)
    )

    # State 3 (doubly occ): need alpha_rem > 0 AND beta_rem > 0,
    #   (alpha_rem - 1) <= after, (beta_rem - 1) <= after
    mask[:, 3] = (
        (alpha_remaining > 0)
        & (beta_remaining > 0)
        & ((alpha_remaining - 1) <= orbitals_after)
        & ((beta_remaining - 1) <= orbitals_after)
    )

    return mask


# Table: how many alpha / beta electrons each state contributes
# state 0 -> (0, 0), state 1 -> (0, 1), state 2 -> (1, 0), state 3 -> (1, 1)
_STATE_ALPHA = torch.tensor([0, 0, 1, 1], dtype=torch.long)
_STATE_BETA = torch.tensor([0, 1, 0, 1], dtype=torch.long)


# ---------------------------------------------------------------------------
# Drop-in sampler
# ---------------------------------------------------------------------------


class AutoregressiveFlowSampler(nn.Module):
    """Drop-in replacement for ``ParticleConservingFlowSampler`` using an autoregressive
    transformer for orbital state prediction.

    This sampler matches the exact public interface of ``ParticleConservingFlowSampler``
    so that it can be used interchangeably in the pipeline and physics-guided trainer.

    Key differences from the non-autoregressive ``ParticleConservingFlowSampler``:

    1. **Full inter-orbital correlations**: Each orbital's state is conditioned on all
       preceding orbitals via transformer self-attention, not independent marginals.
    2. **Exact log probabilities**: ``log_prob()`` computes the exact log P via teacher
       forcing (single forward pass), no Plackett-Luce permutation sums needed.
    3. **Guaranteed particle conservation**: Validity masking at each step ensures
       exactly ``n_alpha`` alpha and ``n_beta`` beta electrons, no rejection sampling.

    Parameters
    ----------
    num_sites : int
        Total number of spin-orbitals (= 2 * n_orbitals).
    n_alpha : int
        Number of alpha electrons.
    n_beta : int
        Number of beta electrons.
    hidden_dims : list, optional
        Ignored (kept for API compatibility with ``ParticleConservingFlowSampler``).
        Use ``transformer_config`` for architecture control.
    temperature : float
        Sampling temperature.  Logits are divided by temperature before softmax.
        Higher = more exploration, lower = more greedy.
    transformer_config : AutoregressiveConfig, optional
        Transformer hyperparameters.  If None, auto-scaled based on system size.
    """

    def __init__(
        self,
        num_sites: int,
        n_alpha: int,
        n_beta: int,
        hidden_dims: Optional[list] = None,
        temperature: float = 1.0,
        transformer_config: Optional[AutoregressiveConfig] = None,
    ):
        super().__init__()
        assert num_sites % 2 == 0, "num_sites must be even (2 * n_orbitals)"

        self.n_orbitals = num_sites // 2
        self.num_sites = num_sites
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.temperature = temperature

        # Auto-scale transformer config if not provided
        if transformer_config is None:
            transformer_config = self._auto_config(self.n_orbitals)

        self.transformer = AutoregressiveTransformer(
            n_orbitals=self.n_orbitals,
            config=transformer_config,
        )

        # Register electron count tables as buffers for device-agnostic access
        self.register_buffer("_state_alpha", _STATE_ALPHA.clone())
        self.register_buffer("_state_beta", _STATE_BETA.clone())

    @staticmethod
    def _auto_config(n_orbitals: int) -> AutoregressiveConfig:
        """Choose transformer hyperparameters based on system size.

        Parameters
        ----------
        n_orbitals : int
            Number of spatial orbitals.

        Returns
        -------
        AutoregressiveConfig
        """
        if n_orbitals <= 8:
            return AutoregressiveConfig(
                n_layers=3,
                n_heads=4,
                d_model=64,
                d_ff=256,
                dropout=0.0,
            )
        elif n_orbitals <= 15:
            return AutoregressiveConfig(
                n_layers=4,
                n_heads=4,
                d_model=128,
                d_ff=512,
                dropout=0.0,
            )
        elif n_orbitals <= 25:
            return AutoregressiveConfig(
                n_layers=6,
                n_heads=8,
                d_model=256,
                d_ff=1024,
                dropout=0.0,
            )
        else:
            return AutoregressiveConfig(
                n_layers=8,
                n_heads=8,
                d_model=256,
                d_ff=1024,
                dropout=0.0,
            )

    # ------------------------------------------------------------------
    # Autoregressive sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_autoregressive(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate configurations autoregressively with particle conservation masking.

        Parameters
        ----------
        n_samples : int
            Number of configurations to generate.

        Returns
        -------
        states : torch.Tensor
            (n_samples, n_orbitals) sampled orbital states (long, values 0-3).
        log_probs : torch.Tensor
            (n_samples,) log P of each sampled configuration.
        """
        device = next(self.parameters()).device
        n_orb = self.n_orbitals

        # Track remaining electron budgets
        alpha_rem = torch.full((n_samples,), self.n_alpha, dtype=torch.long, device=device)
        beta_rem = torch.full((n_samples,), self.n_beta, dtype=torch.long, device=device)

        # Pre-allocate full sequence buffer (avoids O(n^2) torch.cat reallocation)
        generated = torch.full(
            (n_samples, n_orb + 1), BOS_TOKEN_ID, dtype=torch.long, device=device
        )
        log_probs = torch.zeros(n_samples, device=device)

        state_alpha = self._state_alpha  # (4,)
        state_beta = self._state_beta  # (4,)

        for i in range(n_orb):
            # Forward pass on current prefix (positions 0..i inclusive)
            prefix = generated[:, : i + 1]
            logits_all = self.transformer(prefix)  # (batch, i+1, 4)
            # Logits for position i (the last position in the current sequence)
            logits_i = logits_all[:, -1, :]  # (batch, 4)

            # Apply temperature
            logits_i = logits_i / self.temperature

            # Compute validity mask
            orbitals_after = n_orb - i - 1
            valid_mask = compute_validity_mask(alpha_rem, beta_rem, orbitals_after)

            # Mask invalid states to -inf
            logits_i = logits_i.masked_fill(~valid_mask, float("-inf"))

            # Sample from categorical distribution
            log_probs_i = F.log_softmax(logits_i, dim=-1)
            probs_i = log_probs_i.exp()
            sampled_state = torch.multinomial(probs_i, num_samples=1).squeeze(-1)  # (batch,)

            # Accumulate log probability
            log_probs = log_probs + log_probs_i.gather(1, sampled_state.unsqueeze(1)).squeeze(1)

            # Update electron budgets
            alpha_rem = alpha_rem - state_alpha[sampled_state]
            beta_rem = beta_rem - state_beta[sampled_state]

            # Write sampled state into pre-allocated buffer
            generated[:, i + 1] = sampled_state

        # Extract the orbital states (drop BOS at position 0)
        states = generated[:, 1:]  # (n_samples, n_orbitals)
        return states, log_probs

    # ------------------------------------------------------------------
    # Log probability via teacher forcing
    # ------------------------------------------------------------------

    def _log_prob_from_states(self, states: torch.Tensor) -> torch.Tensor:
        """Compute exact log P(config) via teacher forcing.

        Parameters
        ----------
        states : torch.Tensor
            (batch, n_orbitals) long tensor of orbital states (0-3).

        Returns
        -------
        torch.Tensor
            (batch,) log probabilities.
        """
        batch_size, n_orb = states.shape
        device = states.device

        # Prepend BOS token: input = [BOS, s_0, s_1, ..., s_{n-2}]
        bos = torch.full((batch_size, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
        input_seq = torch.cat([bos, states[:, :-1]], dim=1)  # (batch, n_orbitals)

        # Full forward pass (teacher forcing) — pass None to use auto-cropped causal mask
        logits = self.transformer(input_seq)  # (batch, n_orb, 4)

        # Apply temperature
        logits = logits / self.temperature

        # Compute validity masks and apply them at each position
        alpha_rem = torch.full((batch_size,), self.n_alpha, dtype=torch.long, device=device)
        beta_rem = torch.full((batch_size,), self.n_beta, dtype=torch.long, device=device)

        state_alpha = self._state_alpha
        state_beta = self._state_beta

        total_log_prob = torch.zeros(batch_size, device=device)

        for i in range(n_orb):
            orbitals_after = n_orb - i - 1
            valid_mask = compute_validity_mask(alpha_rem, beta_rem, orbitals_after)

            logits_i = logits[:, i, :]  # (batch, 4)
            logits_i = logits_i.masked_fill(~valid_mask, float("-inf"))

            # Guard: if any sample has all states masked, the config has invalid
            # electron count and log_softmax would produce NaN.
            if not valid_mask.any(dim=-1).all():
                raise ValueError(
                    "Invalid config in log_prob: electron count mismatch. "
                    "All 4 states masked at position {i}, likely wrong n_alpha/n_beta."
                )

            log_probs_i = F.log_softmax(logits_i, dim=-1)
            actual_state = states[:, i]  # (batch,)
            total_log_prob = total_log_prob + log_probs_i.gather(
                1, actual_state.unsqueeze(1)
            ).squeeze(1)

            # Update budgets based on actual states (teacher forcing)
            alpha_rem = alpha_rem - state_alpha[actual_state]
            beta_rem = beta_rem - state_beta[actual_state]

        return total_log_prob

    # ------------------------------------------------------------------
    # Public API (matches ParticleConservingFlowSampler)
    # ------------------------------------------------------------------

    def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample configurations from the autoregressive flow.

        Parameters
        ----------
        n_samples : int
            Number of configurations to sample.

        Returns
        -------
        log_probs : torch.Tensor
            (n_samples,) log probabilities of each sample.
        unique_configs : torch.Tensor
            (n_unique, num_sites) unique configurations in binary spin-orbital format.
        """
        states, log_probs = self._sample_autoregressive(n_samples)
        configs = states_to_configs(states, self.n_orbitals)  # (n_samples, 2*n_orb)
        unique_configs = torch.unique(configs.long(), dim=0)
        return log_probs, unique_configs

    def sample_with_probs(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample and return all configs with their log probabilities.

        Parameters
        ----------
        n_samples : int
            Number of configurations to sample.

        Returns
        -------
        configs : torch.Tensor
            (n_samples, num_sites) all sampled configurations (binary, long).
        log_probs : torch.Tensor
            (n_samples,) log probabilities.
        unique_configs : torch.Tensor
            (n_unique, num_sites) unique configurations (binary, long).
        """
        states, log_probs = self._sample_autoregressive(n_samples)
        configs = states_to_configs(states, self.n_orbitals)
        configs_long = configs.long()
        unique_configs = torch.unique(configs_long, dim=0)
        return configs_long, log_probs, unique_configs

    def log_prob(self, configs: torch.Tensor) -> torch.Tensor:
        """Compute exact log probability of given configurations via teacher forcing.

        Parameters
        ----------
        configs : torch.Tensor
            (batch, 2 * n_orbitals) binary configurations.

        Returns
        -------
        torch.Tensor
            (batch,) log probabilities.
        """
        states = configs_to_states(configs, self.n_orbitals)
        return self._log_prob_from_states(states)

    def estimate_discrete_prob(self, configs: torch.Tensor) -> torch.Tensor:
        """Estimate probability of discrete configurations.

        Returns ``exp(log_prob(configs))``.

        Parameters
        ----------
        configs : torch.Tensor
            (batch, 2 * n_orbitals) binary configurations.

        Returns
        -------
        torch.Tensor
            (batch,) probabilities.
        """
        return torch.exp(self.log_prob(configs))

    def set_temperature(self, temperature: float):
        """Set the sampling temperature.

        Parameters
        ----------
        temperature : float
            New temperature value.  Must be positive.
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (same as ``sample()``).

        Parameters
        ----------
        n_samples : int
            Number of configurations to sample.

        Returns
        -------
        log_probs : torch.Tensor
            (n_samples,) log probabilities.
        unique_configs : torch.Tensor
            (n_unique, num_sites) unique configurations.
        """
        return self.sample(n_samples)
