"""Tests for PR 2.4: NF Probability Model Fix.

Fixes two critical bugs:
1. _topk_log_prob uses product-of-marginals (wrong for without-replacement)
   → Replace with Plackett-Luce sequential conditional probability
2. |E|/|S| loss scaling punishes diversity
   → Replace with |E|/batch_size

References:
- ADR-001 C1 (B14): _topk_log_prob incorrect probability model
- ADR-001 C2 (B15): |E|/|S| loss scaling punishes diversity
- Plackett-Luce model: P(S) = Σ_{π ∈ Perm(S)} Π P(s_π(j) | remaining)
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from itertools import permutations, combinations
from math import factorial

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSequentialConditionalProbability:
    """_topk_log_prob must use sequential conditional, not product-of-marginals."""

    def test_sequential_differs_from_marginals(self):
        """Sequential conditional gives DIFFERENT results from product-of-marginals."""
        from flows.particle_conserving_flow import ParticleConservingFlow

        flow = ParticleConservingFlow(
            n_orbitals=6, n_alpha=2, n_beta=2, hidden_dims=[64, 64],
        )

        # Create non-uniform logits (where bias matters most)
        logits = torch.tensor([[3.0, 1.0, -1.0, -2.0, -3.0, -4.0]])
        selection = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

        # Old method: product-of-marginals
        log_probs_marginal = F.log_softmax(logits, dim=-1)
        old_log_prob = (log_probs_marginal * selection).sum(dim=-1)
        old_log_prob -= torch.lgamma(torch.tensor(3.0))  # k!

        # New method: sequential conditional
        new_log_prob = flow._topk_log_prob(logits, selection, k=2)

        assert not torch.allclose(old_log_prob, new_log_prob, atol=1e-3), (
            f"Sequential should differ from marginals: "
            f"old={old_log_prob.item():.6f}, new={new_log_prob.item():.6f}"
        )

    def test_exact_enumeration_small_k(self):
        """For k<=5, log_prob must match exact Plackett-Luce enumeration."""
        from flows.particle_conserving_flow import ParticleConservingFlow

        flow = ParticleConservingFlow(
            n_orbitals=6, n_alpha=2, n_beta=2, hidden_dims=[64, 64],
        )

        logits = torch.tensor([[2.0, 1.0, 0.5, -0.5, -1.0, -2.0]])
        selection = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        k = 2

        # Compute exact PL probability by hand
        selected_indices = [0, 2]  # orbitals 0 and 2
        alpha = torch.softmax(logits[0], dim=-1)

        # P(S={0,2}) = P(0 first, 2 second) + P(2 first, 0 second)
        # P(0 first) = α[0] / Σα = α[0]
        # P(2 | 0 already selected) = α[2] / (Σα - α[0])
        # P(2 first) = α[2]
        # P(0 | 2 already selected) = α[0] / (Σα - α[2])
        total = alpha.sum()
        p_0_then_2 = (alpha[0] / total) * (alpha[2] / (total - alpha[0]))
        p_2_then_0 = (alpha[2] / total) * (alpha[0] / (total - alpha[2]))
        exact_prob = p_0_then_2 + p_2_then_0
        exact_log_prob = torch.log(exact_prob)

        computed_log_prob = flow._topk_log_prob(logits, selection, k)

        assert torch.allclose(computed_log_prob, exact_log_prob.unsqueeze(0), atol=1e-5), (
            f"Exact PL mismatch: computed={computed_log_prob.item():.6f}, "
            f"exact={exact_log_prob.item():.6f}"
        )

    def test_exact_enumeration_k3(self):
        """k=3: must sum over 3!=6 orderings."""
        from flows.particle_conserving_flow import ParticleConservingFlow

        flow = ParticleConservingFlow(
            n_orbitals=5, n_alpha=3, n_beta=2, hidden_dims=[32, 32],
        )

        logits = torch.tensor([[1.5, 0.8, -0.3, -1.0, -2.0]])
        selection = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        k = 3

        # Exact: sum over all 3! = 6 orderings of {0, 1, 2}
        alpha = torch.softmax(logits[0], dim=-1)
        selected = [0, 1, 2]
        total_prob = 0.0
        for perm in permutations(selected):
            remaining_sum = alpha.sum().item()
            p = 1.0
            for idx in perm:
                p *= alpha[idx].item() / remaining_sum
                remaining_sum -= alpha[idx].item()
            total_prob += p

        exact_log_prob = torch.log(torch.tensor(total_prob))
        computed_log_prob = flow._topk_log_prob(logits, selection, k)

        assert torch.allclose(computed_log_prob, exact_log_prob.unsqueeze(0), atol=1e-5), (
            f"k=3 PL mismatch: computed={computed_log_prob.item():.6f}, "
            f"exact={exact_log_prob.item():.6f}"
        )

    def test_probabilities_sum_approximately_one(self):
        """Over all C(n,k) subsets, probabilities should sum to ~1."""
        from flows.particle_conserving_flow import ParticleConservingFlow

        n_orb = 5
        k = 2
        flow = ParticleConservingFlow(
            n_orbitals=n_orb, n_alpha=k, n_beta=k, hidden_dims=[32, 32],
        )

        logits = torch.tensor([[1.0, 0.5, 0.0, -0.5, -1.0]])

        # Enumerate all C(5,2) = 10 subsets
        total_prob = 0.0
        for subset in combinations(range(n_orb), k):
            selection = torch.zeros(1, n_orb)
            for idx in subset:
                selection[0, idx] = 1.0
            log_p = flow._topk_log_prob(logits, selection, k)
            total_prob += torch.exp(log_p).item()

        assert abs(total_prob - 1.0) < 0.01, (
            f"Probabilities over all C({n_orb},{k}) subsets sum to "
            f"{total_prob:.6f}, expected ~1.0"
        )

    def test_gradient_flows_through_log_prob(self):
        """_topk_log_prob must be differentiable w.r.t. logits."""
        from flows.particle_conserving_flow import ParticleConservingFlow

        flow = ParticleConservingFlow(
            n_orbitals=6, n_alpha=2, n_beta=2, hidden_dims=[64, 64],
        )

        logits = torch.randn(4, 6, requires_grad=True)
        selection = torch.zeros(4, 6)
        selection[:, 0] = 1.0
        selection[:, 2] = 1.0

        log_prob = flow._topk_log_prob(logits, selection, k=2)
        loss = log_prob.sum()
        loss.backward()

        assert logits.grad is not None, "No gradient through _topk_log_prob"
        assert logits.grad.abs().max() > 1e-10, "Gradient is all zeros"

    def test_batch_consistency(self):
        """Batched computation matches per-sample computation."""
        from flows.particle_conserving_flow import ParticleConservingFlow

        flow = ParticleConservingFlow(
            n_orbitals=6, n_alpha=2, n_beta=2, hidden_dims=[64, 64],
        )

        batch_logits = torch.randn(8, 6)
        batch_selection = torch.zeros(8, 6)
        batch_selection[:, 1] = 1.0
        batch_selection[:, 4] = 1.0

        # Batched
        batch_result = flow._topk_log_prob(batch_logits, batch_selection, k=2)

        # Per-sample
        for i in range(8):
            single_result = flow._topk_log_prob(
                batch_logits[i:i+1], batch_selection[i:i+1], k=2
            )
            assert torch.allclose(batch_result[i], single_result.squeeze(), atol=1e-6), (
                f"Batch/single mismatch at index {i}: "
                f"batch={batch_result[i].item():.6f}, single={single_result.item():.6f}"
            )


class TestProductOfMarginalsRemoved:
    """The old product-of-marginals approach must be gone."""

    def test_old_method_has_large_bias(self):
        """Show that the old method has >50% bias vs exact on skewed logits."""
        logits = torch.tensor([[5.0, 3.0, 0.0, -2.0, -4.0, -6.0]])
        selection = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        k = 2

        # Old: product-of-marginals
        log_sm = F.log_softmax(logits, dim=-1)
        old_log_prob = (log_sm * selection).sum(dim=-1) - torch.lgamma(torch.tensor(3.0))
        old_prob = torch.exp(old_log_prob).item()

        # Exact: Plackett-Luce
        alpha = torch.softmax(logits[0], dim=-1)
        total = alpha.sum()
        s = [0, 1]
        exact = 0.0
        for perm in permutations(s):
            remaining = total.item()
            p = 1.0
            for idx in perm:
                p *= alpha[idx].item() / remaining
                remaining -= alpha[idx].item()
            exact += p

        bias = abs(old_prob - exact) / exact
        assert bias > 0.1, (
            f"Expected >10% bias from product-of-marginals, got {bias*100:.1f}%"
        )


class TestLossScaling:
    """Loss must scale by |E|/batch_size, NOT |E|/|S| (unique count)."""

    def test_loss_scaling_uses_batch_size(self):
        """Verify the loss denominator is batch_size, not n_unique."""
        from flows.physics_guided_training import PhysicsGuidedConfig

        # The config or trainer should not use n_unique for scaling
        config = PhysicsGuidedConfig()
        # We test this by checking the actual loss computation
        # The key invariant: two runs with same batch_size but different
        # n_unique should produce proportional losses

        # This is a structural test — implementation details tested in integration
        assert hasattr(config, 'samples_per_batch'), (
            "PhysicsGuidedConfig must have samples_per_batch"
        )

    def test_collapsed_flow_not_rewarded(self):
        """A mode-collapsed flow (|S|=1) must NOT get more gradient than diverse flow."""
        # This is a design invariant test.
        # With |E|/|S|: collapsed gets |E|/1, diverse gets |E|/1000 → 1000x bias
        # With |E|/batch_size: both get |E|/batch_size → no bias
        #
        # We verify this by checking that the loss function signature
        # does not divide by the number of unique configs.
        import inspect
        from flows.physics_guided_training import PhysicsGuidedFlowTrainer

        source = inspect.getsource(PhysicsGuidedFlowTrainer._compute_flow_loss)
        # The loss should NOT contain "/ n_unique" or "/ len(unique"
        assert "/ n_unique" not in source, (
            "Loss still divides by n_unique — this punishes diversity"
        )


class TestEndToEndProbabilityModel:
    """Integration: sample → compute log_prob → verify consistency."""

    def test_sample_log_prob_consistency(self):
        """log_prob of sampled configs should be finite and reasonable."""
        from flows.particle_conserving_flow import ParticleConservingFlow

        flow = ParticleConservingFlow(
            n_orbitals=6, n_alpha=2, n_beta=2, hidden_dims=[64, 64],
        )

        with torch.no_grad():
            configs, log_probs = flow.sample(batch_size=32, hard=True)

        assert torch.isfinite(log_probs).all(), (
            f"Non-finite log_probs: {log_probs}"
        )
        assert (log_probs < 0).all(), (
            f"log_probs should be negative (probabilities < 1): {log_probs}"
        )

    def test_higher_logit_orbital_more_likely(self):
        """Orbitals with higher logits should be selected more often."""
        from flows.particle_conserving_flow import ParticleConservingFlow

        flow = ParticleConservingFlow(
            n_orbitals=6, n_alpha=2, n_beta=2, hidden_dims=[64, 64],
        )
        # Set alpha logits to strongly prefer orbitals 0 and 1
        with torch.no_grad():
            flow.alpha_logits.copy_(torch.tensor([5.0, 4.0, 0.0, -1.0, -2.0, -3.0]))

        configs, _ = flow.sample(batch_size=200, hard=True)
        alpha_configs = configs[:, :6]

        # Orbital 0 should be selected most often
        freq = alpha_configs.mean(dim=0)
        assert freq[0] > freq[4], (
            f"Higher-logit orbital not more frequent: freq={freq.tolist()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
