"""Tests for Neural Quantum State implementations."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nqs.dense import DenseNQS, SignedDenseNQS
from nqs.complex_nqs import ComplexNQS, RBMQuantumState


class TestDenseNQS:
    """Test cases for Dense NQS."""

    def test_construction(self):
        """Test basic construction."""
        nqs = DenseNQS(num_sites=4, hidden_dims=[64, 64])
        assert nqs.num_sites == 4

    def test_forward_shape(self):
        """Test output shapes."""
        nqs = DenseNQS(num_sites=4, hidden_dims=[64, 64])

        configs = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])

        log_amp = nqs.log_amplitude(configs)
        assert log_amp.shape == (2,)

        phase = nqs.phase(configs)
        assert phase.shape == (2,)

    def test_psi_normalized(self):
        """Test wavefunction normalization."""
        nqs = DenseNQS(num_sites=3, hidden_dims=[32, 32])

        # All configurations for 3 qubits
        configs = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ])

        probs = nqs.probability(configs)

        # Should sum to some positive value (unnormalized)
        assert probs.sum() > 0

        # Normalized probabilities should sum to 1
        probs_norm = probs / probs.sum()
        assert abs(probs_norm.sum() - 1.0) < 1e-6

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        nqs = DenseNQS(num_sites=4, hidden_dims=[32, 32])

        configs = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        log_amp = nqs.log_amplitude(configs)

        # Backprop
        log_amp.sum().backward()

        # Check gradients exist
        for param in nqs.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestSignedDenseNQS:
    """Test cases for Signed Dense NQS."""

    def test_sign_output(self):
        """Test that signs are in {-1, +1}."""
        nqs = SignedDenseNQS(num_sites=4, hidden_dims=[64, 64])

        configs = torch.tensor([
            [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1],
        ])

        signs = nqs.get_sign(configs)

        # Signs should be approximately ±1
        assert torch.all(torch.abs(torch.abs(signs) - 1.0) < 0.1)


class TestComplexNQS:
    """Test cases for Complex NQS."""

    def test_construction(self):
        """Test construction."""
        nqs = ComplexNQS(num_sites=4, hidden_dims=[64, 64])
        assert nqs.num_sites == 4
        assert nqs.complex_output is True

    def test_complex_output(self):
        """Test that output is complex."""
        nqs = ComplexNQS(num_sites=4, hidden_dims=[64, 64])

        configs = torch.tensor([[0, 1, 0, 1]])
        psi = nqs.psi(configs)

        # Should be complex
        assert psi.is_complex()


class TestRBMQuantumState:
    """Test cases for RBM-based NQS."""

    def test_construction(self):
        """Test construction."""
        rbm = RBMQuantumState(num_sites=4, num_hidden=8)
        assert rbm.num_sites == 4
        assert rbm.num_hidden == 8

    def test_output_shape(self):
        """Test output shapes."""
        rbm = RBMQuantumState(num_sites=4, num_hidden=8)

        configs = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])

        log_amp = rbm.log_amplitude(configs)
        assert log_amp.shape == (2,)

    def test_complex_rbm(self):
        """Test complex RBM."""
        rbm = RBMQuantumState(num_sites=4, num_hidden=8, complex_weights=True)

        configs = torch.tensor([[0, 1, 0, 1]])

        log_amp = rbm.log_amplitude(configs)
        phase = rbm.phase(configs)

        assert log_amp.shape == (1,)
        assert phase.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
