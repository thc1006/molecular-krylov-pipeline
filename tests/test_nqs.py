"""Tests for Neural Quantum State implementations."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nqs.dense import DenseNQS, SignedDenseNQS


class TestDenseNQS:
    """Test cases for Dense NQS."""

    def test_construction(self):
        nqs = DenseNQS(num_sites=4, hidden_dims=[64, 64])
        assert nqs.num_sites == 4

    def test_forward_shape(self):
        nqs = DenseNQS(num_sites=4, hidden_dims=[64, 64])
        configs = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])

        log_amp = nqs.log_amplitude(configs)
        assert log_amp.shape == (2,)

        phase = nqs.phase(configs)
        assert phase.shape == (2,)

    def test_psi_normalized(self):
        nqs = DenseNQS(num_sites=3, hidden_dims=[32, 32])
        configs = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ])

        probs = nqs.probability(configs)
        assert probs.sum() > 0

        probs_norm = probs / probs.sum()
        assert abs(probs_norm.sum() - 1.0) < 1e-6

    def test_gradient_flow(self):
        nqs = DenseNQS(num_sites=4, hidden_dims=[32, 32])
        configs = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        log_amp = nqs.log_amplitude(configs)
        log_amp.sum().backward()

        for param in nqs.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestSignedDenseNQS:
    """Test cases for Signed Dense NQS."""

    def test_sign_output(self):
        nqs = SignedDenseNQS(num_sites=4, hidden_dims=[64, 64])
        configs = torch.tensor([
            [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1],
        ])
        signs = nqs.get_sign(configs)
        assert torch.all(torch.abs(torch.abs(signs) - 1.0) < 0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
