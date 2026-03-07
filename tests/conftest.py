"""Shared fixtures for the Flow-Guided Krylov pipeline test suite."""

import os
import pytest
import torch
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# cu130 resolves NVRTC JIT issues on GB10 (sm_121).
# CUDA is now available for all tests by default.

# Set PyTorch to use all available CPU threads
if hasattr(torch, 'set_num_threads'):
    import multiprocessing
    torch.set_num_threads(multiprocessing.cpu_count())


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "molecular: requires PySCF")
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")


@pytest.fixture(scope="session")
def pyscf_available():
    """Check if PySCF is available."""
    try:
        import pyscf
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def h2_hamiltonian(pyscf_available):
    """H2 Hamiltonian (4 qubits, 4 configs)."""
    if not pyscf_available:
        pytest.skip("PySCF not available")
    from hamiltonians.molecular import create_h2_hamiltonian
    return create_h2_hamiltonian(bond_length=0.74, device="cpu")


@pytest.fixture(scope="session")
def lih_hamiltonian(pyscf_available):
    """LiH Hamiltonian (12 qubits, 225 configs)."""
    if not pyscf_available:
        pytest.skip("PySCF not available")
    from hamiltonians.molecular import create_lih_hamiltonian
    return create_lih_hamiltonian(bond_length=1.6, device="cpu")


@pytest.fixture(scope="session")
def h2o_hamiltonian(pyscf_available):
    """H2O Hamiltonian (14 qubits, 441 configs)."""
    if not pyscf_available:
        pytest.skip("PySCF not available")
    from hamiltonians.molecular import create_h2o_hamiltonian
    return create_h2o_hamiltonian(device="cpu")


@pytest.fixture(scope="session")
def beh2_hamiltonian(pyscf_available):
    """BeH2 Hamiltonian (14 qubits, 1225 configs)."""
    if not pyscf_available:
        pytest.skip("PySCF not available")
    from hamiltonians.molecular import create_beh2_hamiltonian
    return create_beh2_hamiltonian(device="cpu")


@pytest.fixture(scope="session")
def nh3_hamiltonian(pyscf_available):
    """NH3 Hamiltonian (16 qubits, 3136 configs)."""
    if not pyscf_available:
        pytest.skip("PySCF not available")
    from hamiltonians.molecular import create_nh3_hamiltonian
    return create_nh3_hamiltonian(device="cpu")


@pytest.fixture(scope="session")
def ch4_hamiltonian(pyscf_available):
    """CH4 Hamiltonian (18 qubits, 15876 configs)."""
    if not pyscf_available:
        pytest.skip("PySCF not available")
    from hamiltonians.molecular import create_ch4_hamiltonian
    return create_ch4_hamiltonian(device="cpu")


@pytest.fixture(scope="session")
def n2_hamiltonian(pyscf_available):
    """N2 Hamiltonian (20 qubits, 14400 configs)."""
    if not pyscf_available:
        pytest.skip("PySCF not available")
    from hamiltonians.molecular import create_n2_hamiltonian
    return create_n2_hamiltonian(bond_length=1.10, device="cpu")


# GPU fixtures — same molecules but on CUDA
@pytest.fixture(scope="session")
def gpu_device():
    """Return 'cuda' if available, skip otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"


@pytest.fixture(scope="session")
def h2_hamiltonian_gpu(pyscf_available, gpu_device):
    """H2 on GPU."""
    if not pyscf_available:
        pytest.skip("PySCF not available")
    from hamiltonians.molecular import create_h2_hamiltonian
    return create_h2_hamiltonian(bond_length=0.74, device=gpu_device)


@pytest.fixture(scope="session")
def lih_hamiltonian_gpu(pyscf_available, gpu_device):
    """LiH on GPU."""
    if not pyscf_available:
        pytest.skip("PySCF not available")
    from hamiltonians.molecular import create_lih_hamiltonian
    return create_lih_hamiltonian(bond_length=1.6, device=gpu_device)


@pytest.fixture(scope="session")
def beh2_hamiltonian_gpu(pyscf_available, gpu_device):
    """BeH2 on GPU."""
    if not pyscf_available:
        pytest.skip("PySCF not available")
    from hamiltonians.molecular import create_beh2_hamiltonian
    return create_beh2_hamiltonian(device=gpu_device)
