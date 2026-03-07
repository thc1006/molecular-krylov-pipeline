"""Shared pytest fixtures for molecular Hamiltonian tests."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _try_create_hamiltonian(factory_func, **kwargs):
    """Try to create a molecular Hamiltonian, skip if PySCF unavailable."""
    try:
        return factory_func(**kwargs)
    except ImportError:
        pytest.skip("PySCF not available")


@pytest.fixture(scope="session")
def h2_hamiltonian():
    from hamiltonians.molecular import create_h2_hamiltonian
    return _try_create_hamiltonian(create_h2_hamiltonian)


@pytest.fixture(scope="session")
def lih_hamiltonian():
    from hamiltonians.molecular import create_lih_hamiltonian
    return _try_create_hamiltonian(create_lih_hamiltonian)


@pytest.fixture(scope="session")
def beh2_hamiltonian():
    from hamiltonians.molecular import create_beh2_hamiltonian
    return _try_create_hamiltonian(create_beh2_hamiltonian)
