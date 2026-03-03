"""
Flow-Guided Krylov Quantum Diagonalization

A novel pipeline combining Normalizing Flow-Assisted Neural Quantum States (NF-NQS)
with Sample-Based Krylov Quantum Diagonalization (SKQD) for accurate ground state
energy computation of molecular systems.

Modules:
    - nqs: Neural Quantum State implementations
    - flows: Normalizing Flow models for importance sampling
    - hamiltonians: Hamiltonian construction (spin systems and molecular)
    - krylov: Sample-based Krylov quantum diagonalization
    - postprocessing: Projected Hamiltonian construction and eigensolver
    - pipeline: End-to-end integration
"""

__version__ = "0.1.0"

from .pipeline import FlowGuidedKrylovPipeline

__all__ = [
    "FlowGuidedKrylovPipeline",
    "__version__",
]
