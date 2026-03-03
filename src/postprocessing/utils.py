"""Utility functions for SKQD postprocessing."""

import numpy as np
from typing import List, Dict, Union


def bitstring_to_int(bitstring: str) -> int:
    """Convert bitstring to integer."""
    return int(bitstring, 2)


def int_to_bitstring(value: int, num_bits: int) -> str:
    """Convert integer to bitstring with specified number of bits."""
    return format(value, f"0{num_bits}b")


def get_basis_states_as_array(
    measurement_results: Dict[str, int],
    num_qubits: int,
) -> np.ndarray:
    """
    Convert measurement results to array of basis state integers.

    Args:
        measurement_results: Dictionary mapping bitstrings to counts
        num_qubits: Number of qubits

    Returns:
        Array of unique basis states as integers
    """
    states = []
    for bitstring in measurement_results.keys():
        states.append(bitstring_to_int(bitstring))

    return np.array(sorted(set(states)), dtype=np.int64)


def calculate_cumulative_results(
    all_measurement_results: List[Dict[str, int]],
) -> List[Dict[str, int]]:
    """
    Calculate cumulative measurement results across Krylov steps.

    For step k, the cumulative results include all unique bitstrings
    from steps 0, 1, ..., k with their total counts.

    Args:
        all_measurement_results: List of measurement dictionaries per step

    Returns:
        List of cumulative measurement dictionaries
    """
    cumulative = []
    all_counts: Dict[str, int] = {}

    for step_results in all_measurement_results:
        # Merge counts
        for bitstring, count in step_results.items():
            all_counts[bitstring] = all_counts.get(bitstring, 0) + count

        # Store snapshot
        cumulative.append(dict(all_counts))

    return cumulative


def filter_high_probability_states(
    measurement_results: Dict[str, int],
    threshold: float = 0.0,
    max_states: int = None,
) -> Dict[str, int]:
    """
    Filter measurement results to keep high-probability states.

    Args:
        measurement_results: Dictionary mapping bitstrings to counts
        threshold: Minimum probability threshold
        max_states: Maximum number of states to keep

    Returns:
        Filtered measurement dictionary
    """
    total_counts = sum(measurement_results.values())

    # Compute probabilities
    probs = {
        bs: count / total_counts
        for bs, count in measurement_results.items()
    }

    # Filter by threshold
    filtered = {
        bs: count
        for bs, count in measurement_results.items()
        if probs[bs] >= threshold
    }

    # Limit number of states
    if max_states is not None and len(filtered) > max_states:
        sorted_states = sorted(
            filtered.items(), key=lambda x: x[1], reverse=True
        )
        filtered = dict(sorted_states[:max_states])

    return filtered


def compute_basis_overlap(
    basis1: np.ndarray,
    basis2: np.ndarray,
) -> float:
    """
    Compute overlap between two basis sets.

    Returns the fraction of states in basis1 that are also in basis2.

    Args:
        basis1: First basis as array of state integers
        basis2: Second basis as array of state integers

    Returns:
        Overlap fraction (0 to 1)
    """
    set1 = set(basis1.tolist())
    set2 = set(basis2.tolist())

    intersection = len(set1 & set2)
    return intersection / len(set1) if len(set1) > 0 else 0.0


def estimate_ground_state_sparsity(
    ground_state: np.ndarray,
    threshold: float = 1e-6,
) -> Dict[str, float]:
    """
    Estimate sparsity metrics of a ground state vector.

    Args:
        ground_state: Ground state wavefunction
        threshold: Amplitude threshold for significant components

    Returns:
        Dictionary with sparsity metrics:
            - n_significant: Number of significant components
            - sparsity_ratio: Fraction of Hilbert space with significant weight
            - concentration: Weight in top 10% of components
    """
    probs = np.abs(ground_state) ** 2
    probs = probs / probs.sum()

    n_significant = np.sum(probs > threshold)
    sparsity_ratio = n_significant / len(ground_state)

    # Weight in top components
    sorted_probs = np.sort(probs)[::-1]
    n_top = max(1, len(sorted_probs) // 10)
    concentration = np.sum(sorted_probs[:n_top])

    return {
        "n_significant": int(n_significant),
        "sparsity_ratio": float(sparsity_ratio),
        "concentration": float(concentration),
        "total_dimension": len(ground_state),
    }


def merge_basis_sets(*bases: np.ndarray) -> np.ndarray:
    """
    Merge multiple basis sets into one unique set.

    Args:
        *bases: Variable number of basis arrays

    Returns:
        Merged array of unique basis states
    """
    all_states = set()
    for basis in bases:
        all_states.update(basis.tolist())

    return np.array(sorted(all_states), dtype=np.int64)
