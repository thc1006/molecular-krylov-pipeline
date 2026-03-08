"""
Memory estimation and logging for large matrix allocations.

Provides pre-allocation logging so post-mortem analysis can identify
which allocation caused OOM. All output goes through Python logging
(logger name: 'molecular_krylov.memory').

Usage:
    from utils.memory_logger import log_allocation, log_system_memory, check_available_memory

    # Before a large allocation:
    log_allocation("fci_energy", n_configs=63504, dtype="float64", layout="dense")

    # Check if system has enough memory:
    if not check_available_memory(required_gb=30.0):
        raise MemoryError("Insufficient memory")

    # Log current system memory state:
    log_system_memory("before Krylov expansion")
"""

import logging
import os

logger = logging.getLogger("molecular_krylov.memory")

# Only add handler if none exists (avoid duplicate handlers on re-import)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[MEM] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _bytes_to_human(n_bytes: float) -> str:
    """Convert bytes to human-readable string."""
    if n_bytes >= 1e9:
        return f"{n_bytes / 1e9:.2f} GB"
    elif n_bytes >= 1e6:
        return f"{n_bytes / 1e6:.1f} MB"
    else:
        return f"{n_bytes / 1e3:.0f} KB"


def estimate_dense_bytes(n: int, dtype: str = "float64") -> int:
    """Estimate memory for dense n×n matrix."""
    dtype_sizes = {
        "float32": 4, "float64": 8,
        "complex64": 8, "complex128": 16,
    }
    elem_size = dtype_sizes.get(dtype, 8)
    return n * n * elem_size


def estimate_sparse_bytes(n: int, nnz_per_row: int = 200) -> int:
    """Estimate memory for sparse CSR matrix (indices + values)."""
    # CSR: data (8 bytes * nnz) + indices (4 bytes * nnz) + indptr (4 bytes * (n+1))
    nnz = n * nnz_per_row
    return nnz * 12 + (n + 1) * 4


def log_allocation(
    caller: str,
    n_configs: int,
    dtype: str = "float64",
    layout: str = "dense",
    extra: str = "",
):
    """
    Log a pending large matrix allocation.

    Args:
        caller: Function name (e.g. "fci_energy", "compute_ground_state_energy")
        n_configs: Number of configurations (matrix dimension)
        dtype: Data type string
        layout: "dense" or "sparse"
        extra: Additional context
    """
    if layout == "dense":
        est_bytes = estimate_dense_bytes(n_configs, dtype)
    else:
        est_bytes = estimate_sparse_bytes(n_configs)

    msg = f"{caller}: {layout} {n_configs}×{n_configs} ({dtype}) → {_bytes_to_human(est_bytes)}"
    if extra:
        msg += f" [{extra}]"

    logger.info(msg)


def get_available_memory_gb() -> float:
    """
    Get available system memory in GB from /proc/meminfo.

    On DGX Spark UMA, this is the ONLY reliable memory metric —
    cudaMemGetInfo reports GPU-specific values that don't reflect
    the shared 128GB pool correctly.

    Returns:
        Available memory in GB, or -1.0 if unavailable.
    """
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # Value is in kB
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (OSError, ValueError):
        pass
    return -1.0


def check_available_memory(required_gb: float, caller: str = "") -> bool:
    """
    Check if system has enough available memory for an allocation.

    Args:
        required_gb: Required memory in GB
        caller: Caller name for logging

    Returns:
        True if enough memory is available (or if check is unavailable)
    """
    avail = get_available_memory_gb()
    if avail < 0:
        return True  # Can't check, assume OK

    prefix = f"{caller}: " if caller else ""

    if avail < required_gb:
        logger.warning(
            f"{prefix}INSUFFICIENT MEMORY: need {required_gb:.1f} GB, "
            f"available {avail:.1f} GB"
        )
        return False
    else:
        logger.debug(
            f"{prefix}memory OK: need {required_gb:.1f} GB, available {avail:.1f} GB"
        )
        return True


def log_system_memory(context: str = ""):
    """Log current system memory state."""
    avail = get_available_memory_gb()
    if avail < 0:
        return

    msg = f"System memory: {avail:.1f} GB available"
    if context:
        msg = f"{context} — {msg}"

    # Warn if below 10 GB
    if avail < 10.0:
        logger.warning(msg)
    else:
        logger.info(msg)
