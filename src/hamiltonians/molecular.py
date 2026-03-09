"""
Molecular Hamiltonians using PySCF.

Optimizations included:
- Fully vectorized diagonal element computation (no Python loops)
- Precomputed Coulomb/Exchange tensors for GPU acceleration
- Hash-based configuration lookup for O(1) matrix element access
- Batched matrix construction
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

try:
    from .base import Hamiltonian
except ImportError:
    from hamiltonians.base import Hamiltonian

try:
    import numba
    from numba import njit, int64, float64, int32
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Tolerance for filtering computed matrix elements.
# All integral paths (sequential, vectorized, Numba) use FP64, so no
# accumulation-order precision issues. 1e-12 Ha is well below chemical accuracy.
MATRIX_ELEMENT_TOL = 1e-12

# =============================================================================
# Numba JIT-compiled functions for get_connections acceleration
# These are module-level pure functions (no class state) for Numba compatibility.
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def numba_jw_sign_single(config, p, q):
        """
        JW sign for single excitation a+_p a_q.

        Sign = (-1)^(number of occupied sites between p and q).
        """
        if p == q:
            return 1
        low = min(p, q)
        high = max(p, q)
        count = 0
        for i in range(low + 1, high):
            count += config[i]
        if (count & 1) == 0:
            return 1
        return -1

    @njit(cache=True)
    def numba_jw_sign_double(config, p, r, q, s):
        """
        JW sign for double excitation a+_p a+_r a_s a_q.

        Operators applied RIGHT-TO-LEFT:
        1. a_q (annihilate q) on original config
        2. a_s (annihilate s) on config with q removed
        3. a+_r (create r) on config with q,s removed
        4. a+_p (create p) on config with q,s removed, r added
        """
        total_count = 0

        # 1. a_q: count occupied below q
        c_q = 0
        for i in range(q):
            c_q += config[i]
        total_count += c_q

        # 2. a_s: count occupied below s, adjust for q removal
        c_s = 0
        for i in range(s):
            c_s += config[i]
        if q < s:
            c_s -= 1
        total_count += c_s

        # 3. a+_r: count occupied below r, adjust for q,s removal
        c_r = 0
        for i in range(r):
            c_r += config[i]
        if q < r:
            c_r -= 1
        if s < r:
            c_r -= 1
        total_count += c_r

        # 4. a+_p: count occupied below p, adjust for q,s removal and r addition
        c_p = 0
        for i in range(p):
            c_p += config[i]
        if q < p:
            c_p -= 1
        if s < p:
            c_p -= 1
        if r < p:
            c_p += 1
        total_count += c_p

        if (total_count & 1) == 0:
            return 1
        return -1

    @njit(cache=True)
    def _numba_single_excitations(
        config, n_orb, J_single, K_single,
        single_exc_pq, single_exc_hpq,
    ):
        """
        Compute all single excitation connections and matrix elements.

        Returns:
            conns: list of new configs (numpy arrays)
            elems: list of matrix element values (float64)
        """
        n_sites = len(config)

        # Find occupied/virtual orbitals
        occ_alpha = np.empty(n_orb, dtype=np.int64)
        n_occ_a = 0
        virt_alpha = np.empty(n_orb, dtype=np.int64)
        n_virt_a = 0
        for i in range(n_orb):
            if config[i] == 1:
                occ_alpha[n_occ_a] = i
                n_occ_a += 1
            else:
                virt_alpha[n_virt_a] = i
                n_virt_a += 1

        occ_beta = np.empty(n_orb, dtype=np.int64)
        n_occ_b = 0
        virt_beta = np.empty(n_orb, dtype=np.int64)
        n_virt_b = 0
        for i in range(n_orb):
            if config[i + n_orb] == 1:
                occ_beta[n_occ_b] = i
                n_occ_b += 1
            else:
                virt_beta[n_virt_b] = i
                n_virt_b += 1

        # Pre-allocate (upper bound: n_exc * 2 for alpha + beta)
        max_conns = len(single_exc_pq) * 2
        conn_buf = np.empty((max_conns, n_sites), dtype=np.int64)
        elem_buf = np.empty(max_conns, dtype=np.float64)
        count = 0

        # Build occupied sets as boolean arrays for O(1) lookup
        occ_a_set = np.zeros(n_orb, dtype=np.int64)
        for i in range(n_occ_a):
            occ_a_set[occ_alpha[i]] = 1
        virt_a_set = np.zeros(n_orb, dtype=np.int64)
        for i in range(n_virt_a):
            virt_a_set[virt_alpha[i]] = 1
        occ_b_set = np.zeros(n_orb, dtype=np.int64)
        for i in range(n_occ_b):
            occ_b_set[occ_beta[i]] = 1
        virt_b_set = np.zeros(n_orb, dtype=np.int64)
        for i in range(n_virt_b):
            virt_b_set[virt_beta[i]] = 1

        n_exc = len(single_exc_pq)
        for idx in range(n_exc):
            p = single_exc_pq[idx, 0]
            q = single_exc_pq[idx, 1]
            h_pq = single_exc_hpq[idx]

            # Alpha: q -> p
            if occ_a_set[q] == 1 and virt_a_set[p] == 1:
                val = h_pq
                for ri in range(n_occ_a):
                    r = occ_alpha[ri]
                    val += J_single[p, q, r] - K_single[p, q, r]
                for ri in range(n_occ_b):
                    r = occ_beta[ri]
                    val += J_single[p, q, r]

                if abs(val) > 1e-12:
                    for k in range(n_sites):
                        conn_buf[count, k] = config[k]
                    conn_buf[count, q] = 0
                    conn_buf[count, p] = 1
                    sign = numba_jw_sign_single(config, p, q)
                    elem_buf[count] = sign * val
                    count += 1

            # Beta: q -> p
            if occ_b_set[q] == 1 and virt_b_set[p] == 1:
                val = h_pq
                for ri in range(n_occ_b):
                    r = occ_beta[ri]
                    val += J_single[p, q, r] - K_single[p, q, r]
                for ri in range(n_occ_a):
                    r = occ_alpha[ri]
                    val += J_single[p, q, r]

                if abs(val) > 1e-12:
                    p_idx = p + n_orb
                    q_idx = q + n_orb
                    for k in range(n_sites):
                        conn_buf[count, k] = config[k]
                    conn_buf[count, q_idx] = 0
                    conn_buf[count, p_idx] = 1
                    sign = numba_jw_sign_single(config, p_idx, q_idx)
                    elem_buf[count] = sign * val
                    count += 1

        return conn_buf[:count], elem_buf[:count]

    @njit(cache=True)
    def _numba_double_excitations(config, n_orb, h2e):
        """
        Compute all double excitation connections and matrix elements.

        Handles alpha-alpha, beta-beta, and alpha-beta double excitations
        following Slater-Condon rules.
        """
        n_sites = len(config)

        # Find occupied/virtual orbitals
        occ_alpha = np.empty(n_orb, dtype=np.int64)
        n_occ_a = 0
        virt_alpha = np.empty(n_orb, dtype=np.int64)
        n_virt_a = 0
        for i in range(n_orb):
            if config[i] == 1:
                occ_alpha[n_occ_a] = i
                n_occ_a += 1
            else:
                virt_alpha[n_virt_a] = i
                n_virt_a += 1

        occ_beta = np.empty(n_orb, dtype=np.int64)
        n_occ_b = 0
        virt_beta = np.empty(n_orb, dtype=np.int64)
        n_virt_b = 0
        for i in range(n_orb):
            if config[i + n_orb] == 1:
                occ_beta[n_occ_b] = i
                n_occ_b += 1
            else:
                virt_beta[n_virt_b] = i
                n_virt_b += 1

        # Upper bound on doubles
        n_aa = n_occ_a * (n_occ_a - 1) // 2 * n_virt_a * (n_virt_a - 1) // 2
        n_bb = n_occ_b * (n_occ_b - 1) // 2 * n_virt_b * (n_virt_b - 1) // 2
        n_ab = n_occ_a * n_occ_b * n_virt_a * n_virt_b
        max_doubles = n_aa + n_bb + n_ab
        if max_doubles == 0:
            return np.empty((0, n_sites), dtype=np.int64), np.empty(0, dtype=np.float64)

        conn_buf = np.empty((max_doubles, n_sites), dtype=np.int64)
        elem_buf = np.empty(max_doubles, dtype=np.float64)
        count = 0

        # Alpha-Alpha doubles
        for i in range(n_occ_a):
            q = occ_alpha[i]
            for j in range(i + 1, n_occ_a):
                s = occ_alpha[j]
                for k in range(n_virt_a):
                    p = virt_alpha[k]
                    for l in range(k + 1, n_virt_a):
                        r = virt_alpha[l]
                        val = h2e[p, q, r, s] - h2e[p, s, r, q]
                        if abs(val) > 1e-12:
                            for m in range(n_sites):
                                conn_buf[count, m] = config[m]
                            conn_buf[count, q] = 0
                            conn_buf[count, s] = 0
                            conn_buf[count, p] = 1
                            conn_buf[count, r] = 1
                            sign = numba_jw_sign_double(config, p, r, q, s)
                            elem_buf[count] = sign * val
                            count += 1

        # Beta-Beta doubles
        for i in range(n_occ_b):
            q = occ_beta[i]
            for j in range(i + 1, n_occ_b):
                s = occ_beta[j]
                for k in range(n_virt_b):
                    p = virt_beta[k]
                    for l in range(k + 1, n_virt_b):
                        r = virt_beta[l]
                        val = h2e[p, q, r, s] - h2e[p, s, r, q]
                        if abs(val) > 1e-12:
                            q_idx = q + n_orb
                            s_idx = s + n_orb
                            p_idx = p + n_orb
                            r_idx = r + n_orb
                            for m in range(n_sites):
                                conn_buf[count, m] = config[m]
                            conn_buf[count, q_idx] = 0
                            conn_buf[count, s_idx] = 0
                            conn_buf[count, p_idx] = 1
                            conn_buf[count, r_idx] = 1
                            sign = numba_jw_sign_double(config, p_idx, r_idx, q_idx, s_idx)
                            elem_buf[count] = sign * val
                            count += 1

        # Alpha-Beta doubles (no exchange term)
        for i in range(n_occ_a):
            q = occ_alpha[i]
            for j in range(n_occ_b):
                s = occ_beta[j]
                for k in range(n_virt_a):
                    p = virt_alpha[k]
                    for l in range(n_virt_b):
                        r = virt_beta[l]
                        val = h2e[p, q, r, s]
                        if abs(val) > 1e-12:
                            s_idx = s + n_orb
                            r_idx = r + n_orb
                            for m in range(n_sites):
                                conn_buf[count, m] = config[m]
                            conn_buf[count, q] = 0
                            conn_buf[count, s_idx] = 0
                            conn_buf[count, p] = 1
                            conn_buf[count, r_idx] = 1
                            sign = numba_jw_sign_double(config, p, r_idx, q, s_idx)
                            elem_buf[count] = sign * val
                            count += 1

        return conn_buf[:count], elem_buf[:count]

    def numba_get_connections(config_np, n_orb, h1e, h2e, J_single, K_single, single_exc_data):
        """
        Numba-accelerated get_connections.

        Drop-in replacement for MolecularHamiltonian.get_connections(),
        operating on numpy arrays directly.

        Args:
            config_np: Configuration as numpy int64 array (n_sites,)
            n_orb: Number of spatial orbitals
            h1e: One-electron integrals (n_orb, n_orb)
            h2e: Two-electron integrals (n_orb, n_orb, n_orb, n_orb)
            J_single: Precomputed Coulomb tensor J[p,q,r] = h2e[p,q,r,r]
            K_single: Precomputed exchange tensor K[p,q,r] = h2e[p,r,r,q]
            single_exc_data: List of (p, q, h_pq) tuples for non-zero h1e elements

        Returns:
            (connected_configs, matrix_elements) as numpy arrays
        """
        config_np = np.asarray(config_np, dtype=np.int64)

        # Convert single_exc_data to arrays for Numba
        n_exc = len(single_exc_data)
        if n_exc > 0:
            single_exc_pq = np.empty((n_exc, 2), dtype=np.int64)
            single_exc_hpq = np.empty(n_exc, dtype=np.float64)
            for i, (p, q, h_pq) in enumerate(single_exc_data):
                single_exc_pq[i, 0] = p
                single_exc_pq[i, 1] = q
                single_exc_hpq[i] = h_pq
        else:
            single_exc_pq = np.empty((0, 2), dtype=np.int64)
            single_exc_hpq = np.empty(0, dtype=np.float64)

        # Single excitations — preserve original dtype (float32) to match Python path
        s_conns, s_elems = _numba_single_excitations(
            config_np, n_orb,
            np.ascontiguousarray(J_single),
            np.ascontiguousarray(K_single),
            single_exc_pq, single_exc_hpq,
        )

        # Double excitations — preserve original dtype
        d_conns, d_elems = _numba_double_excitations(
            config_np, n_orb,
            np.ascontiguousarray(h2e),
        )

        # Combine
        if len(s_conns) == 0 and len(d_conns) == 0:
            n_sites = len(config_np)
            return np.empty((0, n_sites), dtype=np.int64), np.empty(0, dtype=np.float64)
        elif len(s_conns) == 0:
            return d_conns, d_elems
        elif len(d_conns) == 0:
            return s_conns, s_elems
        else:
            return (
                np.concatenate((s_conns, d_conns), axis=0),
                np.concatenate((s_elems, d_elems)),
            )

else:
    # Fallback stubs when Numba is not available
    def numba_jw_sign_single(config, p, q):
        raise ImportError("numba is required for numba_jw_sign_single")

    def numba_jw_sign_double(config, p, r, q, s):
        raise ImportError("numba is required for numba_jw_sign_double")

    def numba_get_connections(config_np, n_orb, h1e, h2e, J_single, K_single, single_exc_data):
        raise ImportError("numba is required for numba_get_connections")


@dataclass
class MolecularIntegrals:
    """Container for molecular integrals."""

    h1e: np.ndarray  # One-electron integrals (n_orb, n_orb)
    h2e: np.ndarray  # Two-electron integrals (n_orb, n_orb, n_orb, n_orb)
    nuclear_repulsion: float
    n_electrons: int
    n_orbitals: int
    n_alpha: int  # Number of alpha electrons
    n_beta: int   # Number of beta electrons
    # Cache metadata (set by compute_molecular_integrals for FCI caching)
    _geometry: Optional[list] = None
    _basis: Optional[str] = None
    _charge: int = 0
    _spin: int = 0


class MolecularHamiltonian(Hamiltonian):
    """
    Second-quantized molecular Hamiltonian with GPU acceleration.

    H = sum_{pq,s} h_pq a+_{ps} a_{qs}
        + 1/2 sum_{pqrs,st} h_pqrs a+_{ps} a+_{rt} a_{st} a_{qs}
        + E_nuc

    Uses Jordan-Wigner transformation to map to qubits:
    - alpha-spin orbitals on sites 0, 1, ..., n_orb-1
    - beta-spin orbitals on sites n_orb, ..., 2*n_orb-1

    Includes optimizations:
    - Vectorized batch diagonal computation
    - Precomputed Coulomb/Exchange tensors
    - Hash-based configuration lookup

    Args:
        integrals: MolecularIntegrals object
        device: Torch device for GPU acceleration
    """

    def __init__(
        self,
        integrals: MolecularIntegrals,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        n_qubits = 2 * integrals.n_orbitals  # Spin orbitals

        super().__init__(n_qubits, local_dim=2)

        self.device = device
        self.integrals = integrals
        # FP64 for integrals: PySCF provides FP64 and Slater-Condon J-K subtraction
        # suffers catastrophic cancellation in FP32. Memory is trivial (n_orb^4 * 8B).
        self.h1e = torch.from_numpy(integrals.h1e).double().to(device)
        self.h2e = torch.from_numpy(integrals.h2e).double().to(device)
        self.nuclear_repulsion = integrals.nuclear_repulsion
        self.n_orbitals = integrals.n_orbitals
        self.n_electrons = integrals.n_electrons
        self.n_alpha = integrals.n_alpha
        self.n_beta = integrals.n_beta

        # Pre-convert h2e to numpy ONCE (avoids GPU->CPU transfer in get_connections)
        # Already FP64 from h2e tensor above.
        # Must be done BEFORE _precompute_vectorized_integrals which uses it
        self._h2e_np = self.h2e.cpu().numpy()

        # Keep h2e on GPU for vectorized operations
        self._h2e_gpu = self.h2e.to(device)
        self._h1e_gpu = self.h1e.to(device)

        # Precompute vectorized integral tensors (creates h1_offdiag_indices)
        self._precompute_vectorized_integrals()

        # Precompute single excitation data (depends on h1_offdiag_indices)
        self._precompute_single_excitation_data()

        # Precompute excitation index tensors for vectorized batch operations
        # (depends on h1_offdiag_indices and h2e)
        self._precompute_excitation_indices()

    def _precompute_vectorized_integrals(self):
        """Precompute tensors for vectorized energy evaluation."""
        n_orb = self.n_orbitals
        device = self.device

        # One-body diagonal: h_pp
        self.h1_diag = torch.diag(self.h1e)  # (n_orb,)

        # Two-body Coulomb tensor: J_pq = h2e[p,p,q,q] - VECTORIZED
        p_idx = torch.arange(n_orb, device=device)
        q_idx = torch.arange(n_orb, device=device)
        self.J_tensor = self.h2e[p_idx[:, None], p_idx[:, None], q_idx[None, :], q_idx[None, :]]

        # Two-body Exchange tensor: K_pq = h2e[p,q,q,p] - VECTORIZED
        self.K_tensor = self.h2e[p_idx[:, None], q_idx[None, :], q_idx[None, :], p_idx[:, None]]

        # Precompute nonzero off-diagonal h1e indices
        tol = 1e-12
        h1_offdiag_mask = (torch.abs(self.h1e) > tol) & ~torch.eye(n_orb, device=device, dtype=torch.bool)
        self.h1_offdiag_indices = torch.nonzero(h1_offdiag_mask)
        self.h1_offdiag_values = self.h1e[h1_offdiag_mask]

        # Precompute Coulomb/Exchange tensors for single excitation Slater-Condon rules
        # J_single[p,q,r] = h2e[p,q,r,r] (Coulomb-type for excitation p<-q with spectator r)
        # K_single[p,q,r] = h2e[p,r,r,q] (Exchange-type for excitation p<-q with spectator r)
        # Slater-Condon J/K/JK tensors (already FP64 from h2e)
        r_idx = torch.arange(n_orb, device=device)
        self._J_single = self.h2e[:, :, r_idx, r_idx]  # (n_orb, n_orb, n_orb) axes: (p, q, r)
        # h2e[:, r_idx, r_idx, :] gives axes (p, r, q) -> permute to (p, q, r)
        self._K_single = self.h2e[:, r_idx, r_idx, :].permute(0, 2, 1)  # (n_orb, n_orb, n_orb)
        # Combined tensor: J - K for same-spin contribution
        self._JK_single = self._J_single - self._K_single  # (n_orb, n_orb, n_orb)

        # Numpy versions for sequential/Numba get_connections (FP64 from h2e)
        self._J_single_np = self._J_single.cpu().numpy()
        self._K_single_np = self._K_single.cpu().numpy()

        # OPTIMIZATION: Precompute sparse h2e dictionary for double excitations
        # This avoids iterating over all (p,q,r,s) combinations in get_connections
        # For C2H4 (14 orbitals): reduces from 38,416 iterations to ~500-2000 nonzero
        self._precompute_sparse_h2e()

    def _precompute_single_excitation_data(self):
        """Precompute data for fast single excitation enumeration."""
        self.single_exc_data = []
        for idx in range(len(self.h1_offdiag_indices)):
            p, q = self.h1_offdiag_indices[idx]
            h_pq = self.h1_offdiag_values[idx]
            self.single_exc_data.append((p.item(), q.item(), h_pq.item()))

    def _precompute_excitation_indices(self):
        """
        Precompute all possible excitation index combinations as GPU tensors.

        This enables fully vectorized batch operations by pre-generating
        all (occupied, virtual) index pairs that could form valid excitations.

        GPU Memory usage: O(n_orb^2) for singles, O(n_orb^4) for doubles
        For C2H4 (14 orbitals): ~2MB for singles, ~150MB for doubles
        """
        n_orb = self.n_orbitals
        device = self.device

        # === SINGLE EXCITATIONS ===
        # All (p, q) pairs where p != q and h1e[p,q] != 0
        # Store as GPU tensors for vectorized operations
        single_p = self.h1_offdiag_indices[:, 0].to(device)  # target (virtual)
        single_q = self.h1_offdiag_indices[:, 1].to(device)  # source (occupied)
        single_h1e = self.h1_offdiag_values.to(device)

        self._single_p = single_p
        self._single_q = single_q
        self._single_h1e = single_h1e  # Already FP64 from h1e

        # === DOUBLE EXCITATIONS: Same-spin (alpha-alpha, beta-beta) ===
        # Need all (q, s, p, r) where q < s (occupied), p < r (virtual), all distinct
        # Pre-filter by non-zero h2e[p,q,r,s] - h2e[p,s,r,q]

        # Generate all ordered pairs for occupied: (q, s) with q < s
        occ_pairs_q = []
        occ_pairs_s = []
        for q in range(n_orb):
            for s in range(q + 1, n_orb):
                occ_pairs_q.append(q)
                occ_pairs_s.append(s)

        # Generate all ordered pairs for virtual: (p, r) with p < r
        virt_pairs_p = []
        virt_pairs_r = []
        for p in range(n_orb):
            for r in range(p + 1, n_orb):
                virt_pairs_p.append(p)
                virt_pairs_r.append(r)

        # Create all combinations of (q,s) x (p,r)
        n_occ_pairs = len(occ_pairs_q)
        n_virt_pairs = len(virt_pairs_p)

        if n_occ_pairs > 0 and n_virt_pairs > 0:
            # Expand to all combinations
            double_q = torch.tensor(occ_pairs_q, device=device).repeat_interleave(n_virt_pairs)
            double_s = torch.tensor(occ_pairs_s, device=device).repeat_interleave(n_virt_pairs)
            double_p = torch.tensor(virt_pairs_p, device=device).repeat(n_occ_pairs)
            double_r = torch.tensor(virt_pairs_r, device=device).repeat(n_occ_pairs)

            # Filter out cases where indices overlap (p,r must be different from q,s)
            # Valid if: p != q, p != s, r != q, r != s
            valid_mask = (
                (double_p != double_q) & (double_p != double_s) &
                (double_r != double_q) & (double_r != double_s)
            )

            double_q = double_q[valid_mask]
            double_s = double_s[valid_mask]
            double_p = double_p[valid_mask]
            double_r = double_r[valid_mask]

            # Compute h2e values: h2e[p,q,r,s] - h2e[p,s,r,q] (exchange)
            h2e_direct = self._h2e_gpu[double_p, double_q, double_r, double_s]
            h2e_exchange = self._h2e_gpu[double_p, double_s, double_r, double_q]
            double_h2e_same = h2e_direct - h2e_exchange

            # Filter by non-zero values
            nonzero_mask = double_h2e_same.abs() > 1e-12
            self._double_same_q = double_q[nonzero_mask]
            self._double_same_s = double_s[nonzero_mask]
            self._double_same_p = double_p[nonzero_mask]
            self._double_same_r = double_r[nonzero_mask]
            self._double_same_h2e = double_h2e_same[nonzero_mask]
        else:
            self._double_same_q = torch.empty(0, dtype=torch.long, device=device)
            self._double_same_s = torch.empty(0, dtype=torch.long, device=device)
            self._double_same_p = torch.empty(0, dtype=torch.long, device=device)
            self._double_same_r = torch.empty(0, dtype=torch.long, device=device)
            self._double_same_h2e = torch.empty(0, device=device)

        # === DOUBLE EXCITATIONS: Alpha-Beta (no exchange) ===
        # All (q_a, s_b, p_a, r_b) combinations
        all_q = []
        all_s = []
        all_p = []
        all_r = []
        for q in range(n_orb):
            for s in range(n_orb):
                for p in range(n_orb):
                    if p == q:
                        continue
                    for r in range(n_orb):
                        if r == s:
                            continue
                        all_q.append(q)
                        all_s.append(s)
                        all_p.append(p)
                        all_r.append(r)

        if len(all_q) > 0:
            ab_q = torch.tensor(all_q, device=device)
            ab_s = torch.tensor(all_s, device=device)
            ab_p = torch.tensor(all_p, device=device)
            ab_r = torch.tensor(all_r, device=device)

            # h2e values (no exchange for alpha-beta)
            ab_h2e = self._h2e_gpu[ab_p, ab_q, ab_r, ab_s]

            # Filter by non-zero
            nonzero_mask = ab_h2e.abs() > 1e-12
            self._double_ab_q = ab_q[nonzero_mask]
            self._double_ab_s = ab_s[nonzero_mask]
            self._double_ab_p = ab_p[nonzero_mask]
            self._double_ab_r = ab_r[nonzero_mask]
            self._double_ab_h2e = ab_h2e[nonzero_mask]
        else:
            self._double_ab_q = torch.empty(0, dtype=torch.long, device=device)
            self._double_ab_s = torch.empty(0, dtype=torch.long, device=device)
            self._double_ab_p = torch.empty(0, dtype=torch.long, device=device)
            self._double_ab_r = torch.empty(0, dtype=torch.long, device=device)
            self._double_ab_h2e = torch.empty(0, device=device)

        # Precompute powers of 2 for integer encoding (on GPU).
        # For num_sites >= 64, int64 overflows at 2^63, so we split into
        # two halves. See utils.config_hash for details.
        if self.num_sites < 64:
            self._powers_gpu = (
                2 ** torch.arange(self.num_sites, device=device, dtype=torch.long)
            ).flip(0)
            self._powers_gpu_hi = None
            self._powers_gpu_lo = None
        else:
            self._powers_gpu = None  # Cannot use single int64 encoding
            half = self.num_sites // 2
            n_lo = self.num_sites - half
            self._powers_gpu_hi = (
                2 ** torch.arange(half, device=device, dtype=torch.long)
            ).flip(0)
            self._powers_gpu_lo = (
                2 ** torch.arange(n_lo, device=device, dtype=torch.long)
            ).flip(0)

    def _config_int_hash(self, configs: torch.Tensor) -> list:
        """Hash binary configs to integers for fast dict/set lookups.

        Overflow-safe: for num_sites >= 64, returns list of (int, int) tuples
        instead of plain ints. Both types are hashable and usable as dict keys.

        Args:
            configs: (n_configs, num_sites) binary tensor on self.device

        Returns:
            list of int (n_sites < 64) or list of tuple[int, int] (n_sites >= 64)
        """
        if self._powers_gpu is not None:
            # Standard path: single int64 encoding
            return (configs.long() * self._powers_gpu).sum(dim=1).cpu().tolist()
        else:
            # Split path: two halves to avoid int64 overflow
            half = self.num_sites // 2
            hash_hi = (configs[:, :half].long() * self._powers_gpu_hi).sum(dim=1)
            hash_lo = (configs[:, half:].long() * self._powers_gpu_lo).sum(dim=1)
            return list(zip(hash_hi.cpu().tolist(), hash_lo.cpu().tolist()))

    def _precompute_sparse_h2e(self):
        """
        Precompute sparse dictionaries for non-zero h2e elements.

        This optimization provides 5-20x speedup for get_connections() by
        avoiding iteration over all (p,q,r,s) combinations.

        For C2H4 (14 orbitals): reduces from 38,416 to ~500-2000 nonzero entries.

        Creates three dictionaries for same-spin and mixed-spin excitations:
        - h2e_same_spin: (occ_i, occ_j) -> [(virt_k, virt_l, val), ...]
        - h2e_alpha_beta: (occ_a, occ_b) -> [(virt_a, virt_b, val), ...]
        """
        h2e_np = self._h2e_np
        n_orb = self.n_orbitals
        tol = 1e-12

        # For same-spin (alpha-alpha or beta-beta) double excitations:
        # Need: h2e[p,q,r,s] - h2e[p,s,r,q] where q,s occupied, p,r virtual
        # Store as: occ_pair (q,s) -> list of (virt_pair (p,r), exchange_value)
        self._h2e_same_spin_by_occ = {}

        # For alpha-beta double excitations:
        # Need: h2e[p,q,r,s] where q occupied_alpha, s occupied_beta, p virtual_alpha, r virtual_beta
        # Store as: (q, s) -> list of (p, r, val)
        self._h2e_alpha_beta_by_occ = {}

        # Build same-spin lookup: iterate over all orbital quartets
        # (q, s) occupied pair -> (p, r) virtual pair -> value
        for q in range(n_orb):
            for s in range(q + 1, n_orb):  # s > q to avoid double counting
                pairs = []
                for p in range(n_orb):
                    for r in range(p + 1, n_orb):  # r > p to avoid double counting
                        # Skip if any indices overlap
                        if p == q or p == s or r == q or r == s:
                            continue
                        # Exchange integral for same-spin
                        val = h2e_np[p, q, r, s] - h2e_np[p, s, r, q]
                        if abs(val) > tol:
                            pairs.append((p, r, val))
                if pairs:
                    self._h2e_same_spin_by_occ[(q, s)] = pairs

        # Build alpha-beta lookup: no exchange term
        for q in range(n_orb):  # alpha occupied
            for s in range(n_orb):  # beta occupied
                pairs = []
                for p in range(n_orb):  # alpha virtual
                    if p == q:
                        continue
                    for r in range(n_orb):  # beta virtual
                        if r == s:
                            continue
                        val = h2e_np[p, q, r, s]
                        if abs(val) > tol:
                            pairs.append((p, r, val))
                if pairs:
                    self._h2e_alpha_beta_by_occ[(q, s)] = pairs

        # Statistics for debugging
        n_same = sum(len(v) for v in self._h2e_same_spin_by_occ.values())
        n_ab = sum(len(v) for v in self._h2e_alpha_beta_by_occ.values())
        self._h2e_sparsity_stats = {
            'n_same_spin_nonzero': n_same,
            'n_alpha_beta_nonzero': n_ab,
            'n_orbitals': n_orb,
            'full_size': n_orb ** 4,
        }

    def _orbital_to_qubit(self, orbital: int, spin: str) -> int:
        """Map orbital index and spin to qubit index."""
        if spin == "alpha" or spin == "a":
            return orbital
        else:  # beta
            return orbital + self.n_orbitals

    def _qubit_to_orbital(self, qubit: int) -> Tuple[int, str]:
        """Map qubit index to orbital and spin."""
        if qubit < self.n_orbitals:
            return qubit, "alpha"
        else:
            return qubit - self.n_orbitals, "beta"

    @torch.no_grad()
    def diagonal_elements_batch(self, configs: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized diagonal energy computation for a batch.

        E_diag = E_nuc + sum_p h_pp * n_p + 0.5 * sum_{p!=q} (J_pq - K_pq*delta_s) * n_p * n_q

        Args:
            configs: (batch_size, num_sites) occupation numbers

        Returns:
            (batch_size,) diagonal energies
        """
        dtype = self.h1e.dtype  # Match integral precision (FP64)
        configs = configs.to(device=self.device, dtype=dtype)
        batch_size = configs.shape[0]
        n_orb = self.n_orbitals

        # Split into alpha and beta
        n_alpha = configs[:, :n_orb]  # (batch, n_orb)
        n_beta = configs[:, n_orb:]   # (batch, n_orb)

        # Nuclear repulsion
        energies = torch.full((batch_size,), self.nuclear_repulsion,
                             device=self.device, dtype=dtype)

        # One-body: sum_p h_pp * (n_p^alpha + n_p^beta)
        energies += (n_alpha + n_beta) @ self.h1_diag

        # Two-body Coulomb (J)
        # alpha-alpha: 0.5 * sum_{p!=q} J_pq * n_p^a * n_q^a
        J_aa = 0.5 * (torch.einsum('bp,pq,bq->b', n_alpha, self.J_tensor, n_alpha)
                      - torch.sum(n_alpha * torch.diag(self.J_tensor), dim=1))

        # beta-beta
        J_bb = 0.5 * (torch.einsum('bp,pq,bq->b', n_beta, self.J_tensor, n_beta)
                      - torch.sum(n_beta * torch.diag(self.J_tensor), dim=1))

        # alpha-beta (no self-exclusion needed)
        J_ab = torch.einsum('bp,pq,bq->b', n_alpha, self.J_tensor, n_beta)

        energies += J_aa + J_bb + J_ab

        # Two-body Exchange (K): same spin only
        K_aa = -0.5 * (torch.einsum('bp,pq,bq->b', n_alpha, self.K_tensor, n_alpha)
                       - torch.sum(n_alpha * torch.diag(self.K_tensor), dim=1))

        K_bb = -0.5 * (torch.einsum('bp,pq,bq->b', n_beta, self.K_tensor, n_beta)
                       - torch.sum(n_beta * torch.diag(self.K_tensor), dim=1))

        energies += K_aa + K_bb

        return energies

    def diagonal_element(self, config: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal element <x|H|x> for single configuration.

        Uses vectorized batch computation internally.
        """
        return self.diagonal_elements_batch(config.unsqueeze(0))[0]

    def get_connections(
        self, config: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get off-diagonal connections for a configuration.

        Off-diagonal elements arise from:
        1. Single excitations: a+_p a_q with p != q (one-body terms)
        2. Double excitations: a+_p a+_r a_s a_q (two-body terms)

        Returns only configurations with non-zero matrix elements.

        Optimized version: Uses numpy arrays for excitation enumeration
        and minimizes tensor cloning operations.
        """
        device = self.device
        config = config.to(device)
        n_orb = self.n_orbitals

        # Work with numpy for faster loops, convert once at end
        config_np = config.cpu().numpy()

        connected_list = []
        elements_list = []

        # Get occupied/virtual orbitals as numpy arrays (faster iteration)
        occ_alpha = np.where(config_np[:n_orb] == 1)[0]
        occ_beta = np.where(config_np[n_orb:] == 1)[0]
        virt_alpha = np.where(config_np[:n_orb] == 0)[0]
        virt_beta = np.where(config_np[n_orb:] == 0)[0]

        # Use precomputed numpy array (avoids GPU->CPU transfer per call)
        h2e_np = self._h2e_np

        # ===== SINGLE EXCITATIONS (Slater-Condon rules) =====
        # Full matrix element for single excitation D -> D_{p<-q}:
        #   <D_{p<-q}|H|D> = h_pq
        #     + sum_{r in occ_same_spin, r!=q} [<pr|qr> - <pr|rq>]
        #     + sum_{r' in occ_other_spin} <pr'|qr'>
        # where <ab|cd> (physicist) = h2e[a,c,b,d] (chemist notation)
        occ_alpha_set = set(occ_alpha)
        occ_beta_set = set(occ_beta)
        virt_alpha_set = set(virt_alpha)
        virt_beta_set = set(virt_beta)

        # Precomputed Slater-Condon tensors for efficient single excitation computation
        J_single_np = self._J_single_np   # J_single[p,q,r] = h2e[p,q,r,r]
        K_single_np = self._K_single_np   # K_single[p,q,r] = h2e[p,r,r,q]

        for p, q, h_pq in self.single_exc_data:
            # Alpha: q -> p
            if q in occ_alpha_set and p in virt_alpha_set:
                # Full Slater-Condon: h_pq + sum_r n_alpha[r]*(J-K) + sum_r n_beta[r]*J
                # Note: r=q contribution to (J-K) is zero since J[p,q,q]=K[p,q,q]=h2e[p,q,q,q]
                val = h_pq
                for r in occ_alpha:
                    val += J_single_np[p, q, r] - K_single_np[p, q, r]
                for r in occ_beta:
                    val += J_single_np[p, q, r]

                if abs(val) > MATRIX_ELEMENT_TOL:
                    new_config = config_np.copy()
                    new_config[q] = 0
                    new_config[p] = 1
                    sign = self._jw_sign_np(config_np, p, q)
                    connected_list.append(new_config)
                    elements_list.append(sign * val)

            # Beta: q -> p
            if q in occ_beta_set and p in virt_beta_set:
                val = h_pq
                for r in occ_beta:
                    val += J_single_np[p, q, r] - K_single_np[p, q, r]
                for r in occ_alpha:
                    val += J_single_np[p, q, r]

                if abs(val) > MATRIX_ELEMENT_TOL:
                    new_config = config_np.copy()
                    new_config[q + n_orb] = 0
                    new_config[p + n_orb] = 1
                    sign = self._jw_sign_np(config_np, p + n_orb, q + n_orb)
                    connected_list.append(new_config)
                    elements_list.append(sign * val)

        # ===== DOUBLE EXCITATIONS (two-body terms) =====
        # NOTE: Using original 4-nested-loops approach for correctness.
        # The sparse h2e optimization had a bug causing energy to go below ground state.

        # Alpha-Alpha
        n_occ_a = len(occ_alpha)
        n_virt_a = len(virt_alpha)
        for i in range(n_occ_a):
            q = occ_alpha[i]
            for j in range(i + 1, n_occ_a):
                s = occ_alpha[j]
                for k in range(n_virt_a):
                    p = virt_alpha[k]
                    for l in range(k + 1, n_virt_a):
                        r = virt_alpha[l]
                        val = h2e_np[p, q, r, s] - h2e_np[p, s, r, q]
                        if abs(val) > MATRIX_ELEMENT_TOL:
                            new_config = config_np.copy()
                            new_config[q] = 0
                            new_config[s] = 0
                            new_config[p] = 1
                            new_config[r] = 1
                            sign = self._jw_sign_double_np(config_np, p, r, q, s)
                            connected_list.append(new_config)
                            elements_list.append(sign * val)

        # Beta-Beta
        n_occ_b = len(occ_beta)
        n_virt_b = len(virt_beta)
        for i in range(n_occ_b):
            q = occ_beta[i]
            for j in range(i + 1, n_occ_b):
                s = occ_beta[j]
                for k in range(n_virt_b):
                    p = virt_beta[k]
                    for l in range(k + 1, n_virt_b):
                        r = virt_beta[l]
                        val = h2e_np[p, q, r, s] - h2e_np[p, s, r, q]
                        if abs(val) > MATRIX_ELEMENT_TOL:
                            new_config = config_np.copy()
                            q_idx = q + n_orb
                            s_idx = s + n_orb
                            p_idx = p + n_orb
                            r_idx = r + n_orb
                            new_config[q_idx] = 0
                            new_config[s_idx] = 0
                            new_config[p_idx] = 1
                            new_config[r_idx] = 1
                            sign = self._jw_sign_double_np(config_np, p_idx, r_idx, q_idx, s_idx)
                            connected_list.append(new_config)
                            elements_list.append(sign * val)

        # Alpha-Beta (no exchange term)
        for q in occ_alpha:
            for s in occ_beta:
                for p in virt_alpha:
                    for r in virt_beta:
                        val = h2e_np[p, q, r, s]
                        if abs(val) > MATRIX_ELEMENT_TOL:
                            new_config = config_np.copy()
                            s_idx = s + n_orb
                            r_idx = r + n_orb
                            new_config[q] = 0
                            new_config[s_idx] = 0
                            new_config[p] = 1
                            new_config[r_idx] = 1
                            sign = self._jw_sign_double_np(config_np, p, r_idx, q, s_idx)
                            connected_list.append(new_config)
                            elements_list.append(sign * val)

        if len(connected_list) == 0:
            return torch.empty(0, self.num_sites, device=device), torch.empty(0, device=device)

        # Convert to torch tensors once at the end
        connected = torch.from_numpy(np.array(connected_list)).to(device)
        elements = torch.tensor(elements_list, dtype=torch.float64, device=device)

        return connected, elements

    @torch.no_grad()
    def get_all_connections_with_indices(
        self, configs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get ALL off-diagonal connections for ALL configs at once.

        Optimized for batched local energy computation. Returns connections
        in a format ready for scatter_add accumulation.

        Args:
            configs: (n_configs, num_sites) configurations

        Returns:
            all_connected: (total_connections, num_sites) all connected configurations
            all_elements: (total_connections,) corresponding matrix elements
            config_indices: (total_connections,) which original config each connection belongs to
        """
        device = self.device
        configs = configs.to(device)
        n_configs = configs.shape[0]

        all_connected = []
        all_elements = []
        all_indices = []

        for i in range(n_configs):
            connected, elements = self.get_connections(configs[i])
            n_conn = len(connected)

            if n_conn > 0:
                all_connected.append(connected)
                all_elements.append(elements)
                all_indices.append(
                    torch.full((n_conn,), i, dtype=torch.long, device=device)
                )

        if not all_connected:
            return (
                torch.empty(0, self.num_sites, device=device),
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.long, device=device)
            )

        return (
            torch.cat(all_connected, dim=0),
            torch.cat(all_elements, dim=0),
            torch.cat(all_indices, dim=0)
        )

    def _jw_sign_np(self, config: np.ndarray, p: int, q: int) -> int:
        """
        Compute Jordan-Wigner sign for a+_p a_q (numpy version).

        Sign = (-1)^(number of occupied sites between p and q).
        Uses numpy vectorized slice sum — faster than Python for-loop
        for all gap sizes due to C-level array operations.
        """
        if p == q:
            return 1
        low, high = min(p, q), max(p, q)
        count = config[low + 1:high].sum()
        return 1 if (count & 1) == 0 else -1

    def _jw_sign_double_np(
        self, config: np.ndarray, p: int, r: int, q: int, s: int
    ) -> int:
        """
        Compute Jordan-Wigner sign for double excitation a+_p a+_r a_s a_q (numpy version).

        IMPORTANT: Operators are applied RIGHT-TO-LEFT in second quantization:
        1. a_q first (annihilate q) - on original config
        2. a_s second (annihilate s) - on config with q removed
        3. a+_r third (create r) - on config with q,s removed
        4. a+_p fourth (create p) - on config with q,s removed, r added

        Each JW string counts occupied sites to the LEFT of the operator position,
        accounting for modifications from previously applied operators.
        """
        total_count = 0

        # 1. a_q (FIRST operator, applied to original config)
        # JW string: count occupied sites below q in original config
        total_count += config[:q].sum()

        # 2. a_s (second operator, q has been removed)
        # JW string: count occupied sites below s, minus 1 if q < s (since q is now empty)
        count_s = config[:s].sum()
        if q < s:
            count_s -= 1  # q was occupied, now removed
        total_count += count_s

        # 3. a+_r (third operator, q and s have been removed)
        # JW string: count occupied sites below r, minus adjustments for q,s removal
        count_r = config[:r].sum()
        if q < r:
            count_r -= 1  # q was occupied, now removed
        if s < r:
            count_r -= 1  # s was occupied, now removed
        total_count += count_r

        # 4. a+_p (fourth operator, q,s removed, r added)
        # JW string: count occupied sites below p, with all adjustments
        count_p = config[:p].sum()
        if q < p:
            count_p -= 1  # q was occupied, now removed
        if s < p:
            count_p -= 1  # s was occupied, now removed
        if r < p:
            count_p += 1  # r was empty, now occupied
        total_count += count_p

        return (-1) ** int(total_count)

    def _jw_sign_double(
        self, config: torch.Tensor, p: int, r: int, q: int, s: int
    ) -> int:
        """
        Compute Jordan-Wigner sign for double excitation a+_p a+_r a_s a_q.

        IMPORTANT: Operators are applied RIGHT-TO-LEFT in second quantization:
        1. a_q first (annihilate q) - on original config
        2. a_s second (annihilate s) - on config with q removed
        3. a+_r third (create r) - on config with q,s removed
        4. a+_p fourth (create p) - on config with q,s removed, r added

        Each JW string counts occupied sites to the LEFT of the operator position,
        accounting for modifications from previously applied operators.
        """
        total_count = 0

        # 1. a_q (FIRST operator, applied to original config)
        total_count += config[:q].sum().item()

        # 2. a_s (second operator, q has been removed)
        count_s = config[:s].sum().item()
        if q < s:
            count_s -= 1  # q was occupied, now removed
        total_count += count_s

        # 3. a+_r (third operator, q and s have been removed)
        count_r = config[:r].sum().item()
        if q < r:
            count_r -= 1
        if s < r:
            count_r -= 1
        total_count += count_r

        # 4. a+_p (fourth operator, q,s removed, r added)
        count_p = config[:p].sum().item()
        if q < p:
            count_p -= 1
        if s < p:
            count_p -= 1
        if r < p:
            count_p += 1
        total_count += count_p

        return (-1) ** int(total_count)

    def _jw_sign(self, config: torch.Tensor, p: int, q: int) -> int:
        """
        Compute Jordan-Wigner sign for a+_p a_q.

        Sign = (-1)^(number of occupied sites between p and q)
        """
        if p == q:
            return 1
        low, high = min(p, q), max(p, q)
        count = config[low + 1:high].sum().item()
        return (-1) ** int(count)

    @torch.no_grad()
    def get_connections_vectorized_batch(
        self, configs: torch.Tensor, max_memory_mb: float = 2048.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        GPU-accelerated batch computation of Hamiltonian connections.

        Processes configurations in parallel using tensor operations.
        Auto-chunks to stay within ``max_memory_mb`` for intermediate tensors.
        This is 10-50x faster than sequential get_connections() calls for large batches.

        Args:
            configs: (n_configs, num_sites) basis configurations on GPU
            max_memory_mb: memory budget for intermediate tensors (default 2048 MB)

        Returns:
            all_connected: (total_connections, num_sites) connected configurations
            all_elements: (total_connections,) matrix elements H[i,j]
            batch_indices: (total_connections,) index of source config for each connection
        """
        n_configs = configs.shape[0]

        # Estimate peak intermediate tensor size per config
        n_exc_max = max(
            len(self._single_p) if hasattr(self, '_single_p') else 0,
            len(self._double_same_p) if hasattr(self, '_double_same_p') else 0,
            len(self._double_ab_p) if hasattr(self, '_double_ab_p') else 0,
        )
        # 5 intermediate tensors × n_exc × 8 bytes per config (float64)
        bytes_per_config = 5 * n_exc_max * 8
        max_chunk = max(1, int(max_memory_mb * 1e6 / max(bytes_per_config, 1)))

        if n_configs <= max_chunk:
            return self._get_connections_vectorized_batch_impl(configs)

        # Chunk processing to stay within memory budget
        all_connected, all_elements, all_batch_idx = [], [], []
        for start in range(0, n_configs, max_chunk):
            end = min(start + max_chunk, n_configs)
            chunk = configs[start:end]
            c, e, b = self._get_connections_vectorized_batch_impl(chunk)
            if len(c) > 0:
                # Adjust batch indices to global config indices
                all_connected.append(c)
                all_elements.append(e)
                all_batch_idx.append(b + start)

        if not all_connected:
            device = self.device
            num_sites = self.num_sites
            return (
                torch.empty(0, num_sites, device=device),
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )
        return (
            torch.cat(all_connected, dim=0),
            torch.cat(all_elements, dim=0),
            torch.cat(all_batch_idx, dim=0),
        )

    def _get_connections_vectorized_batch_impl(
        self, configs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inner implementation of vectorized batch connections (no chunking)."""
        device = self.device
        configs = configs.to(device)
        n_configs = configs.shape[0]
        n_orb = self.n_orbitals
        num_sites = self.num_sites

        all_connected = []
        all_elements = []
        all_batch_idx = []

        # Split configs into alpha and beta occupations
        alpha_occ = configs[:, :n_orb]  # (n_configs, n_orb)
        beta_occ = configs[:, n_orb:]   # (n_configs, n_orb)

        # ============================================================
        # SINGLE EXCITATIONS (one-body terms)
        # ============================================================
        # For each (p, q) pair with h1e[p,q] != 0, check which configs
        # have q occupied and p unoccupied (for alpha and beta separately)

        n_singles = len(self._single_p)
        if n_singles > 0:
            # Expand to (n_configs, n_singles)
            p_idx = self._single_p.unsqueeze(0).expand(n_configs, -1)  # (n_configs, n_singles)
            q_idx = self._single_q.unsqueeze(0).expand(n_configs, -1)  # (n_configs, n_singles)
            h1e_vals = self._single_h1e.unsqueeze(0).expand(n_configs, -1)  # (n_configs, n_singles)

            # Check alpha excitations: q occupied in alpha, p unoccupied in alpha
            # Use gather to get occupations at p and q indices
            q_occ_alpha = torch.gather(alpha_occ, 1, q_idx)  # (n_configs, n_singles)
            p_occ_alpha = torch.gather(alpha_occ, 1, p_idx)  # (n_configs, n_singles)
            valid_alpha = (q_occ_alpha == 1) & (p_occ_alpha == 0)  # (n_configs, n_singles)

            # Check beta excitations
            q_occ_beta = torch.gather(beta_occ, 1, q_idx)
            p_occ_beta = torch.gather(beta_occ, 1, p_idx)
            valid_beta = (q_occ_beta == 1) & (p_occ_beta == 0)

            # Process alpha singles with full Slater-Condon matrix elements
            if valid_alpha.any():
                config_idx, exc_idx = valid_alpha.nonzero(as_tuple=True)
                p_orb = self._single_p[exc_idx]  # orbital indices (no spin offset)
                q_orb = self._single_q[exc_idx]
                h_vals = self._single_h1e[exc_idx]

                # Full Slater-Condon: h_pq + n_alpha·(J-K) + n_beta·J
                JK_pq = self._JK_single[p_orb, q_orb, :]  # (n_exc, n_orb)
                J_pq = self._J_single[p_orb, q_orb, :]     # (n_exc, n_orb)
                alpha_n = alpha_occ[config_idx]              # (n_exc, n_orb)
                beta_n = beta_occ[config_idx]                # (n_exc, n_orb)
                two_body = (alpha_n * JK_pq).sum(dim=1) + (beta_n * J_pq).sum(dim=1)
                full_vals = h_vals + two_body

                # Filter out negligible elements
                significant = full_vals.abs() > MATRIX_ELEMENT_TOL
                if significant.any():
                    config_idx = config_idx[significant]
                    p_orb = p_orb[significant]
                    q_orb = q_orb[significant]
                    full_vals = full_vals[significant]

                    # Create new configs (use orbital indices for alpha)
                    new_configs = configs[config_idx].clone()
                    arange_idx = torch.arange(len(config_idx), device=device)
                    new_configs[arange_idx, q_orb] = 0
                    new_configs[arange_idx, p_orb] = 1

                    signs = self._jw_sign_vectorized(configs[config_idx], p_orb, q_orb)

                    all_connected.append(new_configs)
                    all_elements.append(signs * full_vals)
                    all_batch_idx.append(config_idx)

            # Process beta singles with full Slater-Condon matrix elements
            if valid_beta.any():
                config_idx, exc_idx = valid_beta.nonzero(as_tuple=True)
                p_orb = self._single_p[exc_idx]  # orbital indices (no spin offset)
                q_orb = self._single_q[exc_idx]
                h_vals = self._single_h1e[exc_idx]

                # Full Slater-Condon: h_pq + n_beta·(J-K) + n_alpha·J
                JK_pq = self._JK_single[p_orb, q_orb, :]
                J_pq = self._J_single[p_orb, q_orb, :]
                alpha_n = alpha_occ[config_idx]
                beta_n = beta_occ[config_idx]
                two_body = (beta_n * JK_pq).sum(dim=1) + (alpha_n * J_pq).sum(dim=1)
                full_vals = h_vals + two_body

                significant = full_vals.abs() > MATRIX_ELEMENT_TOL
                if significant.any():
                    config_idx = config_idx[significant]
                    p_orb = p_orb[significant]
                    q_orb = q_orb[significant]
                    full_vals = full_vals[significant]

                    p_vals = p_orb + n_orb  # Beta offset for qubit indices
                    q_vals = q_orb + n_orb

                    new_configs = configs[config_idx].clone()
                    arange_idx = torch.arange(len(config_idx), device=device)
                    new_configs[arange_idx, q_vals] = 0
                    new_configs[arange_idx, p_vals] = 1

                    signs = self._jw_sign_vectorized(configs[config_idx], p_vals, q_vals)

                    all_connected.append(new_configs)
                    all_elements.append(signs * full_vals)
                    all_batch_idx.append(config_idx)

        # ============================================================
        # DOUBLE EXCITATIONS: Same-spin (alpha-alpha, beta-beta)
        # ============================================================
        n_double_same = len(self._double_same_p)
        if n_double_same > 0:
            p_idx = self._double_same_p
            r_idx = self._double_same_r
            q_idx = self._double_same_q
            s_idx = self._double_same_s
            h2e_vals = self._double_same_h2e

            # Check alpha-alpha: q,s occupied, p,r unoccupied
            for spin_offset, occ_tensor in [(0, alpha_occ), (n_orb, beta_occ)]:
                # Get occupations at each index for all configs
                q_occ = occ_tensor[:, q_idx]  # (n_configs, n_double_same)
                s_occ = occ_tensor[:, s_idx]
                p_occ = occ_tensor[:, p_idx]
                r_occ = occ_tensor[:, r_idx]

                # Valid if q,s occupied AND p,r unoccupied
                valid = (q_occ == 1) & (s_occ == 1) & (p_occ == 0) & (r_occ == 0)

                if valid.any():
                    config_idx, exc_idx = valid.nonzero(as_tuple=True)
                    p_vals = p_idx[exc_idx] + spin_offset
                    r_vals = r_idx[exc_idx] + spin_offset
                    q_vals = q_idx[exc_idx] + spin_offset
                    s_vals = s_idx[exc_idx] + spin_offset
                    h_vals = h2e_vals[exc_idx]

                    # Create new configs
                    new_configs = configs[config_idx].clone()
                    arange_idx = torch.arange(len(config_idx), device=device)
                    new_configs[arange_idx, q_vals] = 0
                    new_configs[arange_idx, s_vals] = 0
                    new_configs[arange_idx, p_vals] = 1
                    new_configs[arange_idx, r_vals] = 1

                    # Compute JW signs for double excitations
                    signs = self._jw_sign_double_vectorized(
                        configs[config_idx], p_vals, r_vals, q_vals, s_vals
                    )

                    all_connected.append(new_configs)
                    all_elements.append(signs * h_vals)
                    all_batch_idx.append(config_idx)

        # ============================================================
        # DOUBLE EXCITATIONS: Alpha-Beta (no exchange)
        # ============================================================
        n_double_ab = len(self._double_ab_p)
        if n_double_ab > 0:
            p_idx = self._double_ab_p  # alpha virtual
            r_idx = self._double_ab_r  # beta virtual
            q_idx = self._double_ab_q  # alpha occupied
            s_idx = self._double_ab_s  # beta occupied
            h2e_vals = self._double_ab_h2e

            # Check: q occupied in alpha, s occupied in beta, p unoccupied in alpha, r unoccupied in beta
            q_occ = alpha_occ[:, q_idx]  # (n_configs, n_double_ab)
            s_occ = beta_occ[:, s_idx]
            p_occ = alpha_occ[:, p_idx]
            r_occ = beta_occ[:, r_idx]

            valid = (q_occ == 1) & (s_occ == 1) & (p_occ == 0) & (r_occ == 0)

            if valid.any():
                config_idx, exc_idx = valid.nonzero(as_tuple=True)
                p_vals = p_idx[exc_idx]           # alpha (no offset)
                r_vals = r_idx[exc_idx] + n_orb   # beta
                q_vals = q_idx[exc_idx]           # alpha
                s_vals = s_idx[exc_idx] + n_orb   # beta
                h_vals = h2e_vals[exc_idx]

                new_configs = configs[config_idx].clone()
                arange_idx = torch.arange(len(config_idx), device=device)
                new_configs[arange_idx, q_vals] = 0
                new_configs[arange_idx, s_vals] = 0
                new_configs[arange_idx, p_vals] = 1
                new_configs[arange_idx, r_vals] = 1

                signs = self._jw_sign_double_vectorized(
                    configs[config_idx], p_vals, r_vals, q_vals, s_vals
                )

                all_connected.append(new_configs)
                all_elements.append(signs * h_vals)
                all_batch_idx.append(config_idx)

        # Concatenate all results
        if len(all_connected) == 0:
            return (
                torch.empty(0, num_sites, device=device),
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.long, device=device)
            )

        return (
            torch.cat(all_connected, dim=0),
            torch.cat(all_elements, dim=0),
            torch.cat(all_batch_idx, dim=0)
        )

    def _jw_sign_vectorized(
        self, configs: torch.Tensor, p: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized JW sign computation for single excitations.

        Args:
            configs: (batch,, num_sites) configurations
            p: (batch,) target site indices
            q: (batch,) source site indices

        Returns:
            (batch,) signs (+1 or -1)
        """
        batch_size = configs.shape[0]
        device = configs.device

        # Compute count of occupied sites between p and q for each config
        low = torch.minimum(p, q)
        high = torch.maximum(p, q)

        # Create mask for sites between low and high (exclusive)
        site_indices = torch.arange(configs.shape[1], device=device).unsqueeze(0)  # (1, num_sites)
        low_expanded = low.unsqueeze(1)   # (batch, 1)
        high_expanded = high.unsqueeze(1)  # (batch, 1)

        # Mask: True for sites in range (low, high) exclusive
        mask = (site_indices > low_expanded) & (site_indices < high_expanded)

        # Count occupied sites in the masked region
        counts = (configs * mask).sum(dim=1)

        # Sign is (-1)^count
        signs = 1 - 2 * (counts % 2)  # Equivalent to (-1)**count but faster
        return signs.float()

    def _jw_sign_double_vectorized(
        self, configs: torch.Tensor, p: torch.Tensor, r: torch.Tensor,
        q: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized JW sign computation for double excitations a+_p a+_r a_s a_q.

        Operators applied right-to-left: a_q, a_s, a+_r, a+_p

        Args:
            configs: (batch, num_sites) original configurations
            p, r: (batch,) creation site indices
            q, s: (batch,) annihilation site indices

        Returns:
            (batch,) signs (+1 or -1)
        """
        batch_size = configs.shape[0]
        device = configs.device
        num_sites = configs.shape[1]

        # Cumulative sum of occupations (for counting occupied sites below index)
        # cumsum[i] = sum of configs[:i] (occupied sites with index < i)
        configs_f = configs.float()
        cumsum = torch.cat([
            torch.zeros(batch_size, 1, device=device),
            configs_f.cumsum(dim=1)[:, :-1]
        ], dim=1)  # (batch, num_sites)

        # Gather cumsum values at each index
        batch_idx = torch.arange(batch_size, device=device)

        # 1. a_q: count = cumsum[q]
        count_q = cumsum[batch_idx, q]

        # 2. a_s: count = cumsum[s] - (q < s).float()
        count_s = cumsum[batch_idx, s] - (q < s).float()

        # 3. a+_r: count = cumsum[r] - (q < r).float() - (s < r).float()
        count_r = cumsum[batch_idx, r] - (q < r).float() - (s < r).float()

        # 4. a+_p: count = cumsum[p] - (q < p).float() - (s < p).float() + (r < p).float()
        count_p = cumsum[batch_idx, p] - (q < p).float() - (s < p).float() + (r < p).float()

        total_count = count_q + count_s + count_r + count_p

        # Sign is (-1)^total_count
        signs = 1 - 2 * (total_count.long() % 2)
        return signs.float()

    @torch.no_grad()
    def matrix_elements_fast(
        self,
        configs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fast Hamiltonian matrix construction with enforced Hermitian symmetry.

        Uses vectorized diagonal and optimized off-diagonal computation.
        For large bases (>200 configs), uses integer hash encoding for speed.

        IMPORTANT: Builds only the lower triangle and mirrors to upper triangle
        to guarantee Hermitian symmetry. This avoids issues where H[i,j] and H[j,i]
        might differ due to JW sign computation from different directions.

        Args:
            configs: (n_configs, num_sites) basis configurations

        Returns:
            (n_configs, n_configs) Hermitian Hamiltonian matrix
        """
        configs = configs.to(self.device)
        n_configs = configs.shape[0]

        # SIZE GUARD: refuse to allocate dense matrices that would cause OOM.
        # n² × 8 bytes (float64): 20000² = 3.2 GB, 50000² = 20 GB, 63504² = 32 GB.
        # On DGX Spark 128GB UMA, 20000 is a safe upper bound for dense allocation.
        MAX_DENSE_CONFIGS = 20000
        if n_configs > MAX_DENSE_CONFIGS:
            mem_gb = n_configs ** 2 * 8 / 1e9
            raise MemoryError(
                f"matrix_elements_fast() refused to build {n_configs}×{n_configs} dense matrix "
                f"({mem_gb:.1f} GB). Use get_sparse_matrix_elements() + diagonal_elements_batch() "
                f"for systems with >{MAX_DENSE_CONFIGS} configs."
            )

        # Memory logging before dense allocation
        try:
            from utils.memory_logger import log_allocation
        except ImportError:
            try:
                from src.utils.memory_logger import log_allocation
            except ImportError:
                log_allocation = None
        dtype_str = "complex128" if self.h1e.is_complex() else "float64"
        if log_allocation:
            log_allocation("matrix_elements_fast", n_configs, dtype=dtype_str, layout="dense")

        H = torch.zeros(n_configs, n_configs, device=self.device, dtype=self.h1e.dtype)

        # Vectorized diagonal (already GPU-accelerated)
        H.diagonal().copy_(self.diagonal_elements_batch(configs))

        # Use overflow-safe integer encoding for hash lookups
        config_ints_cpu = self._config_int_hash(configs)
        config_hash = {config_ints_cpu[i]: i for i in range(n_configs)}

        # Get ALL connections at once using vectorized batch method
        all_connected, all_elements, batch_indices = self.get_connections_vectorized_batch(configs)

        if len(all_connected) > 0:
            # Encode connected configs (overflow-safe)
            connected_ints_cpu = self._config_int_hash(all_connected)
            batch_indices_cpu = batch_indices.cpu().tolist()  # Single transfer

            # B3: Vectorized scatter — build index arrays then assign at once
            row_indices = []
            col_indices = []
            values = []
            processed_pairs = set()

            for k in range(len(all_connected)):
                conn_int = connected_ints_cpu[k]
                if conn_int in config_hash:
                    i = config_hash[conn_int]
                    j = batch_indices_cpu[k]
                    if i != j:
                        pair = (min(i, j), max(i, j))
                        if pair not in processed_pairs:
                            processed_pairs.add(pair)
                            row_indices.append(i)
                            col_indices.append(j)
                            values.append(k)

            if row_indices:
                rows = torch.tensor(row_indices, device=self.device)
                cols = torch.tensor(col_indices, device=self.device)
                vals = all_elements[torch.tensor(values, device=self.device)]
                # Vectorized assignment — both triangles at once
                H[rows, cols] = vals
                H[cols, rows] = vals

        return H

    @torch.no_grad()
    def get_connections_parallel(
        self, configs: torch.Tensor, max_workers: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parallel computation of connections for multiple configurations.

        Uses ThreadPoolExecutor to process configs in parallel, providing
        significant speedup for large batches on multi-core CPUs.

        Args:
            configs: (n_configs, num_sites) configurations
            max_workers: Maximum number of parallel workers

        Returns:
            all_connected: (total_connections, num_sites) connected configs
            all_elements: (total_connections,) matrix elements
            config_indices: (total_connections,) which config each belongs to
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        device = self.device
        configs = configs.to(device)
        n_configs = len(configs)

        # Process configs in parallel
        def process_config(idx):
            connected, elements = self.get_connections(configs[idx])
            return idx, connected, elements

        all_connected = []
        all_elements = []
        all_indices = []

        # Use ThreadPool for parallel processing
        # Note: ThreadPool works well here because get_connections releases GIL during numpy ops
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_config, i): i for i in range(n_configs)}

            for future in as_completed(futures):
                idx, connected, elements = future.result()
                n_conn = len(connected)
                if n_conn > 0:
                    all_connected.append(connected.to(device))
                    all_elements.append(elements.to(device))
                    all_indices.append(
                        torch.full((n_conn,), idx, dtype=torch.long, device=device)
                    )

        if not all_connected:
            return (
                torch.empty(0, self.num_sites, device=device),
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.long, device=device)
            )

        return (
            torch.cat(all_connected, dim=0),
            torch.cat(all_elements, dim=0),
            torch.cat(all_indices, dim=0)
        )

    @torch.no_grad()
    def get_sparse_matrix_elements(
        self, configs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch compute off-diagonal connections for multiple configurations.

        Returns sparse COO format data for efficient matrix construction.
        Uses GPU-accelerated vectorized batch method for speed.

        Args:
            configs: (n_configs, num_sites) configurations

        Returns:
            (row_indices, col_indices, values) for sparse matrix
        """
        device = self.device
        configs = configs.to(device)
        n_configs = configs.shape[0]

        # Use vectorized batch method (GPU-accelerated)
        all_connected, all_elements, batch_indices = self.get_connections_vectorized_batch(configs)

        if len(all_connected) == 0:
            return (
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.float64, device=device),
            )

        # Overflow-safe integer encoding for fast lookup
        config_ints_cpu = self._config_int_hash(configs)
        config_hash = {config_ints_cpu[i]: i for i in range(n_configs)}

        # Encode connected configs (overflow-safe)
        connected_ints_cpu = self._config_int_hash(all_connected)
        batch_indices_cpu = batch_indices.cpu().tolist()

        all_rows = []
        all_cols = []
        val_indices = []

        for k in range(len(all_connected)):
            conn_int = connected_ints_cpu[k]
            if conn_int in config_hash:
                i = config_hash[conn_int]
                j = batch_indices_cpu[k]
                if i != j:
                    all_rows.append(i)
                    all_cols.append(j)
                    val_indices.append(k)

        if len(all_rows) == 0:
            return (
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.float64, device=device),
            )

        return (
            torch.tensor(all_rows, dtype=torch.long, device=device),
            torch.tensor(all_cols, dtype=torch.long, device=device),
            all_elements[torch.tensor(val_indices, device=device)],
        )

    def matrix_elements(
        self,
        configs_bra: torch.Tensor,
        configs_ket: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute matrix of elements H_ij = <x_i|H|x_j>.

        Uses fast path when bra == ket.
        """
        # Fast path for same bra/ket
        if (configs_bra.shape == configs_ket.shape and
            torch.all(configs_bra == configs_ket)):
            return self.matrix_elements_fast(configs_bra)

        # General case
        configs_bra = configs_bra.to(self.device)
        configs_ket = configs_ket.to(self.device)
        n_bra = configs_bra.shape[0]
        n_ket = configs_ket.shape[0]

        H = torch.zeros(n_bra, n_ket, device=self.device, dtype=self.h1e.dtype)

        # Build bra hash
        bra_hash = {tuple(configs_bra[i].cpu().tolist()): i
                    for i in range(n_bra)}

        for j in range(n_ket):
            config_j = configs_ket[j]
            key_j = tuple(config_j.cpu().tolist())

            # Diagonal
            if key_j in bra_hash:
                i = bra_hash[key_j]
                H[i, j] = self.diagonal_elements_batch(config_j.unsqueeze(0))[0]

            # Off-diagonal
            connected, elements = self.get_connections(config_j)
            if len(connected) > 0:
                for k in range(len(connected)):
                    key = tuple(connected[k].cpu().tolist())
                    if key in bra_hash:
                        i = bra_hash[key]
                        H[i, j] = elements[k]

        return H

    def to_pauli_strings(self) -> Tuple[List[float], List[str]]:
        """
        Convert molecular Hamiltonian to Pauli string representation.

        Uses Jordan-Wigner transformation to map fermionic operators to
        Pauli strings for CUDA-Q integration.

        Returns:
            (coefficients, pauli_words): Lists of coefficients and Pauli strings
        """
        n_qubits = self.num_sites
        coefficients = []
        pauli_words = []

        # Nuclear repulsion contributes to identity term
        coefficients.append(self.nuclear_repulsion)
        pauli_words.append("I" * n_qubits)

        # One-body terms
        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                h_pq = self.h1e[p, q].item()
                if abs(h_pq) < 1e-12:
                    continue

                for spin_offset in [0, self.n_orbitals]:
                    p_qubit = p + spin_offset
                    q_qubit = q + spin_offset

                    if p_qubit == q_qubit:
                        coefficients.append(h_pq / 2)
                        pauli_words.append("I" * n_qubits)

                        pauli = ["I"] * n_qubits
                        pauli[p_qubit] = "Z"
                        coefficients.append(-h_pq / 2)
                        pauli_words.append("".join(pauli))
                    else:
                        low, high = min(p_qubit, q_qubit), max(p_qubit, q_qubit)

                        pauli = ["I"] * n_qubits
                        pauli[p_qubit] = "X"
                        pauli[q_qubit] = "X"
                        for k in range(low + 1, high):
                            pauli[k] = "Z"
                        coefficients.append(h_pq / 2)
                        pauli_words.append("".join(pauli))

                        pauli = ["I"] * n_qubits
                        pauli[p_qubit] = "Y"
                        pauli[q_qubit] = "Y"
                        for k in range(low + 1, high):
                            pauli[k] = "Z"
                        coefficients.append(h_pq / 2)
                        pauli_words.append("".join(pauli))

        # Two-body terms (diagonal contributions)
        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                h_pqpq = self.h2e[p, p, q, q].item()
                if abs(h_pqpq) > 1e-12 and p != q:
                    for spin_offset in [0, self.n_orbitals]:
                        pauli = ["I"] * n_qubits
                        pauli[p + spin_offset] = "Z"
                        pauli[q + spin_offset] = "Z"
                        coefficients.append(h_pqpq / 8)
                        pauli_words.append("".join(pauli))

                    pauli = ["I"] * n_qubits
                    pauli[p] = "Z"
                    pauli[q + self.n_orbitals] = "Z"
                    coefficients.append(h_pqpq / 4)
                    pauli_words.append("".join(pauli))

        # Consolidate terms
        consolidated = {}
        for coeff, pauli in zip(coefficients, pauli_words):
            if pauli in consolidated:
                consolidated[pauli] += coeff
            else:
                consolidated[pauli] = coeff

        final_coeffs = []
        final_paulis = []
        for pauli, coeff in consolidated.items():
            if abs(coeff) > 1e-12:
                final_coeffs.append(coeff)
                final_paulis.append(pauli)

        return final_coeffs, final_paulis

    def get_hf_state(self) -> torch.Tensor:
        """
        Get Hartree-Fock reference state configuration.

        Returns the occupation pattern corresponding to the HF determinant.
        """
        config = torch.zeros(self.num_sites, dtype=torch.long, device=self.device)

        for i in range(self.n_alpha):
            config[i] = 1

        for i in range(self.n_beta):
            config[i + self.n_orbitals] = 1

        return config

    def to_sparse(self, device: str = "cpu"):
        """
        Convert to sparse CSR matrix representation.

        Optimized for molecular Hamiltonians using vectorized operations.

        Returns:
            scipy.sparse.csr_matrix
        """
        from scipy.sparse import csr_matrix

        n = self.hilbert_dim
        rows, cols, data = [], [], []

        # Generate all basis states
        basis = self._generate_all_configs(device)

        # Batch compute diagonal elements
        diag_values = self.diagonal_elements_batch(basis).cpu().numpy()

        for j in range(n):
            # Diagonal
            rows.append(j)
            cols.append(j)
            data.append(diag_values[j])

            # Off-diagonal connections
            config_j = basis[j]
            connected, elements = self.get_connections(config_j)

            if len(connected) > 0:
                for conn, elem in zip(connected, elements):
                    # Find index of connected config
                    i = self._config_to_index(conn)
                    rows.append(i)
                    cols.append(j)
                    data.append(elem.item() if hasattr(elem, 'item') else elem)

        return csr_matrix(
            (data, (rows, cols)),
            shape=(n, n),
            dtype=np.complex128
        )

    def exact_ground_state(
        self, device: str = "cpu"
    ) -> Tuple[float, torch.Tensor]:
        """
        Compute exact ground state energy by diagonalizing in particle-conserving subspace.

        This is MUCH faster than dense diagonalization in full Hilbert space:
        - Full space: O(4^n_orbitals)
        - Particle-conserving: O(C(n_orb, n_alpha) * C(n_orb, n_beta))

        Example speedups:
        - NH3 (16 qubits): 65,536 -> 3,136 (21x reduction)
        - N2 (20 qubits): 1,048,576 -> 14,400 (73x reduction)

        Returns:
            (ground_state_energy, ground_state_vector)
            Note: ground_state_vector is in full Hilbert space representation
        """
        fci_energy_val = self.fci_energy()

        # For small systems, also compute the ground state vector in full space
        if self.hilbert_dim <= 16384:  # Up to 14 qubits
            try:
                from scipy.sparse.linalg import eigsh
                H_sparse = self.to_sparse(device)
                eigenvalues, eigenvectors = eigsh(H_sparse, k=1, which="SA")
                psi0 = eigenvectors[:, 0]
                return fci_energy_val, torch.from_numpy(psi0).to(device)
            except Exception as e:
                print(f"WARNING: ground_state eigenvector computation failed: {e}")

        # For larger systems, return None for eigenvector
        return fci_energy_val, None

    def fci_energy(self, use_cache: bool = True) -> float:
        """
        Compute FCI (Full Configuration Interaction) energy.

        This computes FCI by diagonalizing the Hamiltonian in the
        particle-conserving subspace, which is equivalent to FCI and much
        faster than full Hilbert space diagonalization.

        IMPORTANT: Uses the same matrix_elements() function as the pipeline
        to ensure consistency between FCI reference and pipeline energy.

        Args:
            use_cache: If True, check/store FCI energy in disk cache.

        Returns:
            FCI ground state energy in Hartree
        """
        # Import cache module once
        _cache_mod = None
        _has_cache_meta = (
            use_cache
            and hasattr(self.integrals, '_geometry')
            and self.integrals._geometry is not None
        )
        if _has_cache_meta:
            try:
                try:
                    from utils import hamiltonian_cache as _cache_mod
                except ImportError:
                    from src.utils import hamiltonian_cache as _cache_mod
            except Exception:
                _has_cache_meta = False

        # Try loading from cache
        if _has_cache_meta:
            cached_fci = _cache_mod.load_fci_energy(
                self.integrals._geometry, self.integrals._basis,
                self.integrals._charge, self.integrals._spin,
            )
            if cached_fci is not None:
                print(f"[HamiltonianCache] Loaded FCI energy from disk cache: {cached_fci:.8f} Ha")
                return cached_fci

        import time
        from itertools import combinations

        n_orb = self.n_orbitals
        n_alpha = self.n_alpha
        n_beta = self.n_beta

        # Generate all valid determinants
        alpha_configs = list(combinations(range(n_orb), n_alpha))
        beta_configs = list(combinations(range(n_orb), n_beta))

        basis_configs = []
        for alpha_occ in alpha_configs:
            for beta_occ in beta_configs:
                config = torch.zeros(self.num_sites, dtype=torch.long)
                for i in alpha_occ:
                    config[i] = 1
                for i in beta_occ:
                    config[i + n_orb] = 1
                basis_configs.append(config)

        n_configs = len(basis_configs)
        print(f"Computing FCI energy in {n_configs} configuration subspace...")
        start_time = time.time()

        # Memory logging
        try:
            from utils.memory_logger import log_allocation, log_system_memory
        except ImportError:
            try:
                from src.utils.memory_logger import log_allocation, log_system_memory
            except ImportError:
                log_allocation = log_system_memory = None
        if log_allocation:
            log_system_memory("fci_energy start")

        # Stack configs into tensor
        basis_tensor = torch.stack(basis_configs).to(self.device)

        # Memory-safe path selection:
        # Dense n×n matrix = n² × 8 bytes (float64).
        # Threshold 5000: 5000² × 8 = 200 MB (safe).
        # Above that: use direct sparse construction → O(nnz) ≈ O(200n) memory.
        SPARSE_FCI_THRESHOLD = 5000

        if n_configs > SPARSE_FCI_THRESHOLD:
            # Sparse path: build CSR directly from get_sparse_matrix_elements()
            # For CAS(10,10) 63504 configs: ~150 MB instead of ~97 GB
            from scipy.sparse import coo_matrix, diags
            from scipy.sparse.linalg import eigsh

            print(f"  Using direct sparse construction ({n_configs} > {SPARSE_FCI_THRESHOLD})")
            mem_dense_gb = n_configs ** 2 * 8 / 1e9
            mem_sparse_est_mb = n_configs * 200 * 12 / 1e6  # ~200 nnz/row, 12 bytes each
            print(f"  Memory: sparse ~{mem_sparse_est_mb:.0f} MB vs dense ~{mem_dense_gb:.1f} GB")

            rows, cols, vals = self.get_sparse_matrix_elements(basis_tensor)
            rows_np = rows.cpu().numpy()
            cols_np = cols.cpu().numpy()
            vals_np = vals.cpu().numpy().astype(np.float64)

            H_coo = coo_matrix((vals_np, (rows_np, cols_np)), shape=(n_configs, n_configs))
            diag_np = self.diagonal_elements_batch(basis_tensor).cpu().numpy().astype(np.float64)
            H_csr = H_coo.tocsr()
            H_csr = 0.5 * (H_csr + H_csr.T)
            H_csr = H_csr + diags(diag_np, 0, shape=(n_configs, n_configs), format='csr')

            eigenvalues, _ = eigsh(H_csr, k=1, which='SA', tol=1e-12)
            fci_E = float(eigenvalues[0])
        else:
            # Dense path for small systems — exact and fast
            H_fci = self.matrix_elements(basis_tensor, basis_tensor)

            # Convert to numpy with float64 for numerical stability
            H_np = H_fci.cpu().numpy().astype(np.float64)

            # Ensure Hermitian symmetry (critical for correct eigenvalues)
            H_np = 0.5 * (H_np + H_np.T)

            # Verify Hermiticity
            asymmetry = np.abs(H_np - H_np.T).max()
            if asymmetry > 1e-10:
                print(f"WARNING: Hamiltonian asymmetry detected: {asymmetry:.2e}")

            if n_configs <= 2000:
                eigenvalues, _ = np.linalg.eigh(H_np)
                fci_E = float(eigenvalues[0])
            else:
                from scipy.sparse import csr_matrix
                from scipy.sparse.linalg import eigsh
                H_sparse = csr_matrix(H_np)
                eigenvalues, _ = eigsh(H_sparse, k=1, which='SA', tol=1e-12)
                fci_E = float(eigenvalues[0])

        elapsed = time.time() - start_time
        print(f"FCI energy: {fci_E:.8f} Ha (computed in {elapsed:.1f}s)")

        # Save to cache
        if _has_cache_meta:
            try:
                _cache_mod.save_fci_energy(
                    self.integrals._geometry, self.integrals._basis,
                    self.integrals._charge, self.integrals._spin,
                    fci_E,
                )
                print(f"[HamiltonianCache] Saved FCI energy to disk cache")
            except Exception as e:
                print(f"[HamiltonianCache] Failed to save FCI energy: {e}")

        return fci_E


def compute_molecular_integrals(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    use_cache: bool = True,
    cas: Optional[Tuple[int, int]] = None,
    casci: bool = False,
) -> MolecularIntegrals:
    """
    Compute molecular integrals using PySCF, with optional disk caching.

    Args:
        geometry: List of (atom_symbol, (x, y, z)) tuples
        basis: Basis set name
        charge: Molecular charge
        spin: 2S (number of unpaired electrons)
        use_cache: If True, check/store disk cache in ~/.cache/molecular-krylov/
        cas: Optional (nelecas, ncas) tuple for CAS active space selection.
            When provided, runs CASSCF (or CASCI if casci=True) after RHF
            and returns integrals in the active space only (n_orbitals = ncas).
        casci: If True and cas is not None, use CASCI instead of CASSCF.
            CASCI uses HF MOs directly (no orbital optimization), which is
            faster for large active spaces where CASSCF's iterative FCI solver
            is prohibitively expensive (e.g., CAS(10,15)+ with 9M+ configs).

    Returns:
        MolecularIntegrals object
    """
    try:
        from utils.hamiltonian_cache import load_integrals, save_integrals
    except ImportError:
        from src.utils.hamiltonian_cache import load_integrals, save_integrals

    # Try loading from cache first (skip cache for CAS — CASSCF optimization is not cached)
    if use_cache and cas is None:
        cached = load_integrals(geometry, basis, charge, spin)
        if cached is not None:
            print(f"[HamiltonianCache] Loaded integrals from disk cache")
            result = MolecularIntegrals(
                h1e=cached["h1e"],
                h2e=cached["h2e"],
                nuclear_repulsion=cached["nuclear_repulsion"],
                n_electrons=cached["n_electrons"],
                n_orbitals=cached["n_orbitals"],
                n_alpha=cached["n_alpha"],
                n_beta=cached["n_beta"],
            )
            result._geometry = list(geometry)
            result._basis = basis
            result._charge = charge
            result._spin = spin
            return result

    try:
        from pyscf import gto, scf, ao2mo
    except ImportError:
        raise ImportError("PySCF is required for molecular Hamiltonians")

    # Build molecule
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    # Enable symmetry to resolve degenerate orbital rotations (e.g., N2 pi orbitals).
    # PySCF uses Abelian subgroups (D2h etc.) which split degenerate irreps,
    # making MO coefficients deterministic across SCF runs.
    mol.symmetry = True
    mol.build()

    # Run HF to get orbitals
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.kernel()

    # CAS active space path
    if cas is not None:
        from pyscf import mcscf, fci

        nelecas, ncas = cas

        if casci:
            # CASCI: no orbital optimization, uses HF MOs directly.
            # Much faster for large active spaces where CASSCF's iterative
            # FCI solver is prohibitively expensive (e.g., CAS(10,15)+).
            mc = mcscf.CASCI(mf, ncas=ncas, nelecas=nelecas)
        else:
            mc = mcscf.CASSCF(mf, ncas=ncas, nelecas=nelecas)

        # Linear molecules (D∞h/C∞v) with symmetry=True can cause
        # PointGroupSymmetryError when CAS orbital selection breaks
        # an E-irrep pair. Use the non-symmetry FCI solver to avoid this.
        if mol.symmetry and mol.topgroup in ('Dooh', 'Coov'):
            mc.fcisolver = fci.direct_spin1.FCISolver(mol)

        # Compute config space size to decide if FCI is feasible.
        # For CASCI with >50M configs (e.g., CAS(10,20) = 240M), PySCF's
        # FCI solver is infeasible — skip kernel() and extract integrals only.
        from math import comb as _comb

        if isinstance(nelecas, (tuple, list)):
            _na, _nb = nelecas[0], nelecas[1]
        else:
            _na = (nelecas + spin) // 2
            _nb = (nelecas - spin) // 2
        _n_configs = _comb(ncas, _na) * _comb(ncas, _nb)
        _FCI_CONFIG_LIMIT = 50_000_000  # 50M configs

        if casci and _n_configs > _FCI_CONFIG_LIMIT:
            # Integrals-only mode: h1e_for_cas() and ao2mo work without
            # kernel() because CASCI sets mo_coeff = mf.mo_coeff at __init__.
            print(
                f"[CAS] Skipping FCI solve for CAS({nelecas},{ncas}): "
                f"{_n_configs:,} configs > {_FCI_CONFIG_LIMIT:,} limit. "
                f"Extracting integrals only."
            )
        else:
            mc.kernel()

            if not casci and not mc.converged:
                import warnings
                warnings.warn(
                    f"CASSCF did not converge for CAS({nelecas},{ncas}). "
                    "Integrals may be unreliable."
                )

        # h1e_for_cas returns (h1e_cas, e_core)
        # e_core = nuclear_repulsion + frozen core energy
        h1e_cas, e_core = mc.h1e_for_cas()

        # Two-electron integrals in active MO basis
        active_mo = mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
        h2e_cas = ao2mo.full(mol, active_mo)
        h2e_cas = ao2mo.restore(1, h2e_cas, ncas)

        h1e_cas = np.asarray(h1e_cas, dtype=np.float64)
        h2e_cas = np.asarray(h2e_cas, dtype=np.float64)

        # Active electron counts
        if isinstance(nelecas, (tuple, list)):
            n_alpha_cas = nelecas[0]
            n_beta_cas = nelecas[1]
            n_elec_cas = sum(nelecas)
        else:
            n_elec_cas = nelecas
            n_alpha_cas = (nelecas + spin) // 2
            n_beta_cas = (nelecas - spin) // 2

        # Do NOT set _geometry/_basis/_charge/_spin metadata — CAS integrals
        # depend on CASSCF orbital optimization which is non-deterministic,
        # so disk caching would be unsafe.
        return MolecularIntegrals(
            h1e=h1e_cas,
            h2e=h2e_cas,
            nuclear_repulsion=float(e_core),
            n_electrons=n_elec_cas,
            n_orbitals=ncas,
            n_alpha=n_alpha_cas,
            n_beta=n_beta_cas,
        )

    # Full (non-CAS) integral computation
    # Get integrals in MO basis
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff

    # Two-electron integrals
    h2e = ao2mo.kernel(mol, mf.mo_coeff)
    h2e = ao2mo.restore(1, h2e, mol.nao)  # Restore to 4-index tensor

    n_electrons = mol.nelectron
    n_orbitals = mol.nao
    n_alpha = (n_electrons + spin) // 2
    n_beta = (n_electrons - spin) // 2

    nuclear_repulsion = mol.energy_nuc()

    # Save to cache
    if use_cache:
        try:
            mol_hash = save_integrals(
                geometry, basis, charge, spin,
                h1e, h2e, nuclear_repulsion,
                n_electrons, n_orbitals, n_alpha, n_beta,
            )
            print(f"[HamiltonianCache] Saved integrals to disk cache ({mol_hash})")
        except Exception as e:
            print(f"[HamiltonianCache] Failed to save integrals: {e}")

    result = MolecularIntegrals(
        h1e=h1e,
        h2e=h2e,
        nuclear_repulsion=nuclear_repulsion,
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )
    result._geometry = list(geometry)
    result._basis = basis
    result._charge = charge
    result._spin = spin
    return result


def create_h2_hamiltonian(
    bond_length: float = 0.74,
    basis: str = "sto-3g",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MolecularHamiltonian:
    """Create H2 Hamiltonian at given bond length."""
    geometry = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, bond_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis=basis)
    return MolecularHamiltonian(integrals, device=device)


def create_lih_hamiltonian(
    bond_length: float = 1.6,
    basis: str = "sto-3g",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MolecularHamiltonian:
    """Create LiH Hamiltonian at given bond length."""
    geometry = [
        ("Li", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, bond_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis=basis)
    return MolecularHamiltonian(integrals, device=device)


def create_h2o_hamiltonian(
    oh_length: float = 0.96,
    angle: float = 104.5,
    basis: str = "sto-3g",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MolecularHamiltonian:
    """Create H2O Hamiltonian."""
    angle_rad = np.radians(angle)
    geometry = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (oh_length, 0.0, 0.0)),
        ("H", (oh_length * np.cos(angle_rad), oh_length * np.sin(angle_rad), 0.0)),
    ]
    integrals = compute_molecular_integrals(geometry, basis=basis)
    return MolecularHamiltonian(integrals, device=device)


def create_beh2_hamiltonian(
    bond_length: float = 1.33,
    basis: str = "sto-3g",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MolecularHamiltonian:
    """
    Create BeH2 (beryllium hydride) Hamiltonian.

    Linear molecule: H-Be-H
    6 electrons, ~7 orbitals in STO-3G
    Valid configs: C(7,3)² = 1,225
    """
    geometry = [
        ("Be", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, bond_length)),
        ("H", (0.0, 0.0, -bond_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis=basis)
    return MolecularHamiltonian(integrals, device=device)


def create_nh3_hamiltonian(
    nh_length: float = 1.01,
    hnh_angle: float = 107.8,
    basis: str = "sto-3g",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MolecularHamiltonian:
    """
    Create NH3 (ammonia) Hamiltonian.

    Pyramidal molecule with C3v symmetry.
    10 electrons, ~8 orbitals in STO-3G
    Valid configs: C(8,5)² = 3,136
    """
    # Place N at origin, H atoms in pyramidal arrangement
    angle_rad = np.radians(hnh_angle)
    # Height of N above H plane
    h = nh_length * np.cos(np.arcsin(np.sin(angle_rad/2) / np.sin(np.radians(60))))
    r = np.sqrt(nh_length**2 - h**2)  # Radius of H triangle

    geometry = [
        ("N", (0.0, 0.0, h)),
        ("H", (r, 0.0, 0.0)),
        ("H", (r * np.cos(np.radians(120)), r * np.sin(np.radians(120)), 0.0)),
        ("H", (r * np.cos(np.radians(240)), r * np.sin(np.radians(240)), 0.0)),
    ]
    integrals = compute_molecular_integrals(geometry, basis=basis)
    return MolecularHamiltonian(integrals, device=device)


def create_n2_hamiltonian(
    bond_length: float = 1.10,
    basis: str = "sto-3g",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MolecularHamiltonian:
    """
    Create N2 (nitrogen) Hamiltonian.

    Diatomic molecule with strong triple bond.
    14 electrons, ~10 orbitals in STO-3G
    Valid configs: C(10,7)² = 14,400

    This is a challenging strongly-correlated system.
    """
    geometry = [
        ("N", (0.0, 0.0, 0.0)),
        ("N", (0.0, 0.0, bond_length)),
    ]
    integrals = compute_molecular_integrals(geometry, basis=basis)
    return MolecularHamiltonian(integrals, device=device)


def create_ch4_hamiltonian(
    ch_length: float = 1.09,
    basis: str = "sto-3g",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MolecularHamiltonian:
    """
    Create CH4 (methane) Hamiltonian.

    Tetrahedral molecule with Td symmetry.
    10 electrons, ~9 orbitals in STO-3G
    Valid configs: C(9,5)² = 15,876
    """
    # Tetrahedral geometry
    # C at origin, H at vertices of tetrahedron
    a = ch_length / np.sqrt(3)  # Edge length relationship

    geometry = [
        ("C", (0.0, 0.0, 0.0)),
        ("H", (a, a, a)),
        ("H", (a, -a, -a)),
        ("H", (-a, a, -a)),
        ("H", (-a, -a, a)),
    ]
    integrals = compute_molecular_integrals(geometry, basis=basis)
    return MolecularHamiltonian(integrals, device=device)


def create_n2_cas_hamiltonian(
    bond_length: float = 1.10,
    basis: str = "cc-pvdz",
    cas: Tuple[int, int] = (10, 8),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MolecularHamiltonian:
    """
    Create N2 Hamiltonian with CAS (Complete Active Space) reduction.

    Runs CASSCF to select active orbitals, then returns a Hamiltonian
    over the CAS active space only.

    Default: CAS(10, 8) on cc-pVDZ gives 8 active orbitals and
    C(8,5)² = 3,136 configurations — small enough for exact FCI.

    Args:
        bond_length: N-N distance in Angstroms
        basis: Basis set name (default: cc-pvdz)
        cas: (nelecas, ncas) tuple for active space selection
        device: Computation device

    Returns:
        MolecularHamiltonian over the CAS active space
    """
    geometry = [
        ("N", (0.0, 0.0, 0.0)),
        ("N", (0.0, 0.0, bond_length)),
    ]
    # CASCI fallback for large active spaces where CASSCF's internal FCI
    # is too slow (e.g., CAS(10,15) has C(15,5)^2 = 9M+ configs).
    nelecas, ncas = cas
    use_casci = ncas >= 15
    integrals = compute_molecular_integrals(
        geometry, basis=basis, cas=cas, casci=use_casci
    )
    return MolecularHamiltonian(integrals, device=device)


def create_cr2_hamiltonian(
    bond_length: float = 1.68,
    basis: str = "sto-3g",
    cas: Optional[Tuple[int, int]] = (12, 12),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MolecularHamiltonian:
    """
    Create Cr2 (chromium dimer) Hamiltonian.

    Cr2 has a formal sextuple bond and is one of the most challenging
    multi-reference systems in quantum chemistry. The 3d electrons on
    each Cr atom create strong static correlation.

    Default: CAS(12,12) on STO-3G gives 12 active orbitals (3d + 4s)
    and C(12,6)^2 = 853,776 configurations -- 24 qubits.

    Args:
        bond_length: Cr-Cr distance in Angstroms (equilibrium ~1.68 A)
        basis: Basis set (default: sto-3g for method development)
        cas: (nelecas, ncas) for active space. None = full space.
        device: Computation device

    Notes:
        - SCF convergence can be difficult for Cr2; uses max_cycle=300.
        - CASSCF uses ``fix_spin_(ss=0)`` to target the singlet ground state.
          Without this constraint, CASSCF converges to the septet (S=3)
          state which is variationally lower but physically incorrect for
          the ground-state singlet with sextuple bond character.
        - CAS(12,12) captures 3d-3d metal bonding (6 electrons from each Cr:
          3d^5 + 4s^1 = 6 active electrons per atom, 12 total).
        - Cr2 is a classic benchmark for multi-reference methods due to
          strong static correlation from the near-degenerate 3d orbitals.
    """
    if cas is None:
        # Full-space (no CAS) -- delegate to compute_molecular_integrals
        geometry = [
            ("Cr", (0.0, 0.0, 0.0)),
            ("Cr", (0.0, 0.0, bond_length)),
        ]
        integrals = compute_molecular_integrals(geometry, basis=basis)
        return MolecularHamiltonian(integrals, device=device)

    # CAS path: custom CASSCF with fix_spin_ to target singlet ground state
    try:
        from pyscf import gto, scf, mcscf, fci, ao2mo
    except ImportError:
        raise ImportError("PySCF is required for Cr2 Hamiltonian")

    import warnings

    mol = gto.M(
        atom=f"Cr 0 0 0; Cr 0 0 {bond_length}",
        basis=basis,
        spin=0,
        symmetry=True,
        verbose=0,
    )

    mf = scf.RHF(mol)
    mf.max_cycle = 300
    mf.kernel()

    if not mf.converged:
        warnings.warn(
            f"RHF did not converge for Cr2 (bond_length={bond_length}, basis={basis}). "
            "Trying ROHF fallback."
        )
        mf = scf.ROHF(mol)
        mf.max_cycle = 300
        mf.kernel()
        if not mf.converged:
            warnings.warn("ROHF also did not converge for Cr2. Results may be unreliable.")

    nelecas, ncas = cas

    mc = mcscf.CASSCF(mf, ncas=ncas, nelecas=nelecas)

    # Linear molecules (D_inf_h / C_inf_v) need non-symmetry FCI solver
    if mol.symmetry and mol.topgroup in ("Dooh", "Coov"):
        mc.fcisolver = fci.direct_spin1.FCISolver(mol)

    # Cr2 CASSCF without spin constraint converges to septet (S=3, S^2=12)
    # instead of the singlet ground state. fix_spin_ adds a penalty to
    # enforce <S^2> = 0 (singlet).
    mc.fix_spin_(ss=0)
    mc.kernel()

    if not mc.converged:
        warnings.warn(
            f"CASSCF did not converge for Cr2 CAS({nelecas},{ncas}). "
            "Integrals may be unreliable."
        )

    # Extract active-space integrals
    h1e_cas, e_core = mc.h1e_for_cas()
    active_mo = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
    h2e_cas = ao2mo.full(mol, active_mo)
    h2e_cas = ao2mo.restore(1, h2e_cas, ncas)

    h1e_cas = np.asarray(h1e_cas, dtype=np.float64)
    h2e_cas = np.asarray(h2e_cas, dtype=np.float64)

    # Active electron counts
    if isinstance(nelecas, (tuple, list)):
        n_alpha_cas = nelecas[0]
        n_beta_cas = nelecas[1]
        n_elec_cas = sum(nelecas)
    else:
        n_elec_cas = nelecas
        n_alpha_cas = nelecas // 2
        n_beta_cas = nelecas // 2

    integrals = MolecularIntegrals(
        h1e=h1e_cas,
        h2e=h2e_cas,
        nuclear_repulsion=float(e_core),
        n_electrons=n_elec_cas,
        n_orbitals=ncas,
        n_alpha=n_alpha_cas,
        n_beta=n_beta_cas,
    )
    return MolecularHamiltonian(integrals, device=device)


def create_benzene_hamiltonian(
    basis: str = "sto-3g",
    cas: Optional[Tuple[int, int]] = (6, 15),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MolecularHamiltonian:
    """
    Create benzene (C6H6) Hamiltonian with CAS active space reduction.

    Benzene in D6h geometry (regular hexagon). Default active space is
    CAS(6,15) = 6 pi electrons in 15 orbitals (pi + sigma correlation).
    C(15,3)^2 = 455^2 = 207,025 configurations = 30 qubits.

    For CAS with ncas >= 15, automatically uses CASCI (no orbital
    optimization) since CASSCF's iterative FCI solver is prohibitively
    expensive at that scale.

    Args:
        basis: Basis set name (default: sto-3g)
        cas: Optional (nelecas, ncas) for active space. Default: (6, 15).
            Set to None for full-space (not recommended: 42e, 36o in STO-3G).
        device: Computation device

    Returns:
        MolecularHamiltonian over the CAS active space (or full space)
    """
    import math as _math

    # Regular hexagon: R(C-C) = 1.40 A, R(C-H) = 1.08 A
    cc_dist = 1.40
    ch_dist = 1.08

    geometry = []
    for i in range(6):
        angle = _math.pi / 3 * i
        cx = cc_dist * _math.cos(angle)
        cy = cc_dist * _math.sin(angle)
        geometry.append(("C", (cx, cy, 0.0)))
        hx = (cc_dist + ch_dist) * _math.cos(angle)
        hy = (cc_dist + ch_dist) * _math.sin(angle)
        geometry.append(("H", (hx, hy, 0.0)))

    # CASCI fallback for large active spaces
    use_casci = False
    if cas is not None:
        _, ncas = cas
        use_casci = ncas >= 15

    integrals = compute_molecular_integrals(
        geometry, basis=basis, cas=cas, casci=use_casci
    )
    return MolecularHamiltonian(integrals, device=device)
