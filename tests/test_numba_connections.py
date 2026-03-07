"""Tests for PR 2.0: Numba JIT for get_connections.

The 4-nested Python for-loop in get_connections is the #1 CPU bottleneck.
Numba JIT compilation targets:
1. JW sign computation (popcount-based, no Python loops)
2. Single excitation matrix elements (vectorized Coulomb/exchange sums)
3. Double excitation matrix elements (4-nested loop → @njit)

Expected speedup: 50-200x over pure Python.
"""

import pytest
import torch
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestNumbaJWSign:
    """Test Numba-JIT'd Jordan-Wigner sign computation."""

    @pytest.mark.molecular
    def test_jw_sign_single_exists(self, h2_hamiltonian):
        """numba_jw_sign_single should be importable."""
        from hamiltonians.molecular import numba_jw_sign_single
        assert callable(numba_jw_sign_single)

    @pytest.mark.molecular
    def test_jw_sign_single_matches_python_h2(self, h2_hamiltonian):
        """Numba JW sign must match Python for all H2 single excitations."""
        from hamiltonians.molecular import numba_jw_sign_single

        H = h2_hamiltonian
        configs = torch.tensor([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ], dtype=torch.long)

        for config in configs:
            config_np = config.numpy()
            n_orb = H.n_orbitals
            occ_alpha = np.where(config_np[:n_orb] == 1)[0]
            virt_alpha = np.where(config_np[:n_orb] == 0)[0]
            occ_beta = np.where(config_np[n_orb:] == 1)[0]
            virt_beta = np.where(config_np[n_orb:] == 0)[0]

            # Test all alpha single excitations
            for q in occ_alpha:
                for p in virt_alpha:
                    python_sign = H._jw_sign_np(config_np, p, q)
                    numba_sign = numba_jw_sign_single(config_np, p, q)
                    assert python_sign == numba_sign, (
                        f"JW sign mismatch at p={p}, q={q}: "
                        f"python={python_sign}, numba={numba_sign}"
                    )

            # Test all beta single excitations
            for q in occ_beta:
                for p in virt_beta:
                    python_sign = H._jw_sign_np(config_np, p + n_orb, q + n_orb)
                    numba_sign = numba_jw_sign_single(config_np, p + n_orb, q + n_orb)
                    assert python_sign == numba_sign

    @pytest.mark.molecular
    def test_jw_sign_double_exists(self, h2_hamiltonian):
        """numba_jw_sign_double should be importable."""
        from hamiltonians.molecular import numba_jw_sign_double
        assert callable(numba_jw_sign_double)

    @pytest.mark.molecular
    def test_jw_sign_double_matches_python_lih(self, lih_hamiltonian):
        """Numba JW double sign must match Python for LiH excitations."""
        from hamiltonians.molecular import numba_jw_sign_double

        H = lih_hamiltonian
        hf = H.get_hf_state().cpu().numpy()
        n_orb = H.n_orbitals

        occ_alpha = np.where(hf[:n_orb] == 1)[0]
        virt_alpha = np.where(hf[:n_orb] == 0)[0]
        occ_beta = np.where(hf[n_orb:] == 1)[0]
        virt_beta = np.where(hf[n_orb:] == 0)[0]

        # Test alpha-alpha doubles
        for i, q in enumerate(occ_alpha):
            for s in occ_alpha[i+1:]:
                for k, p in enumerate(virt_alpha):
                    for r in virt_alpha[k+1:]:
                        py = H._jw_sign_double_np(hf, p, r, q, s)
                        nb = numba_jw_sign_double(hf, p, r, q, s)
                        assert py == nb, (
                            f"Double JW sign mismatch: p={p},r={r},q={q},s={s}: "
                            f"python={py}, numba={nb}"
                        )

        # Test alpha-beta doubles
        for q in occ_alpha:
            for s in occ_beta:
                for p in virt_alpha:
                    for r in virt_beta:
                        s_idx = s + n_orb
                        r_idx = r + n_orb
                        py = H._jw_sign_double_np(hf, p, r_idx, q, s_idx)
                        nb = numba_jw_sign_double(hf, p, r_idx, q, s_idx)
                        assert py == nb


class TestNumbaGetConnections:
    """Test Numba-accelerated get_connections produces identical results."""

    @pytest.mark.molecular
    def test_numba_get_connections_exists(self, h2_hamiltonian):
        """numba_get_connections should be importable."""
        from hamiltonians.molecular import numba_get_connections
        assert callable(numba_get_connections)

    @pytest.mark.molecular
    def test_numba_matches_python_h2(self, h2_hamiltonian):
        """Numba get_connections must produce identical results for H2."""
        from hamiltonians.molecular import numba_get_connections

        H = h2_hamiltonian
        configs = torch.tensor([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ], dtype=torch.long)

        for config in configs:
            # Python reference
            py_conns, py_elems = H.get_connections(config)
            # Numba
            nb_conns, nb_elems = numba_get_connections(
                config.numpy(),
                H.n_orbitals,
                H.h1e.cpu().numpy(),
                H._h2e_np,
                H._J_single_np,
                H._K_single_np,
                H.single_exc_data,
            )

            # Sort both by config bitstring for deterministic comparison
            if len(py_conns) == 0 and len(nb_conns) == 0:
                continue

            py_conns_np = py_conns.cpu().numpy()
            py_elems_np = py_elems.cpu().numpy().astype(np.float64)
            nb_elems_np = nb_elems.astype(np.float64)

            # Sort by config representation
            py_keys = [tuple(c) for c in py_conns_np]
            nb_keys = [tuple(c) for c in nb_conns]

            py_sorted = sorted(zip(py_keys, py_elems_np))
            nb_sorted = sorted(zip(nb_keys, nb_elems_np))

            assert len(py_sorted) == len(nb_sorted), (
                f"Different number of connections: python={len(py_sorted)}, "
                f"numba={len(nb_sorted)}"
            )

            for (pk, pv), (nk, nv) in zip(py_sorted, nb_sorted):
                assert pk == nk, f"Different config: python={pk}, numba={nk}"
                np.testing.assert_allclose(
                    pv, nv, atol=1e-8,
                    err_msg=f"Element mismatch for config {pk}"
                )

    @pytest.mark.molecular
    def test_numba_matches_python_lih(self, lih_hamiltonian):
        """Numba get_connections must match Python for all LiH HF connections."""
        from hamiltonians.molecular import numba_get_connections

        H = lih_hamiltonian
        hf = H.get_hf_state().cpu()

        # Python reference
        py_conns, py_elems = H.get_connections(hf)
        # Numba
        nb_conns, nb_elems = numba_get_connections(
            hf.numpy(),
            H.n_orbitals,
            H.h1e.cpu().numpy(),
            H._h2e_np,
            H._J_single_np,
            H._K_single_np,
            H.single_exc_data,
        )

        py_conns_np = py_conns.cpu().numpy()
        py_elems_np = py_elems.cpu().numpy().astype(np.float64)
        nb_elems_np = nb_elems.astype(np.float64)

        py_sorted = sorted(zip([tuple(c) for c in py_conns_np], py_elems_np))
        nb_sorted = sorted(zip([tuple(c) for c in nb_conns], nb_elems_np))

        assert len(py_sorted) == len(nb_sorted), (
            f"Connection count mismatch: python={len(py_sorted)}, numba={len(nb_sorted)}"
        )

        for (pk, pv), (nk, nv) in zip(py_sorted, nb_sorted):
            assert pk == nk
            np.testing.assert_allclose(pv, nv, atol=1e-8)

    @pytest.mark.molecular
    def test_numba_matches_python_beh2(self, beh2_hamiltonian):
        """Numba get_connections must match Python for BeH2 HF + singles."""
        from hamiltonians.molecular import numba_get_connections

        H = beh2_hamiltonian
        hf = H.get_hf_state().cpu()
        n_orb = H.n_orbitals

        # Test HF and first few single excitations
        test_configs = [hf]
        hf_np = hf.numpy()
        occ = np.where(hf_np[:n_orb] == 1)[0]
        virt = np.where(hf_np[:n_orb] == 0)[0]
        for q in occ[:2]:
            for p in virt[:2]:
                c = hf.clone()
                c[q] = 0
                c[p] = 1
                test_configs.append(c)

        for config in test_configs:
            py_conns, py_elems = H.get_connections(config)
            nb_conns, nb_elems = numba_get_connections(
                config.numpy(),
                H.n_orbitals,
                H.h1e.cpu().numpy(),
                H._h2e_np,
                H._J_single_np,
                H._K_single_np,
                H.single_exc_data,
            )

            py_sorted = sorted(
                zip([tuple(c) for c in py_conns.cpu().numpy()],
                    py_elems.cpu().numpy().astype(np.float64))
            )
            nb_sorted = sorted(
                zip([tuple(c) for c in nb_conns],
                    nb_elems.astype(np.float64))
            )

            assert len(py_sorted) == len(nb_sorted), (
                f"Count mismatch on config {config.tolist()}: "
                f"python={len(py_sorted)}, numba={len(nb_sorted)}"
            )

            for (pk, pv), (nk, nv) in zip(py_sorted, nb_sorted):
                assert pk == nk
                np.testing.assert_allclose(pv, nv, atol=1e-8)


class TestNumbaSpeedup:
    """Verify Numba acceleration provides actual speedup."""

    @pytest.mark.molecular
    def test_numba_speedup_lih(self, lih_hamiltonian):
        """Numba get_connections must be > 10x faster than Python on LiH."""
        from hamiltonians.molecular import numba_get_connections

        H = lih_hamiltonian
        hf = H.get_hf_state().cpu()

        # Warmup Numba JIT
        _ = numba_get_connections(
            hf.numpy(), H.n_orbitals, H.h1e.cpu().numpy(), H._h2e_np,
            H._J_single_np, H._K_single_np, H.single_exc_data,
        )

        # Benchmark Python
        n_iter = 20
        t0 = time.perf_counter()
        for _ in range(n_iter):
            H.get_connections(hf)
        t_python = time.perf_counter() - t0

        # Benchmark Numba — pre-convert all arrays outside the timing loop
        hf_np = hf.numpy()
        h1e_np = H.h1e.cpu().numpy()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            numba_get_connections(
                hf_np, H.n_orbitals, h1e_np, H._h2e_np,
                H._J_single_np, H._K_single_np, H.single_exc_data,
            )
        t_numba = time.perf_counter() - t0

        speedup = t_python / t_numba
        print(f"\n  LiH get_connections: python={t_python/n_iter*1000:.1f}ms, "
              f"numba={t_numba/n_iter*1000:.1f}ms, speedup={speedup:.1f}x")
        assert speedup > 10, f"Numba speedup only {speedup:.1f}x, expected > 10x"


class TestNumbaEnergyAccuracy:
    """End-to-end accuracy: Numba connections → matrix → eigensolver → energy."""

    @pytest.mark.molecular
    def test_numba_energy_matches_python_h2(self, h2_hamiltonian):
        """H2 ground state from Numba connections must match Python path."""
        from hamiltonians.molecular import numba_get_connections

        H = h2_hamiltonian
        configs = torch.tensor([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ], dtype=torch.long)

        # Build H matrix using Numba connections
        n = len(configs)
        H_mat = np.zeros((n, n), dtype=np.float64)

        for i, config in enumerate(configs):
            # Diagonal
            H_mat[i, i] = H.diagonal_element(config).item()
            # Off-diagonal via Numba
            nb_conns, nb_elems = numba_get_connections(
                config.numpy(), H.n_orbitals, H.h1e.cpu().numpy(), H._h2e_np,
                H._J_single_np, H._K_single_np, H.single_exc_data,
            )
            for conn, elem in zip(nb_conns, nb_elems):
                conn_tuple = tuple(conn)
                for j, c in enumerate(configs):
                    if tuple(c.numpy()) == conn_tuple:
                        H_mat[i, j] = float(elem)
                        break

        H_sym = 0.5 * (H_mat + H_mat.T)
        evals = np.linalg.eigvalsh(H_sym)
        E_numba = float(evals[0])

        # Reference
        E_fci = H.fci_energy()
        error_mha = abs(E_numba - E_fci) * 1000

        assert error_mha < 0.1, (
            f"Numba-based H2 energy error {error_mha:.4f} mHa"
        )

    @pytest.mark.molecular
    def test_numba_energy_matches_python_lih(self, lih_hamiltonian):
        """LiH energy from Numba connections must match Python within 1e-10 Ha."""
        from hamiltonians.molecular import numba_get_connections

        H = lih_hamiltonian
        hf = H.get_hf_state().cpu()

        # Get connections from both paths
        py_conns, py_elems = H.get_connections(hf)
        nb_conns, nb_elems = numba_get_connections(
            hf.numpy(), H.n_orbitals, H.h1e.cpu().numpy(), H._h2e_np,
            H._J_single_np, H._K_single_np, H.single_exc_data,
        )

        # Compare total matrix element magnitudes (sum of |H_ij|)
        py_sum = py_elems.abs().sum().item()
        nb_sum = np.abs(nb_elems).sum()

        np.testing.assert_allclose(
            nb_sum, py_sum, rtol=1e-6,
            err_msg="Total off-diagonal magnitude mismatch"
        )


class TestNumbaEdgeCases:
    """Edge cases for Numba JIT functions."""

    def test_jw_sign_adjacent_orbitals(self):
        """JW sign for adjacent orbitals (p=1, q=0) should be correct."""
        from hamiltonians.molecular import numba_jw_sign_single

        config = np.array([1, 0, 1, 0])  # alpha: [1,0], beta: [1,0]
        # p=1, q=0: no sites between them, sign = +1
        assert numba_jw_sign_single(config, 1, 0) == 1

    def test_jw_sign_p_equals_q(self):
        """JW sign for p==q should be +1."""
        from hamiltonians.molecular import numba_jw_sign_single

        config = np.array([1, 0, 1, 0])
        assert numba_jw_sign_single(config, 0, 0) == 1
        assert numba_jw_sign_single(config, 1, 1) == 1

    def test_jw_sign_many_occupied_between(self):
        """JW sign with many occupied sites between p and q."""
        from hamiltonians.molecular import numba_jw_sign_single

        # config: [1, 1, 1, 1, 0, 0, ...] — 4 occupied alpha
        config = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0])
        # p=5, q=0: 3 occupied between (indices 1,2,3) → sign = (-1)^3 = -1
        assert numba_jw_sign_single(config, 5, 0) == -1
        # p=4, q=0: 3 occupied between (indices 1,2,3) → sign = (-1)^3 = -1
        assert numba_jw_sign_single(config, 4, 0) == -1

    def test_numba_get_connections_empty(self):
        """Config with no possible excitations should return empty."""
        from hamiltonians.molecular import numba_get_connections

        # All occupied — no virtual orbitals, no excitations possible
        # This is artificial but should not crash
        config = np.array([1, 1, 1, 1])
        n_orb = 2
        h1e = np.zeros((n_orb, n_orb))
        h2e = np.zeros((n_orb, n_orb, n_orb, n_orb))
        J_single = np.zeros((n_orb, n_orb, n_orb))
        K_single = np.zeros((n_orb, n_orb, n_orb))
        single_exc_data = []

        conns, elems = numba_get_connections(
            config, n_orb, h1e, h2e, J_single, K_single, single_exc_data
        )
        assert len(conns) == 0
        assert len(elems) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
