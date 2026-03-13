"""MP2-based perturbative pruning for subspace reduction.

Uses MP2 double excitation amplitudes to rank configurations by importance.
Configs with larger |t2| contributions are more important for the ground state.

The key insight: MP2 amplitudes t2[i,j,a,b] quantify how much each double
excitation (occ_i, occ_j) -> (vir_a, vir_b) contributes to the correlation
energy. Configurations corresponding to large |t2| values are physically
more important and should be retained during basis pruning.

Scoring scheme:
  - HF (rank 0): essential_score (effectively infinite)
  - Singles (rank 1): essential_score (always important for orbital relaxation)
  - Doubles (rank 2): |t2[i,j,a,b]| from the MP2 amplitudes
  - Triples+ (rank 3+): product of constituent t2 amplitudes (approximate)

Reference: PIGen-SQD (IBM, 2025) -- perturbative importance generation.
"""

import numpy as np
import torch


def compute_mp2_amplitudes(hamiltonian) -> tuple[np.ndarray, float]:
    """Compute MP2 t2 amplitudes from molecular Hamiltonian integrals.

    Uses the MO-basis one- and two-electron integrals already stored in
    the Hamiltonian to compute MP2 amplitudes directly, without re-running
    PySCF. This works because the integrals are in canonical MO basis where
    the orbital energies are the diagonal elements of h1e.

    For closed-shell (RHF) systems:
        t2[i,j,a,b] = (ia|jb) / (eps_i + eps_j - eps_a - eps_b)
        E_MP2 = sum_{ijab} t2[i,j,a,b] * (2*(ia|jb) - (ib|ja))

    where (pq|rs) = h2e[p,q,r,s] in chemist's notation.

    Args:
        hamiltonian: MolecularHamiltonian with integrals

    Returns:
        t2: (nocc, nocc, nvir, nvir) MP2 double excitation amplitudes
        e_mp2_corr: MP2 correlation energy (should be negative)
    """
    integrals = hamiltonian.integrals
    h1e = integrals.h1e  # numpy (n_orb, n_orb)
    h2e = integrals.h2e  # numpy (n_orb, n_orb, n_orb, n_orb)
    if integrals.n_alpha != integrals.n_beta:
        raise ValueError(
            f"MP2 pruning requires closed-shell (RHF) system. "
            f"Got n_alpha={integrals.n_alpha}, n_beta={integrals.n_beta}"
        )

    # MP2 assumes canonical HF orbitals where the Fock matrix is diagonal.
    # CASSCF active-space orbitals are NOT eigenstates of the Fock operator,
    # so the Fock diagonal reconstruction gives wrong orbital energies.
    # Detect this by building the full Fock matrix and checking diagonality.
    h1e_f64 = h1e.astype(np.float64)
    h2e_f64_check = h2e.astype(np.float64)
    n_orb_check = integrals.n_orbitals
    n_occ_check = integrals.n_alpha
    fock = h1e_f64.copy()
    for p in range(n_orb_check):
        for q in range(n_orb_check):
            for j in range(n_occ_check):
                fock[p, q] += 2.0 * h2e_f64_check[p, q, j, j] - h2e_f64_check[p, j, j, q]
    fock_off_diag = fock - np.diag(np.diag(fock))
    max_fock_off_diag = float(np.max(np.abs(fock_off_diag)))
    if max_fock_off_diag > 1e-6:
        raise ValueError(
            f"MP2 pruning requires canonical (HF) orbitals where the Fock matrix "
            f"is diagonal. Fock off-diagonal elements up to {max_fock_off_diag:.2e}. "
            f"This typically happens with CASSCF active-space integrals."
        )

    n_occ = integrals.n_alpha  # For RHF, n_alpha == n_beta
    n_orb = integrals.n_orbitals
    n_vir = n_orb - n_occ

    h2e_f64 = h2e.astype(np.float64)

    # Reconstruct Fock diagonal: eps_p = h1e[p,p] + sum_j^occ (2*J_pj - K_pj)
    # This gives canonical orbital energies, not bare one-electron energies.
    eps = np.diag(h1e).copy().astype(np.float64)
    for p in range(n_orb):
        for j in range(n_occ):
            eps[p] += 2.0 * h2e_f64[p, p, j, j] - h2e_f64[p, j, j, p]

    # Build t2 amplitudes
    t2 = np.zeros((n_occ, n_occ, n_vir, n_vir), dtype=np.float64)

    for i in range(n_occ):
        for j in range(n_occ):
            for a in range(n_vir):
                for b in range(n_vir):
                    # Orbital indices in full MO space
                    a_full = a + n_occ
                    b_full = b + n_occ

                    # Energy denominator
                    denom = eps[i] + eps[j] - eps[a_full] - eps[b_full]

                    if abs(denom) < 1e-10:
                        continue

                    # (ia|jb) in chemist's notation
                    numerator = h2e_f64[i, a_full, j, b_full]
                    t2[i, j, a, b] = numerator / denom

    # MP2 correlation energy
    # E_MP2 = sum_{ijab} t2[i,j,a,b] * (2*(ia|jb) - (ib|ja))
    e_mp2_corr = 0.0
    for i in range(n_occ):
        for j in range(n_occ):
            for a in range(n_vir):
                for b in range(n_vir):
                    a_full = a + n_occ
                    b_full = b + n_occ
                    ia_jb = h2e_f64[i, a_full, j, b_full]  # (ia|jb)
                    ib_ja = h2e_f64[i, b_full, j, a_full]  # (ib|ja)
                    e_mp2_corr += t2[i, j, a, b] * (2.0 * ia_jb - ib_ja)

    return t2, float(e_mp2_corr)


def _classify_excitation(
    config: torch.Tensor,
    hf_state: torch.Tensor,
    n_orbitals: int,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    """Classify a configuration by excitation type relative to HF.

    Returns the excitation rank and lists of (hole, particle) pairs
    for alpha and beta spin channels separately.

    Args:
        config: (2*n_orbitals,) binary occupation vector
        hf_state: (2*n_orbitals,) HF reference occupation vector
        n_orbitals: number of spatial orbitals

    Returns:
        rank: excitation rank (0=HF, 1=single, 2=double, ...)
        alpha_excitations: list of (hole_idx, particle_idx) in alpha orbitals
        beta_excitations: list of (hole_idx, particle_idx) in beta orbitals
    """
    config_np = config.cpu().numpy()
    hf_np = hf_state.cpu().numpy()

    # Alpha channel: orbitals [0, n_orbitals)
    alpha_config = config_np[:n_orbitals]
    alpha_hf = hf_np[:n_orbitals]

    # Beta channel: orbitals [n_orbitals, 2*n_orbitals)
    beta_config = config_np[n_orbitals:]
    beta_hf = hf_np[n_orbitals:]

    # Find holes (occupied in HF, vacant in config) and particles (vice versa)
    alpha_holes = np.where((alpha_hf == 1) & (alpha_config == 0))[0].tolist()
    alpha_particles = np.where((alpha_hf == 0) & (alpha_config == 1))[0].tolist()
    beta_holes = np.where((beta_hf == 1) & (beta_config == 0))[0].tolist()
    beta_particles = np.where((beta_hf == 0) & (beta_config == 1))[0].tolist()

    alpha_excitations = list(zip(alpha_holes, alpha_particles, strict=True))
    beta_excitations = list(zip(beta_holes, beta_particles, strict=True))

    rank = len(alpha_excitations) + len(beta_excitations)
    return rank, alpha_excitations, beta_excitations


def _score_double_excitation(
    alpha_excitations: list[tuple[int, int]],
    beta_excitations: list[tuple[int, int]],
    t2: np.ndarray,
    n_occ: int,
) -> float:
    """Score a double excitation using MP2 t2 amplitudes.

    Handles three cases:
    - alpha-beta double: |t2[i, j, a-nocc, b-nocc]|
    - alpha-alpha double: |t2[i, j, a-nocc, b-nocc] - t2[i, j, b-nocc, a-nocc]|
    - beta-beta double: |t2[i, j, a-nocc, b-nocc] - t2[i, j, b-nocc, a-nocc]|

    Args:
        alpha_excitations: list of (hole, particle) in alpha orbitals
        beta_excitations: list of (hole, particle) in beta orbitals
        t2: (nocc, nocc, nvir, nvir) MP2 amplitudes
        n_occ: number of occupied orbitals per spin

    Returns:
        importance score (non-negative)
    """
    n_alpha_exc = len(alpha_excitations)
    n_beta_exc = len(beta_excitations)

    if n_alpha_exc == 1 and n_beta_exc == 1:
        # Alpha-beta double: i_alpha -> a_alpha, j_beta -> b_beta
        i, a = alpha_excitations[0]
        j, b = beta_excitations[0]
        return abs(t2[i, j, a - n_occ, b - n_occ])

    elif n_alpha_exc == 2 and n_beta_exc == 0:
        # Alpha-alpha double: (i, j) -> (a, b), antisymmetrized
        i, a = alpha_excitations[0]
        j, b = alpha_excitations[1]
        return abs(t2[i, j, a - n_occ, b - n_occ] - t2[i, j, b - n_occ, a - n_occ])

    elif n_alpha_exc == 0 and n_beta_exc == 2:
        # Beta-beta double: (i, j) -> (a, b), antisymmetrized
        i, a = beta_excitations[0]
        j, b = beta_excitations[1]
        return abs(t2[i, j, a - n_occ, b - n_occ] - t2[i, j, b - n_occ, a - n_occ])

    return 0.0


def _score_higher_excitation(
    alpha_excitations: list[tuple[int, int]],
    beta_excitations: list[tuple[int, int]],
    t2: np.ndarray,
    n_occ: int,
) -> float:
    """Score a triple or higher excitation using products of t2 amplitudes.

    Approximate the importance by decomposing the excitation into pairs and
    taking the product of the corresponding |t2| values. For a triple excitation
    (3 hole-particle pairs), we find the best pairing into one double + one single
    and use the t2 score of the double component.

    For a quadruple, we pair into two doubles and multiply their t2 scores.

    Args:
        alpha_excitations: list of (hole, particle) in alpha orbitals
        beta_excitations: list of (hole, particle) in beta orbitals
        t2: (nocc, nocc, nvir, nvir) MP2 amplitudes
        n_occ: number of occupied orbitals per spin

    Returns:
        importance score (non-negative)
    """
    all_excitations = [(h, p, "alpha") for h, p in alpha_excitations] + [
        (h, p, "beta") for h, p in beta_excitations
    ]
    n_exc = len(all_excitations)

    if n_exc < 2:
        return 0.0

    # Enumerate all pairs and compute max product of t2 amplitudes
    # For efficiency, just compute pairwise t2 scores and take products
    pair_scores = []
    for idx_a in range(n_exc):
        for idx_b in range(idx_a + 1, n_exc):
            h_a, p_a, spin_a = all_excitations[idx_a]
            h_b, p_b, spin_b = all_excitations[idx_b]

            # These two excitations form a "pseudo-double"
            if spin_a == "alpha" and spin_b == "beta":
                # alpha-beta pair
                score = abs(t2[h_a, h_b, p_a - n_occ, p_b - n_occ])
            elif spin_a == "beta" and spin_b == "alpha":
                score = abs(t2[h_b, h_a, p_b - n_occ, p_a - n_occ])
            elif spin_a == spin_b:
                # same-spin pair (antisymmetrized)
                score = abs(
                    t2[h_a, h_b, p_a - n_occ, p_b - n_occ] - t2[h_a, h_b, p_b - n_occ, p_a - n_occ]
                )
            else:
                score = 0.0

            pair_scores.append(score)

    if not pair_scores:
        return 0.0

    # For triples: use max pair score (the dominant double component)
    # For quadruples: use product of top-2 pair scores
    # General: geometric mean of top ceil(n_exc/2) pair scores
    pair_scores.sort(reverse=True)
    n_pairs_needed = n_exc // 2

    result = 1.0
    for k in range(min(n_pairs_needed, len(pair_scores))):
        result *= pair_scores[k]

    return result


def mp2_importance_scores(
    configs: torch.Tensor,
    hamiltonian,
    essential_score: float = 1e10,
) -> torch.Tensor:
    """Score configurations by MP2 importance.

    Scores reflect how much each configuration contributes to the
    ground-state correlation energy, based on MP2 perturbation theory:
    - HF and single excitations: essential_score (always retained)
    - Double excitations: |t2[i,j,a,b]| for the corresponding excitation
    - Higher excitations: product of constituent t2 amplitudes

    Args:
        configs: (n_configs, 2*n_orbitals) binary occupation vectors
        hamiltonian: MolecularHamiltonian with integrals
        essential_score: score assigned to HF + single excitations

    Returns:
        scores: (n_configs,) importance scores (non-negative)
    """
    n_configs = len(configs)
    n_orb = hamiltonian.n_orbitals
    n_occ = hamiltonian.n_alpha  # For RHF, n_alpha == n_beta
    hf_state = hamiltonian.get_hf_state()

    # Compute MP2 amplitudes
    t2, _ = compute_mp2_amplitudes(hamiltonian)

    scores = torch.zeros(n_configs, dtype=torch.float64)

    for idx in range(n_configs):
        config = configs[idx]
        rank, alpha_exc, beta_exc = _classify_excitation(config, hf_state, n_orb)

        if rank == 0:
            # HF configuration
            scores[idx] = essential_score
        elif rank == 1:
            # Single excitation — always essential
            scores[idx] = essential_score
        elif rank == 2:
            # Double excitation — score by |t2|
            scores[idx] = _score_double_excitation(alpha_exc, beta_exc, t2, n_occ)
        else:
            # Triple or higher — approximate with t2 products
            scores[idx] = _score_higher_excitation(alpha_exc, beta_exc, t2, n_occ)

    return scores


def prune_basis(
    configs: torch.Tensor,
    scores: torch.Tensor,
    max_configs: int,
    preserve_essential: bool = True,
) -> torch.Tensor:
    """Prune basis by keeping top-scored configurations.

    Essential configs (HF + singles, identified by score >= essential_score/2)
    are always preserved when preserve_essential=True. Remaining slots are
    filled by highest-scoring non-essential configs.

    Args:
        configs: (n_configs, 2*n_orbitals) configurations
        scores: (n_configs,) importance scores from mp2_importance_scores
        max_configs: maximum number of configs to keep
        preserve_essential: if True, always keep configs with score >= 1e9

    Returns:
        pruned_configs: (n_pruned, 2*n_orbitals) top-scored configurations
    """
    n_configs = len(configs)

    if n_configs <= max_configs:
        return configs.clone()

    essential_threshold = 1e9  # Anything above this is essential

    if preserve_essential:
        # Separate essential and non-essential
        essential_mask = scores >= essential_threshold
        nonessential_mask = ~essential_mask

        essential_indices = torch.where(essential_mask)[0]
        nonessential_indices = torch.where(nonessential_mask)[0]

        n_essential = len(essential_indices)

        if n_essential >= max_configs:
            # Even essentials exceed budget — keep all essentials
            # (this shouldn't happen in practice)
            return configs[essential_indices]

        # Fill remaining slots with top non-essential configs
        n_remaining = max_configs - n_essential
        nonessential_scores = scores[nonessential_indices]

        if len(nonessential_scores) <= n_remaining:
            # Keep all non-essential too
            return configs.clone()

        # Select top-scoring non-essential configs
        _, top_indices = torch.topk(nonessential_scores, n_remaining)
        selected_nonessential = nonessential_indices[top_indices]

        # Combine essential + top non-essential
        all_indices = torch.cat([essential_indices, selected_nonessential])
        return configs[all_indices]

    else:
        # Simple top-k by score
        _, top_indices = torch.topk(scores, min(max_configs, n_configs))
        return configs[top_indices]


# =============================================================================
# Natural Orbitals from MP2 1-RDM (P2.2)
# =============================================================================


def compute_mp2_rdm1(hamiltonian) -> np.ndarray:
    """Compute the MP2 unrelaxed one-particle density matrix in MO basis.

    The MP2 1-RDM is the HF density plus the MP2 correction:
        γ_pq = D^HF_pq + D^(2)_pq

    For closed-shell (RHF):
        D^HF_ij = 2δ_ij  (occupied block)
        D^(2)_ij = -Σ_{kab} (2T_ikab - T_ikba)(2T_jkab - T_jkba)
                         [using Hermitian symmetrized amplitude Tbar]
        D^(2)_ab = +Σ_{ijc} (2T_ijac - T_ijca)(2T_ijbc - T_ijcb)

    Reference: Szabo & Ostlund (1996), Chapter 6.

    Args:
        hamiltonian: MolecularHamiltonian with integrals (closed-shell only)

    Returns:
        rdm1: (n_orb, n_orb) one-particle density matrix in MO basis
    """
    t2, _ = compute_mp2_amplitudes(hamiltonian)
    n_occ, _, n_vir, _ = t2.shape
    n_orb = n_occ + n_vir

    rdm1 = np.zeros((n_orb, n_orb), dtype=np.float64)

    # HF contribution: occupied block = 2*I
    for i in range(n_occ):
        rdm1[i, i] = 2.0

    # MP2 correction: occupied-occupied block
    # D^(2)_ij = -Σ_{kab} Tbar_{ikab} * Tbar_{jkab}
    # where Tbar_{ikab} = 2*t2_{ikab} - t2_{ikba}
    tbar = 2.0 * t2 - t2.transpose(0, 1, 3, 2)  # (nocc, nocc, nvir, nvir)
    # D^(2)_ij = -Σ_{kab} tbar[i,k,a,b] * tbar[j,k,a,b]
    tbar_flat = tbar.reshape(n_occ, -1)  # (nocc, nocc*nvir*nvir)
    d2_oo = -tbar_flat @ tbar_flat.T  # (nocc, nocc)
    rdm1[:n_occ, :n_occ] += d2_oo

    # MP2 correction: virtual-virtual block
    # D^(2)_ab = +Σ_{ijc} tbar[i,j,a,c] * tbar[i,j,b,c]
    # Reshape: tbar has shape (nocc, nocc, nvir, nvir)
    # For each (a,b): sum over i,j,c of tbar[i,j,a,c]*tbar[i,j,b,c]
    tbar_vv = tbar.reshape(n_occ * n_occ, n_vir, n_vir)  # (nocc^2, nvir, nvir)
    # d2_vv[a,b] = Σ_{ij,c} tbar_vv[ij,a,c] * tbar_vv[ij,b,c]
    # Flatten: for each virtual a, collapse (ij, c) dimensions
    tbar_for_vv = tbar_vv.transpose(1, 0, 2).reshape(n_vir, -1)  # (nvir, nocc^2*nvir)
    d2_vv = tbar_for_vv @ tbar_for_vv.T  # (nvir, nvir)
    rdm1[n_occ:, n_occ:] += d2_vv

    return rdm1


def compute_natural_orbitals(hamiltonian) -> tuple[np.ndarray, np.ndarray]:
    """Compute natural orbitals from MP2 1-RDM.

    Natural orbitals (NOs) are eigenvectors of the 1-RDM. Their eigenvalues
    (occupation numbers) measure orbital importance for correlation:
    - n ≈ 2.0: doubly occupied, HF-like
    - n ≈ 0.0: empty, negligible correlation contribution
    - 0 << n << 2: strongly correlated, important for CI

    Orbitals are returned sorted by occupation number (descending).

    Args:
        hamiltonian: MolecularHamiltonian with integrals (closed-shell only)

    Returns:
        occ_numbers: (n_orb,) natural orbital occupation numbers, descending
        no_coeffs: (n_orb, n_orb) transformation matrix from MO to NO basis.
                   Each column is a natural orbital in MO basis.
    """
    rdm1 = compute_mp2_rdm1(hamiltonian)
    # Diagonalize: rdm1 = C @ diag(n) @ C^T
    occ_numbers, no_coeffs = np.linalg.eigh(rdm1)

    # Sort by descending occupation number (eigh returns ascending)
    idx = np.argsort(occ_numbers)[::-1]
    occ_numbers = occ_numbers[idx]
    no_coeffs = no_coeffs[:, idx]

    return occ_numbers, no_coeffs


def transform_integrals_to_no_basis(
    hamiltonian,
    no_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform h1e and h2e integrals from MO basis to Natural Orbital basis.

    Uses the 4-index transformation:
        h1e_NO[p,q] = Σ_{rs} C[r,p] * h1e[r,s] * C[s,q]
        h2e_NO[p,q,r,s] = Σ_{abcd} C[a,p] * C[b,q] * h2e[a,b,c,d] * C[c,r] * C[d,s]

    Args:
        hamiltonian: MolecularHamiltonian with integrals
        no_coeffs: (n_orb, n_orb) MO→NO transformation matrix

    Returns:
        h1e_no: (n_orb, n_orb) one-electron integrals in NO basis
        h2e_no: (n_orb, n_orb, n_orb, n_orb) two-electron integrals in NO basis
    """
    h1e = hamiltonian.integrals.h1e.astype(np.float64)
    h2e = hamiltonian.integrals.h2e.astype(np.float64)
    C = no_coeffs.astype(np.float64)

    # 1-electron: h1e_NO = C^T @ h1e @ C
    h1e_no = C.T @ h1e @ C

    # 2-electron: 4-index transform via sequential contractions
    # Step 1: h2e[a,b,c,d] × C[d,s] → tmp1[a,b,c,s]
    tmp1 = np.einsum("abcd,ds->abcs", h2e, C)
    # Step 2: tmp1[a,b,c,s] × C[c,r] → tmp2[a,b,r,s]
    tmp2 = np.einsum("abcs,cr->abrs", tmp1, C)
    # Step 3: tmp2[a,b,r,s] × C[b,q] → tmp3[a,q,r,s]
    tmp3 = np.einsum("abrs,bq->aqrs", tmp2, C)
    # Step 4: tmp3[a,q,r,s] × C[a,p] → h2e_NO[p,q,r,s]
    h2e_no = np.einsum("aqrs,ap->pqrs", tmp3, C)

    return h1e_no, h2e_no


def no_orbital_importance(occ_numbers: np.ndarray) -> np.ndarray:
    """Rank orbitals by correlation importance from NO occupation numbers.

    Importance = min(n, 2-n), measuring deviation from fully occupied/empty.
    Higher values = more correlated = more important for excitation generation.

    Args:
        occ_numbers: (n_orb,) natural orbital occupation numbers

    Returns:
        importance: (n_orb,) correlation importance scores
    """
    return np.minimum(occ_numbers, 2.0 - occ_numbers)
