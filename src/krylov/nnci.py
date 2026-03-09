"""Neural Network Configuration Importance (NNCI) Active Learning.

Implements the NNCI approach for selecting important electron configurations
for subspace diagonalization. The algorithm:

1. Start with known-important configs (e.g., HF + singles + doubles from CISD)
2. Train a small feedforward NN to predict configuration importance from CI coefficients
3. Generate candidate configs at higher excitation ranks (triples, quadruples, ...)
4. Use the trained NN to score candidates
5. Add top-K scored candidates to the basis
6. Re-diagonalize the Hamiltonian in the expanded basis
7. Repeat until convergence

This is a classical alternative to NF-based sampling for discovering important
configurations beyond CISD. The NN learns which orbital occupation patterns
correlate with large CI coefficients, enabling targeted exploration of the
exponentially large configuration space.

Key advantages over NF:
- Simpler architecture (feedforward NN vs normalizing flow)
- Directly uses CI coefficients as supervision signal
- Active learning loop provides iterative refinement
- No particle-conservation enforcement needed (candidates generated combinatorially)

Limitations:
- Requires initial basis with meaningful CI coefficients
- Candidate generation is combinatorial (may be expensive for high excitation ranks)
- NN generalization from CISD to triples/quadruples is an assumption

Reference: NNCI (NN classifier + active learning) approach from competitive landscape.

Usage:
    from krylov.nnci import NNCIActiveLearning, NNCIConfig
    from hamiltonians.molecular import create_lih_hamiltonian

    H = create_lih_hamiltonian()
    basis = generate_cisd_configs(H)

    config = NNCIConfig(max_iterations=5, top_k=20)
    nnci = NNCIActiveLearning(H, basis, config)
    results = nnci.run()
    print(f"Energy: {results['energy']:.8f} Ha")
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from itertools import combinations

try:
    from ..hamiltonians.base import Hamiltonian
except ImportError:
    from hamiltonians.base import Hamiltonian


@dataclass
class NNCIConfig:
    """Configuration for NNCI active learning.

    Parameters
    ----------
    max_iterations : int
        Maximum number of active learning iterations.
    top_k : int
        Number of top-scored candidates to add per iteration.
    hidden_dims : list[int]
        Hidden layer dimensions for the importance classifier.
    training_epochs : int
        Number of epochs to train the classifier per iteration.
    learning_rate : float
        Learning rate for classifier training.
    max_excitation_rank : int
        Maximum excitation rank for candidate generation.
        3 = up to triples, 4 = up to quadruples.
    max_basis_size : int
        Hard limit on basis size. If 0, no limit.
    max_candidates : int
        Maximum number of candidates to generate per iteration.
        0 means no limit (generate all possible).
    convergence_threshold : float
        Energy convergence threshold in Hartree. Stop if energy change
        between iterations is smaller than this.
    seed : int
        Random seed for reproducibility.
    """

    max_iterations: int = 5
    top_k: int = 20
    hidden_dims: list = field(default_factory=lambda: [128, 64])
    training_epochs: int = 100
    learning_rate: float = 1e-3
    max_excitation_rank: int = 4
    max_basis_size: int = 0
    max_candidates: int = 0
    convergence_threshold: float = 1e-6
    seed: int = 42


class ConfigImportanceClassifier(nn.Module):
    """Neural network that predicts configuration importance.

    Takes a binary configuration vector (n_sites,) and outputs a
    scalar importance score >= 0. The network uses ReLU activations
    in hidden layers and a Softplus output to ensure non-negativity.

    Parameters
    ----------
    n_sites : int
        Number of spin-orbital sites (= 2 * n_orbitals).
    hidden_dims : list[int]
        Sizes of hidden layers.

    Examples
    --------
    >>> clf = ConfigImportanceClassifier(n_sites=12, hidden_dims=[64, 32])
    >>> config = torch.zeros(1, 12)
    >>> score = clf(config)  # shape (1, 1), non-negative
    """

    def __init__(
        self,
        n_sites: int,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        in_dim = n_sites
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        # Output layer: single importance score
        layers.append(nn.Linear(in_dim, 1))
        # Softplus ensures non-negative output
        layers.append(nn.Softplus())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict importance scores for a batch of configurations.

        Parameters
        ----------
        x : torch.Tensor
            Binary configuration vectors, shape (batch, n_sites).

        Returns
        -------
        torch.Tensor
            Importance scores, shape (batch, 1), all >= 0.
        """
        return self.network(x.float())


class CandidateGenerator:
    """Generates candidate configurations via excitation operators.

    Produces higher-excitation configurations (triples, quadruples, etc.)
    from a reference state by applying combinations of occupied -> virtual
    orbital replacements. All generated candidates strictly conserve
    particle number (n_alpha, n_beta).

    Parameters
    ----------
    n_orbitals : int
        Number of spatial orbitals.
    n_alpha : int
        Number of alpha electrons.
    n_beta : int
        Number of beta electrons.
    """

    def __init__(self, n_orbitals: int, n_alpha: int, n_beta: int):
        self.n_orbitals = n_orbitals
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.n_sites = 2 * n_orbitals

    def generate_excitations(
        self,
        reference: torch.Tensor,
        max_rank: int = 4,
        min_rank: int = 1,
        exclude: Optional[torch.Tensor] = None,
        max_candidates: int = 0,
    ) -> torch.Tensor:
        """Generate excitation configurations from a reference state.

        Generates all combinations of alpha and beta excitations up to
        max_rank, filtering by particle conservation and uniqueness.

        Parameters
        ----------
        reference : torch.Tensor
            Reference configuration, shape (n_sites,).
        max_rank : int
            Maximum total excitation rank (alpha + beta excitations).
        min_rank : int
            Minimum total excitation rank.
        exclude : torch.Tensor, optional
            Configurations to exclude, shape (n_exclude, n_sites).
        max_candidates : int
            Maximum number of candidates to return. 0 = no limit.

        Returns
        -------
        torch.Tensor
            Generated candidate configurations, shape (n_candidates, n_sites).
        """
        n_orb = self.n_orbitals
        ref_np = reference.cpu().numpy()

        # Find occupied and virtual orbitals in reference
        occ_alpha = [i for i in range(n_orb) if ref_np[i] == 1]
        virt_alpha = [i for i in range(n_orb) if ref_np[i] == 0]
        occ_beta = [i for i in range(n_orb) if ref_np[i + n_orb] == 1]
        virt_beta = [i for i in range(n_orb) if ref_np[i + n_orb] == 0]

        candidates = []

        # Generate all (n_alpha_exc, n_beta_exc) partitions of rank
        for rank in range(min_rank, max_rank + 1):
            for n_a_exc in range(rank + 1):
                n_b_exc = rank - n_a_exc

                # Check feasibility
                if n_a_exc > min(len(occ_alpha), len(virt_alpha)):
                    continue
                if n_b_exc > min(len(occ_beta), len(virt_beta)):
                    continue

                # Generate alpha excitations
                if n_a_exc == 0:
                    alpha_excitations = [([], [])]
                else:
                    alpha_excitations = [
                        (list(holes), list(particles))
                        for holes in combinations(occ_alpha, n_a_exc)
                        for particles in combinations(virt_alpha, n_a_exc)
                    ]

                # Generate beta excitations
                if n_b_exc == 0:
                    beta_excitations = [([], [])]
                else:
                    beta_excitations = [
                        (list(holes), list(particles))
                        for holes in combinations(occ_beta, n_b_exc)
                        for particles in combinations(virt_beta, n_b_exc)
                    ]

                # Combine alpha and beta excitations
                for a_holes, a_parts in alpha_excitations:
                    for b_holes, b_parts in beta_excitations:
                        new_config = ref_np.copy()
                        # Apply alpha excitations
                        for h in a_holes:
                            new_config[h] = 0
                        for p in a_parts:
                            new_config[p] = 1
                        # Apply beta excitations
                        for h in b_holes:
                            new_config[h + n_orb] = 0
                        for p in b_parts:
                            new_config[p + n_orb] = 1
                        candidates.append(new_config)

                        # Early exit if max_candidates reached (before dedup)
                        if max_candidates > 0 and len(candidates) > max_candidates * 5:
                            break
                    if max_candidates > 0 and len(candidates) > max_candidates * 5:
                        break
                if max_candidates > 0 and len(candidates) > max_candidates * 5:
                    break

        if not candidates:
            return torch.empty(0, self.n_sites, dtype=torch.long)

        result = torch.from_numpy(np.array(candidates)).long()
        # Deduplicate
        result = torch.unique(result, dim=0)

        # Exclude existing basis configs
        if exclude is not None and len(exclude) > 0 and len(result) > 0:
            result = self._exclude_configs(result, exclude)

        # Apply max_candidates limit
        if max_candidates > 0 and len(result) > max_candidates:
            result = result[:max_candidates]

        return result

    def _exclude_configs(
        self, candidates: torch.Tensor, exclude: torch.Tensor
    ) -> torch.Tensor:
        """Remove configs from candidates that appear in exclude set."""
        # Build a set of tuples for O(1) lookup
        exclude_set = set()
        for i in range(len(exclude)):
            exclude_set.add(tuple(exclude[i].cpu().tolist()))

        keep_mask = torch.ones(len(candidates), dtype=torch.bool)
        for i in range(len(candidates)):
            if tuple(candidates[i].cpu().tolist()) in exclude_set:
                keep_mask[i] = False

        return candidates[keep_mask]


class NNCIActiveLearning:
    """Active learning loop for Neural Network Configuration Importance.

    Iteratively trains a classifier on CI coefficients, uses it to score
    candidate configurations, adds top-K candidates to the basis, and
    re-diagonalizes. Repeats until convergence or max iterations.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Molecular Hamiltonian for energy evaluation.
    initial_basis : torch.Tensor
        Initial set of configurations, shape (n_configs, n_sites).
    config : NNCIConfig
        Configuration for the active learning loop.

    Examples
    --------
    >>> from hamiltonians.molecular import create_lih_hamiltonian
    >>> H = create_lih_hamiltonian()
    >>> basis = generate_cisd_basis(H)
    >>> nnci = NNCIActiveLearning(H, basis, NNCIConfig(max_iterations=5))
    >>> results = nnci.run()
    >>> print(f"Energy: {results['energy']:.8f}")
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        initial_basis: torch.Tensor,
        config: Optional[NNCIConfig] = None,
    ):
        self.hamiltonian = hamiltonian
        self.basis = initial_basis.clone()
        self.config = config or NNCIConfig()
        self.n_sites = hamiltonian.num_sites

        # Extract molecular info
        self.n_orbitals = hamiltonian.n_orbitals
        self.n_alpha = hamiltonian.n_alpha
        self.n_beta = hamiltonian.n_beta

        # Candidate generator
        self.generator = CandidateGenerator(
            n_orbitals=self.n_orbitals,
            n_alpha=self.n_alpha,
            n_beta=self.n_beta,
        )

        # Reference state for excitation generation
        self.hf_state = hamiltonian.get_hf_state()

    def run(self) -> Dict[str, Any]:
        """Run the active learning loop.

        Returns
        -------
        dict
            Results containing:
            - energy: final ground state energy estimate
            - basis_size: final number of configurations
            - iterations: number of iterations completed
            - converged: whether convergence was achieved
            - energy_history: list of energies per iteration
            - final_basis: the final configuration tensor
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        energy_history = []
        prev_energy = float('inf')

        # Initial diagonalization
        current_energy, ci_coeffs = self._diagonalize(self.basis)
        energy_history.append(current_energy)
        converged = False

        for iteration in range(cfg.max_iterations):
            # 1. Train classifier on current CI coefficients
            clf = self._train_classifier(self.basis, ci_coeffs)

            # 2. Generate candidate configs (triples, quadruples, ...)
            candidates = self._generate_candidates()

            if len(candidates) == 0:
                # No new candidates available -- converged by exhaustion
                converged = True
                break

            # 3. Score candidates with the classifier
            scores = self._score_candidates(clf, candidates)

            # 4. Select top-K candidates
            n_to_add = min(cfg.top_k, len(candidates))

            # Respect max_basis_size
            if cfg.max_basis_size > 0:
                space_left = cfg.max_basis_size - len(self.basis)
                if space_left <= 0:
                    converged = False
                    break
                n_to_add = min(n_to_add, space_left)

            if n_to_add <= 0:
                break

            _, top_indices = torch.topk(scores, n_to_add)
            new_configs = candidates[top_indices].to(self.basis.device)

            # 5. Expand basis
            self.basis = torch.cat([self.basis, new_configs], dim=0)
            self.basis = torch.unique(self.basis, dim=0)

            # 6. Re-diagonalize
            current_energy, ci_coeffs = self._diagonalize(self.basis)
            energy_history.append(current_energy)

            # 7. Check convergence
            energy_change = abs(prev_energy - current_energy)
            if energy_change < cfg.convergence_threshold:
                converged = True
                prev_energy = current_energy
                break

            prev_energy = current_energy

        return {
            "energy": current_energy,
            "basis_size": len(self.basis),
            "iterations": len(energy_history),
            "converged": converged,
            "energy_history": energy_history,
            "final_basis": self.basis,
        }

    def _diagonalize(
        self, basis: torch.Tensor
    ) -> Tuple[float, np.ndarray]:
        """Diagonalize Hamiltonian in the given basis.

        Returns
        -------
        energy : float
            Ground state energy.
        ci_coeffs : np.ndarray
            Absolute CI coefficients (|c_i|) for each basis config.
        """
        H = self.hamiltonian.matrix_elements(basis, basis)
        H_np = H.cpu().numpy().astype(np.float64)
        H_np = 0.5 * (H_np + H_np.T)  # Symmetrize
        eigenvalues, eigenvectors = np.linalg.eigh(H_np)
        energy = float(eigenvalues[0])
        ci_coeffs = np.abs(eigenvectors[:, 0])
        return energy, ci_coeffs

    def _train_classifier(
        self, basis: torch.Tensor, ci_coeffs: np.ndarray
    ) -> ConfigImportanceClassifier:
        """Train the importance classifier on current CI coefficients.

        Parameters
        ----------
        basis : torch.Tensor
            Current basis configurations, shape (n_basis, n_sites).
        ci_coeffs : np.ndarray
            Absolute CI coefficients, shape (n_basis,).

        Returns
        -------
        ConfigImportanceClassifier
            Trained classifier.
        """
        cfg = self.config
        device = basis.device
        clf = ConfigImportanceClassifier(
            n_sites=self.n_sites,
            hidden_dims=cfg.hidden_dims,
        ).to(device)

        optimizer = torch.optim.Adam(clf.parameters(), lr=cfg.learning_rate)
        criterion = nn.MSELoss()

        configs_float = basis.float()
        labels = torch.from_numpy(ci_coeffs).float().unsqueeze(1).to(device)

        clf.train()
        for _ in range(cfg.training_epochs):
            optimizer.zero_grad()
            predictions = clf(configs_float)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

        clf.eval()
        return clf

    def _generate_candidates(self) -> torch.Tensor:
        """Generate candidate configs beyond the current basis.

        Generates excitations from the HF reference at ranks 3+
        (triples, quadruples, etc.), excluding configs already in the basis.

        Returns
        -------
        torch.Tensor
            Candidate configurations, shape (n_candidates, n_sites).
        """
        cfg = self.config
        max_cands = cfg.max_candidates if cfg.max_candidates > 0 else 0

        # Generate candidates from HF reference
        candidates = self.generator.generate_excitations(
            self.hf_state,
            max_rank=cfg.max_excitation_rank,
            min_rank=1,  # Include all ranks; exclude handles overlap
            exclude=self.basis,
            max_candidates=max_cands,
        )

        return candidates

    @torch.no_grad()
    def _score_candidates(
        self, clf: ConfigImportanceClassifier, candidates: torch.Tensor
    ) -> torch.Tensor:
        """Score candidate configs using the trained classifier.

        Parameters
        ----------
        clf : ConfigImportanceClassifier
            Trained importance classifier.
        candidates : torch.Tensor
            Candidate configurations, shape (n_candidates, n_sites).

        Returns
        -------
        torch.Tensor
            Importance scores, shape (n_candidates,).
        """
        device = next(clf.parameters()).device
        scores = clf(candidates.float().to(device))
        return scores.squeeze(1).cpu()
