"""
active_loop.py
--------------
Closed-loop Bayesian active learning engine for HEA grain-boundary discovery.

Implements the core loop:

  Generate -> Evaluate -> Select -> Oracle -> Retrain -> Repeat

The loop is oracle-agnostic: any callable f(Z) -> E can serve as the oracle
(held-out DFT dataset, live DFT calculation, MD trajectory, or experiment).

Usage
-----
  from active_loop import ActiveLearningLoop, DFTDatasetOracle
  from gb_vae import GBVAE
  from gp_surrogate import GPSurrogate

  # Setup
  oracle = DFTDatasetOracle(Z_holdout, E_holdout)
  loop   = ActiveLearningLoop(vae=model, gp=gp, oracle=oracle,
                               latent_dim=16, n_candidates=500,
                               batch_size=5, acquisition='EI')

  # Seed with a few known configurations
  loop.seed(Z_seed, E_seed)

  # Run
  results = loop.run(n_iterations=20)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ── Oracle wrappers ────────────────────────────────────────────────────────

class DFTDatasetOracle:
    """
    Simulates a DFT oracle using a held-out dataset.
    At each query, finds the dataset point nearest to the queried latent
    vector and returns its known energy. Mimics the real scenario where
    DFT evaluates the structure closest to the proposed configuration.

    Parameters
    ----------
    Z_pool : np.ndarray, shape [N, latent_dim]  - pool of known latent vectors
    E_pool : np.ndarray, shape [N]              - corresponding DFT energies
    """

    def __init__(self, Z_pool: np.ndarray, E_pool: np.ndarray):
        self.Z_pool = Z_pool
        self.E_pool = E_pool
        self._queried_indices = set()

    def query(self, Z_query: np.ndarray) -> tuple:
        """
        For each row in Z_query, return (Z_matched, E_matched).
        Uses nearest-neighbour lookup in latent space.
        Tracks which pool points have been queried (no double-counting).
        """
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(Z_query, self.Z_pool)

        # Mask already-queried points
        for idx in self._queried_indices:
            D[:, idx] = np.inf

        best_idx = np.argmin(D, axis=1)
        self._queried_indices.update(best_idx.tolist())

        return self.Z_pool[best_idx], self.E_pool[best_idx]

    @property
    def n_remaining(self) -> int:
        return len(self.Z_pool) - len(self._queried_indices)

    @property
    def fraction_low_energy_discovered(self) -> float:
        """Fraction of bottom-10% energies discovered so far."""
        threshold = np.percentile(self.E_pool, 10)
        low_e_idx = set(np.where(self.E_pool <= threshold)[0])
        discovered = low_e_idx & self._queried_indices
        return len(discovered) / max(1, len(low_e_idx))


# ── Baselines ──────────────────────────────────────────────────────────────

def random_acquisition(Z_candidates: np.ndarray, **kwargs) -> np.ndarray:
    """Uniform random selection (baseline)."""
    return np.random.rand(len(Z_candidates))


def uncertainty_acquisition(Z_candidates: np.ndarray, gp, **kwargs) -> np.ndarray:
    """Pure uncertainty sampling (exploitation-free baseline)."""
    return gp.bald(Z_candidates)


# ── Results container ──────────────────────────────────────────────────────

@dataclass
class LoopResults:
    iterations:           list = field(default_factory=list)
    best_energy_history:  list = field(default_factory=list)
    gp_variance_history:  list = field(default_factory=list)
    n_queries_history:    list = field(default_factory=list)
    frac_low_e_history:   list = field(default_factory=list)
    Z_discovered:         Optional[np.ndarray] = None
    E_discovered:         Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = [
            "Active Learning Loop Summary",
            "=" * 40,
            f"Total iterations : {len(self.iterations)}",
            f"Total queries    : {self.n_queries_history[-1] if self.n_queries_history else 0}",
            f"Best energy found: {min(self.best_energy_history):.4f} eV/atom"
            if self.best_energy_history else "",
            f"Low-E discovery  : {self.frac_low_e_history[-1]*100:.1f}%"
            if self.frac_low_e_history else "",
        ]
        return "\n".join(lines)


# ── Main loop ──────────────────────────────────────────────────────────────

class ActiveLearningLoop:
    """
    Closed-loop active learning engine.

    Parameters
    ----------
    vae          : GBVAE  (or None to use latent_dim Gaussian sampling)
    gp           : GPSurrogate
    oracle       : object with .query(Z) -> (Z_matched, E_matched)
    latent_dim   : int
    n_candidates : int   - candidates sampled from VAE prior per iteration
    batch_size   : int   - top-k candidates selected per iteration
    acquisition  : str   - 'EI' | 'UCB' | 'BALD' | 'random' | 'uncertainty'
    xi           : float - EI exploration bonus
    kappa        : float - UCB exploration weight
    convergence_tol : float - stop if GP variance sum falls below this
    max_queries  : int   - hard stop on total oracle calls
    device       : str   - 'cpu' or 'cuda'
    """

    ACQUISITION_MAP = {
        'EI':          'expected_improvement',
        'UCB':         'upper_confidence_bound',
        'BALD':        'bald',
        'random':      None,
        'uncertainty': None,
    }

    def __init__(self, gp, oracle, latent_dim: int = 16,
                 vae=None, n_candidates: int = 500, batch_size: int = 5,
                 acquisition: str = 'EI', xi: float = 0.01, kappa: float = 2.0,
                 convergence_tol: float = 1e-3, max_queries: int = 200,
                 device: str = 'cpu', verbose: bool = True):
        self.vae            = vae
        self.gp             = gp
        self.oracle         = oracle
        self.latent_dim     = latent_dim
        self.n_candidates   = n_candidates
        self.batch_size     = batch_size
        self.acquisition    = acquisition
        self.xi             = xi
        self.kappa          = kappa
        self.convergence_tol = convergence_tol
        self.max_queries    = max_queries
        self.device         = device
        self.verbose        = verbose

        self._Z_all = None
        self._E_all = None
        self._total_queries = 0

    def seed(self, Z_seed: np.ndarray, E_seed: np.ndarray):
        """
        Seed the GP with a small initial dataset before running the loop.
        Typically 5-20 randomly selected configurations.
        """
        self._Z_all = Z_seed.copy()
        self._E_all = E_seed.copy()
        self.gp.fit(Z_seed, E_seed)
        self._total_queries += len(Z_seed)
        if self.verbose:
            print(f"[Seed] {len(Z_seed)} configs | "
                  f"best energy: {E_seed.min():.4f} eV/atom")

    def _sample_candidates(self) -> np.ndarray:
        """Sample candidate latent vectors from VAE prior or Gaussian."""
        if self.vae is not None:
            try:
                import torch
                with torch.no_grad():
                    z = self.vae.sample(self.n_candidates, device=self.device)
                return z.cpu().numpy()
            except ImportError:
                pass
        # Fallback: standard Gaussian samples (equivalent to VAE prior)
        return np.random.randn(self.n_candidates, self.latent_dim).astype(np.float32)

    def _score_candidates(self, Z_cand: np.ndarray) -> np.ndarray:
        """Score candidates using the configured acquisition function."""
        acq = self.acquisition
        if acq == 'EI':
            return self.gp.expected_improvement(Z_cand, xi=self.xi)
        elif acq == 'UCB':
            return self.gp.upper_confidence_bound(Z_cand, kappa=self.kappa)
        elif acq == 'BALD':
            return self.gp.bald(Z_cand)
        elif acq == 'random':
            return random_acquisition(Z_cand)
        elif acq == 'uncertainty':
            return uncertainty_acquisition(Z_cand, gp=self.gp)
        else:
            raise ValueError(f"Unknown acquisition: {acq!r}. "
                             f"Choose from {list(self.ACQUISITION_MAP)}")

    def step(self) -> dict:
        """Execute one iteration of the active learning loop."""
        # 1. Generate candidates
        Z_cand = self._sample_candidates()

        # 2. Score with acquisition function
        scores = self._score_candidates(Z_cand)

        # 3. Select top-k
        top_idx = np.argsort(scores)[::-1][:self.batch_size]
        Z_selected = Z_cand[top_idx]

        # 4. Query oracle
        Z_matched, E_new = self.oracle.query(Z_selected)
        self._total_queries += len(Z_matched)

        # 5. Update GP
        self.gp.update(Z_matched, E_new)

        # 6. Accumulate
        if self._Z_all is None:
            self._Z_all = Z_matched
            self._E_all = E_new
        else:
            self._Z_all = np.vstack([self._Z_all, Z_matched])
            self._E_all = np.concatenate([self._E_all, E_new])

        metrics = {
            'best_energy':   float(self._E_all.min()),
            'gp_var_sum':    self.gp.posterior_variance_sum(),
            'n_queries':     self._total_queries,
            'n_remaining':   self.oracle.n_remaining,
            'frac_low_e':    self.oracle.fraction_low_energy_discovered,
        }
        return metrics

    def run(self, n_iterations: int = 20) -> LoopResults:
        """
        Run the full active learning loop.

        Parameters
        ----------
        n_iterations : int
            Maximum number of iterations (each queries batch_size configs).

        Returns
        -------
        LoopResults dataclass with per-iteration metrics.
        """
        assert self._Z_all is not None, \
            "Call loop.seed(Z_seed, E_seed) before loop.run()."

        results = LoopResults()

        for i in range(1, n_iterations + 1):
            if self._total_queries >= self.max_queries:
                if self.verbose:
                    print(f"[Loop] max_queries={self.max_queries} reached. Stopping.")
                break
            if self.oracle.n_remaining == 0:
                if self.verbose:
                    print("[Loop] Oracle pool exhausted. Stopping.")
                break

            metrics = self.step()

            results.iterations.append(i)
            results.best_energy_history.append(metrics['best_energy'])
            results.gp_variance_history.append(metrics['gp_var_sum'])
            results.n_queries_history.append(metrics['n_queries'])
            results.frac_low_e_history.append(metrics['frac_low_e'])

            if self.verbose:
                print(f"[Iter {i:3d}] "
                      f"best={metrics['best_energy']:.4f}  "
                      f"GP_var={metrics['gp_var_sum']:.4f}  "
                      f"queries={metrics['n_queries']}  "
                      f"low-E%={metrics['frac_low_e']*100:.1f}")

            if metrics['gp_var_sum'] < self.convergence_tol:
                if self.verbose:
                    print(f"[Loop] GP variance converged below {self.convergence_tol}.")
                break

        results.Z_discovered = self._Z_all
        results.E_discovered = self._E_all
        return results


# ── Benchmark runner ───────────────────────────────────────────────────────

def run_benchmark(Z_pool: np.ndarray, E_pool: np.ndarray,
                  latent_dim: int = 16, seed_size: int = 10,
                  n_iterations: int = 20, batch_size: int = 5,
                  acquisitions: list = None, random_seed: int = 42) -> dict:
    """
    Compare acquisition functions on the same pool.
    Returns dict mapping acquisition name -> LoopResults.
    """
    from gp_surrogate import GPSurrogate

    if acquisitions is None:
        acquisitions = ['EI', 'UCB', 'random']

    rng = np.random.default_rng(random_seed)
    seed_idx = rng.choice(len(Z_pool), size=seed_size, replace=False)

    all_results = {}
    for acq in acquisitions:
        print(f"\n{'='*50}")
        print(f"Acquisition: {acq}")
        print('='*50)

        oracle = DFTDatasetOracle(Z_pool, E_pool)
        gp     = GPSurrogate(latent_dim=latent_dim)
        loop   = ActiveLearningLoop(
            gp=gp, oracle=oracle, latent_dim=latent_dim,
            batch_size=batch_size, acquisition=acq, verbose=True
        )
        loop.seed(Z_pool[seed_idx], E_pool[seed_idx])
        results = loop.run(n_iterations=n_iterations)
        all_results[acq] = results
        print(results.summary())

    return all_results
