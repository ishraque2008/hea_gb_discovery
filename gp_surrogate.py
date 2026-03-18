"""
gp_surrogate.py
---------------
Gaussian Process surrogate model for predicting DFT formation energies
from latent vectors produced by GBVAE.

Uses scikit-learn GaussianProcessRegressor with a Matern-5/2 kernel,
which is appropriate for physical property prediction:
  - Smooth but not infinitely differentiable (good for rugged energy landscapes)
  - Automatic relevance determination (ARD) via anisotropic length scales

Posterior variance is used downstream by active_loop.py for acquisition.

Usage
-----
  from gp_surrogate import GPSurrogate
  gp = GPSurrogate(latent_dim=16)
  gp.fit(Z_train, E_train)                   # Z: [N, 16], E: [N]
  mu, sigma = gp.predict(Z_query)            # mu: [M], sigma: [M]
  ei = gp.expected_improvement(Z_candidates) # EI acquisition values
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler


class GPSurrogate:
    """
    Gaussian Process surrogate for HEA grain-boundary formation energy.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of input latent vectors (must match GBVAE.latent_dim).
    n_restarts : int
        Number of random restarts for kernel hyperparameter optimisation.
    noise_level : float
        Initial estimate of observation noise (eV/atom).
    """

    def __init__(self, latent_dim: int = 16, n_restarts: int = 5,
                 noise_level: float = 0.05):
        self.latent_dim  = latent_dim
        self.n_restarts  = n_restarts
        self.noise_level = noise_level
        self._scaler     = StandardScaler()
        self._build_gp()
        self.is_fitted   = False
        self.y_best      = None    # best (lowest) energy seen so far

    def _build_gp(self):
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(length_scale=np.ones(self.latent_dim),
                     length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=self.noise_level,
                          noise_level_bounds=(1e-4, 1.0))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=True,
            random_state=42
        )

    def fit(self, Z: np.ndarray, E: np.ndarray):
        """
        Fit (or refit) the GP on latent vectors Z and energies E.

        Parameters
        ----------
        Z : np.ndarray, shape [N, latent_dim]
        E : np.ndarray, shape [N]  (DFT formation energies, eV/atom)
        """
        Z_scaled = self._scaler.fit_transform(Z)
        self.gp.fit(Z_scaled, E)
        self.is_fitted = True
        self.y_best    = float(np.min(E))
        return self

    def update(self, Z_new: np.ndarray, E_new: np.ndarray):
        """
        Incrementally add new observations and refit.
        Maintains the full training set for exact GP inference.
        """
        if not self.is_fitted:
            return self.fit(Z_new, E_new)

        # Inverse-transform existing training data from scaled space
        Z_prev = self._scaler.inverse_transform(self.gp.X_train_)
        E_prev = (self.gp.y_train_ * self.gp._y_train_std
                  + self.gp._y_train_mean)

        Z_all = np.vstack([Z_prev, Z_new])
        E_all = np.concatenate([E_prev, E_new])
        self._build_gp()   # reset hyperparameters for clean refit
        return self.fit(Z_all, E_all)

    def predict(self, Z: np.ndarray) -> tuple:
        """
        Predict mean and standard deviation of formation energy.

        Returns
        -------
        mu    : np.ndarray, shape [M]
        sigma : np.ndarray, shape [M]  (posterior std dev)
        """
        assert self.is_fitted, "GPSurrogate.fit() must be called first."
        Z_scaled = self._scaler.transform(Z)
        mu, sigma = self.gp.predict(Z_scaled, return_std=True)
        return mu, sigma

    # ── Acquisition functions ────────────────────────────────────────

    def expected_improvement(self, Z: np.ndarray,
                             xi: float = 0.01) -> np.ndarray:
        """
        Expected Improvement (EI) acquisition function.
        Minimisation convention (lower energy = better).

        EI(z) = (y_best - mu - xi) * Phi(Z) + sigma * phi(Z)
        where Z = (y_best - mu - xi) / sigma

        Parameters
        ----------
        Z  : np.ndarray, shape [M, latent_dim]
        xi : float, exploration bonus (default 0.01)

        Returns
        -------
        ei : np.ndarray, shape [M], non-negative
        """
        from scipy.stats import norm
        mu, sigma = self.predict(Z)
        sigma = np.maximum(sigma, 1e-9)
        improvement = self.y_best - mu - xi
        Z_score = improvement / sigma
        ei = improvement * norm.cdf(Z_score) + sigma * norm.pdf(Z_score)
        return np.maximum(ei, 0.0)

    def upper_confidence_bound(self, Z: np.ndarray,
                               kappa: float = 2.0) -> np.ndarray:
        """
        Lower Confidence Bound (LCB) for minimisation.
        acquisition = -(mu - kappa * sigma)
        """
        mu, sigma = self.predict(Z)
        return -(mu - kappa * sigma)

    def bald(self, Z: np.ndarray) -> np.ndarray:
        """
        Bayesian Active Learning by Disagreement (proxy):
        returns posterior variance, maximised for highest uncertainty.
        """
        _, sigma = self.predict(Z)
        return sigma ** 2

    def posterior_variance_sum(self) -> float:
        """
        Convergence diagnostic: sum of posterior variance over training points.
        Decreases monotonically as the GP is refined.
        """
        if not self.is_fitted:
            return float('inf')
        _, sigma = self.predict(
            self._scaler.inverse_transform(self.gp.X_train_)
        )
        return float(np.sum(sigma ** 2))
