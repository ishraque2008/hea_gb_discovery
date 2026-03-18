"""
viz_latent.py
-------------
Latent space visualisation for HEA GB discovery pipeline.

Produces three figures:
  1. 2D projection of latent space (UMAP if available, else PCA) coloured by:
       - known GB type labels (interpretability analysis)
       - predicted formation energy (GP posterior mean)
       - GP uncertainty (posterior std dev)
  2. Active learning convergence curves (best energy, GP variance over iterations)
  3. Low-energy discovery efficiency comparison across acquisition functions

Usage
-----
  from viz_latent import (plot_latent_space, plot_convergence,
                           plot_benchmark_comparison)

  plot_latent_space(Z, labels=gb_types, energies=E_pred,
                    uncertainties=sigma, savepath='latent_space.png')

  plot_convergence(results, savepath='convergence.png')

  plot_benchmark_comparison(benchmark_dict, savepath='benchmark.png')
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import warnings


# ── Projection ─────────────────────────────────────────────────────────────

def _project_2d(Z: np.ndarray, method: str = 'auto') -> tuple:
    """
    Project latent vectors Z to 2D for visualisation.
    Tries UMAP first; falls back to PCA if unavailable.

    Returns
    -------
    Z2d     : np.ndarray, shape [N, 2]
    method  : str, 'umap' or 'pca'
    """
    if method in ('auto', 'umap'):
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42,
                                n_neighbors=min(15, len(Z) - 1),
                                min_dist=0.1)
            Z2d = reducer.fit_transform(Z)
            return Z2d, 'umap'
        except ImportError:
            if method == 'umap':
                warnings.warn("umap-learn not installed; falling back to PCA.")

    from sklearn.decomposition import PCA
    pca  = PCA(n_components=2, random_state=42)
    Z2d  = pca.fit_transform(Z)
    return Z2d, 'pca'


# ── Figure 1: Latent space ─────────────────────────────────────────────────

def plot_latent_space(Z: np.ndarray,
                      labels: np.ndarray = None,
                      energies: np.ndarray = None,
                      uncertainties: np.ndarray = None,
                      label_names: dict = None,
                      projection: str = 'auto',
                      savepath: str = 'latent_space.png',
                      dpi: int = 150) -> str:
    """
    Visualise the VAE latent space as a 2D projection.

    Parameters
    ----------
    Z             : np.ndarray, shape [N, latent_dim]
    labels        : int array [N], GB type indices (0,1,2,...)
    energies      : float array [N], predicted formation energies
    uncertainties : float array [N], GP posterior std dev
    label_names   : dict {int: str}, e.g. {0:'Sigma-3 twin', 1:'Random HAGB'}
    projection    : 'auto' | 'umap' | 'pca'
    savepath      : output filename
    dpi           : figure resolution

    Returns
    -------
    savepath : str
    """
    Z2d, proj_method = _project_2d(Z, method=projection)

    n_panels = 1 + (energies is not None) + (uncertainties is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5),
                              constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    proj_label = proj_method.upper()
    ax_idx = 0

    # Panel A: GB type labels
    ax = axes[ax_idx]; ax_idx += 1
    if labels is not None:
        unique_labels = np.unique(labels)
        cmap_l = cm.get_cmap('tab10', len(unique_labels))
        for li, lab in enumerate(unique_labels):
            mask = labels == lab
            name = (label_names or {}).get(lab, f'Type {lab}')
            ax.scatter(Z2d[mask, 0], Z2d[mask, 1],
                       color=cmap_l(li), s=18, alpha=0.75, label=name)
        ax.legend(fontsize=8, framealpha=0.8, markerscale=1.5)
    else:
        ax.scatter(Z2d[:, 0], Z2d[:, 1], s=12, alpha=0.6, color='steelblue')
    ax.set_xlabel(f'{proj_label}-1', fontsize=10)
    ax.set_ylabel(f'{proj_label}-2', fontsize=10)
    ax.set_title(f'Latent Space: GB Type Clusters ({proj_label})', fontsize=11)
    ax.tick_params(labelsize=8)

    # Panel B: Formation energy
    if energies is not None:
        ax = axes[ax_idx]; ax_idx += 1
        sc = ax.scatter(Z2d[:, 0], Z2d[:, 1], c=energies,
                        cmap='RdYlBu_r', s=18, alpha=0.8)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label('Formation energy (eV/atom)', fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax.set_xlabel(f'{proj_label}-1', fontsize=10)
        ax.set_ylabel(f'{proj_label}-2', fontsize=10)
        ax.set_title('GP Predicted Formation Energy', fontsize=11)
        ax.tick_params(labelsize=8)

    # Panel C: GP uncertainty
    if uncertainties is not None:
        ax = axes[ax_idx]; ax_idx += 1
        sc = ax.scatter(Z2d[:, 0], Z2d[:, 1], c=uncertainties,
                        cmap='plasma', s=18, alpha=0.8)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label('GP posterior std dev', fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax.set_xlabel(f'{proj_label}-1', fontsize=10)
        ax.set_ylabel(f'{proj_label}-2', fontsize=10)
        ax.set_title('Exploration Uncertainty', fontsize=11)
        ax.tick_params(labelsize=8)

    fig.suptitle('HEA Grain-Boundary Latent Space', fontsize=13, fontweight='bold')
    plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {savepath}")
    return savepath


# ── Figure 2: Convergence ──────────────────────────────────────────────────

def plot_convergence(results, savepath: str = 'convergence.png',
                     dpi: int = 150) -> str:
    """
    Plot active learning convergence curves.

    Parameters
    ----------
    results : LoopResults (from active_loop.py)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    iters = results.iterations

    axes[0].plot(iters, results.best_energy_history, 'o-', color='royalblue',
                 lw=2, ms=5)
    axes[0].set_xlabel('Iteration', fontsize=11)
    axes[0].set_ylabel('Best energy found (eV/atom)', fontsize=11)
    axes[0].set_title('Best Formation Energy vs. Queries', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, results.gp_variance_history, 's-', color='darkorange',
                 lw=2, ms=5)
    axes[1].set_xlabel('Iteration', fontsize=11)
    axes[1].set_ylabel('GP posterior variance sum', fontsize=11)
    axes[1].set_title('GP Uncertainty Reduction', fontsize=12)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iters,
                 [f * 100 for f in results.frac_low_e_history],
                 '^-', color='seagreen', lw=2, ms=5)
    axes[2].set_xlabel('Iteration', fontsize=11)
    axes[2].set_ylabel('Low-energy configs discovered (%)', fontsize=11)
    axes[2].set_title('Discovery Rate of Low-Energy GBs', fontsize=12)
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Active Learning Convergence', fontsize=13, fontweight='bold')
    plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {savepath}")
    return savepath


# ── Figure 3: Benchmark comparison ────────────────────────────────────────

def plot_benchmark_comparison(benchmark_dict: dict,
                               savepath: str = 'benchmark.png',
                               dpi: int = 150) -> str:
    """
    Compare acquisition strategies side-by-side.

    Parameters
    ----------
    benchmark_dict : dict {str: LoopResults}
        Output of active_loop.run_benchmark().
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    colors = ['royalblue', 'darkorange', 'seagreen', 'crimson', 'purple']
    for ci, (name, results) in enumerate(benchmark_dict.items()):
        c = colors[ci % len(colors)]
        iters = results.iterations

        axes[0].plot(results.n_queries_history, results.best_energy_history,
                     'o-', color=c, lw=2, ms=5, label=name)
        axes[1].plot(results.n_queries_history,
                     [f * 100 for f in results.frac_low_e_history],
                     's-', color=c, lw=2, ms=5, label=name)

    axes[0].set_xlabel('Total oracle queries', fontsize=11)
    axes[0].set_ylabel('Best energy found (eV/atom)', fontsize=11)
    axes[0].set_title('Sample Efficiency: Best Energy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Total oracle queries', fontsize=11)
    axes[1].set_ylabel('Low-energy configs discovered (%)', fontsize=11)
    axes[1].set_title('Sample Efficiency: Discovery Rate', fontsize=12)
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Acquisition Function Benchmark', fontsize=13, fontweight='bold')
    plt.savefig(savepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {savepath}")
    return savepath
