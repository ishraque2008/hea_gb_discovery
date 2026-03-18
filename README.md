# HEA Grain-Boundary Discovery: Closed-Loop Generative AI Pipeline

A proof-of-concept implementation of the research proposal:

> **"Generative Discovery of Grain-Boundary Motifs in High-Entropy Alloys:
> A Closed-Loop AI System for Autonomous Microstructure Exploration"**

Submitted for the Lila Sciences AI Residency, 2026 Cohort.

---

## What This Is

Most ML pipelines in materials science are passive: train a model on a fixed
dataset, predict a property, report accuracy. This repository implements the
opposite: an AI system that **actively participates in scientific discovery** by
generating candidate grain-boundary (GB) configurations, estimating which ones
are worth evaluating, querying a physics oracle (DFT / MD / experiment), and
improving its model iteratively.

The closed loop:

```
Sample from VAE prior
        |
        v
Score candidates (EI / UCB / BALD)
        |
        v
Query oracle (DFT dataset / live DFT / experiment)
        |
        v
Update GP surrogate
        |
        v
Retrain VAE (next round)  <--------|
```

---

## Modules

| File | Purpose |
|------|---------|
| `gb_vae.py` | VAE for atomic-scale SOAP/MBTR descriptor vectors. Learns a 16-dim latent space over HEA GB configurations. Physics-penalized reconstruction loss. |
| `gp_surrogate.py` | Gaussian Process surrogate (Matern-5/2, ARD) mapping latent vectors to DFT formation energies. Provides posterior mean and uncertainty. |
| `active_loop.py` | Closed-loop active learning engine. Oracle-agnostic: plug in DFT dataset, live DFT, MD, or STEM experiment. Supports EI, UCB, BALD, random, and uncertainty acquisition. |
| `viz_latent.py` | Visualisation: UMAP/PCA latent space projections, convergence curves, acquisition function benchmarks. |
| `test_pipeline.py` | Integration test suite (5 tests, fully runnable without PyTorch or real DFT data). |

---

## Quick Start

### 1. Install dependencies

```bash
pip install numpy scipy scikit-learn matplotlib
# For VAE training:
pip install torch
# For SOAP/MBTR descriptors:
pip install dscribe
# For UMAP visualisation:
pip install umap-learn
```

### 2. Run the integration test (no GPU, no DFT data needed)

```bash
python test_pipeline.py
```

Expected output:
```
[PASS] GP surrogate: fit / predict / acquisition functions
[PASS] DFT dataset oracle: query / tracking / discovery rate
[PASS] Active learning loop: EI / UCB / random
[PASS] Benchmark runner: multi-acquisition comparison
[PASS] Visualisation: latent space / convergence / benchmark
Results: 5/5 tests passed
```

### 3. Run on real DFT data

```python
import numpy as np
from gb_vae import train_vae, encode_descriptors
from gp_surrogate import GPSurrogate
from active_loop import ActiveLearningLoop, DFTDatasetOracle

# Load your SOAP descriptors and DFT energies
# (compute descriptors with DScribe from ASE Atoms objects)
X = np.load("soap_descriptors.npy")   # shape [N, descriptor_dim]
E = np.load("dft_energies.npy")       # shape [N], eV/atom

# 1. Train VAE
model, history, scaler = train_vae(X, latent_dim=16, epochs=200)

# 2. Encode to latent space
Z = encode_descriptors(model, X, scaler)

# 3. Split: seed set + holdout pool
seed_idx = np.random.choice(len(Z), size=20, replace=False)
pool_mask = np.ones(len(Z), dtype=bool)
pool_mask[seed_idx] = False

oracle = DFTDatasetOracle(Z[pool_mask], E[pool_mask])
gp     = GPSurrogate(latent_dim=16)
loop   = ActiveLearningLoop(
    vae=model, gp=gp, oracle=oracle, latent_dim=16,
    batch_size=5, acquisition='EI', n_candidates=1000
)

loop.seed(Z[seed_idx], E[seed_idx])
results = loop.run(n_iterations=30)
print(results.summary())
```

---

## Using a Live DFT Oracle

Replace `DFTDatasetOracle` with any object implementing:

```python
class MyDFTOracle:
    def query(self, Z_query: np.ndarray) -> tuple:
        # Z_query: [batch, latent_dim]
        # Returns (Z_matched, E_matched): both numpy arrays
        ...

    @property
    def n_remaining(self) -> int: ...

    @property
    def fraction_low_energy_discovered(self) -> float: ...
```

For VASP, the workflow is: decode Z_query back to SOAP descriptors via the VAE
decoder, reconstruct an ASE Atoms object from the nearest GB in the structural
database, run VASP single-point, return energy.

---

## Data Sources

This demo uses publicly available DFT grain-boundary datasets:

- **Materials Project** grain boundary entries: `mp-api` Python client
- **AFLOW** grain boundary database: `aflow` Python client
- **GBDB** (Grain Boundary Database, Olmsted et al.): https://github.com/usnistgov/gbDB

For Cantor alloy (CrMnFeCoNi) GBs specifically, see:
- Yin et al., *Acta Materialia* 212, 116911 (2021)
- Messina et al., *npj Computational Materials* 8, 84 (2022)

---

## Descriptor Computation (DScribe Example)

```python
from dscribe.descriptors import SOAP
from ase.io import read

# Load GB structure from CIF / POSCAR
atoms = read("gb_structure.vasp")

soap = SOAP(
    species=["Cr", "Mn", "Fe", "Co", "Ni"],
    r_cut=6.0,
    n_max=8,
    l_max=6,
    periodic=True,
    average="outer"
)
descriptor = soap.create(atoms)  # shape [descriptor_dim]
```

---

## Architecture Notes

**Why a VAE, not a diffusion model?**

A VAE provides an explicit, differentiable latent space that the Bayesian
acquisition function can reason over analytically. Diffusion models produce
higher-quality samples but do not expose a tractable latent space for GP-based
acquisition. The natural upgrade path (Month 7+ of the residency) is to replace
the VAE with a flow-matching model and use latent-space EI over the flow's
continuous normalizing flow.

**Why sklearn GP, not GPyTorch?**

The sklearn GaussianProcessRegressor with a Matern-5/2 + ARD kernel is
sufficient and reproducible for up to ~500 training points. For larger datasets
(> 1000 configurations), swap in `botorch.models.SingleTaskGP` (GPyTorch-based)
by implementing the same `fit / predict / expected_improvement` interface.

**Why oracle-agnostic design?**

The oracle interface is deliberately minimal (one method, two properties) so
the same loop runs against a held-out DFT dataset (for benchmarking), a live
VASP calculation (for production), or a STEM experiment (for
simulation-to-experiment transfer). This modularity is the architectural
prerequisite for the autonomous lab scenario.

---

## Relation to Published Work

This codebase directly extends the active learning pipeline from:

> Borshon, I. Z., et al. (2025). Deep learning analysis of solid-electrolyte
> interphase microstructures in lithium-ion batteries.
> *Advanced Materials Interfaces.*

That work used multislice HRTEM simulation as the oracle and a transformer
segmentation model as the learner. The present repository replaces those with
a VAE generative model and GP surrogate, targeting the higher-dimensional HEA
configuration space.

---

## License

MIT
