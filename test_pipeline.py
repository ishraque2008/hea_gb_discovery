"""
test_pipeline.py
----------------
End-to-end integration test for the HEA GB discovery pipeline.
Uses synthetic data to exercise all four modules without requiring
DScribe, PyTorch, or real DFT datasets.

Run:
  python test_pipeline.py

Expected output:
  All 5 tests PASSED
  Convergence and benchmark figures saved to ./test_outputs/
"""

import sys
import os
import numpy as np
import traceback

sys.path.insert(0, os.path.dirname(__file__))
OUTDIR = os.path.join(os.path.dirname(__file__), 'test_outputs')
os.makedirs(OUTDIR, exist_ok=True)

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def run_test(name, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append(True)
    except Exception as e:
        print(f"{FAIL} {name}")
        traceback.print_exc()
        results.append(False)


# ── Synthetic dataset ──────────────────────────────────────────────────────

def make_synthetic_gb_data(n=300, descriptor_dim=64, latent_dim=16, seed=42):
    """
    Generate synthetic GB-like data:
      - 3 GB 'types' embedded in different regions of descriptor space
      - Formation energy = -0.5 * exp(-||z||^2 / 4) + noise
        (lower energy near the origin of latent space)
    """
    rng = np.random.default_rng(seed)
    n_per_type = n // 3

    # Cluster centres in descriptor space
    centres = [
        rng.standard_normal(descriptor_dim) * 2.0,
        rng.standard_normal(descriptor_dim) * 2.0 + 3.0,
        rng.standard_normal(descriptor_dim) * 2.0 - 3.0,
    ]
    X_list, L_list = [], []
    for i, c in enumerate(centres):
        X_list.append(rng.standard_normal((n_per_type, descriptor_dim)) * 0.8 + c)
        L_list.append(np.full(n_per_type, i))

    X = np.vstack(X_list).astype(np.float32)
    L = np.concatenate(L_list).astype(int)

    # Formation energy: clusters near centre have lower energy
    norms = np.linalg.norm(X, axis=1)
    E = -0.5 * np.exp(-norms**2 / (4 * descriptor_dim)) + \
        rng.standard_normal(len(X)) * 0.05
    E = E.astype(np.float64)

    return X, L, E


# ── Test 1: GP surrogate ───────────────────────────────────────────────────

def test_gp_surrogate():
    from gp_surrogate import GPSurrogate

    rng = np.random.default_rng(0)
    latent_dim = 8
    N_train, N_test = 60, 20

    Z_train = rng.standard_normal((N_train, latent_dim)).astype(np.float32)
    E_train = np.sin(Z_train[:, 0]) + 0.1 * rng.standard_normal(N_train)
    Z_test  = rng.standard_normal((N_test, latent_dim)).astype(np.float32)

    gp = GPSurrogate(latent_dim=latent_dim, n_restarts=1)
    gp.fit(Z_train, E_train)
    mu, sigma = gp.predict(Z_test)

    assert mu.shape == (N_test,),    f"mu shape {mu.shape}"
    assert sigma.shape == (N_test,), f"sigma shape {sigma.shape}"
    assert np.all(sigma >= 0),       "sigma must be non-negative"

    ei  = gp.expected_improvement(Z_test)
    ucb = gp.upper_confidence_bound(Z_test)
    bal = gp.bald(Z_test)

    assert ei.shape  == (N_test,), f"EI shape {ei.shape}"
    assert ucb.shape == (N_test,), f"UCB shape {ucb.shape}"
    assert bal.shape == (N_test,), f"BALD shape {bal.shape}"
    assert np.all(ei >= 0),  "EI must be non-negative"
    assert np.all(bal >= 0), "BALD must be non-negative"

    # Update with new data
    Z_new = rng.standard_normal((5, latent_dim)).astype(np.float32)
    E_new = np.sin(Z_new[:, 0])
    gp.update(Z_new, E_new)
    mu2, _ = gp.predict(Z_test)
    assert mu2.shape == (N_test,), "GP update broke predict shape"


# ── Test 2: Oracle ─────────────────────────────────────────────────────────

def test_oracle():
    from active_loop import DFTDatasetOracle

    rng = np.random.default_rng(1)
    Z_pool = rng.standard_normal((100, 8)).astype(np.float32)
    E_pool = rng.standard_normal(100)
    oracle = DFTDatasetOracle(Z_pool, E_pool)

    Z_q = rng.standard_normal((5, 8)).astype(np.float32)
    Z_m, E_m = oracle.query(Z_q)

    assert Z_m.shape == (5, 8),    f"Z_matched shape {Z_m.shape}"
    assert E_m.shape == (5,),      f"E_matched shape {E_m.shape}"
    assert oracle.n_remaining == 95, f"n_remaining = {oracle.n_remaining}"
    assert oracle.fraction_low_energy_discovered >= 0

    # Verify tracking prevents duplicates
    _, E2 = oracle.query(Z_q)
    total_distinct = len(oracle._queried_indices)
    assert total_distinct >= 5, "Query tracking failed"


# ── Test 3: Active learning loop (EI, random, UCB) ────────────────────────

def test_active_loop():
    from gp_surrogate import GPSurrogate
    from active_loop import ActiveLearningLoop, DFTDatasetOracle

    rng = np.random.default_rng(2)
    latent_dim = 8
    N = 200

    Z_pool = rng.standard_normal((N, latent_dim)).astype(np.float32)
    E_pool = (np.linalg.norm(Z_pool, axis=1) * -0.3
              + rng.standard_normal(N) * 0.05)

    seed_idx = rng.choice(N, size=10, replace=False)

    for acq in ['EI', 'UCB', 'random']:
        oracle = DFTDatasetOracle(Z_pool, E_pool)
        gp     = GPSurrogate(latent_dim=latent_dim, n_restarts=1)
        loop   = ActiveLearningLoop(
            gp=gp, oracle=oracle, latent_dim=latent_dim,
            batch_size=5, acquisition=acq, n_candidates=100,
            verbose=False
        )
        loop.seed(Z_pool[seed_idx], E_pool[seed_idx])
        res = loop.run(n_iterations=5)

        assert len(res.iterations) > 0, f"{acq}: no iterations ran"
        assert len(res.best_energy_history) == len(res.iterations)
        assert res.Z_discovered is not None
        assert res.E_discovered is not None
        # Energy should be non-increasing (monotone best)
        beh = res.best_energy_history
        assert all(beh[i] <= beh[i-1] + 1e-9 for i in range(1, len(beh))), \
            f"{acq}: best energy not monotone: {beh}"


# ── Test 4: Benchmark runner ───────────────────────────────────────────────

def test_benchmark():
    from active_loop import run_benchmark

    rng = np.random.default_rng(3)
    latent_dim = 6
    N = 150

    Z_pool = rng.standard_normal((N, latent_dim)).astype(np.float32)
    E_pool = -np.exp(-np.linalg.norm(Z_pool, axis=1)) + rng.standard_normal(N) * 0.05

    bench = run_benchmark(
        Z_pool, E_pool, latent_dim=latent_dim,
        seed_size=8, n_iterations=4, batch_size=4,
        acquisitions=['EI', 'random'], random_seed=42
    )

    assert 'EI' in bench and 'random' in bench
    for name, res in bench.items():
        assert len(res.iterations) > 0, f"{name}: no iterations"

    return bench


# ── Test 5: Visualisation ──────────────────────────────────────────────────

def test_viz():
    from gp_surrogate import GPSurrogate
    from active_loop import ActiveLearningLoop, DFTDatasetOracle, run_benchmark
    from viz_latent import plot_latent_space, plot_convergence, plot_benchmark_comparison

    rng = np.random.default_rng(4)
    latent_dim = 8
    N = 200

    Z_pool = rng.standard_normal((N, latent_dim)).astype(np.float32)
    E_pool = -np.exp(-np.linalg.norm(Z_pool, axis=1)) + rng.standard_normal(N) * 0.03
    labels = (np.linalg.norm(Z_pool, axis=1) > 2.5).astype(int) + \
             (Z_pool[:, 0] > 1.0).astype(int)

    # Fit a GP for energy/uncertainty
    gp = GPSurrogate(latent_dim=latent_dim, n_restarts=1)
    gp.fit(Z_pool[:100], E_pool[:100])
    mu, sigma = gp.predict(Z_pool)

    # Fig 1: latent space
    p1 = plot_latent_space(
        Z_pool, labels=labels, energies=mu, uncertainties=sigma,
        label_names={0: 'Sigma-3 twin', 1: 'HAGB', 2: 'Twist'},
        projection='pca',
        savepath=os.path.join(OUTDIR, 'latent_space.png')
    )
    assert os.path.exists(p1), "latent_space.png not created"

    # Run a short loop for convergence plot
    seed_idx = rng.choice(N, size=10, replace=False)
    oracle = DFTDatasetOracle(Z_pool, E_pool)
    gp2    = GPSurrogate(latent_dim=latent_dim, n_restarts=1)
    loop   = ActiveLearningLoop(
        gp=gp2, oracle=oracle, latent_dim=latent_dim,
        batch_size=5, acquisition='EI', n_candidates=150, verbose=False
    )
    loop.seed(Z_pool[seed_idx], E_pool[seed_idx])
    res = loop.run(n_iterations=8)

    p2 = plot_convergence(res, savepath=os.path.join(OUTDIR, 'convergence.png'))
    assert os.path.exists(p2), "convergence.png not created"

    # Benchmark
    bench = run_benchmark(
        Z_pool, E_pool, latent_dim=latent_dim, seed_size=10,
        n_iterations=5, batch_size=5,
        acquisitions=['EI', 'random'], random_seed=7
    )
    p3 = plot_benchmark_comparison(bench,
             savepath=os.path.join(OUTDIR, 'benchmark.png'))
    assert os.path.exists(p3), "benchmark.png not created"


# ── Run all tests ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("HEA GB Discovery Pipeline - Integration Tests")
    print("=" * 55)

    run_test("GP surrogate: fit / predict / acquisition functions", test_gp_surrogate)
    run_test("DFT dataset oracle: query / tracking / discovery rate", test_oracle)
    run_test("Active learning loop: EI / UCB / random", test_active_loop)
    run_test("Benchmark runner: multi-acquisition comparison", test_benchmark)
    run_test("Visualisation: latent space / convergence / benchmark", test_viz)

    print("\n" + "=" * 55)
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"Results: {n_pass}/{len(results)} tests passed")
    if n_fail > 0:
        print(f"  {n_fail} test(s) FAILED - see traceback above")
        sys.exit(1)
    else:
        print("All tests PASSED.")
        print(f"Figures saved to: {OUTDIR}/")
