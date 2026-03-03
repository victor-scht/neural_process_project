import numpy as np
from pathlib import Path

from hawkes import simulate_multivariate_hawkes_ogata
from jumpdiffusion import simulate_jump_diffusion, f_drift_true, g_sigma_true
from estimation import estimate_parameters, reconstruct_X
import plots


def make_adjacency(M, seed=0):
    rng = np.random.default_rng(seed)
    A = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            same_block = (i < M // 2 and j < M // 2) or (i >= M // 2 and j >= M // 2)
            p = 0.20 if same_block else 0.08
            if rng.random() < p:
                A[i, j] = rng.uniform(0.02, 0.10)
    rs = A.sum(axis=1).max()
    if rs > 0.7:
        A *= 0.7 / rs
    return A


def main(outdir="outputs", seed=1):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    M = 17
    T = 50.0
    dt = 0.01

    mu = np.full(M, 0.15)
    alpha = make_adjacency(M, seed=seed)
    rng = np.random.default_rng(seed)
    beta = rng.uniform(1.0, 3.0, size=(M, M))
    beta[alpha == 0.0] = 2.0

    times, marks = simulate_multivariate_hawkes_ogata(
        T=T, mu=mu, alpha=alpha, beta=beta, seed=seed
    )
    plots.plot_hawkes_raster(times, T, out / "hawkes_raster.png")

    drift_params_true = np.array([0.1, -0.8, 0.7])
    sigma_params_true = np.array([-0.3, 0.9, 1.1])

    a0_true = rng.uniform(-0.20, 0.25, size=M)
    a1_true = rng.uniform(-0.10, 0.10, size=M)

    ts, X, dW, _ = simulate_jump_diffusion(
        T=T,
        dt=dt,
        x0=0.0,
        marks=marks,
        drift_params=drift_params_true,
        sigma_params=sigma_params_true,
        a0=a0_true,
        a1=a1_true,
        seed=seed + 10,
    )

    est = estimate_parameters(ts=ts, X=X, marks=marks, M=M, dt=dt)
    print("Estimation success:", est.success, "|", est.message)
    print("NLL:", est.fun)
    print("drift theta true:", drift_params_true, "hat:", est.drift_theta)
    print("sigma phi true :", sigma_params_true, "hat:", est.sigma_phi)

    Xhat = reconstruct_X(
        ts=ts,
        X0=X[0],
        marks=marks,
        dt=dt,
        drift_theta=est.drift_theta,
        sigma_phi=est.sigma_phi,
        a0=est.a0,
        a1=est.a1,
        dW=dW,
    )

    plots.plot_X(ts, X, Xhat, out / "X_truth_vs_reconstructed.png")

    xgrid = np.linspace(np.percentile(X, 1), np.percentile(X, 99), 300)
    plots.plot_functions(
        drift_true_fn=lambda x: f_drift_true(x, drift_params_true),
        sigma_true_fn=lambda x: g_sigma_true(x, sigma_params_true),
        drift_theta=est.drift_theta,
        sigma_phi=est.sigma_phi,
        xgrid=xgrid,
        out_prefix=str(out / "truth_vs_est"),
    )

    plots.plot_jump_params(
        a0_true, a1_true, est.a0, est.a1, out_prefix=str(out / "jump_params")
    )

    # Save summary JSON
    summary = {
        "M": M,
        "T": T,
        "dt": dt,
        "n_spikes_total": int(sum(len(ti) for ti in times)),
        "drift_params_true": drift_params_true.tolist(),
        "sigma_params_true": sigma_params_true.tolist(),
        "drift_theta_hat": est.drift_theta.tolist(),
        "sigma_phi_hat": est.sigma_phi.tolist(),
    }
    (out / "summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
