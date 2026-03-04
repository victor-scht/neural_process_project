import numpy as np
import matplotlib.pyplot as plt
from estimation import drift_model, sigma_model


def plot_hawkes_raster(times, T, path):
    M = len(times)
    plt.figure(figsize=(10, 4))
    for i, ti in enumerate(times):
        if len(ti) == 0:
            continue

        plt.scatter(ti, np.full(len(ti), i), marker="+", color="red")
    plt.ylim(0, M + 1)
    plt.xlim(0, T)
    plt.xlabel("time")
    plt.grid(alpha=0.3)
    plt.ylabel("neuron")
    plt.title("Hawkes spikes (raster)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_X(ts, X_true, X_hat, path):
    plt.figure(figsize=(10, 4))
    plt.plot(ts, X_true, label="X(t) true", c="red")
    plt.plot(ts, X_hat, label="X(t) reconstructed (estimated)", c="blue")
    plt.xlabel("time")
    plt.ylabel("X")
    plt.title("Jump-diffusion path: truth vs reconstructed")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_functions(
    drift_true_fn, sigma_true_fn, drift_theta, sigma_phi, xgrid, out_prefix
):
    plt.figure(figsize=(6, 4))
    plt.plot(xgrid, drift_true_fn(xgrid), label="f (drift) true", c="red", ls="--")
    plt.plot(
        xgrid,
        drift_model(xgrid, drift_theta),
        label="f (drift) estimated",
        alpha=0.85,
        c="blue",
    )
    plt.xlabel("x")
    plt.ylabel("drift")
    plt.title("Drift function f(x)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_drift.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(xgrid, sigma_true_fn(xgrid), label="g (sigma) true", c="red", ls="--")
    plt.plot(
        xgrid,
        sigma_model(xgrid, sigma_phi),
        label="g (sigma) estimated",
        alpha=0.85,
        c="blue",
    )
    plt.xlabel("x")
    plt.ylabel("sigma")
    plt.title("Diffusion function g(x)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_sigma.png", dpi=150)
    plt.close()


def plot_jump_params(a0_true, a1_true, a0_hat, a1_hat, out_prefix):
    M = len(a0_true)
    idx = np.arange(M)
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar(idx - width / 2, a0_true, width, label="a0 true")
    plt.bar(idx + width / 2, a0_hat, width, label="a0 estimated", alpha=0.85)
    plt.xlabel("neuron")
    plt.ylabel("a0")
    plt.title("Jump intercepts per neuron")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_a0.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(idx - width / 2, a1_true, width, label="a1 true")
    plt.bar(idx + width / 2, a1_hat, width, label="a1 estimated", alpha=0.85)
    plt.xlabel("neuron")
    plt.ylabel("a1")
    plt.title("Jump tanh coefficients per neuron")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_a1.png", dpi=150)
    plt.close()
