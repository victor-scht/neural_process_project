import test_fit_process.hawkes as hawkes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")


def simu_jumpdiff(X0, grid, bfunc, sigfunc, afunc, isjumpN):
    W = np.random.randn(len(grid) - 1)
    X = np.zeros(len(grid))
    X[0] = X0

    for i in range(len(grid) - 1):
        dt = grid[i + 1] - grid[i]
        X[i + 1] = (
            X[i]
            + dt * bfunc(X[i])
            + np.sqrt(dt) * sigfunc(X[i]) * W[i]
            + afunc(X[i]) * isjumpN[i]
        )
    return X


def simu_diff(X0, grid, bfunc, sigfunc):
    W = np.random.randn(len(grid) - 1)
    X = np.zeros(len(grid))
    X[0] = X0

    for i in range(len(grid) - 1):
        dt = grid[i + 1] - grid[i]
        X[i + 1] = X[i] + dt * bfunc(X[i]) + np.sqrt(dt) * sigfunc(X[i]) * W[i]
    return X


def _make_trig_function(c0, cos_coef, sin_coef, freqs, floor=None):
    """
    Build f(x) = c0 + sum_k cos_coef[k] cos(freqs[k] x)
                   + sum_k sin_coef[k] sin(freqs[k] x)
    If floor is not None, return max(floor, f(x)).
    """
    cos_coef = np.asarray(cos_coef, dtype=float)
    sin_coef = np.asarray(sin_coef, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    def f(x):
        x = np.asarray(x)
        val = c0
        for ck, wk in zip(cos_coef, freqs):
            val = val + ck * np.cos(wk * x)
        for sk, wk in zip(sin_coef, freqs):
            val = val + sk * np.sin(wk * x)
        if floor is not None:
            val = np.maximum(floor, val)
        return val

    return f


def generate_basis_functions(K=3, seed=None, m=0.0):
    """
    Generate bfunc, sigfunc, afunc using a sine/cosine basis.

    bfunc is made mean-reverting:
        b(x) = -kappa * (x - m) + small trig perturbations

    so the process tends to come back near m.
    """
    rng = np.random.default_rng(seed)
    freqs = np.arange(1, K + 1, dtype=float)

    # Strong mean-reverting part
    kappa = rng.uniform(0.8, 1.4)
    # kappa = rng.uniform(1.5, 2.5)

    # Small oscillatory perturbation for b
    b_cos = rng.uniform(-0.05, 0.05, size=K)
    b_sin = rng.uniform(-0.05, 0.05, size=K)

    def bfunc(x):
        x = np.asarray(x)
        val = -kappa * (x - m)
        for ck, wk in zip(b_cos, freqs):
            val = val + ck * np.cos(wk * x)
        for sk, wk in zip(b_sin, freqs):
            val = val + sk * np.sin(wk * x)
        return val

    # Diffusion sigma: strictly positive
    sig_c0 = rng.uniform(0.20, 0.35)
    sig_cos = rng.uniform(-0.06, 0.06, size=K)
    sig_sin = rng.uniform(-0.06, 0.06, size=K)

    def sigfunc(x):
        x = np.asarray(x)
        val = sig_c0
        for ck, wk in zip(sig_cos, freqs):
            val = val + ck * np.cos(wk * x)
        for sk, wk in zip(sig_sin, freqs):
            val = val + sk * np.sin(wk * x)
        return np.maximum(0.05, val)

    # Jump amplitude a: positive
    # a_c0 = rng.uniform(0.12, 0.28)
    a_c0 = 2 * rng.uniform(0.12, 0.28)
    a_cos = rng.uniform(-0.05, 0.05, size=K)
    a_sin = rng.uniform(-0.05, 0.05, size=K)

    def afunc(x):
        x = np.asarray(x)
        val = a_c0
        for ck, wk in zip(a_cos, freqs):
            val = val + ck * np.cos(wk * x)
        for sk, wk in zip(a_sin, freqs):
            val = val + sk * np.sin(wk * x)
        return np.maximum(0.01, val)

    params = {
        "freqs": freqs,
        "kappa": kappa,
        "m": m,
        "b": {"cos": b_cos, "sin": b_sin},
        "sigma": {"c0": sig_c0, "cos": sig_cos, "sin": sig_sin},
        "a": {"c0": a_c0, "cos": a_cos, "sin": a_sin},
    }

    return bfunc, sigfunc, afunc, params


def hawkes_to_isjumpN(times, grid):
    """
    Concatenate all Hawkes spike times and convert them into the counting-process
    increments on the Euler grid.

    isjumpN[i] = number of spikes in (grid[i], grid[i+1]]

    Parameters
    ----------
    times : list of lists
        Multivariate Hawkes spike times
    grid : ndarray
        Time grid

    Returns
    -------
    isjumpN : ndarray of length len(grid)-1
    all_spikes : ndarray
        Sorted concatenated spike times
    """
    non_empty = [np.asarray(t, dtype=float) for t in times if len(t) > 0]

    if len(non_empty) == 0:
        all_spikes = np.array([], dtype=float)
        isjumpN = np.zeros(len(grid) - 1, dtype=int)
        return isjumpN, all_spikes

    all_spikes = np.sort(np.concatenate(non_empty))
    isjumpN, _ = np.histogram(all_spikes, bins=grid)
    return isjumpN, all_spikes


def detect_threshold_spikes(X, grid, trigger):
    """
    Define a spike when the membrane potential crosses the trigger upward.

    Returns
    -------
    spike_times : ndarray
    spike_idx   : ndarray
        Indices in the grid
    """
    spike_idx = np.where((X[:-1] <= trigger) & (X[1:] > trigger))[0] + 1
    spike_times = grid[spike_idx]
    return spike_times, spike_idx


def plot_basis_functions(bfunc, sigfunc, afunc, xmin=-3.0, xmax=3.0, npts=1000):
    x = np.linspace(xmin, xmax, npts)

    bvals = bfunc(x)
    sigvals = sigfunc(x)
    avals = afunc(x)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(x, bvals, color="red", linewidth=2)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("bfunc")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("b(x)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, sigvals, color="red", linewidth=2)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("sigfunc")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$\sigma(x)$")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, avals, color="red", linewidth=2)
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("afunc")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("a(x)")
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_jumpdiff_with_spikes(
    grid, X, trigger, spike_times, spike_idx, hawkes_times=None
):
    """
    Left: jump-diffusion trajectory in blue, trigger in red dotted line
    Right: spike raster in red crosses

    If hawkes_times is given, the raster contains:
    - one row per Hawkes neuron
    - one extra row for the membrane neuron
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2.2, 1.2]}
    )

    # ---- Left panel: membrane potential ----
    axes[0].plot(grid, X, color="blue", linewidth=2.0, label=r"$X_t$")
    axes[0].axhline(trigger, color="red", linestyle=":", linewidth=2.0, label="trigger")

    if len(spike_times) > 0:
        axes[0].scatter(
            spike_times,
            X[spike_idx],
            color="red",
            marker="x",
            s=90,
            linewidths=2.5,
            zorder=3,
            label="spikes",
        )

    axes[0].set_xlim(grid[0], grid[-1])
    axes[0].set_xlabel("time")
    axes[0].set_ylabel(r"$X_t$")
    axes[0].set_title("Jump-diffusion membrane potential")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    # ---- Right panel: spikes ----
    if hawkes_times is None:
        if len(spike_times) > 0:
            axes[1].scatter(
                spike_times,
                np.ones(len(spike_times)),
                color="red",
                marker="x",
                s=90,
                linewidths=2.5,
            )
        axes[1].set_yticks([1])
        axes[1].set_yticklabels(["membrane"])
        ymax = 1
    else:
        M = len(hawkes_times)
        for j in range(M):
            tj = np.asarray(hawkes_times[j], dtype=float)
            if len(tj) > 0:
                axes[1].scatter(
                    tj,
                    np.full(len(tj), j + 1),
                    color="red",
                    marker="x",
                    s=70,
                    linewidths=2.2,
                )

        if len(spike_times) > 0:
            axes[1].scatter(
                spike_times,
                np.full(len(spike_times), M + 1),
                color="red",
                marker="x",
                s=95,
                linewidths=2.5,
            )

        axes[1].set_yticks(np.arange(1, M + 2))
        axes[1].set_yticklabels([f"Hawkes {j + 1}" for j in range(M)] + ["membrane"])
        ymax = M + 1

    axes[1].set_xlim(grid[0], grid[-1])
    axes[1].set_ylim(0.5, ymax + 0.5)
    axes[1].set_xlabel("time")
    axes[1].set_title("Spike raster")
    axes[1].grid(alpha=0.20)

    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(7)

    # --------------------------------------------------
    # 1) Multivariate Hawkes process
    # --------------------------------------------------
    M = 1
    Tend = 40.0

    xi = 0.28
    xi_vec = np.full(M, xi)

    beta0 = 0.5
    beta = np.full(M, beta0)

    alpha = np.array(
        [
            [0.00, 0.16, 0.08, 0.05],
            [0.10, 0.00, 0.12, 0.07],
            [0.06, 0.11, 0.00, 0.09],
            [0.04, 0.07, 0.13, 0.00],
        ]
    )

    param = [xi_vec, alpha, beta]
    times = hawkes.simuHawkesExpoM(param, M, Tend, xi)

    # --------------------------------------------------
    # 2) Euler grid and Hawkes jump increments
    # --------------------------------------------------
    ngrid = 5000
    grid = np.linspace(0.0, Tend, ngrid + 1)
    isjumpN, all_spikes = hawkes_to_isjumpN(times, grid)

    # --------------------------------------------------
    # 3) Generate basis functions and plot them
    # --------------------------------------------------
    bfunc, sigfunc, afunc, basis_params = generate_basis_functions(K=3, seed=12, m=0.0)
    plot_basis_functions(bfunc, sigfunc, afunc, xmin=-2.5, xmax=2.5)

    # --------------------------------------------------
    # 4) Simulate membrane potential
    # --------------------------------------------------
    X0 = 0.0
    X = simu_jumpdiff(X0, grid, bfunc, sigfunc, afunc, isjumpN)

    # --------------------------------------------------
    # 5) Threshold spikes
    # --------------------------------------------------
    trigger = 0.5
    spike_times, spike_idx = detect_threshold_spikes(X, grid, trigger)

    # --------------------------------------------------
    # 6) Plot process + raster
    # --------------------------------------------------
    plot_jumpdiff_with_spikes(
        grid=grid,
        X=X,
        trigger=trigger,
        spike_times=spike_times,
        spike_idx=spike_idx,
        hawkes_times=times,
    )


if __name__ == "__main__":
    main()
