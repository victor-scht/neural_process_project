import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def intensM(t, param, times):
    """
    Intensity function for M Hawkes processes (exponential kernel)

    param: [xi, alpha, beta]
        xi    : array of size M
        alpha : M x M matrix
        beta  : array of size M
    times: list of length M, spike times for each neuron
    """
    xi, alpha, beta = param
    M = len(xi)

    intenst = np.array(xi, dtype=float).copy()

    for i in range(M):
        for j in range(M):
            tj = np.array(times[j], dtype=float)
            tj = tj[tj < t]
            if len(tj) > 0:
                intenst[i] += np.sum(
                    alpha[i, j] * beta[i] * np.exp(-beta[i] * (t - tj))
                )

    return intenst


def intens(k, M, s, times, param, xi):
    """
    It sums all intensities up to the kth neuron
    """
    _, alpha, beta = param
    vect = 0.0

    for m in range(k):
        for l in range(M):
            tl = np.array(times[l], dtype=float)
            tl = tl[tl < s]
            if len(tl) > 0:
                vect += np.sum(alpha[m, l] * beta[m] * np.exp(-beta[m] * (s - tl)))

    vect += k * xi
    return vect


def simuHawkesExpoM(param, M, Tend, xi):
    """
    Simulation of a multi-neuron exponential Hawkes process
    using Ogata's thinning method.
    """
    times = [[] for _ in range(M)]
    s = 0.0

    while s < Tend:
        lambda_bar = intens(M, M, s, times, param, xi)
        if lambda_bar <= 0:
            break

        u = np.random.rand()
        w = -np.log(u) / lambda_bar
        s += w

        if s > Tend:
            break

        current_intensities = []
        for k in range(M):
            lam_k = intens(k + 1, M, s, times, param, xi) - intens(
                k, M, s, times, param, xi
            )
            current_intensities.append(lam_k)

        current_intensities = np.array(current_intensities, dtype=float)
        total_intensity = np.sum(current_intensities)

        D = np.random.rand()
        if D * lambda_bar <= total_intensity:
            probs = current_intensities / total_intensity
            neuron = np.random.choice(M, p=probs)
            times[neuron].append(s)

    return times


def plotHawkesM(times, param, Tend, xi, save=False):
    """
    Plot:
    1) adjacency matrix
    2) raster plot with red thick crosses
    3) intensities over time
    """
    xi_vec, alpha, beta = param
    M = len(xi_vec)

    palette = [
        "red",
        "blue",
        "lightgreen",
        "violet",
        "orange",
        "cyan",
        "magenta",
        "gold",
    ]

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 13), gridspec_kw={"height_ratios": [1.1, 1.0, 1.6]}
    )

    # --- Adjacency matrix ---
    sns.heatmap(
        alpha,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.5,
        ax=axes[0],
        cbar=True,
    )
    axes[0].set_title("Adjacency matrix")
    axes[0].set_xlabel("Source neuron j")
    axes[0].set_ylabel("Target neuron i")

    # --- Raster plot ---
    for i in range(M):
        if len(times[i]) > 0:
            y = np.full(len(times[i]), i + 1)
            axes[1].scatter(times[i], y, color="red", marker="x", s=90, linewidths=2.5)

    axes[1].set_xlim(0, Tend)
    axes[1].set_ylim(0.5, M + 0.5)
    axes[1].set_yticks(np.arange(1, M + 1))
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Neuron")
    axes[1].set_title("Spike raster")
    axes[1].grid(alpha=0.25)

    # --- Intensity curves ---
    t_grid = np.linspace(0, Tend, 1200)
    lam_grid = np.array([intensM(t, param, times) for t in t_grid])

    for i in range(M):
        axes[2].plot(
            t_grid,
            lam_grid[:, i],
            color=palette[i % len(palette)],
            linewidth=2,
            label=f"Neuron {i + 1}",
        )

    axes[2].set_xlim(0, Tend)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Intensity")
    axes[2].set_title(f"Intensities (baseline xi = {xi}, beta = {beta[0]})")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    plt.tight_layout()

    if save:
        plt.savefig("./figures/hawkes_simu.png", dpi=200)
    plt.show()


def main():
    np.random.seed(7)

    M = 4
    Tend = 5.0

    # same baseline for every neuron
    xi = 0.35
    xi_vec = np.full(M, xi)

    # same beta for every neuron
    beta0 = 20.0
    beta = np.full(M, beta0)

    # adjacency matrix
    alpha = np.array(
        [
            [0.00, 0.18, 0.10, 0.06],
            [0.12, 0.00, 0.14, 0.08],
            [0.07, 0.15, 0.00, 0.10],
            [0.05, 0.09, 0.16, 0.00],
        ]
    )

    param = [xi_vec, alpha, beta]

    times = simuHawkesExpoM(param, M, Tend, xi)

    print("Adjacency matrix:")
    print(alpha)
    print("\nBaseline xi:", xi)
    print("Beta:", beta0)
    for i in range(M):
        print(f"Neuron {i + 1}: {len(times[i])} spikes")

    plotHawkesM(times, param, Tend, xi, save=True)


if __name__ == "__main__":
    main()
