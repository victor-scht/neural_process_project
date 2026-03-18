import argparse
import math
from dataclasses import dataclass
import hawkes
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
from scipy.stats import norm


# ============================================================
# Paper setup: Amorino et al. (simulation study, Section 6)
# ============================================================
# Default choices follow the paper's numerical study:
# - M = 1 Hawkes process
# - xi = 0.5, c = 0.4, alpha = 5
# - X0 = 2, lambda0 = xi
# - models (a), (b), (c), (d)
# - trigonometric basis, Nn = 20
# - kappa1 = kappa2 = 100
# - truncation beta = 1/4 + 0.01
# - estimation interval A = random data range


@dataclass
class PaperConfig:
    model: str = "b"
    n: int = 10000
    Delta: float = 0.01
    xi: float = 0.5
    hawkes_c: float = 0.4
    hawkes_alpha: float = 5.0
    X0: float = 2.0
    Nn: int = 20
    npas: int = 300
    kap_sigma: float = 100.0
    kap_g: float = 100.0
    beta_trunc: float = 1 / 4 + 0.01
    nrep_plot: int = 3
    seed: int = 7
    bandwidth: float | None = None
    do_cv_bandwidth: bool = True
    cv_bandwidth_grid: Tuple[float, ...] = (0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3)
    cv_max_points: int = 1200


# ============================================================
# Models from the paper
# ============================================================


def get_model_functions(model: str) -> Tuple[Callable, Callable, Callable]:
    model = model.lower()

    if model == "a":

        def bfunc(x):
            return -4.0 * np.asarray(x)

        def sigfunc(x):
            x = np.asarray(x)
            return np.ones_like(x, dtype=float)

        def afunc(x):
            x = np.asarray(x)
            return np.sqrt(2.0 + 0.5 * np.sin(x))

    elif model == "b":

        def bfunc(x):
            x = np.asarray(x)
            return -2.0 * x + np.sin(x)

        def sigfunc(x):
            x = np.asarray(x)
            return np.sqrt((3.0 + x**2) / (1.0 + x**2))

        def afunc(x):
            x = np.asarray(x)
            return np.ones_like(x, dtype=float)

    elif model == "c":

        def bfunc(x):
            return -2.0 * np.asarray(x)

        def sigfunc(x):
            x = np.asarray(x)
            return np.sqrt(1.0 + x**2)

        def afunc(x):
            x = np.asarray(x)
            return np.ones_like(x, dtype=float)

    elif model == "d":

        def bfunc(x):
            return -2.0 * np.asarray(x)

        def sigfunc(x):
            x = np.asarray(x)
            return np.sqrt(1.0 + x**2)

        def afunc(x):
            x = np.asarray(x)
            return np.clip(x, -5.0, 5.0)

    else:
        raise ValueError("model must be one of {'a', 'b', 'c', 'd'}")

    return bfunc, sigfunc, afunc


def estimate_conditional_expectation_f(X, grid, times, param, h=0.1, npas=250):
    """
    Estimate f_i(x) = E[ lambda_i(t) | X_t = x ] by Nadaraya-Watson.

    Parameters
    ----------
    X : array
        Jump-diffusion path, same length as grid
    grid : array
        Time grid
    times : list
        Hawkes spike times, one list per neuron
    param : [xi, alpha, beta]
        Hawkes parameters
    h : float
        Bandwidth for NW
    npas : int
        Number of x-points for plotting

    Returns
    -------
    gridx : array
        x-grid
    condiM : array, shape (M, npas)
        Estimated conditional expectations
    intensity : array, shape (len(grid)-1, M)
        Intensities evaluated on the time grid[:-1]
    """
    xi, alpha, beta = param
    M = len(xi)

    # Regressor: left endpoint X_ti
    Xreg = np.asarray(X[:-1], dtype=float)

    # Response: lambda(t_i)
    intensity = np.array(
        [hawkes.intensM(t, param, times) for t in grid[:-1]], dtype=float
    )
    intensity = intensity.reshape(len(grid) - 1, M)

    q1, q2 = np.min(Xreg), np.max(Xreg)
    gridx = np.linspace(q1, q2, npas)

    condiM = np.zeros((M, npas))
    for i in range(M):
        condiM[i, :] = mNW(x=gridx, X=Xreg, Y=intensity[:, i], h=h)

    return gridx, condiM, intensity


def plot_conditional_expectation_f(res: Dict, model: str):
    gridx = res["gridx"]
    keep = res["keep"]
    x = gridx[keep]

    plt.figure(figsize=(7, 4.8))
    plt.plot(x, res["fhat"][keep], color="purple", lw=2.0, label=r"$\hat f$")
    plt.title(rf"Model {model}: conditional expectation $f(x)$")
    plt.xlabel("x")
    plt.ylabel(r"$f(x)$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Hawkes process (M = 1)
# ============================================================


def hawkes_intensity_scalar(
    t: float, xi: float, c: float, alpha: float, jump_times: np.ndarray
) -> float:
    if jump_times.size == 0:
        return float(xi)
    tj = jump_times[jump_times < t]
    if tj.size == 0:
        return float(xi)
    return float(xi + np.sum(c * np.exp(-alpha * (t - tj))))


def simulate_hawkes_m1(
    xi: float, c: float, alpha: float, Tend: float, seed: int | None = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = 0.0
    jumps: List[float] = []

    while s < Tend:
        jumps_arr = np.asarray(jumps, dtype=float)
        lambda_bar = hawkes_intensity_scalar(s, xi, c, alpha, jumps_arr)
        if lambda_bar <= 0:
            break

        s += -math.log(rng.random()) / lambda_bar
        if s > Tend:
            break

        lambda_s = hawkes_intensity_scalar(s, xi, c, alpha, jumps_arr)
        if rng.random() * lambda_bar <= lambda_s:
            jumps.append(s)

    return np.asarray(jumps, dtype=float)


def hawkes_to_isjumpN(jump_times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    isjumpN, _ = np.histogram(jump_times, bins=grid)
    return isjumpN.astype(int)


# ============================================================
# Jump-diffusion simulation
# ============================================================


def simu_jumpdiff(
    X0: float,
    grid: np.ndarray,
    bfunc: Callable,
    sigfunc: Callable,
    afunc: Callable,
    isjumpN: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    W = rng.standard_normal(len(grid) - 1)
    X = np.zeros(len(grid), dtype=float)
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


# ============================================================
# Estimation helpers (port of the R utilities)
# ============================================================


def projectionSm(x: np.ndarray, q1: float, q2: float, m: int) -> np.ndarray:
    Dm = 2 * m + 1
    c = np.sqrt(2.0) / np.sqrt(q2 - q1)

    x = np.asarray(x, dtype=float)
    proj = np.zeros((Dm, len(x)), dtype=float)
    proj[0, :] = 1.0 / np.sqrt(q2 - q1)

    for l in range(1, m + 1):
        proj[2 * l - 1, :] = c * np.cos(2 * np.pi * l * (x - q1) / (q2 - q1))
        proj[2 * l, :] = c * np.sin(2 * np.pi * l * (x - q1) / (q2 - q1))
    return proj


def alphachapeau(P: np.ndarray, U: np.ndarray) -> np.ndarray:
    AA = P @ P.T
    B = P @ U.reshape(-1, 1)
    return solve(AA, B).flatten()


def collecestimcoeff(X, U, q1, q2, Nn):
    colleccoeffalpha = np.zeros((Nn, 2 * Nn + 1), dtype=float)
    collecP = []
    N = len(U)

    for k in range(1, Nn + 1):
        Pk = projectionSm(X[:N], q1, q2, k)  # was X[1:N+1]
        collecP.append(Pk)
        alpha_hat = np.linalg.lstsq(Pk.T, U, rcond=None)[0]
        colleccoeffalpha[k - 1, : 2 * k + 1] = alpha_hat

    return collecP, colleccoeffalpha


def penaltysig(Nn, n, kap):
    D = 2 * np.arange(1, Nn + 1) + 1
    return kap * D / n


def penaltyg(Nn, n, Delta, kap):
    D = 2 * np.arange(1, Nn + 1) + 1
    return kap * D / (n * Delta)


def adaptiveestim(
    colleccoeffalpha: np.ndarray,
    collecmatP: List[np.ndarray],
    U: np.ndarray,
    penalty: np.ndarray,
):
    ind = len(collecmatP)
    estimmhat = []
    criteremhat = np.zeros(ind, dtype=float)

    for l in range(ind):
        nrows = collecmatP[l].shape[0]
        est = np.sum(collecmatP[l] * colleccoeffalpha[l, :nrows][:, None], axis=0)
        estimmhat.append(est)
        criteremhat[l] = np.mean((U - est) ** 2)

    crit = criteremhat + penalty[:ind]
    mhat = int(np.argmin(crit))
    return estimmhat, criteremhat, crit, mhat


def phifunc(x: float) -> float:
    ax = abs(x)
    if ax < 1:
        return 1.0
    if ax >= 2:
        return 0.0
    return float(np.exp((1.0 / 3.0) + 1.0 / (x**2 - 4.0)))


def mNW(x, X: np.ndarray, Y: np.ndarray, h: float, K=norm.pdf):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if np.isscalar(x):
        weights = K((x - X) / h) / h
        s = weights.sum()
        if s <= 0:
            return float(np.mean(Y))
        weights /= s
        return float(np.dot(weights, Y))

    x = np.asarray(x, dtype=float)
    Kx = K((x[:, None] - X[None, :]) / h) / h
    rowsum = Kx.sum(axis=1)
    rowsum = np.where(rowsum <= 0, 1.0, rowsum)
    W = Kx / rowsum[:, None]
    return W @ Y


def select_bandwidth_loocv(
    X: np.ndarray,
    Y: np.ndarray,
    candidates: Tuple[float, ...],
    max_points: int = 1200,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if len(X) > max_points:
        idx = np.sort(rng.choice(len(X), size=max_points, replace=False))
        X = X[idx]
        Y = Y[idx]

    dmat = X[:, None] - X[None, :]
    best_h = candidates[0]
    best_score = np.inf

    for h in candidates:
        Kmat = norm.pdf(dmat / h) / h
        np.fill_diagonal(Kmat, 0.0)
        denom = Kmat.sum(axis=1)
        numer = Kmat @ Y
        pred = np.where(denom > 0, numer / denom, np.mean(Y))
        score = float(np.mean((Y - pred) ** 2))
        if score < best_score:
            best_score = score
            best_h = h

    return float(best_h)


def build_collection_estimates(
    gridx: np.ndarray,
    q1: float,
    q2: float,
    colleccoeffalpha: np.ndarray,
    ind: int,
    positive: bool = False,
) -> np.ndarray:
    collecestim = np.zeros((ind, len(gridx)), dtype=float)
    for k in range(ind):
        proj = projectionSm(gridx, q1, q2, k + 1)
        coeff = colleccoeffalpha[k, : 2 * (k + 1) + 1]
        vals = np.sum(proj * coeff[:, None], axis=0)
        if positive:
            vals = np.maximum(vals, 0.0)
        collecestim[k, :] = vals
    return collecestim


def build_eval_collection(
    Xeval: np.ndarray,
    q1: float,
    q2: float,
    colleccoeffalpha: np.ndarray,
    ind: int,
    positive: bool = False,
) -> List[np.ndarray]:
    out = []
    for k in range(ind):
        proj = projectionSm(Xeval, q1, q2, k + 1)
        coeff = colleccoeffalpha[k, : 2 * (k + 1) + 1]
        vals = np.sum(proj * coeff[:, None], axis=0)
        if positive:
            vals = np.maximum(vals, 0.0)
        out.append(vals)
    return out


# ============================================================
# One full experiment
# ============================================================


def run_one_experiment(
    config: PaperConfig, seed: int
) -> Dict[str, np.ndarray | float | int]:
    rng = np.random.default_rng(seed)
    bfunc, sigfunc, afunc = get_model_functions(config.model)

    grid = np.linspace(0.0, config.n * config.Delta, config.n + 1)
    Tend = float(grid[-1])

    jump_times = simulate_hawkes_m1(
        xi=config.xi,
        c=config.hawkes_c,
        alpha=config.hawkes_alpha,
        Tend=Tend,
        seed=seed + 1,
    )
    isjumpN = hawkes_to_isjumpN(jump_times, grid)
    X = simu_jumpdiff(
        X0=config.X0,
        grid=grid,
        bfunc=bfunc,
        sigfunc=sigfunc,
        afunc=afunc,
        isjumpN=isjumpN,
        rng=rng,
    )

    # Estimation interval A = random data range
    q1, q2 = float(X.min()), float(X.max())
    gridx = np.linspace(q1, q2, config.npas)
    qq1, qq2 = np.quantile(X, [0.05, 0.95])
    keep = (gridx > qq1) & (gridx < qq2)

    dX = np.diff(X)
    Tquad = dX**2 / config.Delta

    # sigma^2 estimation
    truncphi = np.array([phifunc(dx / (config.Delta**config.beta_trunc)) for dx in dX])
    Tquadphi = Tquad * truncphi

    collecP_sig, collecalphasig = collecestimcoeff(
        X=X, U=Tquadphi, q1=q1, q2=q2, Nn=config.Nn
    )
    ind_sig = len(collecP_sig)
    collecestimsig2 = build_collection_estimates(
        gridx, q1, q2, collecalphasig, ind_sig, positive=True
    )
    res_sig = adaptiveestim(
        collecalphasig,
        collecP_sig,
        Tquadphi,
        penaltysig(config.Nn, config.n, config.kap_sigma),
    )
    mhat_sig = res_sig[3]
    estim_sig2 = collecestimsig2[mhat_sig, :]

    # intensity and NW estimate of f(x) = E[lambda | X = x]
    intensity_on_grid = np.array(
        [
            hawkes_intensity_scalar(
                t, config.xi, config.hawkes_c, config.hawkes_alpha, jump_times
            )
            for t in grid[:-1]
        ]
    )

    X_pred = X[:-1]
    if config.bandwidth is None and config.do_cv_bandwidth:
        h = select_bandwidth_loocv(
            X_pred,
            intensity_on_grid,
            config.cv_bandwidth_grid,
            max_points=config.cv_max_points,
            seed=seed + 2,
        )
    else:
        h = 0.1 if config.bandwidth is None else float(config.bandwidth)

    fhat = mNW(gridx, X_pred, intensity_on_grid, h=h)
    fhat = np.maximum(fhat, 1e-10)

    # g estimation
    collecP_g, collecalphag = collecestimcoeff(X=X, U=Tquad, q1=q1, q2=q2, Nn=config.Nn)
    ind_g = len(collecP_g)
    collecestimg = build_collection_estimates(
        gridx, q1, q2, collecalphag, ind_g, positive=True
    )
    res_g = adaptiveestim(
        collecalphag,
        collecP_g,
        Tquad,
        penaltyg(config.Nn, config.n, config.Delta, config.kap_g),
    )
    mhat_g = res_g[3]
    estim_g = collecestimg[mhat_g, :]

    # Paper compares g-hat to g_tilde = sigma^2 + a^2 * fhat_NW
    true_sig2 = sigfunc(gridx) ** 2
    true_a = afunc(gridx)
    true_g_tilde = true_sig2 + true_a**2 * fhat

    estim_a2 = np.maximum((estim_g - estim_sig2) / fhat, 0.0)
    estim_a = np.sqrt(estim_a2)

    # Oracle dimensions in the paper's spirit
    Xeval = X[1:]
    sig2_eval = sigfunc(Xeval) ** 2
    a_eval = afunc(Xeval)
    fhat_eval = mNW(Xeval, X_pred, intensity_on_grid, h=h)
    true_g_eval = sig2_eval + a_eval**2 * fhat_eval

    sigma_eval_collection = build_eval_collection(
        Xeval, q1, q2, collecalphasig, ind_sig, positive=True
    )
    g_eval_collection = build_eval_collection(
        Xeval, q1, q2, collecalphag, ind_g, positive=True
    )

    sigma_oracle = int(
        np.argmin([np.mean((est - sig2_eval) ** 2) for est in sigma_eval_collection])
    )
    g_oracle = int(
        np.argmin([np.mean((est - true_g_eval) ** 2) for est in g_eval_collection])
    )

    return {
        "grid": grid,
        "gridx": gridx,
        "keep": keep,
        "X": X,
        "jump_times": jump_times,
        "isjumpN": isjumpN,
        "h": h,
        "q1": q1,
        "q2": q2,
        "estim_sig2": estim_sig2,
        "estim_sigma": np.sqrt(np.maximum(estim_sig2, 0.0)),
        "estim_g": estim_g,
        "estim_a": estim_a,
        "true_sig2": true_sig2,
        "true_sigma": sigfunc(gridx),
        "true_a": true_a,
        "true_g_tilde": true_g_tilde,
        "mhat_sig": mhat_sig + 1,
        "mhat_g": mhat_g + 1,
        "m_oracle_sig": sigma_oracle + 1,
        "m_oracle_g": g_oracle + 1,
        "fhat": fhat,
    }


# ============================================================
# Plotting
# ============================================================


def plot_path_and_spikes(res: Dict, model: str):
    grid = res["grid"]
    X = res["X"]
    jump_times = res["jump_times"]

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2.0, 1.0]}
    )

    axes[0].plot(grid, X, color="blue", lw=1.8)
    axes[0].set_title(f"Jump-diffusion path (model {model})")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel(r"$X_t$")
    axes[0].grid(alpha=0.3)

    if len(jump_times) > 0:
        axes[1].scatter(
            jump_times,
            np.ones_like(jump_times),
            color="red",
            marker="x",
            s=80,
            linewidths=2.0,
        )
    axes[1].set_xlim(grid[0], grid[-1])
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_yticks([1])
    axes[1].set_yticklabels(["Hawkes spikes"])
    axes[1].set_xlabel("time")
    axes[1].set_title("Spike train")
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_single_run_estimates(res: Dict, model: str):
    gridx = res["gridx"]
    keep = res["keep"]
    x = gridx[keep]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(
        x, res["estim_sig2"][keep], color="green", lw=2.0, label=r"$\hat\sigma^2$"
    )
    axes[0].plot(
        x,
        res["true_sig2"][keep],
        color="black",
        lw=2.0,
        linestyle=":",
        label=r"$\sigma^2$",
    )
    axes[0].set_title(rf"Model {model}: estimation of $\sigma^2$")
    axes[0].set_xlabel("x")
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 6)
    axes[0].legend()

    axes[1].plot(x, res["estim_g"][keep], color="green", lw=2.0, label=r"$\hat g$")
    axes[1].plot(
        x,
        res["true_g_tilde"][keep],
        color="black",
        lw=2.0,
        linestyle=":",
        label=r"$\tilde g$",
    )
    axes[1].set_title(rf"Model {model}: estimation of $\tilde g$")
    axes[1].set_xlabel("x")
    axes[1].set_ylim(0, 6)

    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(x, res["estim_a"][keep], color="red", lw=2.0, label=r"$\hat a$")
    axes[2].plot(x, res["true_a"][keep], color="blue", lw=2.0, label=r"$a$")
    axes[2].set_title(f"Model {model}: approximation of $a$")
    axes[2].set_xlabel("x")
    axes[2].grid(alpha=0.3)
    axes[1].set_ylim(0, 4)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def plot_three_realizations(config: PaperConfig):
    results = [
        run_one_experiment(config, seed=config.seed + 100 * k)
        for k in range(config.nrep_plot)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for res in results:
        x = res["gridx"][res["keep"]]
        axes[0].plot(x, res["estim_sig2"][res["keep"]], color="lightgreen", lw=1.8)
        axes[1].plot(x, res["estim_g"][res["keep"]], color="lightgreen", lw=1.8)

    ref = results[0]
    x = ref["gridx"][ref["keep"]]
    axes[0].plot(x, ref["true_sig2"][ref["keep"]], color="black", lw=2.0, linestyle=":")
    axes[1].plot(
        x, ref["true_g_tilde"][ref["keep"]], color="black", lw=2.0, linestyle=":"
    )

    axes[0].set_title(r"Three estimators of $\sigma^2$")
    axes[0].set_ylim(0, 6)
    axes[1].set_title(r"Three estimators of $\tilde g$")
    axes[1].set_ylim(0, 6)
    for ax in axes:
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Paper-style setup for self-exciting jump-diffusion estimation."
    )
    parser.add_argument("--model", type=str, default="b", choices=["a", "b", "c", "d"])
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--Delta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nrep-plot", type=int, default=3)
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument("--no-cv", action="store_true")
    args = parser.parse_args()

    config = PaperConfig(
        model=args.model,
        n=args.n,
        Delta=args.Delta,
        seed=args.seed,
        nrep_plot=args.nrep_plot,
        bandwidth=args.bandwidth,
        do_cv_bandwidth=not args.no_cv,
    )

    res = run_one_experiment(config, seed=config.seed)

    print(f"Model: {config.model}")
    print(f"n = {config.n}, Delta = {config.Delta}")
    print(
        f"Hawkes parameters: xi = {config.xi}, c = {config.hawkes_c}, alpha = {config.hawkes_alpha}"
    )
    print(f"Selected bandwidth h = {res['h']:.4f}")
    print(
        f"Selected dimension for sigma^2: {res['mhat_sig']} (oracle: {res['m_oracle_sig']})"
    )
    print(f"Selected dimension for g: {res['mhat_g']} (oracle: {res['m_oracle_g']})")
    print(f"Number of Hawkes spikes: {len(res['jump_times'])}")

    plot_path_and_spikes(res, config.model)
    plot_conditional_expectation_f(res, config.model)
    plot_single_run_estimates(res, config.model)

    if config.nrep_plot >= 2:
        plot_three_realizations(config)


if __name__ == "__main__":
    main()
