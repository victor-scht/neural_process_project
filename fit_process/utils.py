from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
from scipy.stats import norm

Array = np.ndarray


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def intensM(t: float, paramhawkes: Sequence[Array], spikes: Sequence[Array]) -> Array:
    base, adjM, beta = paramhawkes
    base = np.asarray(base, dtype=float).reshape(-1)
    adjM = np.asarray(adjM, dtype=float)
    beta = np.asarray(beta, dtype=float).reshape(-1)

    lam = base.copy()
    M = len(base)

    for j in range(M):
        tj = np.asarray(spikes[j], dtype=float)
        if tj.size == 0:
            continue
        tj = tj[tj < t]
        if tj.size == 0:
            continue
        dt = t - tj
        kern_sum = np.exp(-beta[:, None] * dt[None, :]).sum(axis=1)
        lam += adjM[:, j] * kern_sum

    return np.maximum(lam, 1e-12)


def projectionSm(x: Array, q1: float, q2: float, m: int) -> Array:
    Dm = 2 * m + 1
    c = np.sqrt(2.0) / np.sqrt(q2 - q1)

    x = np.asarray(x, dtype=float)
    proj = np.zeros((Dm, len(x)), dtype=float)
    proj[0, :] = 1.0 / np.sqrt(q2 - q1)

    z = (x - q1) / (q2 - q1)
    for l in range(1, m + 1):
        angle = 2.0 * np.pi * l * z
        proj[2 * l - 1, :] = c * np.cos(angle)
        proj[2 * l, :] = c * np.sin(angle)
    return proj


def alphachapeau(P: Array, U: Array) -> Array:
    AA = P @ P.T
    B = P @ U.reshape(-1, 1)
    return solve(AA, B).flatten()


def collecestimcoeff(X: Array, U: Array, q1: float, q2: float, Nn: int):
    colleccoeffalpha = np.zeros((Nn, 2 * Nn + 1), dtype=float)
    collecP = []

    for k in range(1, Nn + 1):
        Pk = projectionSm(X[1 : len(U) + 1], q1, q2, k)
        collecP.append(Pk)
        try:
            alpha_hat = alphachapeau(Pk, U)
            colleccoeffalpha[k - 1, : 2 * k + 1] = alpha_hat
        except Exception:
            break

    return collecP, colleccoeffalpha


def penaltyb(m: int, n: int, Delta: float, rho: float, sigma02: float) -> float:
    return rho * (2 * m + 1) * sigma02 / (n * Delta)


def penaltyg(Nn: int, n: int, Delta: float, kap: float) -> Array:
    return kap * np.arange(1, Nn + 1) / (n * Delta)


def penaltysig(Nn: int, n: int, kap: float) -> Array:
    return kap * np.arange(1, Nn + 1) / n


def adaptiveestim(
    colleccoeffalpha: Array, collecmatP: List[Array], U: Array, penalty: Array
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


def mNW(x: Array, X: Array, Y: Array, h: float, K=norm.pdf) -> Array:
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
    res = []
    for xi in x:
        weights = K((xi - X) / h) / h
        s = weights.sum()
        if s <= 0:
            res.append(np.mean(Y))
        else:
            weights /= s
            res.append(np.dot(weights, Y))
    return np.asarray(res, dtype=float)


def simu_jumpdiff(
    X0: float,
    grid: Array,
    bfunc,
    sigfunc,
    afunc,
    isjumpN: Array,
    rng: Optional[np.random.Generator] = None,
) -> Array:
    if rng is None:
        rng = np.random.default_rng()

    grid = np.asarray(grid, dtype=float)
    X = np.zeros(len(grid), dtype=float)
    X[0] = X0

    jumps = np.asarray(isjumpN, dtype=float)
    if len(jumps) == len(grid):
        jumps = jumps[:-1]

    W = rng.standard_normal(len(grid) - 1)

    for i in range(len(grid) - 1):
        dt = grid[i + 1] - grid[i]
        X[i + 1] = (
            X[i]
            + dt * bfunc(X[i])
            + np.sqrt(dt) * sigfunc(X[i]) * W[i]
            + afunc(X[i]) * jumps[i]
        )
    return X


def finish_plot(fig, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
