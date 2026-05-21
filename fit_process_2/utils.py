from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear
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


def penaltyb(m: int, n: int, Delta: float, rho: float, sigma02: float) -> float:
    return rho * m * sigma02 / max(n * Delta, 1e-12)


def penaltyg(Nn: int, n: int, Delta: float, kap: float) -> Array:
    return kap * np.arange(1, Nn + 1, dtype=float) / max(n * Delta, 1e-12)


def penaltysig(Nn: int, n: int, kap: float) -> Array:
    return kap * np.arange(1, Nn + 1, dtype=float) / max(n, 1)


def _augmented_knot_vector(q1: float, q2: float, n_basis: int, degree: int) -> Array:
    if q2 <= q1:
        q2 = q1 + 1e-6
    n_internal = max(n_basis - degree - 1, 0)
    if n_internal > 0:
        internal = np.linspace(q1, q2, n_internal + 2)[1:-1]
    else:
        internal = np.array([], dtype=float)
    knots = np.concatenate(
        [np.repeat(q1, degree + 1), internal, np.repeat(q2, degree + 1)]
    )
    return knots.astype(float)


def spline_design_matrix(
    x: Array,
    q1: float,
    q2: float,
    n_basis: int,
    degree: int,
) -> Array:
    x = np.asarray(x, dtype=float)
    x_clip = np.clip(x, q1, q2)
    knots = _augmented_knot_vector(q1, q2, n_basis, degree)
    B = np.zeros((len(x_clip), n_basis), dtype=float)
    for j in range(n_basis):
        coeff = np.zeros(n_basis, dtype=float)
        coeff[j] = 1.0
        spline = BSpline(knots, coeff, degree, extrapolate=False)
        values = spline(x_clip)
        values = np.where(np.isfinite(values), values, 0.0)
        B[:, j] = values
    return B


def fit_positive_spline_least_squares(
    x: Array,
    y: Array,
    q1: float,
    q2: float,
    n_basis: int,
    degree: int,
    floor: float = 1e-8,
    ridge: float = 1e-8,
    sample_weight: Optional[Array] = None,
) -> Dict[str, Array | float | int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    B = spline_design_matrix(x, q1, q2, n_basis, degree)

    if sample_weight is None:
        w = np.ones(len(x), dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)
        w = np.maximum(w, 1e-12)

    sqrtw = np.sqrt(w)
    A = B * sqrtw[:, None]
    target = y * sqrtw

    if ridge > 0.0:
        A = np.vstack([A, np.sqrt(ridge) * np.eye(n_basis)])
        target = np.concatenate([target, np.sqrt(ridge) * np.full(n_basis, floor)])

    res = lsq_linear(A, target, bounds=(floor, np.inf), lsmr_tol="auto", verbose=0)
    coeff = res.x
    fit = B @ coeff
    mse = float(np.mean((y - fit) ** 2))
    return {
        "coeff": coeff,
        "fit": fit,
        "design": B,
        "mse": mse,
        "success": bool(res.success),
        "cost": float(res.cost),
    }


def fit_unconstrained_spline_least_squares(
    x: Array,
    y: Array,
    q1: float,
    q2: float,
    n_basis: int,
    degree: int,
    ridge: float = 1e-8,
    sample_weight: Optional[Array] = None,
) -> Dict[str, Array | float | int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    B = spline_design_matrix(x, q1, q2, n_basis, degree)

    if sample_weight is None:
        w = np.ones(len(x), dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)
        w = np.maximum(w, 1e-12)

    sqrtw = np.sqrt(w)
    A = B * sqrtw[:, None]
    target = y * sqrtw

    if ridge > 0.0:
        A = np.vstack([A, np.sqrt(ridge) * np.eye(n_basis)])
        target = np.concatenate([target, np.zeros(n_basis)])

    coeff, _, _, _ = lstsq(A, target, rcond=None)
    fit = B @ coeff
    mse = float(np.mean((y - fit) ** 2))
    return {
        "coeff": coeff,
        "fit": fit,
        "design": B,
        "mse": mse,
    }


def evaluate_spline_fit(
    x: Array,
    q1: float,
    q2: float,
    coeff: Array,
    degree: int,
) -> Array:
    coeff = np.asarray(coeff, dtype=float)
    B = spline_design_matrix(x, q1, q2, len(coeff), degree)
    return B @ coeff


def adaptive_positive_spline_fit(
    x_train: Array,
    y_train: Array,
    x_eval: Array,
    q1: float,
    q2: float,
    min_basis: int,
    max_basis: int,
    degree: int,
    penalty: Array,
    floor: float = 1e-8,
    ridge: float = 1e-8,
    sample_weight: Optional[Array] = None,
) -> Dict[str, object]:
    fits: List[Dict[str, object]] = []
    criteria = []
    basis_list = []

    for n_basis in range(min_basis, max_basis + 1):
        res = fit_positive_spline_least_squares(
            x=x_train,
            y=y_train,
            q1=q1,
            q2=q2,
            n_basis=n_basis,
            degree=degree,
            floor=floor,
            ridge=ridge,
            sample_weight=sample_weight,
        )
        eval_fit = evaluate_spline_fit(x_eval, q1, q2, res["coeff"], degree)
        fit_info = {
            "n_basis": n_basis,
            "coeff": res["coeff"],
            "train_fit": res["fit"],
            "eval_fit": eval_fit,
            "mse": res["mse"],
            "success": res["success"],
        }
        fits.append(fit_info)
        basis_list.append(n_basis)
        penalty_index = min(n_basis - 1, len(penalty) - 1)
        criteria.append(float(res["mse"] + penalty[penalty_index]))

    criteria = np.asarray(criteria, dtype=float)
    idx = int(np.argmin(criteria))
    return {
        "fits": fits,
        "basis_list": basis_list,
        "criteria": criteria,
        "best_index": idx,
        "best_fit": fits[idx],
    }


def adaptive_unconstrained_spline_fit(
    x_train: Array,
    y_train: Array,
    x_eval: Array,
    q1: float,
    q2: float,
    min_basis: int,
    max_basis: int,
    degree: int,
    penalty: Array,
    ridge: float = 1e-8,
    sample_weight: Optional[Array] = None,
) -> Dict[str, object]:
    fits: List[Dict[str, object]] = []
    criteria = []
    basis_list = []

    for n_basis in range(min_basis, max_basis + 1):
        res = fit_unconstrained_spline_least_squares(
            x=x_train,
            y=y_train,
            q1=q1,
            q2=q2,
            n_basis=n_basis,
            degree=degree,
            ridge=ridge,
            sample_weight=sample_weight,
        )
        eval_fit = evaluate_spline_fit(x_eval, q1, q2, res["coeff"], degree)
        fit_info = {
            "n_basis": n_basis,
            "coeff": res["coeff"],
            "train_fit": res["fit"],
            "eval_fit": eval_fit,
            "mse": res["mse"],
        }
        fits.append(fit_info)
        basis_list.append(n_basis)
        penalty_index = min(n_basis - 1, len(penalty) - 1)
        criteria.append(float(res["mse"] + penalty[penalty_index]))

    criteria = np.asarray(criteria, dtype=float)
    idx = int(np.argmin(criteria))
    return {
        "fits": fits,
        "basis_list": basis_list,
        "criteria": criteria,
        "best_index": idx,
        "best_fit": fits[idx],
    }
