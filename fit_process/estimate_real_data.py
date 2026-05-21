from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from fit_process.config import FitProcessConfig
from fit_process.utils import (
    intensM,
    mNW,
    collecestimcoeff,
    adaptiveestim,
    penaltyb,
    penaltyg,
    penaltysig,
    simu_jumpdiff,
    phifunc,
    projectionSm,
    ensure_dir,
    save_json,
    finish_plot,
)
from fit_process.data_loading import (
    load_processed_data,
    load_adjacency_selection,
    get_local_trial_index,
)


def build_jump_counts(grid: np.ndarray, spikesneurons):
    grid = np.asarray(grid, dtype=float)
    non_empty = [np.asarray(s, float) for s in spikesneurons if len(s) > 0]
    if len(non_empty) == 0:
        return np.zeros(len(grid), dtype=int)

    jumptimes = np.sort(np.concatenate(non_empty))
    left = np.searchsorted(jumptimes, grid[:-1], side="right")
    right = np.searchsorted(jumptimes, grid[1:], side="right")
    counts = (right - left).astype(int)

    isjumpN = np.zeros(len(grid), dtype=int)
    isjumpN[:-1] = counts
    return isjumpN


def estimate_baseline_from_spikes(spikesneurons, duration: float) -> np.ndarray:
    return np.asarray(
        [len(s) / max(duration, 1e-12) for s in spikesneurons], dtype=float
    )


def run_real_data_estimation(cfg: FitProcessConfig):
    out_dir = ensure_dir(cfg.output_path())

    ds = load_processed_data(cfg)
    adj = load_adjacency_selection(cfg)

    data_intra = ds["data_intra"]
    data_intra -= np.mean(data_intra)
    # data_intra /= np.std(data_intra)

    data_extra = ds["data_extra"]
    kept_trial_ids = ds["kept_trial_ids"]

    local_trial = get_local_trial_index(kept_trial_ids, cfg.original_trial_id)

    t0, t1 = cfg.shifted_interval
    model_duration = t1 - t0
    grid = np.arange(0.0, model_duration + cfg.Delta_model, cfg.Delta_model)

    selected_local = adj["selected_local"].astype(int)
    spikes_selected = [
        data_extra[local_trial, i][data_extra[local_trial, i] > 0.0]
        for i in selected_local
    ]

    adjM = np.asarray(adj["small_adjacency"], dtype=float)

    if adjM.shape[0] != len(spikes_selected):
        if adjM.shape[0] == len(spikes_selected) - 1:
            spikes_selected = spikes_selected[:-1]
        else:
            raise ValueError(
                f"Mismatch between selected.npy ({len(spikes_selected)} neurons) and "
                f"small adjacency shape {adjM.shape}."
            )

    baseline_small = adj["small_baseline"]
    if baseline_small is None:
        baseline_small = estimate_baseline_from_spikes(spikes_selected, model_duration)

    M = len(spikes_selected)
    paramhawkes = [baseline_small, adjM, np.repeat(cfg.hawkes_decay_value, M)]

    X_full = data_intra[local_trial]
    start_idx = int(t0 / cfg.Delta_intra)
    end_idx = int(t1 / cfg.Delta_intra)
    X_interval = X_full[start_idx:end_idx]
    X_real = X_interval[:: cfg.downsample_factor]

    if len(X_real) != len(grid):
        L = min(len(X_real), len(grid))
        X_real = X_real[:L]
        grid = grid[:L]

    X = X_real
    isjumpN = build_jump_counts(grid, spikes_selected)

    if cfg.use_synthetic_path:

        def bdrift(x):
            return -20.0 * x - 1080.0

        def sig(x):
            return 11.0

        def ajump(x):
            return -0.07 * x - 2.0

        rng = np.random.default_rng(cfg.synthetic_seed)
        X0 = float(rng.uniform(cfg.synthetic_X0_low, cfg.synthetic_X0_high))
        X = simu_jumpdiff(
            X0=X0,
            grid=grid,
            bfunc=bdrift,
            sigfunc=sig,
            afunc=ajump,
            isjumpN=isjumpN,
            rng=rng,
        )

    n = len(X) - 2
    q1, q2 = float(np.min(X)), float(np.max(X))
    gridx = np.linspace(q1, q2, cfg.npas)

    intensity = np.array(
        [intensM(s, paramhawkes, spikes_selected) for s in grid[: n + 1]]
    )

    condiM = np.zeros((M, cfg.npas))
    for i in range(M):
        condiM[i, :] = mNW(
            x=gridx,
            X=X[:-1],
            Y=intensity[: len(X) - 1, i],
            h=cfg.nw_bandwidth,
        )

    qq1, qq2 = np.quantile(X, [0.05, 0.95])
    keep = (gridx > qq1) & (gridx < qq2)

    Delta = cfg.Delta_model
    Tquad = np.diff(X[1:]) ** 2 / Delta

    collecP_g, collecalphag = collecestimcoeff(X=X, U=Tquad, q1=q1, q2=q2, Nn=cfg.Nn)
    penaltyg_vals = penaltyg(cfg.Nn, n, Delta, cfg.kap)

    collecestimg = np.zeros((cfg.Nn, len(gridx)))
    for k in range(cfg.Nn):
        proj = projectionSm(gridx, q1, q2, k + 1)
        coeff = collecalphag[k, : 2 * (k + 1) + 1]
        collecestimg[k, :] = np.maximum(np.sum(proj * coeff[:, None], axis=0), 0.0)

    _, _, _, mfinal_g0 = adaptiveestim(collecalphag, collecP_g, Tquad, penaltyg_vals)
    estimfinal_g = collecestimg[mfinal_g0, :]
    mfinal_g = mfinal_g0 + 1

    incr = np.diff(X[1:])
    truncphi = np.array(
        [
            phifunc(d / (cfg.truncation_threshold * (Delta**cfg.truncation_beta)))
            for d in incr
        ]
    )
    Tquadphi = Tquad * truncphi

    collecP_sig, collecalphasig = collecestimcoeff(
        X=X, U=Tquadphi, q1=q1, q2=q2, Nn=cfg.Nn
    )
    penaltysig_vals = penaltysig(cfg.Nn, n, cfg.kap)

    collecestimsig = np.zeros((cfg.Nn, len(gridx)))
    for k in range(cfg.Nn):
        proj = projectionSm(gridx, q1, q2, k + 1)
        coeff = collecalphasig[k, : 2 * (k + 1) + 1]
        collecestimsig[k, :] = np.maximum(np.sum(proj * coeff[:, None], axis=0), 0.0)

    _, _, _, mfinal_sig0 = adaptiveestim(
        collecalphasig, collecP_sig, Tquadphi, penaltysig_vals
    )
    estimfinal_sig = collecestimsig[mfinal_sig0, :]
    mfinal_sig = mfinal_sig0 + 1

    sumcondiM = np.sum(condiM, axis=0)
    sumcondiM = np.maximum(sumcondiM, 1e-12)
    estim_a2 = (estimfinal_g - estimfinal_sig) / sumcondiM
    estim_a = np.sqrt(np.maximum(estim_a2, 0.0))

    sigmean = float(np.sqrt(np.mean(estimfinal_sig[keep])))

    def sigesti(x):
        x = np.asarray(x)
        return np.full_like(x, sigmean, dtype=float)

    reg_a = LinearRegression(fit_intercept=True).fit(
        gridx[keep].reshape(-1, 1), estim_a[keep]
    )

    def aesti(x):
        x = np.asarray(x).reshape(-1, 1)
        return reg_a.predict(x)

    aestiX = aesti(X).reshape(-1)
    Y = np.diff(X[1:]) / Delta
    jump_counts_for_Y = isjumpN[1 : 1 + n]
    termT = (aestiX[1 : 1 + n] * jump_counts_for_Y) / Delta
    U = Y - termT

    sigma02 = float(np.max(estimfinal_sig[keep]))
    collecP_b, collecalphab = collecestimcoeff(X=X, U=U, q1=q1, q2=q2, Nn=cfg.Nn)

    collecestimb = np.zeros((cfg.Nn, len(gridx)))
    for k in range(cfg.Nn):
        proj = projectionSm(gridx, q1, q2, k + 1)
        coeff = collecalphab[k, : 2 * (k + 1) + 1]
        collecestimb[k, :] = np.sum(proj * coeff[:, None], axis=0)

    penaltyb_vals = np.array(
        [penaltyb(k + 1, n, Delta, cfg.rho, sigma02) for k in range(cfg.Nn)]
    )
    _, _, _, mfinal_b0 = adaptiveestim(collecalphab, collecP_b, U, penaltyb_vals)
    estim_b = collecestimb[mfinal_b0, :]
    mfinal_b = mfinal_b0 + 1

    reg_b = LinearRegression().fit(gridx.reshape(-1, 1), estim_b)

    def besti(x):
        x = np.asarray(x).reshape(-1, 1)
        return reg_b.predict(x)

    X_est = simu_jumpdiff(
        X0=float(X[0]),
        grid=grid,
        bfunc=lambda x: besti([x])[0],
        sigfunc=lambda x: sigesti(np.array([x]))[0],
        afunc=lambda x: aesti([x])[0],
        isjumpN=isjumpN,
        rng=np.random.default_rng(2024),
    )

    curves = pd.DataFrame(
        {
            "x": gridx,
            "g_hat": estimfinal_g,
            "sigma2_hat": estimfinal_sig,
            "a_hat": estim_a,
            "b_hat": estim_b,
        }
    )
    curves.to_csv(out_dir / "estimated_curves.csv", index=False)

    save_json(
        {
            "original_trial_id": int(cfg.original_trial_id),
            "local_trial_index": int(local_trial),
            "shifted_interval": [float(t0), float(t1)],
            "n_selected_neurons_from_selected_npy": int(len(selected_local)),
            "n_used_neurons_after_alignment": int(M),
            "m_g": int(mfinal_g),
            "m_sig": int(mfinal_sig),
            "m_b": int(mfinal_b),
            "sigma_mean": float(sigmean),
            "grid_length": int(len(grid)),
            "X_length": int(len(X)),
            "truncated_count": int(np.sum(truncphi != 1.0)),
        },
        out_dir / "estimation_summary.json",
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, neuron_spikes in enumerate(spikes_selected, start=1):
        ax.scatter(
            neuron_spikes, np.full(len(neuron_spikes), i), marker="+", color="red"
        )
    ax.set_title("Selected spike trains used for fit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron index")
    ax.grid(alpha=0.3)
    finish_plot(fig, out_dir / "selected_spike_trains.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for i in range(min(4, intensity.shape[1])):
        ax = axes.flat[i]
        ax.plot(grid[:n], intensity[:n, i], color="blue")
        ax.grid(alpha=0.3)
        ax.set_title(f"Intensity neuron {i + 1}")
    finish_plot(fig, out_dir / "intensities_overview.png")

    for y, name in [
        (estimfinal_g, "g_estimation.png"),
        (estimfinal_sig, "sigma2_estimation.png"),
        (estim_a, "a_estimation.png"),
        (estim_b, "b_estimation.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(gridx, y, color="blue")
        ax.set_xlim(float(qq1), float(qq2))
        ax.grid(alpha=0.3)
        ax.set_title(name.replace("_", " ").replace(".png", ""))
        finish_plot(fig, out_dir / name)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(grid, X, color="blue", lw=1.5, label="Observed")
    ax.plot(grid, X_est, color="red", lw=1.2, label="Estimated-process simulation")
    ax.set_title("Observed vs estimated process")
    ax.grid(alpha=0.3)
    ax.legend()
    finish_plot(fig, out_dir / "observed_vs_estimated_process.png")

    report = f"""# Fit process report

## Inputs
- processed dataset only: {cfg.processed_dir}
- selected neurons loaded from: {cfg.adjacency_dir}/selection/selected.npy
- small adjacency loaded from: {cfg.adjacency_dir}/small/interval_2/adjacency_small.npy

## Data selection
- original trial id: {cfg.original_trial_id}
- local trial index: {local_trial}
- shifted interval: [{t0}, {t1}]

## Used neurons
- selected.npy count: {len(selected_local)}
- actually used after alignment with the small adjacency matrix: {M}

## Estimated model dimensions
- m_g = {mfinal_g}
- m_sig = {mfinal_sig}
- m_b = {mfinal_b}
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    return {
        "grid": grid,
        "X": X,
        "X_est": X_est,
        "spikesneurons": spikes_selected,
        "gridx": gridx,
        "g": estimfinal_g,
        "sig2": estimfinal_sig,
        "a": estim_a,
        "b": estim_b,
        "m_g": mfinal_g,
        "m_sig": mfinal_sig,
        "m_b": mfinal_b,
    }
