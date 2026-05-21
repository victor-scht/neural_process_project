from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from build_dataset.utils import ensure_dir


from typing import Sequence

sns.set_style("whitegrid")


def _finish(fig, path=None, show=False):
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        ensure_dir(path.parent)
        fig.savefig(path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_spike_raster(data_extra: np.ndarray, trial: int, path=None, show=False):
    spikes = data_extra[trial]
    fig, ax = plt.subplots(figsize=(10, 5))
    for neuron_id, neuron_spikes in enumerate(spikes):
        valid = neuron_spikes[neuron_spikes > 0.0]
        ax.scatter(valid, np.full_like(valid, neuron_id), s=2, color="black")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron ID")
    ax.set_title(f"Trial {trial} - Spike raster")
    ax.grid(alpha=0.2)
    _finish(fig, path, show)


def plot_overlay(
    data_intra: np.ndarray,
    data_extra: np.ndarray,
    delta: float,
    trial: int,
    neuron_index: int = -1,
    path=None,
    show=False,
):
    signal = data_intra[trial]
    spikes = data_extra[trial, neuron_index]
    valid = spikes[spikes > 0.0]
    time = np.arange(len(signal)) * delta

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time, signal, color="blue", lw=1, label="Membrane")
    ax.vlines(
        valid,
        ymin=float(signal.min()),
        ymax=float(signal.max()),
        color="red",
        alpha=0.5,
        label="Spikes",
    )
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Membrane + spikes overlay (trial {trial}, neuron {neuron_index})")
    ax.legend()
    ax.grid(alpha=0.3)
    _finish(fig, path, show)


def plot_counts_bar(
    values: Sequence[int],
    title: str,
    xlabel: str,
    path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(values)), values, color="blue", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Spike count")
    fig.tight_layout()
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_psth_trial(
    data_extra: np.ndarray,
    trial: int,
    bins: int = 100,
    path=None,
):
    """
    Plot PSTH (spike histogram over time) for one trial.
    """

    spikes = data_extra[trial]

    # --- flatten all spike times ---
    all_spikes = spikes[spikes > 0.0]

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.histplot(all_spikes, bins=bins, color="blue", ax=ax, alpha=0.5)

    ax.set_title(f"PSTH (trial {trial})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spike count")
    ax.grid(alpha=0.3)

    _finish(fig, path)
