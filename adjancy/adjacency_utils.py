import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from tick.hawkes import HawkesADM4
import seaborn as sns

sns.set_style("whitegrid")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_events(data, t0=None, t1=None):
    events = []

    for trial in data:
        trial_events = []
        for neuron in trial:
            spikes = neuron[neuron > 0.0]

            if t0 is not None:
                spikes = spikes[spikes >= t0]
            if t1 is not None:
                spikes = spikes[spikes < t1]

            if t0 is not None:
                spikes = spikes - t0

            trial_events.append(spikes)

        events.append(trial_events)

    return events


def fit_adm4(events, decay_grid):
    best_score = -np.inf
    best_decay = None
    best_model = None
    scores = []

    for decay in tqdm(decay_grid):
        model = HawkesADM4(decay=decay, lasso_nuclear_ratio=1.0)
        model.fit(events)

        score = model.score(events)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_decay = decay
            best_model = model

    return best_model.adjacency, best_model.baseline, best_decay, scores


def plot_adjacency_matrix(A, path):
    ensure_dir(Path(path).parent)

    plt.figure(figsize=(6, 6))
    plt.imshow(A, cmap="Blues")
    plt.colorbar()
    plt.savefig(path)
    plt.close()


def plot_central_row(A, idx, trigger, path):
    row = A[idx]

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(row)), row, color="blue")
    plt.axhline(trigger, color="red", linestyle="--")
    plt.yscale("log")
    plt.savefig(path)
    plt.close()


def plot_trigger_curve(row, trigger_grid, path):
    counts = [np.sum(row > t) for t in trigger_grid]

    plt.figure(figsize=(6, 4))
    plt.plot(trigger_grid, counts, marker="o", c="blue")
    plt.xscale("log")
    plt.savefig(path)
    plt.close()

    return counts


def remove_central(selected, central_index, n):
    idx = central_index if central_index >= 0 else n + central_index
    return selected[selected != idx]
