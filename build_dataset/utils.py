from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat


# ---------------------------------------------------------------------
# generic io
# ---------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config(config: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)


def load_config(path: str | Path, config_class: type) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return config_class(**data)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_text(text: str, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def save_vector_csv(column_name: str, values: Sequence[Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    pd.DataFrame({column_name: list(values)}).to_csv(path, index=False)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------
# time helpers
# ---------------------------------------------------------------------
def interval_to_indices(t_start: float, t_end: float, delta: float) -> Tuple[int, int]:
    """
    Half-open interval [t_start, t_end)
    """
    i0 = int(np.floor(t_start / delta))
    i1 = int(np.floor(t_end / delta))
    return i0, i1


def time_axis_from_length(length: int, delta: float) -> np.ndarray:
    return np.arange(length, dtype=float) * delta


# ---------------------------------------------------------------------
# raw data loading
# ---------------------------------------------------------------------
def load_intra_data(file_path: str | Path) -> np.ndarray:
    return loadmat(file_path)["nerve"][:, 0]


def load_extra_data(file_path: str | Path) -> np.ndarray:
    # Keep all columns as loaded from CSV.
    # The user said the last neuron is the central neuron.
    data = pd.read_csv(file_path, index_col=0)
    return np.asarray(data)


def load_all_intra(config: Any) -> np.ndarray:
    data: List[np.ndarray] = []
    max_len = 0
    for trial in range(1, config.n_trials + 1):
        path = Path(config.intra_dir) / config.intra_pattern.format(trial=trial)
        raw = load_intra_data(path)
        max_len = max(max_len, len(raw))
        data.append(raw)
    padded = [np.pad(x, (0, max_len - len(x)), mode="constant") for x in data]
    return np.stack(padded)


def load_all_extra(config: Any) -> np.ndarray:
    data: List[np.ndarray] = []
    max_cols = 0
    for trial in range(1, config.n_trials + 1):
        path = Path(config.extra_dir) / config.extra_pattern.format(trial=trial)
        raw = load_extra_data(path)
        max_cols = max(max_cols, raw.shape[1])
        data.append(raw)
    padded: List[np.ndarray] = []
    for raw in data:
        rows, cols = raw.shape
        padded.append(np.pad(raw, ((0, 0), (0, max_cols - cols)), mode="constant"))
    return np.stack(padded)


# ---------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------
def clean_extra(data_extra: np.ndarray, trial_duration: float) -> np.ndarray:
    """
    Extracellular timestamps are stored trial-by-trial in a global clock:
    trial k is offset by k * trial_duration.
    """
    n_trials = data_extra.shape[0]
    offsets = np.arange(n_trials, dtype=float).reshape(n_trials, 1, 1) * trial_duration
    return np.where(data_extra > 0.0, data_extra - offsets, data_extra)


def filter_intra(
    data_intra: np.ndarray, t_start: float, t_end: float, delta: float
) -> np.ndarray:
    i0, i1 = interval_to_indices(t_start, t_end, delta)
    i0 = max(0, i0)
    i1 = min(data_intra.shape[1], i1)
    return data_intra[:, i0:i1]


def filter_extra(data_extra: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
    mask = (data_extra >= t_start) & (data_extra < t_end)
    return np.where(mask, data_extra, 0.0)


def shift_valid_spike_times(data_extra: np.ndarray, t0: float) -> np.ndarray:
    out = data_extra.copy()
    mask = out > 0.0
    out[mask] -= t0
    return out


# ---------------------------------------------------------------------
# spike detection / comparison
# ---------------------------------------------------------------------
def detect_spikes_from_potential(
    signal: np.ndarray, delta: float, threshold: float
) -> np.ndarray:
    """
    Detect spikes as local maxima above a threshold.

    A spike is defined as a point i such that:
    - signal[i] > threshold
    - signal[i] is a local maximum
    """

    # Find candidate points above threshold
    above = signal > threshold

    # Local maxima: derivative sign change
    d = np.diff(signal)

    # peak when slope goes from + to -
    peaks = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1

    # Keep only peaks above threshold
    peaks = peaks[above[peaks]]

    return peaks.astype(float) * delta


def detect_all_intra_spikes(
    data_intra: np.ndarray, delta: float, threshold: float
) -> List[np.ndarray]:
    return [
        detect_spikes_from_potential(trial, delta, threshold) for trial in data_intra
    ]


def window_and_shift_spike_lists(
    spikes_by_trial: Sequence[np.ndarray], t_start: float, t_end: float
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for spikes in spikes_by_trial:
        kept = spikes[(spikes >= t_start) & (spikes < t_end)] - t_start
        out.append(kept)
    return out


def trial_spike_times_to_padded_matrix(
    spike_times_by_trial: Sequence[np.ndarray],
) -> np.ndarray:
    max_len = max((len(x) for x in spike_times_by_trial), default=0)
    if max_len == 0:
        return np.zeros((len(spike_times_by_trial), 0), dtype=float)
    rows = [
        np.pad(x, (0, max_len - len(x)), mode="constant") for x in spike_times_by_trial
    ]
    return np.stack(rows)


def padded_matrix_to_list(data: np.ndarray) -> List[np.ndarray]:
    return [row[row > 0.0] for row in data]


def get_central_spikes(
    data_extra: np.ndarray, central_index: int = -1
) -> List[np.ndarray]:
    return [trial[central_index][trial[central_index] > 0.0] for trial in data_extra]


# ---------------------------------------------------------------------
# pruning
# ---------------------------------------------------------------------
def spike_counts_by_trial_and_neuron(data_extra: np.ndarray) -> np.ndarray:
    return np.sum(data_extra > 0.0, axis=2)


def find_inactive_neurons(data_extra: np.ndarray) -> List[int]:
    counts = spike_counts_by_trial_and_neuron(data_extra)
    return np.where(np.sum(counts, axis=0) == 0)[0].astype(int).tolist()


def find_trials_without_target_spikes(data_inferred: np.ndarray) -> List[int]:
    out = []
    for i, spikes in enumerate(data_inferred):
        if not np.any(spikes > 0.0):
            out.append(i)
    return out


def remove_neurons(
    data_extra: np.ndarray, neuron_ids: np.ndarray, neuron_indices: Sequence[int]
):
    if not neuron_indices:
        return data_extra, neuron_ids
    keep = np.ones(len(neuron_ids), dtype=bool)
    keep[list(neuron_indices)] = False
    return data_extra[:, keep, :], neuron_ids[keep]


def remove_trials(
    data_intra: np.ndarray,
    data_extra: np.ndarray,
    inferred_spikes: Sequence[np.ndarray],
    trial_ids: np.ndarray,
    trial_indices: Sequence[int],
):
    if not trial_indices:
        return data_intra, data_extra, list(inferred_spikes), trial_ids
    keep = np.ones(len(trial_ids), dtype=bool)
    keep[list(trial_indices)] = False
    inferred_kept = [x for j, x in enumerate(inferred_spikes) if keep[j]]
    return data_intra[keep], data_extra[keep], inferred_kept, trial_ids[keep]


def merge_spikes_data(data_inferred: np.ndarray, data_extra: np.ndarray) -> np.ndarray:
    """
    Merge inferred spike trains as an additional neuron.

    Parameters
    ----------
    data_inferred : (n_trials, L_inferred)
        Padded inferred spike times
    data_extra : (n_trials, M, L_extra)
        Extracellular spike trains

    Returns
    -------
    data_merged : (n_trials, M+1, L_final)
    """

    n_trials, M, L_extra = data_extra.shape
    _, L_inferred = data_inferred.shape

    # --- Step 1: find common length ---
    L_final = max(L_extra, L_inferred)

    # --- Step 2: pad extra if needed ---
    if L_extra < L_final:
        pad_width = L_final - L_extra
        data_extra = np.pad(
            data_extra, ((0, 0), (0, 0), (0, pad_width)), mode="constant"
        )

    # --- Step 3: pad inferred if needed ---
    if L_inferred < L_final:
        pad_width = L_final - L_inferred
        data_inferred = np.pad(data_inferred, ((0, 0), (0, pad_width)), mode="constant")

    # --- Step 4: reshape inferred to neuron dimension ---
    data_inferred = data_inferred[:, np.newaxis, :]  # (n_trials, 1, L_final)

    # --- Step 5: concatenate ---
    data_merged = np.concatenate([data_extra, data_inferred], axis=1)

    return data_merged


# ---------------------------------------------------------------------
# events for hawkes
# ---------------------------------------------------------------------
def to_events(
    data_extra: np.ndarray,
    t0: Optional[float] = None,
    t1: Optional[float] = None,
    rebase: bool = False,
) -> List[List[np.ndarray]]:
    events: List[List[np.ndarray]] = []
    for trial in data_extra:
        trial_events: List[np.ndarray] = []
        for neuron_spikes in trial:
            spikes = neuron_spikes[neuron_spikes > 0.0]
            if t0 is not None:
                spikes = spikes[spikes >= t0]
            if t1 is not None:
                spikes = spikes[spikes < t1]
            if rebase and t0 is not None:
                spikes = spikes - t0
            trial_events.append(spikes)
        events.append(trial_events)
    return events


# ---------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------
def summarize_dataset(
    data_intra: np.ndarray,
    data_extra: np.ndarray,
    trial_ids: np.ndarray,
    neuron_ids: np.ndarray,
) -> Dict[str, Any]:
    counts = spike_counts_by_trial_and_neuron(data_extra)
    return {
        "data_intra_shape": list(data_intra.shape),
        "data_extra_shape": list(data_extra.shape),
        "kept_trial_ids": trial_ids.astype(int).tolist(),
        "kept_neuron_ids": neuron_ids.astype(int).tolist(),
        "spikes_per_trial": counts.sum(axis=1).astype(int).tolist(),
        "spikes_per_neuron": counts.sum(axis=0).astype(int).tolist(),
    }
