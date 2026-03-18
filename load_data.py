import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ============================================================
# Paths (same logic as your script)
# ============================================================
def setup_paths():
    current_dir = os.getcwd()
    # parent_dir = os.path.dirname(current_dir)
    parent_dir = current_dir
    data_dir = os.path.join(parent_dir, "data")
    data_extra_dir = os.path.join(data_dir, "SignalExtra")
    data_intra_dir = os.path.join(data_dir, "SignalIntra")
    return data_extra_dir, data_intra_dir


# ============================================================
# Helper
# ============================================================
def interval_to_indices(time_interval, time_step, ech):
    t0, t1 = time_interval
    i0 = int(t0 / time_step)
    i1 = int(t1 / time_step) + int(ech / time_step)  # end is exclusive
    return i0, i1


# ============================================================
# LOADERS
# ============================================================
def load_intra_trials(data_intra_dir, files, mat_key="nerve", column=0):
    traces = []
    for fname in files:
        path = os.path.join(data_intra_dir, fname)
        mat = loadmat(path)
        traces.append(mat[mat_key][:, column])
    return traces


def load_extra_trials(
    data_extra_dir,
    n_trials=10,
    base_subdir="DonneesReellesTick/M250",
    trial_dir_fmt="M250_Trial{trial}",
    spikes_file_fmt="SpikesM250_{trial}.csv",
    trial_period=40.0,
):
    raw_trials = []
    for i in range(n_trials):
        trial = i + 1
        path = os.path.join(
            data_extra_dir,
            base_subdir,
            trial_dir_fmt.format(trial=trial),
            spikes_file_fmt.format(trial=trial),
        )
        df = pd.read_csv(path)
        arr = np.array(df)[:, 1:]
        arr[arr > 0] -= trial_period * i
        raw_trials.append(arr)  # drop first column like your code
    return raw_trials


# ============================================================
# EXTRA FILTER 1: detect silent neurons + trials to remove
#   - silent neurons: no spikes across the whole experiment
#   - trials_to_remove: trials where the LAST neuron is silent (your corrected rule)
# ============================================================
def detect_silent_neurons_and_bad_trials(raw_trials, last_neuron_index=-1):
    if not raw_trials:
        return [], []

    M = raw_trials[0].shape[0]
    active_somewhere = np.zeros(M, dtype=bool)

    bad_trials = []
    for ti, mat in enumerate(raw_trials):
        # neuron activity per trial
        active_this_trial = np.any(mat > 0, axis=1)

        active_somewhere |= active_this_trial

        # trial to remove if LAST neuron is silent in that trial
        last_idx = (
            last_neuron_index if last_neuron_index >= 0 else (M + last_neuron_index)
        )
        if last_idx < 0 or last_idx >= M:
            raise IndexError("last_neuron_index is out of range for this dataset.")
        if not active_this_trial[last_idx]:
            bad_trials.append(ti)

    silent_neurons = np.flatnonzero(~active_somewhere).tolist()
    bad_trials = sorted(set(bad_trials))
    return silent_neurons, bad_trials


# ============================================================
# INTRA FILTER 2: remove the same bad trials from intra
#   - "bad trials" means: trials where last neuron is silent in EXTRA
# ============================================================
def remove_bad_trials_from_intra(intra_traces, bad_trials):
    if not bad_trials:
        return list(intra_traces)
    rm = set(bad_trials)
    return [tr for k, tr in enumerate(intra_traces) if k not in rm]


def remove_bad_trials_from_extra(raw_trials, bad_trials):
    if not bad_trials:
        return list(raw_trials)
    rm = set(bad_trials)
    return [tr for k, tr in enumerate(raw_trials) if k not in rm]


# ============================================================
# TIME WINDOWING (applied after trials are aligned/removed)
# ============================================================
def window_intra_trials(intra_traces, time_interval, time_step, ech):
    i0, i1 = interval_to_indices(time_interval, time_step, ech)
    return [np.asarray(tr)[i0:i1] for tr in intra_traces]


def window_extra_trials(raw_trials, time_interval):
    t0, t1 = time_interval
    trials = []

    for trial_idx, spikes_mat in enumerate(raw_trials):
        spikes_mat[(spikes_mat < t0) | (spikes_mat > t1)] = 0.0
        trials.append(spikes_mat)
    return trials


def window_extra_trials_align(raw_trials, time_interval):
    """
    Return: list of trials, each is list of neurons, each is np.array of spike times
    Windowing is identical to your original code:
      - shift by trial_idx * trial_period
      - keep within [t0, t1]
      - rebase to 0 by subtracting t0
    """
    t0, t1 = time_interval
    trials = []
    for trial_idx, spikes_mat in enumerate(raw_trials):
        instants = []
        for row in spikes_mat:
            s = row[row >= 0].astype(float)
            s = s[(s >= t0) & (s <= t1)]
            s -= t0
            instants.append(s)
        trials.append(instants)
    return trials


# ============================================================
# LAST STEP: manually select neurons
#   - Applies to already-windowed extra data (list-of-neurons per trial)
#   - IMPORTANT: this selection is done AFTER any neuron removals you may choose
# ============================================================
def remove_neurons_from_windowed_extra(windowed_extra, remove_neurons):
    if not remove_neurons:
        return windowed_extra
    out = []
    for trial in windowed_extra:
        trial2 = list(trial)
        for ni in sorted(remove_neurons, reverse=True):
            if 0 <= ni < len(trial2):
                trial2.pop(ni)
        out.append(trial2)
    return out


def select_neurons(windowed_extra, indices):
    return [[trial[i] for i in indices] for trial in windowed_extra]


def plot_spikes(data_extra, trial_id=0, indices=None):
    """
    Plot spike raster using crosses instead of ticks.

    Parameters:
    - data_extra: list of trials -> list of neurons -> spike times
    - trial_id: which trial to plot
    - indices: optional list of neuron labels (for y-axis)
    """

    trial = data_extra[trial_id]
    n_neurons = len(trial)

    # plt.figure(figsize=figsize)

    for i, spikes in enumerate(trial):
        if len(spikes) == 0:
            continue

        y = np.full_like(spikes, i, dtype=float)

        plt.scatter(
            spikes,
            y,
            marker="x",  # cross instead of tick
            c="red",
            s=60,  # size (increase if needed)
            linewidths=3,  # thickness of the cross
        )

    # Y axis
    if indices is not None:
        plt.yticks(np.arange(n_neurons), indices)
        plt.ylabel("Neuron (original index)")
    else:
        plt.ylabel("Neuron index")

    plt.xlabel("Time (s)")
    plt.title(f"Spike raster (trial {trial_id})")
    plt.grid(alpha=0.3)


# ============================================================
# EXAMPLE: apply your described pipeline + print + plot
# ============================================================

# ---- your selected neurons
indices = [
    47,
    55,
    70,
    87,
    96,
    112,
    113,
    114,
    115,
    116,
    123,
    137,
    148,
    149,
    150,
    207,
    227,
    229,
    237,
]

# ---- your time params / window
time_step = 4.8e-5
ech = 4.8e-3
time_interval = (11, 24)

# ---- intra files
files = [
    "clampex_2013_09_04_0014.mat",
    "clampex_2013_09_04_0015.mat",
    "clampex_2013_09_04_0016.mat",
    "clampex_2013_09_04_0018.mat",
    "clampex_2013_09_04_0019.mat",
    "clampex_2013_09_04_0020.mat",
    "clampex_2013_09_04_0021.mat",
    "clampex_2013_09_04_0022.mat",
    "clampex_2013_09_04_0023.mat",
    "clampex_2013_09_04_0024.mat",
]

# ---- setup paths
data_extra_dir, data_intra_dir = setup_paths()

# ---- load raw
intra_raw = load_intra_trials(data_intra_dir, files)
extra_raw = load_extra_trials(data_extra_dir, n_trials=10)
extra_raw = window_extra_trials(extra_raw, (11, 24))

# ---- (1) detect silent neurons + bad trials using FULL experiment
silent_neurons, bad_trials = detect_silent_neurons_and_bad_trials(
    extra_raw, last_neuron_index=-1
)

# ---- (2) remove bad trials from intra and extra (alignment step)
intra_aligned = remove_bad_trials_from_intra(intra_raw, bad_trials)
extra_aligned = remove_bad_trials_from_extra(extra_raw, bad_trials)

# ---- (3) window both datasets on your interval
data_intra = window_intra_trials(intra_aligned, time_interval, time_step, ech)
extra_windowed = window_extra_trials_align(extra_aligned, time_interval)

# ---- Optional: remove globally silent neurons (across whole experiment), then select your indices
# NOTE: if you remove neurons here, your indices must match the post-removal indexing (same as your old behavior).
extra_windowed = remove_neurons_from_windowed_extra(extra_windowed, silent_neurons)

data_extra = select_neurons(extra_windowed, indices)

# ===================
# Print info
# ===================
print("===== PIPELINE SUMMARY =====")
print(
    f"time_interval: {time_interval}  -> window = {time_interval[1] - time_interval[0]} s"
)
print(f"raw trials loaded: intra={len(intra_raw)}  extra={len(extra_raw)}")
print(f"bad trials removed (last neuron silent): {bad_trials}")
print(f"silent neurons across experiment: {len(silent_neurons)}")
print(f"remaining trials: intra={len(data_intra)}  extra={len(data_extra)}")
print(f"selected neurons per trial: {len(indices)}")

if len(data_extra) > 0:
    trial_id = 0
    spikes_per_neuron = [len(s) for s in data_extra[trial_id]]
    print(f"\nTrial {trial_id} details:")
    print(f"  intra length: {len(data_intra[trial_id])} samples")
    print(
        f"  spikes/neuron: min={np.min(spikes_per_neuron)}, max={np.max(spikes_per_neuron)}, mean={np.mean(spikes_per_neuron):.2f}"
    )

# ===================
# Plot (intra + raster)
# ===================
trial_id = 7
window = time_interval[1] - time_interval[0]
t_intra = np.arange(len(data_intra[trial_id])) * time_step


plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(t_intra, data_intra[trial_id], c="blue")
plt.grid(lw=1, alpha=0.3)
plt.subplot(1, 2, 2)
plot_spikes(data_extra, trial_id=0, indices=indices)
plt.tight_layout()
plt.show()
