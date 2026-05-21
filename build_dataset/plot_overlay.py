import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import load_config
from config import Config


def plot_overlay(data_intra, data_extra, config, trial=0, neuron_index=-1):
    signal = data_intra[trial]
    spikes = data_extra[trial]

    # --- time axis ---
    time = np.arange(0, len(signal) * config.Delta, config.Delta)

    plt.figure(figsize=(12, 5))

    # --- membrane ---
    plt.plot(time, signal, color="blue", lw=1, label="Membrane")

    # --- spikes (single neuron) ---
    neuron_spikes = spikes[neuron_index]
    valid = neuron_spikes[neuron_spikes > 0.0]

    plt.vlines(
        valid,
        ymin=signal.min(),
        ymax=signal.max(),
        color="red",
        alpha=0.5,
        label="Spikes",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Membrane potential (mV)")
    plt.title(f"Membrane + spikes overlay (trial {trial}, neuron {neuron_index})")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot membrane + spike overlay")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/processed/exp1",
        help="Path to processed dataset",
    )

    parser.add_argument("--trial", type=int, default=0, help="Trial index")

    parser.add_argument(
        "--neuron", type=int, default=-1, help="Neuron index (default: last neuron)"
    )

    args = parser.parse_args()

    # --- load data ---
    data_intra = np.load(os.path.join(args.data_dir, "data_intra.npy"))
    data_extra = np.load(os.path.join(args.data_dir, "data_extra.npy"))
    config = load_config(os.path.join(args.data_dir, "config.yaml"), Config)

    print("Loaded:")
    print("  intra:", data_intra.shape)
    print("  extra:", data_extra.shape)

    plot_overlay(
        data_intra, data_extra, config, trial=args.trial, neuron_index=args.neuron
    )
