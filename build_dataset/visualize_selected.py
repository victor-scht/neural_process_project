from __future__ import annotations

import json
import numpy as np


from build_dataset.config import Config
from build_datset.plot_utils import (
    plot_overlay,
    plot_spike_raster,
    plot_counts_bar,
    plot_psth_trial,
)
from build_datset.utils import ensure_dir, save_config, save_json


def visualize_selected(config: Config, plot_raster: bool = True):
    processed_dir = config.processed_path()
    output_dir = ensure_dir(config.visualize_output_path() / "selected")

    plots_dir = ensure_dir(output_dir / "plots")
    summary_dir = ensure_dir(output_dir / "summary")

    # --- load data ---
    data_intra = np.load(processed_dir / "data_intra.npy")
    data_extra = np.load(processed_dir / "data_extra.npy")
    kept_trial_ids = np.load(processed_dir / "kept_trial_ids.npy")

    with open(processed_dir / "dataset_metadata.json", "r") as f:
        dataset_summary = json.load(f)

    n_trials = data_intra.shape[0]

    # =========================
    # LOOP OVER ALL TRIALS
    # =========================
    for trial in range(n_trials):
        original_trial = int(kept_trial_ids[trial])

        # --- overlay ---
        plot_overlay(
            data_intra=data_intra,
            data_extra=data_extra,
            delta=config.Delta,
            trial=trial,
            neuron_index=config.central_neuron_index,
            path=plots_dir / f"overlay_trial_{original_trial}.png",
        )

        # --- raster ---
        if plot_raster:
            plot_spike_raster(
                data_extra=data_extra,
                trial=trial,
                path=plots_dir / f"raster_trial_{original_trial}.png",
            )

    # =========================
    # SUMMARY PLOTS
    # =========================
    spikes_per_trial = np.sum(data_extra > 0.0, axis=(1, 2))
    spikes_per_neuron = np.sum(data_extra > 0.0, axis=(0, 2))

    plot_counts_bar(
        spikes_per_trial,
        title="Spike counts per kept trial",
        xlabel="Trial index",
        path=plots_dir / "spikes_per_trial.png",
    )

    plot_counts_bar(
        spikes_per_neuron,
        title="Spike counts per kept neuron",
        xlabel="Neuron index",
        path=plots_dir / "spikes_per_neuron.png",
    )

    # =========================
    # SAVE METADATA
    # =========================
    save_config(config, summary_dir / "config_used.yaml")
    save_json(dataset_summary, summary_dir / "dataset_summary.json")

    print("Saved to:", output_dir)

    # =========================
    # PER-TRIAL NEURON COUNTS
    # =========================

    for trial in range(n_trials):
        original_trial = int(kept_trial_ids[trial])

        counts_per_neuron = np.sum(data_extra[trial] > 0.0, axis=1)

        plot_counts_bar(
            counts_per_neuron,
            title=f"Spike counts per neuron (trial {original_trial})",
            xlabel="Neuron index",
            path=plots_dir / f"spikes_per_neuron_trial_{original_trial}.png",
        )

    # =========================
    # PSTH PER TRIAL
    # =========================

    for trial in range(n_trials):
        original_trial = int(kept_trial_ids[trial])

        plot_psth_trial(
            data_extra=data_extra,
            trial=trial,
            bins=50,
            path=plots_dir / f"psth_trial_{original_trial}.png",
        )


def main():
    config = Config()

    visualize_selected(
        config=config,
        plot_raster=True,
    )


if __name__ == "__main__":
    main()
