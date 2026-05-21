from __future__ import annotations

import numpy as np

from build_dataset.config import Config
from build_dataset.utils import (
    clean_extra,
    detect_all_intra_spikes,
    ensure_dir,
    filter_extra,
    filter_intra,
    find_inactive_neurons,
    find_trials_without_target_spikes,
    load_all_extra,
    load_all_intra,
    remove_neurons,
    remove_trials,
    save_config,
    save_json,
    shift_valid_spike_times,
    summarize_dataset,
    trial_spike_times_to_padded_matrix,
    window_and_shift_spike_lists,
    merge_spikes_data,
)


def build_dataset(config: Config):
    processed_dir = ensure_dir(config.processed_path())

    raw_intra = load_all_intra(config)
    raw_extra = load_all_extra(config)

    original_trial_ids = np.arange(config.n_trials, dtype=int)
    original_neuron_ids = np.arange(raw_extra.shape[1], dtype=int)

    # 1) Clean extracellular global clock
    raw_extra = clean_extra(raw_extra, config.trial_duration)

    # 2) Filter to chosen interval
    data_intra = filter_intra(raw_intra, config.T_START, config.T_END, config.Delta)
    data_extra = filter_extra(raw_extra, config.T_START, config.T_END)
    data_extra = shift_valid_spike_times(data_extra, config.T_START)

    # 3) Infer target spikes from membrane, filtered to same interval
    inferred_spikes_all = detect_all_intra_spikes(
        raw_intra, config.Delta, config.membrane_spike_threshold_mv
    )
    inferred_spikes = window_and_shift_spike_lists(
        inferred_spikes_all, config.T_START, config.T_END
    )

    # 4) Remove neurons with no spikes in the interval
    removed_neuron_ids = []
    kept_neuron_ids = original_neuron_ids.copy()
    if config.remove_inactive_neurons:
        inactive_local = find_inactive_neurons(data_extra)
        removed_neuron_ids = kept_neuron_ids[inactive_local].astype(int).tolist()
        data_extra, kept_neuron_ids = remove_neurons(
            data_extra=data_extra,
            neuron_ids=kept_neuron_ids,
            neuron_indices=inactive_local,
        )

    # 5) Remove trials where target neuron has no spikes
    removed_trial_ids = []
    kept_trial_ids = original_trial_ids.copy()
    if config.remove_trials_without_target_spikes:
        empty_trial_local = find_trials_without_target_spikes(inferred_spikes)
        removed_trial_ids = kept_trial_ids[empty_trial_local].astype(int).tolist()
        data_intra, data_extra, inferred_spikes, kept_trial_ids = remove_trials(
            data_intra=data_intra,
            data_extra=data_extra,
            inferred_spikes=inferred_spikes,
            trial_ids=kept_trial_ids,
            trial_indices=empty_trial_local,
        )

    inferred_padded = trial_spike_times_to_padded_matrix(inferred_spikes)

    # 6) save everything
    data_merged = merge_spikes_data(inferred_padded, data_extra)

    np.save(processed_dir / "data_intra.npy", data_intra)
    np.save(processed_dir / "data_extra.npy", data_merged)
    np.save(processed_dir / "central_spikes_from_intra.npy", inferred_padded)
    np.save(processed_dir / "kept_trial_ids.npy", kept_trial_ids)
    np.save(processed_dir / "kept_neuron_ids.npy", kept_neuron_ids)

    meta = summarize_dataset(data_intra, data_extra, kept_trial_ids, kept_neuron_ids)
    meta.update(
        {
            "original_n_trials": int(config.n_trials),
            "original_n_neurons": int(len(original_neuron_ids)),
            "removed_trial_ids": removed_trial_ids,
            "removed_neuron_ids": removed_neuron_ids,
        }
    )
    save_json(meta, processed_dir / "dataset_metadata.json")
    save_config(config, processed_dir / "config.yaml")

    print("Processed dataset saved to:", processed_dir)
    print("Kept trial ids:", kept_trial_ids.tolist())
    print("Kept neuron ids shape:", kept_neuron_ids.shape)
    return meta


def main():
    config = Config()
    build_dataset(config)


if __name__ == "__main__":
    main()
