from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # raw data
    intra_dir: str = "./data/SignalIntra"
    intra_pattern: str = "membrane_potential_trial{trial}.mat"
    extra_dir: str = "./data/SignalExtra"
    extra_pattern: str = "SpikesM249_trial_{trial}.csv"

    # acquisition / dataset
    Delta: float = 4.8e-5
    trial_duration: float = 40.0
    n_trials: int = 10
    membrane_spike_threshold_mv: float = -20.0
    T_START: float = 11.0
    T_END: float = 24.0

    # build stage
    remove_trials_without_target_spikes: bool = True
    remove_inactive_neurons: bool = True
    central_neuron_index: int = -1  # confirmed by user

    # visualization
    visualize_trial: int = 0
    match_tolerance: float = 5e-3
    n_trials_to_plot: int = 9

    # outputs
    processed_dir: str = "./data/processed/exp1"
    visualize_output_dir: str = "./outputs/visualize/exp1"

    @property
    def shifted_duration(self) -> float:
        return self.T_END - self.T_START

    def processed_path(self) -> Path:
        return Path(self.processed_dir)

    def visualize_output_path(self) -> Path:
        return Path(self.visualize_output_dir)
