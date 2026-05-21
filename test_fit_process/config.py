from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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

    # adjacency
    adjacency_selection_threshold: float = 1e-5
    adjacency_t0: Optional[float] = None
    adjacency_t1: Optional[float] = None
    adjacency_decay_grid: tuple[float, ...] = (0.5, 1.0, 10.0, 20.0, 50.0)

    # outputs
    processed_dir: str = "./data/processed/exp1"
    visualize_output_dir: str = "./outputs/visualize/exp1"
    adjacency_output_dir: str = "./outputs/adjacency/exp1"

    def validate(self) -> None:
        if self.Delta <= 0:
            raise ValueError("Delta must be positive.")
        if self.trial_duration <= 0:
            raise ValueError("trial_duration must be positive.")
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive.")
        if self.T_START >= self.T_END:
            raise ValueError("T_START must be smaller than T_END.")
        if self.adjacency_t0 is not None and self.adjacency_t1 is not None:
            if not (
                0 <= self.adjacency_t0 < self.adjacency_t1 <= self.T_END - self.T_START
            ):
                raise ValueError(
                    "Adjacency sub-window must lie inside the shifted dataset window."
                )

    @property
    def shifted_duration(self) -> float:
        return self.T_END - self.T_START

    def processed_path(self) -> Path:
        return Path(self.processed_dir)

    def visualize_output_path(self) -> Path:
        return Path(self.visualize_output_dir)

    def adjacency_output_path(self) -> Path:
        return Path(self.adjacency_output_dir)
