from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class FitProcessConfig:
    processed_dir: str = "./data/processed/exp1"
    adjacency_dir: str = "./adjancy/outputs"
    output_dir: str = "./fit_process/outputs"

    original_trial_id: int = 0
    shifted_interval: Tuple[float, float] = (0.0, 5.0)

    Delta_intra: float = 4.8e-5
    Delta_model: float = 4.8e-5
    downsample_factor: int = 1

    hawkes_decay_value: float = 10.0

    npas: int = 200
    Nn: int = 20
    kap: float = 100.0
    rho: float = 3.0
    truncation_threshold: float = 1.0
    truncation_beta: float = 1.0 / 8.0
    nw_bandwidth: float = 0.1

    use_synthetic_path: bool = False
    synthetic_X0_low: float = -55.0
    synthetic_X0_high: float = -45.0
    synthetic_seed: int = 123

    def processed_path(self) -> Path:
        return Path(self.processed_dir)

    def adjacency_path(self) -> Path:
        return Path(self.adjacency_dir)

    def output_path(self) -> Path:
        return Path(self.output_dir)
