from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class AdjacencyConfig:
    processed_dir: str = "./data/processed/exp1"
    output_dir: str = "./adjancy/outputs"

    central_neuron_index: int = -1

    trigger: float = 1e-2

    trigger_grid: Tuple[float, ...] = (
        1e-8,
        5e-8,
        1e-7,
        5e-7,
        1e-6,
        5e-6,
        1e-5,
        5e-5,
        1e-4,
        5e-4,
        1e-3,
        5e-3,
        1e-2,
        5e-2,
        1e-1,
    )

    decay_grid: Tuple[float, ...] = (0.5, 1.0, 10.0, 20.0, 50.0)

    small_timeframes: Tuple[Tuple[float, float], ...] = (
        (0.0, 4.0),
        (0.0, 8.0),
        (0.0, 13.0),
    )

    def processed_path(self) -> Path:
        return Path(self.processed_dir)

    def output_path(self) -> Path:
        return Path(self.output_dir)
