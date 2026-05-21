from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from fit_process_2.config import FitProcessConfig


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_processed_data(cfg: FitProcessConfig):
    pdir = cfg.processed_path()
    return {
        "data_intra": np.load(pdir / "data_intra.npy"),
        "data_extra": np.load(pdir / "data_extra.npy"),
        "kept_trial_ids": np.load(pdir / "kept_trial_ids.npy"),
        "kept_neuron_ids": np.load(pdir / "kept_neuron_ids.npy"),
        "processed_cfg": _load_yaml(pdir / "config.yaml"),
    }


def load_adjacency_selection(cfg: FitProcessConfig):
    adir = cfg.adjacency_path()

    selected_local = np.load(adir / "selection" / "selected.npy")
    small_dir = adir / "small" / "interval_2"
    small_adj = np.load(small_dir / "A_small.npy")

    baseline_small = None
    baseline_path = small_dir / "b_small.npy"
    if baseline_path.exists():
        baseline_small = np.load(baseline_path)

    return {
        "selected_local": selected_local,
        "small_adjacency": small_adj,
        "small_baseline": baseline_small,
        "small_dir": small_dir,
    }


def get_local_trial_index(kept_trial_ids: np.ndarray, original_trial_id: int) -> int:
    kept_trial_ids = kept_trial_ids.astype(int)
    match = np.where(kept_trial_ids == int(original_trial_id))[0]
    if len(match) == 0:
        raise ValueError(
            f"original_trial_id={original_trial_id} is not available. "
            f"Available kept_trial_ids={kept_trial_ids.tolist()}"
        )
    return int(match[0])
