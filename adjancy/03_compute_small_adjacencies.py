import numpy as np
from config import AdjacencyConfig
from adjacency_utils import *

cfg = AdjacencyConfig()
cfg.decay_grid = (20.0,)

data = np.load(cfg.processed_path() / "data_extra.npy")

selected = np.load(cfg.output_path() / "selection" / "selected.npy")

# IMPORTANT: remove central neuron
selected = remove_central(selected, cfg.central_neuron_index, data.shape[1])

data_small = data[:, selected, :]

out = cfg.output_path() / "small"
ensure_dir(out)

for i, (t0, t1) in enumerate(cfg.small_timeframes):
    events = to_events(data_small, t0, t1)

    A, b, best_decay, _ = fit_adm4(events, cfg.decay_grid)

    sub = out / f"interval_{i}"
    ensure_dir(sub)

    np.save(sub / "A_small.npy", A)
    np.save(sub / "b_small.npy", b)
    plot_adjacency_matrix(A, sub / "A_small.png")

    print(f"interval {i} done | decay={best_decay}")
