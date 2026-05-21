import numpy as np
from adjancy.config import AdjacencyConfig
from adjancy.adjacency_utils import *

cfg = AdjacencyConfig()

data = np.load(cfg.processed_path() / "data_extra.npy")

events = to_events(data)

A, b, best_decay, scores = fit_adm4(events, cfg.decay_grid)

out = cfg.output_path() / "full"
ensure_dir(out)

np.save(out / "A_full.npy", A)

plot_adjacency_matrix(A, out / "A_full.png")
plot_central_row(A, cfg.central_neuron_index, cfg.trigger, out / "central_row.png")

print("best decay:", best_decay)
