import numpy as np
import pandas as pd
from adjancy.config import AdjacencyConfig
from adjancy.adjacency_utils import *

cfg = AdjacencyConfig()

A = np.load(cfg.output_path() / "full" / "A_full.npy")
kept_ids = np.load(cfg.processed_path() / "kept_neuron_ids.npy")

row = A[cfg.central_neuron_index]

# plot evolution
counts = plot_trigger_curve(
    row,
    cfg.trigger_grid,
    cfg.output_path() / "selection" / "trigger_curve.png",
)

# selection
selected = np.where(row > cfg.trigger)[0][:-1]
selected_original = kept_ids[selected]

out = cfg.output_path() / "selection"
ensure_dir(out)

pd.DataFrame(
    {"local": selected, "original": selected_original, "weight": row[selected]}
).to_csv(out / "selected_neurons.csv", index=False)

np.save(out / "selected.npy", selected)

print("selected neurons:", len(selected))
