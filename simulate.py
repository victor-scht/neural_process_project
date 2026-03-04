import numpy as np
from utils import simuHawkesExpoM, simu_jumpdiff, intensM

def make_sparse_adjacency(M: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            if rng.random() < 0.12:
                A[i, j] = rng.uniform(0.01, 0.06)
    # mild scaling for stability
    rs = A.sum(axis=1).max()
    if rs > 0.7:
        A *= 0.7 / rs
    return A

def build_isjumpN_from_times(times, grid):
    # times is list of arrays of spike times (per neuron)
    nonempty = [np.asarray(t) for t in times if len(t) > 0]
    jumptimes = np.sort(np.concatenate(nonempty)) if len(nonempty) > 0 else np.array([])
    isjumpN = np.zeros(len(grid))
    for i in range(len(grid) - 1):
        isjumpN[i] = np.sum((jumptimes > grid[i]) & (jumptimes <= grid[i + 1]))
    return isjumpN, jumptimes

def simulate_synthetic(T=15.0, Delta=4.8e-3, M=19, seed=123):
    rng = np.random.default_rng(seed)
    grid = np.arange(0.0, T + Delta, Delta)

    # Hawkes parameters
    base = np.full(M, 0.15)
    adjM = make_sparse_adjacency(M, seed=seed)
    beta = np.repeat(20.0, M)
    paramhawkes = [base, adjM, beta]

    # simulate spikes
    # NOTE: utils.simuHawkesExpoM expects param already containing xi; pass xi=0
    spikesneurons = simuHawkesExpoM(param=paramhawkes, M=M, Tend=T, xi=0.0)

    isjumpN, jumptimes = build_isjumpN_from_times(spikesneurons, grid)

    # jump-diffusion using user's model functions
    from model import bdrift, sig, ajump
    X0 = rng.uniform(-55, -45)
    X = simu_jumpdiff(X0=X0, grid=grid, bfunc=bdrift, sigfunc=sig, afunc=ajump, isjumpN=isjumpN)

    return {
        "grid": grid,
        "Delta": Delta,
        "M": M,
        "paramhawkes": paramhawkes,
        "spikesneurons": spikesneurons,
        "jumptimes": jumptimes,
        "isjumpN": isjumpN,
        "X": X,
    }

def compute_intensities_on_grid(grid, paramhawkes, spikesneurons, n_steps):
    # intensities at grid[:n_steps] inclusive
    intensity = np.array([intensM(s, paramhawkes, spikesneurons) for s in grid[:n_steps]])
    return intensity
