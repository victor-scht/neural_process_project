import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tick.hawkes import SimuHawkesExpKernels, HawkesADM4

# -------------------------
# CONFIG
# -------------------------

M = 10  # neurons (small for testing)
DECAY = 1.0
BASELINE = 20.0  # Hz (as in paper)

DURATIONS = [5, 10, 20, 40, 80]  # increasing time
N_TRIALS = 5

SUBSAMPLE_RATIO = 2 / 3


# -------------------------
# SIMULATION
# -------------------------


def simulate_hawkes(T):
    baseline = np.ones(M) * BASELINE

    adjacency = np.random.rand(M, M) * 0.1
    np.fill_diagonal(adjacency, 0.02)

    sim = SimuHawkesExpKernels(
        adjacency=adjacency,
        decays=DECAY,
        baseline=baseline,
        end_time=T,
        verbose=False,
    )

    sim.simulate()

    return sim.timestamps


# -------------------------
# FIT INTENSITY
# -------------------------


def fit_intensity(events):
    model = HawkesADM4(decay=DECAY, lasso_nuclear_ratio=1.0)
    model.fit(events)
    return model


# -------------------------
# TIME RESCALING
# -------------------------


def compute_rescaled_times(events, model):
    rescaled = []

    for j in range(M):
        spikes = events[j]
        if len(spikes) == 0:
            rescaled.append(np.array([]))
            continue

        lam = model.baseline[j]

        # crude approx (constant intensity)
        rescaled.append(np.cumsum(np.ones_like(spikes) * lam))

    return rescaled


# -------------------------
# TEST STATISTIC
# -------------------------


def compute_Z(rescaled):
    Zs = []

    for j in range(M):
        times = rescaled[j]
        if len(times) < 10:
            continue

        N = len(times)
        u = np.linspace(0, 1, N)

        empirical = np.arange(1, N + 1) / N

        deviation = np.max(np.abs(empirical - u))

        Z = np.sqrt(N) * deviation
        Zs.append(Z)

    return np.array(Zs)


# -------------------------
# MAIN EXPERIMENT
# -------------------------

results = []

for T in DURATIONS:
    Z_all = []

    print(f"\n=== Duration {T} ===")

    for _ in tqdm(range(N_TRIALS)):
        events = simulate_hawkes(T)

        model = fit_intensity(events)

        rescaled = compute_rescaled_times(events, model)

        Z = compute_Z(rescaled)

        if len(Z) > 0:
            Z_all.append(np.mean(Z))

    results.append(np.mean(Z_all))

# -------------------------
# PLOT CONVERGENCE
# -------------------------

plt.figure(figsize=(6, 4))
plt.plot(DURATIONS, results, marker="o", color="blue")

plt.xlabel("Duration T")
plt.ylabel("Mean Z statistic")
plt.title("Convergence of Hawkes validation test")

plt.grid(alpha=0.3)
plt.show()
