
import numpy as np

def simulate_multivariate_hawkes_ogata(T, mu, alpha, beta, seed=0, max_events=200000):
    '''
    Ogata thinning for exponential-kernel multivariate Hawkes:
      lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_{i,j} * beta_{i,j} * exp(-beta_{i,j}*(t - t_k^j))

    Returns
    -------
    times : list length M; times[j] are event times for neuron j
    marks : list of (t, j) events sorted by time
    '''
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)

    M = mu.shape[0]
    assert alpha.shape == (M, M)
    assert beta.shape == (M, M)

    t = 0.0
    H = np.zeros((M, M), dtype=float)  # H[i,j] memory from source j to target i

    times = [[] for _ in range(M)]
    marks = []

    def lambdas(H_):
        return mu + H_.sum(axis=1)

    n_events = 0
    while t < T and n_events < max_events:
        lam = lambdas(H)
        lam_sum = float(lam.sum())
        if lam_sum <= 0.0:
            break

        w = rng.exponential(1.0 / lam_sum)
        t_cand = t + w
        if t_cand > T:
            break

        dt = t_cand - t
        H = H * np.exp(-beta * dt)

        lam_cand = lambdas(H)
        lam_sum_cand = float(lam_cand.sum())

        if rng.random() * lam_sum <= lam_sum_cand:
            probs = lam_cand / lam_sum_cand
            j = int(rng.choice(M, p=probs))
            times[j].append(t_cand)
            marks.append((t_cand, j))
            n_events += 1

            H[:, j] += alpha[:, j] * beta[:, j]
            t = t_cand
        else:
            t = t_cand

    marks.sort(key=lambda x: x[0])
    return times, marks
