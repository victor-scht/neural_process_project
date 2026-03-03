
import numpy as np

def f_drift_true(x, params):
    b0, b1, b2 = params
    return b0 + b1 * np.tanh(b2 * x)

def g_sigma_true(x, params):
    s0, s1, s2 = params
    sig = 1.0 / (1.0 + np.exp(-s2 * x))
    return np.maximum(1e-6, s0 + s1 * sig)

def ajum_true(x, neuron, a0, a1):
    return a0[neuron] + a1[neuron] * np.tanh(x)

def simulate_jump_diffusion(T, dt, x0, marks, drift_params, sigma_params, a0, a1, seed=0):
    '''
    Simulate 1D jump-diffusion driven by marked point process 'marks' (t, neuron):
      dX = f(X) dt + g(X) dW + sum_{events at t} a_neuron(X_{t-})

    Euler-Maruyama on a grid; all jumps in (t_n, t_{n+1}] are applied using X_n as pre-jump state.
    Returns: ts, X, dW, jump_contrib_per_step
    '''
    rng = np.random.default_rng(seed)
    N = int(np.floor(T / dt))
    ts = np.linspace(0.0, N * dt, N + 1)
    X = np.zeros(N + 1, dtype=float)
    X[0] = float(x0)

    dW = rng.normal(0.0, np.sqrt(dt), size=N)
    jump_contrib = np.zeros(N, dtype=float)

    marks = sorted(marks, key=lambda x: x[0])
    ptr = 0

    for n in range(N):
        t0, t1 = ts[n], ts[n+1]
        x_prev = X[n]

        jc = 0.0
        while ptr < len(marks) and marks[ptr][0] <= t1:
            t_ev, neuron = marks[ptr]
            if t_ev > t0:
                jc += ajum_true(x_prev, neuron, a0, a1)
            ptr += 1

        drift = f_drift_true(x_prev, drift_params)
        sig = g_sigma_true(x_prev, sigma_params)
        X[n+1] = x_prev + drift * dt + sig * dW[n] + jc
        jump_contrib[n] = jc

    return ts, X, dW, jump_contrib
