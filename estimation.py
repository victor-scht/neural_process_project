
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize

def drift_model(x, theta):
    b0, b1, b2 = theta
    return b0 + b1 * np.tanh(b2 * x)

def sigma_model(x, phi):
    s0, s1, s2 = phi
    sig = 1.0 / (1.0 + np.exp(-s2 * x))
    raw = s0 + s1 * sig
    return np.log1p(np.exp(raw)) + 1e-6  # softplus => positive

def ajum_model(x, neuron, a0, a1):
    return a0[neuron] + a1[neuron] * np.tanh(x)

@dataclass
class EstimationResult:
    drift_theta: np.ndarray
    sigma_phi: np.ndarray
    a0: np.ndarray
    a1: np.ndarray
    success: bool
    message: str
    fun: float

def _marks_to_steps(ts, marks, dt):
    N = len(ts) - 1
    step_events = [[] for _ in range(N)]
    for t_ev, neuron in marks:
        if t_ev <= ts[0] or t_ev > ts[-1]:
            continue
        n = int(np.ceil((t_ev - ts[0]) / dt)) - 1
        n = max(0, min(N-1, n))
        if ts[n] < t_ev <= ts[n+1]:
            step_events[n].append(int(neuron))
    return step_events

def estimate_parameters(ts, X, marks, M, dt):
    '''
    Gaussian quasi-MLE under Euler scheme:
      ΔX_n = f(X_n) dt + J_n + sigma(X_n) * ε_n,  ε_n ~ N(0, dt)

    Unknowns:
      theta (3) for drift f
      phi   (3) for sigma g
      a0(M), a1(M) for jump size a_i(x) = a0_i + a1_i*tanh(x)
    '''
    X = np.asarray(X, float)
    ts = np.asarray(ts, float)
    N = len(ts) - 1
    assert len(X) == N + 1

    step_events = _marks_to_steps(ts, marks, dt)

    X_left = X[:-1]
    dX = X[1:] - X[:-1]

    def unpack(p):
        p = np.asarray(p, float)
        theta = p[0:3]
        phi = p[3:6]
        a0 = p[6:6+M]
        a1 = p[6+M:6+2*M]
        return theta, phi, a0, a1

    def nll(p):
        theta, phi, a0, a1 = unpack(p)
        f = drift_model(X_left, theta)
        sig = sigma_model(X_left, phi)

        J = np.zeros(N, float)
        for n in range(N):
            if step_events[n]:
                x_state = X_left[n]
                for neuron in step_events[n]:
                    J[n] += ajum_model(x_state, neuron, a0, a1)

        resid = dX - f*dt - J
        var = (sig**2) * dt
        return 0.5*np.sum(np.log(2*np.pi*var) + (resid**2)/var)

    theta0 = np.array([0.0, 0.5, 0.5])
    phi0 = np.array([0.0, 0.5, 0.5])
    a00 = np.zeros(M)
    a10 = np.zeros(M)
    p0 = np.concatenate([theta0, phi0, a00, a10])

    bounds = []
    bounds += [(-5, 5), (-5, 5), (1e-3, 5)]         # theta
    bounds += [(-5, 5), (-5, 5), (1e-3, 5)]         # phi
    bounds += [(-3, 3)] * M                         # a0
    bounds += [(-3, 3)] * M                         # a1

    res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 700})

    theta_hat, phi_hat, a0_hat, a1_hat = unpack(res.x)
    return EstimationResult(
        drift_theta=theta_hat,
        sigma_phi=phi_hat,
        a0=a0_hat,
        a1=a1_hat,
        success=bool(res.success),
        message=str(res.message),
        fun=float(res.fun),
    )

def reconstruct_X(ts, X0, marks, dt, drift_theta, sigma_phi, a0, a1, dW):
    '''
    Reconstruct X using estimated parameters and the same Brownian increments dW.
    '''
    ts = np.asarray(ts, float)
    N = len(ts) - 1
    step_events = _marks_to_steps(ts, marks, dt)

    Xhat = np.zeros(N+1, float)
    Xhat[0] = float(X0)

    for n in range(N):
        x_prev = Xhat[n]
        f = drift_model(x_prev, drift_theta)
        sig = sigma_model(x_prev, sigma_phi)

        jc = 0.0
        for neuron in step_events[n]:
            jc += ajum_model(x_prev, neuron, a0, a1)

        Xhat[n+1] = x_prev + f*dt + sig*dW[n] + jc

    return Xhat
