import numpy as np
from numpy.linalg import solve
from scipy.stats import norm


# ====================
# Hawkes process
# ====================


def intensM(t, param, times):
    """
    Intensity function for M Hawkes processes (exponential kernel)

    param: [xi, alpha, beta]
        xi    : array of size M
        alpha : M x M matrix
        beta  : array of size M
    times: list of length M, spike times for each neuron
    """
    xi, alpha, beta = param
    M = len(xi)

    intenst = [0.0] * M

    for i in range(M):
        if len(times[i]) == 0:
            """
            Confusing because it should not have any impact on
            the computation of the intensity for the ith neuron

            Maybe it is for a gain of time
            """
            intenst[i] = xi[i]

        else:
            intenst[i] = xi[i]
            for j in range(M):
                if i == j:
                    continue
                # impact of other neurons
                tj = np.array(times[j])
                tj = tj[tj < t]
                if len(tj) > 0:
                    intenst[i] += np.sum(
                        alpha[i, j] * beta[i] * np.exp(-beta[i] * (t - tj))
                    )
    return intenst


def intens_original(k, M, s, times, param, xi):
    _, alpha, beta = param
    vect = 0.0

    for m in range(k):
        for l in range(M):
            tm = np.array(times[m])
            tm = tm[tm < s]
            if len(tm) > 0:
                # very weird to have alpha[m,l] instead of alpha[l,m]
                # because it is the impact of the spikes of the mth neuron
                # on neuron l
                vect += np.sum(
                    alpha[m, l] * beta[m] * np.exp(-beta[m] * (s - tm))
                )  # + sign!

    vect += k * xi
    return vect


def intens(k, M, s, times, param, xi):
    """
    It sums all intensities up to the kth neuron
    """
    _, alpha, beta = param
    vect = 0.0

    for m in range(k):
        for l in range(M):
            tl = np.array(times[l])
            tl = tl[tl < s]
            if len(tl) > 0:
                vect += np.sum(
                    alpha[m, l] * beta[l] * np.exp(-beta[l] * (s - tl))
                )  # + sign!

    vect += k * xi
    return vect


def simuHawkesExpoM(param, M, Tend, xi):
    """
    Simulation of a multi-neuron exponential Hawkes process
    using Ogata's thinning method.
    """
    times = [[] for _ in range(M)]
    s = 0.0

    while s < Tend:
        # Compute total intensity upper bound
        lambda_bar = intens(M, M, s, times, param, xi)
        if lambda_bar <= 0:
            break  # avoid division by zero

        # Generate next candidate event
        u = np.random.rand()
        w = -np.log(u) / lambda_bar
        s += w

        # Compute current intensities per neuron
        current_intensities = []
        for k in range(M):
            lam_k = intens(k + 1, M, s, times, param, xi) - intens(
                k, M, s, times, param, xi
            )
            current_intensities.append(lam_k)
        total_intensity = sum(current_intensities)

        # Thinning: accept event with probability
        D = np.random.rand()
        if D * lambda_bar <= total_intensity:
            # Choose which neuron fires
            probs = np.array(current_intensities) / total_intensity
            neuron = np.random.choice(M, p=probs)
            times[neuron].append(s)

    return times


# ====================
# Test function
# ====================


def a_func(x, namea):
    x = np.asarray(x)

    if namea == "none":
        return np.zeros_like(x)

    if namea == "constant":
        return np.ones_like(x)

    if namea == "lin":
        return np.clip(x, -5, 5)

    if namea == "lin2":
        return -0.1 * x

    if namea == "lin3":
        theta = np.mean(x)
        return theta - x


# ====================
# Estimation
# ====================


def projectionSm(x, q1, q2, m):
    Dm = 2 * m + 1
    c = np.sqrt(2) / np.sqrt(q2 - q1)

    proj = np.zeros((Dm, len(x)))
    proj[0, :] = 1 / np.sqrt(q2 - q1)

    for l in range(1, m + 1):
        proj[2 * l - 1, :] = c * np.cos(2 * np.pi * l * (x - q1) / (q2 - q1))
        proj[2 * l, :] = c * np.sin(2 * np.pi * l * (x - q1) / (q2 - q1))

    return proj


def alphachapeau(P, U):
    AA = P @ P.T
    B = P @ U.reshape(-1, 1)
    return solve(AA, B).flatten()


def collecestimcoeff(X, U, q1, q2, Nn):
    colleccoeffalpha = np.zeros((Nn, 2 * Nn + 1))
    collecP = []

    # N = len(U)

    for k in range(1, Nn + 1):
        Pk = projectionSm(X[1 : len(U) + 1], q1, q2, k)
        # Pk = projectionSm(X[1 : N + 1], q1, q2, k)
        collecP.append(Pk)

        try:
            alpha_hat = alphachapeau(Pk, U)
            colleccoeffalpha[k - 1, : 2 * k + 1] = alpha_hat
        except Exception:
            break

    return collecP, colleccoeffalpha


# ====================
# Adaptation
# ====================


def penaltyb(m, n, Delta, rho, sigma02):
    return rho * (2 * m + 1) * sigma02 / (n * Delta)


def penaltyg(Nn, n, Delta, kap):
    return kap * np.arange(1, Nn + 1) / (n * Delta)


def penaltysig(Nn, n, kap):
    return kap * np.arange(1, Nn + 1) / n


def adaptiveestim(colleccoeffalpha, collecmatP, U, penalty):
    ind = len(collecmatP)
    estimmhat = []
    criteremhat = np.zeros(ind)

    for l in range(ind):
        # number of rows = number of basis functions
        nrows = collecmatP[l].shape[0]
        est = np.sum(collecmatP[l] * colleccoeffalpha[l, :nrows][:, None], axis=0)
        estimmhat.append(est)
        criteremhat[l] = np.mean((U - est) ** 2)

    crit = criteremhat + penalty[:ind]
    mhat = np.argmin(crit)

    return estimmhat, criteremhat, crit, mhat


# ====================
# Supplementary functions
# ====================


def phifunc(x):
    ax = abs(x)
    if ax < 1:
        return 1
    if ax >= 2:
        return 0
    return np.exp((1 / 3) + 1 / (x**2 - 4))


def mNW(x, X, Y, h, K=norm.pdf):
    X = np.asarray(X)
    Y = np.asarray(Y)

    if np.isscalar(x):
        weights = K((x - X) / h) / h
        weights /= weights.sum()
        return np.dot(weights, Y)

    else:
        res = []
        for xi in x:
            weights = K((xi - X) / h) / h
            weights /= weights.sum()
            res.append(np.dot(weights, Y))
        return np.array(res)


def simu_jumpdiff(X0, grid, bfunc, sigfunc, afunc, isjumpN):
    W = np.random.randn(len(grid) - 1)
    X = np.zeros(len(grid))
    X[0] = X0

    for i in range(len(grid) - 1):
        dt = grid[i + 1] - grid[i]
        X[i + 1] = (
            X[i]
            + dt * bfunc(X[i])
            + np.sqrt(dt) * sigfunc(X[i]) * W[i]
            + afunc(X[i]) * isjumpN[i]
        )
    return X


def simu_diff(X0, grid, bfunc, sigfunc):
    W = np.random.randn(len(grid) - 1)
    X = np.zeros(len(grid))
    X[0] = X0

    for i in range(len(grid) - 1):
        dt = grid[i + 1] - grid[i]
        X[i + 1] = X[i] + dt * bfunc(X[i]) + np.sqrt(dt) * sigfunc(X[i]) * W[i]
    return X
