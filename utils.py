import numpy as np

from numpy.linalg import solve
from scipy.stats import norm

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

    for k in range(1, Nn + 1):
        Pk = projectionSm(X[1 : len(U) + 1], q1, q2, k)
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
        s = weights.sum()
        if s <= 0:
            return np.mean(Y)
        weights /= s
        return np.dot(weights, Y)

    res = []
    for xi in x:
        weights = K((xi - X) / h) / h
        s = weights.sum()
        if s <= 0:
            res.append(np.mean(Y))
        else:
            weights /= s
            res.append(np.dot(weights, Y))
    return np.array(res)
