import numpy as np
import matplotlib.pyplot as plt
import hawkes
import jumpdiff
from scipy.stats import norm

np.random.seed(7)

# --------------------------------------------------
# 1) Multivariate Hawkes process
# --------------------------------------------------
M = 1
Tend = 5

xi = 0.5
xi_vec = np.full(M, xi)

beta0 = 5
beta = np.full(M, beta0)

alpha = np.array(
    [
        [0.00, 0.16, 0.08, 0.05],
        [0.10, 0.00, 0.12, 0.07],
        [0.06, 0.11, 0.00, 0.09],
        [0.04, 0.07, 0.13, 0.00],
    ]
)
alpha = np.array([[0.4]])

param = [xi_vec, alpha, beta]
times = hawkes.simuHawkesExpoM(param, M, Tend, xi)

# --------------------------------------------------
# 2) Euler grid and Hawkes jump increments
# --------------------------------------------------
ngrid = int(1e3)
grid = np.linspace(0.0, Tend, ngrid + 1)
isjumpN, all_spikes = jumpdiff.hawkes_to_isjumpN(times, grid)

# --------------------------------------------------
# 3) Generate basis functions
# --------------------------------------------------
bfunc, sigfunc, afunc, basis_params = jumpdiff.generate_basis_functions(
    K=6, seed=12, m=0.0
)

# --------------------------------------------------
# 4) Simulate membrane potential
# --------------------------------------------------
X0 = 2.0
Xt = jumpdiff.simu_jumpdiff(X0, grid, bfunc, sigfunc, afunc, isjumpN)


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
    A = P.T  # shape (n, Dm)
    alpha_hat, *_ = np.linalg.lstsq(A, U, rcond=None)
    return alpha_hat


def collecestimcoeff(X, U, q1, q2, Nn):
    colleccoeffalpha = np.zeros((Nn, 2 * Nn + 1))
    collecP = []

    for k in range(1, Nn + 1):
        Pk = projectionSm(X[0 : len(U)], q1, q2, k)
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


def build_collection_estimates(gridx, q1, q2, colleccoeffalpha, ind, positive=False):
    collecestim = np.zeros((ind, len(gridx)))
    for k in range(ind):
        proj = projectionSm(gridx, q1, q2, k + 1)
        coeff = colleccoeffalpha[k, : 2 * (k + 1) + 1]
        vals = np.sum(proj * coeff[:, None], axis=0)
        if positive:
            vals = np.maximum(vals, 0.0)
        collecestim[k, :] = vals
    return collecestim


def make_interp_function(xgrid, ygrid, floor=None):
    xgrid = np.asarray(xgrid, dtype=float)
    ygrid = np.asarray(ygrid, dtype=float)

    def f(x):
        xarr = np.asarray(x, dtype=float)
        y = np.interp(xarr, xgrid, ygrid, left=ygrid[0], right=ygrid[-1])
        if floor is not None:
            y = np.maximum(y, floor)
        if np.ndim(xarr) == 0:
            return float(y)
        return y

    return f


def recover_noise_from_path(X, grid, bfunc, sigfunc, afunc, isjumpN):
    W = np.zeros(len(grid) - 1)
    for i in range(len(grid) - 1):
        dt = grid[i + 1] - grid[i]
        denom = np.sqrt(dt) * sigfunc(X[i])
        W[i] = (X[i + 1] - X[i] - dt * bfunc(X[i]) - afunc(X[i]) * isjumpN[i]) / denom
    return W


def simu_jumpdiff_given_noise(X0, grid, bfunc, sigfunc, afunc, isjumpN, W):
    Xsim = np.zeros(len(grid))
    Xsim[0] = X0

    for i in range(len(grid) - 1):
        dt = grid[i + 1] - grid[i]
        Xsim[i + 1] = (
            Xsim[i]
            + dt * bfunc(Xsim[i])
            + np.sqrt(dt) * sigfunc(Xsim[i]) * W[i]
            + afunc(Xsim[i]) * isjumpN[i]
        )
    return Xsim


def main():
    Delta = grid[1] - grid[0]
    n = len(Xt) - 2
    npas = 1000
    Nn = 10

    q1, q2 = Xt.min(), Xt.max()
    gridx = np.linspace(q1, q2, npas)

    qq1, qq2 = np.quantile(Xt, [0.05, 0.95])
    keep = (gridx > qq1) & (gridx < qq2)

    Xstate = Xt[1:-1]
    dX = np.diff(Xt[1:])
    Y = dX / Delta
    Tquad = dX**2 / Delta

    # --------------------------------------------------
    # Hawkes intensities and conditional expectation f
    # --------------------------------------------------
    intensity = np.array([hawkes.intensM(s, param, times) for s in grid])
    h_nw = max(0.05 * (q2 - q1), 1e-3)
    print(h_nw)

    condiM = np.zeros((M, npas))
    for i in range(M):
        condiM[i, :] = mNW(
            x=gridx,
            X=Xstate,
            Y=intensity[1:-1, i],
            h=h_nw,
        )

    sumcondiM = np.maximum(np.sum(condiM, axis=0), 1e-8)

    # --------------------------------------------------
    # Estimation of g
    # --------------------------------------------------
    kap = 100

    collecP_g, collecalphag = collecestimcoeff(X=Xt, U=Tquad, q1=q1, q2=q2, Nn=Nn)
    ind_g = len(collecP_g)
    collecestimg = build_collection_estimates(
        gridx, q1, q2, collecalphag, ind_g, positive=True
    )

    penaltyg_vals = penaltyg(Nn, n, Delta, kap)
    res_g = adaptiveestim(
        colleccoeffalpha=collecalphag,
        collecmatP=collecP_g,
        U=Tquad,
        penalty=penaltyg_vals,
    )
    mhat_g = res_g[3]
    estimfinal_g = collecestimg[mhat_g, :]

    # --------------------------------------------------
    # Estimation of sigma^2
    # --------------------------------------------------
    seuil = 1.0
    beta_trunc = 1 / 4

    truncphi = np.array([phifunc(d / (seuil * Delta**beta_trunc)) for d in dX])
    Tquadphi = Tquad * truncphi

    collecP_sig, collecalphasig = collecestimcoeff(
        X=Xt, U=Tquadphi, q1=q1, q2=q2, Nn=Nn
    )

    ind_sig = len(collecP_sig)
    collecestimsig2 = build_collection_estimates(
        gridx, q1, q2, collecalphasig, ind_sig, positive=True
    )

    penaltysig_vals = penaltysig(Nn, n, kap)
    res_sig = adaptiveestim(
        colleccoeffalpha=collecalphasig,
        collecmatP=collecP_sig,
        U=Tquadphi,
        penalty=penaltysig_vals,
    )
    mhat_sig = res_sig[3]
    estimfinal_sig2 = collecestimsig2[mhat_sig, :]
    estimfinal_sigma = np.sqrt(np.maximum(estimfinal_sig2, 0.0))

    # --------------------------------------------------
    # Estimation of a
    # --------------------------------------------------
    estim_a2 = np.maximum((estimfinal_g - estimfinal_sig2) / sumcondiM, 0.0)
    estim_a = np.sqrt(estim_a2)

    # interpolation functions for plug-in estimation
    gridx_keep = gridx[keep]
    estim_a_keep = estim_a[keep]
    estim_sigma_keep = estimfinal_sigma[keep]

    aesti = make_interp_function(gridx_keep, estim_a_keep, floor=0.0)
    sigesti = make_interp_function(gridx_keep, estim_sigma_keep, floor=0.05)

    # --------------------------------------------------
    # Estimation of b
    # --------------------------------------------------
    U_b = Y - (aesti(Xstate) * isjumpN[1:]) / Delta

    collecP_b, collecalphab = collecestimcoeff(X=Xt, U=U_b, q1=q1, q2=q2, Nn=Nn)
    ind_b = len(collecP_b)
    collecestimb = build_collection_estimates(
        gridx, q1, q2, collecalphab, ind_b, positive=False
    )

    rho = 3.0
    sigma02 = max(np.max(estimfinal_sig2[keep]), 1e-6)
    penaltyb_vals = np.array(
        [penaltyb(k + 1, n, Delta, rho, sigma02) for k in range(Nn)]
    )

    res_b = adaptiveestim(
        colleccoeffalpha=collecalphab,
        collecmatP=collecP_b,
        U=U_b,
        penalty=penaltyb_vals,
    )
    mhat_b = res_b[3]
    estim_b = collecestimb[mhat_b, :]

    besti = make_interp_function(gridx_keep, estim_b[keep], floor=None)

    # --------------------------------------------------
    # Simulate with estimated parameters
    # Use the same recovered noise as the original path
    # --------------------------------------------------
    Wrec = recover_noise_from_path(Xt, grid, bfunc, sigfunc, afunc, isjumpN)
    X_est = simu_jumpdiff_given_noise(
        X0=Xt[0],
        grid=grid,
        bfunc=besti,
        sigfunc=sigesti,
        afunc=aesti,
        isjumpN=isjumpN,
        W=Wrec,
    )

    # --------------------------------------------------
    # Truth on the plotting support
    # --------------------------------------------------
    xplot = gridx_keep
    true_a = afunc(xplot)
    true_b = bfunc(xplot)
    true_sigma = sigfunc(xplot)

    est_a_plot = aesti(xplot)
    est_b_plot = besti(xplot)
    est_sigma_plot = sigesti(xplot)

    # --------------------------------------------------
    # Plot estimated vs truth
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    axes[0].plot(xplot, est_a_plot, color="red", lw=2, label="estimated")
    axes[0].plot(xplot, true_a, color="blue", lw=2, label="truth")
    axes[0].set_title("Estimation of a")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("a(x)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(xplot, est_b_plot, color="red", lw=2, label="estimated")
    axes[1].plot(xplot, true_b, color="blue", lw=2, label="truth")
    axes[1].set_title("Estimation of b")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("b(x)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(xplot, est_sigma_plot, color="red", lw=2, label="estimated")
    axes[2].plot(xplot, true_sigma, color="blue", lw=2, label="truth")
    axes[2].set_title("Estimation of sigma")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel(r"$\sigma(x)$")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # Plot old and estimated jump-diffusion together
    # --------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(grid, Xt, color="blue", lw=1.8, label="old process (truth)")
    plt.plot(
        grid, X_est, color="red", lw=1.5, alpha=0.9, label="new process (estimated)"
    )
    plt.title("Jump-diffusion process: truth vs estimated")
    plt.xlabel("time")
    plt.ylabel("membrane potential")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Selected dimension for g: {mhat_g + 1}")
    print(f"Selected dimension for sigma^2: {mhat_sig + 1}")
    print(f"Selected dimension for b: {mhat_b + 1}")
    print(f"Number of truncated increments: {np.sum(truncphi != 1)}")
    print(f"Path RMSE: {np.sqrt(np.mean((X_est - Xt) ** 2)):.6f}")


if __name__ == "__main__":
    main()
