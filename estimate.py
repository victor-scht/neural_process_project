import numpy as np
from utils import (
    collecestimcoeff, adaptiveestim,
    penaltyb, penaltyg, penaltysig,
    projectionSm, phifunc
)

def projected_curve_from_coeffs(gridx, q1, q2, m, alpha_row):
    proj = projectionSm(gridx, q1, q2, m)
    coeff = alpha_row[: 2*m + 1]
    return np.sum(proj * coeff[:, None], axis=0)

def fit_projected(X, U, q1, q2, gridx, Nn, penalty_vals, positivity=False):
    '''
    Fit h(x) = E[U | X=x] by projection on trig basis S_m with adaptive m (penalized).
    Returns dict with selected m (1..Nn), coefficients, curve, and all curves.
    '''
    collecP, collecalpha = collecestimcoeff(X=X, U=U, q1=q1, q2=q2, Nn=Nn)

    K = len(collecP)
    curves = np.zeros((K, len(gridx)))
    for k in range(K):
        m = k + 1
        curves[k, :] = projected_curve_from_coeffs(gridx, q1, q2, m, collecalpha[k, :])
        if positivity:
            curves[k, :] = np.maximum(curves[k, :], 0)

    res = adaptiveestim(colleccoeffalpha=collecalpha, collecmatP=collecP, U=U, penalty=penalty_vals)
    m0 = int(res[3])         # 0-based
    m = m0 + 1               # 1-based
    curve = curves[m-1, :].copy()
    coeffs = collecalpha[m-1, : 2*m + 1].copy()
    return {
        "m": m,
        "curve": curve,
        "coeffs": coeffs,
        "curves": curves,
        "all_coeffs": collecalpha,
        "crit": res[2],
        "m0": m0,
    }

def estimate_all(grid, Delta, X, isjumpN, intensity, Nn=20, kap=100.0, rho=3.0, seuil=1.0, beta=1/8):
    '''
    Replicates your original structure:
      - estimate f in trig basis: f(x) ≈ E[sum intensity | X=x]
      - estimate g in trig basis: g(x) ≈ E[(ΔX)^2/Δ | X=x]
      - estimate sigma^2 in trig basis with truncation
      - estimate a via sqrt((g - sigma^2)/f)   (as in your code)
      - estimate b via projection of Y - (1/Δ)*a(X)*ΔN

    IMPORTANT: This keeps the same conventions as your main_estimation.py
    (including the sigma^2 "/5" scaling that appears in your code).
    '''
    n = len(X) - 2
    q1, q2 = float(np.min(X)), float(np.max(X))
    gridx = np.linspace(q1, q2, 200)

    # f: sum of intensities
    sum_intensity = np.sum(intensity[:len(X)-1, :], axis=1)
    penalty_f = penaltyg(Nn, n, Delta, kap)
    fit_f = fit_projected(X=X, U=sum_intensity, q1=q1, q2=q2, gridx=gridx, Nn=Nn, penalty_vals=penalty_f, positivity=True)

    # g: quadratic increments
    Tquad = np.diff(X[1:])**2 / Delta
    penalty_g = penaltyg(Nn, n, Delta, kap)
    fit_g = fit_projected(X=X, U=Tquad, q1=q1, q2=q2, gridx=gridx, Nn=Nn, penalty_vals=penalty_g, positivity=True)

    # sigma^2: truncated quadratic increments
    truncphi = np.array([phifunc(d / (seuil * Delta**beta)) for d in np.diff(X[1:])])
    Tquadphi = Tquad * truncphi
    penalty_s = penaltysig(Nn, n, kap)
    fit_s = fit_projected(X=X, U=Tquadphi, q1=q1, q2=q2, gridx=gridx, Nn=Nn, penalty_vals=penalty_s, positivity=True)

    # keep sigma^2 scaling exactly as in your code
    sig2_hat = fit_s["curve"] / 5.0

    # a: from (g - sig^2) / f
    f_hat = np.maximum(fit_f["curve"], 1e-12)
    a2_hat = (fit_g["curve"] - sig2_hat) / f_hat
    a_hat = np.sqrt(np.maximum(a2_hat, 0.0))

    # b: from Y - (1/Δ)*a(X)*ΔN
    # match your alignment choices: Y = diff(X[1:])/Δ and isjumpN[1:]
    Y = np.diff(X[1:]) / Delta
    a_at_X = np.interp(X, gridx, a_hat)
    termT = (1/Delta) * a_at_X[:-1] * isjumpN[1:]
    U_b = Y - termT[:-1]

    sigma02 = float(np.max(sig2_hat))
    penalty_b = np.array([penaltyb(m=k+1, n=n, Delta=Delta, rho=rho, sigma02=sigma02) for k in range(Nn)])
    fit_b = fit_projected(X=X, U=U_b, q1=q1, q2=q2, gridx=gridx, Nn=Nn, penalty_vals=penalty_b, positivity=False)

    return {
        "gridx": gridx, "q1": q1, "q2": q2,
        "fit_f": fit_f,
        "fit_g": fit_g,
        "fit_sig2_raw": fit_s,
        "sig2_hat": sig2_hat,
        "a_hat": a_hat,
        "fit_b": fit_b,
    }
