import numpy as np

# ============================================================
# DO NOT CHANGE: original a, b, sig definitions (as in user's code)
# ============================================================

def bdrift(x):
    return -20 * np.asarray(x) - 1080

def sig(x):
    # constant diffusion coefficient (NOT sigma^2)
    x = np.asarray(x)
    return np.full_like(x, 11.0, dtype=float)

def ajump(x):
    return -0.07 * np.asarray(x) - 2
