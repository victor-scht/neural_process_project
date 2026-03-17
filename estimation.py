import numpy as np
import hawkes
import jumpdiff

np.random.seed(7)

# --------------------------------------------------
# 1) Multivariate Hawkes process
# --------------------------------------------------
M = 4
Tend = 40.0

xi = 0.28
xi_vec = np.full(M, xi)

beta0 = 0.5
beta = np.full(M, beta0)

alpha = np.array(
    [
        [0.00, 0.16, 0.08, 0.05],
        [0.10, 0.00, 0.12, 0.07],
        [0.06, 0.11, 0.00, 0.09],
        [0.04, 0.07, 0.13, 0.00],
    ]
)

param = [xi_vec, alpha, beta]
times = hawkes.simuHawkesExpoM(param, M, Tend, xi)

# --------------------------------------------------
# 2) Euler grid and Hawkes jump increments
# --------------------------------------------------
ngrid = 5000
grid = np.linspace(0.0, Tend, ngrid + 1)
isjumpN, all_spikes = jumpdiff.hawkes_to_isjumpN(times, grid)

# --------------------------------------------------
# 3) Generate basis functions and plot them
# --------------------------------------------------
bfunc, sigfunc, afunc, basis_params = jumpdiff.generate_basis_functions(
    K=3, seed=12, m=0.0
)
