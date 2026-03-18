set.seed(7)

# =============================================================
# Self-contained R translation of the Python script.
# Includes Hawkes-process helpers and jump-diffusion helpers.
# =============================================================

# -------------------------------------------------------------
# Hawkes utilities
# -------------------------------------------------------------
intensM <- function(t, param, times) {
  # Intensity function for M Hawkes processes (exponential kernel)
  xi <- param[[1]]
  alpha <- param[[2]]
  beta <- param[[3]]
  M <- length(xi)

  intenst <- as.numeric(xi)

  for (i in seq_len(M)) {
    for (j in seq_len(M)) {
      tj <- as.numeric(times[[j]])
      tj <- tj[tj < t]
      if (length(tj) > 0) {
        intenst[i] <- intenst[i] + sum(alpha[i, j] * beta[i] * exp(-beta[i] * (t - tj)))
      }
    }
  }

  intenst
}

intens_prefix <- function(k, M, s, times, param, xi) {
  # Sum of intensities up to the kth neuron
  alpha <- param[[2]]
  beta <- param[[3]]
  vect <- 0.0

  if (k >= 1) {
    for (m in seq_len(k)) {
      for (l in seq_len(M)) {
        tl <- as.numeric(times[[l]])
        tl <- tl[tl < s]
        if (length(tl) > 0) {
          vect <- vect + sum(alpha[m, l] * beta[m] * exp(-beta[m] * (s - tl)))
        }
      }
    }
  }

  vect <- vect + k * xi
  vect
}

simuHawkesExpoM <- function(param, M, Tend, xi) {
  # Simulation of a multi-neuron exponential Hawkes process
  # using Ogata's thinning method.
  times <- replicate(M, numeric(0), simplify = FALSE)
  s <- 0.0

  while (s < Tend) {
    lambda_bar <- intens_prefix(M, M, s, times, param, xi)
    if (lambda_bar <= 0) {
      break
    }

    u <- runif(1)
    w <- -log(u) / lambda_bar
    s <- s + w

    if (s > Tend) {
      break
    }

    current_intensities <- numeric(M)
    for (k in seq_len(M)) {
      lam_k <- intens_prefix(k, M, s, times, param, xi) -
        intens_prefix(k - 1, M, s, times, param, xi)
      current_intensities[k] <- lam_k
    }

    total_intensity <- sum(current_intensities)
    D <- runif(1)

    if (D * lambda_bar <= total_intensity) {
      probs <- current_intensities / total_intensity
      neuron <- sample.int(M, size = 1, prob = probs)
      times[[neuron]] <- c(times[[neuron]], s)
    }
  }

  times
}

# -------------------------------------------------------------
# Jump-diffusion utilities
# -------------------------------------------------------------
simu_jumpdiff <- function(X0, grid, bfunc, sigfunc, afunc, isjumpN) {
  W <- rnorm(length(grid) - 1)
  X <- numeric(length(grid))
  X[1] <- X0

  for (i in seq_len(length(grid) - 1)) {
    dt <- grid[i + 1] - grid[i]
    X[i + 1] <- X[i] +
      dt * bfunc(X[i]) +
      sqrt(dt) * sigfunc(X[i]) * W[i] +
      afunc(X[i]) * isjumpN[i]
  }

  X
}

simu_diff <- function(X0, grid, bfunc, sigfunc) {
  W <- rnorm(length(grid) - 1)
  X <- numeric(length(grid))
  X[1] <- X0

  for (i in seq_len(length(grid) - 1)) {
    dt <- grid[i + 1] - grid[i]
    X[i + 1] <- X[i] + dt * bfunc(X[i]) + sqrt(dt) * sigfunc(X[i]) * W[i]
  }

  X
}

make_trig_function <- function(c0, cos_coef, sin_coef, freqs, floor = NULL) {
  cos_coef <- as.numeric(cos_coef)
  sin_coef <- as.numeric(sin_coef)
  freqs <- as.numeric(freqs)

  function(x) {
    x <- as.numeric(x)
    val <- rep(c0, length(x))

    for (idx in seq_along(freqs)) {
      val <- val + cos_coef[idx] * cos(freqs[idx] * x)
    }
    for (idx in seq_along(freqs)) {
      val <- val + sin_coef[idx] * sin(freqs[idx] * x)
    }

    if (!is.null(floor)) {
      val <- pmax(floor, val)
    }
    val
  }
}

generate_basis_functions <- function(K = 3, seed = NULL, m = 0.0) {
  # Match the Python helper structure as closely as possible.
  if (!is.null(seed)) {
    old_seed_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    if (old_seed_exists) {
      old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    }
    set.seed(seed)
    on.exit({
      if (old_seed_exists) {
        assign(".Random.seed", old_seed, envir = .GlobalEnv)
      } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    }, add = TRUE)
  }

  freqs <- seq_len(K)

  # Strong mean-reverting part
  kappa <- runif(1, min = 0.8, max = 1.4)

  # Small oscillatory perturbation for b
  b_cos <- runif(K, min = -0.05, max = 0.05)
  b_sin <- runif(K, min = -0.05, max = 0.05)

  bfunc <- function(x) {
    x <- as.numeric(x)
    val <- -kappa * (x - m)
    for (idx in seq_along(freqs)) {
      val <- val + b_cos[idx] * cos(freqs[idx] * x)
    }
    for (idx in seq_along(freqs)) {
      val <- val + b_sin[idx] * sin(freqs[idx] * x)
    }
    val
  }

  # Diffusion sigma: strictly positive
  sig_c0 <- runif(1, min = 0.20, max = 0.35)
  sig_cos <- runif(K, min = -0.06, max = 0.06)
  sig_sin <- runif(K, min = -0.06, max = 0.06)

  sigfunc <- function(x) {
    x <- as.numeric(x)
    val <- rep(sig_c0, length(x))
    for (idx in seq_along(freqs)) {
      val <- val + sig_cos[idx] * cos(freqs[idx] * x)
    }
    for (idx in seq_along(freqs)) {
      val <- val + sig_sin[idx] * sin(freqs[idx] * x)
    }
    pmax(0.05, val)
  }

  # Jump amplitude a: positive
  a_c0 <- runif(1, min = 0.12, max = 0.28)
  a_cos <- runif(K, min = -0.05, max = 0.05)
  a_sin <- runif(K, min = -0.05, max = 0.05)

  afunc <- function(x) {
    x <- as.numeric(x)
    val <- rep(a_c0, length(x))
    for (idx in seq_along(freqs)) {
      val <- val + a_cos[idx] * cos(freqs[idx] * x)
    }
    for (idx in seq_along(freqs)) {
      val <- val + a_sin[idx] * sin(freqs[idx] * x)
    }
    pmax(0.01, val)
  }

  params <- list(
    freqs = freqs,
    kappa = kappa,
    m = m,
    b = list(cos = b_cos, sin = b_sin),
    sigma = list(c0 = sig_c0, cos = sig_cos, sin = sig_sin),
    a = list(c0 = a_c0, cos = a_cos, sin = a_sin)
  )

  list(bfunc = bfunc, sigfunc = sigfunc, afunc = afunc, basis_params = params)
}

hawkes_to_isjumpN <- function(times, grid) {
  # isjumpN[i] = number of spikes in (grid[i], grid[i+1]]
  non_empty <- lapply(times, function(t) as.numeric(t))
  non_empty <- Filter(function(x) length(x) > 0, non_empty)

  if (length(non_empty) == 0) {
    all_spikes <- numeric(0)
    isjumpN <- integer(length(grid) - 1)
    return(list(isjumpN = isjumpN, all_spikes = all_spikes))
  }

  all_spikes <- sort(unlist(non_empty, use.names = FALSE))
  h <- hist(all_spikes, breaks = grid, plot = FALSE, right = TRUE, include.lowest = TRUE)
  isjumpN <- as.integer(h$counts)

  list(isjumpN = isjumpN, all_spikes = all_spikes)
}

# -------------------------------------------------------------
# Estimation helpers
# -------------------------------------------------------------
projectionSm <- function(x, q1, q2, m) {
  Dm <- 2 * m + 1
  cst <- sqrt(2) / sqrt(q2 - q1)

  proj <- matrix(0, nrow = Dm, ncol = length(x))
  proj[1, ] <- 1 / sqrt(q2 - q1)

  if (m >= 1) {
    for (l in seq_len(m)) {
      proj[2 * l, ] <- cst * cos(2 * pi * l * (x - q1) / (q2 - q1))
      proj[2 * l + 1, ] <- cst * sin(2 * pi * l * (x - q1) / (q2 - q1))
    }
  }

  proj
}

alphachapeau <- function(P, U) {
  A <- t(P)
  fit <- lm.fit(x = A, y = U)
  alpha_hat <- fit$coefficients
  alpha_hat[is.na(alpha_hat)] <- 0
  alpha_hat
}

collecestimcoeff <- function(X, U, q1, q2, Nn) {
  colleccoeffalpha <- matrix(0, nrow = Nn, ncol = 2 * Nn + 1)
  collecP <- vector("list", Nn)
  actual_n <- 0

  for (k in seq_len(Nn)) {
    Pk <- projectionSm(X[seq_len(length(U))], q1, q2, k)
    collecP[[k]] <- Pk

    alpha_hat <- tryCatch(
      alphachapeau(Pk, U),
      error = function(e) NULL
    )

    if (is.null(alpha_hat)) {
      break
    }

    colleccoeffalpha[k, seq_len(2 * k + 1)] <- alpha_hat
    actual_n <- k
  }

  if (actual_n < Nn) {
    collecP <- collecP[seq_len(actual_n)]
    colleccoeffalpha <- colleccoeffalpha[seq_len(actual_n), , drop = FALSE]
  }

  list(collecP = collecP, colleccoeffalpha = colleccoeffalpha)
}

# -------------------------------------------------------------
# Adaptation helpers
# -------------------------------------------------------------
penaltyb <- function(m, n, Delta, rho, sigma02) {
  rho * (2 * m + 1) * sigma02 / (n * Delta)
}

penaltyg <- function(Nn, n, Delta, kap) {
  kap * seq_len(Nn) / (n * Delta)
}

penaltysig <- function(Nn, n, kap) {
  kap * seq_len(Nn) / n
}

adaptiveestim <- function(colleccoeffalpha, collecmatP, U, penalty) {
  ind <- length(collecmatP)
  estimmhat <- vector("list", ind)
  criteremhat <- numeric(ind)

  for (l in seq_len(ind)) {
    nrows <- nrow(collecmatP[[l]])
    coeff <- colleccoeffalpha[l, seq_len(nrows)]
    est <- colSums(sweep(collecmatP[[l]], 1, coeff, `*`))
    estimmhat[[l]] <- est
    criteremhat[l] <- mean((U - est)^2)
  }

  crit <- criteremhat + penalty[seq_len(ind)]
  mhat <- which.min(crit)

  list(estimmhat = estimmhat, criteremhat = criteremhat, crit = crit, mhat = mhat)
}

# -------------------------------------------------------------
# Supplementary functions
# -------------------------------------------------------------
phifunc <- function(x) {
  ax <- abs(x)
  if (ax < 1) {
    return(1)
  }
  if (ax >= 2) {
    return(0)
  }
  exp((1 / 3) + 1 / (x^2 - 4))
}

mNW <- function(x, X, Y, h, K = dnorm) {
  X <- as.numeric(X)
  Y <- as.numeric(Y)

  eval_one <- function(xi) {
    weights <- K((xi - X) / h) / h
    s <- sum(weights)
    if (s <= 0) {
      return(mean(Y))
    }
    weights <- weights / s
    sum(weights * Y)
  }

  if (length(x) == 1) {
    return(eval_one(x))
  }

  vapply(x, eval_one, numeric(1))
}

build_collection_estimates <- function(gridx, q1, q2, colleccoeffalpha, ind, positive = FALSE) {
  collecestim <- matrix(0, nrow = ind, ncol = length(gridx))

  for (k in seq_len(ind)) {
    proj <- projectionSm(gridx, q1, q2, k)
    coeff <- colleccoeffalpha[k, seq_len(2 * k + 1)]
    vals <- colSums(sweep(proj, 1, coeff, `*`))
    if (positive) {
      vals <- pmax(vals, 0.0)
    }
    collecestim[k, ] <- vals
  }

  collecestim
}

make_interp_function <- function(xgrid, ygrid, floor = NULL) {
  f0 <- approxfun(xgrid, ygrid, rule = 2)

  function(x) {
    y <- f0(x)
    if (!is.null(floor)) {
      y <- pmax(y, floor)
    }
    y
  }
}

recover_noise_from_path <- function(X, grid, bfunc, sigfunc, afunc, isjumpN) {
  W <- numeric(length(grid) - 1)

  for (i in seq_len(length(grid) - 1)) {
    dt <- grid[i + 1] - grid[i]
    denom <- sqrt(dt) * sigfunc(X[i])
    W[i] <- (X[i + 1] - X[i] - dt * bfunc(X[i]) - afunc(X[i]) * isjumpN[i]) / denom
  }

  W
}

simu_jumpdiff_given_noise <- function(X0, grid, bfunc, sigfunc, afunc, isjumpN, W) {
  Xsim <- numeric(length(grid))
  Xsim[1] <- X0

  for (i in seq_len(length(grid) - 1)) {
    dt <- grid[i + 1] - grid[i]
    Xsim[i + 1] <- Xsim[i] +
      dt * bfunc(Xsim[i]) +
      sqrt(dt) * sigfunc(Xsim[i]) * W[i] +
      afunc(Xsim[i]) * isjumpN[i]
  }

  Xsim
}

# -------------------------------------------------------------
# Main script
# -------------------------------------------------------------
main <- function() {
  # --------------------------------------------------
  # 1) Multivariate Hawkes process
  # --------------------------------------------------
  M <- 1
  Tend <- 5

  xi <- 0.5
  xi_vec <- rep(xi, M)

  beta0 <- 5
  beta <- rep(beta0, M)

  alpha <- matrix(
    c(
      0.00, 0.16, 0.08, 0.05,
      0.10, 0.00, 0.12, 0.07,
      0.06, 0.11, 0.00, 0.09,
      0.04, 0.07, 0.13, 0.00
    ),
    nrow = 4,
    byrow = TRUE
  )

  # Kept exactly as in the Python script: the 4x4 matrix above is overwritten.
  alpha <- matrix(0.4, nrow = 1, ncol = 1)

  param <- list(xi_vec, alpha, beta)
  times <- simuHawkesExpoM(param, M, Tend, xi)

  # --------------------------------------------------
  # 2) Euler grid and Hawkes jump increments
  # --------------------------------------------------
  ngrid <- as.integer(1e3)
  grid <- seq(0.0, Tend, length.out = ngrid + 1)
  hawkes_conv <- hawkes_to_isjumpN(times, grid)
  isjumpN <- hawkes_conv$isjumpN
  all_spikes <- hawkes_conv$all_spikes

  # --------------------------------------------------
  # 3) Generate basis functions
  # --------------------------------------------------
  basis_out <- generate_basis_functions(K = 6, seed = 12, m = 0.0)
  bfunc <- basis_out$bfunc
  sigfunc <- basis_out$sigfunc
  afunc <- basis_out$afunc
  basis_params <- basis_out$basis_params

  # --------------------------------------------------
  # 4) Simulate membrane potential
  # --------------------------------------------------
  X0 <- 2.0
  Xt <- simu_jumpdiff(X0, grid, bfunc, sigfunc, afunc, isjumpN)

  # ====================
  # Estimation
  # ====================
  Delta <- grid[2] - grid[1]
  n <- length(Xt) - 2
  npas <- 1000
  Nn <- 10

  q1 <- min(Xt)
  q2 <- max(Xt)
  gridx <- seq(q1, q2, length.out = npas)

  qq <- quantile(Xt, probs = c(0.05, 0.95))
  qq1 <- qq[1]
  qq2 <- qq[2]
  keep <- (gridx > qq1) & (gridx < qq2)

  Xstate <- Xt[2:(length(Xt) - 1)]
  dX <- diff(Xt[2:length(Xt)])
  Y <- dX / Delta
  Tquad <- dX^2 / Delta

  # --------------------------------------------------
  # Hawkes intensities and conditional expectation f
  # --------------------------------------------------
  # Build intensity as a length(grid) x M matrix even when M = 1.
  intensity_vals <- unlist(lapply(grid, function(s) as.numeric(intensM(s, param, times))), use.names = FALSE)
  intensity <- matrix(intensity_vals, nrow = length(grid), ncol = M, byrow = TRUE)

  h_nw <- max(0.05 * (q2 - q1), 1e-3)
  print(h_nw)

  condiM <- matrix(0, nrow = M, ncol = npas)
  for (i in seq_len(M)) {
    condiM[i, ] <- mNW(
      x = gridx,
      X = Xstate,
      Y = intensity[2:(nrow(intensity) - 1), i],
      h = h_nw
    )
  }

  sumcondiM <- pmax(colSums(condiM), 1e-8)

  # --------------------------------------------------
  # Estimation of g
  # --------------------------------------------------
  kap <- 100

  est_g_obj <- collecestimcoeff(X = Xt, U = Tquad, q1 = q1, q2 = q2, Nn = Nn)
  collecP_g <- est_g_obj$collecP
  collecalphag <- est_g_obj$colleccoeffalpha

  ind_g <- length(collecP_g)
  collecestimg <- build_collection_estimates(
    gridx = gridx,
    q1 = q1,
    q2 = q2,
    colleccoeffalpha = collecalphag,
    ind = ind_g,
    positive = TRUE
  )

  penaltyg_vals <- penaltyg(Nn, n, Delta, kap)
  res_g <- adaptiveestim(
    colleccoeffalpha = collecalphag,
    collecmatP = collecP_g,
    U = Tquad,
    penalty = penaltyg_vals
  )
  mhat_g <- res_g$mhat
  estimfinal_g <- collecestimg[mhat_g, ]

  # --------------------------------------------------
  # Estimation of sigma^2
  # --------------------------------------------------
  seuil <- 1.0
  beta_trunc <- 1 / 4

  truncphi <- vapply(dX / (seuil * Delta^beta_trunc), phifunc, numeric(1))
  Tquadphi <- Tquad * truncphi

  est_sig_obj <- collecestimcoeff(X = Xt, U = Tquadphi, q1 = q1, q2 = q2, Nn = Nn)
  collecP_sig <- est_sig_obj$collecP
  collecalphasig <- est_sig_obj$colleccoeffalpha

  ind_sig <- length(collecP_sig)
  collecestimsig2 <- build_collection_estimates(
    gridx = gridx,
    q1 = q1,
    q2 = q2,
    colleccoeffalpha = collecalphasig,
    ind = ind_sig,
    positive = TRUE
  )

  penaltysig_vals <- penaltysig(Nn, n, kap)
  res_sig <- adaptiveestim(
    colleccoeffalpha = collecalphasig,
    collecmatP = collecP_sig,
    U = Tquadphi,
    penalty = penaltysig_vals
  )
  mhat_sig <- res_sig$mhat
  estimfinal_sig2 <- collecestimsig2[mhat_sig, ]
  estimfinal_sigma <- sqrt(pmax(estimfinal_sig2, 0.0))

  # --------------------------------------------------
  # Estimation of a
  # --------------------------------------------------
  estim_a2 <- pmax((estimfinal_g - estimfinal_sig2) / sumcondiM, 0.0)
  estim_a <- sqrt(estim_a2)

  gridx_keep <- gridx[keep]
  estim_a_keep <- estim_a[keep]
  estim_sigma_keep <- estimfinal_sigma[keep]

  aesti <- make_interp_function(gridx_keep, estim_a_keep, floor = 0.0)
  sigesti <- make_interp_function(gridx_keep, estim_sigma_keep, floor = 0.05)

  # --------------------------------------------------
  # Estimation of b
  # --------------------------------------------------
  U_b <- Y - (aesti(Xstate) * isjumpN[2:length(isjumpN)]) / Delta

  est_b_obj <- collecestimcoeff(X = Xt, U = U_b, q1 = q1, q2 = q2, Nn = Nn)
  collecP_b <- est_b_obj$collecP
  collecalphab <- est_b_obj$colleccoeffalpha

  ind_b <- length(collecP_b)
  collecestimb <- build_collection_estimates(
    gridx = gridx,
    q1 = q1,
    q2 = q2,
    colleccoeffalpha = collecalphab,
    ind = ind_b,
    positive = FALSE
  )

  rho <- 3.0
  sigma02 <- max(estimfinal_sig2[keep], 1e-6)
  penaltyb_vals <- vapply(seq_len(Nn), function(k) penaltyb(k, n, Delta, rho, sigma02), numeric(1))

  res_b <- adaptiveestim(
    colleccoeffalpha = collecalphab,
    collecmatP = collecP_b,
    U = U_b,
    penalty = penaltyb_vals
  )
  mhat_b <- res_b$mhat
  estim_b <- collecestimb[mhat_b, ]

  besti <- make_interp_function(gridx_keep, estim_b[keep], floor = NULL)

  # --------------------------------------------------
  # Simulate with estimated parameters
  # Use the same recovered noise as the original path
  # --------------------------------------------------
  Wrec <- recover_noise_from_path(Xt, grid, bfunc, sigfunc, afunc, isjumpN)
  X_est <- simu_jumpdiff_given_noise(
    X0 = Xt[1],
    grid = grid,
    bfunc = besti,
    sigfunc = sigesti,
    afunc = aesti,
    isjumpN = isjumpN,
    W = Wrec
  )

  # --------------------------------------------------
  # Truth on the plotting support
  # --------------------------------------------------
  xplot <- gridx_keep
  true_a <- afunc(xplot)
  true_b <- bfunc(xplot)
  true_sigma <- sigfunc(xplot)

  est_a_plot <- aesti(xplot)
  est_b_plot <- besti(xplot)
  est_sigma_plot <- sigesti(xplot)

  # --------------------------------------------------
  # Plot estimated vs truth
  # --------------------------------------------------
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)

  par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))

  plot(xplot, est_a_plot, type = "l", col = "red", lwd = 2,
       main = "Estimation of a", xlab = "x", ylab = "a(x)")
  lines(xplot, true_a, col = "blue", lwd = 2)
  grid()
  legend("topright", legend = c("estimated", "truth"),
         col = c("red", "blue"), lwd = 2, bty = "n")

  plot(xplot, est_b_plot, type = "l", col = "red", lwd = 2,
       main = "Estimation of b", xlab = "x", ylab = "b(x)")
  lines(xplot, true_b, col = "blue", lwd = 2)
  grid()
  legend("topright", legend = c("estimated", "truth"),
         col = c("red", "blue"), lwd = 2, bty = "n")

  plot(xplot, est_sigma_plot, type = "l", col = "red", lwd = 2,
       main = "Estimation of sigma", xlab = "x", ylab = expression(sigma(x)))
  lines(xplot, true_sigma, col = "blue", lwd = 2)
  grid()
  legend("topright", legend = c("estimated", "truth"),
         col = c("red", "blue"), lwd = 2, bty = "n")

  # --------------------------------------------------
  # Plot old and estimated jump-diffusion together
  # --------------------------------------------------
  dev.new()
  plot(grid, Xt, type = "l", col = "blue", lwd = 1.8,
       main = "Jump-diffusion process: truth vs estimated",
       xlab = "time", ylab = "membrane potential")
  lines(grid, X_est, col = "red", lwd = 1.5)
  grid()
  legend("topright", legend = c("old process (truth)", "new process (estimated)"),
         col = c("blue", "red"), lwd = c(1.8, 1.5), bty = "n")

  cat(sprintf("Selected dimension for g: %d\n", mhat_g))
  cat(sprintf("Selected dimension for sigma^2: %d\n", mhat_sig))
  cat(sprintf("Selected dimension for b: %d\n", mhat_b))
  cat(sprintf("Number of truncated increments: %d\n", sum(truncphi != 1)))
  cat(sprintf("Path RMSE: %.6f\n", sqrt(mean((X_est - Xt)^2))))

  invisible(list(
    Xt = Xt,
    X_est = X_est,
    grid = grid,
    times = times,
    all_spikes = all_spikes,
    basis_params = basis_params,
    mhat_g = mhat_g,
    mhat_sig = mhat_sig,
    mhat_b = mhat_b
  ))
}

if (sys.nframe() == 0) {
  main()
}
