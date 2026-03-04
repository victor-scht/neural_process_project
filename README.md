# Projected estimation (a, b, sigma^2, f, g) with L2 trigonometric bases

This project mirrors the structure of your original `main_estimation.py` / `utils.py`:

- Uses the **same projection basis** `S_m` implemented in `utils.projectionSm`
- Uses the **same adaptive (penalized) selection** `utils.adaptiveestim` with `penaltyb`, `penaltyg`, `penaltysig`
- Keeps the original model functions **unchanged** (see `model.py`):
  - `bdrift(x) = -20x - 1080`
  - `sig(x) = 11.0`  (sigma, not sigma^2)
  - `ajump(x) = -0.07x - 2`

## Files

- `utils.py` (copied from your upload, unchanged)
- `model.py` (your original a/b/sig)
- `simulate.py` (synthetic Hawkes + jump-diffusion generator using utils simulators)
- `estimate.py` (projection + penalization estimation of f, g, sigma^2, a, b)
- `plots.py` (plots of projected curves + basis coefficients)
- `run.py` (entry point)

## Run

```bash
python run.py
```

Outputs are saved into `outputs/`.

## Notes on conventions (kept as in your script)

- sigma^2 estimate is `sig2_hat = (projected_sigma2_raw)/5.0` (as in your code).
- `a_hat` is computed as `sqrt( max( (g_hat - sig2_hat)/f_hat , 0 ) )`.
- drift uses `Y = diff(X[1:])/Delta` and jump term aligned with `isjumpN[1:]` (same indexing style).
