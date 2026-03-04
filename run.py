from pathlib import Path
from simulate import simulate_synthetic, compute_intensities_on_grid
from estimate import estimate_all
import plots

def main():
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    sim = simulate_synthetic(T=15.0, Delta=4.8e-3, M=19, seed=123)
    grid = sim["grid"]
    X = sim["X"]
    spikesneurons = sim["spikesneurons"]
    paramhawkes = sim["paramhawkes"]
    Delta = sim["Delta"]
    isjumpN = sim["isjumpN"]

    # intensities aligned with estimation n=len(X)-2 -> need grid[:n+1] in original
    n = len(X) - 2
    intensity = compute_intensities_on_grid(grid, paramhawkes, spikesneurons, n_steps=n+1)

    est = estimate_all(grid=grid, Delta=Delta, X=X, isjumpN=isjumpN, intensity=intensity,
                       Nn=20, kap=100.0, rho=3.0, seuil=1.0, beta=1/8)

    # Plots
    plots.plot_raster(spikesneurons, out/"raster.png")
    plots.plot_X(grid, X, out/"X.png")
    plots.plot_all(est, str(out))

    print("Done. Plots saved to outputs/")

if __name__ == "__main__":
    main()
