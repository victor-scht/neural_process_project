import numpy as np
import matplotlib.pyplot as plt
from utils import projectionSm
from model import bdrift, sig, ajump

def plot_raster(spikesneurons, path):
    plt.figure(figsize=(12, 4))
    for i, s in enumerate(spikesneurons):
        if len(s) == 0:
            continue
        plt.vlines(s, i+0.1, i+0.9, lw=0.6)
    plt.title("Hawkes spikes (raster)")
    plt.xlabel("t")
    plt.ylabel("neuron")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def plot_X(grid, X, path):
    plt.figure(figsize=(12, 4))
    plt.plot(grid, X, lw=1.2)
    plt.title("Jump-diffusion trajectory X(t)")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def plot_function_truth_vs_est(gridx, truth, est, title, ylabel, path):
    plt.figure(figsize=(7, 4))
    plt.plot(gridx, truth, label="truth")
    plt.plot(gridx, est, label="estimated (projected)", alpha=0.9)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def plot_projected_basis(gridx, q1, q2, m, coeffs, title, path):
    # show reconstruction + basis components magnitude
    proj = projectionSm(gridx, q1, q2, m)
    recon = (proj * coeffs[:, None]).sum(axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(gridx, recon, lw=1.5, label="projected curve")
    plt.title(title + f" (m={m})")
    plt.xlabel("x")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path.replace(".png", "_curve.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.stem(np.arange(len(coeffs)), coeffs, use_line_collection=True)
    plt.title(title + f" basis coefficients (m={m})")
    plt.xlabel("basis index")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path.replace(".png", "_coeffs.png"), dpi=160)
    plt.close()

def plot_all(est, path_dir):
    gridx = est["gridx"]
    q1, q2 = est["q1"], est["q2"]

    # b truth vs est
    plot_function_truth_vs_est(
        gridx, bdrift(gridx), est["fit_b"]["curve"],
        title="b(x) drift: truth vs projected estimate", ylabel="b(x)",
        path=f"{path_dir}/b_truth_vs_est.png"
    )
    plot_projected_basis(
        gridx, q1, q2, est["fit_b"]["m"], est["fit_b"]["coeffs"],
        title="b(x) projected", path=f"{path_dir}/b_basis.png"
    )

    # a truth vs est
    plot_function_truth_vs_est(
        gridx, ajump(gridx), est["a_hat"],
        title="a(x) jump amplitude: truth vs estimated (from g, sigma^2, f)", ylabel="a(x)",
        path=f"{path_dir}/a_truth_vs_est.png"
    )

    # sigma^2 truth vs est
    sig2_true = sig(gridx)**2
    plot_function_truth_vs_est(
        gridx, sig2_true, est["sig2_hat"],
        title="sigma^2(x): truth vs projected estimate", ylabel="sigma^2(x)",
        path=f"{path_dir}/sig2_truth_vs_est.png"
    )
    plot_projected_basis(
        gridx, q1, q2, est["fit_sig2_raw"]["m"], est["fit_sig2_raw"]["coeffs"],
        title="sigma^2(x) projected (raw before /5)", path=f"{path_dir}/sig2_basis.png"
    )

    # g
    plot_function_truth_vs_est(
        gridx, est["fit_g"]["curve"], est["fit_g"]["curve"],
        title="g(x): projected estimate (no closed-form truth here)", ylabel="g(x)",
        path=f"{path_dir}/g_est.png"
    )
    plot_projected_basis(
        gridx, q1, q2, est["fit_g"]["m"], est["fit_g"]["coeffs"],
        title="g(x) projected", path=f"{path_dir}/g_basis.png"
    )

    # f
    plot_function_truth_vs_est(
        gridx, est["fit_f"]["curve"], est["fit_f"]["curve"],
        title="f(x): projected estimate of E[sum intensity | X=x]", ylabel="f(x)",
        path=f"{path_dir}/f_est.png"
    )
    plot_projected_basis(
        gridx, q1, q2, est["fit_f"]["m"], est["fit_f"]["coeffs"],
        title="f(x) projected", path=f"{path_dir}/f_basis.png"
    )
