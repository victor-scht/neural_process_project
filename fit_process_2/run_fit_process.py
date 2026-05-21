from fit_process_2.config import FitProcessConfig
from fit_process_2.estimate_real_data import run_real_data_estimation

if __name__ == "__main__":
    cfg = FitProcessConfig()
    res = run_real_data_estimation(cfg)
    print("Done.")
    print(
        {
            "basis_g": res["basis_g"],
            "basis_sig": res["basis_sig"],
            "basis_a": res["basis_a"],
            "basis_b": res["basis_b"],
        }
    )
