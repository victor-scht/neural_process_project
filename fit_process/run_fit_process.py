from fit_process.config import FitProcessConfig
from fit_process.estimate_real_data import run_real_data_estimation

if __name__ == "__main__":
    cfg = FitProcessConfig()
    res = run_real_data_estimation(cfg)
    print("Done.")
    print({"m_g": res["m_g"], "m_sig": res["m_sig"], "m_b": res["m_b"]})
