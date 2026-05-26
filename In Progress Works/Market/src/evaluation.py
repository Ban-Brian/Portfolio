"""
Evaluation metrics for HTE estimator performance.

Since the synthetic DGP provides ground-truth CATE, we can directly
measure estimation accuracy. Also includes diagnostics for
time-series robustness and confidence interval calibration.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def cate_rmse(estimated: np.ndarray, true: np.ndarray) -> float:
    """Root mean squared error between estimated and true CATE."""
    return np.sqrt(np.mean((estimated - true) ** 2))


def cate_mae(estimated: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error between estimated and true CATE."""
    return np.mean(np.abs(estimated - true))


def cate_bias(estimated: np.ndarray, true: np.ndarray) -> float:
    """Mean bias (estimated - true). Positive = overestimation."""
    return np.mean(estimated - true)


def rank_correlation(estimated: np.ndarray, true: np.ndarray) -> dict:
    """
    Spearman and Kendall rank correlations between estimated and true CATE.
    Measures whether the estimator correctly ranks observations by effect size.
    """
    spearman_r, spearman_p = sp_stats.spearmanr(estimated, true)
    kendall_tau, kendall_p = sp_stats.kendalltau(estimated, true)
    return {
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "kendall_tau": kendall_tau,
        "kendall_p": kendall_p,
    }


def ci_coverage(true: np.ndarray, lower: np.ndarray,
                upper: np.ndarray) -> float:
    """Fraction of true CATEs falling inside the confidence interval."""
    covered = (true >= lower) & (true <= upper)
    return np.mean(covered)


def gates_analysis(estimated: np.ndarray, true: np.ndarray,
                    n_bins: int = 5) -> pd.DataFrame:
    """
    Group Average Treatment Effects (GATES) analysis.
    Bin observations by estimated CATE quintile, then report
    average estimated and true CATE within each bin.
    """
    bin_labels = pd.qcut(estimated, q=n_bins, labels=False, duplicates="drop")
    df = pd.DataFrame({
        "estimated": estimated,
        "true": true,
        "bin": bin_labels,
    })
    summary = df.groupby("bin").agg(
        mean_estimated=("estimated", "mean"),
        mean_true=("true", "mean"),
        count=("estimated", "count"),
    ).reset_index()
    summary["abs_error"] = np.abs(summary["mean_estimated"] - summary["mean_true"])
    return summary


def evaluate_all(cate_dict: dict, true_cate: np.ndarray,
                  ci_dict: dict = None, n_bins: int = 5) -> pd.DataFrame:
    """
    Run all evaluation metrics for every estimator.
    Returns a summary DataFrame with one row per estimator.
    """
    rows = []
    for name, est_cate in cate_dict.items():
        row = {"Estimator": name}
        row["RMSE"] = cate_rmse(est_cate, true_cate)
        row["MAE"] = cate_mae(est_cate, true_cate)
        row["Bias"] = cate_bias(est_cate, true_cate)

        rank_corr = rank_correlation(est_cate, true_cate)
        row["Spearman_r"] = rank_corr["spearman_r"]
        row["Kendall_tau"] = rank_corr["kendall_tau"]

        # Add CI coverage if available
        if ci_dict and name in ci_dict:
            lower, upper = ci_dict[name]
            row["CI_Coverage"] = ci_coverage(true_cate, lower, upper)
        else:
            row["CI_Coverage"] = np.nan

        rows.append(row)

    results = pd.DataFrame(rows)
    results = results.set_index("Estimator")
    return results


def iid_vs_dependent_comparison(estimator_suite, X_test, true_cate,
                                  cfg, n_shuffles: int = 5):
    """
    Compare estimator performance on original (dependent) vs shuffled (IID) data.
    Quantifies how much time-series dependence degrades CATE estimation.
    """
    from .dgp import generate_data, split_data
    from .features import engineer_features

    results = []

    for i in range(n_shuffles):
        # Generate fresh data with same DGP but different seed
        cfg_copy = cfg.copy()
        df_shuffled = generate_data(cfg_copy, seed=cfg["random_seed"] + i + 100)
        # Shuffle to break time dependence
        df_shuffled = df_shuffled.sample(frac=1.0, random_state=i).reset_index(drop=True)

        _, test_shuffled = split_data(df_shuffled, cfg_copy)
        test_shuffled, _ = engineer_features(test_shuffled, cfg_copy)

        X_shuf = test_shuffled[["spread", "depth", "volatility"]].values
        true_shuf = test_shuffled["true_cate"].values

        cate_dict = estimator_suite.estimate_cate(X_shuf)
        for name, est_cate in cate_dict.items():
            results.append({
                "shuffle": i,
                "estimator": name,
                "rmse": cate_rmse(est_cate, true_shuf),
            })

    return pd.DataFrame(results)


# --- Quick check ---
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    true = rng.normal(0.001, 0.0005, 1000)
    estimated = true + rng.normal(0, 0.0002, 1000)

    print("=== Evaluation Check ===")
    print(f"RMSE: {cate_rmse(estimated, true):.6f}")
    print(f"MAE:  {cate_mae(estimated, true):.6f}")
    print(f"Bias: {cate_bias(estimated, true):.6f}")
    print(f"Rank: {rank_correlation(estimated, true)}")
