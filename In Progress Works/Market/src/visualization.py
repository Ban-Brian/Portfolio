"""
Visualization module for HTE analysis results.

Generates publication-quality plots:
  1. CATE heatmap (spread × volatility)
  2. Estimated vs true CATE scatter
  3. Partial dependence of CATE on each covariate
  4. Estimator comparison bar chart
  5. Confidence interval coverage by CATE quintile
  6. GATES analysis plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path


# --- Global plot style ---
def set_style():
    """Configure a clean, publication-ready matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_cate_heatmap(df: pd.DataFrame, cate_col: str, title: str,
                       save_path: str = None):
    """
    2D heatmap of CATE as a function of (depth, volatility).
    Uses unstandardized data for interpretability.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Bin depth and volatility into grid cells
    df_plot = df.copy()
    df_plot["depth_bin"] = pd.qcut(df_plot["depth"], q=20, duplicates="drop")
    df_plot["vol_bin"] = pd.qcut(df_plot["volatility"], q=20, duplicates="drop")

    pivot = df_plot.groupby(["vol_bin", "depth_bin"])[cate_col].mean().unstack()
    sns.heatmap(pivot, cmap="RdYlBu_r", center=0, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Order Book Depth (binned)")
    ax.set_ylabel("Volatility (binned)")

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_cate_scatter(estimated: np.ndarray, true: np.ndarray,
                       estimator_name: str, save_path: str = None):
    """Scatter plot of estimated vs true CATE with 45° reference line."""
    set_style()
    fig, ax = plt.subplots(figsize=(7, 7))

    # Subsample for readability if too many points
    n = len(true)
    idx = np.random.choice(n, size=min(5000, n), replace=False)

    ax.scatter(true[idx], estimated[idx], alpha=0.3, s=8, color="#2196F3")
    lims = [
        min(true[idx].min(), estimated[idx].min()),
        max(true[idx].max(), estimated[idx].max()),
    ]
    ax.plot(lims, lims, "k--", alpha=0.7, label="Perfect estimation")
    ax.set_xlabel("True CATE")
    ax.set_ylabel("Estimated CATE")
    ax.set_title(f"{estimator_name}: Estimated vs True CATE")
    ax.legend()

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_estimator_comparison(results_df: pd.DataFrame, metric: str = "RMSE",
                                save_path: str = None):
    """Bar chart comparing all estimators on a given metric."""
    set_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = sns.color_palette("viridis", n_colors=len(results_df))
    results_df[metric].plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel(metric)
    ax.set_title(f"Estimator Comparison — {metric}")
    ax.invert_yaxis()

    # Add value labels on bars
    for i, v in enumerate(results_df[metric]):
        ax.text(v + 0.0001, i, f"{v:.5f}", va="center", fontsize=9)

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_gates(gates_df: pd.DataFrame, estimator_name: str,
                save_path: str = None):
    """Plot GATES: mean estimated vs mean true CATE by quintile bin."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    x = gates_df["bin"]
    width = 0.35
    ax.bar(x - width / 2, gates_df["mean_estimated"], width,
           label="Estimated", color="#4CAF50", alpha=0.8)
    ax.bar(x + width / 2, gates_df["mean_true"], width,
           label="True", color="#FF5722", alpha=0.8)

    ax.set_xlabel("CATE Quintile Bin")
    ax.set_ylabel("Mean CATE")
    ax.set_title(f"GATES Analysis — {estimator_name}")
    ax.legend()
    ax.set_xticks(x)

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_partial_dependence(X: np.ndarray, cate: np.ndarray,
                             feature_names: list, save_path: str = None):
    """
    Partial dependence: binned average CATE as a function of each covariate.
    Shows how each market-state variable drives heterogeneity in price impact.
    """
    set_style()
    n_features = min(len(feature_names), X.shape[1])
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4))
    if n_features == 1:
        axes = [axes]

    for i, (ax, fname) in enumerate(zip(axes, feature_names[:n_features])):
        # Bin the feature and compute mean CATE per bin
        bins = pd.qcut(X[:, i], q=20, duplicates="drop")
        df_tmp = pd.DataFrame({"feature": X[:, i], "cate": cate, "bin": bins})
        agg = df_tmp.groupby("bin")["cate"].mean()

        ax.plot(range(len(agg)), agg.values, "o-", color="#673AB7", markersize=4)
        ax.set_xlabel(fname)
        ax.set_ylabel("Mean Estimated CATE")
        ax.set_title(f"Partial Dependence: {fname}")
        ax.tick_params(axis="x", rotation=45)
        ax.set_xticks([])

    fig.suptitle("Partial Dependence of CATE on Market-State Features", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def generate_all_plots(test_df, cate_dict, true_cate, results_df,
                        ci_dict, gates_dict, cfg):
    """Generate and save all analysis plots to the results directory."""
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = cfg["output"]["figure_format"]

    # 1. CATE heatmap (true)
    plot_cate_heatmap(
        test_df, "true_cate", "True CATE (Ground Truth)",
        save_path=str(out_dir / f"cate_heatmap_true.{fmt}")
    )

    # 2. Scatter plots and partial dependence for each estimator
    feature_names = ["spread", "depth", "volatility"]
    X_test = test_df[feature_names].values

    for name, est_cate in cate_dict.items():
        safe_name = name.replace(" ", "_").replace("-", "_").lower()

        plot_cate_scatter(
            est_cate, true_cate, name,
            save_path=str(out_dir / f"scatter_{safe_name}.{fmt}")
        )

    # 3. Estimator comparison
    plot_estimator_comparison(
        results_df, "RMSE",
        save_path=str(out_dir / f"estimator_comparison_rmse.{fmt}")
    )
    plot_estimator_comparison(
        results_df, "MAE",
        save_path=str(out_dir / f"estimator_comparison_mae.{fmt}")
    )

    # 4. Partial dependence for best estimator (lowest RMSE)
    best = results_df["RMSE"].idxmin()
    plot_partial_dependence(
        X_test, cate_dict[best], feature_names,
        save_path=str(out_dir / f"partial_dependence_{best.lower().replace('-','_')}.{fmt}")
    )

    # 5. GATES plots
    for name, gates_df in gates_dict.items():
        safe_name = name.replace(" ", "_").replace("-", "_").lower()
        plot_gates(
            gates_df, name,
            save_path=str(out_dir / f"gates_{safe_name}.{fmt}")
        )

    print(f"All plots saved to {out_dir}/")
