"""
Main pipeline — end-to-end execution of the Price Impact HTE analysis.

Steps:
  1. Load configuration
  2. Generate synthetic data with known CATE
  3. Engineer features
  4. Fit all HTE estimators
  5. Evaluate against ground truth
  6. Generate publication-quality figures
  7. Save summary results
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path so src modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from src.dgp import load_config, generate_data, split_data
from src.features import engineer_features
from src.estimators import EstimatorSuite
from src.evaluation import evaluate_all, gates_analysis
from src.visualization import generate_all_plots


def main(config_path: str = "config/params.yaml"):
    """Run the full analysis pipeline."""
    t0 = time.time()

    # --- Step 1: Load config ---
    print("=" * 60)
    print("  Price Impact HTE Analysis Pipeline")
    print("=" * 60)
    cfg = load_config(config_path)
    print(f"✓ Config loaded from {config_path}")

    # --- Step 2: Generate synthetic data ---
    print("\n[1/6] Generating synthetic data...")
    df = generate_data(cfg)
    train_raw, test_raw = split_data(df, cfg)
    print(f"  Total: {len(df):,}  |  Train: {len(train_raw):,}  |  Test: {len(test_raw):,}")

    # --- Step 3: Feature engineering ---
    print("\n[2/6] Engineering features...")
    train, fit_stats = engineer_features(train_raw, cfg)
    test, _ = engineer_features(test_raw, cfg, fit_stats=fit_stats)
    print(f"  Features: {list(train.columns)}")

    # --- Step 4: Prepare arrays and fit estimators ---
    print("\n[3/6] Fitting HTE estimators...")
    # Covariates for the causal model
    covariate_cols = ["spread", "depth", "volatility"]
    X_train = train[covariate_cols].values
    X_test = test[covariate_cols].values
    Y_train = train["price_change"].values
    T_train = train["trade_size"].values
    true_cate_test = test["true_cate"].values

    suite = EstimatorSuite(cfg)
    suite.fit_all(Y_train, T_train, X_train)

    # --- Step 5: Estimate CATE on test set ---
    print("\n[4/6] Estimating CATE on test set...")
    cate_dict = suite.estimate_cate(X_test)
    ci_dict = suite.confidence_intervals(X_test, alpha=1 - cfg["evaluation"]["ci_level"])

    # --- Step 6: Evaluate ---
    print("\n[5/6] Evaluating estimators...")
    results_df = evaluate_all(cate_dict, true_cate_test, ci_dict,
                               n_bins=cfg["evaluation"]["n_cate_bins"])
    print("\n" + "=" * 60)
    print("  ESTIMATOR PERFORMANCE SUMMARY")
    print("=" * 60)
    print(results_df.to_string(float_format="%.6f"))
    print()

    # GATES analysis for each estimator
    gates_dict = {}
    for name, est_cate in cate_dict.items():
        gates_dict[name] = gates_analysis(
            est_cate, true_cate_test, n_bins=cfg["evaluation"]["n_cate_bins"]
        )

    # --- Step 7: Generate plots ---
    print("\n[6/6] Generating plots...")
    # Use unstandardized test data for interpretable heatmaps
    generate_all_plots(test_raw, cate_dict, true_cate_test, results_df,
                        ci_dict, gates_dict, cfg)

    # Save results table
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "estimator_results.csv")
    print(f"\nResults table saved to {out_dir / 'estimator_results.csv'}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
