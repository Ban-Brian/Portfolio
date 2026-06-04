"""Feature engineering — liquidity proxies, volatility, standardization."""

import numpy as np
import pandas as pd
from scipy import stats


def winsorize(series: pd.Series, pct: float = 0.01) -> pd.Series:
    """Clip extreme values at the given percentile on both tails."""
    lower = series.quantile(pct)
    upper = series.quantile(1 - pct)
    return series.clip(lower, upper)


def add_rolling_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Backward-looking rolling standard deviation of price changes."""
    df = df.copy()
    df["rolling_vol"] = (
        df["price_change"]
        .rolling(window=window, min_periods=1)
        .std()
        .bfill()
    )
    return df


def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive log_spread, inv_depth, and spread_depth_ratio."""
    df = df.copy()
    df["log_spread"] = np.log1p(df["spread"])
    df["inv_depth"] = 1.0 / (df["depth"] + 1e-6)
    df["spread_depth_ratio"] = df["spread"] / (df["depth"] + 1e-6)
    return df


def standardize(df: pd.DataFrame, columns: list,
                 fit_stats: dict = None) -> tuple[pd.DataFrame, dict]:
    """Z-score standardize columns. Reuses fit_stats from training if provided."""
    df = df.copy()
    if fit_stats is None:
        fit_stats = {}
        for col in columns:
            fit_stats[col] = {"mean": df[col].mean(), "std": df[col].std()}

    for col in columns:
        mu = fit_stats[col]["mean"]
        sigma = fit_stats[col]["std"]
        df[col] = (df[col] - mu) / (sigma + 1e-10)

    return df, fit_stats


def engineer_features(df: pd.DataFrame, cfg: dict,
                       fit_stats: dict = None) -> tuple[pd.DataFrame, dict]:
    """Full pipeline: winsorize → derive features → standardize."""
    feat_cfg = cfg["features"]

    pct = feat_cfg["winsorize_percentile"]
    for col in ["spread", "depth", "volatility", "trade_size"]:
        df[col] = winsorize(df[col], pct)

    df = add_liquidity_features(df)
    df = add_rolling_volatility(df, window=feat_cfg["volatility_window"])

    covariate_cols = [
        "spread", "depth", "volatility",
        "log_spread", "inv_depth", "spread_depth_ratio", "rolling_vol"
    ]
    df, fit_stats = standardize(df, covariate_cols, fit_stats)

    return df, fit_stats


if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    from dgp import load_config, generate_data

    cfg = load_config()
    df = generate_data(cfg)
    df_feat, stats = engineer_features(df, cfg)

    print("=== Feature Engineering Check ===")
    print(f"Columns: {list(df_feat.columns)}")
    print(f"\nStandardized covariate means (should be ~0):")
    for col in ["spread", "depth", "volatility", "log_spread", "inv_depth"]:
        print(f"  {col}: {df_feat[col].mean():.4f}")
