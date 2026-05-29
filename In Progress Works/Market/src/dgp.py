"""Synthetic data generation with known ground-truth CATE for benchmarking."""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "config/params.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_true_cate(depth: np.ndarray, volatility: np.ndarray,
                       cfg: dict) -> np.ndarray:
    """Compute ground-truth CATE: τ(X) = β₀ + β₁/depth + β₂·volatility."""
    dgp = cfg["dgp"]
    cate = (
        dgp["cate_intercept"]
        + dgp["cate_depth_coef"] / (depth + 1e-6)
        + dgp["cate_vol_coef"] * volatility
    )
    return cate


def generate_data(cfg: dict, seed: int = None) -> pd.DataFrame:
    """Generate synthetic trade data with heterogeneous price impact."""
    if seed is None:
        seed = cfg["random_seed"]
    rng = np.random.default_rng(seed)
    dgp = cfg["dgp"]
    n = dgp["n_observations"]

    # Market-state covariates
    spread = rng.lognormal(mean=dgp["spread_loc"], sigma=dgp["spread_scale"], size=n)
    depth = rng.exponential(scale=dgp["depth_scale"], size=n)
    volatility = np.abs(rng.normal(0, dgp["volatility_scale"], size=n))

    # Treatment (trade size) — confounded by market state
    treatment_mean = (
        dgp["treatment_depth_coef"] * depth
        + dgp["treatment_vol_coef"] * volatility
    )
    treatment_noise = rng.normal(0, dgp["treatment_noise_scale"], size=n)
    trade_size = np.maximum(treatment_mean + treatment_noise, 1.0)

    # Ground-truth CATE
    true_cate = compute_true_cate(depth, volatility, cfg)

    # Outcome: Y = g(X) + τ(X)·T + ε
    baseline_drift = -dgp["baseline_drift_scale"] * spread

    # Noise with optional AR(1) autocorrelation
    iid_noise = rng.normal(0, dgp["outcome_noise_scale"], size=n)
    ar1 = dgp.get("ar1_coefficient", 0.0)
    if ar1 != 0.0:
        noise = np.zeros(n)
        noise[0] = iid_noise[0]
        for t_idx in range(1, n):
            noise[t_idx] = ar1 * noise[t_idx - 1] + iid_noise[t_idx]
    else:
        noise = iid_noise

    price_change = baseline_drift + true_cate * trade_size + noise

    df = pd.DataFrame({
        "spread": spread,
        "depth": depth,
        "volatility": volatility,
        "trade_size": trade_size,
        "price_change": price_change,
        "true_cate": true_cate,
    })

    return df


def split_data(df: pd.DataFrame, cfg: dict):
    """Temporal train/test split to respect time ordering."""
    frac = cfg["dgp"]["train_fraction"]
    split_idx = int(len(df) * frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


if __name__ == "__main__":
    cfg = load_config()
    df = generate_data(cfg)
    train, test = split_data(df, cfg)

    print("=== DGP Sanity Check ===")
    print(f"Total observations: {len(df):,}")
    print(f"Train / Test:       {len(train):,} / {len(test):,}")
    print(f"\nCovariate summary:\n{df[['spread', 'depth', 'volatility']].describe().round(4)}")
    print(f"\nTreatment (trade_size):\n{df['trade_size'].describe().round(4)}")
    print(f"\nOutcome (price_change):\n{df['price_change'].describe().round(6)}")
    print(f"\nTrue CATE:\n{df['true_cate'].describe().round(6)}")
