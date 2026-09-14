"""
Price Impact Estimation — LOBSTER-Style Dataset
================================================
Implements Steps 1–5 of the pseudocode guide using a synthetic
order-book dataset modeled after LOBSTER message/orderbook files.
Swap in real LOBSTER CSVs by replacing the generate_lobster_data() call.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)


# ── STEP 1: Generate synthetic LOBSTER-style data ────────────────────────────

def generate_lobster_data(n=5000):
    """
    Creates a DataFrame mimicking cleaned LOBSTER order-book features.
    Replace this function with a real CSV loader when data is available.
    """
    # Market condition features
    bid_ask_spread = np.abs(np.random.normal(0.02, 0.008, n))
    order_book_depth = np.random.exponential(500, n)
    volatility = np.abs(np.random.normal(0.01, 0.005, n))
    liquidity_score = order_book_depth / (bid_ask_spread + 1e-9) / 1e4

    # Trade size with a fat tail (log-normal)
    trade_size = np.random.lognormal(mean=5, sigma=1.2, size=n)

    # Price change depends on trade size, spread, and volatility
    base_impact = 0.0003 * trade_size / (order_book_depth + 1)
    spread_effect = 0.5 * bid_ask_spread
    vol_effect = 2.0 * volatility
    noise = np.random.normal(0, 0.001, n)
    price_change = base_impact + spread_effect + vol_effect + noise

    df = pd.DataFrame({
        "trade_size": trade_size,
        "price_change": price_change,
        "bid_ask_spread": bid_ask_spread,
        "order_book_depth": order_book_depth,
        "volatility": volatility,
        "liquidity_score": liquidity_score,
    })
    return df


data = generate_lobster_data()
print(f"Loaded {len(data)} rows")
print(data.describe().round(4))
print()


# ── STEP 2: Define treatment and outcome ─────────────────────────────────────

threshold = data["trade_size"].quantile(0.75)
data["treatment"] = (data["trade_size"] > threshold).astype(int)
data["outcome"] = data["price_change"]

covariates = ["bid_ask_spread", "order_book_depth", "volatility", "liquidity_score"]
X = data[covariates].values
T = data["treatment"].values
Y = data["outcome"].values

print(f"Treatment threshold (75th pct): {threshold:.1f}")
print(f"Treated: {T.sum()}  |  Control: {(1 - T).sum():.0f}")
print()


# ── STEP 4: Meta-learners ───────────────────────────────────────────────────

results = {}

# --- S-Learner ---
X_with_T = np.column_stack([X, T])
s_model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
s_model.fit(X_with_T, Y)

X_t1 = np.column_stack([X, np.ones(len(X))])
X_t0 = np.column_stack([X, np.zeros(len(X))])
tau_s = s_model.predict(X_t1) - s_model.predict(X_t0)
results["S-Learner"] = tau_s
print(f"S-Learner  — avg CATE: {tau_s.mean():.6f}")


# --- T-Learner ---
treated_mask = T == 1
control_mask = T == 0

model_t1 = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
model_t0 = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
model_t1.fit(X[treated_mask], Y[treated_mask])
model_t0.fit(X[control_mask], Y[control_mask])

tau_t = model_t1.predict(X) - model_t0.predict(X)
results["T-Learner"] = tau_t
print(f"T-Learner  — avg CATE: {tau_t.mean():.6f}")


# --- X-Learner ---
imputed_treated = Y[treated_mask] - model_t0.predict(X[treated_mask])
imputed_control = model_t1.predict(X[control_mask]) - Y[control_mask]

effect_model_t = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
effect_model_c = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
effect_model_t.fit(X[treated_mask], imputed_treated)
effect_model_c.fit(X[control_mask], imputed_control)

# Propensity score for blending
prop_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
prop_model.fit(X, T)
propensity = prop_model.predict_proba(X)[:, 1]

tau_x = propensity * effect_model_t.predict(X) + (1 - propensity) * effect_model_c.predict(X)
results["X-Learner"] = tau_x
print(f"X-Learner  — avg CATE: {tau_x.mean():.6f}")


# --- DR-Learner ---
mu1 = model_t1.predict(X)
mu0 = model_t0.predict(X)
ps = np.clip(propensity, 0.05, 0.95)

tau_dr = (mu1 - mu0
          + T * (Y - mu1) / ps
          - (1 - T) * (Y - mu0) / (1 - ps))
results["DR-Learner"] = tau_dr
print(f"DR-Learner — avg CATE: {tau_dr.mean():.6f}")
print()


# ── STEP 5: Heterogeneity analysis by market condition ───────────────────────

def bin_and_average(series, tau, labels):
    """Split a feature into quantile bins and compute mean CATE per bin."""
    bins = pd.qcut(series, q=len(labels), labels=labels, duplicates="drop")
    return pd.DataFrame({"bin": bins, "cate": tau}).groupby("bin", observed=False)["cate"].mean()


condition_labels = {
    "liquidity_score": ["Low", "Medium", "High"],
    "volatility":      ["Low", "Medium", "High"],
    "bid_ask_spread":  ["Narrow", "Medium", "Wide"],
}

fig, axes = plt.subplots(len(condition_labels), len(results),
                         figsize=(16, 10), sharey="row")

for col_idx, (learner, tau) in enumerate(results.items()):
    for row_idx, (feature, labels) in enumerate(condition_labels.items()):
        ax = axes[row_idx, col_idx]
        grouped = bin_and_average(data[feature], tau, labels)
        grouped.plot.bar(ax=ax, color=["#3b82f6", "#8b5cf6", "#ef4444"])
        ax.set_title(f"{learner}" if row_idx == 0 else "")
        ax.set_ylabel(feature if col_idx == 0 else "")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)

fig.suptitle("Heterogeneous Treatment Effects by Market Condition", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig("heterogeneity_analysis.png", dpi=150, bbox_inches="tight")
print("Saved: heterogeneity_analysis.png")


# ── Summary table ────────────────────────────────────────────────────────────

summary_rows = []
for learner, tau in results.items():
    row = {"Learner": learner, "Mean CATE": f"{tau.mean():.6f}",
           "Std CATE": f"{tau.std():.6f}"}
    for feature, labels in condition_labels.items():
        grouped = bin_and_average(data[feature], tau, labels)
        for label in labels:
            row[f"{feature}_{label}"] = f"{grouped.get(label, 0):.6f}"
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
print("\n=== Treatment Effect Summary ===")
print(summary.to_string(index=False))
