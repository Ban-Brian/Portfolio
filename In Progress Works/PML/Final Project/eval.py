"""
05_evaluate.py
Consolidated evaluation across the three modelling families. Produces the
head-to-head comparison table, calibration analysis, uncertainty diagnostics,
and the numeric answers to the four business questions stated in the report.
Every artefact written here is re-generated from files produced by stages 02,
03, and 04, so this script is safe to re-run on its own.
"""

from __future__ import annotations

import pathlib

import arviz as az
import numpy as np
import pandas as pd

PROC = pathlib.Path("data/processed")
BASE = pathlib.Path("results/baselines")
HIER = pathlib.Path("results/hierarchical")
VBNN = pathlib.Path("results/vbnn")
OUT = pathlib.Path("results/final")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# Load test-set targets and metadata for evaluation
def load_targets() -> dict:
    blob = np.load(PROC / "features.npz", allow_pickle=True)
    cols = list(blob["struct_cols"])
    return {
        "stars": blob["stars"],
        "is_open": blob["is_open"],
        "test": blob["test_idx"],
        "city": blob["city"],
        "cuisine": blob["cuisine"],
        "struct_cols": cols,
        "X_struct": blob["X_struct"],
        "review_count_log": blob["X_struct"][:, cols.index("review_count_log")],
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

# Build the headline model-vs-model comparison from saved CSV results
def headline_table() -> pd.DataFrame:
    base_reg = pd.read_csv(BASE / "star_regression.csv")
    base_clf = pd.read_csv(BASE / "survival_classification.csv")
    hier = pd.read_csv(HIER / "test_performance.csv")
    vbnn = pd.read_csv(VBNN / "test_performance.csv")

    rows = [
        {"model": "Ridge",
         "test_rmse": base_reg.loc[base_reg.model == "Ridge", "test_rmse"].iloc[0],
         "test_logloss": base_clf.loc[base_clf.model == "Logistic-L2", "test_logloss"].iloc[0],
         "ece": base_clf.loc[base_clf.model == "Logistic-L2", "ece"].iloc[0]},
        {"model": "ElasticNet",
         "test_rmse": base_reg.loc[base_reg.model == "ElasticNet", "test_rmse"].iloc[0],
         "test_logloss": base_clf.loc[base_clf.model == "Logistic-L1", "test_logloss"].iloc[0],
         "ece": base_clf.loc[base_clf.model == "Logistic-L1", "ece"].iloc[0]},
        {"model": "Hierarchical",
         "test_rmse": hier["value"].iloc[0],
         "test_logloss": np.nan,
         "ece": np.nan},
        {"model": "VBNN",
         "test_rmse": vbnn["test_rmse"].iloc[0],
         "test_logloss": vbnn["test_logloss"].iloc[0],
         "ece": vbnn["ece"].iloc[0]},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Uncertainty-vs-error analysis
# ---------------------------------------------------------------------------

# Correlate epistemic uncertainty with prediction error and build risk bands
def uncertainty_vs_error(t: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = np.load(VBNN / "predictions.npz")
    yte_stars = t["stars"][t["test"]]
    yte_open = t["is_open"][t["test"]]

    abs_err_stars = np.abs(yte_stars - preds["mean"])
    epi_std = np.sqrt(preds["epistemic_var"])
    corr_stars = float(np.corrcoef(epi_std, abs_err_stars)[0, 1])

    # Bin predicted probability of being open and check calibration per band
    p_open = preds["prob_open"]
    bands = np.array([0.0, 0.3, 0.5, 0.7, 0.8, 1.01])
    band_idx = np.digitize(p_open, bands) - 1
    band_idx = np.clip(band_idx, 0, len(bands) - 2)
    rows = []
    for b in range(len(bands) - 1):
        m = band_idx == b
        if not m.any():
            continue
        rows.append({
            "band": f"[{bands[b]:.1f}, {bands[b+1]:.2f})",
            "n": int(m.sum()),
            "mean_prob_open": float(p_open[m].mean()),
            "observed_open_rate": float(yte_open[m].mean()),
            "mean_epistemic_sd": float(preds["prob_open_std"][m].mean()),
        })
    return pd.DataFrame([{"metric": "corr_epistemic_abs_err_stars", "value": corr_stars}]), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Variance decomposition from the hierarchical posterior
# ---------------------------------------------------------------------------

# Decompose total variance into city, cuisine, and residual components
def variance_decomposition() -> pd.DataFrame:
    idata = az.from_netcdf(HIER / "trace.nc")
    tau_city = float(idata.posterior["tau_city"].mean())
    tau_cuis = float(idata.posterior["tau_cuisine"].mean())
    sigma = float(idata.posterior["sigma"].mean())
    total = tau_city ** 2 + tau_cuis ** 2 + sigma ** 2
    return pd.DataFrame([{
        "tau_city": tau_city,
        "tau_cuisine": tau_cuis,
        "sigma": sigma,
        "icc_city": tau_city ** 2 / total,
        "icc_cuisine": tau_cuis ** 2 / total,
        "share_residual": sigma ** 2 / total,
    }])


# ---------------------------------------------------------------------------
# Embedding ablation: fit Ridge with and without the 384-dim block
# ---------------------------------------------------------------------------

# Compare Ridge RMSE with structured features only vs. with embeddings added
def embedding_ablation() -> pd.DataFrame:
    from sklearn.linear_model import RidgeCV

    blob = np.load(PROC / "features.npz", allow_pickle=True)
    Xs = blob["X_struct"]
    Xe = blob["X_embed"]
    stars = blob["stars"]
    tr, te = blob["train_idx"], blob["test_idx"]
    alphas = np.logspace(-3, 2, 25)

    rows = []
    for name, X in [("structured_only", Xs),
                    ("structured_plus_embeddings", np.concatenate([Xs, Xe], axis=1))]:
        model = RidgeCV(alphas=alphas, cv=5).fit(X[tr], stars[tr])
        yhat = model.predict(X[te])
        rows.append({
            "feature_set": name,
            "test_rmse": float(np.sqrt(np.mean((stars[te] - yhat) ** 2))),
            "alpha": float(model.alpha_),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Business-question summary
# ---------------------------------------------------------------------------

# Extract the strongest fixed effects from the hierarchical posterior
def top_fixed_effects(k: int = 6) -> pd.DataFrame:
    idata = az.from_netcdf(HIER / "trace.nc")
    summary = az.summary(idata, var_names=["beta"], round_to=4)
    summary["abs_mean"] = summary["mean"].abs()
    summary = summary.sort_values("abs_mean", ascending=False).head(k)
    return summary[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    t = load_targets()

    head = headline_table()
    head.to_csv(OUT / "headline_comparison.csv", index=False)
    print("\n=== Headline comparison ===")
    print(head.to_string(index=False))

    unc_scalar, risk_bands = uncertainty_vs_error(t)
    unc_scalar.to_csv(OUT / "uncertainty_correlation.csv", index=False)
    risk_bands.to_csv(OUT / "survival_risk_bands.csv", index=False)
    print("\n=== Risk bands ===")
    print(risk_bands.to_string(index=False))

    var = variance_decomposition()
    var.to_csv(OUT / "variance_decomposition.csv", index=False)
    print("\n=== Variance decomposition ===")
    print(var.to_string(index=False))

    abl = embedding_ablation()
    abl.to_csv(OUT / "embedding_ablation.csv", index=False)
    print("\n=== Embedding ablation ===")
    print(abl.to_string(index=False))

    top = top_fixed_effects()
    top.to_csv(OUT / "top_fixed_effects.csv")
    print("\n=== Top fixed effects ===")
    print(top.to_string())


if __name__ == "__main__":
    main()
