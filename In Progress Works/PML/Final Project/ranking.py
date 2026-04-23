"""
03_hierarchical.py
Bayesian hierarchical regression on the star-rating and survival targets.
City and cuisine enter as random intercepts with a non-centred parameterisation;
two fixed features (price range and weekend hours) additionally carry
city-level random slopes. NUTS via PyMC 5. ArviZ is used for diagnostics.
"""

from __future__ import annotations

import pathlib

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

SEED = 42
PROC = pathlib.Path("data/processed")
OUT = pathlib.Path("results/hierarchical")
OUT.mkdir(parents=True, exist_ok=True)


# Load structured features and group labels from preprocessed data
def load_data() -> dict:
    blob = np.load(PROC / "features.npz", allow_pickle=True)
    cols = list(blob["struct_cols"])
    X = blob["X_struct"]

    return {
        "X": X,
        "cols": cols,
        "stars": blob["stars"],
        "is_open": blob["is_open"],
        "city": blob["city"],
        "cuisine": blob["cuisine"],
        "train": blob["train_idx"],
        "test": blob["test_idx"],
    }


# Convert string labels to integer indices for PyMC indexing
def make_indices(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    uniq = sorted(set(labels.tolist()))
    lookup = {name: i for i, name in enumerate(uniq)}
    return np.array([lookup[x] for x in labels]), uniq


# Build the hierarchical regression model with random intercepts and slopes
def build_model(d: dict) -> tuple[pm.Model, dict]:
    tr = d["train"]
    X = d["X"][tr]
    y = d["stars"][tr]
    z_features = ["price_range", "weekend_open_hours"]
    z_idx = [d["cols"].index(c) for c in z_features]
    Z = X[:, z_idx]

    city_idx, cities = make_indices(d["city"][tr])
    cuis_idx, cuisines = make_indices(d["cuisine"][tr])

    coords = {"city": cities, "cuisine": cuisines,
              "feature": d["cols"], "slope_feature": z_features}

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X", X)
        Z_data = pm.Data("Z", Z)
        city_data = pm.Data("city_idx", city_idx)
        cuis_data = pm.Data("cuisine_idx", cuis_idx)

        # Global intercept and fixed-effect coefficients
        alpha0 = pm.Normal("alpha0", 0.0, 5.0)
        beta = pm.Normal("beta", 0.0, 2.5, dims="feature")

        # Non-centred city random intercepts
        tau_city = pm.HalfNormal("tau_city", 1.0)
        alpha_city_raw = pm.Normal("alpha_city_raw", 0.0, 1.0, dims="city")
        alpha_city = pm.Deterministic("alpha_city", tau_city * alpha_city_raw, dims="city")

        # Non-centred cuisine random intercepts
        tau_cuis = pm.HalfNormal("tau_cuisine", 1.0)
        alpha_cuis_raw = pm.Normal("alpha_cuis_raw", 0.0, 1.0, dims="cuisine")
        alpha_cuis = pm.Deterministic("alpha_cuisine",
                                      tau_cuis * alpha_cuis_raw, dims="cuisine")

        # Non-centred city-level random slopes on price_range and weekend hours
        tau_slope = pm.HalfNormal("tau_slope", 1.0)
        gamma_raw = pm.Normal("gamma_raw", 0.0, 1.0, dims=("city", "slope_feature"))
        gamma_city = pm.Deterministic("gamma_city",
                                      tau_slope * gamma_raw, dims=("city", "slope_feature"))

        # Linear predictor combining all components
        mu = (
            alpha0
            + alpha_city[city_data]
            + alpha_cuis[cuis_data]
            + pm.math.dot(X_data, beta)
            + pm.math.sum(Z_data * gamma_city[city_data], axis=1)
        )

        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    meta = {
        "cities": cities,
        "cuisines": cuisines,
        "z_features": z_features,
        "z_idx": z_idx,
        "train_city_idx": city_idx,
        "train_cuis_idx": cuis_idx,
    }
    return model, meta


# Run NUTS sampling with 4 chains and return the InferenceData
def sample(model: pm.Model) -> az.InferenceData:
    with model:
        idata = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.95,
            max_treedepth=12,
            random_seed=SEED,
            init="jitter+adapt_diag",
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
        )
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=SEED))
    return idata


# Compute convergence diagnostics (Rhat, ESS, divergences)
def diagnostics(idata: az.InferenceData) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = az.summary(
        idata,
        var_names=["alpha0", "beta", "alpha_city", "alpha_cuisine",
                   "gamma_city", "tau_city", "tau_cuisine", "tau_slope", "sigma"],
        round_to=4,
    )
    diverging = int(idata.sample_stats["diverging"].sum())
    summary_tail = pd.DataFrame({
        "worst_rhat": [summary["r_hat"].max()],
        "min_ess_bulk": [summary["ess_bulk"].min()],
        "min_ess_tail": [summary["ess_tail"].min()],
        "divergences": [diverging],
    })
    return summary, summary_tail


# Posterior predictive mean for the held-out test set
def predict_test(idata, d, meta) -> np.ndarray:
    post = idata.posterior

    X_test = d["X"][d["test"]]
    Z_test = X_test[:, meta["z_idx"]]

    # Map test labels into training index space; unseen groups default to 0
    city_lookup = {c: i for i, c in enumerate(meta["cities"])}
    cuis_lookup = {c: i for i, c in enumerate(meta["cuisines"])}
    city_test = np.array([city_lookup.get(c, -1) for c in d["city"][d["test"]]])
    cuis_test = np.array([cuis_lookup.get(c, -1) for c in d["cuisine"][d["test"]]])

    # Average across chains and draws to get point estimates
    alpha0 = post["alpha0"].mean(("chain", "draw")).values
    beta = post["beta"].mean(("chain", "draw")).values
    ac = post["alpha_city"].mean(("chain", "draw")).values
    aq = post["alpha_cuisine"].mean(("chain", "draw")).values
    gc = post["gamma_city"].mean(("chain", "draw")).values

    mu = alpha0 + X_test @ beta
    mu += np.where(city_test >= 0, ac[city_test.clip(min=0)], 0.0)
    mu += np.where(cuis_test >= 0, aq[cuis_test.clip(min=0)], 0.0)
    mu += np.where(
        city_test >= 0,
        np.sum(Z_test * gc[city_test.clip(min=0)], axis=1),
        0.0,
    )
    return mu


def main() -> None:
    d = load_data()
    model, meta = build_model(d)
    idata = sample(model)
    idata.to_netcdf(OUT / "trace.nc")

    summary, tail = diagnostics(idata)
    summary.to_csv(OUT / "posterior_summary.csv")
    tail.to_csv(OUT / "diagnostics.csv", index=False)
    print(tail.to_string(index=False))

    yhat = predict_test(idata, d, meta)
    yte = d["stars"][d["test"]]
    rmse = float(np.sqrt(np.mean((yte - yhat) ** 2)))
    pd.DataFrame({"metric": ["test_rmse"], "value": [rmse]}).to_csv(
        OUT / "test_performance.csv", index=False
    )
    print(f"Hierarchical test RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
