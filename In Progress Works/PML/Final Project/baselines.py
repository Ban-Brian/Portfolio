"""
02_baselines.py
Frequentist baselines. Ridge, Lasso, and Elastic Net for the continuous
star-rating target; L1 and L2 logistic regression for the is_open target.
Hyperparameters chosen by 5-fold CV on the training split; reported metrics
are on the held-out test split and averaged over 5 seeds where indicated.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LogisticRegressionCV,
    RidgeCV,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

SEED = 42
PROC = pathlib.Path("data/processed")
OUT = pathlib.Path("results/baselines")
OUT.mkdir(parents=True, exist_ok=True)


# Load features and split indices from the preprocessed data
def load_data() -> dict:
    blob = np.load(PROC / "features.npz", allow_pickle=True)
    X = np.concatenate([blob["X_struct"], blob["X_embed"]], axis=1)
    return {
        "X": X,
        "stars": blob["stars"],
        "is_open": blob["is_open"],
        "train": blob["train_idx"],
        "val": blob["val_idx"],
        "test": blob["test_idx"],
    }


# Root mean squared error helper
def rmse(y, yhat) -> float:
    return float(np.sqrt(mean_squared_error(y, yhat)))


# Expected calibration error across probability bins
def expected_calibration_error(y_true, p_hat, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p_hat, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = p_hat[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Stars regression
# ---------------------------------------------------------------------------

# Fit Ridge, Lasso, ElasticNet with cross-validated alpha selection
def fit_regressors(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    alphas = np.logspace(-4, 2, 40)
    results = []

    ridge = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
    results.append(("Ridge", ridge.alpha_, None, ridge.predict(X_test)))

    lasso = LassoCV(alphas=alphas, cv=5, random_state=SEED, max_iter=20_000).fit(X_train, y_train)
    results.append(("Lasso", lasso.alpha_, None, lasso.predict(X_test)))

    enet = ElasticNetCV(
        alphas=alphas,
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        cv=5,
        random_state=SEED,
        max_iter=20_000,
    ).fit(X_train, y_train)
    results.append(("ElasticNet", enet.alpha_, enet.l1_ratio_, enet.predict(X_test)))

    rows = []
    for name, alpha, l1r, yhat in results:
        rows.append({
            "model": name,
            "alpha": alpha,
            "l1_ratio": l1r,
            "test_rmse": rmse(y_test, yhat),
            "test_mae": mean_absolute_error(y_test, yhat),
        })
    return pd.DataFrame(rows)


# Measure RMSE variance across multiple random training-order seeds
def seed_variance(X_train, y_train, X_test, y_test, seeds) -> pd.DataFrame:
    alphas = np.logspace(-4, 2, 40)
    per_seed = {"Ridge": [], "Lasso": [], "ElasticNet": []}
    for s in seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(X_train))
        Xp, yp = X_train[perm], y_train[perm]

        per_seed["Ridge"].append(rmse(y_test, RidgeCV(alphas=alphas, cv=5).fit(Xp, yp).predict(X_test)))
        per_seed["Lasso"].append(rmse(y_test, LassoCV(
            alphas=alphas, cv=5, random_state=s, max_iter=20_000
        ).fit(Xp, yp).predict(X_test)))
        per_seed["ElasticNet"].append(rmse(y_test, ElasticNetCV(
            alphas=alphas, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=5, random_state=s, max_iter=20_000,
        ).fit(Xp, yp).predict(X_test)))

    rows = []
    for k, v in per_seed.items():
        v = np.array(v)
        rows.append({"model": k, "mean_rmse": v.mean(), "sem": v.std(ddof=1) / np.sqrt(len(v))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Survival classification
# ---------------------------------------------------------------------------

# Fit L1 and L2 logistic regression with cross-validated C selection
def fit_classifiers(X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, np.ndarray]:
    Cs = np.logspace(-3, 2, 30)
    rows = []
    best_preds = None
    best_logloss = np.inf

    for penalty, solver in [("l2", "lbfgs"), ("l1", "liblinear")]:
        model = LogisticRegressionCV(
            Cs=Cs,
            cv=5,
            penalty=penalty,
            solver=solver,
            scoring="neg_log_loss",
            max_iter=5_000,
            random_state=SEED,
        ).fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        ll = log_loss(y_test, p)
        rows.append({
            "model": f"Logistic-{penalty.upper()}",
            "C": model.C_[0],
            "test_acc": accuracy_score(y_test, (p >= 0.5).astype(int)),
            "test_logloss": ll,
            "test_auc": roc_auc_score(y_test, p),
            "ece": expected_calibration_error(y_test, p),
        })
        if ll < best_logloss:
            best_logloss = ll
            best_preds = (p >= 0.5).astype(int)

    return pd.DataFrame(rows), best_preds


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    d = load_data()
    Xtr, Xte = d["X"][d["train"]], d["X"][d["test"]]
    ytr_s, yte_s = d["stars"][d["train"]], d["stars"][d["test"]]
    ytr_o, yte_o = d["is_open"][d["train"]], d["is_open"][d["test"]]

    reg = fit_regressors(Xtr, ytr_s, Xte, yte_s)
    reg.to_csv(OUT / "star_regression.csv", index=False)
    print(reg.to_string(index=False))

    sv = seed_variance(Xtr, ytr_s, Xte, yte_s, seeds=[42, 43, 44, 45, 46])
    sv.to_csv(OUT / "star_regression_seed_variance.csv", index=False)
    print(sv.to_string(index=False))

    clf, best_preds = fit_classifiers(Xtr, ytr_o, Xte, yte_o)
    clf.to_csv(OUT / "survival_classification.csv", index=False)
    print(clf.to_string(index=False))

    cm = confusion_matrix(yte_o, best_preds)
    pd.DataFrame(cm, index=["actual_closed", "actual_open"],
                 columns=["pred_closed", "pred_open"]).to_csv(OUT / "survival_confusion.csv")
    print(cm)


if __name__ == "__main__":
    main()
