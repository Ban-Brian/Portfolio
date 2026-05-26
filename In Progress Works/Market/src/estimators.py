"""
HTE estimator wrappers for price impact analysis.

Provides a unified interface around five estimators:
  1. CausalForestDML  — forest-based CATE with double ML debiasing
  2. S-Learner        — single model, treatment as a feature
  3. T-Learner        — separate models per treatment arm
  4. X-Learner        — cross-fitted imputed effects
  5. DR-Learner       — doubly robust CATE estimation

Each wrapper exposes fit(), estimate_cate(), and (where available)
confidence_intervals() methods.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)


def _get_base_model(model_name: str, n_estimators: int = 200):
    """Instantiate a scikit-learn base model by name."""
    models = {
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=4, random_state=42
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=n_estimators, max_depth=10, random_state=42
        ),
    }
    return models.get(model_name, GradientBoostingRegressor(
        n_estimators=n_estimators, random_state=42
    ))


def binarize_treatment(trade_size: np.ndarray, quantile: float = 0.75) -> np.ndarray:
    """Convert continuous trade size to binary: 1 if above quantile threshold."""
    threshold = np.quantile(trade_size, quantile)
    return (trade_size >= threshold).astype(int)


# ──────────────────────────────────────────────
# Estimator wrapper class
# ──────────────────────────────────────────────
class EstimatorSuite:
    """Fits and evaluates all five HTE estimators."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.estimators = {}
        self.fitted = {}

    def fit_all(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray):
        """
        Fit all estimators on the training data.
        Y: outcome (price_change), T: treatment (trade_size), X: covariates.
        """
        est_cfg = self.cfg["estimators"]

        # Binary treatment for meta-learners
        T_binary = binarize_treatment(T, est_cfg["binary_treatment_quantile"])

        # --- 1. Causal Forest DML (continuous treatment) ---
        self._fit_causal_forest(Y, T, X, est_cfg["causal_forest"])

        # --- 2-5. Meta-learners (binary treatment) ---
        self._fit_s_learner(Y, T_binary, X, est_cfg["s_learner"])
        self._fit_t_learner(Y, T_binary, X, est_cfg["t_learner"])
        self._fit_x_learner(Y, T_binary, X, est_cfg["x_learner"])
        self._fit_dr_learner(Y, T_binary, X, est_cfg["dr_learner"])

        return self

    def _fit_causal_forest(self, Y, T, X, params):
        """Fit CausalForestDML for continuous treatment effect estimation."""
        from econml.dml import CausalForestDML

        est = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, random_state=42),
            model_t=RandomForestRegressor(n_estimators=100, random_state=42),
            n_estimators=params["n_estimators"],
            min_samples_leaf=params["min_samples_leaf"],
            max_depth=params.get("max_depth"),
            discrete_treatment=params["discrete_treatment"],
            random_state=42,
        )
        est.fit(Y, T, X=X)
        self.fitted["CausalForest"] = est
        print("  ✓ CausalForestDML fitted")

    def _fit_s_learner(self, Y, T, X, params):
        """Fit S-Learner: single model with treatment as a feature."""
        from econml.metalearners import SLearner

        base = _get_base_model(params["base_model"], params["n_estimators"])
        est = SLearner(overall_model=base)
        est.fit(Y, T, X=X)
        self.fitted["S-Learner"] = est
        print("  ✓ S-Learner fitted")

    def _fit_t_learner(self, Y, T, X, params):
        """Fit T-Learner: separate models for treated and control."""
        from econml.metalearners import TLearner

        base = _get_base_model(params["base_model"], params["n_estimators"])
        est = TLearner(models=base)
        est.fit(Y, T, X=X)
        self.fitted["T-Learner"] = est
        print("  ✓ T-Learner fitted")

    def _fit_x_learner(self, Y, T, X, params):
        """Fit X-Learner: cross-fitted imputed treatment effects."""
        from econml.metalearners import XLearner

        base = _get_base_model(params["base_model"], params["n_estimators"])
        est = XLearner(models=base)
        est.fit(Y, T, X=X)
        self.fitted["X-Learner"] = est
        print("  ✓ X-Learner fitted")

    def _fit_dr_learner(self, Y, T, X, params):
        """Fit DR-Learner: doubly robust CATE estimation."""
        from econml.metalearners import DRLearner

        base = _get_base_model(params["base_model"], params["n_estimators"])
        est = DRLearner(models=base)
        est.fit(Y, T, X=X)
        self.fitted["DR-Learner"] = est
        print("  ✓ DR-Learner fitted")

    def estimate_cate(self, X: np.ndarray) -> dict:
        """Predict CATE for all fitted estimators on new covariates."""
        results = {}
        for name, est in self.fitted.items():
            cate = est.effect(X)
            # Flatten to 1D if needed
            results[name] = cate.flatten()
        return results

    def confidence_intervals(self, X: np.ndarray, alpha: float = 0.1) -> dict:
        """
        Get confidence intervals where supported.
        Returns dict of {name: (lower, upper)} arrays.
        """
        ci_results = {}
        for name, est in self.fitted.items():
            if hasattr(est, "effect_interval"):
                try:
                    lower, upper = est.effect_interval(X, alpha=alpha)
                    ci_results[name] = (lower.flatten(), upper.flatten())
                except Exception:
                    pass  # some estimators may not support intervals
        return ci_results
