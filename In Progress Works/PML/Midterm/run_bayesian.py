"""
Run the Bayesian Part C models and print exact values for Tables 2 & 3.
Mirrors the notebook logic with all fixes applied.
"""
import os
import subprocess
sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path']).decode().strip()
os.environ['CPATH'] = f"{sdk_path}/usr/include/c++/v1:{sdk_path}/usr/include"
os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64,optimizer=fast_compile'

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
import pymc as pm
import arviz as az

SEED = 42
np.random.seed(SEED)
N_SAMPLES = 5000
N_COMPONENTS = 30

# ── 1. Reproduce the exact data splits from the notebook ──────────
print("Loading MNIST …")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_full, y_full = mnist.data, mnist.target.astype(int)

idx = np.random.choice(len(X_full), N_SAMPLES, replace=False)
X, y = X_full[idx], y_full[idx]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"  Train+Val: {len(X_train_val)}, Test: {len(X_test)}")

# ── 2. Binary subset from X_train_val ─────────────────────────────
DIGIT1, DIGIT2 = 3, 8

train_mask = (y_train_val == DIGIT1) | (y_train_val == DIGIT2)
test_mask  = (y_test == DIGIT1)      | (y_test == DIGIT2)

X_train_bin = X_train_val[train_mask]
y_train_bin = (y_train_val[train_mask] == DIGIT2).astype(int)
X_test_bin  = X_test[test_mask]
y_test_bin  = (y_test[test_mask] == DIGIT2).astype(int)

print(f"  Binary train: {len(y_train_bin)} "
      f"(class 0: {(y_train_bin==0).sum()}, class 1: {(y_train_bin==1).sum()})")
print(f"  Binary test:  {len(y_test_bin)}")

# ── 3. Scale + PCA ────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_bin)
X_test_s  = scaler.transform(X_test_bin)

pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca  = pca.transform(X_test_s)

pca_var = pca.explained_variance_ratio_.sum()
print(f"  PCA variance explained by {N_COMPONENTS} components: {pca_var:.4f} ({pca_var*100:.1f}%)")

# ── Helper: proper posterior predictive ────────────────────────────
def posterior_predictive(trace, X_test_pca, y_test_bin):
    """Average sigmoid across all posterior samples."""
    post = trace.posterior
    beta_s = post["beta"].stack(s=("chain", "draw")).values     # (Q, S)
    a_s    = post["intercept"].stack(s=("chain", "draw")).values # (S,)
    eta_s  = X_test_pca @ beta_s + a_s                          # (N, S)
    probs_s = 1 / (1 + np.exp(-eta_s))                          # (N, S)
    probs     = probs_s.mean(axis=1)
    probs_std = probs_s.std(axis=1)
    preds     = (probs > 0.5).astype(int)
    acc = accuracy_score(y_test_bin, preds)
    ll  = log_loss(y_test_bin, probs)
    return acc, ll, probs, probs_std

def print_diagnostics(trace, name):
    """Print convergence diagnostics."""
    summary = az.summary(trace, var_names=["beta", "intercept"])
    rhat_min, rhat_max = summary["r_hat"].min(), summary["r_hat"].max()
    ess_min = summary["ess_bulk"].min()
    divs = 0
    if hasattr(trace, "sample_stats"):
        d = trace.sample_stats.get("diverging", None)
        if d is not None:
            divs = int(d.sum().values)
    print(f"  R-hat: [{rhat_min:.3f}, {rhat_max:.3f}]  "
          f"ESS bulk min: {ess_min:.0f}  Divergences: {divs}")

# ── 4. Fit Bayesian models ────────────────────────────────────────
results = {}

# (i) Gaussian Prior
print("\n═══ Gaussian Prior (Ridge analogue) ═══")
with pm.Model() as model_gaussian:
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=1, shape=N_COMPONENTS)
    eta = intercept + pm.math.dot(X_train_pca, beta)
    p = pm.math.sigmoid(eta)
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train_bin)
    trace_g = pm.sample(draws=1000, tune=500, cores=1, chains=2,
                        random_seed=SEED, return_inferencedata=True)

print_diagnostics(trace_g, "Gaussian")
acc, ll, probs_g, std_g = posterior_predictive(trace_g, X_test_pca, y_test_bin)
beta_mean_g = trace_g.posterior["beta"].mean(dim=["chain", "draw"]).values
results["Gaussian Prior"] = {"acc": acc, "ll": ll, "probs": probs_g, "std": std_g,
                             "beta_mean": beta_mean_g}
print(f"  Accuracy: {acc:.4f}  Log-Loss: {ll:.4f}")

# (ii) Laplace Prior
print("\n═══ Laplace Prior (Lasso analogue) ═══")
with pm.Model() as model_laplace:
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    beta = pm.Laplace("beta", mu=0, b=1, shape=N_COMPONENTS)
    eta = intercept + pm.math.dot(X_train_pca, beta)
    p = pm.math.sigmoid(eta)
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train_bin)
    trace_l = pm.sample(draws=1000, tune=500, cores=1, chains=2,
                        random_seed=SEED, return_inferencedata=True)

print_diagnostics(trace_l, "Laplace")
acc, ll, probs_l, std_l = posterior_predictive(trace_l, X_test_pca, y_test_bin)
beta_mean_l = trace_l.posterior["beta"].mean(dim=["chain", "draw"]).values
results["Laplace Prior"] = {"acc": acc, "ll": ll, "probs": probs_l, "std": std_l,
                            "beta_mean": beta_mean_l}
print(f"  Accuracy: {acc:.4f}  Log-Loss: {ll:.4f}")

# (iii) Horseshoe Prior (non-centered)
print("\n═══ Horseshoe Prior (non-centered) ═══")
with pm.Model() as model_horseshoe:
    tau = pm.HalfCauchy("tau", beta=1)
    lam = pm.HalfCauchy("lam", beta=1, shape=N_COMPONENTS)
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    z = pm.Normal("z", 0, 1, shape=N_COMPONENTS)
    beta = pm.Deterministic("beta", z * lam * tau)
    eta = intercept + pm.math.dot(X_train_pca, beta)
    p = pm.math.sigmoid(eta)
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train_bin)
    trace_h = pm.sample(draws=1000, tune=500, cores=1, chains=2,
                        random_seed=SEED, return_inferencedata=True,
                        target_accept=0.9)

print_diagnostics(trace_h, "Horseshoe")
acc, ll, probs_h, std_h = posterior_predictive(trace_h, X_test_pca, y_test_bin)
beta_mean_h = trace_h.posterior["beta"].mean(dim=["chain", "draw"]).values
results["Horseshoe Prior"] = {"acc": acc, "ll": ll, "probs": probs_h, "std": std_h,
                              "beta_mean": beta_mean_h}
print(f"  Accuracy: {acc:.4f}  Log-Loss: {ll:.4f}")

# ── 5. Print table values ─────────────────────────────────────────
print("\n" + "=" * 72)
print("TABLE 2 VALUES (paste into report):")
print("=" * 72)
print(f"{'Model':<20} {'Test Acc':>10} {'Log-Loss':>10}")
print("-" * 42)
for name, r in results.items():
    print(f"{name:<20} {r['acc']:>10.4f} {r['ll']:>10.4f}")

print("\n" + "=" * 72)
print("TABLE 3 VALUES — Top 5 most uncertain (Gaussian Prior):")
print("=" * 72)
difficult = np.argsort(std_g)[-5:][::-1]
print(f"{'Rank':>4} {'True':>5} {'p_hat':>8} {'sigma_hat':>10} {'Pred':>5} {'Correct':>8}")
print("-" * 42)
for rank, idx in enumerate(difficult, 1):
    pred = 1 if probs_g[idx] > 0.5 else 0
    correct = "yes" if pred == y_test_bin[idx] else "NO"
    label = f"Digit {DIGIT2}" if y_test_bin[idx] == 1 else f"Digit {DIGIT1}"
    print(f"{rank:>4} {label:>10} {probs_g[idx]:>8.3f} {std_g[idx]:>10.3f} "
          f"{pred:>5} {correct:>8}")

print("\n" + "=" * 72)
print(f"PCA variance explained: {pca_var:.4f} ({pca_var*100:.1f}%)")
print("=" * 72)
print("\nDone.")
