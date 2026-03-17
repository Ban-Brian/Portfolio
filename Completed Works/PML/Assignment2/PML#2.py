"""
PML Assignment 2 – Self-Supervised Denoising: Proxy vs Downstream Noise Selection
Iris dataset (K=3). Standardization fit on training data only.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

SEED        = 42
SIGMA_GRID  = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
M_PER_CLASS = 5
N_SEEDS     = 10
np.random.seed(SEED)


def add_noise(X, sigma, seed):
    """Add Gaussian noise to X (identity when σ=0)."""
    if sigma == 0:
        return X.copy()
    return X + np.random.RandomState(seed).normal(0, sigma, size=X.shape)


def train_denoiser(X_clean, sigma, alpha=1.0):
    """Fit a Ridge denoiser: noisy input → clean target."""
    X_noisy = add_noise(X_clean, sigma, seed=SEED)
    model = Ridge(alpha=alpha)
    model.fit(X_noisy, X_clean)
    return model


def eval_reconstruction_mse(denoiser, X_clean, sigma):
    """Reconstruction MSE on a held-out set."""
    X_noisy = add_noise(X_clean, sigma, seed=SEED + 1)
    return np.mean((X_clean - denoiser.predict(X_noisy)) ** 2)


def sample_labeled_set(X, y, m_per_class, rng):
    """Stratified sampling of m examples per class."""
    idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        idx.extend(rng.choice(cls_idx, size=m_per_class, replace=False))
    return np.array(idx)


def print_table(title, df):
    """Pretty-print a results table."""
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(df.to_string(index=False))
    print()


# ─── Data ─────────────────────────────────────────────────────────────────────
iris  = load_iris()
X_raw = iris.data.astype(np.float64)
y_all = iris.target


# ─── Part A: Proxy-Task Selection (no labels) ────────────────────────────────
X_ptr_raw, X_pval_raw = train_test_split(X_raw, test_size=0.25, random_state=SEED)

scaler_A    = StandardScaler().fit(X_ptr_raw)
X_ptr       = scaler_A.transform(X_ptr_raw)
X_pval      = scaler_A.transform(X_pval_raw)

proxy_results = []
for sigma in SIGMA_GRID:
    denoiser = train_denoiser(X_ptr, sigma)
    mse = eval_reconstruction_mse(denoiser, X_pval, sigma)
    proxy_results.append({"sigma": sigma, "proxy_val_MSE": round(mse, 6)})

proxy_df    = pd.DataFrame(proxy_results)
non_trivial = proxy_df[proxy_df["sigma"] > 0]
sigma_proxy = non_trivial.loc[non_trivial["proxy_val_MSE"].idxmin(), "sigma"]

print_table("TABLE 1 – σ vs Proxy-Validation Reconstruction MSE", proxy_df)
print(f"  Selection rule : smallest proxy-val MSE among σ > 0")
print(f"  σ_proxy        = {sigma_proxy}\n")


# ─── Part B: Downstream Selection (few labels) ───────────────────────────────
# gσ used as consistent feature transformer on both train and test sets
X_pre_raw, X_tst_raw, y_pre, y_tst = train_test_split(
    X_raw, y_all, test_size=0.50, random_state=SEED, stratify=y_all
)

scaler_B = StandardScaler().fit(X_pre_raw)
X_pre    = scaler_B.transform(X_pre_raw)
X_tst    = scaler_B.transform(X_tst_raw)

rng     = np.random.RandomState(SEED)
lab_idx = sample_labeled_set(X_pre, y_pre, M_PER_CLASS, rng)
X_lab   = X_pre[lab_idx]
y_lab   = y_pre[lab_idx]

down_results = []

for sigma in SIGMA_GRID:
    denoiser    = train_denoiser(X_pre, sigma)
    X_lab_repr  = denoiser.predict(X_lab)
    X_tst_repr  = denoiser.predict(X_tst)

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_lab_repr, y_lab)
    acc = accuracy_score(y_tst, clf.predict(X_tst_repr))
    down_results.append({"sigma": sigma, "test_accuracy": round(acc, 4)})

down_df    = pd.DataFrame(down_results)
best_acc   = down_df["test_accuracy"].max()
sigma_down = down_df[down_df["test_accuracy"] == best_acc]["sigma"].min()

print_table(f"TABLE 2 – σ vs Downstream Test Accuracy (m={M_PER_CLASS}/class)", down_df)
print(f"  Selection rule : highest test accuracy (ties → smallest σ)")
print(f"  σ_down         = {sigma_down}\n")


# ─── Multi-Seed Variance (Part B over N_SEEDS labeled sets) ──────────────────
multi_acc = {sigma: [] for sigma in SIGMA_GRID}

for s in range(N_SEEDS):
    rng_s = np.random.RandomState(SEED + 100 + s)
    idx_s = sample_labeled_set(X_pre, y_pre, M_PER_CLASS, rng_s)
    X_s, y_s = X_pre[idx_s], y_pre[idx_s]

    for sigma in SIGMA_GRID:
        denoiser = train_denoiser(X_pre, sigma)
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(denoiser.predict(X_s), y_s)
        multi_acc[sigma].append(accuracy_score(y_tst, clf.predict(denoiser.predict(X_tst))))

multi_rows = []
for sigma in SIGMA_GRID:
    arr = np.array(multi_acc[sigma])
    multi_rows.append({"sigma": sigma, "mean_acc": round(arr.mean(), 4), "std_acc": round(arr.std(), 4)})

multi_df = pd.DataFrame(multi_rows)
print(f"\n{'='*60}")
print(f"  MULTI-SEED VARIANCE ({N_SEEDS} seeds, m={M_PER_CLASS}/class)")
print(f"{'='*60}\n")
print(multi_df.to_string(index=False))

best_mean    = multi_df["mean_acc"].max()
sigma_down_m = multi_df[multi_df["mean_acc"] == best_mean]["sigma"].min()
print(f"\n  Best σ by mean accuracy = {sigma_down_m}\n")


# ─── Summary ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  σ_proxy = {sigma_proxy}  (proxy-val MSE, no labels)")
print(f"  σ_down  = {sigma_down}  (downstream accuracy, {M_PER_CLASS} labels/class)")
print()
if sigma_proxy == sigma_down:
    print("  ✓ Both methods agree on the optimal noise level.")
else:
    print("  ✗ The two methods select different noise levels, showing that")
    print("    proxy-task and downstream objectives can favour different σ.")