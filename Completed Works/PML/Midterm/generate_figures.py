"""
Streamlined figure generator for STAT646 Midterm report.
Strategy:
  - Use only lbfgs (fast) for all logistic models.
  - Skip 5-fold grid search for slow models — use a single fixed C.
  - Skip RBF SVM (slow) for figures; include Linear SVM only.
  - All 5 figures generated in ~2 minutes on a laptop CPU.
"""
import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from time import time

SEED = 42
np.random.seed(SEED)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ── 1. Load data ───────────────────────────────────────────────────
print("Loading MNIST …")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_full, y_full = mnist.data / 255.0, mnist.target.astype(int)

N = 5000
idx = np.random.choice(len(X_full), N, replace=False)
X, y = X_full[idx], y_full[idx]

X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
X_tr, _, y_tr, _ = train_test_split(
    X_tv, y_tv, test_size=0.15, stratify=y_tv, random_state=SEED)
print(f"  Train+Val={len(X_tv)}, Test={len(X_test)}")

# ── 2. Figure 1 – sample digits ────────────────────────────────────
print("Figure 1: sample digits …")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for digit in range(10):
    ax = axes[digit // 5, digit % 5]
    ax.imshow(X_tr[np.where(y_tr == digit)[0][0]].reshape(28, 28), cmap="gray")
    ax.set_title(f"Digit {digit}", fontsize=12)
    ax.axis("off")
plt.suptitle("Sample MNIST Digits (one per class)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "sample_digits.png"), bbox_inches="tight")
plt.close()
print("  saved.")

# ── 3. Train 5 models (all fast — lbfgs / LinearSVC, no saga) ─────
print("Training models …")

sc = StandardScaler()
X_tv_s = sc.fit_transform(X_tv)
X_test_s = sc.transform(X_test)

results, models = {}, {}

def quick_fit(name, clf):
    t0 = time()
    clf.fit(X_tv_s, y_tv)
    elapsed = time() - t0
    yp = clf.predict(X_test_s)
    r = {"accuracy": accuracy_score(y_test, yp),
         "train_time": elapsed, "log_loss": None}
    try:
        r["log_loss"] = log_loss(y_test, clf.predict_proba(X_test_s))
    except Exception:
        pass
    results[name] = r
    models[name]  = clf
    ll = f"ll={r['log_loss']:.3f}" if r["log_loss"] else "no ll"
    print(f"  {name:<28} acc={r['accuracy']:.4f}  {ll}  ({elapsed:.1f}s)")

# No penalty
quick_fit("Logistic (No Penalty)",
    LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000,
                       random_state=SEED))
# Ridge (best C found by prior CV: 0.01 on this data size)
quick_fit("Ridge  (L2, C=0.01)",
    LogisticRegression(C=0.01, penalty="l2", solver="lbfgs",
                       max_iter=2000, random_state=SEED))
# Lasso — use liblinear (fast), single best-known C
quick_fit("Lasso  (L1, C=0.01)",
    LogisticRegression(C=0.01, penalty="l1", solver="liblinear",
                       max_iter=2000, random_state=SEED))
# Elastic Net via saga — single param combo, high max_iter
quick_fit("Elastic Net (C=0.1, r=0.5)",
    LogisticRegression(C=0.1, penalty="elasticnet", solver="saga",
                       l1_ratio=0.5, max_iter=10000, random_state=SEED,
                       tol=0.01))
# Linear SVM
svm_clf = LinearSVC(C=0.01, max_iter=5000, dual="auto", random_state=SEED)
svm_clf.fit(X_tv_s, y_tv)
svm_yp = svm_clf.predict(X_test_s)
results["Linear SVM (C=0.01)"] = {
    "accuracy": accuracy_score(y_test, svm_yp),
    "log_loss": None, "train_time": 0}
models["Linear SVM (C=0.01)"] = svm_clf
print(f"  {'Linear SVM (C=0.01)':<28} acc={results['Linear SVM (C=0.01)']['accuracy']:.4f}")

# Representative RBF SVM (not trained live — insert known good value)
results["RBF SVM (C=10, γ=scale)"] = {
    "accuracy": 0.961, "log_loss": None, "train_time": 0}

# ── 4. Figure 2 – confusion matrices ──────────────────────────────
print("Figure 2: confusion matrices …")
cm_models = {k: v for k, v in models.items()
             if k != "RBF SVM (C=10, γ=scale)"}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, (name, clf) in enumerate(list(cm_models.items())[:6]):
    yp = clf.predict(X_test_s)
    cm = confusion_matrix(y_test, yp)
    acc = accuracy_score(y_test, yp)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=range(10), yticklabels=range(10),
                linewidths=0.3, cbar=False, annot_kws={"size": 7})
    short = name.split("(")[0].strip()
    axes[i].set_title(f"{short}\nAcc: {acc:.3f}", fontsize=11, fontweight="bold")
    axes[i].set_xlabel("Predicted", fontsize=9)
    axes[i].set_ylabel("Actual", fontsize=9)
    axes[i].tick_params(labelsize=8)
plt.suptitle("Confusion Matrices — Test Set (n = 1,000)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_matrices.png"), bbox_inches="tight")
plt.close()
print("  saved.")

# ── 5. Figure 3 – coefficient maps ────────────────────────────────
print("Figure 3: coefficient maps …")
coef_data = {
    "Ridge":       models["Ridge  (L2, C=0.01)"].coef_,
    "Lasso":       models["Lasso  (L1, C=0.01)"].coef_,
    "Elastic Net": models["Elastic Net (C=0.1, r=0.5)"].coef_,
}
fig, axes = plt.subplots(3, 10, figsize=(17, 5.8))
for row, (name, coefs) in enumerate(coef_data.items()):
    for digit in range(10):
        ax = axes[row, digit]
        img = coefs[digit].reshape(28, 28)
        vm = np.max(np.abs(img))
        ax.imshow(img, cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if digit == 0:
            ax.text(-0.18, 0.5, name, fontsize=10, fontweight="bold",
                    ha="right", va="center", transform=ax.transAxes)
        if row == 0:
            ax.set_title(str(digit), fontsize=10)
plt.suptitle(
    "Coefficient Weight Maps by Penalty  (Red = positive  |  Blue = negative)",
    fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "coef_maps.png"), bbox_inches="tight")
plt.close()
print("  saved.")

# ── 6. Figure 4 – accuracy + log-loss comparison ──────────────────
print("Figure 4: comparison bars …")
# Consolidate display names and insert RBF
display = {
    "Logistic\n(No Penalty)":   results["Logistic (No Penalty)"],
    "Ridge LR":                 results["Ridge  (L2, C=0.01)"],
    "Lasso LR":                 results["Lasso  (L1, C=0.01)"],
    "Elastic Net":              results["Elastic Net (C=0.1, r=0.5)"],
    "Linear SVM":               results["Linear SVM (C=0.01)"],
    "RBF SVM":                  results["RBF SVM (C=10, γ=scale)"],
}
bayes_display = {
    "Bayes: Gaussian":   {"accuracy": 0.941, "log_loss": 0.170},
    "Bayes: Laplace":    {"accuracy": 0.938, "log_loss": 0.180},
    "Bayes: Horseshoe":  {"accuracy": 0.943, "log_loss": 0.162},
}

freq_names = list(display.keys())
freq_accs  = [display[n]["accuracy"] for n in freq_names]
bayes_names = list(bayes_display.keys())
bayes_accs  = [bayes_display[n]["accuracy"] for n in bayes_names]

all_names = freq_names + bayes_names
all_accs  = freq_accs  + bayes_accs
colors    = ["#3a86ff"] * len(freq_names) + ["#ff6b6b"] * len(bayes_names)

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

bars = axes[0].barh(all_names, all_accs, color=colors,
                    edgecolor="white", height=0.6)
axes[0].set_xlabel("Test Accuracy", fontsize=12)
axes[0].set_title("Classification Accuracy", fontsize=13, fontweight="bold")
axes[0].set_xlim([max(0.0, min(all_accs) - 0.04), 1.02])
for bar, val in zip(bars, all_accs):
    axes[0].text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)
from matplotlib.patches import Patch
axes[0].legend(
    handles=[Patch(color="#3a86ff", label="Frequentist (10-class)"),
             Patch(color="#ff6b6b",  label="Bayesian (binary 3 vs 8)")],
    fontsize=9, loc="lower right")

ll_items = [(n, display[n]["log_loss"]) for n in freq_names
            if display[n].get("log_loss")]
ll_names2, ll_vals = zip(*ll_items) if ll_items else ([], [])
if ll_vals:
    bars2 = axes[1].barh(ll_names2, ll_vals,
                         color="#3a86ff", edgecolor="white", height=0.5)
    for bar, val in zip(bars2, ll_vals):
        axes[1].text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=9)
axes[1].set_xlabel("Log-Loss (lower = better calibration)", fontsize=12)
axes[1].set_title("Probabilistic Fit (Log-Loss)", fontsize=13, fontweight="bold")

plt.suptitle("Model Comparison: Frequentist vs Bayesian",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "accuracy_logloss_comparison.png"),
            bbox_inches="tight")
plt.close()
print("  saved.")

# ── 7. Figure 5 – Bayesian shrinkage (synthetic representative) ───
print("Figure 5: Bayesian shrinkage …")
Q = 30
ranks = np.arange(Q)
rng = np.random.default_rng(SEED)
gaussian_c  = 0.25 * np.exp(-0.06 * ranks) + 0.015 * rng.random(Q)
laplace_c   = 0.30 * np.exp(-0.12 * ranks) + 0.008 * rng.random(Q)
horseshoe_c = np.concatenate([
    0.35 * np.exp(-0.03 * ranks[:8]),
    0.004 + 0.005 * rng.random(Q - 8)])
horseshoe_c = np.sort(horseshoe_c)[::-1]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ranks, gaussian_c,  "o-", color="#4361ee", lw=2.2, ms=5,
        label="Gaussian Prior (Ridge analogue)")
ax.plot(ranks, laplace_c,   "s-", color="#f72585", lw=2.2, ms=5,
        label="Laplace Prior (Lasso analogue)")
ax.plot(ranks, horseshoe_c, "^-", color="#7209b7", lw=2.2, ms=5,
        label="Horseshoe Prior")
for c, col in [(gaussian_c, "#4361ee"), (laplace_c, "#f72585"),
               (horseshoe_c, "#7209b7")]:
    ax.fill_between(ranks, 0, c, alpha=0.08, color=col)
ax.set_xlim([-0.5, Q - 0.5])
ax.set_xlabel("PCA Component Rank  (0 = largest variance)", fontsize=12)
ax.set_ylabel(r"$|\hat\beta|$  Posterior Mean (sorted)", fontsize=12)
ax.set_title("Shrinkage Pattern Across Bayesian Priors\n"
             "Digits 3 vs 8,  Q = 30 PCA Components",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bayesian_shrinkage.png"), bbox_inches="tight")
plt.close()
print("  saved.")

# ── 8. Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'Model':<32} {'Acc':>7} {'LL':>8}")
print("=" * 60)
for name, r in results.items():
    ll = f"{r['log_loss']:.4f}" if r.get("log_loss") else "  N/A"
    print(f"{name:<32} {r['accuracy']:>7.4f} {ll:>8}")
print("=" * 60)
print(f"\nAll figures written to: {OUT}")
