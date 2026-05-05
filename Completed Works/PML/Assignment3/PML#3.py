import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold,
    learning_curve, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
SEED       = 42
TEST_SIZE  = 0.20
N_FOLDS    = 5
OUT_DIR    = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# Hyperparameter search grids
C_GRID     = [0.001, 0.01, 0.1, 1, 10, 100]
GAMMA_GRID = [0.001, 0.01, 0.1, 1]

np.random.seed(SEED)

# Plot style
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "serif",
})


# ─── 1. Data Loading & EDA ───────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names  = data.target_names

print("=" * 65)
print("  DATASET: Breast Cancer Wisconsin (Diagnostic)")
print("=" * 65)
print(f"  Samples : {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes : {list(target_names)} (0=malignant, 1=benign)")
class_dist = dict(zip(*np.unique(y, return_counts=True)))
print(f"  Class distribution: {class_dist}")
print()


# ─── 2. Train / Test Split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)
print(f"  Train set: {X_train.shape[0]} samples")
print(f"  Test  set: {X_test.shape[0]} samples  (held out, used once)")
print(f"  Split    : {1 - TEST_SIZE:.0%} / {TEST_SIZE:.0%}, stratified, seed={SEED}")
print()

# Cross-validation strategy
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
print(f"  CV strategy: {N_FOLDS}-fold StratifiedKFold (shuffle=True, seed={SEED})")
print()


# ─── 3. Model Definitions ────────────────────────────────────────────────────

# Each entry: (display name, Pipeline, param_grid)
model_specs = {
    "Logistic Regression": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=10_000, random_state=SEED))
        ]),
        {"clf__C": C_GRID}
    ),

    "Linear SVC": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(max_iter=50_000, dual="auto", random_state=SEED),
                cv=3
            ))
        ]),
        {"clf__estimator__C": C_GRID}
    ),

    "RBF SVC": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=SEED))
        ]),
        {"clf__C": [0.01, 0.1, 1, 10, 100],
         "clf__gamma": GAMMA_GRID}
    ),

    "Poly SVC (deg=3)": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="poly", degree=3, probability=True, random_state=SEED))
        ]),
        {"clf__C": [0.01, 0.1, 1, 10, 100],
         "clf__gamma": GAMMA_GRID}
    ),
}


# ─── 4. Hyperparameter Tuning ────────────────────────────────────────────────

def tune_model(name, pipeline, param_grid):
    """Runs GridSearchCV and returns the fitted search object."""
    print("=" * 65)
    print(f"  MODEL: {name}")
    print("=" * 65)

    gs = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="accuracy",
        refit=True, return_train_score=True, n_jobs=-1
    )
    gs.fit(X_train, y_train)

    print(f"  Best CV accuracy : {gs.best_score_:.4f}")
    print(f"  Best params      : {gs.best_params_}")
    print()
    return gs


fitted_models = {}
for name, (pipe, grid) in model_specs.items():
    fitted_models[name] = tune_model(name, pipe, grid)


# ─── 5. Test-Set Evaluation ──────────────────────────────────────────────────

print("=" * 65)
print("  FINAL TEST-SET EVALUATION")
print("=" * 65)

results = []

for name, gs in fitted_models.items():
    y_pred = gs.predict(X_test)
    y_prob = gs.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    # Readable hyperparameter string
    best = gs.best_params_
    if name == "Logistic Regression":
        hp_str = f"C={best['clf__C']}"
    elif name == "Linear SVC":
        hp_str = f"C={best['clf__estimator__C']}"
    else:
        hp_str = f"C={best['clf__C']}, γ={best['clf__gamma']}"

    results.append({
        "Model": name,
        "Best Params": hp_str,
        "CV Accuracy": gs.best_score_,
        "Test Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": auc,
        "TN": cm[0, 0], "FP": cm[0, 1],
        "FN": cm[1, 0], "TP": cm[1, 1],
    })

    print(f"\n  ── {name} ({hp_str}) ──")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Confusion : TN={cm[0,0]}  FP={cm[0,1]}  |  FN={cm[1,0]}  TP={cm[1,1]}")

results_df = pd.DataFrame(results)


# ─── 6. Summary Table ────────────────────────────────────────────────────────

print("\n")
print("=" * 65)
print("  SUMMARY TABLE")
print("=" * 65)

display_cols = ["Model", "Best Params", "CV Accuracy",
                "Test Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
fmt_df = results_df[display_cols].copy()
for col in ["CV Accuracy", "Test Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
    fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.4f}")
print(fmt_df.to_string(index=False))

# Save to CSV for report use
results_df.to_csv(OUT_DIR / "results_summary.csv", index=False)
print(f"\n  Results saved → {OUT_DIR / 'results_summary.csv'}")


# ─── 7. Visualizations ───────────────────────────────────────────────────────

# ── 7a. ROC Curves ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

for (name, gs), color in zip(fitted_models.items(), colors):
    y_prob = gs.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name}  (AUC = {auc_val:.4f})")

ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Test Set")
ax.legend(loc="lower right", framealpha=0.9)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "roc_curves.png", bbox_inches="tight")
plt.close(fig)
print(f"  Plot saved → {OUT_DIR / 'roc_curves.png'}")


# ── 7b. Confusion Matrix Heatmaps ────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

for ax, (name, gs) in zip(axes, fitted_models.items()):
    y_pred = gs.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names,
                cbar=False, ax=ax, annot_kws={"size": 14})
    ax.set_title(name, fontsize=10)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

fig.suptitle("Confusion Matrices — Test Set", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrices.png", bbox_inches="tight")
plt.close(fig)
print(f"  Plot saved → {OUT_DIR / 'confusion_matrices.png'}")


# ── 7c. CV Accuracy Heatmap for RBF SVC ──────────────────────────────────────
# Visualize how C and gamma jointly affect RBF SVC cross-validation accuracy
gs_rbf = fitted_models["RBF SVC"]
cv_results = pd.DataFrame(gs_rbf.cv_results_)

# Build the heatmap pivot table
pivot = cv_results.pivot_table(
    index="param_clf__gamma",
    columns="param_clf__C",
    values="mean_test_score"
)

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd",
            ax=ax, cbar_kws={"label": "Mean CV Accuracy"})
ax.set_title("RBF SVC — CV Accuracy by C and γ")
ax.set_xlabel("C")
ax.set_ylabel("γ")
fig.tight_layout()
fig.savefig(OUT_DIR / "rbf_cv_heatmap.png", bbox_inches="tight")
plt.close(fig)
print(f"  Plot saved → {OUT_DIR / 'rbf_cv_heatmap.png'}")


# ── 7d. Learning Curves ──────────────────────────────────────────────────────
# Show how training size affects performance for each model
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
train_sizes_frac = np.linspace(0.1, 1.0, 10)

for ax, (name, gs), color in zip(axes, fitted_models.items(), colors):
    best_pipe = gs.best_estimator_

    train_sizes, train_scores, val_scores = learning_curve(
        best_pipe, X_train, y_train,
        train_sizes=train_sizes_frac,
        cv=cv, scoring="accuracy", n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.15, color=color)
    ax.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.15, color=color)
    ax.plot(train_sizes, train_mean, "o-", color=color, label="Train", markersize=4)
    ax.plot(train_sizes, val_mean, "s--", color=color, label="Val", markersize=4)

    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Training Samples")
    ax.set_ylim(0.90, 1.01)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Accuracy")
fig.suptitle("Learning Curves (best estimator per model)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "learning_curves.png", bbox_inches="tight")
plt.close(fig)
print(f"  Plot saved → {OUT_DIR / 'learning_curves.png'}")


# ── 7e. Feature Importance — Logistic Regression Coefficients ─────────────────
# Logistic regression coefficients show which features drive the decision boundary
best_lr = fitted_models["Logistic Regression"].best_estimator_
coefs = best_lr.named_steps["clf"].coef_[0]

# Top 10 features by absolute coefficient magnitude
top_idx = np.argsort(np.abs(coefs))[-10:][::-1]

fig, ax = plt.subplots(figsize=(7, 4))
colors_bar = ["#2563eb" if c > 0 else "#dc2626" for c in coefs[top_idx]]
ax.barh(range(len(top_idx)), coefs[top_idx], color=colors_bar, edgecolor="white")
ax.set_yticks(range(len(top_idx)))
ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Coefficient (standardized)")
ax.set_title("Logistic Regression — Top 10 Feature Coefficients")
ax.axvline(0, color="black", lw=0.8)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "lr_feature_importance.png", bbox_inches="tight")
plt.close(fig)
print(f"  Plot saved → {OUT_DIR / 'lr_feature_importance.png'}")


# ── 7f. Metric Comparison Bar Chart ──────────────────────────────────────────
metrics_to_plot = ["Test Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
x = np.arange(len(metrics_to_plot))
width = 0.18

fig, ax = plt.subplots(figsize=(9, 4.5))
for i, (_, row) in enumerate(results_df.iterrows()):
    vals = [row[m] for m in metrics_to_plot]
    ax.bar(x + i * width, vals, width, label=row["Model"], color=colors[i],
           edgecolor="white", linewidth=0.5)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylim(0.94, 1.005)
ax.set_ylabel("Score")
ax.set_title("Test-Set Metrics Comparison")
ax.legend(loc="lower left", fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "metrics_comparison.png", bbox_inches="tight")
plt.close(fig)
print(f"  Plot saved → {OUT_DIR / 'metrics_comparison.png'}")


# ─── 8. Statistical Significance — Paired CV Fold Tests ──────────────────────
# Compare each model pair using a paired t-test on per-fold CV accuracy scores
print("\n")
print("=" * 65)
print("  STATISTICAL SIGNIFICANCE (paired t-test on CV folds)")
print("=" * 65)

fold_scores = {}
for name, gs in fitted_models.items():
    scores = cross_val_score(gs.best_estimator_, X_train, y_train,
                             cv=cv, scoring="accuracy", n_jobs=-1)
    fold_scores[name] = scores
    print(f"  {name:>22s}: fold scores = {scores.round(4)}")

model_names = list(fold_scores.keys())
sig_rows = []
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        a, b = model_names[i], model_names[j]
        t_stat, p_val = stats.ttest_rel(fold_scores[a], fold_scores[b])
        sig = "*" if p_val < 0.05 else "ns"
        sig_rows.append({"Model A": a, "Model B": b,
                         "t-stat": f"{t_stat:.3f}", "p-value": f"{p_val:.4f}",
                         "Sig (α=0.05)": sig})
        print(f"  {a} vs {b}: t={t_stat:.3f}, p={p_val:.4f} [{sig}]")

sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv(OUT_DIR / "significance_tests.csv", index=False)
print(f"  Saved → {OUT_DIR / 'significance_tests.csv'}")


# ─── 9. Calibration Curves ───────────────────────────────────────────────────
# Reliability diagrams: do predicted probabilities match observed frequencies?
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfectly calibrated")

for (name, gs), color in zip(fitted_models.items(), colors):
    y_prob = gs.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=8,
                                             strategy="uniform")
    ax.plot(prob_pred, prob_true, "o-", color=color, lw=2, label=name)

ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curves — Test Set")
ax.legend(loc="lower right", fontsize=8)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "calibration_curves.png", bbox_inches="tight")
plt.close(fig)
print(f"  Plot saved → {OUT_DIR / 'calibration_curves.png'}")


# ─── 10. PCA Decision Boundaries ─────────────────────────────────────────────
# Project data to 2D via PCA and visualize each model's decision surface
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

pca = PCA(n_components=2, random_state=SEED)
X_train_2d = pca.fit_transform(X_train_sc)
X_test_2d  = pca.transform(X_test_sc)

print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.2%} "
      f"({pca.explained_variance_ratio_[0]:.2%} + {pca.explained_variance_ratio_[1]:.2%})")

# Build a mesh for contour plotting
h = 0.15
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# Fit lightweight clones on the 2D data for visualization only
from sklearn.base import clone
vis_models = {
    "Logistic Regression": LogisticRegression(C=results_df.iloc[0]["CV Accuracy"] and
                           fitted_models["Logistic Regression"].best_params_["clf__C"],
                           max_iter=10_000, random_state=SEED),
    "Linear SVC":          LinearSVC(C=fitted_models["Linear SVC"].best_params_["clf__estimator__C"],
                                     max_iter=50_000, dual="auto", random_state=SEED),
    "RBF SVC":             SVC(kernel="rbf",
                               C=fitted_models["RBF SVC"].best_params_["clf__C"],
                               gamma=fitted_models["RBF SVC"].best_params_["clf__gamma"],
                               random_state=SEED),
    "Poly SVC (deg=3)":    SVC(kernel="poly", degree=3,
                               C=fitted_models["Poly SVC (deg=3)"].best_params_["clf__C"],
                               gamma=fitted_models["Poly SVC (deg=3)"].best_params_["clf__gamma"],
                               random_state=SEED),
}

for ax, (name, clf), color in zip(axes, vis_models.items(), colors):
    clf.fit(X_train_2d, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.contour(xx, yy, Z, colors="k", linewidths=0.5, alpha=0.4)
    ax.scatter(X_test_2d[y_test == 0, 0], X_test_2d[y_test == 0, 1],
               c="#dc2626", s=20, edgecolors="k", linewidths=0.3, label="Malignant")
    ax.scatter(X_test_2d[y_test == 1, 0], X_test_2d[y_test == 1, 1],
               c="#2563eb", s=20, edgecolors="k", linewidths=0.3, label="Benign")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("PC1")
    if ax == axes[0]:
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, loc="lower left")

fig.suptitle("Decision Boundaries (PCA 2D projection, test points)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "pca_decision_boundaries.png", bbox_inches="tight")
plt.close(fig)
print(f"  Plot saved → {OUT_DIR / 'pca_decision_boundaries.png'}")


# ─── 11. Reproducibility Summary ─────────────────────────────────────────────
print("\n")
print("=" * 65)
print("  REPRODUCIBILITY")
print("=" * 65)
print(f"  SEED={SEED}, TEST_SIZE={TEST_SIZE}, CV={N_FOLDS}-fold StratifiedKFold")
print(f"  Pipeline: StandardScaler → model (no data leakage)")
print(f"  All outputs saved to: {OUT_DIR.resolve()}")
print("=" * 65)