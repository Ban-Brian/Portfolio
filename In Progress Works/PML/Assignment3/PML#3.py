import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_auc_score, classification_report)
import warnings
warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5

# Hyperparameter search grids
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
GAMMA_GRID = [0.001, 0.01, 0.1, 1]

np.random.seed(SEED)

# ─── Data Loading ─────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

print("=" * 65)
print("  DATASET: Breast Cancer Wisconsin (Diagnostic)")
print("=" * 65)
print(f"  Samples : {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes : {target_names} (0=malignant, 1=benign)")
print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print()

# ─── Train / Test Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

print(f"  Train set: {X_train.shape[0]} samples")
print(f"  Test  set: {X_test.shape[0]} samples  (held out, used once)")
print(f"  Split    : {1-TEST_SIZE:.0%} / {TEST_SIZE:.0%}, stratified, seed={SEED}")
print()

# ─── Cross-Validation Setup ──────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print(f"  CV strategy: {N_FOLDS}-fold StratifiedKFold (shuffle=True, seed={SEED})")
print()


# ─── Helper: run GridSearchCV and report results ─────────────────────────────
def tune_model(name, pipeline, param_grid):
    """Run GridSearchCV on the training set and return the fitted searcher."""
    print("=" * 65)
    print(f"  MODEL: {name}")
    print("=" * 65)
    print(f"  Search grid: {param_grid}")

    gs = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="accuracy",
        refit=True, return_train_score=False, n_jobs=-1
    )
    gs.fit(X_train, y_train)

    print(f"  Best CV accuracy : {gs.best_score_:.4f}")
    print(f"  Best params      : {gs.best_params_}")
    print()
    return gs


# ─── Model 1: Logistic Regression ────────────────────────────────────────────
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=10000, random_state=SEED))
])
param_lr = {"clf__C": C_GRID}
gs_lr = tune_model("Logistic Regression", pipe_lr, param_lr)


# ─── Model 2: Linear SVC ─────────────────────────────────────────────────────
# Wrap LinearSVC in CalibratedClassifierCV to get predict_proba for ROC-AUC
pipe_lsvc = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", CalibratedClassifierCV(
        LinearSVC(max_iter=50000, dual="auto", random_state=SEED),
        cv=3
    ))
])
param_lsvc = {"clf__estimator__C": C_GRID}
gs_lsvc = tune_model("Linear SVC", pipe_lsvc, param_lsvc)


# ─── Model 3: RBF SVC ────────────────────────────────────────────────────────
pipe_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True, random_state=SEED))
])
param_rbf = {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__gamma": GAMMA_GRID}
gs_rbf = tune_model("RBF SVC", pipe_rbf, param_rbf)


# ─── Final Evaluation on Held-Out Test Set ────────────────────────────────────
print("=" * 65)
print("  FINAL TEST-SET EVALUATION")
print("=" * 65)

models = {
    "Logistic Regression": gs_lr,
    "Linear SVC":          gs_lsvc,
    "RBF SVC":             gs_rbf,
}

summary_rows = []

for name, gs in models.items():
    y_pred = gs.predict(X_test)
    y_prob = gs.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # Extract best hyperparameters in readable form
    best = gs.best_params_
    if name == "Logistic Regression":
        hp_str = f"C={best['clf__C']}"
    elif name == "Linear SVC":
        hp_str = f"C={best['clf__estimator__C']}"
    else:
        hp_str = f"C={best['clf__C']}, γ={best['clf__gamma']}"

    summary_rows.append({
        "Model": name,
        "Best Params": hp_str,
        "CV Accuracy": f"{gs.best_score_:.4f}",
        "Test Accuracy": f"{acc:.4f}",
        "ROC-AUC": f"{auc:.4f}",
        "TN": cm[0, 0], "FP": cm[0, 1],
        "FN": cm[1, 0], "TP": cm[1, 1],
    })

    print(f"\n  ── {name} ({hp_str}) ──")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

# ─── Summary Table ────────────────────────────────────────────────────────────
print("\n")
print("=" * 65)
print("  SUMMARY TABLE")
print("=" * 65)

summary_df = pd.DataFrame(summary_rows)
display_cols = ["Model", "Best Params", "CV Accuracy", "Test Accuracy", "ROC-AUC"]
print(summary_df[display_cols].to_string(index=False))
print()

# Confusion matrices side-by-side
print("  Confusion Matrices (rows=actual, cols=predicted):")
print(f"  {'':>20}  {'Malignant':>10} {'Benign':>7}")
for row in summary_rows:
    print(f"  {row['Model']:>20}  TN={row['TN']:>2} FP={row['FP']:>2}  |  FN={row['FN']:>2} TP={row['TP']:>2}")
print()

print("  Reproducibility settings:")
print(f"    SEED={SEED}, TEST_SIZE={TEST_SIZE}, CV={N_FOLDS}-fold StratifiedKFold")
print(f"    Pipeline: StandardScaler → model (no leakage)")
