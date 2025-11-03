"""
Improved Model for Predicting Loan Payback
Playground Series - Season 5, Episode 11

Improvements:
- Better hyperparameters
- Simple feature engineering
- LightGBM + XGBoost ensemble
- Target encoding for high-cardinality categoricals
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Separate features and target
X = train.drop(['id', 'loan_paid_back'], axis=1)
y = train['loan_paid_back']
X_test = test.drop(['id'], axis=1)

print(f"Train shape: {X.shape}")
print(f"Test shape: {X_test.shape}")

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")


# Simple feature engineering
def create_features(df):
    df = df.copy()

    # Interaction features for numerical columns
    if len(numerical_cols) >= 2:
        # Sum and mean of all numerical features
        df['num_sum'] = df[numerical_cols].sum(axis=1)
        df['num_mean'] = df[numerical_cols].mean(axis=1)
        df['num_std'] = df[numerical_cols].std(axis=1)
        df['num_max'] = df[numerical_cols].max(axis=1)
        df['num_min'] = df[numerical_cols].min(axis=1)

    return df


print("\nCreating features...")
X = create_features(X)
X_test = create_features(X_test)

# Handle categorical variables with label encoding
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    le_dict[col] = le

print(f"Final train shape: {X.shape}")

# Parameters for LightGBM (tuned for better performance)
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': 42
}

# Parameters for XGBoost
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'max_depth': 7,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42
}

# Cross-validation setup
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

cv_scores_lgb = []
cv_scores_xgb = []

print("\nTraining models with cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n{'=' * 50}")
    print(f"Fold {fold}/{n_folds}")
    print(f"{'=' * 50}")

    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Train LightGBM
    print("\nTraining LightGBM...")
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold)

    model_lgb = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(200)]
    )

    oof_lgb[val_idx] = model_lgb.predict(X_val_fold)
    test_lgb += model_lgb.predict(X_test) / n_folds

    score_lgb = roc_auc_score(y_val_fold, oof_lgb[val_idx])
    cv_scores_lgb.append(score_lgb)
    print(f"LightGBM Fold {fold} AUC: {score_lgb:.6f}")

    # Train XGBoost
    print("\nTraining XGBoost...")
    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
    dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
    dtest = xgb.DMatrix(X_test)

    model_xgb = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, 'eval')],
        early_stopping_rounds=100,
        verbose_eval=200
    )

    oof_xgb[val_idx] = model_xgb.predict(dval)
    test_xgb += model_xgb.predict(dtest) / n_folds

    score_xgb = roc_auc_score(y_val_fold, oof_xgb[val_idx])
    cv_scores_xgb.append(score_xgb)
    print(f"XGBoost Fold {fold} AUC: {score_xgb:.6f}")

# Calculate overall scores
cv_lgb = roc_auc_score(y, oof_lgb)
cv_xgb = roc_auc_score(y, oof_xgb)

print(f"\n{'=' * 60}")
print("RESULTS")
print(f"{'=' * 60}")
print(f"LightGBM CV AUC: {cv_lgb:.6f} (+/- {np.std(cv_scores_lgb):.6f})")
print(f"XGBoost CV AUC:  {cv_xgb:.6f} (+/- {np.std(cv_scores_xgb):.6f})")

# Ensemble predictions (weighted average)
# Find optimal weights for ensemble
best_score = 0
best_weight = 0.5

for weight in np.linspace(0, 1, 21):
    oof_blend = weight * oof_lgb + (1 - weight) * oof_xgb
    score = roc_auc_score(y, oof_blend)
    if score > best_score:
        best_score = score
        best_weight = weight

print(f"\nBest ensemble weight (LGB): {best_weight:.2f}")
print(f"Ensemble CV AUC: {best_score:.6f}")
print(f"{'=' * 60}")

# Create final predictions
test_predictions = best_weight * test_lgb + (1 - best_weight) * test_xgb

# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
print(f"Predictions range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")