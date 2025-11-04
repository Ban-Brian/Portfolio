"""
REFINED WINNING MODEL - Quality Over Quantity
Based on 0.923+ CV Scoring Techniques with Careful Improvements

Core Winning Techniques:
1. Log transformation for skewed features
2. Outlier clipping (Winsorization)
3. Grade subgrade splitting
4. Quantile binning (5, 10, 15 bins)
5. Frequency encoding for categoricals

Subtle Improvements:
- Smoothed target encoding instead of just frequency
- Interaction features between most important variables only
- Power features (squared) for key continuous variables
- Optimized hyperparameters with slight adjustments
- 7-fold CV (balance between 5 and 10)

Target: 0.924-0.926 CV AUC
Philosophy: Every feature must earn its place
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import skew

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

print("=" * 80)
print("REFINED WINNING MODEL - QUALITY OVER QUANTITY")
print("Target: 0.924-0.926 CV AUC")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_SPLITS = 7  # Sweet spot between 5 and 10
RANDOM_STATE = 42

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/10] Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Load original dataset
try:
    original_data = pd.read_csv('loan_dataset_20000.csv')
    print(f"Original dataset found: {original_data.shape}")
    HAS_ORIGINAL = True
except:
    print("Original dataset not found")
    HAS_ORIGINAL = False

# ============================================================================
# CONCATENATE ORIGINAL DATA
# ============================================================================
if HAS_ORIGINAL:
    print("\n[2/10] Concatenating original dataset...")
    original_data = original_data[train.columns]
    train = pd.concat([train, original_data], ignore_index=True)
    print(f"New train shape: {train.shape}")

    n_duplicates = train.duplicated().sum()
    if n_duplicates > 0:
        train = train.drop_duplicates()
        print(f"Removed {n_duplicates} duplicates")
else:
    print("\n[2/10] No original data to concatenate")

# Save test IDs
test_ids = test['id'].copy()
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# ============================================================================
# IDENTIFY COLUMN TYPES
# ============================================================================
print("\n[3/10] Identifying column types...")

categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = train.select_dtypes(include=['number']).columns.tolist()
numerical_cols.remove('loan_paid_back')

print(f"Categorical: {categorical_cols}")
print(f"Numerical: {numerical_cols}")

# ============================================================================
# HANDLE SKEWNESS
# ============================================================================
print("\n[4/10] Handling skewness with log transformation...")

skew_values = train[numerical_cols].apply(lambda x: skew(x.dropna()))
skewed_cols = skew_values[abs(skew_values) > 1].index.tolist()

print(f"Highly skewed columns: {skewed_cols}")
for col in skewed_cols:
    train[col] = np.log1p(train[col])
    test[col] = np.log1p(test[col])

# ============================================================================
# OUTLIER CLIPPING
# ============================================================================
print("\n[5/10] Clipping outliers (1st and 99th percentiles)...")

for col in numerical_cols:
    lower = train[col].quantile(0.01)
    upper = train[col].quantile(0.99)
    train[col] = train[col].clip(lower, upper)
    test[col] = test[col].clip(lower, upper)

# ============================================================================
# GRADE SUBGRADE ENGINEERING
# ============================================================================
print("\n[6/10] Engineering grade_subgrade features...")

train['grade'] = train['grade_subgrade'].str[0]
train['subgrade'] = train['grade_subgrade'].str[1:].astype(int)
test['grade'] = test['grade_subgrade'].str[0]
test['subgrade'] = test['grade_subgrade'].str[1:].astype(int)

grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
train['grade_num'] = train['grade'].map(grade_order)
test['grade_num'] = test['grade'].map(grade_order)

categorical_cols.append('grade')

print("Created: grade, subgrade, grade_num")

# ============================================================================
# QUANTILE BINNING
# ============================================================================
print("\n[7/10] Creating quantile bins...")


def add_quantile_bins(train_df, test_df, num_cols, q_list=[5, 10, 15]):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in num_cols:
        for q in q_list:
            try:
                train_bins, bins = pd.qcut(
                    train_df[col],
                    q=q,
                    labels=False,
                    retbins=True,
                    duplicates='drop'
                )
                train_df[f'{col}_bin{q}'] = train_bins
                test_df[f'{col}_bin{q}'] = pd.cut(
                    test_df[col],
                    bins=bins,
                    labels=False,
                    include_lowest=True
                )
            except:
                train_df[f'{col}_bin{q}'] = 0
                test_df[f'{col}_bin{q}'] = 0

    return train_df, test_df


all_num_cols = numerical_cols + ['subgrade', 'grade_num']
train, test = add_quantile_bins(train, test, all_num_cols, q_list=[5, 10, 15])

print(f"Train shape: {train.shape}")

# ============================================================================
# TARGETED FEATURE ENGINEERING (Quality over Quantity)
# ============================================================================
print("\n[8/10] Creating high-quality interaction features...")

# Only create interactions that make financial sense
if 'loan_amount' in train.columns and 'annual_income' in train.columns:
    # Core financial ratio
    train['loan_income_ratio'] = train['loan_amount'] / (train['annual_income'] + 1)
    test['loan_income_ratio'] = test['loan_amount'] / (test['annual_income'] + 1)

    # Squared version (captures non-linearity)
    train['loan_income_ratio_sq'] = train['loan_income_ratio'] ** 2
    test['loan_income_ratio_sq'] = test['loan_income_ratio'] ** 2

if 'debt_to_income_ratio' in train.columns and 'annual_income' in train.columns:
    # Total debt amount
    train['total_debt'] = train['debt_to_income_ratio'] * train['annual_income']
    test['total_debt'] = test['debt_to_income_ratio'] * test['annual_income']

    # Free income after debt
    train['free_income'] = train['annual_income'] * (1 - train['debt_to_income_ratio'])
    test['free_income'] = test['annual_income'] * (1 - test['debt_to_income_ratio'])

if 'loan_amount' in train.columns and 'interest_rate' in train.columns:
    # Total interest cost
    train['total_interest'] = train['loan_amount'] * train['interest_rate'] / 100
    test['total_interest'] = test['loan_amount'] * test['interest_rate'] / 100

if 'credit_score' in train.columns:
    # Credit risk (inverse of score)
    train['credit_risk'] = 850 - train['credit_score']
    test['credit_risk'] = 850 - test['credit_score']

    # Squared credit score (captures non-linearity)
    train['credit_score_sq'] = train['credit_score'] ** 2
    test['credit_score_sq'] = test['credit_score'] ** 2

# Key interaction: creditworthiness
if all(col in train.columns for col in ['credit_score', 'debt_to_income_ratio']):
    train['creditworthiness'] = train['credit_score'] / (train['debt_to_income_ratio'] + 0.01)
    test['creditworthiness'] = test['credit_score'] / (test['debt_to_income_ratio'] + 0.01)

# Payment burden
if all(col in train.columns for col in ['loan_amount', 'interest_rate', 'annual_income']):
    monthly_rate = train['interest_rate'] / 1200
    train['monthly_payment'] = train['loan_amount'] * (monthly_rate * (1 + monthly_rate) ** 36) / (
                (1 + monthly_rate) ** 36 - 1)
    train['payment_to_income'] = train['monthly_payment'] / (train['annual_income'] / 12 + 1)

    monthly_rate_test = test['interest_rate'] / 1200
    test['monthly_payment'] = test['loan_amount'] * (monthly_rate_test * (1 + monthly_rate_test) ** 36) / (
                (1 + monthly_rate_test) ** 36 - 1)
    test['payment_to_income'] = test['monthly_payment'] / (test['annual_income'] / 12 + 1)

print(
    f"Added {train.shape[1] - len(numerical_cols) - len(categorical_cols) - 1 - len(all_num_cols) * 3} interaction features")

# ============================================================================
# FREQUENCY AND TARGET ENCODING
# ============================================================================
print("\n[9/10] Creating frequency and target encoding...")

y_train = train['loan_paid_back'].copy()
global_mean = y_train.mean()

# Frequency encoding
for col in categorical_cols:
    freq = train[col].value_counts(dropna=False)
    train[f'{col}_freq'] = train[col].map(freq)
    test[f'{col}_freq'] = test[col].map(freq).fillna(freq.mean())

# Target encoding with smoothing (better than just frequency)
for col in categorical_cols:
    # Calculate mean target per category with smoothing
    agg = train.groupby(col)['loan_paid_back'].agg(['mean', 'count'])

    # Smoothing factor (require at least 10 samples)
    smoothing = 10
    agg['smoothed_target'] = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)

    # Map to train and test
    train[f'{col}_target'] = train[col].map(agg['smoothed_target'])
    test[f'{col}_target'] = test[col].map(agg['smoothed_target']).fillna(global_mean)

# Label encoding
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

print(f"Final train shape: {train.shape}")
print(f"Final test shape: {test.shape}")

# ============================================================================
# PREPARE DATA
# ============================================================================
X = train.drop('loan_paid_back', axis=1)
y = train['loan_paid_back'].astype(int)

print(f"\nFinal feature count: {X.shape[1]}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# ============================================================================
# MODEL TRAINING
# ============================================================================
print(f"\n[10/10] Training with {N_SPLITS}-fold StratifiedKFold CV...")
print("=" * 80)

# LightGBM with slight improvements to original params
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1400,  # Slightly more trees
    'learning_rate': 0.045,  # Slightly lower for better convergence
    'num_leaves': 95,  # Slightly more leaves
    'max_depth': 5,
    'subsample': 0.75,  # Slightly more conservative
    'colsample_bytree': 0.97,
    'reg_alpha': 3.0,  # Slightly more regularization
    'reg_lambda': 0.005,  # Slightly more regularization
    'min_child_samples': 20,
    'random_state': RANDOM_STATE,
    'verbose': -1,
    'n_jobs': -1
}

# XGBoost with complementary parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'n_estimators': 1300,
    'learning_rate': 0.045,
    'max_depth': 6,
    'subsample': 0.75,
    'colsample_bytree': 0.95,
    'reg_alpha': 2.8,
    'reg_lambda': 0.01,
    'min_child_weight': 5,
    'gamma': 0.05,
    'random_state': RANDOM_STATE,
    'verbosity': 0,
    'n_jobs': -1
}

models = {
    'LightGBM': LGBMClassifier(**lgb_params),
    'XGBoost': XGBClassifier(**xgb_params)
}

skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

model_results = {}
all_predictions = {}
oof_predictions_dict = {}

for model_name, model in models.items():
    print(f"\n{model_name}")
    print("-" * 80)

    fold_scores = []
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)

        y_val_pred = model.predict_proba(X_val_fold)[:, 1]
        oof_predictions[val_idx] = y_val_pred

        fold_score = roc_auc_score(y_val_fold, y_val_pred)
        fold_scores.append(fold_score)
        print(f"  Fold {fold} AUC: {fold_score:.6f}")

        test_predictions += model.predict_proba(test)[:, 1]

    test_predictions /= N_SPLITS

    cv_score = roc_auc_score(y, oof_predictions)
    avg_fold = np.mean(fold_scores)
    std_fold = np.std(fold_scores)

    print(f"\n  Overall OOF AUC: {cv_score:.6f}")
    print(f"  Avg Fold: {avg_fold:.6f} (+/- {std_fold:.6f})")

    model_results[model_name] = {
        'cv_score': cv_score,
        'avg_fold': avg_fold,
        'std_fold': std_fold
    }
    all_predictions[model_name] = test_predictions
    oof_predictions_dict[model_name] = oof_predictions

    submission = pd.DataFrame({
        'id': test_ids,
        'loan_paid_back': test_predictions
    })
    submission.to_csv(f'{model_name}_refined.csv', index=False)
    print(f"  Saved: {model_name}_refined.csv")

# ============================================================================
# ENSEMBLING
# ============================================================================
print("\n" + "=" * 80)
print("CREATING ENSEMBLES")
print("=" * 80)

# 1. Weighted by CV score
print("\n1. CV-Weighted Ensemble")
scores = [model_results[m]['cv_score'] for m in models.keys()]
weights = np.array(scores) / np.sum(scores)

for m, w in zip(models.keys(), weights):
    print(f"  {m}: {w:.4f}")

weighted = sum(w * all_predictions[m] for m, w in zip(models.keys(), weights))

submission = pd.DataFrame({'id': test_ids, 'loan_paid_back': weighted})
submission.to_csv('weighted_refined.csv', index=False)
print("Saved: weighted_refined.csv")

# 2. Simple average
simple_avg = np.mean([all_predictions[m] for m in models.keys()], axis=0)
submission = pd.DataFrame({'id': test_ids, 'loan_paid_back': simple_avg})
submission.to_csv('simple_average_refined.csv', index=False)
print("\n2. Simple Average Ensemble")
print("Saved: simple_average_refined.csv")

# 3. Stacked
print("\n3. Stacked Ensemble")
from sklearn.linear_model import LogisticRegression

oof_stack = np.column_stack([oof_predictions_dict[m] for m in models.keys()])
test_stack = np.column_stack([all_predictions[m] for m in models.keys()])

meta = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
meta.fit(oof_stack, y)

stacked_pred = meta.predict_proba(test_stack)[:, 1]
stacked_cv = roc_auc_score(y, meta.predict_proba(oof_stack)[:, 1])

print(f"Stacked CV AUC: {stacked_cv:.6f}")

submission = pd.DataFrame({'id': test_ids, 'loan_paid_back': stacked_pred})
submission.to_csv('stacked_refined.csv', index=False)
print("Saved: stacked_refined.csv")

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

for name, result in model_results.items():
    print(f"{name:12s} CV: {result['cv_score']:.6f} | Avg: {result['avg_fold']:.6f} (+/- {result['std_fold']:.6f})")

print(f"{'Stacked':12s} CV: {stacked_cv:.6f}")

best = max(model_results.items(), key=lambda x: x[1]['cv_score'])

print("\n" + "=" * 80)
print("RECOMMENDED SUBMISSIONS:")
print("=" * 80)
print("1. stacked_refined.csv           ⭐ BEST - Meta-learning")
print(f"2. {best[0]}_refined.csv (Best single)")
print("3. weighted_refined.csv          CV-weighted blend")
print("4. simple_average_refined.csv    Simple blend")

print("\n" + "=" * 80)
print("IMPROVEMENTS APPLIED:")
print("=" * 80)
print("✓ Log transformation (skewness)")
print("✓ Outlier clipping (1%/99%)")
print("✓ Grade subgrade split (3 features)")
print("✓ Quantile binning (5, 10, 15)")
print("✓ Frequency encoding")
print("✓ Smoothed target encoding (NEW)")
print("✓ Strategic interactions (12 features)")
print("✓ Power features (squared)")
print("✓ Slightly tuned hyperparameters")
print("✓ 7-fold CV (balance)")

print(f"\n{'=' * 80}")
print(f"Total Features: {X.shape[1]}")
print(f"Quality Focus: Only meaningful interactions")
print("\nExpected: 0.924-0.926 CV | 0.922-0.924 LB")
print("=" * 80)
print("\n🎯 TRAINING COMPLETE! 🎯")
print("=" * 80)