import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, rankdata
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

print("=" * 80)
print("ENHANCED MODEL - HIERARCHICAL BLENDING APPROACH")
print("Target: 0.927+ Public LB")
print("=" * 80)

N_SPLITS = 10
RANDOM_STATE = 42

# ========== DATA LOADING ==========
print("\n[1/12] Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Try to load original dataset for data augmentation
try:
    original_data = pd.read_csv('loan_dataset_20000.csv')
    print(f"Original dataset found: {original_data.shape}")
    original_data = original_data[train.columns]
    train = pd.concat([train, original_data], ignore_index=True)
    print(f"Combined train shape: {train.shape}")

    n_dup = train.duplicated().sum()
    if n_dup > 0:
        train = train.drop_duplicates()
        print(f"Removed {n_dup} duplicates")
except:
    print("Original dataset not found - proceeding without augmentation")

test_ids = test['id'].copy()
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# ========== PREPROCESSING ==========
print("\n[2/12] Basic preprocessing...")
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = train.select_dtypes(include=['number']).columns.tolist()
numerical_cols.remove('loan_paid_back')

print(f"Categorical: {len(categorical_cols)}")
print(f"Numerical: {len(numerical_cols)}")

print("\n[3/12] Log transformation...")
skew_values = train[numerical_cols].apply(lambda x: skew(x.dropna()))
skewed_cols = skew_values[abs(skew_values) > 1].index.tolist()

for col in skewed_cols:
    train[col] = np.log1p(train[col])
    test[col] = np.log1p(test[col])
print(f"Transformed: {skewed_cols}")

print("\n[4/12] Advanced outlier handling...")
for col in numerical_cols:
    # More sophisticated outlier handling with IQR method
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Also consider percentile clipping
    lower_pct = train[col].quantile(0.01)
    upper_pct = train[col].quantile(0.99)

    # Use the less aggressive clipping
    lower = max(lower, lower_pct)
    upper = min(upper, upper_pct)

    train[col] = train[col].clip(lower, upper)
    test[col] = test[col].clip(lower, upper)

# ========== FEATURE ENGINEERING ==========
print("\n[5/12] Enhanced grade features...")
train['grade'] = train['grade_subgrade'].str[0]
train['subgrade'] = train['grade_subgrade'].str[1:].astype(int)
test['grade'] = test['grade_subgrade'].str[0]
test['subgrade'] = test['grade_subgrade'].str[1:].astype(int)

grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
train['grade_num'] = train['grade'].map(grade_order)
test['grade_num'] = test['grade'].map(grade_order)

# Additional grade features
train['grade_subgrade_num'] = train['grade_num'] * 10 + train['subgrade']
test['grade_subgrade_num'] = test['grade_num'] * 10 + test['subgrade']

categorical_cols.append('grade')

print("\n[6/12] Advanced quantile binning...")


def add_advanced_bins(train_df, test_df, cols, q_lists):
    """Enhanced binning with multiple quantile strategies"""
    for col in cols:
        for q in q_lists:
            try:
                train_bins, bins = pd.qcut(train_df[col], q=q, labels=False,
                                           retbins=True, duplicates='drop')
                train_df[f'{col}_q{q}'] = train_bins
                test_df[f'{col}_q{q}'] = pd.cut(test_df[col], bins=bins,
                                                labels=False, include_lowest=True)

                # Add bin counts as features
                bin_counts = train_df[f'{col}_q{q}'].value_counts()
                train_df[f'{col}_q{q}_count'] = train_df[f'{col}_q{q}'].map(bin_counts)
                test_df[f'{col}_q{q}_count'] = test_df[f'{col}_q{q}'].map(bin_counts).fillna(0)

            except:
                train_df[f'{col}_q{q}'] = 0
                test_df[f'{col}_q{q}'] = 0
                train_df[f'{col}_q{q}_count'] = 0
                test_df[f'{col}_q{q}_count'] = 0
    return train_df, test_df


num_cols_for_bins = numerical_cols + ['subgrade', 'grade_num', 'grade_subgrade_num']
train, test = add_advanced_bins(train, test, num_cols_for_bins, q_lists=[5, 10, 15, 20])

print("\n[7/12] Enhanced interaction features...")
# Basic ratios
train['loan_to_income'] = train['loan_amount'] / (train['annual_income'] + 1)
test['loan_to_income'] = test['loan_amount'] / (test['annual_income'] + 1)

train['total_debt'] = train['debt_to_income_ratio'] * train['annual_income']
test['total_debt'] = test['debt_to_income_ratio'] * test['annual_income']

train['credit_risk'] = 850 - train['credit_score']
test['credit_risk'] = 850 - test['credit_score']

train['total_interest'] = train['loan_amount'] * train['interest_rate'] / 100
test['total_interest'] = test['loan_amount'] * test['interest_rate'] / 100

# Advanced interactions
train['payment_burden'] = train['total_interest'] / (train['annual_income'] + 1)
test['payment_burden'] = test['total_interest'] / (test['annual_income'] + 1)

train['credit_utilization'] = train['total_debt'] / (train['annual_income'] * 2 + 1)
test['credit_utilization'] = test['total_debt'] / (test['annual_income'] * 2 + 1)

train['risk_score'] = (train['credit_risk'] * train['debt_to_income_ratio'] *
                       train['interest_rate']) / 1000
test['risk_score'] = (test['credit_risk'] * test['debt_to_income_ratio'] *
                      test['interest_rate']) / 1000

# Polynomial features for key variables
for col in ['loan_amount', 'annual_income', 'credit_score']:
    train[f'{col}_squared'] = train[col] ** 2
    test[f'{col}_squared'] = test[col] ** 2
    train[f'{col}_sqrt'] = np.sqrt(np.abs(train[col]))
    test[f'{col}_sqrt'] = np.sqrt(np.abs(test[col]))

print("\n[8/12] Advanced encoding strategies...")
y_full = train['loan_paid_back'].copy()
global_mean = y_full.mean()

# Frequency encoding
for col in categorical_cols:
    freq = train[col].value_counts(dropna=False)
    train[f'{col}_freq'] = train[col].map(freq)
    test[f'{col}_freq'] = test[col].map(freq).fillna(freq.mean())

# Enhanced target encoding with multiple smoothing parameters
smoothing_params = [5, 10, 20]
for col in categorical_cols:
    for smoothing in smoothing_params:
        agg = train.groupby(col)['loan_paid_back'].agg(['mean', 'count'])
        agg[f'target_s{smoothing}'] = (agg['mean'] * agg['count'] +
                                       global_mean * smoothing) / (agg['count'] + smoothing)

        train[f'{col}_target_s{smoothing}'] = train[col].map(agg[f'target_s{smoothing}'])
        test[f'{col}_target_s{smoothing}'] = test[col].map(agg[f'target_s{smoothing}']).fillna(global_mean)

# Label encoding
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

X = train.drop('loan_paid_back', axis=1)
y = train['loan_paid_back'].astype(int)

print(f"\nTotal features: {X.shape[1]}")

# ========== MODEL CONFIGURATIONS ==========
print(f"\n[9/12] Preparing optimized model configurations...")
print("=" * 80)

# Enhanced model configurations based on leaderboard insights
models_config = {
    # LightGBM variants
    'LGB_1': {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 1500,
        'learning_rate': 0.04,
        'num_leaves': 95,
        'max_depth': 7,
        'subsample': 0.75,
        'colsample_bytree': 0.95,
        'reg_alpha': 3.0,
        'reg_lambda': 0.01,
        'min_child_samples': 20,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    },
    'LGB_2': {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 1400,
        'learning_rate': 0.045,
        'num_leaves': 85,
        'max_depth': 6,
        'subsample': 0.77,
        'colsample_bytree': 0.93,
        'reg_alpha': 2.7,
        'reg_lambda': 0.015,
        'min_child_samples': 25,
        'random_state': 123,
        'verbose': -1,
        'n_jobs': -1
    },
    'LGB_3': {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 1300,
        'learning_rate': 0.05,
        'num_leaves': 75,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'reg_alpha': 2.3,
        'reg_lambda': 0.02,
        'min_child_samples': 30,
        'random_state': 456,
        'verbose': -1,
        'n_jobs': -1
    },
    'LGB_4': {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 1600,
        'learning_rate': 0.035,
        'num_leaves': 100,
        'max_depth': 8,
        'subsample': 0.73,
        'colsample_bytree': 0.92,
        'reg_alpha': 3.5,
        'reg_lambda': 0.008,
        'min_child_samples': 15,
        'random_state': 789,
        'verbose': -1,
        'n_jobs': -1
    },

    # XGBoost variants
    'XGB_1': {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 1400,
        'learning_rate': 0.04,
        'max_depth': 8,
        'subsample': 0.75,
        'colsample_bytree': 0.93,
        'reg_alpha': 2.8,
        'reg_lambda': 0.02,
        'min_child_weight': 3,
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': -1
    },
    'XGB_2': {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 1300,
        'learning_rate': 0.045,
        'max_depth': 7,
        'subsample': 0.77,
        'colsample_bytree': 0.9,
        'reg_alpha': 2.5,
        'reg_lambda': 0.025,
        'min_child_weight': 4,
        'random_state': 123,
        'verbosity': 0,
        'n_jobs': -1
    },
    'XGB_3': {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 1500,
        'learning_rate': 0.038,
        'max_depth': 9,
        'subsample': 0.73,
        'colsample_bytree': 0.91,
        'reg_alpha': 3.2,
        'reg_lambda': 0.018,
        'min_child_weight': 2,
        'random_state': 456,
        'verbosity': 0,
        'n_jobs': -1
    },

    # CatBoost variants
    'CAT_1': {
        'iterations': 1500,
        'depth': 5,
        'learning_rate': 0.09,
        'l2_leaf_reg': 12.0,
        'border_count': 240,
        'random_strength': 4.8,
        'bagging_temperature': 0.25,
        'random_state': 42,
        'eval_metric': 'AUC',
        'verbose': 0
    },
    'CAT_2': {
        'iterations': 1400,
        'depth': 4,
        'learning_rate': 0.095,
        'l2_leaf_reg': 11.5,
        'border_count': 235,
        'random_strength': 4.5,
        'bagging_temperature': 0.24,
        'random_state': 123,
        'eval_metric': 'AUC',
        'verbose': 0
    },
    'CAT_3': {
        'iterations': 1600,
        'depth': 6,
        'learning_rate': 0.085,
        'l2_leaf_reg': 13.0,
        'border_count': 250,
        'random_strength': 5.0,
        'bagging_temperature': 0.23,
        'random_state': 456,
        'eval_metric': 'AUC',
        'verbose': 0
    }
}

# Initialize models
models = {
    name: (LGBMClassifier(**params) if 'LGB' in name else
           XGBClassifier(**params) if 'XGB' in name else
           CatBoostClassifier(**params))
    for name, params in models_config.items()
}

# ========== TRAINING ==========
print(f"\n[10/12] Training {N_SPLITS}-fold CV with {len(models)} models...")

skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

results = {}
predictions = {}
oof_preds = {}

for name, model in models.items():
    print(f"\n{name}")
    print("-" * 80)

    scores = []
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred

        score = roc_auc_score(y_val, val_pred)
        scores.append(score)

        test_pred += model.predict_proba(test)[:, 1]

    test_pred /= N_SPLITS

    cv = roc_auc_score(y, oof)
    avg = np.mean(scores)
    std = np.std(scores)

    print(f"CV: {cv:.6f} | Avg: {avg:.6f} | Std: {std:.6f}")

    results[name] = cv
    predictions[name] = test_pred
    oof_preds[name] = oof

# ========== HIERARCHICAL BLENDING ==========
print("\n[11/12] Hierarchical blending strategy...")
print("=" * 80)


def hierarchical_blend(predictions, weights, sort_ratio=(0.3, 0.7), sub_weights=None):
    """
    Advanced hierarchical blending with sorting strategy
    """
    if sub_weights is None:
        sub_weights = [0.07, -0.01, -0.02, -0.04]

    n_samples = len(list(predictions.values())[0])
    blend = np.zeros(n_samples)

    # Get model names and their predictions
    model_names = list(predictions.keys())
    preds_array = np.array([predictions[m] for m in model_names])

    # For each sample, sort models by their prediction value
    for i in range(n_samples):
        sample_preds = [(model_names[j], preds_array[j, i]) for j in range(len(model_names))]
        sample_preds_sorted = sorted(sample_preds, key=lambda x: x[1], reverse=True)

        # Apply hierarchical weights based on sort position
        weighted_sum = 0
        total_weight = 0

        for idx, (model_name, pred_value) in enumerate(sample_preds_sorted):
            base_weight = weights[model_name]

            # Apply sub-weights based on position
            if idx < len(sub_weights):
                position_weight = base_weight + sub_weights[idx]
            else:
                position_weight = base_weight

            # Apply sort ratio
            if idx < len(model_names) * sort_ratio[0]:
                position_weight *= 1.1  # Boost top predictions
            elif idx > len(model_names) * sort_ratio[1]:
                position_weight *= 0.9  # Reduce bottom predictions

            weighted_sum += pred_value * max(position_weight, 0)
            total_weight += max(position_weight, 0)

        if total_weight > 0:
            blend[i] = weighted_sum / total_weight
        else:
            blend[i] = np.mean([pred_value for _, pred_value in sample_preds])

    return blend


# Display results
df_results = pd.DataFrame({
    'Model': list(results.keys()),
    'CV': list(results.values())
}).sort_values('CV', ascending=False)

print("\nModel Performance Summary:")
print(df_results.to_string(index=False))

# Group models by performance tiers
top_tier = df_results.head(4)['Model'].tolist()
mid_tier = df_results.iloc[4:7]['Model'].tolist() if len(df_results) > 4 else []
all_models = df_results['Model'].tolist()

print("\n[12/12] Creating ensemble submissions...")
print("=" * 80)

# 1. Simple average blend
simple = np.mean([predictions[m] for m in all_models], axis=0)
pd.DataFrame({'id': test_ids, 'loan_paid_back': simple}).to_csv('submission_simple.csv', index=False)
print("✓ Simple average blend saved")

# 2. CV-weighted blend
cvs = [results[m] for m in all_models]
weights_cv = np.array(cvs) / np.sum(cvs)
weighted = sum(w * predictions[m] for m, w in zip(all_models, weights_cv))
pd.DataFrame({'id': test_ids, 'loan_paid_back': weighted}).to_csv('submission_weighted.csv', index=False)
print("✓ CV-weighted blend saved")

# 3. Top-tier blend
top_pred = np.mean([predictions[m] for m in top_tier], axis=0)
pd.DataFrame({'id': test_ids, 'loan_paid_back': top_pred}).to_csv('submission_top_tier.csv', index=False)
print("✓ Top-tier blend saved")

# 4. Rank average blend
rank = np.zeros(len(test))
for m in all_models:
    rank += rankdata(predictions[m]) / len(predictions[m])
rank /= len(all_models)
pd.DataFrame({'id': test_ids, 'loan_paid_back': rank}).to_csv('submission_rank.csv', index=False)
print("✓ Rank average blend saved")

# 5. Stacked meta-learner
oof_stack = np.column_stack([oof_preds[m] for m in all_models])
test_stack = np.column_stack([predictions[m] for m in all_models])

meta = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, C=0.5)
meta.fit(oof_stack, y)
stacked = meta.predict_proba(test_stack)[:, 1]
stacked_cv = roc_auc_score(y, meta.predict_proba(oof_stack)[:, 1])
print(f"✓ Stacked blend (CV: {stacked_cv:.6f}) saved")
pd.DataFrame({'id': test_ids, 'loan_paid_back': stacked}).to_csv('submission_stacked.csv', index=False)

# 6. Hierarchical blend (competition-style)
weights_dict = {m: 1 / len(all_models) for m in all_models}
hierarchical = hierarchical_blend(predictions, weights_dict, sort_ratio=(0.3, 0.7))
pd.DataFrame({'id': test_ids, 'loan_paid_back': hierarchical}).to_csv('submission_hierarchical.csv', index=False)
print("✓ Hierarchical blend saved")

# 7. Optimized blend (based on leaderboard insights)
# Using weights similar to version 19/20 from the notebook
if len(all_models) >= 4:
    optimized_weights = {
        all_models[0]: 0.32,  # Best model
        all_models[1]: 0.28,  # Second best
        all_models[2]: 0.28,  # Third
        all_models[3]: 0.12  # Fourth
    }
    for m in all_models[4:]:
        optimized_weights[m] = 0  # Exclude lower performers

    optimized = hierarchical_blend(predictions, optimized_weights,
                                   sort_ratio=(0.3, 0.7),
                                   sub_weights=[0.07, -0.01, -0.02, -0.04])
    pd.DataFrame({'id': test_ids, 'loan_paid_back': optimized}).to_csv('submission_optimized.csv', index=False)
    print("✓ Optimized hierarchical blend saved")

# ========== FINAL SUMMARY ==========
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
print(f"Total features: {X.shape[1]}")
print(f"Models trained: {len(models)}")
print(f"Best single model: {df_results.iloc[0]['Model']} (CV: {df_results.iloc[0]['CV']:.6f})")
print(f"Stacked ensemble CV: {stacked_cv:.6f}")
print("\nGenerated submissions:")
print("1. submission_simple.csv - Simple average")
print("2. submission_weighted.csv - CV-weighted")
print("3. submission_top_tier.csv - Top 4 models")
print("4. submission_rank.csv - Rank average")
print("5. submission_stacked.csv - Meta-learner stacking")
print("6. submission_hierarchical.csv - Hierarchical blend")
print("7. submission_optimized.csv - Competition-optimized blend")
print("\nTarget: 0.927+ Public LB")
print("=" * 80)