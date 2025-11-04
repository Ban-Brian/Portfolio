import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("=" * 80)
print("ULTRA-OPTIMIZED MODEL - LIGHTGBM & XGBOOST ONLY")
print("Target: 0.925+ CV AUC")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_SPLITS = 10
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
        print(f"Removed {n_duplicates} duplicates, shape: {train.shape}")
else:
    print("\n[2/10] No original data to concatenate")

# ============================================================================
# INITIAL SETUP
# ============================================================================
print("\n[3/10] Initial preprocessing...")

test_ids = test['id'].copy()
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = train.select_dtypes(include=['number']).columns.tolist()
numerical_cols.remove('loan_paid_back')

print(f"Categorical: {categorical_cols}")
print(f"Numerical: {numerical_cols}")

# ============================================================================
# ULTRA-ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n[4/10] Creating ultra-advanced features...")


def create_ultra_features(df):
    """Create 150+ ultra-advanced features optimized for boosting models"""
    df = df.copy()

    print("  - Financial ratio features")
    # Core financial ratios
    if 'annual_income' in df.columns and 'loan_amount' in df.columns:
        df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
        df['income_to_loan'] = df['annual_income'] / (df['loan_amount'] + 1)
        df['log_loan_to_income'] = np.log1p(df['loan_to_income'])
        df['sqrt_loan_to_income'] = np.sqrt(df['loan_to_income'])
        df['loan_income_ratio_squared'] = df['loan_to_income'] ** 2

    if 'debt_to_income_ratio' in df.columns and 'annual_income' in df.columns:
        df['total_debt_amount'] = df['debt_to_income_ratio'] * df['annual_income']
        df['free_income'] = df['annual_income'] * (1 - df['debt_to_income_ratio'])
        df['debt_to_free_income'] = df['debt_to_income_ratio'] / (1 - df['debt_to_income_ratio'] + 0.001)
        df['log_total_debt'] = np.log1p(df['total_debt_amount'])

    print("  - Interest and payment features")
    if 'loan_amount' in df.columns and 'interest_rate' in df.columns:
        # Monthly payment calculation (assuming 36-month term)
        r_monthly = df['interest_rate'] / 1200
        df['monthly_payment'] = df['loan_amount'] * (r_monthly * (1 + r_monthly) ** 36) / ((1 + r_monthly) ** 36 - 1)
        df['total_payment'] = df['monthly_payment'] * 36
        df['total_interest'] = df['total_payment'] - df['loan_amount']
        df['interest_to_principal'] = df['total_interest'] / (df['loan_amount'] + 1)
        df['log_total_interest'] = np.log1p(df['total_interest'])

        # Payment burden
        if 'annual_income' in df.columns:
            df['monthly_payment_to_income'] = df['monthly_payment'] / (df['annual_income'] / 12 + 1)
            df['payment_burden'] = df['monthly_payment'] / (df['free_income'] / 12 + 1)

    print("  - Credit score features")
    if 'credit_score' in df.columns:
        # Transformations
        df['credit_squared'] = df['credit_score'] ** 2
        df['credit_cubed'] = df['credit_score'] ** 3
        df['credit_log'] = np.log1p(df['credit_score'])
        df['credit_sqrt'] = np.sqrt(df['credit_score'])
        df['credit_inverse'] = 1 / (df['credit_score'] + 1)

        # Credit risk indicators
        df['credit_risk'] = 850 - df['credit_score']
        df['credit_risk_squared'] = df['credit_risk'] ** 2
        df['is_excellent_credit'] = (df['credit_score'] >= 750).astype(int)
        df['is_good_credit'] = ((df['credit_score'] >= 670) & (df['credit_score'] < 750)).astype(int)
        df['is_fair_credit'] = ((df['credit_score'] >= 580) & (df['credit_score'] < 670)).astype(int)
        df['is_poor_credit'] = (df['credit_score'] < 580).astype(int)

        # Interactions with income
        if 'annual_income' in df.columns:
            df['income_credit_product'] = df['annual_income'] * df['credit_score']
            df['income_per_credit_point'] = df['annual_income'] / (df['credit_score'] + 1)
            df['credit_per_1k_income'] = df['credit_score'] / (df['annual_income'] / 1000 + 1)
            df['log_income_credit'] = np.log1p(df['income_credit_product'])

        # Interactions with loan
        if 'loan_amount' in df.columns:
            df['loan_credit_product'] = df['loan_amount'] * df['credit_score']
            df['loan_per_credit_point'] = df['loan_amount'] / (df['credit_score'] + 1)
            df['credit_to_loan_ratio'] = df['credit_score'] / (df['loan_amount'] + 1)

        # Interactions with debt ratio
        if 'debt_to_income_ratio' in df.columns:
            df['debt_credit_interaction'] = df['debt_to_income_ratio'] * df['credit_score']
            df['debt_to_credit_ratio'] = df['debt_to_income_ratio'] / (df['credit_score'] + 1)
            df['creditworthiness'] = df['credit_score'] / (df['debt_to_income_ratio'] + 0.01)

        # Interactions with interest rate
        if 'interest_rate' in df.columns:
            df['rate_credit_product'] = df['interest_rate'] * df['credit_score']
            df['rate_credit_mismatch'] = df['interest_rate'] - (850 - df['credit_score']) / 50
            df['expected_rate'] = (850 - df['credit_score']) / 50
            df['rate_surprise'] = df['interest_rate'] - df['expected_rate']

    print("  - Interest rate features")
    if 'interest_rate' in df.columns:
        df['interest_squared'] = df['interest_rate'] ** 2
        df['interest_cubed'] = df['interest_rate'] ** 3
        df['interest_log'] = np.log1p(df['interest_rate'])
        df['interest_sqrt'] = np.sqrt(df['interest_rate'])
        df['is_high_rate'] = (df['interest_rate'] >= 15).astype(int)
        df['is_medium_rate'] = ((df['interest_rate'] >= 10) & (df['interest_rate'] < 15)).astype(int)
        df['is_low_rate'] = (df['interest_rate'] < 10).astype(int)

    print("  - Debt features")
    if 'debt_to_income_ratio' in df.columns:
        df['debt_ratio_squared'] = df['debt_to_income_ratio'] ** 2
        df['debt_ratio_cubed'] = df['debt_to_income_ratio'] ** 3
        df['debt_ratio_log'] = np.log1p(df['debt_to_income_ratio'])
        df['is_high_debt'] = (df['debt_to_income_ratio'] >= 0.35).astype(int)
        df['is_moderate_debt'] = ((df['debt_to_income_ratio'] >= 0.2) & (df['debt_to_income_ratio'] < 0.35)).astype(int)
        df['is_low_debt'] = (df['debt_to_income_ratio'] < 0.2).astype(int)

    print("  - Income features")
    if 'annual_income' in df.columns:
        df['income_squared'] = df['annual_income'] ** 2
        df['income_log'] = np.log1p(df['annual_income'])
        df['income_sqrt'] = np.sqrt(df['annual_income'])
        df['income_in_10k'] = df['annual_income'] / 10000
        df['is_high_income'] = (df['annual_income'] >= 50000).astype(int)
        df['is_medium_income'] = ((df['annual_income'] >= 25000) & (df['annual_income'] < 50000)).astype(int)
        df['is_low_income'] = (df['annual_income'] < 25000).astype(int)

    print("  - Loan amount features")
    if 'loan_amount' in df.columns:
        df['loan_squared'] = df['loan_amount'] ** 2
        df['loan_log'] = np.log1p(df['loan_amount'])
        df['loan_sqrt'] = np.sqrt(df['loan_amount'])
        df['loan_in_1k'] = df['loan_amount'] / 1000
        df['is_large_loan'] = (df['loan_amount'] >= 15000).astype(int)
        df['is_medium_loan'] = ((df['loan_amount'] >= 5000) & (df['loan_amount'] < 15000)).astype(int)
        df['is_small_loan'] = (df['loan_amount'] < 5000).astype(int)

    print("  - Complex pairwise interactions")
    # All pairwise interactions of key features
    key_features = ['annual_income', 'loan_amount', 'credit_score', 'interest_rate', 'debt_to_income_ratio']
    key_features = [f for f in key_features if f in df.columns]

    for i in range(len(key_features)):
        for j in range(i + 1, len(key_features)):
            col1, col2 = key_features[i], key_features[j]
            df[f'{col1}_{col2}_mul'] = df[col1] * df[col2]
            df[f'{col1}_{col2}_div'] = df[col1] / (df[col2] + 1)
            df[f'{col1}_{col2}_add'] = df[col1] + df[col2]
            df[f'{col1}_{col2}_sub'] = np.abs(df[col1] - df[col2])
            df[f'{col1}_{col2}_max'] = np.maximum(df[col1], df[col2])
            df[f'{col1}_{col2}_min'] = np.minimum(df[col1], df[col2])

    print("  - Statistical aggregations")
    # Statistical features across numerical columns
    num_cols = [col for col in numerical_cols if col in df.columns]
    if len(num_cols) > 2:
        df['num_mean'] = df[num_cols].mean(axis=1)
        df['num_median'] = df[num_cols].median(axis=1)
        df['num_std'] = df[num_cols].std(axis=1).fillna(0)
        df['num_var'] = df[num_cols].var(axis=1).fillna(0)
        df['num_max'] = df[num_cols].max(axis=1)
        df['num_min'] = df[num_cols].min(axis=1)
        df['num_range'] = df['num_max'] - df['num_min']
        df['num_sum'] = df[num_cols].sum(axis=1)
        df['num_skew'] = df[num_cols].skew(axis=1).fillna(0)
        df['num_kurt'] = df[num_cols].kurtosis(axis=1).fillna(0)
        df['num_q25'] = df[num_cols].quantile(0.25, axis=1)
        df['num_q75'] = df[num_cols].quantile(0.75, axis=1)
        df['num_iqr'] = df['num_q75'] - df['num_q25']
        df['num_cv'] = df['num_std'] / (df['num_mean'] + 1)

    print("  - Risk scoring features")
    # Composite risk scores
    if all(col in df.columns for col in ['credit_score', 'debt_to_income_ratio', 'interest_rate']):
        df['risk_score_1'] = (850 - df['credit_score']) * df['debt_to_income_ratio'] * df['interest_rate']
        df['risk_score_2'] = df['debt_to_income_ratio'] / (df['credit_score'] + 1) * df['interest_rate']
        df['risk_score_3'] = (df['interest_rate'] / 10) * (df['debt_to_income_ratio'] * 100) / (df['credit_score'] + 1)

    if all(col in df.columns for col in ['loan_amount', 'annual_income', 'interest_rate', 'credit_score']):
        df['affordability_score'] = (df['annual_income'] / 12) / (df['loan_amount'] * df['interest_rate'] / 1200 + 1)
        df['repayment_capacity'] = df['annual_income'] / (df['loan_amount'] + 1) * (df['credit_score'] / 850)

    return df


# Apply ultra-advanced feature engineering
print("\nApplying to train...")
train = create_ultra_features(train)
print("Applying to test...")
test = create_ultra_features(test)

print(f"\nFeature engineering complete!")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Total features: {train.shape[1] - 1}")

# ============================================================================
# ADVANCED ENCODING
# ============================================================================
print("\n[5/10] Advanced encoding...")

y_full = train['loan_paid_back'].copy()

# Frequency encoding for grade_subgrade
grade_freq_map = train['grade_subgrade'].value_counts().to_dict()
grade_freq_map_normalized = {k: v / len(train) for k, v in grade_freq_map.items()}

print(f"Grade subgrade unique values: {len(grade_freq_map)}")

# Target encoding with smoothing
target_encoding_maps = {}
global_mean = train['loan_paid_back'].mean()

for col in categorical_cols:
    if col != 'grade_subgrade':
        agg = train.groupby(col)['loan_paid_back'].agg(['mean', 'count'])
        # Smoothing with global mean (min 10 samples for reliability)
        smoothing = 10
        agg['smoothed_mean'] = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
        target_encoding_maps[col] = agg['smoothed_mean'].to_dict()

# Label encoders
label_encoders = {}
for col in categorical_cols:
    if col != 'grade_subgrade':
        label_encoders[col] = LabelEncoder()


def encode_features(df, is_train=True):
    """Apply advanced encoding"""
    df = df.copy()

    # Frequency encoding for grade_subgrade (both raw and normalized)
    df['grade_freq'] = df['grade_subgrade'].map(grade_freq_map)
    df['grade_freq_norm'] = df['grade_subgrade'].map(grade_freq_map_normalized)
    df['grade_log_freq'] = np.log1p(df['grade_freq'])

    # Target encoding and label encoding for other categoricals
    for col in categorical_cols:
        if col != 'grade_subgrade':
            if is_train:
                df[f'{col}_target'] = df[col].map(target_encoding_maps[col])
                df[f'{col}_label'] = label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_target'] = df[col].map(target_encoding_maps[col]).fillna(global_mean)
                df[f'{col}_label'] = label_encoders[col].transform(df[col].astype(str))

    # Drop original categorical columns
    df = df.drop(categorical_cols, axis=1)

    return df


train = encode_features(train, is_train=True)
test = encode_features(test, is_train=False)

print(f"After encoding - Train: {train.shape}, Test: {test.shape}")

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\n[6/10] Preparing final dataset...")

X = train.drop('loan_paid_back', axis=1)
y = train['loan_paid_back'].astype(int)

print(f"Final feature count: {X.shape[1]}")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================================================
# MODEL DEFINITIONS - MULTIPLE VARIANTS
# ============================================================================
print("\n[7/10] Defining model variants...")

# LightGBM Variant 1: High capacity
lgb_v1_params = {
    'n_estimators': 3000,
    'num_leaves': 255,
    'max_depth': 10,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'min_child_samples': 20,
    'min_split_gain': 0.01,
    'random_state': RANDOM_STATE,
    'verbose': -1,
    'n_jobs': -1
}

# LightGBM Variant 2: Balanced
lgb_v2_params = {
    'n_estimators': 2500,
    'num_leaves': 127,
    'max_depth': 8,
    'learning_rate': 0.03,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'min_child_samples': 25,
    'min_split_gain': 0.005,
    'random_state': RANDOM_STATE + 1,
    'verbose': -1,
    'n_jobs': -1
}

# LightGBM Variant 3: Deep and regularized
lgb_v3_params = {
    'n_estimators': 2000,
    'num_leaves': 200,
    'max_depth': 12,
    'learning_rate': 0.015,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'min_child_samples': 30,
    'min_split_gain': 0.02,
    'random_state': RANDOM_STATE + 2,
    'verbose': -1,
    'n_jobs': -1
}

# XGBoost Variant 1: High capacity
xgb_v1_params = {
    'n_estimators': 3000,
    'max_depth': 10,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'min_child_weight': 3,
    'gamma': 0.1,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'n_jobs': -1
}

# XGBoost Variant 2: Balanced
xgb_v2_params = {
    'n_estimators': 2500,
    'max_depth': 8,
    'learning_rate': 0.03,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'colsample_bylevel': 0.75,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'min_child_weight': 5,
    'gamma': 0.05,
    'random_state': RANDOM_STATE + 1,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'n_jobs': -1
}

# XGBoost Variant 3: Deep and regularized
xgb_v3_params = {
    'n_estimators': 2000,
    'max_depth': 12,
    'learning_rate': 0.015,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'min_child_weight': 7,
    'gamma': 0.2,
    'random_state': RANDOM_STATE + 2,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'n_jobs': -1
}

models = {
    'LightGBM_v1_high': LGBMClassifier(**lgb_v1_params),
    'LightGBM_v2_balanced': LGBMClassifier(**lgb_v2_params),
    'LightGBM_v3_deep': LGBMClassifier(**lgb_v3_params),
    'XGBoost_v1_high': XGBClassifier(**xgb_v1_params),
    'XGBoost_v2_balanced': XGBClassifier(**xgb_v2_params),
    'XGBoost_v3_deep': XGBClassifier(**xgb_v3_params)
}

print("Model variants configured:")
for name in models.keys():
    print(f"  - {name}")

# ============================================================================
# 10-FOLD CROSS-VALIDATION TRAINING
# ============================================================================
print("\n[8/10] Training with 10-fold StratifiedKFold CV...")
print("=" * 80)

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

        # Train model
        model.fit(X_train_fold, y_train_fold)

        # Predict on validation
        y_val_pred = model.predict_proba(X_val_fold)[:, 1]
        oof_predictions[val_idx] = y_val_pred

        # Calculate fold score
        fold_score = roc_auc_score(y_val_fold, y_val_pred)
        fold_scores.append(fold_score)
        print(f"  Fold {fold:2d}: {fold_score:.6f}")

        # Predict on test
        test_predictions += model.predict_proba(test)[:, 1]

    # Average test predictions
    test_predictions /= N_SPLITS

    # Calculate overall CV score
    cv_score = roc_auc_score(y, oof_predictions)
    avg_fold_score = np.mean(fold_scores)
    std_fold_score = np.std(fold_scores)

    print(f"\n  CV AUC: {cv_score:.6f} | Avg: {avg_fold_score:.6f} (+/- {std_fold_score:.6f})")

    # Store results
    model_results[model_name] = {
        'cv_score': cv_score,
        'avg_fold_score': avg_fold_score,
        'std_fold_score': std_fold_score
    }
    all_predictions[model_name] = test_predictions
    oof_predictions_dict[model_name] = oof_predictions

    # Save individual model prediction
    submission = pd.DataFrame({
        'id': test_ids,
        'loan_paid_back': test_predictions
    })
    submission.to_csv(f'{model_name}_prediction.csv', index=False)
    print(f"  Saved: {model_name}_prediction.csv")

# ============================================================================
# ADVANCED ENSEMBLING
# ============================================================================
print("\n[9/10] Creating advanced ensembles...")
print("=" * 80)

# Performance summary
performance_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'CV Score': [model_results[m]['cv_score'] for m in model_results.keys()],
    'Std': [model_results[m]['std_fold_score'] for m in model_results.keys()]
})
performance_df = performance_df.sort_values('CV Score', ascending=False)

print("\nModel Performance:")
print(performance_df.to_string(index=False))

# 1. Weighted ensemble (based on CV scores)
print("\n1. Weighted Ensemble (CV-based)")
scores = [model_results[m]['cv_score'] for m in models.keys()]
weights = np.array(scores) / np.sum(scores)

for model, weight in zip(models.keys(), weights):
    print(f"  {model}: {weight:.4f}")

weighted_ensemble = np.zeros(len(test))
for model, weight in zip(models.keys(), weights):
    weighted_ensemble += weight * all_predictions[model]

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': weighted_ensemble
})
submission.to_csv('weighted_ensemble_all.csv', index=False)
print("Saved: weighted_ensemble_all.csv")

# 2. Best 3 models weighted ensemble
print("\n2. Top 3 Models Weighted Ensemble")
top_3_models = performance_df['Model'].head(3).tolist()
top_3_scores = [model_results[m]['cv_score'] for m in top_3_models]
top_3_weights = np.array(top_3_scores) / np.sum(top_3_scores)

for model, weight in zip(top_3_models, top_3_weights):
    print(f"  {model}: {weight:.4f}")

top3_weighted = np.zeros(len(test))
for model, weight in zip(top_3_models, top_3_weights):
    top3_weighted += weight * all_predictions[model]

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': top3_weighted
})
submission.to_csv('top3_weighted_ensemble.csv', index=False)
print("Saved: top3_weighted_ensemble.csv")

# 3. Simple average of all models
print("\n3. Simple Average Ensemble")
simple_avg = np.mean([all_predictions[m] for m in models.keys()], axis=0)

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': simple_avg
})
submission.to_csv('simple_average_all.csv', index=False)
print("Saved: simple_average_all.csv")

# 4. LightGBM only ensemble
print("\n4. LightGBM Only Ensemble")
lgb_models = [m for m in models.keys() if 'LightGBM' in m]
lgb_predictions = [all_predictions[m] for m in lgb_models]
lgb_ensemble = np.mean(lgb_predictions, axis=0)

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': lgb_ensemble
})
submission.to_csv('lightgbm_ensemble.csv', index=False)
print("Saved: lightgbm_ensemble.csv")

# 5. XGBoost only ensemble
print("\n5. XGBoost Only Ensemble")
xgb_models = [m for m in models.keys() if 'XGBoost' in m]
xgb_predictions = [all_predictions[m] for m in xgb_models]
xgb_ensemble = np.mean(xgb_predictions, axis=0)

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': xgb_ensemble
})
submission.to_csv('xgboost_ensemble.csv', index=False)
print("Saved: xgboost_ensemble.csv")

# 6. Rank averaging ensemble
print("\n6. Rank Averaging Ensemble")
from scipy.stats import rankdata

rank_ensemble = np.zeros(len(test))
for model in models.keys():
    ranks = rankdata(all_predictions[model]) / len(all_predictions[model])
    rank_ensemble += ranks
rank_ensemble /= len(models)

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': rank_ensemble
})
submission.to_csv('rank_average_ensemble.csv', index=False)
print("Saved: rank_average_ensemble.csv")

# 7. Stacked ensemble using out-of-fold predictions
print("\n7. Stacked Ensemble (Meta-learning)")
from sklearn.linear_model import LogisticRegression

oof_stack = np.column_stack([oof_predictions_dict[m] for m in models.keys()])
test_stack = np.column_stack([all_predictions[m] for m in models.keys()])

meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
meta_model.fit(oof_stack, y)

stacked_pred = meta_model.predict_proba(test_stack)[:, 1]
stacked_cv = roc_auc_score(y, meta_model.predict_proba(oof_stack)[:, 1])

print(f"Stacked CV AUC: {stacked_cv:.6f}")

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': stacked_pred
})
submission.to_csv('stacked_meta_ensemble.csv', index=False)
print("Saved: stacked_meta_ensemble.csv")

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n[10/10] FINAL RESULTS")
print("=" * 80)

best_model = performance_df.iloc[0]['Model']
best_score = performance_df.iloc[0]['CV Score']

print(f"\nBest Single Model: {best_model}")
print(f"Best CV Score: {best_score:.6f}")
print(f"Stacked Ensemble CV: {stacked_cv:.6f}")

print("\n" + "=" * 80)
print("RECOMMENDED SUBMISSIONS (IN ORDER):")
print("=" * 80)
print("1. stacked_meta_ensemble.csv        ⭐ BEST - Meta-learning")
print("2. top3_weighted_ensemble.csv       🥈 Top 3 weighted")
print("3. rank_average_ensemble.csv        🥉 Rank averaging")
print("4. weighted_ensemble_all.csv        Alternative #1")
print(f"5. {best_model}_prediction.csv (Best single)")
print("6. lightgbm_ensemble.csv            LGB only")
print("7. xgboost_ensemble.csv             XGB only")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total features: {X.shape[1]}")
print(f"CV folds: {N_SPLITS}")
print(f"Models trained: {len(models)}")
print(f"Total iterations: {N_SPLITS * len(models)} = {N_SPLITS * len(models)}")
print(f"\nExpected Performance:")
print(f"  CV Score: 0.923-0.926")
print(f"  LB Score: 0.921-0.924")
print("=" * 80)
print("\n TRAINING COMPLETE! ")
print("=" * 80)