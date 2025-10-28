import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBRegressor
from category_encoders import TargetEncoder as CETargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import time
from datetime import datetime
from itertools import combinations
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 80)
print("BLENDED XGBOOST MODEL - ACCIDENT RISK PREDICTION")
print("=" * 80)

# ==================== DATA LOADING ====================
print("\n[1/8] Loading data...")
dtype_dict = {
    'speed_limit': 'float32',
    'curvature': 'float32',
    'num_lanes': 'int8',
    'num_reported_accidents': 'int8'
}

# Update these paths for your environment
TRAIN_PATH = "/Users/brianbutler/PycharmProjects/My Work/In Progress Works/Kaggle/train.csv"
TEST_PATH = "/Users/brianbutler/PycharmProjects/My Work/In Progress Works/Kaggle/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Store IDs
train_ids = train["id"].copy() if "id" in train.columns else None
test_ids = test["id"].copy()

# Drop IDs
if "id" in train.columns:
    train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)

# Remove duplicates from train
train = train.drop_duplicates()

# Extract target
y_train = train["accident_risk"].values
train.drop("accident_risk", axis=1, inplace=True)

print(f"✓ Train: {train.shape}, Test: {test.shape}")
print(f"✓ Target range: [{y_train.min():.4f}, {y_train.max():.4f}]")

# ==================== COMPREHENSIVE FEATURE ENGINEERING ====================
print("\n[2/8] Creating comprehensive features...")


def create_blended_features(train_df, test_df):
    """Combines multiple feature engineering approaches"""
    train_new, test_new = train_df.copy(), test_df.copy()

    # Convert dtypes for efficiency
    for col in ['speed_limit', 'curvature']:
        if col in train_new.columns:
            train_new[col] = train_new[col].astype('float32')
            test_new[col] = test_new[col].astype('float32')

    for col in ['num_lanes', 'num_reported_accidents']:
        if col in train_new.columns:
            train_new[col] = train_new[col].astype('int8')
            test_new[col] = test_new[col].astype('int8')

    # Identify column types
    num_cols = train_new.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train_new.select_dtypes(include=["object", "bool"]).columns.tolist()

    print(f"  → Found {len(num_cols)} numerical and {len(cat_cols)} categorical columns")

    # ========== NUMERICAL FEATURES ==========
    print("  → Creating numerical transformations...")

    # Power transforms
    for col in num_cols:
        if col in ['speed_limit', 'curvature']:
            # Square and cube
            train_new[f'{col}_squared'] = train_new[col] ** 2
            test_new[f'{col}_squared'] = test_new[col] ** 2
            train_new[f'{col}_cubed'] = train_new[col] ** 3
            test_new[f'{col}_cubed'] = test_new[col] ** 3

            # Log and sqrt
            train_new[f'{col}_log'] = np.log1p(train_new[col])
            test_new[f'{col}_log'] = np.log1p(test_new[col])
            train_new[f'{col}_sqrt'] = np.sqrt(train_new[col])
            test_new[f'{col}_sqrt'] = np.sqrt(test_new[col])

            # Exponential (clipped for safety)
            train_new[f'{col}_exp'] = np.exp(np.clip(train_new[col] / train_new[col].max(), 0, 5))
            test_new[f'{col}_exp'] = np.exp(np.clip(test_new[col] / test_new[col].max(), 0, 5))

    # ========== NUMERICAL INTERACTIONS ==========
    print("  → Creating numerical interactions...")
    for r in [2]:  # Pairwise interactions
        for pair in combinations(num_cols, r):
            name = '_x_'.join(pair)
            train_new[name] = train_new[pair[0]]
            for col in pair[1:]:
                train_new[name] = train_new[name] * train_new[col]

            test_new[name] = test_new[pair[0]]
            for col in pair[1:]:
                test_new[name] = test_new[name] * test_new[col]

            # Also create division features for key pairs
            if pair[0] in ['speed_limit', 'curvature'] and pair[1] != pair[0]:
                ratio_name = f'{pair[0]}_div_{pair[1]}'
                train_new[ratio_name] = train_new[pair[0]] / (train_new[pair[1]] + 1)
                test_new[ratio_name] = test_new[pair[0]] / (test_new[pair[1]] + 1)

    # Three-way interactions for most important features
    if all(col in num_cols for col in ['speed_limit', 'curvature', 'num_lanes']):
        train_new['speed_curv_lanes'] = train_new['speed_limit'] * train_new['curvature'] * train_new['num_lanes']
        test_new['speed_curv_lanes'] = test_new['speed_limit'] * test_new['curvature'] * test_new['num_lanes']

    # ========== CATEGORICAL COMBINATIONS ==========
    print("  → Creating categorical combinations...")
    TE_columns = []

    # Convert booleans to strings for combinations
    bool_cols = train_new.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        train_new[col] = train_new[col].astype(str)
        test_new[col] = test_new[col].astype(str)

    # Update categorical columns list
    cat_cols = train_new.select_dtypes(include=["object"]).columns.tolist()

    # Categorical-Categorical combinations
    if len(cat_cols) > 1:
        for r in [2]:
            cat_combinations = list(combinations(cat_cols, r))[:20]  # Limit to prevent memory issues
            for cols in tqdm(cat_combinations, desc="    Cat combinations", leave=False):
                name = '_+_'.join(cols)

                train_new[name] = train_new[cols[0]].astype('str')
                for col in cols[1:]:
                    train_new[name] = train_new[name] + '_' + train_new[col].astype('str')

                test_new[name] = test_new[cols[0]].astype('str')
                for col in cols[1:]:
                    test_new[name] = test_new[name] + '_' + test_new[col].astype('str')

                # Factorize
                combined = pd.concat([train_new[name], test_new[name]], ignore_index=True)
                combined, _ = combined.factorize()

                train_new[name] = combined[:len(train_new)].astype('int16')
                test_new[name] = combined[len(train_new):].astype('int16')
                TE_columns.append(name)

    # Categorical-Numerical combinations (binned numerical)
    print("  → Creating cat-num combinations...")
    for cat_col in cat_cols[:5]:  # Limit to top categorical columns
        for num_col in ['speed_limit', 'curvature']:
            if num_col in num_cols:
                # Bin the numerical column
                try:
                    train_binned, bins = pd.qcut(train_new[num_col], q=5, labels=False,
                                                 duplicates='drop', retbins=True)
                    test_binned = pd.cut(test_new[num_col], bins=bins, labels=False,
                                         include_lowest=True).fillna(2)

                    name = f'{cat_col}_+_{num_col}_bin'
                    train_new[name] = train_new[cat_col].astype(str) + '_' + train_binned.astype(str)
                    test_new[name] = test_new[cat_col].astype(str) + '_' + test_binned.astype(str)

                    # Factorize
                    combined = pd.concat([train_new[name], test_new[name]], ignore_index=True)
                    combined, _ = combined.factorize()

                    train_new[name] = combined[:len(train_new)].astype('int16')
                    test_new[name] = combined[len(train_new):].astype('int16')
                    TE_columns.append(name)
                except:
                    pass

    # ========== FREQUENCY ENCODING ==========
    print("  → Creating frequency encodings...")
    freq_cols = cat_cols + TE_columns[:10]  # Limit to prevent too many features

    for col in freq_cols:
        if col in train_new.columns:
            freq = train_new[col].value_counts(normalize=True).to_dict()
            train_new[f'{col}_freq'] = train_new[col].map(freq).astype('float32')
            test_new[f'{col}_freq'] = test_new[col].map(freq).fillna(0).astype('float32')

    # ========== BINNING NUMERICAL FEATURES ==========
    print("  → Creating binned features...")
    for col in num_cols:
        for n_bins in [5, 10]:
            try:
                train_new[f'{col}_bin{n_bins}'], bins = pd.qcut(
                    train_new[col], q=n_bins, labels=False, duplicates='drop', retbins=True
                )
                test_new[f'{col}_bin{n_bins}'] = pd.cut(
                    test_new[col], bins=bins, labels=False, include_lowest=True
                ).fillna(0).astype('int8')
                train_new[f'{col}_bin{n_bins}'] = train_new[f'{col}_bin{n_bins}'].astype('int8')
            except:
                train_new[f'{col}_bin{n_bins}'] = 0
                test_new[f'{col}_bin{n_bins}'] = 0

    # ========== STATISTICAL AGGREGATIONS ==========
    print("  → Creating statistical aggregations...")
    for col in cat_cols[:5]:  # Limit to prevent memory issues
        for num_col in num_cols:
            # Mean encoding
            agg_mean = train_new.groupby(col)[num_col].transform('mean')
            train_new[f'{col}_{num_col}_mean'] = agg_mean.astype('float32')

            group_means = train_new.groupby(col)[num_col].mean().to_dict()
            test_new[f'{col}_{num_col}_mean'] = test_new[col].map(group_means).fillna(agg_mean.mean()).astype('float32')

            # Std encoding
            agg_std = train_new.groupby(col)[num_col].transform('std').fillna(0)
            train_new[f'{col}_{num_col}_std'] = agg_std.astype('float32')

            group_stds = train_new.groupby(col)[num_col].std().fillna(0).to_dict()
            test_new[f'{col}_{num_col}_std'] = test_new[col].map(group_stds).fillna(0).astype('float32')

    # ========== RISK-BASED FEATURES ==========
    print("  → Creating risk-based features...")

    # Composite risk score
    if all(col in train_new.columns for col in ['speed_limit', 'curvature', 'num_reported_accidents']):
        train_new['risk_score'] = (
                0.4 * train_new['curvature'] +
                0.35 * (train_new['speed_limit'] / train_new['speed_limit'].max()) +
                0.25 * (train_new['num_reported_accidents'] / train_new['num_reported_accidents'].max())
        )
        test_new['risk_score'] = (
                0.4 * test_new['curvature'] +
                0.35 * (test_new['speed_limit'] / test_new['speed_limit'].max()) +
                0.25 * (test_new['num_reported_accidents'] / test_new['num_reported_accidents'].max())
        )

    # High risk flags
    if 'speed_limit' in train_new.columns and 'curvature' in train_new.columns:
        train_new['high_speed_curve'] = ((train_new['speed_limit'] > 50) & (train_new['curvature'] > 0.5)).astype(
            'int8')
        test_new['high_speed_curve'] = ((test_new['speed_limit'] > 50) & (test_new['curvature'] > 0.5)).astype('int8')

    # Store TE columns for later use
    train_new.attrs['TE_columns'] = TE_columns

    print(f"  → Total features created: {train_new.shape[1]}")

    return train_new, test_new


X_train_full, X_test_full = create_blended_features(train, test)
TE_columns = X_train_full.attrs.get('TE_columns', [])

print(f"✓ Total features created: {X_train_full.shape[1]}")

# ==================== PREPROCESSING ====================
print("\n[3/8] Processing categorical features...")

# Convert remaining object columns to category
for col in X_train_full.select_dtypes(include=['object']).columns:
    # Use LabelEncoder for safety
    le = LabelEncoder()

    # Fit on combined data to ensure consistency
    combined_vals = pd.concat([X_train_full[col], X_test_full[col]])
    le.fit(combined_vals)

    X_train_full[col] = le.transform(X_train_full[col])
    X_test_full[col] = le.transform(X_test_full[col])

# Convert to category dtype for XGBoost
cat_features = []
for col in X_train_full.columns:
    if X_train_full[col].dtype in ['object', 'int8', 'int16'] or 'bin' in col or '_+_' in col:
        X_train_full[col] = X_train_full[col].astype('category')
        X_test_full[col] = pd.Categorical(X_test_full[col])
        cat_features.append(col)

print(f"✓ Categorical features: {len(cat_features)}")

# ==================== MODEL CONFIGURATION ====================
print("\n[4/8] Configuring ensemble models...")

# Model 1: Deep trees with more regularization (better for complex patterns)
MODEL1_PARAMS = {
    'max_depth': 9,
    'learning_rate': 0.015,
    'n_estimators': 100000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.85,
    'colsample_bynode': 0.88,
    'min_child_weight': 4,
    'gamma': 0.01,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'max_delta_step': 1,
    'max_bin': 512,
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'random_state': 42,
    'enable_categorical': True,
    'early_stopping_rounds': 200,
    'verbosity': 0
}

# Model 2: Shallower trees, faster learning (better for general patterns)
MODEL2_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.03,
    'n_estimators': 100000,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 2,
    'gamma': 0.001,
    'reg_alpha': 0.01,
    'reg_lambda': 0.1,
    'random_state': 43,
    'eval_metric': 'rmse',
    'enable_categorical': True,
    'early_stopping_rounds': 200,
    'verbosity': 0
}

# Model 3: Very shallow, aggressive learning (captures different patterns)
MODEL3_PARAMS = {
    'max_depth': 4,
    'learning_rate': 0.05,
    'n_estimators': 100000,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0.001,
    'reg_lambda': 0.01,
    'random_state': 44,
    'eval_metric': 'rmse',
    'enable_categorical': True,
    'early_stopping_rounds': 200,
    'verbosity': 0
}

print("✓ Configured 3 diverse XGBoost models")

# ==================== TRAINING WITH TARGET ENCODING ====================
print("\n[5/8] Training ensemble with cross-validation...")

FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Initialize OOF and test predictions
oof_model1 = np.zeros(len(X_train_full), dtype='float32')
oof_model2 = np.zeros(len(X_train_full), dtype='float32')
oof_model3 = np.zeros(len(X_train_full), dtype='float32')
test_preds_model1 = np.zeros(len(X_test_full), dtype='float32')
test_preds_model2 = np.zeros(len(X_test_full), dtype='float32')
test_preds_model3 = np.zeros(len(X_test_full), dtype='float32')

fold_scores = {'model1': [], 'model2': [], 'model3': [], 'blended': []}
feature_importance_model1 = []
feature_importance_model2 = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train), 1):
    print(f"\n{'=' * 60}")
    print(f"Fold {fold}/{FOLDS}")
    print(f"{'=' * 60}")

    X_train_fold = X_train_full.iloc[train_idx].copy()
    X_val = X_train_full.iloc[val_idx].copy()
    y_train_fold = y_train[train_idx]
    y_val = y_train[val_idx]

    # Target encoding for high-cardinality features
    if len(TE_columns) > 0:
        print(f"  Applying target encoding to {len(TE_columns)} features...")
        te = CETargetEncoder(cols=TE_columns[:20])  # Limit to prevent overfitting
        X_train_fold = te.fit_transform(X_train_fold, y_train_fold)
        X_val = te.transform(X_val)

    # Train Model 1
    print("  Training Model 1 (Deep)...")
    model1 = XGBRegressor(**MODEL1_PARAMS)
    model1.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    pred1_val = model1.predict(X_val)
    pred1_test = model1.predict(X_test_full)

    oof_model1[val_idx] = pred1_val
    test_preds_model1 += pred1_test / FOLDS

    rmse1 = np.sqrt(np.mean((pred1_val - y_val) ** 2))
    fold_scores['model1'].append(rmse1)
    print(
        f"    Model 1 RMSE: {rmse1:.6f} ({model1.n_features_in_} features, {len(model1.get_booster().get_score())} used)")

    # Store feature importance
    if hasattr(model1, 'feature_importances_'):
        feature_importance_model1.append(model1.feature_importances_)

    # Train Model 2
    print("  Training Model 2 (Shallow)...")
    model2 = XGBRegressor(**MODEL2_PARAMS)
    model2.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    pred2_val = model2.predict(X_val)
    pred2_test = model2.predict(X_test_full)

    oof_model2[val_idx] = pred2_val
    test_preds_model2 += pred2_test / FOLDS

    rmse2 = np.sqrt(np.mean((pred2_val - y_val) ** 2))
    fold_scores['model2'].append(rmse2)
    print(
        f"    Model 2 RMSE: {rmse2:.6f} ({model2.n_features_in_} features, {len(model2.get_booster().get_score())} used)")

    if hasattr(model2, 'feature_importances_'):
        feature_importance_model2.append(model2.feature_importances_)

    # Train Model 3
    print("  Training Model 3 (Very Shallow)...")
    model3 = XGBRegressor(**MODEL3_PARAMS)
    model3.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    pred3_val = model3.predict(X_val)
    pred3_test = model3.predict(X_test_full)

    oof_model3[val_idx] = pred3_val
    test_preds_model3 += pred3_test / FOLDS

    rmse3 = np.sqrt(np.mean((pred3_val - y_val) ** 2))
    fold_scores['model3'].append(rmse3)
    print(
        f"    Model 3 RMSE: {rmse3:.6f} ({model3.n_features_in_} features, {len(model3.get_booster().get_score())} used)")

    # Blend predictions for this fold
    blended_val = (pred1_val + pred2_val + pred3_val) / 3
    rmse_blended = np.sqrt(np.mean((blended_val - y_val) ** 2))
    fold_scores['blended'].append(rmse_blended)
    print(f"    Simple Blend RMSE: {rmse_blended:.6f}")

# ==================== OPTIMAL BLENDING ====================
print("\n[6/8] Finding optimal blend weights...")

from scipy.optimize import minimize


# Function to optimize
def blend_rmse(weights):
    w1, w2 = weights
    w3 = 1 - w1 - w2
    if w3 < 0:
        return 1e10
    blended = w1 * oof_model1 + w2 * oof_model2 + w3 * oof_model3
    return np.sqrt(np.mean((blended - y_train) ** 2))


# Find optimal weights
result = minimize(blend_rmse, x0=[0.33, 0.33], bounds=[(0, 1), (0, 1)], method='L-BFGS-B')
w1, w2 = result.x
w3 = 1 - w1 - w2

print(f"✓ Optimal weights:")
print(f"    Model 1 (Deep):         {w1:.3f}")
print(f"    Model 2 (Shallow):      {w2:.3f}")
print(f"    Model 3 (Very Shallow): {w3:.3f}")

# Final blended predictions
oof_blended = w1 * oof_model1 + w2 * oof_model2 + w3 * oof_model3
test_preds_blended = w1 * test_preds_model1 + w2 * test_preds_model2 + w3 * test_preds_model3

# Clip to valid range
test_preds_blended = np.clip(test_preds_blended, 0, 1)

best_rmse = np.sqrt(np.mean((oof_blended - y_train) ** 2))
print(f"✓ Optimal blend CV RMSE: {best_rmse:.6f}")

# ==================== SAVE SUBMISSION ====================
print("\n[7/8] Saving submission...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': test_preds_blended
})
submission.to_csv('submission_xgb.csv', index=False)
print(f"✓ Main submission saved: submission_xgb.csv")

backup_filename = f"submission_xgb_{timestamp}_cv{best_rmse:.6f}.csv"
submission.to_csv(backup_filename, index=False)
print(f"✓ Backup saved: {backup_filename}")

# Save OOF predictions
pd.DataFrame({
    'model1_oof': oof_model1,
    'model2_oof': oof_model2,
    'model3_oof': oof_model3,
    'blended_oof': oof_blended,
    'target': y_train
}).to_csv(f'oof_predictions_xgb_{timestamp}.csv', index=False)

# ==================== ANALYSIS & VISUALIZATION ====================
print("\n[8/8] Creating analysis and visualizations...")

residuals = y_train - oof_blended
abs_residuals = np.abs(residuals)

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(f"\nModel Performance:")
print(f"  Model 1 CV RMSE:    {np.mean(fold_scores['model1']):.6f} (±{np.std(fold_scores['model1']):.6f})")
print(f"  Model 2 CV RMSE:    {np.mean(fold_scores['model2']):.6f} (±{np.std(fold_scores['model2']):.6f})")
print(f"  Model 3 CV RMSE:    {np.mean(fold_scores['model3']):.6f} (±{np.std(fold_scores['model3']):.6f})")
print(f"  Simple Avg RMSE:    {np.mean(fold_scores['blended']):.6f}")
print(f"  Optimal Blend RMSE: {best_rmse:.6f}")
print(f"\nTarget: 0.05537")
print(f"Gap:    {(best_rmse - 0.05537):.6f}")

pearson_corr, _ = pearsonr(y_train, oof_blended)
print(f"\nCorrelation (Pearson): {pearson_corr:.6f}")

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 12))

# 1. Model Comparison
plt.subplot(3, 4, 1)
models = ['Model 1\n(Deep)', 'Model 2\n(Shallow)', 'Model 3\n(V.Shallow)', 'Simple\nAvg', 'Optimal\nBlend']
means = [
    np.mean(fold_scores['model1']),
    np.mean(fold_scores['model2']),
    np.mean(fold_scores['model3']),
    np.mean(fold_scores['blended']),
    best_rmse
]
colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']
bars = plt.bar(models, means, alpha=0.7, edgecolor='black', color=colors)
plt.ylabel('RMSE')
plt.title('Model Comparison', fontweight='bold', fontsize=12)
plt.axhline(y=0.05537, color='red', linestyle='--', label='Target', alpha=0.5)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, mean in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
             f'{mean:.5f}', ha='center', va='bottom', fontsize=9)

# 2. Actual vs Predicted
plt.subplot(3, 4, 2)
plt.scatter(y_train, oof_blended, alpha=0.3, s=2, c='#3498db')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs Predicted (r={pearson_corr:.4f})', fontweight='bold')
plt.grid(True, alpha=0.3)

# 3. Residuals Distribution
plt.subplot(3, 4, 3)
plt.hist(residuals, bins=100, edgecolor='black', alpha=0.7, color='#2ecc71')
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title(f'Residual Distribution (μ={np.mean(residuals):.6f})', fontweight='bold')
plt.grid(True, alpha=0.3)

# 4. Model Agreement (1 vs 2)
plt.subplot(3, 4, 4)
plt.scatter(oof_model1, oof_model2, alpha=0.3, s=2, c=y_train, cmap='viridis')
plt.plot([0, 1], [0, 1], 'r--', lw=2)
plt.xlabel('Model 1 Predictions')
plt.ylabel('Model 2 Predictions')
plt.title('Model 1 vs Model 2', fontweight='bold')
plt.colorbar(label='Actual Risk')
plt.grid(True, alpha=0.3)

# 5. Fold Performance
plt.subplot(3, 4, 5)
x = np.arange(1, FOLDS + 1)
width = 0.2
plt.bar(x - 1.5 * width, fold_scores['model1'], width, label='Model 1', alpha=0.7, color='#3498db')
plt.bar(x - 0.5 * width, fold_scores['model2'], width, label='Model 2', alpha=0.7, color='#2ecc71')
plt.bar(x + 0.5 * width, fold_scores['model3'], width, label='Model 3', alpha=0.7, color='#f39c12')
plt.bar(x + 1.5 * width, fold_scores['blended'], width, label='Simple Avg', alpha=0.7, color='#9b59b6')
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('Fold Performance', fontweight='bold')
plt.legend(loc='best', fontsize=8)
plt.grid(True, alpha=0.3, axis='y')

# 6. Feature Importance (if available)
plt.subplot(3, 4, 6)
if len(feature_importance_model1) > 0:
    avg_imp1 = np.mean(feature_importance_model1, axis=0)
    top_indices = np.argsort(avg_imp1)[-15:][::-1]
    feature_names = X_train_fold.columns
    plt.barh(range(len(top_indices)), avg_imp1[top_indices], alpha=0.7, edgecolor='black', color='#e74c3c')
    plt.yticks(range(len(top_indices)), [str(feature_names[i])[:20] for i in top_indices], fontsize=7)
    plt.xlabel('Importance')
    plt.title('Top 15 Features (Model 1)', fontweight='bold')
    plt.gca().invert_yaxis()
else:
    plt.text(0.5, 0.5, 'Feature importance\nnot available', ha='center', va='center')
plt.grid(True, alpha=0.3, axis='x')

# 7. Cumulative Error
plt.subplot(3, 4, 7)
sorted_abs = np.sort(abs_residuals)
cumulative = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs) * 100
plt.plot(sorted_abs, cumulative, lw=2, color='#3498db')
plt.axhline(y=95, color='r', linestyle='--', label='95%')
plt.axhline(y=99, color='orange', linestyle='--', label='99%')
plt.xlabel('Absolute Error')
plt.ylabel('Cumulative %')
plt.title('Cumulative Error Distribution', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 8. Distribution Comparison
plt.subplot(3, 4, 8)
plt.hist(y_train, bins=50, alpha=0.5, label='Actual', edgecolor='black', color='#3498db')
plt.hist(oof_blended, bins=50, alpha=0.5, label='Predicted', edgecolor='black', color='#e74c3c')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution Comparison', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 9. Residuals vs Predicted
plt.subplot(3, 4, 9)
plt.scatter(oof_blended, residuals, alpha=0.3, s=2, c='#2ecc71')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
# Add confidence bands
std_residuals = np.std(residuals)
plt.axhline(y=2 * std_residuals, color='orange', linestyle=':', alpha=0.5, label='±2σ')
plt.axhline(y=-2 * std_residuals, color='orange', linestyle=':', alpha=0.5)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 10. Q-Q Plot
plt.subplot(3, 4, 10)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot', fontweight='bold')
plt.grid(True, alpha=0.3)

# 11. Model Correlation Matrix
plt.subplot(3, 4, 11)
model_preds = pd.DataFrame({
    'Model 1': oof_model1,
    'Model 2': oof_model2,
    'Model 3': oof_model3,
    'Actual': y_train
})
corr_matrix = model_preds.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0.8,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Model Correlation Matrix', fontweight='bold')

# 12. Prediction Ranges
plt.subplot(3, 4, 12)
ranges_data = {
    'Actual': [y_train.min(), y_train.max()],
    'Model 1': [oof_model1.min(), oof_model1.max()],
    'Model 2': [oof_model2.min(), oof_model2.max()],
    'Model 3': [oof_model3.min(), oof_model3.max()],
    'Blended': [oof_blended.min(), oof_blended.max()],
    'Test': [test_preds_blended.min(), test_preds_blended.max()]
}
x_pos = np.arange(len(ranges_data))
mins = [v[0] for v in ranges_data.values()]
maxs = [v[1] for v in ranges_data.values()]
plt.bar(x_pos - 0.2, mins, 0.4, alpha=0.7, label='Min', edgecolor='black', color='#3498db')
plt.bar(x_pos + 0.2, maxs, 0.4, alpha=0.7, label='Max', edgecolor='black', color='#e74c3c')
plt.xticks(x_pos, ranges_data.keys(), rotation=45, ha='right')
plt.ylabel('Value')
plt.title('Prediction Range Coverage', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
viz_filename = f"xgb_analysis_{timestamp}.png"
plt.savefig(viz_filename, dpi=120, bbox_inches='tight')
print(f"✓ Visualization saved: {viz_filename}")

print("\n" + "=" * 80)
print("✅ XGB ENSEMBLE MODEL COMPLETE")
print("=" * 80)
print(f"Final Blended CV RMSE: {best_rmse:.6f}")
print(f"Target RMSE:           0.05537")
print(f"Gap to target:         {(best_rmse - 0.05537):.6f}")
print(f"Submission ready:      submission_xgb.csv")
print("=" * 80)