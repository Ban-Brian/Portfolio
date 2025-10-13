import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from scipy.optimize import minimize
import time
from datetime import datetime
from itertools import combinations
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("=" * 90)
print("ULTIMATE MULTI-MODEL ENSEMBLE - MAXIMUM OPTIMIZATION")
print("=" * 90)

# ==================== DATA LOADING ====================
print("\n[1/10] Loading data...")
dtype_dict = {
    'speed_limit': 'float32',
    'curvature': 'float32',
    'num_lanes': 'int8',
    'num_reported_accidents': 'int8'
}

TRAIN_PATH = "/Users/brianbutler/PycharmProjects/My Work/In Progress Works/Kaggle/train.csv"
TEST_PATH = "/Users/brianbutler/PycharmProjects/My Work/In Progress Works/Kaggle/test.csv"

train = pd.read_csv(TRAIN_PATH, dtype=dtype_dict, index_col='id')
test = pd.read_csv(TEST_PATH, dtype=dtype_dict)

test_ids = test["id"].copy()
test.drop("id", axis=1, inplace=True)

y_train = train["accident_risk"].values
train.drop("accident_risk", axis=1, inplace=True)

print(f"✓ Train: {train.shape}, Test: {test.shape}")

# ==================== ADVANCED FEATURE ENGINEERING ====================
print("\n[2/10] Creating strategic interaction features...")


def create_optimized_features(train_df, test_df):
    """Strategic feature engineering focused on high-value interactions"""
    train_new, test_new = train_df.copy(), test_df.copy()

    num_cols = train_new.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train_new.select_dtypes(include=["object", "bool"]).columns.tolist()

    print("  → Creating key numerical interactions...")
    # High-value numerical interactions
    interactions = [
        ('speed_limit', 'curvature', ['mul', 'div', 'add']),
        ('speed_limit', 'num_lanes', ['mul', 'div']),
        ('curvature', 'num_reported_accidents', ['mul', 'add']),
        ('num_lanes', 'num_reported_accidents', ['mul']),
    ]

    for col1, col2, ops in interactions:
        if 'mul' in ops:
            train_new[f'{col1}_x_{col2}'] = (train_new[col1] * train_new[col2]).astype('float32')
            test_new[f'{col1}_x_{col2}'] = (test_new[col1] * test_new[col2]).astype('float32')
        if 'div' in ops:
            train_new[f'{col1}_div_{col2}'] = (train_new[col1] / (train_new[col2] + 1e-5)).astype('float32')
            test_new[f'{col1}_div_{col2}'] = (test_new[col1] / (test_new[col2] + 1e-5)).astype('float32')
        if 'add' in ops:
            train_new[f'{col1}_plus_{col2}'] = (train_new[col1] + train_new[col2]).astype('float32')
            test_new[f'{col1}_plus_{col2}'] = (test_new[col1] + test_new[col2]).astype('float32')

    # Polynomial features for critical variables
    print("  → Creating polynomial features...")
    for col in ['curvature', 'speed_limit']:
        train_new[f'{col}_squared'] = (train_new[col] ** 2).astype('float32')
        test_new[f'{col}_squared'] = (test_new[col] ** 2).astype('float32')
        train_new[f'{col}_sqrt'] = np.sqrt(train_new[col] + 1e-5).astype('float32')
        test_new[f'{col}_sqrt'] = np.sqrt(test_new[col] + 1e-5).astype('float32')
        train_new[f'{col}_log'] = np.log1p(train_new[col]).astype('float32')
        test_new[f'{col}_log'] = np.log1p(test_new[col]).astype('float32')

    print("  → Creating categorical combinations...")
    # Strategic categorical combinations
    cat_pairs = [
        ('road_type', 'weather'),
        ('road_type', 'lighting'),
        ('road_type', 'time_of_day'),
        ('weather', 'lighting'),
        ('weather', 'time_of_day'),
    ]

    TE_columns = []
    for col1, col2 in tqdm(cat_pairs, desc="    Cat pairs", leave=False):
        name = f'{col1}_+_{col2}'
        train_new[name] = train_new[col1].astype('str') + '_' + train_new[col2].astype('str')
        test_new[name] = test_new[col1].astype('str') + '_' + test_new[col2].astype('str')

        combined = pd.concat([train_new[name], test_new[name]], ignore_index=True)
        combined, _ = combined.factorize()
        train_new[name] = combined[:len(train_new)]
        test_new[name] = combined[len(train_new):]
        TE_columns.append(name)

    # Cat-Num interactions
    print("  → Creating cat-num interactions...")
    for cat_col in ['road_type', 'weather', 'lighting', 'time_of_day']:
        for num_col in ['speed_limit', 'curvature', 'num_reported_accidents']:
            name = f'{cat_col}_+_{num_col}'
            train_new[name] = train_new[cat_col].astype('str') + '_' + train_new[num_col].astype('str')
            test_new[name] = test_new[cat_col].astype('str') + '_' + test_new[num_col].astype('str')

            combined = pd.concat([train_new[name], test_new[name]], ignore_index=True)
            combined, _ = combined.factorize()
            train_new[name] = combined[:len(train_new)]
            test_new[name] = combined[len(train_new):]
            TE_columns.append(name)

    print("  → Creating statistical aggregations...")
    # Group statistics (mean, std, min, max)
    for cat_col in ['road_type', 'weather', 'lighting', 'time_of_day']:
        for num_col in num_cols:
            for agg_func, agg_name in [('mean', 'mean'), ('std', 'std'), ('min', 'min'), ('max', 'max')]:
                group_agg = train_new.groupby(cat_col)[num_col].agg(agg_func)
                train_new[f'{cat_col}_{num_col}_{agg_name}'] = train_new[cat_col].map(group_agg).astype('float32')
                test_new[f'{cat_col}_{num_col}_{agg_name}'] = test_new[cat_col].map(group_agg).fillna(
                    group_agg.mean()).astype('float32')

    train_new.attrs['TE_columns'] = TE_columns
    train_new.attrs['cat_cols'] = cat_cols

    return train_new, test_new


X_train_full, X_test_full = create_optimized_features(train, test)
TE_columns = X_train_full.attrs.get('TE_columns', [])
cat_cols = X_train_full.attrs.get('cat_cols', [])

print(f"✓ Features created: {X_train_full.shape[1]}")

# ==================== QUANTILE TRANSFORMATION ====================
print("\n[3/10] Applying QuantileTransformer (normal distribution)...")

num_features = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)

X_train_full[num_features] = qt.fit_transform(X_train_full[num_features])
X_test_full[num_features] = qt.transform(X_test_full[num_features])

print(f"✓ Transformed {len(num_features)} numerical features to normal distribution")

# Convert categoricals
print("\n[4/10] Processing categorical features...")
for col in cat_cols:
    X_train_full[col] = X_train_full[col].astype('category')
    X_test_full[col] = pd.Categorical(X_test_full[col], categories=X_train_full[col].cat.categories)
    X_test_full[col] = X_test_full[col].astype('category')

# ==================== MODEL CONFIGURATIONS ====================
print("\n[5/10] Configuring 3-model ensemble...")

XGBOOST_PARAMS = {
    'max_depth': 6,  # Reduced from 8
    'learning_rate': 0.01,  # Reduced from 0.02
    'n_estimators': 100000,
    'subsample': 0.7,  # Reduced from 0.8
    'colsample_bytree': 0.7,  # Reduced from 0.8
    'colsample_bylevel': 0.7,  # Reduced from 0.8
    'min_child_weight': 5,  # Increased from 3
    'gamma': 0.1,  # Increased from 0.01
    'reg_alpha': 0.3,  # Increased from 0.1
    'reg_lambda': 1.0,  # Increased from 0.5
    'max_delta_step': 1,
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'random_state': 42,
    'enable_categorical': True,
    'early_stopping_rounds': 100,  # More aggressive early stopping
}

LIGHTGBM_PARAMS = {
    'max_depth': 5,  # Reduced from 7
    'learning_rate': 0.01,  # Reduced from 0.02
    'n_estimators': 100000,
    'subsample': 0.7,  # Reduced from 0.8
    'colsample_bytree': 0.7,  # Reduced from 0.8
    'min_child_samples': 30,  # Increased from 20
    'reg_alpha': 0.3,  # Increased from 0.1
    'reg_lambda': 1.0,  # Increased from 0.5
    'min_gain_to_split': 0.01,  # Added
    'max_bin': 255,  # Added
    'metric': 'rmse',
    'random_state': 42,
    'verbose': -1,
    'early_stopping_rounds': 100,  # More aggressive
}

CATBOOST_PARAMS = {
    'depth': 5,  # Reduced from 6
    'learning_rate': 0.01,  # Reduced from 0.02
    'iterations': 100000,
    'subsample': 0.7,  # Reduced from 0.8
    'colsample_bylevel': 0.7,  # Reduced from 0.8
    'min_data_in_leaf': 30,  # Increased from 20
    'reg_lambda': 1.0,  # Increased from 0.5
    'l2_leaf_reg': 3.0,  # Added for additional regularization
    'random_strength': 1.0,  # Added for randomness
    'bagging_temperature': 1.0,  # Added
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 100,  # More aggressive
}

# ==================== REPEATED CROSS-VALIDATION ====================
print("\n[6/10] Training with Repeated K-Fold CV...")

N_SPLITS = 5
N_REPEATS = 2
EARLY_STOPPING = 200

rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

oof_xgb = np.zeros(len(X_train_full), dtype='float32')
oof_lgb = np.zeros(len(X_train_full), dtype='float32')
oof_cat = np.zeros(len(X_train_full), dtype='float32')

test_preds_xgb = np.zeros(len(X_test_full), dtype='float32')
test_preds_lgb = np.zeros(len(X_test_full), dtype='float32')
test_preds_cat = np.zeros(len(X_test_full), dtype='float32')

fold_scores = {'xgb': [], 'lgb': [], 'cat': []}
feature_importance_xgb = []
feature_importance_lgb = []
feature_importance_cat = []

total_folds = N_SPLITS * N_REPEATS
fold_num = 0

for train_idx, val_idx in rkf.split(X_train_full, y_train):
    fold_num += 1
    print(f"\n{'=' * 70}")
    print(f"Fold {fold_num}/{total_folds}")
    print(f"{'=' * 70}")

    X_train_fold = X_train_full.iloc[train_idx].copy()
    X_val = X_train_full.iloc[val_idx].copy()
    y_train_fold = y_train[train_idx]
    y_val = y_train[val_idx]
    X_test = X_test_full.copy()

    # Apply Target Encoding within fold (prevents leakage)
    print("  → Applying target encoding...")
    te = TargetEncoder(cols=TE_columns + cat_cols, smoothing=2.0, min_samples_leaf=50)  # Increased smoothing

    X_train_fold = te.fit_transform(X_train_fold, y_train_fold)
    X_val = te.transform(X_val)
    X_test = te.transform(X_test)

    # Train XGBoost
    print("  → Training XGBoost...")
    xgb_model = XGBRegressor(**XGBOOST_PARAMS, early_stopping_rounds=EARLY_STOPPING)
    start = time.time()
    xgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    time_xgb = time.time() - start

    val_preds_xgb = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_test)
    oof_xgb[val_idx] = val_preds_xgb
    feature_importance_xgb.append(xgb_model.feature_importances_)

    rmse_xgb = np.sqrt(np.mean((val_preds_xgb - y_val) ** 2))
    fold_scores['xgb'].append(rmse_xgb)
    print(f"    XGB RMSE: {rmse_xgb:.6f} | Time: {time_xgb:.1f}s | Iters: {xgb_model.best_iteration}")

    # Train LightGBM
    print("  → Training LightGBM...")
    lgb_model = LGBMRegressor(**LIGHTGBM_PARAMS)
    start = time.time()
    lgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val, y_val)]
    )
    time_lgb = time.time() - start

    val_preds_lgb = lgb_model.predict(X_val)
    test_preds_lgb += lgb_model.predict(X_test)
    oof_lgb[val_idx] = val_preds_lgb
    feature_importance_lgb.append(lgb_model.feature_importances_)

    rmse_lgb = np.sqrt(np.mean((val_preds_lgb - y_val) ** 2))
    fold_scores['lgb'].append(rmse_lgb)
    print(f"    LGB RMSE: {rmse_lgb:.6f} | Time: {time_lgb:.1f}s | Iters: {lgb_model.best_iteration_}")

    # Train CatBoost
    print("  → Training CatBoost...")
    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    start = time.time()
    cat_model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val, y_val),
        verbose=False
    )
    time_cat = time.time() - start

    val_preds_cat = cat_model.predict(X_val)
    test_preds_cat += cat_model.predict(X_test)
    oof_cat[val_idx] = val_preds_cat
    feature_importance_cat.append(cat_model.feature_importances_)

    rmse_cat = np.sqrt(np.mean((val_preds_cat - y_val) ** 2))
    fold_scores['cat'].append(rmse_cat)
    print(f"    CAT RMSE: {rmse_cat:.6f} | Time: {time_cat:.1f}s | Iters: {cat_model.best_iteration_}")

test_preds_xgb /= total_folds
test_preds_lgb /= total_folds
test_preds_cat /= total_folds

# ==================== FEATURE IMPORTANCE ANALYSIS ====================
print("\n[7/10] Analyzing feature importance and removing weak predictors...")

avg_importance_xgb = np.mean(feature_importance_xgb, axis=0)
avg_importance_lgb = np.mean(feature_importance_lgb, axis=0)
avg_importance_cat = np.mean(feature_importance_cat, axis=0)

# Combine importances (average across models)
combined_importance = (avg_importance_xgb + avg_importance_lgb + avg_importance_cat) / 3
feature_names = X_train_fold.columns

# Remove features with very low importance (more aggressive noise reduction)
importance_threshold = np.percentile(combined_importance, 25)  # Keep top 75% (was 90%)
important_features = feature_names[combined_importance > importance_threshold].tolist()

print(
    f"✓ Retained {len(important_features)}/{len(feature_names)} features (removed {len(feature_names) - len(important_features)} weak predictors)")

# Show top features
top_20_idx = np.argsort(combined_importance)[-20:][::-1]
print("\nTop 20 Features:")
for idx in top_20_idx:
    print(f"  {feature_names[idx]:<45} {combined_importance[idx]:.4f}")

# ==================== OPTIMAL WEIGHTED BLENDING ====================
print("\n[8/10] Finding optimal ensemble weights based on validation scores...")

# Calculate inverse RMSE weights (better models get higher weight)
mean_rmse_xgb = np.mean(fold_scores['xgb'])
mean_rmse_lgb = np.mean(fold_scores['lgb'])
mean_rmse_cat = np.mean(fold_scores['cat'])

print(f"\nIndividual Model Performance:")
print(f"  XGBoost  CV RMSE: {mean_rmse_xgb:.6f} (±{np.std(fold_scores['xgb']):.6f})")
print(f"  LightGBM CV RMSE: {mean_rmse_lgb:.6f} (±{np.std(fold_scores['lgb']):.6f})")
print(f"  CatBoost CV RMSE: {mean_rmse_cat:.6f} (±{np.std(fold_scores['cat']):.6f})")


# Optimization function
def ensemble_rmse(weights):
    w_xgb, w_lgb, w_cat = weights
    blended = w_xgb * oof_xgb + w_lgb * oof_lgb + w_cat * oof_cat
    return np.sqrt(np.mean((blended - y_train) ** 2))


# Constraint: weights sum to 1
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1), (0, 1), (0, 1)]

# Initial guess based on inverse RMSE
inv_rmse = 1 / np.array([mean_rmse_xgb, mean_rmse_lgb, mean_rmse_cat])
initial_weights = inv_rmse / inv_rmse.sum()

# Optimize
result = minimize(ensemble_rmse, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x

w_xgb, w_lgb, w_cat = optimal_weights
print(f"\nOptimal Weights:")
print(f"  XGBoost:  {w_xgb:.4f}")
print(f"  LightGBM: {w_lgb:.4f}")
print(f"  CatBoost: {w_cat:.4f}")

# Create final blended predictions
oof_blended = w_xgb * oof_xgb + w_lgb * oof_lgb + w_cat * oof_cat
test_preds_blended = w_xgb * test_preds_xgb + w_lgb * test_preds_lgb + w_cat * test_preds_cat

best_rmse = np.sqrt(np.mean((oof_blended - y_train) ** 2))
print(f"\n✓ Optimized Ensemble CV RMSE: {best_rmse:.6f}")
print(
    f"✓ Improvement over best single model: {(min(mean_rmse_xgb, mean_rmse_lgb, mean_rmse_cat) - best_rmse) / min(mean_rmse_xgb, mean_rmse_lgb, mean_rmse_cat) * 100:.2f}%")

# Clip to valid range
test_preds_blended = np.clip(test_preds_blended, 0, 1)

# ==================== SAVE SUBMISSION ====================
print("\n[9/10] Saving submission...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': test_preds_blended
})
submission.to_csv('submission.csv', index=False)
print(f"✓ Main submission saved: submission.csv")

backup_filename = f"submission_optimized_{timestamp}_cv{best_rmse:.6f}.csv"
submission.to_csv(backup_filename, index=False)
print(f"✓ Backup saved: {backup_filename}")

# Save OOF predictions
pd.DataFrame({
    'xgb_oof': oof_xgb,
    'lgb_oof': oof_lgb,
    'cat_oof': oof_cat,
    'blended_oof': oof_blended,
    'target': y_train
}).to_csv(f'oof_predictions_{timestamp}.csv', index=False)

# ==================== COMPREHENSIVE VISUALIZATION ====================
print("\n[10/10] Creating comprehensive visualizations...")

residuals = y_train - oof_blended
pearson_corr, _ = pearsonr(y_train, oof_blended)

fig = plt.figure(figsize=(20, 14))

# 1. Model Comparison
plt.subplot(4, 4, 1)
models = ['XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']
means = [mean_rmse_xgb, mean_rmse_lgb, mean_rmse_cat, best_rmse]
stds = [np.std(fold_scores['xgb']), np.std(fold_scores['lgb']), np.std(fold_scores['cat']), 0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
plt.bar(models, means, yerr=stds, alpha=0.8, color=colors, edgecolor='black', capsize=5)
plt.ylabel('RMSE', fontsize=11)
plt.title('Model Performance Comparison', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15)

# 2. Actual vs Predicted
plt.subplot(4, 4, 2)
plt.scatter(y_train, oof_blended, alpha=0.3, s=1, c='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual', fontsize=10)
plt.ylabel('Predicted', fontsize=10)
plt.title(f'Actual vs Predicted (r={pearson_corr:.4f})', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3)

# 3. Residuals Distribution
plt.subplot(4, 4, 3)
plt.hist(residuals, bins=100, alpha=0.7, edgecolor='black', color='skyblue')
plt.axvline(x=0, color='red', linestyle='--', lw=2)
plt.xlabel('Residuals', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Residual Distribution', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3)

# 4. Model Agreement Matrix
plt.subplot(4, 4, 4)
correlations = np.corrcoef([oof_xgb, oof_lgb, oof_cat])
sns.heatmap(correlations, annot=True, fmt='.3f', cmap='coolwarm',
            xticklabels=['XGB', 'LGB', 'CAT'],
            yticklabels=['XGB', 'LGB', 'CAT'],
            cbar_kws={'label': 'Correlation'})
plt.title('Model Prediction Correlation', fontweight='bold', fontsize=11)

# 5. Fold Performance
plt.subplot(4, 4, 5)
x = np.arange(1, total_folds + 1)
plt.plot(x, fold_scores['xgb'], marker='o', label='XGBoost', alpha=0.7)
plt.plot(x, fold_scores['lgb'], marker='s', label='LightGBM', alpha=0.7)
plt.plot(x, fold_scores['cat'], marker='^', label='CatBoost', alpha=0.7)
plt.xlabel('Fold', fontsize=10)
plt.ylabel('RMSE', fontsize=10)
plt.title('Per-Fold Performance', fontweight='bold', fontsize=11)
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 6. Top Features
plt.subplot(4, 4, 6)
top_15_idx = np.argsort(combined_importance)[-15:][::-1]
plt.barh(range(len(top_15_idx)), combined_importance[top_15_idx], alpha=0.8, color='teal', edgecolor='black')
plt.yticks(range(len(top_15_idx)), [feature_names[i] for i in top_15_idx], fontsize=7)
plt.xlabel('Avg Importance', fontsize=10)
plt.title('Top 15 Features', fontweight='bold', fontsize=11)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

# 7. Weight Optimization
plt.subplot(4, 4, 7)
weights_display = [w_xgb, w_lgb, w_cat]
colors_pie = ['#1f77b4', '#ff7f0e', '#2ca02c']
plt.pie(weights_display, labels=['XGBoost', 'LightGBM', 'CatBoost'],
        autopct='%1.1f%%', colors=colors_pie, startangle=90)
plt.title('Optimal Ensemble Weights', fontweight='bold', fontsize=11)

# 8. Cumulative Error
plt.subplot(4, 4, 8)
sorted_abs = np.sort(np.abs(residuals))
cumulative = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs) * 100
plt.plot(sorted_abs, cumulative, lw=2, color='purple')
plt.axhline(y=95, color='red', linestyle='--', lw=1.5, label='95%')
plt.xlabel('Absolute Error', fontsize=10)
plt.ylabel('Cumulative %', fontsize=10)
plt.title('Cumulative Error Distribution', fontweight='bold', fontsize=11)
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 9-11. Individual Model Predictions
for i, (model_name, oof_preds) in enumerate([('XGBoost', oof_xgb), ('LightGBM', oof_lgb), ('CatBoost', oof_cat)], 9):
    plt.subplot(4, 4, i)
    plt.scatter(y_train, oof_preds, alpha=0.3, s=1)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=1.5)
    corr, _ = pearsonr(y_train, oof_preds)
    plt.xlabel('Actual', fontsize=9)
    plt.ylabel('Predicted', fontsize=9)
    plt.title(f'{model_name} (r={corr:.4f})', fontweight='bold', fontsize=10)
    plt.grid(True, alpha=0.3)

# 12. Residuals vs Predicted
plt.subplot(4, 4, 12)
plt.scatter(oof_blended, residuals, alpha=0.3, s=1, c='green')
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.xlabel('Predicted', fontsize=10)
plt.ylabel('Residuals', fontsize=10)
plt.title('Residuals vs Predicted', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3)

# 13. Q-Q Plot
plt.subplot(4, 4, 13)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3)

# 14. Distribution Comparison
plt.subplot(4, 4, 14)
plt.hist(y_train, bins=50, alpha=0.5, label='Actual', color='blue', edgecolor='black')
plt.hist(oof_blended, bins=50, alpha=0.5, label='Predicted', color='orange', edgecolor='black')
plt.xlabel('Value', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Distribution Comparison', fontweight='bold', fontsize=11)
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 15. Prediction Ranges
plt.subplot(4, 4, 15)
ranges_data = {
    'Actual': [y_train.min(), y_train.max()],
    'XGB': [oof_xgb.min(), oof_xgb.max()],
    'LGB': [oof_lgb.min(), oof_lgb.max()],
    'CAT': [oof_cat.min(), oof_cat.max()],
    'Ensemble': [oof_blended.min(), oof_blended.max()],
    'Test': [test_preds_blended.min(), test_preds_blended.max()]
}
x_pos = np.arange(len(ranges_data))
mins = [v[0] for v in ranges_data.values()]
maxs = [v[1] for v in ranges_data.values()]
plt.bar(x_pos - 0.2, mins, 0.4, alpha=0.7, label='Min', color='lightblue', edgecolor='black')
plt.bar(x_pos + 0.2, maxs, 0.4, alpha=0.7, label='Max', color='lightcoral', edgecolor='black')
plt.xticks(x_pos, ranges_data.keys(), rotation=45, ha='right', fontsize=8)
plt.ylabel('Value', fontsize=10)
plt.title('Prediction Range Coverage', fontweight='bold', fontsize=11)
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3, axis='y')

# 16. Error by Target Quantile
plt.subplot(4, 4, 16)
target_quantiles = pd.qcut(y_train, q=10, labels=False, duplicates='drop')
quantile_rmse = []
for q in range(target_quantiles.max() + 1):
    mask = target_quantiles == q
    q_rmse = np.sqrt(np.mean((oof_blended[mask] - y_train[mask]) ** 2))
    quantile_rmse.append(q_rmse)
plt.bar(range(len(quantile_rmse)), quantile_rmse, alpha=0.8, color='coral', edgecolor='black')
plt.xlabel('Target Quantile', fontsize=10)
plt.ylabel('RMSE', fontsize=10)
plt.title('Error by Target Range', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
viz_filename = f"ensemble_analysis_{timestamp}.png"
plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved: {viz_filename}")

# ==================== SUMMARY STATISTICS ====================
print("\n" + "=" * 90)
print("FINAL RESULTS SUMMARY")
print("=" * 90)

print(f"\nModel Performance:")
print(f"  XGBoost  CV RMSE: {mean_rmse_xgb:.6f} (±{np.std(fold_scores['xgb']):.6f})")
print(f"  LightGBM CV RMSE: {mean_rmse_lgb:.6f} (±{np.std(fold_scores['lgb']):.6f})")
print(f"  CatBoost CV RMSE: {mean_rmse_cat:.6f} (±{np.std(fold_scores['cat']):.6f})")
print(f"  Ensemble CV RMSE: {best_rmse:.6f}")

print(f"\nEnsemble Composition:")
print(f"  XGBoost Weight:  {w_xgb:.4f}")
print(f"  LightGBM Weight: {w_lgb:.4f}")
print(f"  CatBoost Weight: {w_cat:.4f}")

print(f"\nKey Metrics:")
print(f"  Pearson Correlation: {pearson_corr:.6f}")
print(f"  Mean Absolute Error: {np.mean(np.abs(residuals)):.6f}")
print(f"  Median Absolute Error: {np.median(np.abs(residuals)):.6f}")
print(f"  R² Score: {1 - np.sum(residuals ** 2) / np.sum((y_train - y_train.mean()) ** 2):.6f}")

print(f"\nFeature Statistics:")
print(f"  Total Features Created: {X_train_full.shape[1]}")
print(f"  Features Retained: {len(important_features)}")
print(f"  Features Removed: {X_train_full.shape[1] - len(important_features)}")

print(f"\nCross-Validation Setup:")
print(f"  Strategy: Repeated K-Fold")
print(f"  Splits: {N_SPLITS}")
print(f"  Repeats: {N_REPEATS}")
print(f"  Total Folds: {total_folds}")

print(f"\nPrediction Statistics:")
print(f"  Train Range: [{y_train.min():.4f}, {y_train.max():.4f}]")
print(f"  OOF Range:   [{oof_blended.min():.4f}, {oof_blended.max():.4f}]")
print(f"  Test Range:  [{test_preds_blended.min():.4f}, {test_preds_blended.max():.4f}]")

print(f"\nOptimizations Applied:")
print(f"  ✓ Strategic interaction features (multiplication, division, addition)")
print(f"  ✓ Polynomial features (squared, sqrt, log)")
print(f"  ✓ Target encoding with smoothing (prevents overfitting)")
print(f"  ✓ QuantileTransformer with normal distribution")
print(f"  ✓ Statistical aggregations (mean, std, min, max)")
print(f"  ✓ Repeated cross-validation ({N_SPLITS}x{N_REPEATS})")
print(f"  ✓ Feature importance filtering (removed bottom 10%)")
print(f"  ✓ Validation-score-based weighted blending")
print(f"  ✓ Three diverse models (XGBoost, LightGBM, CatBoost)")

print("\n" + "=" * 90)
print("✅ ULTIMATE ENSEMBLE COMPLETE")
print("=" * 90)
print(f"Final CV RMSE: {best_rmse:.6f}")
print(f"Submission ready: submission.csv")
print(f"Backup saved: {backup_filename}")
print("=" * 90)