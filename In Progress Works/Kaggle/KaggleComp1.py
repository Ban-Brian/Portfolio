import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import time
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 80)
print("OPTIMIZED XGBOOST MODEL - ENHANCED PERFORMANCE")
print("=" * 80)

# Load data with optimized dtypes
print("\n[1/7] Loading data...")
dtype_dict = {
    'speed_limit': 'float32',
    'curvature': 'float32',
    'num_lanes': 'int8',
    'num_reported_accidents': 'int8'
}

# Update these paths to match your local file locations
TRAIN_PATH = "/Users/brianbutler/PycharmProjects/My Work/In Progress Works/Kaggle/train.csv"
TEST_PATH = "/Users/brianbutler/PycharmProjects/My Work/In Progress Works/Kaggle/test.csv"
SUBMISSION_PATH = "/Users/brianbutler/PycharmProjects/My Work/In Progress Works/Kaggle/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH, dtype=dtype_dict)
test = pd.read_csv(TEST_PATH, dtype=dtype_dict)

# Create submission template from test file
test_ids = test["id"].copy()
sub = pd.DataFrame({"id": test_ids, "accident_risk": 0.0})

train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)
print(f"✓ Train: {train.shape}, Test: {test.shape}")

# Enhanced Feature Engineering
print("\n[2/7] Creating features...")


def create_enhanced_features(train_df, test_df, target=None):
    """Optimized feature engineering with memory efficiency"""
    train_new, test_new = train_df.copy(), test_df.copy()

    # Separate numeric and categorical
    num_cols = train_new.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train_new.select_dtypes(include=["object", "bool"]).columns.tolist()

    # === 1. Ordinal Encoding (faster than LabelEncoder for multiple columns) ===
    encoders = {}
    for col in cat_cols:
        encoders[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        train_new[col] = encoders[col].fit_transform(train_new[[col]]).astype('int8')
        test_new[col] = encoders[col].transform(test_new[[col]]).astype('int8')

    # === 2. Enhanced Interactions (key patterns) ===
    # Categorical interactions
    cat_features = ['road_type', 'lighting', 'weather', 'time_of_day']
    for i, col1 in enumerate(cat_features):
        for col2 in cat_features[i + 1:]:
            # More efficient: multiply encoded values instead of string concat
            train_new[f'{col1}_{col2}'] = (train_new[col1] * 100 + train_new[col2]).astype('int16')
            test_new[f'{col1}_{col2}'] = (test_new[col1] * 100 + test_new[col2]).astype('int16')

    # Numerical interactions (most important ones)
    train_new['speed_curvature'] = (train_new['speed_limit'] * train_new['curvature']).astype('float32')
    test_new['speed_curvature'] = (test_new['speed_limit'] * test_new['curvature']).astype('float32')

    train_new['speed_lanes'] = (train_new['speed_limit'] * train_new['num_lanes']).astype('float32')
    test_new['speed_lanes'] = (test_new['speed_limit'] * test_new['num_lanes']).astype('float32')

    train_new['curv_accidents'] = (train_new['curvature'] * train_new['num_reported_accidents']).astype('float32')
    test_new['curv_accidents'] = (test_new['curvature'] * test_new['num_reported_accidents']).astype('float32')

    # === 3. Frequency Encoding (vectorized) ===
    all_cols = train_new.columns.tolist()
    for col in all_cols:
        freq = train_new[col].value_counts(normalize=True).to_dict()
        train_new[f'{col}_freq'] = train_new[col].map(freq).astype('float32')
        test_new[f'{col}_freq'] = test_new[col].map(freq).fillna(train_new[f'{col}_freq'].mean()).astype('float32')

    # === 4. Optimized Binning (only for numeric columns) ===
    for col in num_cols:
        # 5 bins
        try:
            train_new[f'{col}_bin5'], bins5 = pd.qcut(train_new[col], q=5, labels=False,
                                                      duplicates='drop', retbins=True)
            test_new[f'{col}_bin5'] = pd.cut(test_new[col], bins=bins5, labels=False,
                                             include_lowest=True).fillna(0).astype('int8')
            train_new[f'{col}_bin5'] = train_new[f'{col}_bin5'].astype('int8')
        except:
            train_new[f'{col}_bin5'] = 0
            test_new[f'{col}_bin5'] = 0

        # 10 bins
        try:
            train_new[f'{col}_bin10'], bins10 = pd.qcut(train_new[col], q=10, labels=False,
                                                        duplicates='drop', retbins=True)
            test_new[f'{col}_bin10'] = pd.cut(test_new[col], bins=bins10, labels=False,
                                              include_lowest=True).fillna(0).astype('int8')
            train_new[f'{col}_bin10'] = train_new[f'{col}_bin10'].astype('int8')
        except:
            train_new[f'{col}_bin10'] = 0
            test_new[f'{col}_bin10'] = 0

    # === 5. Target Encoding (if target provided - for train folds) ===
    if target is not None:
        target_series = pd.Series(target, index=train_new.index)
        for col in cat_features:
            target_mean = target_series.groupby(train_new[col]).mean()
            train_new[f'{col}_target'] = train_new[col].map(target_mean).astype('float32')
            test_new[f'{col}_target'] = test_new[col].map(target_mean).fillna(target_series.mean()).astype('float32')

    # === 6. Statistical aggregations by groups ===
    for col in ['road_type', 'weather', 'lighting']:
        # Mean aggregations
        for num_col in ['speed_limit', 'curvature', 'num_reported_accidents']:
            agg_mean = train_new.groupby(col)[num_col].transform('mean')
            train_new[f'{col}_{num_col}_mean'] = agg_mean.astype('float32')

            # For test, use train group means
            group_means = train_new.groupby(col)[num_col].mean().to_dict()
            test_new[f'{col}_{num_col}_mean'] = test_new[col].map(group_means).fillna(agg_mean.mean()).astype('float32')

    # === 7. Scaling (memory efficient) ===
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_new).astype('float32')
    test_scaled = scaler.transform(test_new).astype('float32')

    train_new = pd.DataFrame(train_scaled, columns=train_new.columns, index=train_new.index)
    test_new = pd.DataFrame(test_scaled, columns=test_new.columns, index=test_new.index)

    return train_new, test_new


y_train = train["accident_risk"].values
X_train_full, X_test_full = create_enhanced_features(
    train.drop("accident_risk", axis=1),
    test.copy(),
    target=y_train
)
print(f"✓ Features: {X_train_full.shape[1]}")

# Create stratification bins
y_bins = pd.qcut(y_train, q=10, labels=False, duplicates='drop')

# Optimized XGBoost parameters
print("\n[3/7] Configuring model...")
BEST_PARAMS = {
    'max_depth': 9,
    'learning_rate': 0.015,  # Slightly higher for faster convergence
    'n_estimators': 2000,  # More trees with early stopping
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.85,
    'colsample_bynode': 0.88,
    'min_child_weight': 4,
    'gamma': 0.01,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'max_delta_step': 1,
    'scale_pos_weight': 0.83,
    'max_bin': 512,
    'tree_method': 'hist',  # Changed from gpu_hist for compatibility
    'eval_metric': 'rmse',
    'random_state': 42,
    'n_jobs': -1  # Use all CPU cores
}

# Training with optimized CV
print("\n[4/7] Training model...")
FOLDS = 10  # Increased folds for better generalization
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train_full), dtype='float32')
test_preds = np.zeros(len(X_test_full), dtype='float32')
fold_scores = []
fold_details = []
feature_importance_list = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_bins), 1):
    print(f"\nFold {fold}/{FOLDS}")

    X_train_fold = X_train_full.iloc[train_idx]
    X_val = X_train_full.iloc[val_idx]
    y_train_fold = y_train[train_idx]
    y_val = y_train[val_idx]

    model = XGBRegressor(**BEST_PARAMS, early_stopping_rounds=100)

    start = time.time()
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    train_time = time.time() - start

    # Predictions
    val_preds = model.predict(X_val).astype('float32')
    test_preds += model.predict(X_test_full).astype('float32')
    oof_preds[val_idx] = val_preds

    # Metrics
    rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
    mae = np.mean(np.abs(val_preds - y_val))
    r2 = 1 - (np.sum((y_val - val_preds) ** 2) / np.sum((y_val - y_val.mean()) ** 2))

    fold_scores.append(rmse)
    fold_details.append({
        'fold': fold, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'train_time': train_time, 'best_iteration': model.best_iteration
    })
    feature_importance_list.append(model.feature_importances_)

    print(f"  RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f} | Time: {train_time:.1f}s")

test_preds /= FOLDS
cv_rmse = np.mean(fold_scores)
cv_std = np.std(fold_scores)

print(f"\n{'=' * 80}")
print(f"CV RMSE: {cv_rmse:.6f} (±{cv_std:.6f})")
print(f"{'=' * 80}")

# Post-processing: Clip predictions to valid range
test_preds = np.clip(test_preds, 0, 1)

# Save submission
print("\n[5/7] Saving submission...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub["accident_risk"] = test_preds

sub.to_csv("submission.csv", index=False)
print(f"✓ Main submission saved: submission.csv")

backup_filename = f"submission_{timestamp}_cv{cv_rmse:.6f}.csv"
sub.to_csv(backup_filename, index=False)
print(f"✓ Backup saved: {backup_filename}")

# Comprehensive Analysis
print("\n[6/7] Performing analysis...")
residuals = y_train - oof_preds
abs_residuals = np.abs(residuals)
squared_residuals = residuals ** 2

print("\n" + "=" * 80)
print("ANALYSIS RESULTS")
print("=" * 80)

# Basic Statistics
print(f"\nPerformance Metrics:")
print(f"  RMSE:           {np.sqrt(squared_residuals.mean()):.6f}")
print(f"  MAE:            {abs_residuals.mean():.6f}")
print(f"  Mean Residual:  {residuals.mean():.6f}")
print(f"  Std Residual:   {residuals.std():.6f}")

# Correlations
pearson_corr, _ = pearsonr(y_train, oof_preds)
print(f"\nCorrelations:")
print(f"  Pearson:        {pearson_corr:.6f}")

# Normality test
_, p_val = stats.normaltest(residuals)
print(f"\nResidual Normality: {'Yes' if p_val > 0.05 else 'No'} (p={p_val:.6f})")

# Fold consistency
print(f"\nFold Performance:")
print(f"  Best Fold:      {min(fold_scores):.6f}")
print(f"  Worst Fold:     {max(fold_scores):.6f}")
print(f"  Std Dev:        {cv_std:.6f}")
print(f"  Avg Time:       {np.mean([f['train_time'] for f in fold_details]):.1f}s")

# Top features
print(f"\nTop 10 Features:")
avg_importance = np.mean(feature_importance_list, axis=0)
feature_names = X_train_full.columns
top_indices = np.argsort(avg_importance)[-10:][::-1]
for idx in top_indices:
    print(f"  {feature_names[idx]:<35} {avg_importance[idx]:.4f}")

# Visualizations
print("\n[7/7] Creating visualizations...")
fig = plt.figure(figsize=(20, 12))

# 1. Actual vs Predicted
plt.subplot(3, 4, 1)
plt.scatter(y_train, oof_preds, alpha=0.3, s=2)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual');
plt.ylabel('Predicted')
plt.title(f'Actual vs Predicted (r={pearson_corr:.4f})', fontweight='bold')
plt.grid(True, alpha=0.3)

# 2. Residuals Distribution
plt.subplot(3, 4, 2)
plt.hist(residuals, bins=100, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.xlabel('Residuals');
plt.ylabel('Frequency')
plt.title('Residual Distribution', fontweight='bold')
plt.grid(True, alpha=0.3)

# 3. Residuals vs Predicted
plt.subplot(3, 4, 3)
plt.scatter(oof_preds, residuals, alpha=0.3, s=2)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted');
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted', fontweight='bold')
plt.grid(True, alpha=0.3)

# 4. Q-Q Plot
plt.subplot(3, 4, 4)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot', fontweight='bold')
plt.grid(True, alpha=0.3)

# 5. Feature Importance
plt.subplot(3, 4, 5)
top_features = 15
imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': avg_importance
}).sort_values('importance', ascending=False).head(top_features)
plt.barh(range(len(imp_df)), imp_df['importance'], alpha=0.7, edgecolor='black')
plt.yticks(range(len(imp_df)), imp_df['feature'], fontsize=8)
plt.xlabel('Importance')
plt.title(f'Top {top_features} Features', fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

# 6. Fold Performance
plt.subplot(3, 4, 6)
fold_nums = [f['fold'] for f in fold_details]
fold_rmses = [f['rmse'] for f in fold_details]
plt.bar(fold_nums, fold_rmses, alpha=0.7, edgecolor='black')
plt.axhline(y=cv_rmse, color='r', linestyle='--', lw=2, label=f'Mean: {cv_rmse:.6f}')
plt.xlabel('Fold');
plt.ylabel('RMSE')
plt.title('RMSE by Fold', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 7. Cumulative Error
plt.subplot(3, 4, 7)
sorted_abs = np.sort(abs_residuals)
cumulative = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs) * 100
plt.plot(sorted_abs, cumulative, lw=2)
plt.axhline(y=95, color='r', linestyle='--', label='95%')
plt.xlabel('Absolute Error');
plt.ylabel('Cumulative %')
plt.title('Cumulative Error Distribution', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 8. RMSE by Target Bins
plt.subplot(3, 4, 8)
target_bins = pd.qcut(y_train, q=15, labels=False, duplicates='drop')
bin_rmse = [np.sqrt(squared_residuals[target_bins == i].mean())
            for i in range(target_bins.max() + 1)]
plt.bar(range(len(bin_rmse)), bin_rmse, alpha=0.7, edgecolor='black')
plt.xlabel('Target Bins');
plt.ylabel('RMSE')
plt.title('RMSE by Target Bins', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 9. Distribution Comparison
plt.subplot(3, 4, 9)
plt.hist(y_train, bins=50, alpha=0.5, label='Actual', edgecolor='black')
plt.hist(oof_preds, bins=50, alpha=0.5, label='Predicted', edgecolor='black')
plt.xlabel('Value');
plt.ylabel('Frequency')
plt.title('Distribution Comparison', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 10. Training Time
plt.subplot(3, 4, 10)
plt.plot(fold_nums, [f['train_time'] for f in fold_details],
         marker='o', lw=2, markersize=8)
plt.xlabel('Fold');
plt.ylabel('Time (seconds)')
plt.title('Training Time per Fold', fontweight='bold')
plt.grid(True, alpha=0.3)

# 11. R² by Fold
plt.subplot(3, 4, 11)
plt.bar(fold_nums, [f['r2'] for f in fold_details],
        alpha=0.7, edgecolor='black', color='purple')
plt.axhline(y=np.mean([f['r2'] for f in fold_details]),
            color='r', linestyle='--', lw=2)
plt.xlabel('Fold');
plt.ylabel('R²')
plt.title('R² by Fold', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 12. Prediction Range
plt.subplot(3, 4, 12)
ranges = {
    'Actual': [y_train.min(), y_train.max()],
    'OOF': [oof_preds.min(), oof_preds.max()],
    'Test': [test_preds.min(), test_preds.max()]
}
x_pos = np.arange(len(ranges))
mins = [v[0] for v in ranges.values()]
maxs = [v[1] for v in ranges.values()]
plt.bar(x_pos - 0.2, mins, 0.4, alpha=0.7, label='Min', edgecolor='black')
plt.bar(x_pos + 0.2, maxs, 0.4, alpha=0.7, label='Max', edgecolor='black')
plt.xticks(x_pos, ranges.keys())
plt.ylabel('Value')
plt.title('Prediction Range Coverage', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
viz_filename = f"analysis_{timestamp}.png"
plt.savefig(viz_filename, dpi=120, bbox_inches='tight')
print(f"✓ Visualization saved: {viz_filename}")

print("\n" + "=" * 80)
print("✅ OPTIMIZATION COMPLETE")
print("=" * 80)
print(f"Final CV RMSE: {cv_rmse:.6f}")
print(f"Submission ready: submission.csv")
print("=" * 80)