import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("PROVEN WINNING APPROACH - EXACT NOTEBOOK IMPLEMENTATION")
print("=" * 80)

# ==================== DATA LOADING ====================
print("\n[1/5] Loading data...")
TRAIN_PATH = "/Users/brianbutler/PycharmProjects/My Work/In Progress Works/Kaggle/train.csv"
TEST_PATH = "/Users/brianbutler/PycharmProjects/My Work/In Progress Works/Kaggle/test.csv"

df = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

# Drop ID and duplicates
df = df.drop("id", axis=1)
df_test_ids = df_test["id"].copy()
df_test = df_test.drop("id", axis=1)
df = df.drop_duplicates()

print(f"✓ Train: {df.shape}, Test: {df_test.shape}")

# ==================== FEATURE ENGINEERING (FROM NOTEBOOK) ====================
print("\n[2/5] Creating features (exact from notebook)...")

# Danger Condition
df['weather_lighting'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
df_test['weather_lighting'] = df_test['weather'].astype(str) + '_' + df_test['lighting'].astype(str)

# Polynomial features
df['speed_squared'] = df['speed_limit'] ** 2
df_test['speed_squared'] = df_test['speed_limit'] ** 2

df['curvature_squared'] = df['curvature'] ** 2
df_test['curvature_squared'] = df_test['curvature'] ** 2

df['log_curvature'] = np.log1p(df['curvature'])
df_test['log_curvature'] = np.log1p(df_test['curvature'])


# Meta feature (Very good FE from notebook)
def f(X):
    return \
            0.3 * X["curvature"] + \
            0.2 * (X["lighting"] == "night").astype(int) + \
            0.1 * (X["weather"] != "clear").astype(int) + \
            0.2 * (X["speed_limit"] >= 60).astype(int) + \
            0.1 * (X["num_reported_accidents"] > 2).astype(int)


df['meta'] = f(df)
df_test['meta'] = f(df_test)

print(f"✓ Created features: {df.shape[1]} columns")

# ==================== ENCODING ====================
print("\n[3/5] Encoding...")

# Convert Boolean columns
bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]
for col in bool_cols:
    df[col] = df[col].astype(int)
    df_test[col] = df_test[col].astype(int)

# Label encoding categorical features
le = LabelEncoder()
cate_cols = df.select_dtypes(exclude="number").columns.tolist()
if 'accident_risk' in cate_cols:
    cate_cols.remove('accident_risk')

for col in cate_cols:
    df[col] = le.fit_transform(df[col])
    df_test[col] = le.transform(df_test[col])

# Prepare X, y
X = df.drop('accident_risk', axis=1)
y = df['accident_risk']
X_test = df_test

print(f"✓ Final shape - X: {X.shape}, Test: {X_test.shape}")

# ==================== MODELS (EXACT FROM NOTEBOOK) ====================
print("\n[4/5] Training with exact notebook parameters...")

# LightGBM best params from notebook
param_lgb = {
    'n_estimators': 2700,
    'learning_rate': 0.01,
    'num_leaves': 99,
    'max_depth': 13,
    'min_child_samples': 10,
    'min_child_weight': 0.002,
    'subsample': 0.60,
    'subsample_freq': 1,
    'colsample_bytree': 0.83,
    'reg_alpha': 0.01,
    'reg_lambda': 0.70,
    'min_split_gain': 0.004,
    'feature_fraction': 0.9,
}

# CatBoost best params from notebook
param_cat = {
    'bagging_temperature': 0.20,
    'border_count': 178,
    'depth': 8,
    'iterations': 1600,
    'l2_leaf_reg': 4,
    'learning_rate': 0.04,
    'random_strength': 0.32,
}

# Initialize models
cat_model = CatBoostRegressor(**param_cat,
                              loss_function='RMSE',
                              random_seed=42,
                              verbose=False,
                              thread_count=-1)

lgb_model = lgb.LGBMRegressor(**param_lgb,
                              objective='regression',
                              metric='rmse',
                              boosting_type='gbdt',
                              random_state=42,
                              n_jobs=-1,
                              verbose=-1)


# Stacking function (from notebook)
def create_meta_features(models, X_train, X_test, y_train, n_splits=5):
    """Create out-of-fold predictions for training meta-model"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    meta_train = np.zeros((len(X_train), len(models)))
    meta_test = np.zeros((len(X_test), len(models)))

    for i, model in enumerate(models):
        print(f"  Processing model {i + 1}/{len(models)}...")
        test_preds = np.zeros(len(X_test))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]

            # Train model on fold
            model.fit(X_tr, y_tr)

            # Get out-of-fold predictions
            meta_train[val_idx, i] = model.predict(X_val)

            # Get test predictions for this fold
            test_preds += model.predict(X_test) / n_splits

        meta_test[:, i] = test_preds

    return meta_train, meta_test


# Create meta features
print("\n  Creating meta features with stacking...")
models = [cat_model, lgb_model]
meta_train, meta_test = create_meta_features(models, X, X_test, y)

print(f"✓ Meta features shape: train={meta_train.shape}, test={meta_test.shape}")

# Train meta-model (Ridge from notebook)
print("\n  Training meta-model (Ridge)...")
meta_model = Ridge(alpha=0.1)
meta_model.fit(meta_train, y)

train_rmse = np.sqrt(mean_squared_error(y, meta_model.predict(meta_train)))
print(f"✓ Meta-model train RMSE: {train_rmse:.6f}")

# Final predictions
final_predictions = meta_model.predict(meta_test)
final_predictions = np.clip(final_predictions, 0, 1)

# ==================== SAVE ====================
print("\n[5/5] Saving submission...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

submission = pd.DataFrame({
    'id': df_test_ids,
    'accident_risk': final_predictions
})
submission.to_csv('submission.csv', index=False)

backup = f"submission_notebook_{timestamp}.csv"
submission.to_csv(backup, index=False)

print(f"✓ Main submission: submission.csv")
print(f"✓ Backup: {backup}")

# ==================== RESULTS ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Train RMSE:     {train_rmse:.6f}")
print(f"Expected CV:    ~0.0559-0.0560 (from notebook)")
print(f"\nModel Weights:")
print(f"  CatBoost:  {meta_model.coef_[0]:.4f}")
print(f"  LightGBM:  {meta_model.coef_[1]:.4f}")
print(f"  Intercept: {meta_model.intercept_:.4f}")
print(f"\nPrediction Range:")
print(f"  Min: {final_predictions.min():.4f}")
print(f"  Max: {final_predictions.max():.4f}")
print("=" * 80)
print("✅ COMPLETE - Using exact winning approach from notebook")
print("=" * 80)

# Quick validation check
print("\nValidation Check:")
print(f"  Features match notebook: {'✓' if X.shape[1] == 18 else '✗'}")
print(f"  Using Ridge meta-model: ✓")
print(f"  Using CAT + LGB: ✓")
print(f"  5-fold stacking: ✓")
print(f"  Predictions in [0,1]: ✓")
print("\nThis is the EXACT approach that achieved 0.0559 RMSE in the notebook!")
print("If score is different, it may be due to:")
print("  - Different train/test split from competition")
print("  - Random seed differences")
print("  - Version differences in libraries")