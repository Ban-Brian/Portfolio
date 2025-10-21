import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("REFINED MODEL - TARGET 0.05537 RMSE")
print("=" * 80)

# ==================== DATA LOADING ====================
print("\n[1/5] Loading data...")
TRAIN_PATH = "/Users/brianbutler/PycharmProjects/My Work/In Progress Works/Kaggle/train.csv"
TEST_PATH = "/Users/brianbutler/PycharmProjects/My Work/In Progress Works/Kaggle/test.csv"

df = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

df = df.drop("id", axis=1)
df_test_ids = df_test["id"].copy()
df_test = df_test.drop("id", axis=1)
df = df.drop_duplicates()

print(f"✓ Train: {df.shape}, Test: {df_test.shape}")

# ==================== FOCUSED FEATURE ENGINEERING ====================
print("\n[2/5] Creating features...")


def create_features(train_df, test_df):
    train_new, test_new = train_df.copy(), test_df.copy()

    # Polynomial features
    for col in ['speed_limit', 'curvature']:
        train_new[f'{col}_sq'] = train_new[col] ** 2
        test_new[f'{col}_sq'] = test_new[col] ** 2
        train_new[f'log_{col}'] = np.log1p(train_new[col])
        test_new[f'log_{col}'] = np.log1p(test_new[col])

    # Key interactions
    train_new['speed_curv'] = train_new['speed_limit'] * train_new['curvature']
    test_new['speed_curv'] = test_new['speed_limit'] * test_new['curvature']

    train_new['speed_lanes'] = train_new['speed_limit'] * train_new['num_lanes']
    test_new['speed_lanes'] = test_new['speed_limit'] * test_new['num_lanes']

    train_new['curv_acc'] = train_new['curvature'] * train_new['num_reported_accidents']
    test_new['curv_acc'] = test_new['curvature'] * test_new['num_reported_accidents']

    # Ratios
    train_new['speed_per_lane'] = train_new['speed_limit'] / (train_new['num_lanes'] + 1)
    test_new['speed_per_lane'] = test_new['speed_limit'] / (test_new['num_lanes'] + 1)

    train_new['acc_per_lane'] = train_new['num_reported_accidents'] / (train_new['num_lanes'] + 1)
    test_new['acc_per_lane'] = test_new['num_reported_accidents'] / (test_new['num_lanes'] + 1)

    # Risk score
    train_new['risk'] = (train_new['curvature'] * 0.4 +
                         (train_new['speed_limit'] / 70) * 0.35 +
                         (train_new['num_reported_accidents'] / 7) * 0.25)
    test_new['risk'] = (test_new['curvature'] * 0.4 +
                        (test_new['speed_limit'] / 70) * 0.35 +
                        (test_new['num_reported_accidents'] / 7) * 0.25)

    # Categorical combinations
    train_new['weather_light'] = train_new['weather'].astype(str) + '_' + train_new['lighting'].astype(str)
    test_new['weather_light'] = test_new['weather'].astype(str) + '_' + test_new['lighting'].astype(str)

    # Flags
    train_new['high_risk'] = ((train_new['curvature'] > 0.7) & (train_new['speed_limit'] >= 60)).astype(int)
    test_new['high_risk'] = ((test_new['curvature'] > 0.7) & (test_new['speed_limit'] >= 60)).astype(int)

    return train_new, test_new


df, df_test = create_features(df, df_test)
print(f"✓ Features: {df.shape[1]} columns")

# ==================== ENCODING ====================
print("\n[3/5] Encoding...")

bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]
for col in bool_cols:
    df[col] = df[col].astype(int)
    df_test[col] = df_test[col].astype(int)

le = LabelEncoder()
cate_cols = df.select_dtypes(exclude="number").columns.tolist()
if 'accident_risk' in cate_cols:
    cate_cols.remove('accident_risk')

for col in cate_cols:
    df[col] = le.fit_transform(df[col])
    df_test[col] = le.transform(df_test[col])

X = df.drop('accident_risk', axis=1)
y = df['accident_risk']
X_test = df_test

print(f"✓ X: {X.shape}, Test: {X_test.shape}")

# ==================== CROSS-VALIDATED ENSEMBLE ====================
print("\n[4/5] Training with 10-fold CV...")

n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Storage for OOF and test predictions
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

# LightGBM params - conservative
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 100,
    'max_depth': 12,
    'min_child_samples': 18,
    'subsample': 0.75,
    'colsample_bytree': 0.80,
    'reg_alpha': 0.08,
    'reg_lambda': 0.9,
    'min_split_gain': 0.008,
    'verbosity': -1,
    'random_state': 42
}

# CatBoost params - conservative
cat_params = {
    'loss_function': 'RMSE',
    'learning_rate': 0.03,
    'depth': 8,
    'l2_leaf_reg': 4.5,
    'bagging_temperature': 0.18,
    'random_strength': 0.45,
    'min_data_in_leaf': 18,
    'rsm': 0.82,
    'verbose': False,
    'random_seed': 42
}

# XGBoost params - conservative
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 8,
    'min_child_weight': 2.5,
    'gamma': 0.08,
    'subsample': 0.77,
    'colsample_bytree': 0.77,
    'reg_alpha': 0.08,
    'reg_lambda': 0.9,
    'random_state': 42,
    'n_jobs': -1
}

print("\nLightGBM:")
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_tr, y_tr)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)

    model = lgb.train(lgb_params, train_data, num_boost_round=3000,
                      valid_sets=[val_data], callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])

    oof_lgb[val_idx] = model.predict(X_val)
    test_lgb += model.predict(X_test) / n_folds

    fold_rmse = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    print(f"  Fold {fold}: {fold_rmse:.6f}")

lgb_cv_rmse = np.sqrt(mean_squared_error(y, oof_lgb))
print(f"  CV RMSE: {lgb_cv_rmse:.6f}")

print("\nCatBoost:")
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostRegressor(**cat_params, iterations=2500, early_stopping_rounds=150)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

    oof_cat[val_idx] = model.predict(X_val)
    test_cat += model.predict(X_test) / n_folds

    fold_rmse = np.sqrt(mean_squared_error(y_val, oof_cat[val_idx]))
    print(f"  Fold {fold}: {fold_rmse:.6f}")

cat_cv_rmse = np.sqrt(mean_squared_error(y, oof_cat))
print(f"  CV RMSE: {cat_cv_rmse:.6f}")

print("\nXGBoost:")
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(**xgb_params, n_estimators=2500, early_stopping_rounds=150)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)

    oof_xgb[val_idx] = model.predict(X_val)
    test_xgb += model.predict(X_test) / n_folds

    fold_rmse = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
    print(f"  Fold {fold}: {fold_rmse:.6f}")

xgb_cv_rmse = np.sqrt(mean_squared_error(y, oof_xgb))
print(f"  CV RMSE: {xgb_cv_rmse:.6f}")

# ==================== OPTIMAL BLEND ====================
print("\n[5/5] Optimizing blend weights...")

from scipy.optimize import minimize


def blend_rmse(weights):
    w = np.abs(weights) / np.sum(np.abs(weights))
    blend = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
    return np.sqrt(mean_squared_error(y, blend))


result = minimize(blend_rmse, [1 / 3, 1 / 3, 1 / 3], method='Nelder-Mead')
weights = np.abs(result.x) / np.sum(np.abs(result.x))

print(f"\nOptimal Weights:")
print(f"  LightGBM: {weights[0]:.4f}")
print(f"  CatBoost: {weights[1]:.4f}")
print(f"  XGBoost:  {weights[2]:.4f}")

# Final predictions
oof_blend = weights[0] * oof_lgb + weights[1] * oof_cat + weights[2] * oof_xgb
test_blend = weights[0] * test_lgb + weights[1] * test_cat + weights[2] * test_xgb

final_predictions = np.clip(test_blend, 0, 1)
final_rmse = np.sqrt(mean_squared_error(y, oof_blend))

print(f"\nBlended CV RMSE: {final_rmse:.6f}")

# ==================== SAVE ====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

submission = pd.DataFrame({
    'id': df_test_ids,
    'accident_risk': final_predictions
})
submission.to_csv('submission.csv', index=False)
backup = f"submission_{timestamp}_rmse{final_rmse:.6f}.csv"
submission.to_csv(backup, index=False)

print(f"\n✓ Saved: submission.csv")

# ==================== RESULTS ====================
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(f"CV RMSE:        {final_rmse:.6f}")
print(f"Target:         0.05537")
print(f"Gap:            {(final_rmse - 0.05537):.6f}")
print(f"\nIndividual Models:")
print(f"  LightGBM: {lgb_cv_rmse:.6f}")
print(f"  CatBoost: {cat_cv_rmse:.6f}")
print(f"  XGBoost:  {xgb_cv_rmse:.6f}")
print(f"\nPrediction Stats:")
print(f"  Train: [{y.min():.4f}, {y.max():.4f}]")
print(f"  OOF:   [{oof_blend.min():.4f}, {oof_blend.max():.4f}]")
print(f"  Test:  [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
print("=" * 80)

if final_rmse <= 0.05537:
    print("TARGET ACHIEVED!")
elif final_rmse <= 0.0556:
    print("Very close! Within 0.0002 of target.")
else:
    print(f"Gap: {(final_rmse - 0.05537):.6f}")

print("=" * 80)