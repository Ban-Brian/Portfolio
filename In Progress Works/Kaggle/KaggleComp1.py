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
print("DIRECT AVERAGING ENSEMBLE - TARGET 0.05537 RMSE")
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

    # Meta risk feature
    def meta_risk(X):
        return (0.35 * X["curvature"] +
                0.25 * (X["lighting"] == "night").astype(int) +
                0.15 * (X["weather"] != "clear").astype(int) +
                0.15 * (X["speed_limit"] >= 60).astype(int) +
                0.10 * (X["num_reported_accidents"] > 2).astype(int))

    train_new['meta'] = meta_risk(train_new)
    test_new['meta'] = meta_risk(test_new)

    # Polynomial features
    for power in [2, 3]:
        train_new[f'speed_p{power}'] = train_new['speed_limit'] ** power
        test_new[f'speed_p{power}'] = test_new['speed_limit'] ** power
        train_new[f'curvature_p{power}'] = train_new['curvature'] ** power
        test_new[f'curvature_p{power}'] = test_new['curvature'] ** power

    # Transforms
    train_new['log_curvature'] = np.log1p(train_new['curvature'])
    test_new['log_curvature'] = np.log1p(test_new['curvature'])
    train_new['log_speed'] = np.log1p(train_new['speed_limit'])
    test_new['log_speed'] = np.log1p(test_new['speed_limit'])
    train_new['sqrt_curvature'] = np.sqrt(train_new['curvature'])
    test_new['sqrt_curvature'] = np.sqrt(test_new['curvature'])
    train_new['sqrt_speed'] = np.sqrt(train_new['speed_limit'])
    test_new['sqrt_speed'] = np.sqrt(test_new['speed_limit'])

    # Critical interactions
    interactions = [
        ('speed_limit', 'curvature'),
        ('speed_limit', 'num_lanes'),
        ('speed_limit', 'num_reported_accidents'),
        ('curvature', 'num_reported_accidents'),
        ('curvature', 'num_lanes'),
        ('num_lanes', 'num_reported_accidents'),
        ('meta', 'curvature'),
        ('meta', 'speed_limit')
    ]

    for col1, col2 in interactions:
        train_new[f'{col1}_x_{col2}'] = train_new[col1] * train_new[col2]
        test_new[f'{col1}_x_{col2}'] = test_new[col1] * test_new[col2]

    # Three-way
    train_new['speed_curv_lanes'] = train_new['speed_limit'] * train_new['curvature'] * train_new['num_lanes']
    test_new['speed_curv_lanes'] = test_new['speed_limit'] * test_new['curvature'] * test_new['num_lanes']

    # Ratios
    train_new['speed_per_lane'] = train_new['speed_limit'] / (train_new['num_lanes'] + 1)
    test_new['speed_per_lane'] = test_new['speed_limit'] / (test_new['num_lanes'] + 1)
    train_new['acc_per_lane'] = train_new['num_reported_accidents'] / (train_new['num_lanes'] + 1)
    test_new['acc_per_lane'] = test_new['num_reported_accidents'] / (test_new['num_lanes'] + 1)

    # Categorical combos
    train_new['weather_lighting'] = train_new['weather'].astype(str) + '_' + train_new['lighting'].astype(str)
    test_new['weather_lighting'] = test_new['weather'].astype(str) + '_' + test_new['lighting'].astype(str)
    train_new['road_weather'] = train_new['road_type'].astype(str) + '_' + train_new['weather'].astype(str)
    test_new['road_weather'] = test_new['road_type'].astype(str) + '_' + test_new['weather'].astype(str)

    # Flags
    train_new['high_risk'] = ((train_new['curvature'] > 0.7) & (train_new['speed_limit'] >= 60)).astype(int)
    test_new['high_risk'] = ((test_new['curvature'] > 0.7) & (test_new['speed_limit'] >= 60)).astype(int)
    train_new['night_bad'] = ((train_new['lighting'] == 'night') & (train_new['weather'] != 'clear')).astype(int)
    test_new['night_bad'] = ((test_new['lighting'] == 'night') & (test_new['weather'] != 'clear')).astype(int)

    # Risk score
    train_new['risk_score'] = (train_new['curvature'] * 0.4 +
                               (train_new['speed_limit'] / 70) * 0.35 +
                               (train_new['num_reported_accidents'] / 7) * 0.25)
    test_new['risk_score'] = (test_new['curvature'] * 0.4 +
                              (test_new['speed_limit'] / 70) * 0.35 +
                              (test_new['num_reported_accidents'] / 7) * 0.25)

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

# ==================== DIRECT CV ENSEMBLE ====================
print("\n[4/5] Training models with direct averaging...")

n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

# LightGBM
param_lgb = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.008,
    'num_leaves': 120,
    'max_depth': 13,
    'min_child_samples': 12,
    'subsample': 0.75,
    'colsample_bytree': 0.82,
    'reg_alpha': 0.05,
    'reg_lambda': 0.7,
    'min_split_gain': 0.005,
    'verbosity': -1,
    'random_state': 42
}

print("\nLightGBM:")
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_tr, y_tr)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)

    model = lgb.train(param_lgb, train_data, num_boost_round=3500,
                      valid_sets=[val_data], callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])

    oof_lgb[val_idx] = model.predict(X_val)
    test_lgb += model.predict(X_test) / n_folds

    print(f"  Fold {fold}: {np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx])):.6f}")

print(f"  CV: {np.sqrt(mean_squared_error(y, oof_lgb)):.6f}")

# CatBoost
param_cat = {
    'loss_function': 'RMSE',
    'learning_rate': 0.025,
    'depth': 9,
    'l2_leaf_reg': 3.5,
    'bagging_temperature': 0.12,
    'random_strength': 0.3,
    'min_data_in_leaf': 12,
    'rsm': 0.85,
    'verbose': False,
    'random_seed': 42
}

print("\nCatBoost:")
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostRegressor(**param_cat, iterations=3000, early_stopping_rounds=200)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

    oof_cat[val_idx] = model.predict(X_val)
    test_cat += model.predict(X_test) / n_folds

    print(f"  Fold {fold}: {np.sqrt(mean_squared_error(y_val, oof_cat[val_idx])):.6f}")

print(f"  CV: {np.sqrt(mean_squared_error(y, oof_cat)):.6f}")

# XGBoost
param_xgb = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.008,
    'max_depth': 8,
    'min_child_weight': 2.0,
    'gamma': 0.05,
    'subsample': 0.76,
    'colsample_bytree': 0.78,
    'reg_alpha': 0.05,
    'reg_lambda': 0.7,
    'random_state': 42,
    'n_jobs': -1
}

print("\nXGBoost:")
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(**param_xgb, n_estimators=3000, early_stopping_rounds=200)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)

    oof_xgb[val_idx] = model.predict(X_val)
    test_xgb += model.predict(X_test) / n_folds

    print(f"  Fold {fold}: {np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx])):.6f}")

print(f"  CV: {np.sqrt(mean_squared_error(y, oof_xgb)):.6f}")

# ==================== WEIGHTED AVERAGING ====================
print("\n[5/5] Finding optimal weights...")

from scipy.optimize import minimize


def objective(weights):
    w = np.abs(weights) / np.sum(np.abs(weights))
    blend = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_xgb
    return np.sqrt(mean_squared_error(y, blend))


# Try multiple starting points
best_result = None
best_score = float('inf')

starting_points = [
    [1 / 3, 1 / 3, 1 / 3],
    [0.4, 0.3, 0.3],
    [0.3, 0.4, 0.3],
    [0.3, 0.3, 0.4],
    [0.5, 0.25, 0.25],
]

for start in starting_points:
    result = minimize(objective, start, method='Powell')
    if result.fun < best_score:
        best_score = result.fun
        best_result = result

weights = np.abs(best_result.x) / np.sum(np.abs(best_result.x))

print(f"\nOptimal Weights:")
print(f"  LightGBM: {weights[0]:.4f}")
print(f"  CatBoost: {weights[1]:.4f}")
print(f"  XGBoost:  {weights[2]:.4f}")

# Final blend
oof_blend = weights[0] * oof_lgb + weights[1] * oof_cat + weights[2] * oof_xgb
test_blend = weights[0] * test_lgb + weights[1] * test_cat + weights[2] * test_xgb

final_predictions = np.clip(test_blend, 0, 1)
final_rmse = np.sqrt(mean_squared_error(y, oof_blend))

print(f"\nWeighted Blend CV RMSE: {final_rmse:.6f}")

# Also try simple average
simple_avg = (oof_lgb + oof_cat + oof_xgb) / 3
simple_rmse = np.sqrt(mean_squared_error(y, simple_avg))
print(f"Simple Average CV RMSE: {simple_rmse:.6f}")

# Use whichever is better
if simple_rmse < final_rmse:
    print("\n✓ Using simple average (better performance)")
    final_predictions = np.clip((test_lgb + test_cat + test_xgb) / 3, 0, 1)
    final_rmse = simple_rmse
    oof_blend = simple_avg
else:
    print("\n✓ Using weighted average (better performance)")

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
print(f"  LightGBM: {np.sqrt(mean_squared_error(y, oof_lgb)):.6f}")
print(f"  CatBoost: {np.sqrt(mean_squared_error(y, oof_cat)):.6f}")
print(f"  XGBoost:  {np.sqrt(mean_squared_error(y, oof_xgb)):.6f}")
print(f"\nPrediction Stats:")
print(f"  Train: [{y.min():.4f}, {y.max():.4f}]")
print(f"  OOF:   [{oof_blend.min():.4f}, {oof_blend.max():.4f}]")
print(f"  Test:  [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
print(f"\nApproach:")
print(f"  ✓ Direct CV averaging (no stacking)")
print(f"  ✓ Conservative hyperparameters")
print(f"  ✓ 10-fold cross-validation")
print(f"  ✓ Simple weighted/average blend")
print("=" * 80)

if final_rmse <= 0.05537:
    print("TARGET ACHIEVED!")
else:a
    print(f"Gap: {(final_rmse - 0.05537):.6f}")

print("=" * 80)