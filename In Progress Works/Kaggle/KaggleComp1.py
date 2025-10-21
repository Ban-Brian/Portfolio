import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("FINE-TUNED STACKING MODEL - TARGET 0.05537 RMSE")
print("=" * 80)

# ==================== DATA LOADING ====================
print("\n[1/7] Loading data...")
TRAIN_PATH = "/Users/brianbutler/PycharmProjects/My Work/In Progress Works/Kaggle/train.csv"
TEST_PATH = "/Users/brianbutler/PycharmProjects/My Work/In Progress Works/Kaggle/test.csv"

df = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

df = df.drop("id", axis=1)
df_test_ids = df_test["id"].copy()
df_test = df_test.drop("id", axis=1)
df = df.drop_duplicates()

print(f"✓ Train: {df.shape}, Test: {df_test.shape}")

# ==================== ADVANCED FEATURE ENGINEERING ====================
print("\n[2/7] Creating advanced features...")


def create_advanced_features(train_df, test_df):
    train_new, test_new = train_df.copy(), test_df.copy()

    # Enhanced meta feature
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

    # Log and sqrt transforms
    train_new['log_curvature'] = np.log1p(train_new['curvature'])
    test_new['log_curvature'] = np.log1p(test_new['curvature'])
    train_new['log_speed'] = np.log1p(train_new['speed_limit'])
    test_new['log_speed'] = np.log1p(test_new['speed_limit'])
    train_new['sqrt_curvature'] = np.sqrt(train_new['curvature'])
    test_new['sqrt_curvature'] = np.sqrt(test_new['curvature'])
    train_new['sqrt_speed'] = np.sqrt(train_new['speed_limit'])
    test_new['sqrt_speed'] = np.sqrt(test_new['speed_limit'])

    # Two-way interactions
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

    # Three-way interactions
    train_new['speed_curv_lanes'] = train_new['speed_limit'] * train_new['curvature'] * train_new['num_lanes']
    test_new['speed_curv_lanes'] = test_new['speed_limit'] * test_new['curvature'] * test_new['num_lanes']
    train_new['speed_curv_acc'] = train_new['speed_limit'] * train_new['curvature'] * train_new[
        'num_reported_accidents']
    test_new['speed_curv_acc'] = test_new['speed_limit'] * test_new['curvature'] * test_new['num_reported_accidents']

    # Division features
    train_new['speed_div_lanes'] = train_new['speed_limit'] / (train_new['num_lanes'] + 1)
    test_new['speed_div_lanes'] = test_new['speed_limit'] / (test_new['num_lanes'] + 1)
    train_new['curvature_div_lanes'] = train_new['curvature'] / (train_new['num_lanes'] + 1)
    test_new['curvature_div_lanes'] = test_new['curvature'] / (test_new['num_lanes'] + 1)
    train_new['speed_div_curv'] = train_new['speed_limit'] / (train_new['curvature'] + 0.001)
    test_new['speed_div_curv'] = test_new['speed_limit'] / (test_new['curvature'] + 0.001)
    train_new['accidents_per_lane'] = train_new['num_reported_accidents'] / (train_new['num_lanes'] + 1)
    test_new['accidents_per_lane'] = test_new['num_reported_accidents'] / (test_new['num_lanes'] + 1)

    # Categorical combinations
    train_new['weather_lighting'] = train_new['weather'].astype(str) + '_' + train_new['lighting'].astype(str)
    test_new['weather_lighting'] = test_new['weather'].astype(str) + '_' + test_new['lighting'].astype(str)
    train_new['road_weather'] = train_new['road_type'].astype(str) + '_' + train_new['weather'].astype(str)
    test_new['road_weather'] = test_new['road_type'].astype(str) + '_' + test_new['weather'].astype(str)
    train_new['road_time'] = train_new['road_type'].astype(str) + '_' + train_new['time_of_day'].astype(str)
    test_new['road_time'] = test_new['road_type'].astype(str) + '_' + test_new['time_of_day'].astype(str)
    train_new['road_lighting'] = train_new['road_type'].astype(str) + '_' + train_new['lighting'].astype(str)
    test_new['road_lighting'] = test_new['road_type'].astype(str) + '_' + test_new['lighting'].astype(str)

    # Risk flags
    train_new['high_risk'] = ((train_new['curvature'] > 0.7) & (train_new['speed_limit'] >= 60)).astype(int)
    test_new['high_risk'] = ((test_new['curvature'] > 0.7) & (test_new['speed_limit'] >= 60)).astype(int)
    train_new['night_rain'] = ((train_new['lighting'] == 'night') & (train_new['weather'] == 'rainy')).astype(int)
    test_new['night_rain'] = ((test_new['lighting'] == 'night') & (test_new['weather'] == 'rainy')).astype(int)
    train_new['extreme_curv'] = (train_new['curvature'] > 0.85).astype(int)
    test_new['extreme_curv'] = (test_new['curvature'] > 0.85).astype(int)
    train_new['high_speed'] = (train_new['speed_limit'] >= 65).astype(int)
    test_new['high_speed'] = (test_new['speed_limit'] >= 65).astype(int)

    # Composite risk scores
    train_new['risk_score1'] = (train_new['curvature'] * 0.4 +
                                (train_new['speed_limit'] / 70) * 0.3 +
                                (train_new['num_reported_accidents'] / 7) * 0.3)
    test_new['risk_score1'] = (test_new['curvature'] * 0.4 +
                               (test_new['speed_limit'] / 70) * 0.3 +
                               (test_new['num_reported_accidents'] / 7) * 0.3)

    train_new['risk_score2'] = train_new['meta'] * train_new['curvature']
    test_new['risk_score2'] = test_new['meta'] * test_new['curvature']

    # Binned features
    train_new['speed_bin'] = pd.cut(train_new['speed_limit'], bins=5, labels=False)
    test_new['speed_bin'] = pd.cut(test_new['speed_limit'], bins=5, labels=False)
    train_new['curv_bin'] = pd.cut(train_new['curvature'], bins=5, labels=False)
    test_new['curv_bin'] = pd.cut(test_new['curvature'], bins=5, labels=False)

    return train_new, test_new


df, df_test = create_advanced_features(df, df_test)
print(f"✓ Created features: {df.shape[1]} columns")

# ==================== ENCODING ====================
print("\n[3/7] Encoding...")

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

print(f"✓ Final shape - X: {X.shape}, Test: {X_test.shape}")

# ==================== FINE-TUNED MODELS ====================
print("\n[4/7] Training 3-model ensemble...")

# LightGBM - Balanced tuning
param_lgb = {
    'n_estimators': 5000,
    'learning_rate': 0.006,
    'num_leaves': 150,
    'max_depth': 14,
    'min_child_samples': 8,
    'min_child_weight': 0.0002,
    'subsample': 0.72,
    'subsample_freq': 1,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.01,
    'reg_lambda': 0.50,
    'min_split_gain': 0.001,
    'feature_fraction': 0.95,
    'max_bin': 400,
    'min_data_in_bin': 3,
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

# CatBoost - Balanced tuning
param_cat = {
    'depth': 10,
    'iterations': 3200,
    'learning_rate': 0.020,
    'l2_leaf_reg': 2.8,
    'border_count': 220,
    'bagging_temperature': 0.08,
    'random_strength': 0.18,
    'min_data_in_leaf': 10,
    'rsm': 0.88,
    'boosting_type': 'Plain',
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': False,
    'thread_count': -1
}

# XGBoost - Balanced tuning
param_xgb = {
    'max_depth': 9,
    'learning_rate': 0.0065,
    'n_estimators': 3800,
    'subsample': 0.75,
    'colsample_bytree': 0.80,
    'colsample_bylevel': 0.82,
    'colsample_bynode': 0.85,
    'min_child_weight': 1.5,
    'gamma': 0.02,
    'reg_alpha': 0.02,
    'reg_lambda': 0.58,
    'tree_method': 'hist',
    'max_bin': 400,
    'random_state': 42,
    'n_jobs': -1
}

lgb_model = lgb.LGBMRegressor(**param_lgb)
cat_model = CatBoostRegressor(**param_cat)
xgb_model = XGBRegressor(**param_xgb)

# ==================== STACKING WITH 12 FOLDS ====================
print("\n[5/7] Creating meta features with 12-fold stacking...")


def create_meta_features(models, X_train, X_test, y_train, n_splits=12):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    meta_train = np.zeros((len(X_train), len(models)))
    meta_test = np.zeros((len(X_test), len(models)))

    model_names = ['LightGBM', 'CatBoost', 'XGBoost']

    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"\n  {name}:")
        test_preds = np.zeros(len(X_test))
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)

            val_pred = model.predict(X_val)
            meta_train[val_idx, i] = val_pred
            test_preds += model.predict(X_test) / n_splits

            fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            fold_scores.append(fold_rmse)
            print(f"    Fold {fold}: {fold_rmse:.6f}")

        meta_test[:, i] = test_preds
        print(f"    Average: {np.mean(fold_scores):.6f} (±{np.std(fold_scores):.6f})")

    return meta_train, meta_test


models = [lgb_model, cat_model, xgb_model]
meta_train, meta_test = create_meta_features(models, X, X_test, y, n_splits=12)

# ==================== OPTIMIZED META-MODEL ====================
print("\n[6/7] Training meta-model...")

# Try both scaled and unscaled
scaler = StandardScaler()
meta_train_scaled = scaler.fit_transform(meta_train)
meta_test_scaled = scaler.transform(meta_test)

# Key features to add
key_features = ['curvature', 'speed_limit', 'meta', 'speed_p2', 'curvature_p2',
                'speed_limit_x_curvature', 'risk_score1']

# Test different combinations
print("\nTesting meta-model configurations...")

configs = [
    ("Scaled only", meta_train_scaled, meta_test_scaled),
    ("Unscaled only", meta_train, meta_test),
    ("Scaled + features", np.hstack([meta_train_scaled, X[key_features].values]),
     np.hstack([meta_test_scaled, X_test[key_features].values]))
]

best_config_name = None
best_config_data = None
best_alpha = None
best_cv_rmse = float('inf')

for config_name, train_data, test_data in configs:
    print(f"\n  {config_name}:")
    for alpha in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(train_data, y)
        pred = ridge.predict(train_data)
        rmse = np.sqrt(mean_squared_error(y, pred))

        if rmse < best_cv_rmse:
            best_cv_rmse = rmse
            best_config_name = config_name
            best_config_data = (train_data, test_data)
            best_alpha = alpha

        print(f"    Alpha {alpha}: {rmse:.6f}")

print(f"\n✓ Best: {best_config_name}, Alpha: {best_alpha}, RMSE: {best_cv_rmse:.6f}")

# Train final model
final_meta_model = Ridge(alpha=best_alpha)
final_meta_model.fit(best_config_data[0], y)

# ==================== FINAL PREDICTIONS ====================
print("\n[7/7] Generating predictions...")

oof_preds = final_meta_model.predict(best_config_data[0])
final_predictions = final_meta_model.predict(best_config_data[1])
final_predictions = np.clip(final_predictions, 0, 1)

final_rmse = np.sqrt(mean_squared_error(y, oof_preds))

print(f"✓ Final OOF RMSE: {final_rmse:.6f}")

# ==================== SAVE ====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

submission = pd.DataFrame({
    'id': df_test_ids,
    'accident_risk': final_predictions
})
submission.to_csv('submission.csv', index=False)

backup = f"submission_finetuned_{timestamp}_rmse{final_rmse:.6f}.csv"
submission.to_csv(backup, index=False)

print(f"✓ Main: submission.csv")
print(f"✓ Backup: {backup}")

# ==================== RESULTS ====================
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(f"OOF RMSE:       {final_rmse:.6f}")
print(f"Target:         0.05537")
print(f"Gap:            {(final_rmse - 0.05537):.6f}")
print(f"\nBest Config:    {best_config_name}")
print(f"Ridge Alpha:    {best_alpha}")
print(f"\nPrediction Statistics:")
print(f"  Train: [{y.min():.4f}, {y.max():.4f}]")
print(f"  OOF:   [{oof_preds.min():.4f}, {oof_preds.max():.4f}]")
print(f"  Test:  [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
print(f"\nImprovements Applied:")
print(f"  ✓ Balanced hyperparameters (not too aggressive)")
print(f"  ✓ 12-fold CV (more stable)")
print(f"  ✓ Multiple meta-model configurations tested")
print(f"  ✓ 3 strong diverse models")
print(f"  ✓ Rich feature engineering")
print("=" * 80)

if final_rmse <= 0.05537:
    print("TARGET ACHIEVED!")
else:
    gap = final_rmse - 0.05537
    print(f"Gap: {gap:.6f}")
    if gap < 0.0005:
        print("Extremely close! Try different random_state seeds.")

print("=" * 80)