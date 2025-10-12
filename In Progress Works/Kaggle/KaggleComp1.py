import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, RobustScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from scipy.optimize import minimize, differential_evolution
import warnings

warnings.filterwarnings('ignore')

# === Configuration ===
TRAIN_PATH = "/Users/brianbutler/Desktop/My Work/In Progress Works/Kaggle/train.csv"
TEST_PATH = "/Users/brianbutler/Desktop/My Work/In Progress Works/Kaggle/test.csv"
SUBMISSION_PATH = "/Users/brianbutler/Desktop/My Work/In Progress Works/Kaggle/submission.csv"

N_FOLDS = 15  # Increased for better generalization
RANDOM_STATE = 42

print("=" * 80)
print("ENHANCED COMPETITION MODEL - ADVANCED TECHNIQUES")
print("=" * 80)

# === 1. Load Data ===
print("\n[1/10] Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"✓ Train: {train_df.shape}, Test: {test_df.shape}")

# === 2. Advanced Feature Engineering ===
print("\n[2/10] Creating enhanced feature set...")


def create_advanced_features(df):
    """Enhanced feature engineering with additional interactions"""
    df = df.copy()

    # Convert booleans
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    # === All 2-way Categorical Interactions ===
    cat_features = ['road_type', 'lighting', 'weather', 'time_of_day']

    for i, col1 in enumerate(cat_features):
        for col2 in cat_features[i + 1:]:
            df[f'{col1}_{col2}'] = df[col1].astype(str) + '_' + df[col2].astype(str)

    # === 3-way and 4-way Interactions ===
    df['weather_lighting_time'] = (df['weather'].astype(str) + '_' +
                                   df['lighting'].astype(str) + '_' +
                                   df['time_of_day'].astype(str))
    df['road_weather_lighting'] = (df['road_type'].astype(str) + '_' +
                                   df['weather'].astype(str) + '_' +
                                   df['lighting'].astype(str))
    df['road_speed_weather'] = (df['road_type'].astype(str) + '_' +
                                df['speed_limit'].astype(str) + '_' +
                                df['weather'].astype(str))
    df['full_interaction'] = (df['road_type'].astype(str) + '_' + df['weather'].astype(str) + '_' +
                              df['lighting'].astype(str) + '_' + df['time_of_day'].astype(str))

    # === Enhanced Numerical Interactions ===
    df['speed_lanes'] = df['speed_limit'] * df['num_lanes']
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['curvature_lanes'] = df['curvature'] * df['num_lanes']
    df['speed_accidents'] = df['speed_limit'] * df['num_reported_accidents']
    df['curvature_accidents'] = df['curvature'] * df['num_reported_accidents']
    df['lanes_accidents'] = df['num_lanes'] * df['num_reported_accidents']
    df['speed_curv_lanes'] = df['speed_limit'] * df['curvature'] * df['num_lanes']
    df['all_numeric'] = df['speed_limit'] * df['curvature'] * df['num_lanes'] * (df['num_reported_accidents'] + 1)

    # === Polynomial Features (up to 4th degree) ===
    for col in ['speed_limit', 'curvature', 'num_lanes', 'num_reported_accidents']:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_cubed'] = df[col] ** 3
        df[f'{col}_fourth'] = df[col] ** 4
        df[f'{col}_sqrt'] = np.sqrt(df[col])

    # === Advanced Ratios ===
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 0.1)
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 0.1)
    df['curvature_per_speed'] = df['curvature'] / (df['speed_limit'] + 1)
    df['speed_curvature_ratio'] = df['speed_limit'] / (df['curvature'] + 0.01)
    df['accidents_per_speed'] = df['num_reported_accidents'] / (df['speed_limit'] + 1)
    df['lanes_per_speed'] = df['num_lanes'] / (df['speed_limit'] + 1)

    # === Log and Exponential Transforms ===
    df['log_speed'] = np.log1p(df['speed_limit'])
    df['log_accidents'] = np.log1p(df['num_reported_accidents'])
    df['log_lanes'] = np.log1p(df['num_lanes'])
    df['log_curvature'] = np.log1p(df['curvature'])
    df['exp_curvature'] = np.expm1(df['curvature'])

    # === Binned Features ===
    df['speed_bin'] = pd.cut(df['speed_limit'], bins=5, labels=False)
    df['curvature_bin'] = pd.cut(df['curvature'], bins=5, labels=False)
    df['accidents_bin'] = pd.cut(df['num_reported_accidents'], bins=5, labels=False)

    # === Risk Indicators (Extended) ===
    df['high_speed'] = (df['speed_limit'] >= 60).astype(int)
    df['very_high_speed'] = (df['speed_limit'] >= 70).astype(int)
    df['very_curved'] = (df['curvature'] > 0.7).astype(int)
    df['extremely_curved'] = (df['curvature'] > 0.9).astype(int)
    df['many_accidents'] = (df['num_reported_accidents'] >= 2).astype(int)
    df['very_many_accidents'] = (df['num_reported_accidents'] >= 3).astype(int)
    df['few_lanes'] = (df['num_lanes'] <= 2).astype(int)
    df['single_lane'] = (df['num_lanes'] == 1).astype(int)

    # === Weather Risk ===
    df['bad_weather'] = ((df['weather'] == 'rainy') | (df['weather'] == 'foggy')).astype(int)
    df['poor_lighting'] = ((df['lighting'] == 'dim') | (df['lighting'] == 'night')).astype(int)
    df['dangerous_conditions'] = df['bad_weather'] * df['poor_lighting']
    df['extreme_conditions'] = df['bad_weather'] * df['poor_lighting'] * df['very_curved']

    # === Road Characteristics ===
    df['urban_road'] = (df['road_type'] == 'urban').astype(int)
    df['highway_road'] = (df['road_type'] == 'highway').astype(int)
    df['rural_road'] = (df['road_type'] == 'rural').astype(int)

    # === Time Features ===
    df['morning'] = (df['time_of_day'] == 'morning').astype(int)
    df['evening'] = (df['time_of_day'] == 'evening').astype(int)
    df['afternoon'] = (df['time_of_day'] == 'afternoon').astype(int)
    df['night'] = (df['time_of_day'] == 'night').astype(int)

    # === Advanced Risk Scores ===
    df['risk_score_1'] = (df['high_speed'] + df['very_curved'] +
                          df['bad_weather'] + df['poor_lighting'])
    df['risk_score_2'] = (df['speed_limit'] / 70 + df['curvature'] +
                          df['bad_weather'] * 0.5 + df['poor_lighting'] * 0.5)
    df['risk_score_3'] = (df['speed_limit'] * df['curvature'] *
                          (1 + df['bad_weather']) * (1 + df['poor_lighting']))
    df['weighted_risk'] = (df['speed_limit'] * 0.3 + df['curvature'] * 100 * 0.3 +
                           df['num_reported_accidents'] * 10 * 0.4)

    # === Boolean Combinations ===
    df['signs_public'] = df['road_signs_present'] * df['public_road']
    df['holiday_school'] = df['holiday'] * df['school_season']
    df['no_signs_not_public'] = (1 - df['road_signs_present']) * (1 - df['public_road'])
    df['signs_holiday'] = df['road_signs_present'] * df['holiday']
    df['public_school'] = df['public_road'] * df['school_season']

    # === Complex Conditions ===
    df['high_speed_curved'] = df['high_speed'] * df['very_curved']
    df['night_rainy'] = ((df['lighting'] == 'night') & (df['weather'] == 'rainy')).astype(int)
    df['foggy_morning'] = ((df['weather'] == 'foggy') & (df['time_of_day'] == 'morning')).astype(int)
    df['urban_high_speed'] = df['urban_road'] * df['high_speed']
    df['rural_curved'] = df['rural_road'] * df['very_curved']
    df['highway_accidents'] = df['highway_road'] * df['num_reported_accidents']

    # === Aggregations by Category ===
    for cat_col in ['road_type', 'weather', 'lighting', 'time_of_day']:
        df[f'{cat_col}_accidents_sum'] = df.groupby(cat_col)['num_reported_accidents'].transform('sum')
        df[f'{cat_col}_accidents_mean'] = df.groupby(cat_col)['num_reported_accidents'].transform('mean')
        df[f'{cat_col}_accidents_std'] = df.groupby(cat_col)['num_reported_accidents'].transform('std')
        df[f'{cat_col}_speed_mean'] = df.groupby(cat_col)['speed_limit'].transform('mean')
        df[f'{cat_col}_speed_std'] = df.groupby(cat_col)['speed_limit'].transform('std')
        df[f'{cat_col}_curvature_mean'] = df.groupby(cat_col)['curvature'].transform('mean')
        df[f'{cat_col}_curvature_std'] = df.groupby(cat_col)['curvature'].transform('std')

    # === Deviation from Group Means ===
    for cat_col in ['road_type', 'weather']:
        df[f'speed_dev_{cat_col}'] = df['speed_limit'] - df[f'{cat_col}_speed_mean']
        df[f'curv_dev_{cat_col}'] = df['curvature'] - df[f'{cat_col}_curvature_mean']

    return df


X = train_df.drop(["id", "accident_risk"], axis=1)
y = train_df["accident_risk"]
test_ids = test_df["id"]
X_test = test_df.drop(["id"], axis=1)

X = create_advanced_features(X)
X_test = create_advanced_features(X_test)
print(f"✓ Created {X.shape[1]} features")

# === 3. Multi-Level Encoding ===
print("\n[3/10] Adding advanced encoding...")

cat_cols_original = ['road_type', 'lighting', 'weather', 'time_of_day']

# Frequency encoding
for col in cat_cols_original:
    freq = X[col].value_counts(normalize=True)
    X[f'{col}_freq'] = X[col].map(freq)
    X_test[f'{col}_freq'] = X_test[col].map(freq)

# Target encoding with multiple folds and smoothing
print("\n[4/10] Adding smoothed target encoding...")
kf_target = KFold(n_splits=7, shuffle=True, random_state=42)

for col in cat_cols_original:
    X[f'{col}_target'] = 0
    X[f'{col}_target_smooth'] = 0

    for train_idx, val_idx in kf_target.split(X):
        # Regular target encoding
        means = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).mean()
        X.loc[X.index[val_idx], f'{col}_target'] = X[col].iloc[val_idx].map(means)

        # Smoothed target encoding
        counts = X[col].iloc[train_idx].value_counts()
        global_mean = y.iloc[train_idx].mean()
        smoothing = 10
        smooth_means = (counts * means + smoothing * global_mean) / (counts + smoothing)
        X.loc[X.index[val_idx], f'{col}_target_smooth'] = X[col].iloc[val_idx].map(smooth_means)

    # For test set
    means = y.groupby(X[col]).mean()
    counts = X[col].value_counts()
    global_mean = y.mean()
    smooth_means = (counts * means + 10 * global_mean) / (counts + 10)

    X_test[f'{col}_target'] = X_test[col].map(means).fillna(global_mean)
    X_test[f'{col}_target_smooth'] = X_test[col].map(smooth_means).fillna(global_mean)

# === 5. Label Encoding ===
print("\n[5/10] Encoding categorical features...")
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

print(f"✓ Final feature count: {X.shape[1]}")

# === 6. Enhanced Multi-Model Training ===
print(f"\n[6/10] Training {N_FOLDS}-Fold CV with 8 models...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

models_oof = {f'model_{i}': np.zeros(len(X)) for i in range(8)}
models_test = {f'model_{i}': np.zeros(len(X_test)) for i in range(8)}

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n{'=' * 60}")
    print(f"Fold {fold}/{N_FOLDS}")
    print(f"{'=' * 60}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Model 0: XGBoost (tuned)
    print("Training XGBoost-1...")
    m0 = xgb.XGBRegressor(
        n_estimators=1500, learning_rate=0.015, max_depth=8,
        min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
        gamma=0.05, reg_alpha=0.4, reg_lambda=1.2,
        random_state=RANDOM_STATE + fold, tree_method='hist',
        early_stopping_rounds=75, n_jobs=-1
    )
    m0.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models_oof['model_0'][val_idx] = m0.predict(X_val)
    models_test['model_0'] += m0.predict(X_test) / N_FOLDS

    # Model 1: LightGBM (tuned)
    print("Training LightGBM-1...")
    m1 = lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.015, max_depth=8,
        num_leaves=50, min_child_samples=15, subsample=0.85,
        colsample_bytree=0.85, reg_alpha=0.4, reg_lambda=1.2,
        random_state=RANDOM_STATE + fold, n_jobs=-1, verbose=-1
    )
    m1.fit(X_train, y_train, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(75, verbose=False)])
    models_oof['model_1'][val_idx] = m1.predict(X_val)
    models_test['model_1'] += m1.predict(X_test) / N_FOLDS

    # Model 2: CatBoost (tuned)
    print("Training CatBoost...")
    m2 = CatBoostRegressor(
        iterations=1500, learning_rate=0.015, depth=8,
        l2_leaf_reg=4, random_seed=RANDOM_STATE + fold,
        verbose=0, early_stopping_rounds=75
    )
    m2.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    models_oof['model_2'][val_idx] = m2.predict(X_val)
    models_test['model_2'] += m2.predict(X_test) / N_FOLDS

    # Model 3: HistGradientBoosting
    print("Training HistGradientBoosting...")
    m3 = HistGradientBoostingRegressor(
        max_iter=700, learning_rate=0.015, max_depth=9,
        min_samples_leaf=15, l2_regularization=1.2,
        random_state=RANDOM_STATE + fold
    )
    m3.fit(X_train, y_train)
    models_oof['model_3'][val_idx] = m3.predict(X_val)
    models_test['model_3'] += m3.predict(X_test) / N_FOLDS

    # Model 4: Extra Trees
    print("Training ExtraTrees...")
    m4 = ExtraTreesRegressor(
        n_estimators=300, max_depth=18, min_samples_split=8,
        min_samples_leaf=4, random_state=RANDOM_STATE + fold, n_jobs=-1
    )
    m4.fit(X_train, y_train)
    models_oof['model_4'][val_idx] = m4.predict(X_val)
    models_test['model_4'] += m4.predict(X_test) / N_FOLDS

    # Model 5: XGBoost-2 (alternative config)
    print("Training XGBoost-2...")
    m5 = xgb.XGBRegressor(
        n_estimators=1200, learning_rate=0.02, max_depth=6,
        min_child_weight=4, subsample=0.75, colsample_bytree=0.75,
        gamma=0.15, reg_alpha=0.6, reg_lambda=1.8,
        random_state=RANDOM_STATE + fold + 100, tree_method='hist',
        early_stopping_rounds=75, n_jobs=-1
    )
    m5.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models_oof['model_5'][val_idx] = m5.predict(X_val)
    models_test['model_5'] += m5.predict(X_test) / N_FOLDS

    # Model 6: LightGBM-2 (alternative config)
    print("Training LightGBM-2...")
    m6 = lgb.LGBMRegressor(
        n_estimators=1200, learning_rate=0.02, max_depth=6,
        num_leaves=30, min_child_samples=25, subsample=0.75,
        colsample_bytree=0.75, reg_alpha=0.6, reg_lambda=1.8,
        random_state=RANDOM_STATE + fold + 200, n_jobs=-1, verbose=-1
    )
    m6.fit(X_train, y_train, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(75, verbose=False)])
    models_oof['model_6'][val_idx] = m6.predict(X_val)
    models_test['model_6'] += m6.predict(X_test) / N_FOLDS

    # Model 7: Ridge (for diversity)
    print("Training Ridge...")
    m7 = Ridge(alpha=10.0, random_state=RANDOM_STATE + fold)
    m7.fit(X_train, y_train)
    models_oof['model_7'][val_idx] = m7.predict(X_val)
    models_test['model_7'] += m7.predict(X_test) / N_FOLDS

# === 7. Model Scores ===
print("\n[7/10] Individual model scores...")
model_scores = {}
for i in range(8):
    score = np.sqrt(mean_squared_error(y, models_oof[f'model_{i}']))
    model_scores[f'model_{i}'] = score
    print(f"  Model {i}: {score:.6f}")

# === 8. Advanced Ensemble Optimization ===
print("\n[8/10] Optimizing ensemble with differential evolution...")

oof_matrix = np.column_stack([models_oof[f'model_{i}'] for i in range(8)])


def objective_de(weights):
    weights = np.abs(weights)
    weights = weights / weights.sum()
    pred = oof_matrix @ weights
    pred = np.clip(pred, 0, 1)
    return np.sqrt(mean_squared_error(y, pred))


# Use differential evolution for global optimization
bounds_de = [(0, 1) for _ in range(8)]
result_de = differential_evolution(objective_de, bounds_de, seed=RANDOM_STATE,
                                   maxiter=300, popsize=20, atol=1e-7, tol=1e-7)
optimal_weights = np.abs(result_de.x)
optimal_weights = optimal_weights / optimal_weights.sum()

print("Optimal weights (Differential Evolution):")
for i, w in enumerate(optimal_weights):
    print(f"  Model {i}: {w:.4f}")

best_oof = oof_matrix @ optimal_weights
best_oof = np.clip(best_oof, 0, 1)
best_rmse = np.sqrt(mean_squared_error(y, best_oof))

print(f"\n✓ Optimized Ensemble RMSE: {best_rmse:.6f}")

# === 9. Post-Processing Calibration ===
print("\n[9/10] Applying calibration...")

# Quantile transformation for better distribution matching
qt = QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=RANDOM_STATE)
best_oof_reshaped = best_oof.reshape(-1, 1)
qt.fit(best_oof_reshaped)

# Apply to test predictions
test_matrix = np.column_stack([models_test[f'model_{i}'] for i in range(8)])
final_preds = test_matrix @ optimal_weights
final_preds = np.clip(final_preds, 0, 1)

# Calibrate
final_preds_reshaped = final_preds.reshape(-1, 1)
final_preds_cal = qt.transform(final_preds_reshaped).flatten()
final_preds_cal = np.clip(final_preds_cal, 0, 1)

print(f"  Uncalibrated - Min: {final_preds.min():.6f}, Max: {final_preds.max():.6f}, Mean: {final_preds.mean():.6f}")
print(
    f"  Calibrated   - Min: {final_preds_cal.min():.6f}, Max: {final_preds_cal.max():.6f}, Mean: {final_preds_cal.mean():.6f}")

# === 10. Create Submission ===
print("\n[10/10] Generating final predictions...")

submission_df = pd.DataFrame({
    "id": test_ids,
    "accident_risk": final_preds_cal
})

submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"\n✓ Submission saved: {SUBMISSION_PATH}")
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Cross-Validation RMSE: {best_rmse:.6f}")
print(f"Target Score:          0.05537")
print(f"Difference:            {abs(best_rmse - 0.05537):.6f}")
print(f"Features:              {X.shape[1]}")
print(f"Models:                8")
print(f"CV Folds:              {N_FOLDS}")
print("=" * 80)
print("\n🏆 ENHANCED MODEL READY - PUSHING FOR LOWER SCORE!")
print("=" * 80)