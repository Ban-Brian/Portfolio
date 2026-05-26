import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import warnings
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

warnings.filterwarnings('ignore')

# === Configuration ===
TRAIN_PATH = "/Users/brianbutler/Desktop/My Work/In Progress Works/Kaggle/train.csv"
TEST_PATH = "/Users/brianbutler/Desktop/My Work/In Progress Works/Kaggle/test.csv"
SUBMISSION_PATH = "/Users/brianbutler/Desktop/My Work/In Progress Works/Kaggle/submission.csv"

N_FOLDS = 10
RANDOM_STATE = 42

print("=" * 80)
print("COMPETITION-WINNING MODEL - TARGETING TOP LEADERBOARD")
print("=" * 80)

# === 1. Load Data ===
print("\n[1/9] Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"✓ Train: {train_df.shape}, Test: {test_df.shape}")

# === 2. Comprehensive Feature Engineering ===
print("\n[2/9] Creating extensive feature set...")


def create_all_features(df):
    """Create comprehensive features for top performance"""
    df = df.copy()

    # Convert booleans
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)

    # === All 2-way Categorical Interactions ===
    cat_features = ['road_type', 'lighting', 'weather', 'time_of_day']

    for i, col1 in enumerate(cat_features):
        for col2 in cat_features[i + 1:]:
            df[f'{col1}_{col2}'] = df[col1].astype(str) + '_' + df[col2].astype(str)

    # === 3-way Critical Interactions ===
    df['weather_lighting_time'] = (df['weather'].astype(str) + '_' +
                                   df['lighting'].astype(str) + '_' +
                                   df['time_of_day'].astype(str))
    df['road_weather_lighting'] = (df['road_type'].astype(str) + '_' +
                                   df['weather'].astype(str) + '_' +
                                   df['lighting'].astype(str))
    df['road_speed_weather'] = (df['road_type'].astype(str) + '_' +
                                df['speed_limit'].astype(str) + '_' +
                                df['weather'].astype(str))

    # === Numerical Interactions ===
    df['speed_lanes'] = df['speed_limit'] * df['num_lanes']
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['curvature_lanes'] = df['curvature'] * df['num_lanes']
    df['speed_accidents'] = df['speed_limit'] * df['num_reported_accidents']
    df['curvature_accidents'] = df['curvature'] * df['num_reported_accidents']
    df['lanes_accidents'] = df['num_lanes'] * df['num_reported_accidents']

    # === Polynomial Features ===
    df['speed_squared'] = df['speed_limit'] ** 2
    df['curvature_squared'] = df['curvature'] ** 2
    df['lanes_squared'] = df['num_lanes'] ** 2
    df['accidents_squared'] = df['num_reported_accidents'] ** 2

    df['speed_cubed'] = df['speed_limit'] ** 3
    df['curvature_cubed'] = df['curvature'] ** 3

    # === Ratios ===
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 0.1)
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 0.1)
    df['curvature_per_speed'] = df['curvature'] / (df['speed_limit'] + 1)
    df['speed_curvature_ratio'] = df['speed_limit'] / (df['curvature'] + 0.01)

    # === Log Transforms ===
    df['log_speed'] = np.log1p(df['speed_limit'])
    df['log_accidents'] = np.log1p(df['num_reported_accidents'])
    df['log_lanes'] = np.log1p(df['num_lanes'])

    # === Risk Indicators ===
    df['high_speed'] = (df['speed_limit'] >= 60).astype(int)
    df['very_curved'] = (df['curvature'] > 0.7).astype(int)
    df['many_accidents'] = (df['num_reported_accidents'] >= 2).astype(int)
    df['few_lanes'] = (df['num_lanes'] <= 2).astype(int)

    # === Weather Risk ===
    df['bad_weather'] = ((df['weather'] == 'rainy') | (df['weather'] == 'foggy')).astype(int)
    df['poor_lighting'] = ((df['lighting'] == 'dim') | (df['lighting'] == 'night')).astype(int)
    df['dangerous_conditions'] = df['bad_weather'] * df['poor_lighting']

    # === Road Characteristics ===
    df['urban_road'] = (df['road_type'] == 'urban').astype(int)
    df['highway_road'] = (df['road_type'] == 'highway').astype(int)
    df['rural_road'] = (df['road_type'] == 'rural').astype(int)

    # === Time Features ===
    df['morning'] = (df['time_of_day'] == 'morning').astype(int)
    df['evening'] = (df['time_of_day'] == 'evening').astype(int)
    df['afternoon'] = (df['time_of_day'] == 'afternoon').astype(int)

    # === Combined Risk Scores ===
    df['risk_score_1'] = (df['high_speed'] + df['very_curved'] +
                          df['bad_weather'] + df['poor_lighting'])
    df['risk_score_2'] = (df['speed_limit'] / 70 + df['curvature'] +
                          df['bad_weather'] * 0.5 + df['poor_lighting'] * 0.5)

    # === Boolean Combinations ===
    df['signs_public'] = df['road_signs_present'] * df['public_road']
    df['holiday_school'] = df['holiday'] * df['school_season']
    df['no_signs_not_public'] = (1 - df['road_signs_present']) * (1 - df['public_road'])

    # === Special Conditions ===
    df['high_speed_curved'] = df['high_speed'] * df['very_curved']
    df['night_rainy'] = ((df['lighting'] == 'night') & (df['weather'] == 'rainy')).astype(int)
    df['foggy_morning'] = ((df['weather'] == 'foggy') & (df['time_of_day'] == 'morning')).astype(int)

    # === Aggregations by Category ===
    for cat_col in ['road_type', 'weather', 'lighting']:
        df[f'{cat_col}_accidents_sum'] = df.groupby(cat_col)['num_reported_accidents'].transform('sum')
        df[f'{cat_col}_accidents_mean'] = df.groupby(cat_col)['num_reported_accidents'].transform('mean')
        df[f'{cat_col}_speed_mean'] = df.groupby(cat_col)['speed_limit'].transform('mean')
        df[f'{cat_col}_curvature_mean'] = df.groupby(cat_col)['curvature'].transform('mean')

    return df


X = train_df.drop(["id", "accident_risk"], axis=1)
y = train_df["accident_risk"]
test_ids = test_df["id"]
X_test = test_df.drop(["id"], axis=1)

X = create_all_features(X)
X_test = create_all_features(X_test)
print(f"✓ Created {X.shape[1]} features")

# === 3. Frequency Encoding ===
print("\n[3/9] Adding frequency encoding...")

cat_cols_original = ['road_type', 'lighting', 'weather', 'time_of_day']
for col in cat_cols_original:
    freq = X[col].value_counts(normalize=True)
    X[f'{col}_freq'] = X[col].map(freq)
    X_test[f'{col}_freq'] = X_test[col].map(freq)

# === 3.5. Hyperparameter Tuning with Optuna (LightGBM) ===
def optuna_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'verbose': -1
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    losses = []
    for train_idx, val_idx in cv.split(X, y):
        train_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50)]
        )
        preds = model.predict(X.iloc[val_idx])
        losses.append(mean_squared_error(y.iloc[val_idx], preds, squared=False))
    return np.mean(losses)

# === 4. Target Encoding (CV) ===
print("\n[4/9] Adding target encoding...")

kf_target = KFold(n_splits=5, shuffle=True, random_state=42)
for col in cat_cols_original:
    X[f'{col}_target'] = 0
    for train_idx, val_idx in kf_target.split(X):
        means = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).mean()
        X.loc[X.index[val_idx], f'{col}_target'] = X[col].iloc[val_idx].map(means)

    # For test
    means = y.groupby(X[col]).mean()
    X_test[f'{col}_target'] = X_test[col].map(means)
    X_test[f'{col}_target'].fillna(y.mean(), inplace=True)

# === 5. Label Encoding ===
print("\n[5/9] Encoding categorical features...")

cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

print(f"✓ Final feature count: {X.shape[1]}")

# === 6. Multi-Model Training ===
print(f"\n[6/9] Training {N_FOLDS}-Fold CV with 6 models...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

models_oof = {f'model_{i}': np.zeros(len(X)) for i in range(6)}
models_test = {f'model_{i}': np.zeros(len(X_test)) for i in range(6)}

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n{'=' * 60}")
    print(f"Fold {fold}/{N_FOLDS}")
    print(f"{'=' * 60}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Model 0: XGBoost
    print("Training XGBoost...")
    m0 = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.02, max_depth=7,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, reg_alpha=0.3, reg_lambda=1.0,
        random_state=RANDOM_STATE + fold, tree_method='hist',
        early_stopping_rounds=50, n_jobs=-1
    )
    m0.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models_oof['model_0'][val_idx] = m0.predict(X_val)
    models_test['model_0'] += m0.predict(X_test) / N_FOLDS

    # Model 1: LightGBM
    print("Training LightGBM...")
    m1 = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.02, max_depth=7,
        num_leaves=40, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=1.0,
        random_state=RANDOM_STATE + fold, n_jobs=-1, verbose=-1
    )
    m1.fit(X_train, y_train, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(50, verbose=False)])
    models_oof['model_1'][val_idx] = m1.predict(X_val)
    models_test['model_1'] += m1.predict(X_test) / N_FOLDS

    # Model 2: CatBoost
    print("Training CatBoost...")
    m2 = CatBoostRegressor(
        iterations=1000, learning_rate=0.02, depth=7,
        l2_leaf_reg=3, random_seed=RANDOM_STATE + fold,
        verbose=0, early_stopping_rounds=50
    )
    m2.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    models_oof['model_2'][val_idx] = m2.predict(X_val)
    models_test['model_2'] += m2.predict(X_test) / N_FOLDS

    # Model 3: HistGradientBoosting
    print("Training HistGradientBoosting...")
    m3 = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.02, max_depth=8,
        min_samples_leaf=20, l2_regularization=1.0,
        random_state=RANDOM_STATE + fold
    )
    m3.fit(X_train, y_train)
    models_oof['model_3'][val_idx] = m3.predict(X_val)
    models_test['model_3'] += m3.predict(X_test) / N_FOLDS

    # Model 4: Extra Trees
    print("Training ExtraTrees...")
    m4 = ExtraTreesRegressor(
        n_estimators=200, max_depth=15, min_samples_split=10,
        min_samples_leaf=5, random_state=RANDOM_STATE + fold, n_jobs=-1
    )
    m4.fit(X_train, y_train)
    models_oof['model_4'][val_idx] = m4.predict(X_val)
    models_test['model_4'] += m4.predict(X_test) / N_FOLDS

    # Model 5: XGBoost (alternative config)
    print("Training XGBoost-2...")
    m5 = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.025, max_depth=6,
        min_child_weight=5, subsample=0.75, colsample_bytree=0.75,
        gamma=0.2, reg_alpha=0.5, reg_lambda=1.5,
        random_state=RANDOM_STATE + fold + 100, tree_method='hist',
        early_stopping_rounds=50, n_jobs=-1
    )
    m5.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models_oof['model_5'][val_idx] = m5.predict(X_val)
    models_test['model_5'] += m5.predict(X_test) / N_FOLDS

# === 7. Model Scores ===
print("\n[7/9] Individual model scores...")
model_scores = {}
for i in range(6):
    score = np.sqrt(mean_squared_error(y, models_oof[f'model_{i}']))
    model_scores[f'model_{i}'] = score
    print(f"  Model {i}: {score:.6f}")

# === 8. Optimal Ensemble via Optimization ===
print("\n[8/9] Optimizing ensemble weights...")

oof_matrix = np.column_stack([models_oof[f'model_{i}'] for i in range(6)])


def objective(weights):
    weights = np.abs(weights)
    weights = weights / weights.sum()
    pred = oof_matrix @ weights
    pred = np.clip(pred, 0, 1)
    return np.sqrt(mean_squared_error(y, pred))


initial = np.ones(6) / 6
bounds = [(0, 1) for _ in range(6)]
constraints = {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1}

result = minimize(objective, initial, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = np.abs(result.x)
optimal_weights = optimal_weights / optimal_weights.sum()

print("Optimal weights:")
for i, w in enumerate(optimal_weights):
    print(f"  Model {i}: {w:.4f}")

best_oof = oof_matrix @ optimal_weights
best_oof = np.clip(best_oof, 0, 1)
best_rmse = np.sqrt(mean_squared_error(y, best_oof))

print(f"\n✓ Optimized Ensemble RMSE: {best_rmse:.6f}")

# === 9. Final Predictions ===
print("\n[9/9] Generating final predictions...")

test_matrix = np.column_stack([models_test[f'model_{i}'] for i in range(6)])
final_preds = test_matrix @ optimal_weights
final_preds = np.clip(final_preds, 0, 1)

print(f"  Min:  {final_preds.min():.6f}")
print(f"  Max:  {final_preds.max():.6f}")
print(f"  Mean: {final_preds.mean():.6f}")

# === 10. Create Submission ===
submission_df = pd.DataFrame({
    "id": test_ids,
    "accident_risk": final_preds
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
print(f"Models:                6")
print("=" * 80)
print("\n🏆 COMPETITION MODEL READY - AIMING FOR TOP OF LEADERBOARD!")
print("=" * 80)