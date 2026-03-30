#!/usr/bin/env python3
"""
Kaggle Diabetes Prediction - Improved Model
Based on CatBoost approach with LightGBM + XGBoost ensemble
Features:
- Native categorical handling with LightGBM
- Feature engineering
- 5-fold stratified CV
- XGBoost + LightGBM ensemble for better accuracy
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# Define paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")

# Configuration
N_FOLDS = 5
SEED = 42
np.random.seed(SEED)


def load_data():
    """Load training and test data."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def create_features(df):
    """Create engineered features to boost accuracy."""
    df = df.copy()
    
    # ===== Health Risk Features =====
    # BMI risk categories
    df['bmi_risk'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 100], 
                            labels=[0, 1, 2, 3, 4]).astype(float)
    
    # Blood pressure categories
    df['bp_systolic_risk'] = pd.cut(df['systolic_bp'], bins=[0, 120, 130, 140, 200], 
                                     labels=[0, 1, 2, 3]).astype(float)
    df['bp_diastolic_risk'] = pd.cut(df['diastolic_bp'], bins=[0, 80, 85, 90, 150], 
                                      labels=[0, 1, 2, 3]).astype(float)
    df['bp_risk'] = df['bp_systolic_risk'] + df['bp_diastolic_risk']
    
    # ===== Cholesterol Features =====
    df['chol_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
    df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)
    df['non_hdl_chol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    # ===== Lifestyle Features =====
    df['sedentary'] = (df['physical_activity_minutes_per_week'] < 150).astype(int)
    df['highly_active'] = (df['physical_activity_minutes_per_week'] >= 300).astype(int)
    df['poor_sleep'] = ((df['sleep_hours_per_day'] < 6) | (df['sleep_hours_per_day'] > 9)).astype(int)
    df['high_screen_time'] = (df['screen_time_hours_per_day'] > 6).astype(int)
    df['poor_diet'] = (df['diet_score'] < 5).astype(int)
    df['heavy_drinker'] = (df['alcohol_consumption_per_week'] >= 7).astype(int)
    
    # Activity to screen ratio
    df['activity_screen_ratio'] = df['physical_activity_minutes_per_week'] / (df['screen_time_hours_per_day'] * 60 + 1)
    
    # ===== Age Features =====
    df['age_group'] = pd.cut(df['age'], bins=[0, 35, 45, 55, 65, 100], 
                             labels=[0, 1, 2, 3, 4]).astype(float)
    df['age_bmi'] = df['age'] * df['bmi']
    df['age_bp'] = df['age'] * df['systolic_bp']
    df['age_waist'] = df['age'] * df['waist_to_hip_ratio']
    
    # ===== Risk Factor Count =====
    df['risk_count'] = (
        (df['bmi'] >= 30).astype(int) +
        (df['systolic_bp'] >= 130).astype(int) +
        (df['diastolic_bp'] >= 85).astype(int) +
        (df['triglycerides'] >= 150).astype(int) +
        (df['hdl_cholesterol'] < 40).astype(int) +
        (df['family_history_diabetes'] == 1).astype(int) +
        (df['hypertension_history'] == 1).astype(int) +
        (df['cardiovascular_history'] == 1).astype(int) +
        (df['age'] >= 45).astype(int) +
        df['sedentary'] +
        df['poor_diet']
    )
    
    # ===== Interaction Features =====
    df['waist_bmi'] = df['waist_to_hip_ratio'] * df['bmi']
    df['heart_bmi'] = df['heart_rate'] * df['bmi']
    
    # Metabolic syndrome proxy
    df['metabolic_score'] = (
        (df['triglycerides'] / 150) +
        (100 - df['hdl_cholesterol']) / 60 +
        (df['systolic_bp'] / 130) +
        (df['bmi'] / 30)
    ) / 4
    
    return df


def main():
    print("="*60)
    print("Loading data...")
    print("="*60)
    train, test = load_data()
    
    y = train["diagnosed_diabetes"]
    X = train.drop(columns=["diagnosed_diabetes"])
    test_ids = test["id"]
    
    # Detect categorical features
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"Categorical features: {cat_features}")
    
    # Fill missing values
    for c in X.columns:
        if c in cat_features:
            X[c] = X[c].fillna("Unknown")
            test[c] = test[c].fillna("Unknown")
        else:
            X[c] = X[c].fillna(X[c].median())
            test[c] = test[c].fillna(X[c].median())
    
    print("Creating features...")
    X = create_features(X)
    test = create_features(test)
    
    # Label encode for XGBoost (LightGBM can use native categoricals)
    X_xgb = X.copy()
    test_xgb = test.copy()
    le_dict = {}
    for col in cat_features:
        le = LabelEncoder()
        combined = pd.concat([X_xgb[col], test_xgb[col]])
        le.fit(combined.astype(str))
        X_xgb[col] = le.transform(X_xgb[col].astype(str))
        test_xgb[col] = le.transform(test_xgb[col].astype(str))
        le_dict[col] = le
    
    # For LightGBM - convert to category dtype
    for col in cat_features:
        X[col] = X[col].astype('category')
        test[col] = test[col].astype('category')
    
    print(f"Training shape: {X.shape}")
    print(f"Test shape: {test.shape}")
    
    # ==================== Cross Validation ====================
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    test_pred_lgb = np.zeros(len(test))
    test_pred_xgb = np.zeros(len(test))
    
    # LightGBM parameters (similar to CatBoost settings)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.02,
        'num_leaves': 64,
        'max_depth': 8,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 5.0,
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.02,
        'max_depth': 8,
        'min_child_weight': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 5.0,
        'seed': SEED,
        'tree_method': 'hist',
    }
    
    print("\n" + "="*60)
    print("Training LightGBM + XGBoost Ensemble with 5-fold CV...")
    print("="*60)
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1} ---")
        
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        X_tr_xgb, X_va_xgb = X_xgb.iloc[tr_idx], X_xgb.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        
        # ===== LightGBM =====
        lgb_train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
        lgb_valid = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_features, reference=lgb_train)
        
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=3000,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=0)
            ]
        )
        
        oof_lgb[va_idx] = lgb_model.predict(X_va)
        test_pred_lgb += lgb_model.predict(test) / N_FOLDS
        lgb_auc = roc_auc_score(y_va, oof_lgb[va_idx])
        
        # ===== XGBoost =====
        dtrain = xgb.DMatrix(X_tr_xgb, label=y_tr)
        dvalid = xgb.DMatrix(X_va_xgb, label=y_va)
        
        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=3000,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            early_stopping_rounds=100,
            verbose_eval=False
        )
        
        oof_xgb[va_idx] = xgb_model.predict(dvalid)
        test_pred_xgb += xgb_model.predict(xgb.DMatrix(test_xgb)) / N_FOLDS
        xgb_auc = roc_auc_score(y_va, oof_xgb[va_idx])
        
        print(f"  LightGBM AUC: {lgb_auc:.5f} | XGBoost AUC: {xgb_auc:.5f}")
    
    # ==================== Results ====================
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    lgb_overall = roc_auc_score(y, oof_lgb)
    xgb_overall = roc_auc_score(y, oof_xgb)
    
    print(f"LightGBM OOF AUC:  {lgb_overall:.5f}")
    print(f"XGBoost OOF AUC:   {xgb_overall:.5f}")
    
    # Ensemble with weighted average (try different weights)
    best_weight = 0.5
    best_ensemble_auc = 0
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        oof_ensemble = w * oof_lgb + (1 - w) * oof_xgb
        ensemble_auc = roc_auc_score(y, oof_ensemble)
        if ensemble_auc > best_ensemble_auc:
            best_ensemble_auc = ensemble_auc
            best_weight = w
    
    print(f"Best Ensemble AUC: {best_ensemble_auc:.5f} (LGB weight: {best_weight})")
    
    # Generate final predictions with best weight
    final_pred = best_weight * test_pred_lgb + (1 - best_weight) * test_pred_xgb
    
    # Save submission
    submission = pd.DataFrame({
        "id": test_ids,
        "diagnosed_diabetes": final_pred
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSubmission saved to {SUBMISSION_PATH}")
    print(f"Submission shape: {submission.shape}")


if __name__ == "__main__":
    main()
