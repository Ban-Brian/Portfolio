import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier

print("="*80)
print("LOAN PAYBACK PREDICTION - COMPREHENSIVE OPTIMIZED MODEL")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/8] Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Try to load sample submission with flexible naming
try:
    sample_submission = pd.read_csv('sample_submission.csv')
except:
    try:
        sample_submission = pd.read_csv('sample_submission__1_.csv')
    except:
        # Create sample submission from test IDs if not found
        print("Sample submission not found - creating from test data")
        sample_submission = pd.DataFrame({
            'id': pd.read_csv('test.csv')['id'],
            'loan_paid_back': 0
        })

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Check for original dataset (if available)
try:
    original_data = pd.read_csv('loan_dataset_20000.csv')
    print(f"Original dataset found: {original_data.shape}")
    HAS_ORIGINAL = True
except:
    print("Original dataset not found - proceeding with train data only")
    HAS_ORIGINAL = False

# ============================================================================
# DATA EXPLORATION
# ============================================================================
print("\n[2/8] Data exploration...")
print("\nTrain columns:", train.columns.tolist())
print("\nFirst few rows:")
print(train.head(3))

print("\nNull values in train:")
print(train.isnull().sum())

print("\nNull values in test:")
print(test.isnull().sum())

print("\nTarget distribution:")
print(train['loan_paid_back'].value_counts())
print(f"Positive class ratio: {train['loan_paid_back'].mean():.4f}")

# ============================================================================
# CONCATENATE ORIGINAL DATA (if available)
# ============================================================================
if HAS_ORIGINAL:
    print("\n[3/8] Concatenating original dataset...")
    # Select only columns that exist in train
    original_data = original_data[train.columns]
    print(f"Original data shape after column selection: {original_data.shape}")

    # Concatenate
    train = pd.concat([train, original_data], ignore_index=True)
    print(f"New train shape after concatenation: {train.shape}")

    # Check for duplicates
    n_duplicates = train.duplicated().sum()
    print(f"Duplicate rows: {n_duplicates}")
    if n_duplicates > 0:
        train = train.drop_duplicates()
        print(f"Shape after removing duplicates: {train.shape}")
else:
    print("\n[3/8] Skipping original data concatenation (not available)")

# ============================================================================
# FEATURE ENGINEERING & PREPROCESSING
# ============================================================================
print("\n[4/8] Feature engineering and preprocessing...")

# Save test IDs before dropping
test_ids = test['id'].copy()

# Drop ID column
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# Identify categorical and numerical columns
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = train.select_dtypes(include=['number']).columns.tolist()
numerical_cols.remove('loan_paid_back')  # Remove target

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# Special handling for grade_subgrade (frequency encoding)
print("\nApplying frequency encoding to grade_subgrade...")
grade_subgrade_freq_map = train['grade_subgrade'].value_counts().to_dict()
print(f"Unique grade_subgrade values: {len(grade_subgrade_freq_map)}")
print("Top 5 most frequent grades:")
for grade, count in list(grade_subgrade_freq_map.items())[:5]:
    print(f"  {grade}: {count}")

# Initialize label encoders
label_encoders = {}
for col in categorical_cols:
    if col != 'grade_subgrade':
        label_encoders[col] = LabelEncoder()

# Preprocessing function
def preprocess_data(df, is_train=True):
    """Apply all preprocessing steps"""
    df = df.copy()

    # Frequency encoding for grade_subgrade
    df['grade_subgrade'] = df['grade_subgrade'].map(grade_subgrade_freq_map)

    # Label encoding for other categorical features
    for col in categorical_cols:
        if col != 'grade_subgrade':
            if is_train:
                df[col] = label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = label_encoders[col].transform(df[col].astype(str))

    return df

# Apply preprocessing
train = preprocess_data(train, is_train=True)
test = preprocess_data(test, is_train=False)

print("\nPreprocessing complete!")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ============================================================================
# PREPARE DATA FOR MODELING
# ============================================================================
print("\n[5/8] Preparing data for modeling...")

X = train.drop('loan_paid_back', axis=1)
y = train['loan_paid_back'].astype(int)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {X.columns.tolist()}")

# ============================================================================
# MODEL TRAINING WITH STRATIFIED K-FOLD
# ============================================================================
print("\n[6/8] Training multiple models with 5-fold StratifiedKFold CV...")
print("="*80)

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

# Define models
models = {
    "XGBClassifier": XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    ),
    "LGBMClassifier": LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    ),
    "CatBoostClassifier": CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        random_state=42,
        verbose=0
    ),
    "GradientBoostingClassifier": GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42
    ),
    "ExtraTreesClassifier": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),
    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),
    "AdaBoostClassifier": AdaBoostClassifier(
        n_estimators=500,
        learning_rate=0.5,
        random_state=42
    )
}

# Store results
model_results = {}
all_predictions = {}

# Train each model
for model_name, model in models.items():
    print(f"\n{model_name}")
    print("-" * 80)

    fold_scores = []
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Split data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Train model
        model.fit(X_train_fold, y_train_fold)

        # Predict on validation set
        y_val_pred = model.predict_proba(X_val_fold)[:, 1]
        oof_predictions[val_idx] = y_val_pred

        # Calculate fold score
        fold_score = roc_auc_score(y_val_fold, y_val_pred)
        fold_scores.append(fold_score)
        print(f"  Fold {fold} ROC AUC: {fold_score:.6f}")

        # Predict on test set
        test_predictions += model.predict_proba(test)[:, 1]

    # Average test predictions
    test_predictions /= N_SPLITS

    # Calculate overall CV score
    cv_score = roc_auc_score(y, oof_predictions)
    avg_fold_score = np.mean(fold_scores)
    std_fold_score = np.std(fold_scores)

    print(f"\n  Overall CV ROC AUC: {cv_score:.6f}")
    print(f"  Average Fold Score: {avg_fold_score:.6f} (+/- {std_fold_score:.6f})")

    # Store results
    model_results[model_name] = {
        'cv_score': cv_score,
        'avg_fold_score': avg_fold_score,
        'std_fold_score': std_fold_score,
        'fold_scores': fold_scores
    }
    all_predictions[model_name] = test_predictions

    # Save individual model prediction
    submission = pd.DataFrame({
        'id': test_ids,
        'loan_paid_back': test_predictions
    })
    submission.to_csv(f'{model_name}_prediction.csv', index=False)
    print(f"  Saved: {model_name}_prediction.csv")

# ============================================================================
# MODEL PERFORMANCE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("[7/8] MODEL PERFORMANCE SUMMARY")
print("="*80)

performance_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'CV Score': [model_results[m]['cv_score'] for m in model_results.keys()],
    'Avg Fold': [model_results[m]['avg_fold_score'] for m in model_results.keys()],
    'Std': [model_results[m]['std_fold_score'] for m in model_results.keys()]
})

performance_df = performance_df.sort_values('CV Score', ascending=False)
print("\n", performance_df.to_string(index=False))

# ============================================================================
# ENSEMBLE TOP MODELS
# ============================================================================
print("\n" + "="*80)
print("[8/8] CREATING ENSEMBLE OF TOP MODELS")
print("="*80)

# Get top 3 models
top_3_models = performance_df['Model'].head(3).tolist()
print(f"\nTop 3 models for ensemble: {top_3_models}")

# Simple average ensemble
print("\nCreating simple average ensemble...")
ensemble_pred = np.mean([all_predictions[model] for model in top_3_models], axis=0)

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': ensemble_pred
})
submission.to_csv('ensemble_top3_prediction.csv', index=False)
print("Saved: ensemble_top3_prediction.csv")

# Weighted ensemble (based on CV scores)
print("\nCreating weighted ensemble...")
top_3_scores = [model_results[model]['cv_score'] for model in top_3_models]
weights = np.array(top_3_scores) / np.sum(top_3_scores)
print(f"Weights: {dict(zip(top_3_models, weights))}")

weighted_ensemble_pred = np.zeros(len(test))
for model, weight in zip(top_3_models, weights):
    weighted_ensemble_pred += weight * all_predictions[model]

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': weighted_ensemble_pred
})
submission.to_csv('ensemble_weighted_prediction.csv', index=False)
print("Saved: ensemble_weighted_prediction.csv")

# All models ensemble
print("\nCreating ensemble of all models...")
all_models_pred = np.mean([all_predictions[model] for model in all_predictions.keys()], axis=0)

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': all_models_pred
})
submission.to_csv('ensemble_all_models_prediction.csv', index=False)
print("Saved: ensemble_all_models_prediction.csv")

# ============================================================================
# BEST SINGLE MODEL SUBMISSION
# ============================================================================
best_model = performance_df.iloc[0]['Model']
best_score = performance_df.iloc[0]['CV Score']

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nBest Single Model: {best_model}")
print(f"Best CV Score: {best_score:.6f}")
print(f"\nRecommended submission: {best_model}_prediction.csv")
print("Alternative: ensemble_weighted_prediction.csv (often performs better on LB)")

print("\n" + "="*80)
print("All submission files created:")
print("="*80)
for model in models.keys():
    print(f"  - {model}_prediction.csv")
print("  - ensemble_top3_prediction.csv")
print("  - ensemble_weighted_prediction.csv")
print("  - ensemble_all_models_prediction.csv")

print("\n" + "="*80)
print("PREDICTION STATISTICS")
print("="*80)
print(f"\nBest model predictions:")
print(f"  Mean: {all_predictions[best_model].mean():.6f}")
print(f"  Std: {all_predictions[best_model].std():.6f}")
print(f"  Min: {all_predictions[best_model].min():.6f}")
print(f"  Max: {all_predictions[best_model].max():.6f}")

print(f"\nWeighted ensemble predictions:")
print(f"  Mean: {weighted_ensemble_pred.mean():.6f}")
print(f"  Std: {weighted_ensemble_pred.std():.6f}")
print(f"  Min: {weighted_ensemble_pred.min():.6f}")
print(f"  Max: {weighted_ensemble_pred.max():.6f}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)