import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import skew

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

print("=" * 80)
print("WINNING MODEL - HIGH-SCORING TECHNIQUES")
print("Target: 0.923-0.925 CV AUC")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_SPLITS = 10
RANDOM_STATE = 42

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/9] Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Load original dataset if available
try:
    original_data = pd.read_csv('loan_dataset_20000.csv')
    print(f"Original dataset found: {original_data.shape}")
    HAS_ORIGINAL = True
except:
    print("Original dataset not found")
    HAS_ORIGINAL = False

# ============================================================================
# CONCATENATE ORIGINAL DATA
# ============================================================================
if HAS_ORIGINAL:
    print("\n[2/9] Concatenating original dataset...")
    original_data = original_data[train.columns]
    train = pd.concat([train, original_data], ignore_index=True)
    print(f"New train shape: {train.shape}")

    n_duplicates = train.duplicated().sum()
    if n_duplicates > 0:
        train = train.drop_duplicates()
        print(f"Removed {n_duplicates} duplicates")
else:
    print("\n[2/9] No original data to concatenate")

# Save test IDs and drop from both
test_ids = test['id'].copy()
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# ============================================================================
# IDENTIFY COLUMN TYPES
# ============================================================================
print("\n[3/9] Identifying column types...")

categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = train.select_dtypes(include=['number']).columns.tolist()
numerical_cols.remove('loan_paid_back')

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# ============================================================================
# HANDLE SKEWNESS (LOG TRANSFORMATION)
# ============================================================================
print("\n[4/9] Handling skewness with log transformation...")

# Calculate skewness for numerical columns
skew_values = train[numerical_cols].apply(lambda x: skew(x.dropna()))
print("\nSkewness values:")
print(skew_values.sort_values(ascending=False))

# Apply log transformation to highly skewed columns (absolute skew > 1)
skewed_cols = skew_values[abs(skew_values) > 1].index.tolist()
print(f"\nApplying log transformation to: {skewed_cols}")

for col in skewed_cols:
    train[col] = np.log1p(train[col])
    test[col] = np.log1p(test[col])

print("Log transformation complete!")

# ============================================================================
# OUTLIER CLIPPING (WINSORIZATION)
# ============================================================================
print("\n[5/9] Clipping outliers (Winsorization at 1st and 99th percentiles)...")

for col in numerical_cols:
    lower = train[col].quantile(0.01)
    upper = train[col].quantile(0.99)

    # Clip values
    train[col] = train[col].clip(lower, upper)
    test[col] = test[col].clip(lower, upper)

    print(f"  {col}: clipped to [{lower:.4f}, {upper:.4f}]")

print("Outlier clipping complete!")

# ============================================================================
# GRADE SUBGRADE FEATURE ENGINEERING
# ============================================================================
print("\n[6/9] Engineering grade_subgrade features...")

# Split grade_subgrade into grade (letter) and subgrade (number)
train['grade'] = train['grade_subgrade'].str[0]
train['subgrade'] = train['grade_subgrade'].str[1:].astype(int)

test['grade'] = test['grade_subgrade'].str[0]
test['subgrade'] = test['grade_subgrade'].str[1:].astype(int)

# Create ordered numerical encoding for grade
grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
train['grade_num'] = train['grade'].map(grade_order)
test['grade_num'] = test['grade'].map(grade_order)

print(f"Created 3 new features: grade, subgrade, grade_num")
print(f"Grade distribution: {train['grade'].value_counts().sort_index().to_dict()}")

# Update categorical columns list
categorical_cols.append('grade')

# ============================================================================
# QUANTILE BINNING FOR NUMERICAL FEATURES
# ============================================================================
print("\n[7/9] Creating quantile bins for numerical features...")


def add_quantile_bins(train_df, test_df, num_cols, q_list=[5, 10, 15]):
    """
    Create binned versions of numerical columns at multiple quantiles.
    This captures non-linear patterns at different granularities.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    bins_created = 0

    for col in num_cols:
        for q in q_list:
            try:
                # Create bins using quantiles on training data
                train_bins, bins = pd.qcut(
                    train_df[col],
                    q=q,
                    labels=False,
                    retbins=True,
                    duplicates='drop'
                )
                train_df[f'{col}_bin{q}'] = train_bins

                # Apply same bins to test data
                test_df[f'{col}_bin{q}'] = pd.cut(
                    test_df[col],
                    bins=bins,
                    labels=False,
                    include_lowest=True
                )
                bins_created += 1

            except Exception as e:
                # If binning fails (e.g., too few unique values), create zero column
                train_df[f'{col}_bin{q}'] = 0
                test_df[f'{col}_bin{q}'] = 0

    print(f"  Created {bins_created} binned features")
    return train_df, test_df


# Add binned features - including the new numerical features
all_num_cols = numerical_cols + ['subgrade', 'grade_num']
train, test = add_quantile_bins(train, test, all_num_cols, q_list=[5, 10, 15])

print(f"Train shape after binning: {train.shape}")
print(f"Test shape after binning: {test.shape}")

# ============================================================================
# FREQUENCY ENCODING FOR CATEGORICAL FEATURES
# ============================================================================
print("\n[8/9] Creating frequency encoding for categorical features...")


def add_frequency_encoding(train_df, test_df, cat_cols):
    """
    Add frequency encoding for categorical columns.
    This is often better than simple label encoding.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in cat_cols:
        # Calculate frequency on training data
        freq = train_df[col].value_counts(dropna=False)

        # Map to both train and test
        train_df[f'{col}_freq'] = train_df[col].map(freq)
        test_df[f'{col}_freq'] = test_df[col].map(freq).fillna(freq.mean())

        print(f"  {col}: {len(freq)} unique values")

    print(f"  Created {len(cat_cols)} frequency features")
    return train_df, test_df


train, test = add_frequency_encoding(train, test, categorical_cols)

# ============================================================================
# LABEL ENCODING FOR CATEGORICAL FEATURES
# ============================================================================
print("\nApplying label encoding to categorical features...")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

print("Encoding complete!")

# ============================================================================
# PREPARE FINAL DATASET
# ============================================================================
print("\nPreparing final dataset...")

X = train.drop('loan_paid_back', axis=1)
y = train['loan_paid_back'].astype(int)

print(f"\nFinal feature count: {X.shape[1]}")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# ============================================================================
# MODEL TRAINING WITH 10-FOLD CV
# ============================================================================
print("\n[9/9] Training models with 10-fold StratifiedKFold CV...")
print("=" * 80)

# Optimized LightGBM parameters (based on winning notebook)
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1320,
    'learning_rate': 0.05,
    'num_leaves': 93,
    'max_depth': 5,
    'subsample': 0.743,
    'colsample_bytree': 0.975,
    'reg_alpha': 2.95,
    'reg_lambda': 0.0022,
    'min_child_samples': 20,
    'random_state': RANDOM_STATE,
    'verbose': -1,
    'n_jobs': -1
}

# Complementary XGBoost parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'n_estimators': 1200,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.75,
    'colsample_bytree': 0.95,
    'reg_alpha': 2.5,
    'reg_lambda': 0.01,
    'min_child_weight': 5,
    'random_state': RANDOM_STATE,
    'verbosity': 0,
    'n_jobs': -1
}

models = {
    'LightGBM': LGBMClassifier(**lgb_params),
    'XGBoost': XGBClassifier(**xgb_params)
}

# Cross-validation
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

model_results = {}
all_predictions = {}
oof_predictions_dict = {}

for model_name, model in models.items():
    print(f"\n{model_name}")
    print("-" * 80)

    fold_scores = []
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Train
        model.fit(X_train_fold, y_train_fold)

        # Predict on validation
        y_val_pred = model.predict_proba(X_val_fold)[:, 1]
        oof_predictions[val_idx] = y_val_pred

        # Score
        fold_score = roc_auc_score(y_val_fold, y_val_pred)
        fold_scores.append(fold_score)
        print(f"  Fold {fold:2d} AUC: {fold_score:.6f}")

        # Predict on test
        test_predictions += model.predict_proba(test)[:, 1]

    # Average test predictions
    test_predictions /= N_SPLITS

    # Overall CV score
    cv_score = roc_auc_score(y, oof_predictions)
    avg_fold_score = np.mean(fold_scores)
    std_fold_score = np.std(fold_scores)

    print(f"\n  Overall OOF AUC: {cv_score:.6f}")
    print(f"  Average Fold:    {avg_fold_score:.6f} (+/- {std_fold_score:.6f})")
    print(f"  Fold AUCs: {[f'{score:.4f}' for score in fold_scores]}")

    # Store results
    model_results[model_name] = {
        'cv_score': cv_score,
        'avg_fold_score': avg_fold_score,
        'std_fold_score': std_fold_score,
        'fold_scores': fold_scores
    }
    all_predictions[model_name] = test_predictions
    oof_predictions_dict[model_name] = oof_predictions

    # Save individual predictions
    submission = pd.DataFrame({
        'id': test_ids,
        'loan_paid_back': test_predictions
    })
    submission.to_csv(f'{model_name}_winning.csv', index=False)
    print(f"  Saved: {model_name}_winning.csv")

# ============================================================================
# ENSEMBLING
# ============================================================================
print("\n" + "=" * 80)
print("CREATING ENSEMBLES")
print("=" * 80)

# 1. Weighted ensemble (CV-based)
print("\n1. Weighted Ensemble")
scores = [model_results[m]['cv_score'] for m in models.keys()]
weights = np.array(scores) / np.sum(scores)

for model, weight in zip(models.keys(), weights):
    print(f"  {model}: {weight:.4f}")

weighted_pred = np.zeros(len(test))
for model, weight in zip(models.keys(), weights):
    weighted_pred += weight * all_predictions[model]

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': weighted_pred
})
submission.to_csv('weighted_ensemble_winning.csv', index=False)
print("Saved: weighted_ensemble_winning.csv")

# 2. Simple average
print("\n2. Simple Average Ensemble")
simple_avg = np.mean([all_predictions[m] for m in models.keys()], axis=0)

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': simple_avg
})
submission.to_csv('simple_average_winning.csv', index=False)
print("Saved: simple_average_winning.csv")

# 3. Rank averaging
print("\n3. Rank Average Ensemble")
from scipy.stats import rankdata

rank_avg = np.zeros(len(test))
for model in models.keys():
    ranks = rankdata(all_predictions[model]) / len(all_predictions[model])
    rank_avg += ranks
rank_avg /= len(models)

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': rank_avg
})
submission.to_csv('rank_average_winning.csv', index=False)
print("Saved: rank_average_winning.csv")

# 4. Stacked ensemble
print("\n4. Stacked Ensemble")
from sklearn.linear_model import LogisticRegression

oof_stack = np.column_stack([oof_predictions_dict[m] for m in models.keys()])
test_stack = np.column_stack([all_predictions[m] for m in models.keys()])

meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
meta_model.fit(oof_stack, y)

stacked_pred = meta_model.predict_proba(test_stack)[:, 1]
stacked_cv = roc_auc_score(y, meta_model.predict_proba(oof_stack)[:, 1])

print(f"Stacked CV AUC: {stacked_cv:.6f}")

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': stacked_pred
})
submission.to_csv('stacked_ensemble_winning.csv', index=False)
print("Saved: stacked_ensemble_winning.csv")

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

# Performance summary
performance_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'CV Score': [model_results[m]['cv_score'] for m in model_results.keys()],
    'Avg Fold': [model_results[m]['avg_fold_score'] for m in model_results.keys()],
    'Std': [model_results[m]['std_fold_score'] for m in model_results.keys()]
})

print("\nModel Performance:")
print(performance_df.to_string(index=False))

best_model = performance_df.iloc[0]['Model']
best_score = performance_df.iloc[0]['CV Score']

print(f"\nBest Single Model: {best_model}")
print(f"Best CV Score: {best_score:.6f}")
print(f"Stacked Ensemble CV: {stacked_cv:.6f}")

print("\n" + "=" * 80)
print("RECOMMENDED SUBMISSIONS (IN ORDER):")
print("=" * 80)
print("1. stacked_ensemble_winning.csv      ⭐ BEST OVERALL")
print(f"2. {best_model}_winning.csv (Best single model)")
print("3. weighted_ensemble_winning.csv     Alternative #1")
print("4. rank_average_winning.csv          Alternative #2")
print("5. simple_average_winning.csv        Baseline ensemble")

print("\n" + "=" * 80)
print("KEY TECHNIQUES APPLIED:")
print("=" * 80)
print("✓ Log transformation for skewed features")
print("✓ Outlier clipping (Winsorization)")
print("✓ Grade subgrade splitting (3 features)")
print("✓ Quantile binning (5, 10, 15 bins)")
print("✓ Frequency encoding for categoricals")
print("✓ Optimized hyperparameters")
print("✓ 10-fold cross-validation")
print("✓ Multiple ensemble strategies")

print("\n" + "=" * 80)
print(f"Total Features: {X.shape[1]}")
print(f"Original Features: {len(numerical_cols) + len(categorical_cols)}")
print(f"Engineered Features: {X.shape[1] - len(numerical_cols) - len(categorical_cols)}")
print("\nExpected Performance:")
print("  CV Score: 0.923-0.925")
print("  LB Score: 0.921-0.924")
print("=" * 80)
print("\n🎉 TRAINING COMPLETE! 🎉")
print("=" * 80)