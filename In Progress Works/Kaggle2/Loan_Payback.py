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
from catboost import CatBoostClassifier

print("=" * 80)
print("EXPERIMENTAL APPROACH - TARGET 0.927+ PUBLIC")
print("=" * 80)

N_SPLITS = 10
RANDOM_STATE = 42

print("\n[1/10] Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

try:
    original_data = pd.read_csv('loan_dataset_20000.csv')
    print(f"Original dataset found: {original_data.shape}")
    original_data = original_data[train.columns]
    train = pd.concat([train, original_data], ignore_index=True)
    print(f"Combined train shape: {train.shape}")

    n_dup = train.duplicated().sum()
    if n_dup > 0:
        train = train.drop_duplicates()
        print(f"Removed {n_dup} duplicates")
except:
    print("Original dataset not found")

test_ids = test['id'].copy()
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

print("\n[2/10] Basic preprocessing...")
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = train.select_dtypes(include=['number']).columns.tolist()
numerical_cols.remove('loan_paid_back')

print(f"Categorical: {len(categorical_cols)}")
print(f"Numerical: {len(numerical_cols)}")

print("\n[3/10] Log transformation...")
skew_values = train[numerical_cols].apply(lambda x: skew(x.dropna()))
skewed_cols = skew_values[abs(skew_values) > 1].index.tolist()

for col in skewed_cols:
    train[col] = np.log1p(train[col])
    test[col] = np.log1p(test[col])
print(f"Transformed: {skewed_cols}")

print("\n[4/10] Outlier clipping...")
for col in numerical_cols:
    lower = train[col].quantile(0.01)
    upper = train[col].quantile(0.99)
    train[col] = train[col].clip(lower, upper)
    test[col] = test[col].clip(lower, upper)

print("\n[5/10] Grade features...")
train['grade'] = train['grade_subgrade'].str[0]
train['subgrade'] = train['grade_subgrade'].str[1:].astype(int)
test['grade'] = test['grade_subgrade'].str[0]
test['subgrade'] = test['grade_subgrade'].str[1:].astype(int)

grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
train['grade_num'] = train['grade'].map(grade_order)
test['grade_num'] = test['grade'].map(grade_order)

categorical_cols.append('grade')

print("\n[6/10] Quantile bins...")


def add_bins(train_df, test_df, cols, q_list=[5, 10, 15]):
    for col in cols:
        for q in q_list:
            try:
                train_bins, bins = pd.qcut(train_df[col], q=q, labels=False, retbins=True, duplicates='drop')
                train_df[f'{col}_q{q}'] = train_bins
                test_df[f'{col}_q{q}'] = pd.cut(test_df[col], bins=bins, labels=False, include_lowest=True)
            except:
                train_df[f'{col}_q{q}'] = 0
                test_df[f'{col}_q{q}'] = 0
    return train_df, test_df


num_cols_for_bins = numerical_cols + ['subgrade', 'grade_num']
train, test = add_bins(train, test, num_cols_for_bins)

print("\n[7/10] Key interactions...")
train['loan_to_income'] = train['loan_amount'] / (train['annual_income'] + 1)
test['loan_to_income'] = test['loan_amount'] / (test['annual_income'] + 1)

train['total_debt'] = train['debt_to_income_ratio'] * train['annual_income']
test['total_debt'] = test['debt_to_income_ratio'] * test['annual_income']

train['credit_risk'] = 850 - train['credit_score']
test['credit_risk'] = 850 - test['credit_score']

train['total_interest'] = train['loan_amount'] * train['interest_rate'] / 100
test['total_interest'] = test['loan_amount'] * test['interest_rate'] / 100

print("\n[8/10] Encoding...")
y_full = train['loan_paid_back'].copy()
global_mean = y_full.mean()

for col in categorical_cols:
    freq = train[col].value_counts(dropna=False)
    train[f'{col}_freq'] = train[col].map(freq)
    test[f'{col}_freq'] = test[col].map(freq).fillna(freq.mean())

for col in categorical_cols:
    agg = train.groupby(col)['loan_paid_back'].agg(['mean', 'count'])
    smoothing = 10
    agg['target'] = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)

    train[f'{col}_target'] = train[col].map(agg['target'])
    test[f'{col}_target'] = test[col].map(agg['target']).fillna(global_mean)

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

X = train.drop('loan_paid_back', axis=1)
y = train['loan_paid_back'].astype(int)

print(f"\nFeatures: {X.shape[1]}")

print(f"\n[9/10] Training {N_SPLITS}-fold CV...")
print("=" * 80)

models_config = {
    'LGB_1': {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 1500,
        'learning_rate': 0.04,
        'num_leaves': 95,
        'max_depth': 7,
        'subsample': 0.75,
        'colsample_bytree': 0.95,
        'reg_alpha': 3.0,
        'reg_lambda': 0.01,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    },
    'LGB_2': {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 1400,
        'learning_rate': 0.045,
        'num_leaves': 85,
        'max_depth': 6,
        'subsample': 0.77,
        'colsample_bytree': 0.93,
        'reg_alpha': 2.7,
        'reg_lambda': 0.015,
        'random_state': 123,
        'verbose': -1,
        'n_jobs': -1
    },
    'LGB_3': {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 1300,
        'learning_rate': 0.05,
        'num_leaves': 75,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'reg_alpha': 2.3,
        'reg_lambda': 0.02,
        'random_state': 456,
        'verbose': -1,
        'n_jobs': -1
    },
    'XGB_1': {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 1400,
        'learning_rate': 0.04,
        'max_depth': 8,
        'subsample': 0.75,
        'colsample_bytree': 0.93,
        'reg_alpha': 2.8,
        'reg_lambda': 0.02,
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': -1
    },
    'XGB_2': {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 1300,
        'learning_rate': 0.045,
        'max_depth': 7,
        'subsample': 0.77,
        'colsample_bytree': 0.9,
        'reg_alpha': 2.5,
        'reg_lambda': 0.025,
        'random_state': 123,
        'verbosity': 0,
        'n_jobs': -1
    },
    'CAT_1': {
        'iterations': 1500,
        'depth': 5,
        'learning_rate': 0.09,
        'l2_leaf_reg': 12.0,
        'border_count': 240,
        'random_strength': 4.8,
        'bagging_temperature': 0.25,
        'random_state': 42,
        'eval_metric': 'AUC',
        'verbose': 0
    },
    'CAT_2': {
        'iterations': 1400,
        'depth': 4,
        'learning_rate': 0.095,
        'l2_leaf_reg': 11.5,
        'border_count': 235,
        'random_strength': 4.5,
        'bagging_temperature': 0.24,
        'random_state': 123,
        'eval_metric': 'AUC',
        'verbose': 0
    }
}

models = {
    name: (LGBMClassifier(**params) if 'LGB' in name else
           XGBClassifier(**params) if 'XGB' in name else
           CatBoostClassifier(**params))
    for name, params in models_config.items()
}

skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

results = {}
predictions = {}
oof_preds = {}

for name, model in models.items():
    print(f"\n{name}")
    print("-" * 80)

    scores = []
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_pred

        score = roc_auc_score(y_val, val_pred)
        scores.append(score)

        test_pred += model.predict_proba(test)[:, 1]

    test_pred /= N_SPLITS

    cv = roc_auc_score(y, oof)
    avg = np.mean(scores)

    print(f"CV: {cv:.6f} | Avg: {avg:.6f}")

    results[name] = cv
    predictions[name] = test_pred
    oof_preds[name] = oof

print("\n[10/10] Creating blends...")
print("=" * 80)

df = pd.DataFrame({'Model': list(results.keys()), 'CV': list(results.values())}).sort_values('CV', ascending=False)
print("\n", df.to_string(index=False))

print("\nSimple average blend")
simple = np.mean([predictions[m] for m in models.keys()], axis=0)
pd.DataFrame({'id': test_ids, 'loan_paid_back': simple}).to_csv('simple_blend.csv', index=False)

print("CV-weighted blend")
cvs = [results[m] for m in models.keys()]
weights = np.array(cvs) / np.sum(cvs)
weighted = sum(w * predictions[m] for m, w in zip(models.keys(), weights))
pd.DataFrame({'id': test_ids, 'loan_paid_back': weighted}).to_csv('weighted_blend.csv', index=False)

print("Top 5 blend")
top5 = df['Model'].head(5).tolist()
top5_pred = np.mean([predictions[m] for m in top5], axis=0)
pd.DataFrame({'id': test_ids, 'loan_paid_back': top5_pred}).to_csv('top5_blend.csv', index=False)

print("Rank blend")
from scipy.stats import rankdata

rank = np.zeros(len(test))
for m in models.keys():
    rank += rankdata(predictions[m]) / len(predictions[m])
rank /= len(models)
pd.DataFrame({'id': test_ids, 'loan_paid_back': rank}).to_csv('rank_blend.csv', index=False)

print("Stacked blend")
from sklearn.linear_model import LogisticRegression

oof_stack = np.column_stack([oof_preds[m] for m in models.keys()])
test_stack = np.column_stack([predictions[m] for m in models.keys()])

meta = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
meta.fit(oof_stack, y)
stacked = meta.predict_proba(test_stack)[:, 1]
stacked_cv = roc_auc_score(y, meta.predict_proba(oof_stack)[:, 1])

print(f"Stacked CV: {stacked_cv:.6f}")
pd.DataFrame({'id': test_ids, 'loan_paid_back': stacked}).to_csv('stacked_blend.csv', index=False)

print("\n" + "=" * 80)
print("FINAL")
print("=" * 80)
print(f"Best single: {df.iloc[0]['Model']} ({df.iloc[0]['CV']:.6f})")
print(f"Stacked: {stacked_cv:.6f}")

print("\nSubmissions:")
print("1. simple_blend.csv")
print("2. stacked_blend.csv")
print("3. top5_blend.csv")
print("4. weighted_blend.csv")
print("5. rank_blend.csv")

print("\n" + "=" * 80)
print(f"Features: {X.shape[1]}")
print("Target: 0.927+ Public")
print("=" * 80)