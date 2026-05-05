import os
import subprocess
import json
import warnings
warnings.filterwarnings("ignore")
sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path']).decode().strip()
os.environ['CPATH'] = f"{sdk_path}/usr/include/c++/v1:{sdk_path}/usr/include"
os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64,optimizer=fast_compile'

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import pymc as pm
import arviz as az

SEED = 42
np.random.seed(SEED)
N_SAMPLES = 5000
N_COMPONENTS = 30

# Load data
print("Loading data...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_full, y_full = mnist.data, mnist.target.astype(int)
idx = np.random.choice(len(X_full), N_SAMPLES, replace=False)
X, y = X_full[idx], y_full[idx]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# Train RBF SVM
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train_val)
X_te_s = scaler.transform(X_test)

svm = SVC(C=10, gamma='scale')
svm.fit(X_tr_s, y_train_val)
y_pred = svm.predict(X_te_s)

print("\n--- RBF SVM Metrics ---")
print("Support Vectors total:", svm.support_.shape[0])
print(classification_report(y_test, y_pred, digits=3))

# Part C Bayesian
print("\n--- Bayesian Data prep ---")
DIGIT1, DIGIT2 = 3, 8
train_mask = (y_train_val == DIGIT1) | (y_train_val == DIGIT2)
test_mask  = (y_test == DIGIT1)      | (y_test == DIGIT2)
X_train_bin = X_train_val[train_mask]
y_train_bin = (y_train_val[train_mask] == DIGIT2).astype(int)
# scale & PCA for binary
scaler2 = StandardScaler()
X_tr2_s = scaler2.fit_transform(X_train_bin)
pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
X_tr2_pca = pca.fit_transform(X_tr2_s)
pca_var = pca.explained_variance_ratio_.sum()

print("PCA var:", pca_var)

# Bayesian Model: Gaussian
with pm.Model() as model_g:
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=1, shape=N_COMPONENTS)
    eta = intercept + pm.math.dot(X_tr2_pca, beta)
    p = pm.math.sigmoid(eta)
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train_bin)
    trace_g = pm.sample(draws=1000, tune=500, cores=1, chains=2, random_seed=SEED, progressbar=False)

# Bayesian Model: Laplace 
with pm.Model() as model_l:
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    beta = pm.Laplace("beta", mu=0, b=1, shape=N_COMPONENTS)
    eta = intercept + pm.math.dot(X_tr2_pca, beta)
    p = pm.math.sigmoid(eta)
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train_bin)
    trace_l = pm.sample(draws=1000, tune=500, cores=1, chains=2, random_seed=SEED, progressbar=False)

# Bayesian Model: Horseshoe
with pm.Model() as model_h:
    tau = pm.HalfCauchy("tau", beta=1)
    lam = pm.HalfCauchy("lam", beta=1, shape=N_COMPONENTS)
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    z = pm.Normal("z", 0, 1, shape=N_COMPONENTS)
    beta = pm.Deterministic("beta", z * lam * tau)
    eta = intercept + pm.math.dot(X_tr2_pca, beta)
    p = pm.math.sigmoid(eta)
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train_bin)
    trace_h = pm.sample(draws=1000, tune=500, cores=1, chains=2, random_seed=SEED, target_accept=0.9, progressbar=False)

def print_hdi(trace, name):
    summ = az.summary(trace, var_names=["beta"], hdi_prob=0.94)
    # Get top 5 by absolute mean
    summ['abs_mean'] = np.abs(summ['mean'])
    top5 = summ.sort_values('abs_mean', ascending=False).head(5)
    print(f"\n--- {name} Top 5 HDIs ---")
    print(top5[['mean', 'hdi_3%', 'hdi_97%']])

print_hdi(trace_g, "Gaussian")
print_hdi(trace_l, "Laplace")
print_hdi(trace_h, "Horseshoe")
