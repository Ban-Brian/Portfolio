"""Second patch: Weibull AFT, leave-one-city-out, elastic net survival,
decision-band CIs, MC dropout benchmark."""
import json

with open("yelp_restaurant_success.ipynb", "r") as f:
    nb = json.load(f)

cells = nb["cells"]

# Verify current structure
print(f"Cells before patch: {len(cells)}")
for i, c in enumerate(cells):
    preview = "".join(c["source"][:1]).strip()[:70] if c["source"] else "(empty)"
    print(f"  [{i}] {c['cell_type']:8s} | {preview}")

# ── A. Fix elastic net survival via SGDClassifier (insert after cell 14, survival clf) ──
enet_surv_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "enet_survival",
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- Elastic Net survival via SGDClassifier ---\n",
        "from sklearn.linear_model import SGDClassifier\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "\n",
        "sgd_enet = GridSearchCV(\n",
        "    SGDClassifier(loss='log_loss', penalty='elasticnet', max_iter=5000,\n",
        "                  random_state=SEED, class_weight='balanced'),\n",
        "    param_grid={'alpha': np.logspace(-5, -1, 20),\n",
        "                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},\n",
        "    cv=5, scoring='neg_log_loss', n_jobs=-1,\n",
        ").fit(Xtr, ytr_o)\n",
        "\n",
        "enet_surv_p = sgd_enet.predict_proba(Xte)[:, 1]\n",
        "enet_surv_row = {\n",
        '    "model": "ElasticNet-Surv",\n',
        '    "test_acc": accuracy_score(yte_o, enet_surv_p >= 0.5),\n',
        '    "test_logloss": log_loss(yte_o, enet_surv_p),\n',
        '    "test_auc": roc_auc_score(yte_o, enet_surv_p),\n',
        '    "ece": expected_calibration_error(yte_o, enet_surv_p),\n',
        '    "brier": brier_score_loss(yte_o, enet_surv_p),\n',
        "}\n",
        "print(f'Elastic Net survival: {enet_surv_row}')\n",
    ],
}
cells.insert(15, enet_surv_cell)

# ── B. Weibull AFT survival model (insert after hierarchical extras) ──
# Find the hier_extras cell
hier_extras_idx = None
for i, c in enumerate(cells):
    if c.get("id") == "hier_extras":
        hier_extras_idx = i
        break
assert hier_extras_idx is not None, "Could not find hier_extras cell"

weibull_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "weibull_aft",
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- Weibull AFT survival model with explicit censoring ---\n",
        "# age_years = time-to-event (or time-to-censoring)\n",
        "# is_open=1 means right-censored (event not observed), is_open=0 means event (closure)\n",
        "import warnings\n",
        "try:\n",
        "    from lifelines import WeibullAFTFitter\n",
        "    HAS_LIFELINES = True\n",
        "except ImportError:\n",
        "    HAS_LIFELINES = False\n",
        "    warnings.warn('lifelines not installed; skipping Weibull AFT')\n",
        "\n",
        "if HAS_LIFELINES:\n",
        "    # Build survival dataframe\n",
        "    struct_cols = d['struct_cols']\n",
        "    surv_df_tr = pd.DataFrame(d['X_struct'][d['train']], columns=struct_cols)\n",
        "    surv_df_tr['duration'] = surv_df_tr['age_years'].clip(lower=0.1)\n",
        "    # event=1 means closure observed; censored=0 means still open\n",
        "    surv_df_tr['event'] = 1 - d['is_open'][d['train']]\n",
        "    # Drop age_years from covariates (it's now the duration)\n",
        "    covar_cols = [c for c in struct_cols if c != 'age_years']\n",
        "\n",
        "    aft = WeibullAFTFitter()\n",
        "    aft.fit(surv_df_tr[covar_cols + ['duration', 'event']],\n",
        "            duration_col='duration', event_col='event')\n",
        "    aft.print_summary()\n",
        "\n",
        "    # Test-set evaluation: predict median survival time\n",
        "    surv_df_te = pd.DataFrame(d['X_struct'][d['test']], columns=struct_cols)\n",
        "    surv_df_te['duration'] = surv_df_te['age_years'].clip(lower=0.1)\n",
        "    surv_df_te['event'] = 1 - d['is_open'][d['test']]\n",
        "\n",
        "    median_surv = aft.predict_median(surv_df_te[covar_cols])\n",
        "    # Concordance index\n",
        "    from lifelines.utils import concordance_index\n",
        "    c_index = concordance_index(\n",
        "        surv_df_te['duration'], median_surv, surv_df_te['event']\n",
        "    )\n",
        "    print(f'\\nWeibull AFT concordance index: {c_index:.4f}')\n",
        "\n",
        "    # Survival probability at t=5 years for risk stratification\n",
        "    surv_5yr = aft.predict_survival_function(\n",
        "        surv_df_te[covar_cols], times=[5.0]\n",
        "    ).iloc[0].values\n",
        "    # Compare with binary classification\n",
        "    aft_pred_open = (surv_5yr > 0.5).astype(int)\n",
        "    aft_acc = accuracy_score(d['is_open'][d['test']], aft_pred_open)\n",
        "    print(f'Weibull AFT 5yr survival acc: {aft_acc:.4f}')\n",
        "\n",
        "    pd.DataFrame([{'concordance': c_index, 'acc_5yr': aft_acc}]).to_csv(\n",
        "        HIER / 'weibull_aft_results.csv', index=False)\n",
        "else:\n",
        "    print('Skipping Weibull AFT (install lifelines: pip install lifelines)')\n",
    ],
}
cells.insert(hier_extras_idx + 1, weibull_cell)

# ── C. Leave-one-city-out cross-validation (insert before head-to-head) ──
# Find head-to-head cell
h2h_idx = None
for i, c in enumerate(cells):
    src = "".join(c["source"])
    if "headline_comparison" in src:
        h2h_idx = i
        break
assert h2h_idx is not None

loco_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "leave_one_city_out",
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- Leave-one-city-out cross-validation ---\n",
        "loco_rows = []\n",
        "for holdout_city in CITIES:\n",
        "    # Split by city\n",
        "    train_mask = d['city'] != holdout_city\n",
        "    test_mask = d['city'] == holdout_city\n",
        "    Xtr_loco = d['X'][train_mask]\n",
        "    Xte_loco = d['X'][test_mask]\n",
        "    ytr_loco = d['stars'][train_mask]\n",
        "    yte_loco = d['stars'][test_mask]\n",
        "    ytr_o_loco = d['is_open'][train_mask]\n",
        "    yte_o_loco = d['is_open'][test_mask]\n",
        "\n",
        "    # Ridge regression\n",
        "    r_loco = RidgeCV(alphas=np.logspace(-3, 2, 20)).fit(Xtr_loco, ytr_loco)\n",
        "    r_rmse = rmse(yte_loco, r_loco.predict(Xte_loco))\n",
        "\n",
        "    # XGBoost\n",
        "    xgb_loco = xgb.XGBRegressor(\n",
        "        n_estimators=300, max_depth=6, learning_rate=0.05,\n",
        "        random_state=SEED, n_jobs=-1,\n",
        "    ).fit(Xtr_loco, ytr_loco, verbose=False)\n",
        "    x_rmse = rmse(yte_loco, xgb_loco.predict(Xte_loco))\n",
        "\n",
        "    loco_rows.append({\n",
        '        "holdout_city": holdout_city,\n',
        '        "n_test": int(test_mask.sum()),\n',
        '        "ridge_rmse": r_rmse,\n',
        '        "xgb_rmse": x_rmse,\n',
        "    })\n",
        "    print(f'{holdout_city:15s} (n={test_mask.sum():5d}): '\n",
        "          f'Ridge={r_rmse:.4f}  XGB={x_rmse:.4f}')\n",
        "\n",
        "loco_df = pd.DataFrame(loco_rows)\n",
        "loco_df.to_csv(FINAL / 'leave_one_city_out.csv', index=False)\n",
        "print(f'\\nMean LOCO Ridge RMSE: {loco_df[\"ridge_rmse\"].mean():.4f}')\n",
        "print(f'Mean LOCO XGB RMSE:   {loco_df[\"xgb_rmse\"].mean():.4f}')\n",
    ],
}
cells.insert(h2h_idx, loco_cell)

# ── D. MC dropout comparison (insert after map_temp_coverage cell) ──
mc_drop_idx = None
for i, c in enumerate(cells):
    if c.get("id") == "map_temp_coverage":
        mc_drop_idx = i + 1
        break
assert mc_drop_idx is not None

mc_dropout_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "mc_dropout",
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- MC Dropout uncertainty benchmark ---\n",
        "class MCDropoutNN(nn.Module):\n",
        "    def __init__(self, in_dim, hidden=128, drop_p=0.2):\n",
        "        super().__init__()\n",
        "        self.fc1 = nn.Linear(in_dim, hidden)\n",
        "        self.fc2 = nn.Linear(hidden, hidden)\n",
        "        self.mean_head = nn.Linear(hidden, 1)\n",
        "        self.logit_head = nn.Linear(hidden, 1)\n",
        "        self.drop = nn.Dropout(drop_p)\n",
        "        self.act = nn.ReLU()\n",
        "\n",
        "    def forward(self, x):\n",
        "        h = self.drop(self.act(self.fc1(x)))\n",
        "        h = self.drop(self.act(self.fc2(h)))\n",
        "        return self.mean_head(h).squeeze(-1), self.logit_head(h).squeeze(-1)\n",
        "\n",
        "# Train MC dropout NN (same architecture minus Bayesian weights)\n",
        "mc_nn = MCDropoutNN(Xtr_t.shape[1])\n",
        "opt_mc = torch.optim.Adam(mc_nn.parameters(), lr=1e-3)\n",
        "for epoch in range(150):\n",
        "    mc_nn.train()\n",
        "    for xb, ysb, yob in loader:\n",
        "        mu, logit = mc_nn(xb)\n",
        "        loss = nn.MSELoss()(mu, ysb) + nn.BCEWithLogitsLoss()(logit, yob)\n",
        "        opt_mc.zero_grad()\n",
        "        loss.backward()\n",
        "        opt_mc.step()\n",
        "\n",
        "# MC dropout inference (keep dropout on)\n",
        "mc_nn.train()  # keep dropout active\n",
        "mc_mus = []\n",
        "for _ in range(50):\n",
        "    with torch.no_grad():\n",
        "        mu_mc, _ = mc_nn(Xte_t)\n",
        "        mc_mus.append(mu_mc.numpy())\n",
        "mc_mus = np.stack(mc_mus)\n",
        "mc_mean = mc_mus.mean(axis=0)\n",
        "mc_epist_var = mc_mus.var(axis=0)\n",
        "\n",
        "mc_rmse = rmse(yte_s_np, mc_mean)\n",
        "# 90% PI coverage using epistemic std only (no learned aleatoric)\n",
        "mc_std = np.sqrt(mc_epist_var)\n",
        "mc_lo90 = mc_mean - 1.645 * mc_std\n",
        "mc_hi90 = mc_mean + 1.645 * mc_std\n",
        "mc_coverage = float(np.mean((yte_s_np >= mc_lo90) & (yte_s_np <= mc_hi90)))\n",
        "\n",
        "# Epistemic-error correlation\n",
        "mc_abs_err = np.abs(yte_s_np - mc_mean)\n",
        "mc_r = float(np.corrcoef(mc_std, mc_abs_err)[0, 1])\n",
        "\n",
        "print(f'MC Dropout RMSE: {mc_rmse:.4f}')\n",
        "print(f'MC Dropout 90% PI coverage: {mc_coverage:.4f}')\n",
        "print(f'MC Dropout epistemic-error r: {mc_r:.4f}, R²: {mc_r**2:.4f}')\n",
        "print(f'\\nComparison:')\n",
        "print(f'  VBNN     coverage={coverage:.4f}  epist-error r={r_corr:.4f}')\n",
        "print(f'  MC Drop  coverage={mc_coverage:.4f}  epist-error r={mc_r:.4f}')\n",
    ],
}
cells.insert(mc_drop_idx, mc_dropout_cell)

# ── E. Add CIs to decision bands (find and update the band cell) ──
for i, c in enumerate(cells):
    src = "".join(c["source"])
    if "survival_decision_bands" in src and "band_rows" in src:
        # Add CI computation to the band loop
        old_src = c["source"]
        new_src = []
        for line in old_src:
            new_src.append(line)
            # After the observed_close_rate line, add CI
            if '"observed_close_rate"' in line:
                new_src.append(
                    '        "closure_ci_lo": float(max(0, (1-yte[m].mean())\n'
                )
                new_src.append(
                    '            - 1.96 * np.sqrt((1-yte[m].mean()) * yte[m].mean() / m.sum()))),\n'
                )
                new_src.append(
                    '        "closure_ci_hi": float(min(1, (1-yte[m].mean())\n'
                )
                new_src.append(
                    '            + 1.96 * np.sqrt((1-yte[m].mean()) * yte[m].mean() / m.sum()))),\n'
                )
        c["source"] = new_src
        break

# Save
with open("yelp_restaurant_success.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"\nCells after patch: {len(cells)}")
print("Notebook v2 patch complete.")
