"""
Comprehensive fixer for the Midterm MNIST notebook.
Addresses ALL issues from the line-by-line review:
  - Remove deprecated multi_class='multinomial'
  - Replace penalty=None with C=np.inf
  - Fix axis('off') label bug in coef maps (cell 18)
  - Use X_train_val for binary subset
  - Fix posterior predictive: average sigmoids across samples
  - Store probs_std, compute Bayesian log-loss
  - Fix uncertainty ranking to use probs_std
  - Non-centered horseshoe parameterization
  - Autoscale xlim
  - Drop redundant val split reassembly
  - Add convergence diagnostics (R-hat, ESS, divergences via ArviZ)
  - Drop the inner val split entirely (train/test + CV only)
  - Add savefig calls for all figures
"""
import json, re, os

SRC = "/Users/brianbutler/Portfolio/In Progress Works/PML/Midterm/mnist_classification_project.ipynb"
DEST = SRC

with open(SRC) as f:
    nb = json.load(f)

cells = nb["cells"]

def src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s

def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)

def find_cell(marker):
    for c in cells:
        if c["cell_type"] == "code" and marker in src(c):
            return c
    return None

fixes = []

# ══════════════════════════════════════════════════════════════════
# FIX 1: Data split — drop the inner val split entirely
#   Keep only X_train_val / X_test.  Remove X_train/X_val creation.
# ══════════════════════════════════════════════════════════════════
c_data = find_cell("# Train/Val split")
if c_data:
    s = src(c_data)
    # Remove the val split lines
    s = re.sub(
        r'# Train/Val split\n.*?X_train, X_val.*?\n.*?\n.*?\n',
        '',
        s,
        flags=re.DOTALL
    )
    # Remove val print line
    s = re.sub(r'.*Val:.*X_val.*\n', '', s)
    # Fix train print to use X_train_val
    s = s.replace(
        "print(f\"  Train: {len(X_train)} ({len(X_train)/N_SAMPLES*100:.1f}%)\")",
        "print(f\"  Train: {len(X_train_val)} ({len(X_train_val)/N_SAMPLES*100:.1f}%)\")"
    )
    set_src(c_data, s)
    fixes.append("FIX 1: Dropped inner val split — train/test + CV only")

# Also fix the sample digit visualization that uses y_train
c_digits = find_cell("y_train == digit")
if c_digits:
    s = src(c_digits)
    s = s.replace("y_train == digit", "y_train_val == digit")
    s = s.replace("X_train[idx]", "X_train_val[idx]")
    set_src(c_digits, s)
    fixes.append("FIX 1b: Sample digit viz uses X_train_val")

# ══════════════════════════════════════════════════════════════════
# FIX 2: Remove redundant X_train_val reassembly in cell 7
# ══════════════════════════════════════════════════════════════════
c7 = find_cell("X_train_val = np.vstack")
if c7:
    new = (
        "# Initialize result containers — X_train_val already defined in Part A\n"
        "results = {}\n"
        "models = {}"
    )
    set_src(c7, new)
    fixes.append("FIX 2: Removed redundant X_train_val reassembly")

# ══════════════════════════════════════════════════════════════════
# FIX 3: penalty=None → C=np.inf, remove all multi_class='multinomial'
# ══════════════════════════════════════════════════════════════════
for c in cells:
    if c["cell_type"] != "code":
        continue
    s = src(c)
    changed = False

    # Remove multi_class='multinomial' everywhere
    if "multi_class='multinomial'" in s:
        s = re.sub(r",?\s*multi_class='multinomial'", "", s)
        changed = True

    # Replace penalty=None with C=np.inf
    if re.search(r'LogisticRegression\([^)]*penalty=None', s):
        s = re.sub(
            r"LogisticRegression\(penalty=None,",
            "LogisticRegression(C=np.inf,",
            s
        )
        changed = True

    if changed:
        set_src(c, s)

fixes.append("FIX 3: Removed multi_class, replaced penalty=None with C=np.inf")

# ══════════════════════════════════════════════════════════════════
# FIX 4: Coefficient map labels — axis('off') kills ylabel
# ══════════════════════════════════════════════════════════════════
c_coef = find_cell("ax.axis('off')\n        if digit == 0:")
if c_coef:
    s = src(c_coef)
    old = (
        "        ax.axis('off')\n"
        "        if digit == 0:\n"
        "            ax.set_ylabel(name, fontsize=10, rotation=0, ha='right', va='center')\n"
    )
    new = (
        "        ax.set_xticks([])\n"
        "        ax.set_yticks([])\n"
        "        for spine in ax.spines.values():\n"
        "            spine.set_visible(False)\n"
        "        if digit == 0:\n"
        "            ax.text(-0.15, 0.5, name, fontsize=10, rotation=0,\n"
        "                    ha='right', va='center', transform=ax.transAxes)\n"
    )
    s = s.replace(old, new)
    # Fix the caveat comment
    s = s.replace(
        "# Visualize coefficients as images\n",
        "# Visualize coefficients as images\n"
        "# Note: StandardScaler sets zero-variance corner pixels to 0 (no amplification);\n"
        "# low-but-nonzero-variance pixels at stroke edges can be amplified.\n"
        "# Spatial patterns are qualitatively valid for comparing regularization.\n",
    )
    set_src(c_coef, s)
    fixes.append("FIX 4: Coefficient map labels fixed, caveat corrected")

# ══════════════════════════════════════════════════════════════════
# FIX 5: Binary subset uses X_train_val (not X_train)
# ══════════════════════════════════════════════════════════════════
c_bin = find_cell("train_mask = (y_train == DIGIT1)")
if c_bin:
    new = (
        "# Prepare binary subset from X_train_val (same pool as Part B)\n"
        "DIGIT1, DIGIT2 = 3, 8\n"
        "\n"
        "train_mask = (y_train_val == DIGIT1) | (y_train_val == DIGIT2)\n"
        "test_mask  = (y_test == DIGIT1)      | (y_test == DIGIT2)\n"
        "\n"
        "X_train_bin = X_train_val[train_mask]\n"
        "y_train_bin = (y_train_val[train_mask] == DIGIT2).astype(int)\n"
        "\n"
        "X_test_bin = X_test[test_mask]\n"
        "y_test_bin = (y_test[test_mask] == DIGIT2).astype(int)\n"
        "\n"
        'print(f"Binary classification: {DIGIT1} vs {DIGIT2}")\n'
        'print(f"Train: {len(y_train_bin)} (class 0: {(y_train_bin==0).sum()}, class 1: {(y_train_bin==1).sum()})")\n'
        'print(f"Test:  {len(y_test_bin)}")'
    )
    set_src(c_bin, new)
    fixes.append("FIX 5: Binary subset now uses X_train_val")

# ══════════════════════════════════════════════════════════════════
# FIX 5b: PCA cell — print actual explained variance
# ══════════════════════════════════════════════════════════════════
c_pca = find_cell("N_COMPONENTS = 30")
if c_pca:
    s = src(c_pca)
    if 'Variance explained:' in s and 'pca_var_explained' not in s:
        s = s.replace(
            'print(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}")',
            'pca_var_explained = pca.explained_variance_ratio_.sum()\n'
            'print(f"PCA variance explained by {N_COMPONENTS} components: {pca_var_explained:.4f} ({pca_var_explained*100:.1f}%)")'
        )
        set_src(c_pca, s)
        fixes.append("FIX 5b: PCA cell stores and prints exact variance explained")

# ══════════════════════════════════════════════════════════════════
# FIX 6-8: All three Bayesian models — proper posterior predictive
#   Average sigmoid across all posterior samples, store probs_std,
#   compute log_loss. Also add convergence diagnostics.
# ══════════════════════════════════════════════════════════════════

def make_pp_block(trace_var, result_key, model_var):
    """Generate proper posterior-predictive + convergence diagnostics block."""
    return (
        f"\n# ── Convergence diagnostics ──\n"
        f"summary = az.summary({trace_var}, var_names=['beta', 'intercept'])\n"
        f"print(f\"R-hat range: [{{summary['r_hat'].min():.3f}}, {{summary['r_hat'].max():.3f}}]\")\n"
        f"print(f\"ESS bulk min: {{summary['ess_bulk'].min():.0f}}\")\n"
        f"if hasattr({trace_var}, 'sample_stats'):\n"
        f"    divs = {trace_var}.sample_stats.get('diverging', None)\n"
        f"    if divs is not None:\n"
        f"        print(f\"Divergences: {{int(divs.sum().values)}}\")\n"
        f"\n"
        f"# ── Posterior predictive: E[sigma(eta)] across all S samples ──\n"
        f"# This is the correct Bayesian predictive, NOT the plug-in sigma(E[eta])\n"
        f"post = {trace_var}.posterior\n"
        f"beta_s = post['beta'].stack(s=('chain', 'draw')).values     # (Q, S)\n"
        f"a_s    = post['intercept'].stack(s=('chain', 'draw')).values # (S,)\n"
        f"eta_s  = X_test_pca @ beta_s + a_s                          # (N_test, S)\n"
        f"probs_s = 1 / (1 + np.exp(-eta_s))                          # (N_test, S)\n"
        f"probs     = probs_s.mean(axis=1)  # E[sigma(eta)]\n"
        f"probs_std = probs_s.std(axis=1)   # posterior predictive std\n"
        f"preds = (probs > 0.5).astype(int)\n"
        f"\n"
        f"bayesian_results['{result_key}'] = {{\n"
        f"    'accuracy':  accuracy_score(y_test_bin, preds),\n"
        f"    'log_loss':  log_loss(y_test_bin, probs),\n"
        f"    'trace':     {trace_var},\n"
        f"    'probs':     probs,\n"
        f"    'probs_std': probs_std,\n"
        f"    'beta_mean': post['beta'].mean(dim=['chain', 'draw']).values,\n"
        f"    'beta_std':  post['beta'].std(dim=['chain', 'draw']).values,\n"
        f"}}\n"
        f"\n"
        f"print(f\"Test Accuracy: {{bayesian_results['{result_key}']['accuracy']:.4f}}\")\n"
        f"print(f\"Log-Loss:     {{bayesian_results['{result_key}']['log_loss']:.4f}}\")"
    )

# --- Gaussian Prior ---
c_gauss = find_cell("# (i) Gaussian Prior")
if c_gauss:
    s = src(c_gauss)
    cut = s.find("# Predictions")
    if cut == -1:
        cut = s.find("beta_mean = trace_gaussian")
    if cut != -1:
        preamble = s[:cut]
        set_src(c_gauss, preamble + make_pp_block("trace_gaussian", "Gaussian Prior", "model_gaussian"))
        fixes.append("FIX 6: Gaussian prior — proper posterior predictive + diagnostics")

# --- Laplace Prior ---
c_lap = find_cell("# (ii) Laplace Prior")
if c_lap:
    s = src(c_lap)
    cut = s.find("beta_mean = trace_laplace")
    if cut != -1:
        preamble = s[:cut]
        set_src(c_lap, preamble + make_pp_block("trace_laplace", "Laplace Prior", "model_laplace"))
        fixes.append("FIX 7: Laplace prior — proper posterior predictive + diagnostics")

# --- Horseshoe Prior (also non-centered parameterization) ---
c_hs = find_cell("# (Optional) Horseshoe Prior")
if c_hs:
    new_hs = (
        "# (iii) Horseshoe Prior — non-centered parameterization\n"
        "# beta = z * lam * tau reduces NUTS divergences vs centered form\n"
        'print("Fitting: Bayesian Logistic Regression with Horseshoe Prior")\n'
        "\n"
        "with pm.Model() as model_horseshoe:\n"
        "    tau = pm.HalfCauchy('tau', beta=1)\n"
        "    lam = pm.HalfCauchy('lam', beta=1, shape=N_COMPONENTS)\n"
        "    intercept = pm.Normal('intercept', mu=0, sigma=10)\n"
        "    z    = pm.Normal('z', 0, 1, shape=N_COMPONENTS)\n"
        "    beta = pm.Deterministic('beta', z * lam * tau)\n"
        "    eta = intercept + pm.math.dot(X_train_pca, beta)\n"
        "    p = pm.math.sigmoid(eta)\n"
        "    y_obs = pm.Bernoulli('y_obs', p=p, observed=y_train_bin)\n"
        "    trace_horseshoe = pm.sample(draws=1000, tune=500, cores=2,\n"
        "                                 random_seed=RANDOM_STATE, return_inferencedata=True,\n"
        "                                 target_accept=0.9)\n"
        + make_pp_block("trace_horseshoe", "Horseshoe Prior", "model_horseshoe")
    )
    set_src(c_hs, new_hs)
    fixes.append("FIX 8: Horseshoe — non-centered + proper posterior predictive + diagnostics")

# ══════════════════════════════════════════════════════════════════
# FIX 9: Uncertainty cell — rank by probs_std, not point-estimate hack
# ══════════════════════════════════════════════════════════════════
c_unc = find_cell("1 - np.abs(probs - 0.5)")
if c_unc:
    new = (
        '# Identify difficult examples by posterior predictive std (genuinely Bayesian)\n'
        '# probs_std reflects how much posterior samples disagree on sigma(eta)\n'
        'print("\\nDifficult Examples (highest posterior predictive std):")\n'
        'for name, res in bayesian_results.items():\n'
        '    probs     = res["probs"]\n'
        '    std_vals  = res["probs_std"]\n'
        '    difficult = np.argsort(std_vals)[-5:][::-1]  # top 5 by std\n'
        '    print(f"\\n{name}:")\n'
        '    print(f"  {\'Idx\':>5} {\'P(y=1)\':>8} {\'Pred Std\':>9} {\'True\':>5} {\'Correct\':>8}")\n'
        '    for idx in difficult:\n'
        '        correct = "yes" if (probs[idx] > 0.5) == y_test_bin[idx] else "NO"\n'
        '        print(f"  {idx:>5} {probs[idx]:>8.3f} {std_vals[idx]:>9.3f} {y_test_bin[idx]:>5} {correct:>8}")'
    )
    set_src(c_unc, new)
    fixes.append("FIX 9: Uncertainty ranked by probs_std, not 1-|p-0.5|*2")

# ══════════════════════════════════════════════════════════════════
# FIX 10: Final summary table — use actual Bayesian log-loss
# ══════════════════════════════════════════════════════════════════
c_summary = find_cell("[np.nan] * len(bayesian_results)")
if c_summary:
    s = src(c_summary)
    s = s.replace(
        "'Log-Loss': [np.nan] * len(bayesian_results)",
        "'Log-Loss': [r.get('log_loss', np.nan) for r in bayesian_results.values()]"
    )
    set_src(c_summary, s)
    fixes.append("FIX 10: Bayesian log-loss now computed in summary table")

# ══════════════════════════════════════════════════════════════════
# FIX 11: Comparison plot — autoscale xlim, include Bayesian bars
# ══════════════════════════════════════════════════════════════════
c_comp = find_cell("axes[0].set_xlim([0.85, 1.0])")
if c_comp:
    s = src(c_comp)
    s = s.replace(
        "axes[0].set_xlim([0.85, 1.0])",
        "axes[0].set_xlim([max(0.0, min(all_accs) - 0.03), 1.01])"
    )
    set_src(c_comp, s)
    fixes.append("FIX 11: Autoscaled xlim")

# ══════════════════════════════════════════════════════════════════
# Add savefig calls before every plt.show()
# ══════════════════════════════════════════════════════════════════
FIGURE_MAP = {
    "Sample Digits from MNIST":           "outputs/sample_digits.png",
    "Coefficient Maps by Digit":          "outputs/coef_maps.png",
    "Confusion Matrices":                 "outputs/confusion_matrices.png",
    "Shrinkage Pattern Across Priors":    "outputs/bayesian_shrinkage.png",
    "Model Comparison":                   "outputs/accuracy_logloss_comparison.png",
}

for c in cells:
    if c["cell_type"] != "code":
        continue
    s = src(c)
    for marker, figname in FIGURE_MAP.items():
        if marker in s and f"savefig('{figname}')" not in s:
            save = (
                f"import os; os.makedirs('outputs', exist_ok=True)\n"
                f"plt.savefig('{figname}', dpi=150, bbox_inches='tight')\n"
            )
            s = s.replace("plt.show()", save + "plt.show()")
            set_src(c, s)
            fixes.append(f"SAVEFIG: {figname}")
            break

# ══════════════════════════════════════════════════════════════════
# Insert outputs-setup cell if not present
# ══════════════════════════════════════════════════════════════════
full_src = "\n".join(src(c) for c in cells)
if "os.makedirs" not in full_src or "Output directory" not in full_src:
    setup = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "os.makedirs('outputs', exist_ok=True)\n",
            "print(f'Output directory: {os.path.abspath(\"outputs\")}')\n"
        ]
    }
    cells.insert(2, setup)
    fixes.append("Setup cell: outputs/ directory creation")

# ══════════════════════════════════════════════════════════════════
# Write
# ══════════════════════════════════════════════════════════════════
with open(DEST, "w") as f:
    json.dump(nb, f, indent=1)

print("Applied fixes:")
for fix in fixes:
    print(f"  ✓ {fix}")
print(f"\nNotebook written to: {DEST}")
