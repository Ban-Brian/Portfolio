"""
Apply all bug-fixes and methodological corrections to the MNIST classification notebook.
Fixes are applied directly to the cell source strings without requiring notebook execution.
"""
import json, re, copy, sys

SRC  = "/Users/brianbutler/Portfolio/mnist_classification_project.ipynb"
DEST = "/Users/brianbutler/Portfolio/mnist_classification_project_fixed.ipynb"

with open(SRC) as f:
    nb = json.load(f)

cells = nb["cells"]

def src(cell):
    """Return cell source as a single string."""
    return "".join(cell["source"])

def set_src(cell, text):
    """Replace cell source (split into lines as Jupyter stores them)."""
    lines = text.splitlines(keepends=True)
    # Ensure last line has no trailing newline duplicate
    cell["source"] = lines

def find_cell(marker):
    """Return first code cell whose source contains *marker*."""
    for c in cells:
        if c["cell_type"] == "code" and marker in src(c):
            return c
    return None

# ──────────────────────────────────────────────────────────────────
# FIX 1 – Cell 7: remove redundant X_train_val reassignment
# ──────────────────────────────────────────────────────────────────
c7 = find_cell("X_train_val = np.vstack([X_train, X_val])")
if c7:
    new = (
        "# X_train_val is already assembled in Part A (train+val).\n"
        "# We reference it directly here — no need to reassemble.\n"
        "results = {}\n"
        "models = {}"
    )
    set_src(c7, new)
    print("FIX 1 applied: removed redundant X_train_val reassignment in cell 7")

# ──────────────────────────────────────────────────────────────────
# FIX 2 – Cell 8: penalty=None → C=np.inf, remove multi_class
# ──────────────────────────────────────────────────────────────────
c8 = find_cell("penalty=None")
if c8:
    s = src(c8)
    # Replace the LogisticRegression(...) call
    s = re.sub(
        r"LogisticRegression\(penalty=None,\s*solver='lbfgs',\s*max_iter=1000,\s*multi_class='multinomial',\s*random_state=RANDOM_STATE\)",
        "LogisticRegression(C=np.inf, solver='lbfgs', max_iter=1000,\n"
        "                                random_state=RANDOM_STATE)",
        s,
    )
    # Update the print comment
    s = s.replace(
        "# (i) Unpenalized Logistic Regression",
        "# (i) Unpenalized Logistic Regression\n"
        "# C=np.inf is equivalent to no regularization (replaces deprecated penalty=None)",
    )
    set_src(c8, s)
    print("FIX 2 applied: penalty=None → C=np.inf, removed multi_class (cell 8)")

# ──────────────────────────────────────────────────────────────────
# FIX 3 – Cells 9, 10, 11: remove multi_class='multinomial'
# ──────────────────────────────────────────────────────────────────
for marker in [
    "# (ii) Ridge-Penalized",
    "# (iii) Lasso-Penalized",
    "# (Bonus) Elastic Net",
]:
    c = find_cell(marker)
    if c:
        s = src(c)
        s = re.sub(r",\s*multi_class='multinomial'", "", s)
        set_src(c, s)
        print(f"FIX 3 applied: removed multi_class from '{marker}' cell")

# ──────────────────────────────────────────────────────────────────
# FIX 4 – Cell 18: axis('off') hides ylabel → use ax.text + selective tick hide
# ──────────────────────────────────────────────────────────────────
c18 = find_cell("ax.axis('off')\n        if digit == 0:")
if c18:
    s = src(c18)
    old_block = (
        "        ax.axis('off')\n"
        "        if digit == 0:\n"
        "            ax.set_ylabel(name, fontsize=10, rotation=0, ha='right', va='center')\n"
    )
    new_block = (
        "        # Hide ticks/spines without axis('off'), which also removes set_ylabel\n"
        "        ax.set_xticks([])\n"
        "        ax.set_yticks([])\n"
        "        for spine in ax.spines.values():\n"
        "            spine.set_visible(False)\n"
        "        if digit == 0:\n"
        "            # ax.text survives after we suppress ticks\n"
        "            ax.text(-0.15, 0.5, name, fontsize=10, rotation=0,\n"
        "                    ha='right', va='center', transform=ax.transAxes)\n"
    )
    s = s.replace(old_block, new_block)
    # Also add scaling caveat comment at top of the cell
    s = s.replace(
        "# Visualize coefficients as images\n",
        "# Visualize coefficients as images\n"
        "# Note: pixels were StandardScaler-transformed, so low-variance background pixels\n"
        "# can dominate after scaling. Spatial patterns are qualitatively valid.\n",
    )
    set_src(c18, s)
    print("FIX 4 applied: axis('off') label bug fixed (cell 18)")

# ──────────────────────────────────────────────────────────────────
# FIX 5 – Cell 22: use X_train_val instead of X_train for binary subset
# ──────────────────────────────────────────────────────────────────
c22 = find_cell("train_mask = (y_train == DIGIT1)")
if c22:
    new = (
        "# Prepare binary subset\n"
        "# Use X_train_val (full train set) so Bayesian models see as much data\n"
        "# as the frequentist models in Part B — consistent comparison\n"
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
        'print(f"Train: {len(y_train_bin)}, Test: {len(y_test_bin)}")'
    )
    set_src(c22, new)
    print("FIX 5 applied: binary subset now uses X_train_val (cell 22)")

# ──────────────────────────────────────────────────────────────────
# Helper: posterior-predictive block (replaces plug-in prediction)
# ──────────────────────────────────────────────────────────────────
def pp_block(trace_var, result_key, extra_comment=""):
    """Generate the corrected posterior-predictive snippet."""
    return (
        f"{extra_comment}"
        f"# True posterior predictive: E[sigmoid(eta)] over posterior samples,\n"
        f"# not sigmoid(E[eta]) which is just a plug-in / MAP approximation\n"
        f"post    = {trace_var}.posterior\n"
        f"beta_s  = post['beta'].stack(s=('chain', 'draw')).values      # (N_COMPONENTS, S)\n"
        f"a_s     = post['intercept'].stack(s=('chain', 'draw')).values  # (S,)\n"
        f"eta_s   = X_test_pca @ beta_s + a_s                           # (N_test, S)\n"
        f"probs_s = 1 / (1 + np.exp(-eta_s))                            # (N_test, S)\n"
        f"probs      = probs_s.mean(axis=1)   # posterior predictive mean\n"
        f"probs_std  = probs_s.std(axis=1)    # predictive uncertainty\n"
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
        f'print(f"Test Accuracy: {{bayesian_results[\'{result_key}\'][\'accuracy\']:.4f}}")\n'
        f'print(f"Log-Loss:     {{bayesian_results[\'{result_key}\'][\'log_loss\']:.4f}}")'
    )

# ──────────────────────────────────────────────────────────────────
# FIX 6 – Cell 25: Gaussian Prior — fix posterior predictive
# ──────────────────────────────────────────────────────────────────
c25 = find_cell("# (i) Gaussian Prior")
if c25:
    s = src(c25)
    # Keep everything up to (and including) the pm.sample line
    # Cut off from "# Predictions" onward and replace
    cutoff = "# Predictions\n"
    idx = s.find(cutoff)
    if idx == -1:
        cutoff = "# predictions\n"
        idx = s.find(cutoff)
    if idx != -1:
        preamble = s[:idx]
        replacement = preamble + pp_block("trace_gaussian", "Gaussian Prior")
        set_src(c25, replacement)
        print("FIX 6 applied: Gaussian Prior posterior predictive (cell 25)")

# ──────────────────────────────────────────────────────────────────
# FIX 7 – Cell 26: Laplace Prior — fix posterior predictive
# ──────────────────────────────────────────────────────────────────
c26 = find_cell("# (ii) Laplace Prior")
if c26:
    s = src(c26)
    cutoff = "beta_mean = trace_laplace"
    idx = s.find(cutoff)
    if idx != -1:
        preamble = s[:idx]
        replacement = preamble + pp_block("trace_laplace", "Laplace Prior")
        set_src(c26, replacement)
        print("FIX 7 applied: Laplace Prior posterior predictive (cell 26)")

# ──────────────────────────────────────────────────────────────────
# FIX 8 – Cell 27: Horseshoe — non-centered + fix posterior predictive
# ──────────────────────────────────────────────────────────────────
c27 = find_cell("# (Optional) Horseshoe Prior")
if c27:
    new_horseshoe = (
        "# (Optional) Horseshoe Prior\n"
        "# Non-centered parameterization (z * lam * tau) reduces divergences with NUTS\n"
        'print("Fitting: Bayesian Logistic Regression with Horseshoe Prior")\n'
        "\n"
        "with pm.Model() as model_horseshoe:\n"
        "    tau = pm.HalfCauchy('tau', beta=1)\n"
        "    lam = pm.HalfCauchy('lam', beta=1, shape=N_COMPONENTS)\n"
        "    intercept = pm.Normal('intercept', mu=0, sigma=10)\n"
        "    # Non-centered: z ~ N(0,1) auxiliary; beta = z * lam * tau\n"
        "    z    = pm.Normal('z', 0, 1, shape=N_COMPONENTS)\n"
        "    beta = pm.Deterministic('beta', z * lam * tau)\n"
        "    eta = intercept + pm.math.dot(X_train_pca, beta)\n"
        "    p = pm.math.sigmoid(eta)\n"
        "    y_obs = pm.Bernoulli('y_obs', p=p, observed=y_train_bin)\n"
        "    trace_horseshoe = pm.sample(draws=1000, tune=500, cores=2,\n"
        "                                 random_seed=RANDOM_STATE, return_inferencedata=True,\n"
        "                                 target_accept=0.9)\n"
        "\n"
        + pp_block("trace_horseshoe", "Horseshoe Prior")
    )
    set_src(c27, new_horseshoe)
    print("FIX 8 applied: Horseshoe non-centered + posterior predictive (cell 27)")

# ──────────────────────────────────────────────────────────────────
# FIX 9 – Cell 30: uncertainty ranked by probs_std not point-estimate
# ──────────────────────────────────────────────────────────────────
c30 = find_cell("uncertainty = 1 - np.abs(probs - 0.5) * 2")
if c30:
    new = (
        "# Uncertainty for difficult examples — ranked by posterior predictive std,\n"
        "# which is a genuinely Bayesian notion of difficulty\n"
        'print("\\nUncertainty for Difficult Examples (ranked by posterior predictive std):")\n'
        "for name, res in bayesian_results.items():\n"
        "    probs      = res['probs']\n"
        "    std_scores = res['probs_std']\n"
        "    difficult  = np.argsort(std_scores)[-5:]  # highest posterior std = most uncertain\n"
        '    print(f"\\n{name}:")\n'
        "    for idx in difficult:\n"
        "        print(f\"  Example {idx}: P(class=1)={probs[idx]:.3f}, \"\n"
        "              f\"Pred Std={std_scores[idx]:.3f}, True={y_test_bin[idx]}\")"
    )
    set_src(c30, new)
    print("FIX 9 applied: uncertainty ranked by probs_std (cell 30)")

# ──────────────────────────────────────────────────────────────────
# FIX 10 – Cell 32: Bayesian log-loss computed (not hardcoded NaN)
# ──────────────────────────────────────────────────────────────────
c32 = find_cell("'Log-Loss': [np.nan] * len(bayesian_results)")
if c32:
    s = src(c32)
    s = s.replace(
        "'Log-Loss': [np.nan] * len(bayesian_results)",
        "'Log-Loss': [r.get('log_loss', np.nan) for r in bayesian_results.values()]",
    )
    set_src(c32, s)
    print("FIX 10 applied: Bayesian log-loss now computed (cell 32)")

# ──────────────────────────────────────────────────────────────────
# FIX 11 – Cell 34: autoscale xlim instead of hard-coded 0.85
# ──────────────────────────────────────────────────────────────────
c34 = find_cell("axes[0].set_xlim([0.85, 1.0])")
if c34:
    s = src(c34)
    s = s.replace(
        "axes[0].set_xlim([0.85, 1.0])",
        "# Autoscale so bars are never silently clipped for low-accuracy models\n"
        "axes[0].set_xlim([max(0.0, min(all_accs) - 0.02), 1.0])",
    )
    set_src(c34, s)
    print("FIX 11 applied: autoscaled xlim (cell 34)")

# ──────────────────────────────────────────────────────────────────
# Write output
# ──────────────────────────────────────────────────────────────────
with open(DEST, "w") as f:
    json.dump(nb, f, indent=1)

print(f"\nFixed notebook written to: {DEST}")
