"""
Patch the Midterm MNIST notebook so every plt.show() also saves the figure
to the outputs/ directory.  Running this script produces a self-contained
notebook that (a) displays figures inline AND (b) writes PNGs to disk for
the LaTeX report.
"""
import json, re, os

SRC  = "/Users/brianbutler/Portfolio/In Progress Works/PML/Midterm/mnist_classification_project.ipynb"
DEST = SRC  # overwrite in-place

with open(SRC) as f:
    nb = json.load(f)

cells = nb["cells"]

# ── Figure name table ──────────────────────────────────────────────────────────
# We identify each figure-generating code cell by a unique string in its source,
# then insert savefig + mkdir before its plt.show().
FIGURE_MAP = {
    # marker substring → output filename
    "Sample Digits from MNIST":          "outputs/sample_digits.png",
    "Coefficient Maps by Digit":         "outputs/coef_maps.png",
    "Confusion Matrices":                "outputs/confusion_matrices.png",  # added below
    "Shrinkage Pattern Across Priors":   "outputs/bayesian_shrinkage.png",
    # Accuracy comparison chart (cell 34)
    "Model Comparison":                  "outputs/accuracy_logloss_comparison.png",
}

# Confusion matrix cell uses a different title string in the code
CM_MARKER = "Confusion Matrices"

def patch_show(source_str, figname):
    """Insert plt.savefig(...) immediately before each plt.show()."""
    save_line = (
        f"import os; os.makedirs('outputs', exist_ok=True)\n"
        f"plt.savefig('{figname}', dpi=150, bbox_inches='tight')\n"
    )
    # Replace every plt.show() with the save+show pair
    patched = source_str.replace(
        "plt.show()",
        save_line + "plt.show()"
    )
    return patched

patched_count = 0
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])

    for marker, figname in FIGURE_MAP.items():
        if marker in src and f"savefig('{figname}')" not in src:
            src = patch_show(src, figname)
            patched_count += 1
            print(f"  Patched → {figname}")
            break  # one figure per cell

    # Write back as list of lines
    cell["source"] = src.splitlines(keepends=True)

# ── Also add an outputs-setup cell right after the import cell ─────────────────
setup_src = (
    "import os\n"
    "os.makedirs(\n"
    "    os.path.join(os.path.dirname(os.path.abspath('__file__')), 'outputs'),\n"
    "    exist_ok=True\n"
    ")\n"
    "# Resolve output directory relative to notebook location\n"
    "OUT = 'outputs'\n"
    "os.makedirs(OUT, exist_ok=True)\n"
    "print(f'Output directory: {os.path.abspath(OUT)}')"
)

# Insert after cell index 1 (the imports cell)
setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": setup_src.splitlines(keepends=True)
}

# Only insert if not already present
if "Output directory" not in "".join("".join(c.get("source", [])) for c in cells):
    cells.insert(2, setup_cell)
    print("Inserted output-directory setup cell at position 2")

with open(DEST, "w") as f:
    json.dump(nb, f, indent=1)

print(f"\nDone — {patched_count} figure cells patched.")
print(f"Notebook written to: {DEST}")
