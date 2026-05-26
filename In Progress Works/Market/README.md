# Heterogeneous Price Impact via Causal Machine Learning

Estimating how the short-term effect of large trades on price changes depends
on liquidity and volatility, using causal forests and meta-learner techniques.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

Results (figures + CSV) are saved to `results/`.

## Project Structure

```
Market/
├── config/params.yaml       # All tunable parameters
├── src/
│   ├── dgp.py               # Synthetic data generation (known CATE)
│   ├── features.py           # Feature engineering
│   ├── causal_graph.py       # DAG specification & assumptions
│   ├── estimators.py         # HTE estimator wrappers
│   ├── evaluation.py         # Metrics (RMSE, coverage, GATES)
│   └── visualization.py      # Publication-quality plots
├── main.py                   # End-to-end pipeline
└── results/                  # Output directory
```

## Estimators

| Method | Type | Treatment |
|---|---|---|
| CausalForestDML | Double ML + forest | Continuous |
| S-Learner | Meta-learner | Binary |
| T-Learner | Meta-learner | Binary |
| X-Learner | Meta-learner | Binary |
| DR-Learner | Meta-learner (doubly robust) | Binary |

## Key References

- Athey & Imbens (2019). *Generalized Random Forests.* Annals of Statistics.
- Künzel et al. (2019). *Metalearners for estimating HTE.* PNAS.
- Almgren & Chriss (2001). *Optimal execution of portfolio transactions.* J. Risk.
