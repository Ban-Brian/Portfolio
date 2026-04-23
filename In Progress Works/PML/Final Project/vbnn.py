"""
04_vbnn.py
Variational Bayesian neural network. Two hidden layers of width 128,
mean-field Gaussian variational posterior over the weights, standard-normal
priors. Two output heads: a heteroscedastic Gaussian over star rating
(predicting mean and log-variance) and a Bernoulli over is_open. Trained
by maximising the ELBO with Adam; KL annealed over the first 20 epochs.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule, PyroSample

SEED = 42
PROC = pathlib.Path("data/processed")
OUT = pathlib.Path("results/vbnn")
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 128
EPOCHS = 150
BATCH = 256
LR = 1e-3
KL_ANNEAL_EPOCHS = 20
POSTERIOR_SAMPLES = 50

torch.manual_seed(SEED)
np.random.seed(SEED)
pyro.set_rng_seed(SEED)


# Load features and targets, cast to float32 for PyTorch
def load_data() -> dict:
    blob = np.load(PROC / "features.npz", allow_pickle=True)
    X = np.concatenate([blob["X_struct"], blob["X_embed"]], axis=1).astype(np.float32)
    return {
        "X": X,
        "stars": blob["stars"].astype(np.float32),
        "is_open": blob["is_open"].astype(np.float32),
        "train": blob["train_idx"],
        "val": blob["val_idx"],
        "test": blob["test_idx"],
    }


class BayesianLinear(PyroModule):
    """Dense layer with a standard-normal prior on every weight."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = PyroSample(
            dist.Normal(torch.zeros(out_dim, in_dim), torch.ones(out_dim, in_dim)).to_event(2)
        )
        self.bias = PyroSample(
            dist.Normal(torch.zeros(out_dim), torch.ones(out_dim)).to_event(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias


class VBNN(PyroModule):
    """Two-hidden-layer BNN with heteroscedastic Gaussian + Bernoulli heads."""

    def __init__(self, in_dim: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.fc1 = BayesianLinear(in_dim, hidden)
        self.fc2 = BayesianLinear(hidden, hidden)
        self.mean_head = BayesianLinear(hidden, 1)
        self.logvar_head = BayesianLinear(hidden, 1)
        self.logit_head = BayesianLinear(hidden, 1)
        self.act = nn.ReLU()

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc2(self.act(self.fc1(x))))

    def forward(self, x: torch.Tensor,
                y_stars: torch.Tensor | None = None,
                y_open: torch.Tensor | None = None,
                kl_weight: float = 1.0) -> torch.Tensor:
        h = self.trunk(x)
        mu = self.mean_head(h).squeeze(-1)
        log_var = self.logvar_head(h).squeeze(-1)
        sigma = torch.exp(0.5 * log_var).clamp(min=1e-3, max=3.0)
        logit = self.logit_head(h).squeeze(-1)

        with pyro.plate("data", x.shape[0]):
            if y_stars is not None:
                pyro.sample("stars", dist.Normal(mu, sigma), obs=y_stars)
            if y_open is not None:
                pyro.sample("is_open", dist.Bernoulli(logits=logit), obs=y_open)
        return mu, sigma, logit


# Wrap numpy arrays into a DataLoader for mini-batch training
def make_loader(X, ys, yo, batch: int = BATCH, shuffle: bool = True):
    X_t = torch.from_numpy(X)
    ys_t = torch.from_numpy(ys)
    yo_t = torch.from_numpy(yo)
    ds = torch.utils.data.TensorDataset(X_t, ys_t, yo_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle,
                                       num_workers=0, drop_last=False)


# Train the model via SVI with KL annealing over the first 20 epochs
def train(model: VBNN, guide, Xtr, ys_tr, yo_tr) -> list[float]:
    optim = pyro.optim.Adam({"lr": LR})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())
    loader = make_loader(Xtr, ys_tr, yo_tr)
    history = []

    for epoch in range(EPOCHS):
        kl_weight = min(1.0, (epoch + 1) / KL_ANNEAL_EPOCHS)
        running = 0.0
        n = 0
        for xb, ysb, yob in loader:
            loss = svi.step(xb, ysb, yob, kl_weight=kl_weight)
            running += loss
            n += xb.shape[0]
        history.append(running / n)
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:3d}  neg_elbo/N = {history[-1]:.4f}  kl_w = {kl_weight:.2f}")
    return history


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

# Draw posterior samples and decompose predictive uncertainty
def posterior_predict(model: VBNN, guide, X: np.ndarray, n_samples: int = POSTERIOR_SAMPLES):
    X_t = torch.from_numpy(X)
    mus, sigmas, logits = [], [], []
    for _ in range(n_samples):
        guide_trace = pyro.poutine.trace(guide).get_trace(X_t)
        replayed = pyro.poutine.replay(model, trace=guide_trace)
        mu, sigma, logit = replayed(X_t)
        mus.append(mu.detach().numpy())
        sigmas.append(sigma.detach().numpy())
        logits.append(logit.detach().numpy())

    mus = np.stack(mus)
    sigmas = np.stack(sigmas)
    probs = 1.0 / (1.0 + np.exp(-np.stack(logits)))

    # Decompose into aleatoric (average data noise) and epistemic (weight uncertainty)
    mean_pred = mus.mean(axis=0)
    aleatoric = (sigmas ** 2).mean(axis=0)
    epistemic = mus.var(axis=0)
    total_std = np.sqrt(aleatoric + epistemic)
    return {
        "mean": mean_pred,
        "aleatoric_var": aleatoric,
        "epistemic_var": epistemic,
        "total_std": total_std,
        "prob_open": probs.mean(axis=0),
        "prob_open_std": probs.std(axis=0),
    }


# Expected calibration error across probability bins
def expected_calibration_error(y_true, p_hat, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_hat, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += (m.sum() / n) * abs(y_true[m].mean() - p_hat[m].mean())
    return float(ece)


def main() -> None:
    d = load_data()
    Xtr = d["X"][d["train"]]
    Xte = d["X"][d["test"]]
    ys_tr = d["stars"][d["train"]]
    yo_tr = d["is_open"][d["train"]]
    ys_te = d["stars"][d["test"]]
    yo_te = d["is_open"][d["test"]]

    pyro.clear_param_store()
    model = VBNN(in_dim=Xtr.shape[1])
    guide = AutoNormal(model, init_scale=0.05)

    history = train(model, guide, Xtr, ys_tr, yo_tr)
    pd.DataFrame({"epoch": np.arange(1, len(history) + 1),
                  "neg_elbo_per_obs": history}).to_csv(OUT / "elbo_history.csv", index=False)

    preds = posterior_predict(model, guide, Xte)
    rmse = float(np.sqrt(np.mean((ys_te - preds["mean"]) ** 2)))
    pred_open = (preds["prob_open"] >= 0.5).astype(int)
    acc = float((pred_open == yo_te).mean())

    # Manual binary cross-entropy with clipping to avoid log(0)
    p_clipped = preds["prob_open"].clip(1e-7, 1 - 1e-7)
    ll = float(-np.mean(
        yo_te * np.log(p_clipped)
        + (1 - yo_te) * np.log(1 - p_clipped)
    ))
    ece = expected_calibration_error(yo_te, preds["prob_open"])

    summary = pd.DataFrame([{
        "test_rmse": rmse,
        "test_acc": acc,
        "test_logloss": ll,
        "ece": ece,
        "mean_aleatoric_std": float(np.sqrt(preds["aleatoric_var"]).mean()),
        "mean_epistemic_std": float(np.sqrt(preds["epistemic_var"]).mean()),
    }])
    summary.to_csv(OUT / "test_performance.csv", index=False)
    print(summary.to_string(index=False))

    np.savez_compressed(
        OUT / "predictions.npz",
        mean=preds["mean"],
        aleatoric_var=preds["aleatoric_var"],
        epistemic_var=preds["epistemic_var"],
        prob_open=preds["prob_open"],
        prob_open_std=preds["prob_open_std"],
    )


if __name__ == "__main__":
    main()
