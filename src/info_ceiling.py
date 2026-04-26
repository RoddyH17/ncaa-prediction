"""
Information-theoretic ceiling estimation for binary outcome prediction.

Provides three estimators of the irreducible Bayes risk under feature set X:

  (1) MINE: Mutual Information Neural Estimator (Belghazi et al., 2018).
      Estimates I(X; Y) via neural lower bound on KL divergence; converted to
      a Brier lower bound via Pinsker-type inequality.

  (2) KSG-style discrete approximation: bins X into K cells via KMeans,
      computes discrete I(X; Y) and entropy H(Y|X). For binary Y, the optimal
      Brier risk is BS* = E[p(X)(1-p(X))] which is bounded below using H(Y|X).

  (3) Direct flexible-model estimation: train high-capacity classifiers
      (random forest, k-NN, GBM) and report their cross-validated Brier as a
      consistent estimator of BS* (under universal consistency).

The ceiling claim is supported when (1), (2), (3) converge, and our linear
LOTO Brier ≈ this convergence point.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def binary_entropy(p: float | np.ndarray, eps: float = 1e-12) -> float | np.ndarray:
    """Shannon entropy of Bernoulli(p) in nats."""
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def brier_lower_bound_from_cond_entropy(H_Y_given_X: float) -> float:
    """Lower bound on Bayes Brier risk given E[H(Y|X)].

    Using inequality h(p) ≥ 2 ln 2 · p(1-p) for p ∈ [0,1] (binary entropy
    bound), so E[p(1-p)] ≤ E[h(p)] / (2 ln 2) = H(Y|X) / (2 ln 2).
    Equivalent for nats: E[p(1-p)] ≤ H(Y|X) / 2.
    Bayes Brier = E[p(1-p)], so:
        BS_lower = ?

    Reverse direction: h(p) ≤ ln 2 always; h(p) ≥ 2 p(1-p) (in nats).
    So p(1-p) ≤ h(p)/2, meaning BS_Bayes ≤ H(Y|X)/2 (upper bound).

    For LOWER bound on Bayes Brier, use Fano-type:
        h(p) ≤ -2 (p - 0.5)^2 + ln 2 (Taylor)
        => (p - 0.5)^2 ≤ (ln 2 - h(p)) / 2
        => p(1-p) = 0.25 - (p-0.5)^2 ≥ 0.25 - (ln 2 - h(p))/2
                  = 0.25 + h(p)/2 - ln(2)/2
        BS_Bayes = E[p(1-p)] ≥ 0.25 + H(Y|X)/2 - ln(2)/2
    This is non-trivial when H(Y|X) > ln(2) - 0.5 (i.e. close to ln 2).

    We return the upper bound H(Y|X) / 2 as the loose ceiling estimate,
    plus the Fano lower bound for context.
    """
    upper_BS = H_Y_given_X / 2.0
    fano_lower_BS = 0.25 + H_Y_given_X / 2.0 - math.log(2) / 2.0
    return upper_BS, max(0.0, fano_lower_BS)


# ---------------------------------------------------------------------------
# MINE estimator
# ---------------------------------------------------------------------------

class MINENetwork(nn.Module):
    def __init__(self, x_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, y):
        # y is (N, 1) binary -> use as continuous input
        z = torch.cat([x, y], dim=1)
        return self.net(z).squeeze(-1)


def estimate_mi_mine(
    X: np.ndarray,
    Y: np.ndarray,
    n_epochs: int = 1500,
    batch_size: int = 256,
    lr: float = 5e-4,
    hidden: int = 64,
    ema_decay: float = 0.99,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Estimate I(X; Y) via MINE-f (with EMA bias correction).

    Returns dict with keys:
      mi_nats, mi_bits, H_Y_given_X, history
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Standardize X
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    X = np.nan_to_num(X, nan=0.0)

    N, d = X.shape
    Y = Y.astype(np.float32).reshape(-1, 1)

    Xt = torch.from_numpy(X.astype(np.float32))
    Yt = torch.from_numpy(Y)

    net = MINENetwork(d, hidden=hidden)
    opt = optim.Adam(net.parameters(), lr=lr)
    ema_term = None
    history = []

    for epoch in range(n_epochs):
        idx = rng.permutation(N)[:batch_size]
        idx_marg = rng.permutation(N)[:batch_size]
        xb = Xt[idx]
        yb = Yt[idx]
        y_marg = Yt[idx_marg]

        T_joint = net(xb, yb)
        T_marg = net(xb, y_marg)

        # EMA bias correction (Belghazi et al.)
        with torch.no_grad():
            mean_exp = torch.exp(T_marg).mean()
            if ema_term is None:
                ema_term = mean_exp
            else:
                ema_term = ema_decay * ema_term + (1 - ema_decay) * mean_exp

        # Loss: -(E[T_joint] - log E[exp(T_marg)]); use EMA grad
        loss = -(T_joint.mean() - torch.log(torch.exp(T_marg).mean() + 1e-8) *
                 (mean_exp.detach() / ema_term.detach()))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if (epoch + 1) % 100 == 0:
            mi_estimate = T_joint.mean().item() - math.log(mean_exp.item() + 1e-12)
            history.append({"epoch": epoch + 1, "mi_nats": mi_estimate})
            if verbose:
                print(f"  MINE epoch {epoch+1}: I={mi_estimate:.4f} nats")

    # Final estimate via full-sample evaluation
    net.eval()
    with torch.no_grad():
        # Batch through to avoid OOM
        all_T_joint = []
        all_T_marg = []
        for i in range(0, N, 512):
            xb = Xt[i:i+512]
            yb = Yt[i:i+512]
            y_marg = Yt[rng.permutation(N)[:xb.shape[0]]]
            all_T_joint.append(net(xb, yb))
            all_T_marg.append(net(xb, y_marg))
        T_joint_full = torch.cat(all_T_joint)
        T_marg_full = torch.cat(all_T_marg)
        mi_nats = T_joint_full.mean().item() - math.log(torch.exp(T_marg_full).mean().item() + 1e-12)

    H_Y = float(binary_entropy(Y.mean()))
    H_Y_given_X = max(0.0, H_Y - mi_nats)

    return {
        "mi_nats": mi_nats,
        "mi_bits": mi_nats / math.log(2),
        "H_Y": H_Y,
        "H_Y_given_X": H_Y_given_X,
        "history": history,
    }


# ---------------------------------------------------------------------------
# KMeans-based discrete MI estimator
# ---------------------------------------------------------------------------

def estimate_mi_kmeans(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters: int = 50,
    seed: int = 42,
) -> dict:
    """Estimate I(X; Y) by quantizing X into K cells via KMeans.

    For each cluster k: p_k = N_k / N, p(Y=1|k) = mean(Y[cluster=k]).
    H(Y|X) ≈ Σ_k p_k h(p(Y=1|k))
    BS_Bayes_estimate ≈ Σ_k p_k p(Y=1|k)(1-p(Y=1|k))
    """
    X = np.nan_to_num(
        (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8),
        nan=0.0,
    )
    N = len(X)
    n_clusters = min(n_clusters, N // 5)  # avoid over-fragmentation

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(X)

    H_Y = float(binary_entropy(Y.mean()))
    H_Y_given_X = 0.0
    BS_Bayes_est = 0.0
    for k in range(n_clusters):
        mask = labels == k
        N_k = mask.sum()
        if N_k == 0:
            continue
        p_k = N_k / N
        py = Y[mask].mean() if N_k > 0 else 0.5
        H_Y_given_X += p_k * float(binary_entropy(py))
        BS_Bayes_est += p_k * py * (1 - py)

    mi_nats = H_Y - H_Y_given_X
    return {
        "mi_nats": mi_nats,
        "mi_bits": mi_nats / math.log(2),
        "H_Y": H_Y,
        "H_Y_given_X": H_Y_given_X,
        "BS_Bayes_est": BS_Bayes_est,
        "n_clusters": n_clusters,
    }


# ---------------------------------------------------------------------------
# Flexible-model estimator of Bayes Brier
# ---------------------------------------------------------------------------

def estimate_bs_flexible(
    X: np.ndarray,
    Y: np.ndarray,
    seasons: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Estimate Bayes Brier via cross-validated flexible models.

    Universal consistency: as N → ∞, k-NN, RF, and GBM all converge to BS*.
    With finite N, their CV Brier is a (slightly biased) estimate.

    If `seasons` is provided, uses leave-one-season-out CV to mirror
    distribution-shift behaviour seen in tournament data.
    """
    X = np.nan_to_num(X, nan=0.0)
    if seasons is None:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(X))
    else:
        splits = []
        for s in np.unique(seasons):
            train_mask = seasons != s
            test_mask = seasons == s
            splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))

    models = {
        "RandomForest_500": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3,
            n_jobs=-1, random_state=seed
        ),
        "GBM_300": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed
        ),
        "kNN_30": KNeighborsClassifier(n_neighbors=30, weights="distance"),
    }
    results = {}
    for name, model in models.items():
        fold_brier = []
        for train_idx, test_idx in splits:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = Y[train_idx], Y[test_idx]
            mu, sigma = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-8
            X_tr_z = (X_tr - mu) / sigma
            X_te_z = (X_te - mu) / sigma
            model.fit(X_tr_z, y_tr)
            p = model.predict_proba(X_te_z)[:, 1]
            p = np.clip(p, 0.01, 0.99)
            fold_brier.append(brier_score_loss(y_te, p))
        results[name] = {"brier_mean": float(np.mean(fold_brier)),
                         "brier_std": float(np.std(fold_brier)),
                         "n_folds": len(fold_brier)}

    bs_floor = min(r["brier_mean"] for r in results.values())
    return {"models": results, "bs_floor_estimate": bs_floor}
