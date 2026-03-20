"""
Model definitions for NCAA tournament prediction.

Hierarchy:
  Baseline: SeedLogistic, Elo, KenPomLogistic
  Advanced: GradientBoosting, MixtureOfExperts, TransformerSeq
  Meta: WeightedEnsemble, MarketCalibrated, PortfolioOptimizer
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

class SeedLogistic(BaseEstimator, ClassifierMixin):
    """Logistic regression on seed difference only. The simplest meaningful model."""

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        # X should have column 'seed_diff'
        self.model.fit(X[["seed_diff"]], y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[["seed_diff"]])


class EloRating:
    """Season-long Elo rating system with tunable K-factor and HCA."""

    def __init__(self, k: float = 20.0, hca: float = 100.0, regression: float = 0.6):
        self.k = k
        self.hca = hca
        self.regression = regression  # regress toward mean between seasons
        self.ratings: dict[int, float] = {}

    def expected(self, team_a: int, team_b: int, home_a: bool = False) -> float:
        ra = self.ratings.get(team_a, 1500)
        rb = self.ratings.get(team_b, 1500)
        adj = self.hca if home_a else 0
        return 1.0 / (1.0 + 10 ** ((rb - ra - adj) / 400))

    def update(self, winner: int, loser: int, home_winner: bool = False):
        e = self.expected(winner, loser, home_winner)
        delta = self.k * (1 - e)
        self.ratings[winner] = self.ratings.get(winner, 1500) + delta
        self.ratings[loser] = self.ratings.get(loser, 1500) - delta

    def regress_to_mean(self):
        mean = np.mean(list(self.ratings.values())) if self.ratings else 1500
        for team in self.ratings:
            self.ratings[team] = mean + self.regression * (self.ratings[team] - mean)


# ---------------------------------------------------------------------------
# Advanced models (stubs — to be implemented)
# ---------------------------------------------------------------------------

class GradientBoostingModel:
    """XGBoost/LightGBM on full feature set with walk-forward CV."""

    def __init__(self, framework: str = "xgboost"):
        self.framework = framework
        self.model = None

    def fit(self, X, y):
        if self.framework == "xgboost":
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
            )
        else:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
            )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class MixtureOfExperts:
    """
    Mixture-of-Experts: gating network classifies matchup type,
    specialized expert models handle each cluster.

    Game types (learned via EM):
      - Chalk (1-4 vs 13-16): high-seed dominance
      - Competitive (5-8 vs 9-12): upset-prone
      - Toss-up (close seeds): coin-flip dynamics
      - Cinderella (mid-major vs power): conference mismatch
    """

    def __init__(self, n_experts: int = 4):
        self.n_experts = n_experts
        self.gating = None
        self.experts = []

    def fit(self, X, y):
        # TODO: Implement EM-style training
        # Step 1: Initialize clusters (e.g., by seed gap bins)
        # Step 2: Train expert on each cluster
        # Step 3: Train gating network on cluster assignments
        # Step 4: Iterate
        raise NotImplementedError("MoE training — Week 3-4 milestone")

    def predict_proba(self, X):
        raise NotImplementedError


class TransformerSequenceModel:
    """
    Transformer encoder on team season sequences.
    Input: sequence of game embeddings → [CLS] team embedding → matchup MLP.

    Architecture: 4 layers, 64-dim, 4 heads.
    """

    def __init__(self, d_model: int = 64, n_layers: int = 4, n_heads: int = 4):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.model = None

    def build(self):
        # TODO: Build PyTorch Transformer encoder
        raise NotImplementedError("Transformer model — Week 3-4 milestone")


# ---------------------------------------------------------------------------
# Meta models
# ---------------------------------------------------------------------------

class WeightedEnsemble:
    """Weighted average of base model probabilities, weights tuned on holdout."""

    def __init__(self, models: list, weights: list[float] | None = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict_proba(self, X):
        probs = np.array([m.predict_proba(X)[:, 1] for m in self.models])
        return np.average(probs, axis=0, weights=self.weights)
