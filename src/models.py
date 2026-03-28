"""
Model definitions for NCAA tournament prediction.

Hierarchy:
  Baseline: SeedLogistic, KenPomLogistic, Elo, GradientBoosting
  Advanced: MixtureOfExperts, TransformerSeq
  Meta: WeightedEnsemble, MarketCalibrated
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

_ID_COLS = ["Season", "TeamA", "TeamB"]


def _drop_id_cols(X):
    """Drop identifier columns, return feature-only DataFrame or array."""
    if isinstance(X, pd.DataFrame):
        return X.drop(columns=[c for c in _ID_COLS if c in X.columns])
    return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from a DataFrame."""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.columns].values
        return X


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

class SeedLogistic(BaseEstimator, ClassifierMixin):
    """Logistic regression on seed difference only."""

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X[["seed_diff"]], y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[["seed_diff"]])


class KenPomLogistic(BaseEstimator, ClassifierMixin):
    """Logistic regression on KenPom (POM) rank difference."""

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        col = X[["rank_diff_POM"]].fillna(0)
        self.model.fit(col, y)
        return self

    def predict_proba(self, X):
        col = X[["rank_diff_POM"]].fillna(0)
        return self.model.predict_proba(col)


class EfficiencyLogistic(BaseEstimator, ClassifierMixin):
    """Logistic regression on our computed net efficiency difference."""

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        col = X[["net_eff_diff"]].fillna(0)
        self.model.fit(col, y)
        return self

    def predict_proba(self, X):
        col = X[["net_eff_diff"]].fillna(0)
        return self.model.predict_proba(col)


class MultiFeatureLogistic(BaseEstimator, ClassifierMixin):
    """Logistic regression on selected continuous features with scaling."""

    _FEATURE_COLS = [
        "seed_diff", "net_eff_diff", "off_eff_diff", "def_eff_diff", "tempo_diff",
        "efg_pct_diff", "to_pct_diff", "or_pct_diff", "ft_rate_diff",
        "opp_efg_pct_diff", "opp_to_pct_diff", "opp_or_pct_diff", "opp_ft_rate_diff",
        "rank_diff_POM",
        "momentum_margin_diff", "momentum_winpct_diff",
    ]

    def __init__(self, C=1.0):
        self.C = C
        self.pipe = None

    def fit(self, X, y):
        cols = [c for c in self._FEATURE_COLS if c in X.columns]
        self.cols_ = cols
        self.pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs")),
        ])
        self.pipe.fit(X[cols], y)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X[self.cols_])


class EloRating:
    """Season-long Elo rating system with tunable K-factor and HCA."""

    def __init__(self, k: float = 20.0, hca: float = 100.0, regression: float = 0.6):
        self.k = k
        self.hca = hca
        self.regression = regression
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


class EloSklearnWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper around EloRating for use with LOTO backtest.

    On fit(): replays regular-season games to build ratings.
    On predict_proba(): uses final Elo ratings to predict tournament matchups.

    Note: fit() processes regular-season data for ALL seasons up to the max
    season in the training set + 1 (to include the holdout season's regular
    season). This is not leakage because we only hold out tournament outcomes.
    """

    def __init__(self, data, k=20.0, hca=100.0, regression=0.6):
        self.data = data
        self.k = k
        self.hca = hca
        self.regression = regression
        self.elo = None

    def fit(self, X, y):
        self.elo = EloRating(k=self.k, hca=self.hca, regression=self.regression)
        reg = self.data["regular_compact"]

        # Process all seasons up to max_train+1 to cover holdout regular season
        train_seasons = sorted(X["Season"].unique())
        max_season = int(max(train_seasons)) + 1
        all_seasons = sorted(reg["Season"].unique())
        seasons_to_process = [s for s in all_seasons if s <= max_season]

        for i, season in enumerate(seasons_to_process):
            if i > 0:
                self.elo.regress_to_mean()
            games = reg[reg["Season"] == season].sort_values("DayNum")
            for _, g in games.iterrows():
                home = g.get("WLoc", "N") == "H"
                self.elo.update(g["WTeamID"], g["LTeamID"], home)
        return self

    def predict_proba(self, X):
        probs = []
        for _, row in X.iterrows():
            p = self.elo.expected(int(row["TeamA"]), int(row["TeamB"]))
            probs.append([1 - p, p])
        return np.array(probs)


# ---------------------------------------------------------------------------
# Advanced models
# ---------------------------------------------------------------------------

class GradientBoostingModel(BaseEstimator, ClassifierMixin):
    """XGBoost/LightGBM on full feature set."""

    def __init__(self, framework: str = "xgboost"):
        self.framework = framework
        self.model = None
        self.feature_cols_ = None

    def fit(self, X, y):
        X_feat = _drop_id_cols(X)
        if isinstance(X_feat, pd.DataFrame):
            self.feature_cols_ = list(X_feat.columns)
            X_feat = X_feat.values
        else:
            self.feature_cols_ = None

        if self.framework == "xgboost":
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                verbosity=0,
            )
            self.model.fit(X_feat, y)
        else:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=-1,
            )
            self.model.fit(X_feat, y)
        return self

    def predict_proba(self, X):
        X_feat = _drop_id_cols(X)
        if isinstance(X_feat, pd.DataFrame):
            X_feat = X_feat.values
        return self.model.predict_proba(X_feat)


class MixtureOfExperts(BaseEstimator, ClassifierMixin):
    """
    Mixture-of-Experts with EM training.

    Gating network classifies matchups into K game types,
    specialized logistic regression experts handle each.
    """

    def __init__(self, n_experts=4, n_em_iters=10, expert_C=0.1, gating_C=1.0):
        self.n_experts = n_experts
        self.n_em_iters = n_em_iters
        self.expert_C = expert_C
        self.gating_C = gating_C
        self.experts = []
        self.gating = None
        self.imputer_ = None
        self.scaler_ = None

    def _init_assignments(self, X):
        """Initialize expert assignments by seed gap bins."""
        n = len(X)
        assignments = np.zeros((n, self.n_experts))
        seed_diff = np.abs(X["seed_diff"].values) if "seed_diff" in X.columns else np.zeros(n)

        for i in range(n):
            sd = seed_diff[i]
            if sd >= 8:
                assignments[i, 0] = 1.0   # Chalk
            elif sd >= 4:
                assignments[i, 1] = 1.0   # Competitive
            elif sd >= 1:
                assignments[i, 2] = 1.0   # Toss-up
            else:
                assignments[i, 3] = 1.0   # Cinderella / same-seed
        return assignments

    def fit(self, X, y):
        X_feat = _drop_id_cols(X)
        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        X_imp = self.scaler_.fit_transform(self.imputer_.fit_transform(X_feat))

        assignments = self._init_assignments(X)

        self.experts = [
            LogisticRegression(C=self.expert_C, max_iter=1000, solver="lbfgs")
            for _ in range(self.n_experts)
        ]
        self.gating = LogisticRegression(
            C=self.gating_C, max_iter=1000, solver="lbfgs"
        )

        for iteration in range(self.n_em_iters):
            # M-step: train experts with sample weights
            for k in range(self.n_experts):
                w = assignments[:, k]
                if w.sum() < 2:
                    continue
                self.experts[k].fit(X_imp, y, sample_weight=w)

            # M-step: train gating on hard assignments
            hard_labels = assignments.argmax(axis=1)
            self.gating.fit(X_imp, hard_labels)

            # E-step: recompute responsibilities
            gate_probs = self.gating.predict_proba(X_imp)  # (n, K)
            expert_liks = np.zeros((len(X_imp), self.n_experts))
            for k in range(self.n_experts):
                try:
                    p = self.experts[k].predict_proba(X_imp)[:, 1]
                except:
                    p = np.full(len(X_imp), 0.5)
                expert_liks[:, k] = y * p + (1 - y) * (1 - p) + 1e-10

            # Responsibility = gate_prob * expert_likelihood, normalized
            resp = gate_probs * expert_liks
            resp = resp / resp.sum(axis=1, keepdims=True)
            assignments = resp

        return self

    def predict_proba(self, X):
        X_feat = _drop_id_cols(X)
        X_imp = self.scaler_.transform(self.imputer_.transform(X_feat))
        gate_probs = self.gating.predict_proba(X_imp)

        weighted_probs = np.zeros(len(X_imp))
        for k in range(self.n_experts):
            try:
                p = self.experts[k].predict_proba(X_imp)[:, 1]
            except:
                p = np.full(len(X_imp), 0.5)
            weighted_probs += gate_probs[:, k] * p

        return np.column_stack([1 - weighted_probs, weighted_probs])


# ---------------------------------------------------------------------------
# Meta models
# ---------------------------------------------------------------------------

class WeightedEnsemble:
    """Weighted average of base model probabilities."""

    def __init__(self, models: list, weights: list[float] | None = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict_proba(self, X):
        probs = np.array([m.predict_proba(X)[:, 1] for m in self.models])
        weighted = np.average(probs, axis=0, weights=self.weights)
        return np.column_stack([1 - weighted, weighted])

    def optimize_weights(self, X_val, y_val):
        """Optimize weights on validation set to minimize Brier score."""
        from scipy.optimize import minimize
        from sklearn.metrics import brier_score_loss

        cached_probs = [m.predict_proba(X_val)[:, 1] for m in self.models]

        def objective(w):
            w = np.abs(w)
            w = w / w.sum()
            blended = np.average(cached_probs, axis=0, weights=w)
            return brier_score_loss(y_val, blended)

        n = len(self.models)
        result = minimize(
            objective, self.weights, method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        )
        self.weights = list(result.x / result.x.sum())


class MarketCalibratedEnsemble:
    """Bayesian update of model prediction with market-implied probability.

    p_final = (p_model^alpha * p_market^(1-alpha)) /
              (p_model^alpha * p_market^(1-alpha) + (1-p_model)^alpha * (1-p_market)^(1-alpha))
    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def calibrate(self, p_model: np.ndarray, p_market: np.ndarray) -> np.ndarray:
        a = self.alpha
        num = (p_model ** a) * (p_market ** (1 - a))
        den = num + ((1 - p_model) ** a) * ((1 - p_market) ** (1 - a))
        return num / den
