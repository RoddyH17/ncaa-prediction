"""
Sequential Bayesian update for live tournament prediction.

Model: each team has a latent strength theta_i ~ Normal(prior_mu_i, prior_sigma^2).
Game outcome: P(team_i beats team_j) = sigmoid(theta_i - theta_j + home_bias).

Initial prior: prior_mu_i = beta * Barttorvik AdjEM_i (or any pre-tournament rating).
Per-game likelihood: Bernoulli(sigmoid(theta_i - theta_j)).

Update rule (per game): Laplace approximation. Closed-form linear update on
the natural parameters (information form), exact for Gaussian + linearized BCE.

Round-by-round procedure:
  1. Initialize prior over all teams from pre-tournament features
  2. After each round: update posterior using observed game outcomes
  3. Predict next round using updated posterior

Key novelty: live in-tournament Bayesian update beats fixed pre-tournament
predictions because it incorporates new evidence as the tournament progresses.
"""

import numpy as np
import pandas as pd


class SequentialBayesianTournament:
    """Bayesian Bradley-Terry style team strength tracker.

    Each team has Gaussian belief over its strength theta_i.
    Game outcomes update posterior via Laplace approximation.
    """

    def __init__(self, prior_means: dict[int, float], prior_var: float = 0.5,
                 obs_scale: float = 1.0):
        """
        prior_means: dict mapping team_id -> prior mean of theta
        prior_var: scalar variance for all teams' priors (could be team-specific)
        obs_scale: scaling constant for likelihood (1.0 = standard logistic)
        """
        self.team_ids = list(prior_means.keys())
        self.tid_to_idx = {tid: i for i, tid in enumerate(self.team_ids)}

        n = len(self.team_ids)
        # Mean vector and covariance matrix
        self.mu = np.array([prior_means[tid] for tid in self.team_ids])
        self.cov = np.eye(n) * prior_var
        self.obs_scale = obs_scale

    def predict(self, team_a: int, team_b: int) -> float:
        """P(team_a beats team_b) given current posterior."""
        if team_a not in self.tid_to_idx or team_b not in self.tid_to_idx:
            return 0.5
        ia = self.tid_to_idx[team_a]
        ib = self.tid_to_idx[team_b]
        # Marginal logit difference: mean and variance
        mean_diff = self.mu[ia] - self.mu[ib]
        var_diff = self.cov[ia, ia] + self.cov[ib, ib] - 2 * self.cov[ia, ib]
        # Probit/logistic approximation: integrate Bernoulli(sigmoid) over Gaussian
        # Use approximate: sigmoid(mean_diff / sqrt(1 + pi*var_diff/8))
        scaled = mean_diff / np.sqrt(1 + np.pi * var_diff / 8)
        return 1.0 / (1.0 + np.exp(-self.obs_scale * scaled))

    def update_with_game(self, winner: int, loser: int, lr: float = 1.0):
        """Update posterior with one observed game outcome (winner beats loser).

        Uses Laplace approximation:
          - Linearize log-likelihood around current mu
          - Treat as Gaussian observation in (theta_winner - theta_loser) space
        """
        if winner not in self.tid_to_idx or loser not in self.tid_to_idx:
            return
        iw = self.tid_to_idx[winner]
        il = self.tid_to_idx[loser]

        n = len(self.mu)
        # Selection vector h: theta_winner - theta_loser
        h = np.zeros(n)
        h[iw] = 1.0
        h[il] = -1.0

        # Current predicted prob and gradient
        diff = self.mu[iw] - self.mu[il]
        p = 1.0 / (1.0 + np.exp(-self.obs_scale * diff))

        # Gradient of log-likelihood (winner=1):
        # d log p / d theta_w = (1 - p), d / d theta_l = -(1-p)
        residual = (1.0 - p) * self.obs_scale
        grad = h * residual

        # Hessian (negative): -p*(1-p) * h h^T
        # In information form: precision update += p*(1-p) * h h^T
        info = p * (1.0 - p) * (self.obs_scale ** 2)

        # Update via Kalman-style: posterior covariance shrinks along h direction
        Sh = self.cov @ h
        denom = 1.0 / info + h @ Sh
        # Mean update: mu += grad direction scaled
        kalman_gain = Sh / denom
        # Innovation = (1 - p) is the effective residual after seeing winner=1
        innovation = 1.0 - p
        self.mu = self.mu + kalman_gain * innovation * lr

        # Covariance update: rank-1 reduction
        self.cov = self.cov - np.outer(kalman_gain, Sh)

        # Symmetrize to avoid numerical drift
        self.cov = 0.5 * (self.cov + self.cov.T)

    def get_team_strength(self, team_id: int) -> tuple[float, float]:
        """Return (mean, variance) of team strength."""
        if team_id not in self.tid_to_idx:
            return 0.0, 1.0
        i = self.tid_to_idx[team_id]
        return float(self.mu[i]), float(self.cov[i, i])


def fit_prior_from_features(features_df: pd.DataFrame, outcomes: np.ndarray,
                             team_a_col: str = "TeamA", team_b_col: str = "TeamB"):
    """Fit a logistic on training data, then convert to per-team prior strengths.

    Approach: train logistic, then derive each team's "fitted strength" as the
    model's logit prediction averaged over historical opponents.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    # Use Barttorvik net rating as the team's "intrinsic" strength signal
    # If we have bart_net_diff = bart_net_a - bart_net_b, infer per-team values
    # by averaging predictions over many opponents.

    # Simpler: just use AdjOE - AdjDE per team if available.
    # For now, use a logistic on bart_net_diff.

    if "bart_net_diff" in features_df.columns:
        X = features_df[["bart_net_diff"]].fillna(0).values
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X, outcomes)
        # The coefficient * bart_net_diff = log-odds difference
        # So effective theta_i = lr.coef * bart_net_i (per-team)
        return float(lr.coef_[0, 0])  # scaling factor for theta
    return 1.0
