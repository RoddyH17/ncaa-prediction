# Project Proposal: NCAA March Madness Prediction System

## 1. Idea

### 1.1 Motivation

NCAA March Madness is one of the highest-variance sporting events in the world — 68 teams, single-elimination, 63 games. The Kaggle "March Machine Learning Mania" competition (running since 2014, $50K prize pool) asks participants to predict P(Team A beats Team B) for every possible matchup, scored by Brier Score.

Key insight from 12 years of competition history: **simple models with strong features consistently outperform complex ML**. The 2017 champion won via meta-prediction (modeling competitors' submissions), not better game modeling. Most top solutions use logistic regression or Bradley-Terry with KenPom/Massey features.

This project aims to push beyond that ceiling by:
1. **Mixture-of-Experts (MoE)**: Different game "types" (e.g., chalk matchups vs upset-prone) require different models — a single global model underfits heterogeneity
2. **Temporal sequence modeling**: A team's trajectory through the season (momentum, injury impact, late-season form) carries predictive signal that static ratings miss
3. **Market-informed calibration**: Betting market odds (Kalshi, Vegas) are the single strongest predictor — our model should learn to disagree with the market only when it has genuine edge
4. **Portfolio optimization**: For Kaggle, optimize submission portfolios to maximize P(finishing top-K), not just minimize Brier score — the Landgraf 2017 meta-strategy

### 1.2 Connection to Prior Work

This extends `~/sports_project/` (NBA CW-RAPM + volatility regime detection):
- **CW-RAPM** → analogous context-weighting for college basketball (schedule strength, home/away, rest days)
- **Volatility regime switches** → tournament volatility is structurally different from regular season — our HMM/mixture approach directly applies
- **6-factor championship model** → adapted factors (bench depth, rotation entropy, stagger index) for NCAA

### 1.3 Deliverables

| Deliverable | Target Date |
|-------------|-------------|
| Data pipeline + EDA notebook | Week 1-2 (by Apr 3) |
| Baseline models (Elo + Logistic + XGBoost) | Week 2-3 (by Apr 10) |
| MoE + Transformer sequence model | Week 3-4 (by Apr 17) |
| Backtest on 2014-2026 tournaments | Week 4-5 (by Apr 24) |
| Kalshi API integration + paper draft | Week 5-6 (by May 1) |
| Kaggle 2027 ready pipeline | May 2027 |

---

## 2. Datasets

### 2.1 Primary: Kaggle March ML Mania (2014-2026)

12 years of competition data, ~35 CSVs per year. The 2026 edition data is the most complete and subsumes all prior years.

**Download:** `kaggle competitions download -c march-machine-learning-mania-2026`

**Core files:**
| File | Description |
|------|-------------|
| `MRegularSeasonDetailedResults.csv` | Box scores: FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF |
| `MNCAATourneyDetailedResults.csv` | Same stats for tournament games |
| `MNCAATourneySeeds.csv` | Tournament seedings (W01-W16, X01-X16, Y01-Y16, Z01-Z16) |
| `MMasseyOrdinals.csv` | Daily rankings from 100+ systems (KenPom, Sagarin, BPI, RPI, etc.) |
| `MNCAATourneySlots.csv` | Bracket structure mapping |
| `MTeamConferences.csv` | Conference affiliations per season |
| `MGameCities.csv` | Game locations (for travel distance features) |
| `MTeamCoaches.csv` | Coaching history |

**Same structure for Women's (W prefix).**

### 2.2 Historical Kaggle Competitions (for backtest validation)

#### Era 1: Community (2014-2017)
| Year | URL |
|------|-----|
| 2014 | https://www.kaggle.com/competitions/march-machine-learning-mania-2014 |
| 2015 | https://www.kaggle.com/c/march-machine-learning-mania-2015 |
| 2016 | https://www.kaggle.com/c/march-machine-learning-mania-2016 |
| 2017 | https://www.kaggle.com/competitions/march-machine-learning-mania-2017 |

#### Era 2: Google Cloud & NCAA (2018-2020)
| Year | Men's | Women's |
|------|-------|---------|
| 2018 | https://www.kaggle.com/c/mens-machine-learning-competition-2018 | https://www.kaggle.com/c/womens-machine-learning-competition-2018 |
| 2019 | https://www.kaggle.com/c/mens-machine-learning-competition-2019 | https://www.kaggle.com/c/womens-machine-learning-competition-2019 |
| 2020 | https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament | https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament |

**2020 Analytics Track:** https://www.kaggle.com/c/march-madness-analytics-2020

#### Era 3: Community Return (2021-2022)
| Year | Men's | Women's |
|------|-------|---------|
| 2021 | https://www.kaggle.com/competitions/ncaam-march-mania-2021 | https://www.kaggle.com/competitions/ncaaw-march-mania-2021 |
| 2021 Spread | https://www.kaggle.com/competitions/ncaam-march-mania-2021-spread | https://www.kaggle.com/c/ncaaw-march-mania-2021-spread |
| 2022 | https://www.kaggle.com/competitions/mens-march-mania-2022 | https://www.kaggle.com/competitions/womens-march-mania-2022 |

#### Era 4: Unified (2023-2026)
| Year | URL |
|------|-----|
| 2023 | https://www.kaggle.com/competitions/march-machine-learning-mania-2023 |
| 2024 | https://www.kaggle.com/competitions/march-machine-learning-mania-2024 |
| 2025 | https://www.kaggle.com/competitions/march-machine-learning-mania-2025 |
| 2026 | https://www.kaggle.com/competitions/march-machine-learning-mania-2026 |
| 2026 Semi Spherical | https://www.kaggle.com/competitions/march-machine-learning-mania-2026-semi-spherical-scoring |
| 2026 Logistic Brier | https://www.kaggle.com/competitions/march-machine-learning-mania-2026-logistic-brier |

### 2.3 External Rating Systems

| Source | URL | Cost | Key Metrics |
|--------|-----|------|-------------|
| **KenPom** | https://kenpom.com | $24.95/yr | AdjO, AdjD, AdjEM, AdjT, Luck, SOS |
| **Barttorvik** | https://barttorvik.com | Free | T-Rank, Barthag, adjusted efficiency |
| **Massey Ratings** | https://masseyratings.com/cb/ncaa-d1/ratings | Free | 100+ system composite |
| **EvanMiya** | https://evanmiya.com | $29.99/mo | BPR (Bayesian Performance Rating) |
| **Haslametrics** | https://haslametrics.com | Free | Team-level unique stats |
| **Sports Reference** | https://www.sports-reference.com/cbb/ | Free | Complete historical box scores |

**Python access:**
- KenPom: `kenpompy` library (unofficial scraper)
- Barttorvik: `toRvik` (R) or direct scraping
- Massey: already in Kaggle `MMasseyOrdinals.csv`

### 2.4 BigQuery Public Dataset

- **Project:** `bigquery-public-data`
- **Dataset:** `ncaa_basketball`
- **Key tables:** `mbb_games_sr`, `mbb_players_games_sr`, `mbb_pbp_sr`, `mbb_historical_tournament_games`
- **Coverage:** Play-by-play back to 2009, final scores to 1996, some records to 1894-95
- **Access:** Free tier, query via `google-cloud-bigquery` Python client

### 2.5 Prediction Markets / Betting Odds

| Platform | URL | API | Notes |
|----------|-----|-----|-------|
| **Kalshi** | https://kalshi.com | https://docs.kalshi.com | CFTC-regulated, 0% maker fee, game-by-game + futures |
| **Polymarket** | https://polymarket.com | https://github.com/Polymarket/py-clob-client | CLOB on Polygon, NOT available to US users for sports |
| **DraftKings** | https://sportsbook.draftkings.com | Odds API | Spreads, totals, moneylines |
| **Historical odds** | Kaggle datasets / covers.com | Scraping | For backtesting |

---

## 3. Methodology

### 3.1 Feature Engineering

#### Layer 1: Static Team Ratings (per season)
- KenPom AdjEM, AdjO, AdjD, AdjT
- Barttorvik T-Rank, Barthag
- Massey composite rank (from MMasseyOrdinals — use latest pre-tournament snapshot)
- Seed number (1-16) — extremely strong baseline feature
- Conference strength (avg AdjEM of conference)

#### Layer 2: Game-Level Features (for each potential matchup)
- Seed difference (Δseed)
- Rating difference (ΔKenPom, ΔBarttorvik, ΔMassey)
- Tempo mismatch (|AdjT_A - AdjT_B|)
- Offensive vs Defensive style clash (AdjO_A vs AdjD_B and vice versa)
- Geographic distance to game site (from MGameCities)
- Coach tournament experience (years in tournament from MTeamCoaches)
- Conference tournament performance (momentum signal)

#### Layer 3: Temporal / Momentum Features
- Rolling 10-game win% and margin
- Late-season rating trend (Δrating over last 30 days)
- Injury-adjusted ratings (if EvanMiya data available)
- Days of rest between games (within tournament)

#### Layer 4: Market Features
- Pre-tournament futures odds (Kalshi championship price)
- Game-by-game line / moneyline (historical from covers.com, live from Kalshi)
- Market-implied probability vs model probability → "disagreement signal"

### 3.2 Models

#### Baseline Tier
1. **Seed-based logistic regression**: P(upset) = f(seed_diff) — the simplest meaningful model
2. **Elo ratings**: Season-long Elo with K-factor tuning, regressed at season start
3. **KenPom logistic**: Logistic regression on KenPom AdjEM difference — the standard strong baseline

#### Advanced Tier
4. **XGBoost / LightGBM**: Gradient boosting on full feature set (Layer 1-3). Walk-forward cross-validation (train on seasons 1-T, validate on season T+1).
5. **Mixture-of-Experts (MoE)**:
   - **Gating network**: Classifies each matchup into K "game types" (e.g., chalk, upset-prone, toss-up, Cinderella) based on seed/rating gap, conference, tempo
   - **Expert networks**: Each expert is a specialized logistic/XGBoost model trained on its cluster
   - **Training**: EM algorithm — alternate between assigning games to experts and updating expert parameters
   - Why MoE: A 1-seed vs 16-seed game has fundamentally different dynamics than a 5 vs 12 — forcing one model to handle both is suboptimal
6. **Transformer sequence model**:
   - Input: Team's season as a sequence of game embeddings (opponent rating, margin, location, stats)
   - Architecture: Small Transformer encoder (4 layers, 64-dim) → [CLS] token as team embedding
   - Matchup prediction: MLP on concatenated [CLS_A, CLS_B] embeddings
   - Why: Captures momentum, trajectory, and schedule-dependent context that static ratings miss

#### Meta Tier
7. **Ensemble**: Weighted average of models 1-6, weights optimized on holdout Brier score
8. **Market-calibrated ensemble**: Bayesian update of ensemble prediction with market-implied probability (shrink toward market when model confidence is low)
9. **Portfolio optimization** (Kaggle-specific): Generate N bracket portfolios that maximize P(top-K finish) given ensemble probabilities + estimated competitor distribution (Landgraf 2017 strategy)

### 3.3 Evaluation Framework

- **Metric**: Brier Score (MSE of probabilities vs 0/1 outcomes)
- **Backtest**: Leave-one-tournament-out (LOTO) on 2014-2026 data (12 tournaments, ~756 games)
- **Calibration**: Reliability diagram — predicted 70% games should win ~70% of the time
- **Comparison benchmarks**:
  - Seed baseline: ~0.200 Brier
  - KenPom logistic: ~0.185 Brier
  - Vegas line: ~0.175 Brier
  - Target: < 0.170 Brier (would be competitive for Kaggle top-10)

### 3.4 Kalshi Live Trading Strategy

- **Signal**: Model probability vs Kalshi market price → trade when |Δp| > threshold
- **Sizing**: Kelly criterion (fractional Kelly ~0.25 for safety)
- **Execution**: REST API via `docs.kalshi.com`, limit orders only (0% maker fee)
- **Risk**: Max 5% of bankroll per game, max 20% total exposure
- **Backtesting**: Simulate on 2025-2026 tournament with historical Kalshi prices

---

## 4. Technical Stack

```
Python 3.11+
├── Data: pandas, polars, google-cloud-bigquery
├── ML: scikit-learn, xgboost, lightgbm
├── DL: pytorch (Transformer, MoE)
├── Stats: scipy, statsmodels
├── Viz: matplotlib, seaborn, plotly
├── API: requests, httpx (Kalshi), kenpompy
└── Infra: jupyter, pytest, ruff
```

---

## 5. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| 63 games too few for DL | Model overfits | Heavy regularization + pretrain on regular season |
| KenPom paywall changes | Lose key features | Barttorvik (free) as fallback |
| Kalshi regulatory shutdown | No live trading | Backtest-only mode |
| Simple models beat complex | Wasted DL effort | Always benchmark against Elo + Logistic |
| Luck dominates in 1 tournament | Bad result despite good model | Evaluate on 12-year backtest, not single year |

---

## 6. References

1. Sill, J. (2010). Improved NBA Adjusted +/- Using Regularization and Out-of-Sample Testing. MIT SSAC.
2. Stern, H. (1994). A Brownian Motion Model for the Progress of Sports Scores. JASA.
3. Landgraf, A. (2017). March ML Mania 1st Place Solution — Meta-Prediction Strategy.
4. Pomeroy, K. (2004-present). KenPom.com — Adjusted Efficiency Ratings.
5. Jacobs, R. et al. (1991). Adaptive Mixtures of Local Experts. Neural Computation.
6. Vaswani, A. et al. (2017). Attention Is All You Need.
7. Sokol, J. et al. Georgia Tech LRMC — Logistic Regression / Markov Chain for NCAA.
