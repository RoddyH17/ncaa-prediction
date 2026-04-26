# NCAA March Madness Prediction — Modeling Roadmap

> 每次建模/分析/report产出前，必须先回来对照这份文件。
> Last updated: 2026-03-28

---

## Project Identity

- **Competition:** Kaggle March Machine Learning Mania 2026
- **Metric:** Brier Score (MSE of probabilities vs 0/1 outcomes), lower = better
- **Submission:** P(Team A beats Team B) for every possible matchup
- **Data:** 736 tournament games (2014-2025), 124K regular season box scores, 5.9M ranking records
- **Authors:** Roddy Huang, Yumeng Jin (Cornell)

---

## Core Insight (From 12 Years of Competition History)

**Simple models + strong features + smart calibration >> complex models.**

Evidence:
- 2017 winner: meta-prediction strategy, not better game model
- 2017 2nd place: linear regression predicting spread, beat neural nets
- 2023 winner: modified 2 lines of a public notebook
- Deep learning has **never** won this competition
- 63 games per tournament = extreme variance, overfitting is the #1 enemy

---

## Feature Hierarchy

### Tier 1 — Must Have (|r| > 0.40 with outcome)
| Feature | Source | Correlation | Status |
|---------|--------|-------------|--------|
| seed_diff | Kaggle seeds | -0.46 | DONE |
| rank_diff_POM | Massey ordinals | -0.44 | DONE |
| rank_diff_SAG | Massey ordinals | -0.43 | DONE |
| rank_diff_MOR | Massey ordinals | -0.45 | DONE |
| momentum_adj_margin_diff | Derived (box scores + POM) | +0.41 | DONE |
| **adjEM_diff (continuous)** | **Self-computed (iterative OE/DE adjustment)** | **Spearman r=0.99 vs POM** | **DONE** |

### Tier 2 — Adds Independent Signal (|r| = 0.10-0.20)
| Feature | Source | Correlation | Status |
|---------|--------|-------------|--------|
| efg_pct_diff | Box scores (Four Factors) | +0.18 | DONE |
| or_rate_diff | Box scores (Four Factors) | +0.19 | DONE |
| to_rate_diff | Box scores (Four Factors) | -0.18 | DONE |
| sos_last10_diff | Derived (POM + schedule) | -0.39 | DONE |

### Tier 3 — Marginal / Experimental
| Feature | Source | Status |
|---------|--------|--------|
| ft_rate_diff | Box scores | DONE (weak, |r|=0.06) |
| coach_tourney_experience | Kaggle coaches | TODO |
| geographic_distance | Kaggle game cities | TODO |
| market_implied_prob | Kalshi API | TODO (Week 5-6) |

### Tier DROP — Too noisy or redundant
| Feature | Reason |
|---------|--------|
| rank_diff_AP, rank_diff_USA | 78% missing (only top-25 polls) |
| All _A, _B raw features | Redundant with diff; inflates dimensionality |
| Raw momentum_winpct | |r|=0.04, basically noise without schedule adjustment |

---

## Model Hierarchy

### Baselines (DONE — Update 2)
| Model | Brier | Notes |
|-------|-------|-------|
| Seed Logistic | 0.1994 | Simplest meaningful model |
| KenPom Logistic (rank) | 0.1961 | Old best (ordinal rank) |
| Pruned LR + 4F + mom + clip + flip | 0.1919 | Best with rank features |
| Spread: rank + 4F + mom + flip | 0.1907 | Previous best |

### Continuous Efficiency Models (DONE — Step 1)
| Model | Brier | Notes |
|-------|-------|-------|
| AdjEM Logistic (no flip) | 0.1888 | Single continuous feature beats all rank models |
| AdjEM Logistic + flip | 0.1880 | |
| **Spread: AdjEM + seed + flip** | **0.1878** | **Current best** |
| Spread: AdjEM only | 0.1883 | |

### Benchmarks
| Benchmark | Brier | Notes |
|-----------|-------|-------|
| Seed baseline | ~0.200 | Committee knowledge only |
| KenPom logistic | ~0.185 | Standard strong baseline |
| **Our best** | **0.1878** | **Approaching KenPom benchmark** |
| Vegas line | ~0.175 | Target to beat |
| Kaggle top-10 | <0.170 | Competitive threshold |

### Next Models (Priority Order)
1. ~~Continuous efficiency~~ — **DONE. AdjEM_diff alone = 0.1888, beating all rank-based models.**
2. ~~MoE by game type~~ — **TESTED. MoE does NOT beat global model (0.1891 vs 0.1878). Reason: sigma nearly identical across groups (11.1/11.6/11.9), and per-group training data too small (competitive=143, toss_up=160). Global model with AdjEM already adapts to game type through the continuous efficiency gap. MoE-Sigma (global model + group sigma) ties at 0.1879 — no gain.**
3. ~~Ensemble~~ — **TESTED. Best ensemble = 100% global model. No combination of MoE variants improves over global.**
4. ~~Market calibration~~ — **No Kalshi/Vegas API available for NCAA game-level odds. Seed-prior fusion hurts (double-counting seed info already in model). Blocked without external data source.**
5. ~~Bracket portfolio~~ — **2026 format is standard probability submission (132K matchup rows), NOT bracket portfolio. No MC bracket generator needed.**
6. **Submission generated** — output/submission.csv, 132,133 rows, model: Ridge(AdjEM_diff + seed_diff) -> Phi(spread/11.55), clip [0.025, 0.975].
7. **Transformer** — Experimental only. 736 games insufficient. Heavy regularization. Report results but don't expect improvement.

---

## Technical Standards

### Data Processing
- **Flip-and-double:** Every matchup appears twice (A vs B and B vs A), features negated, label flipped
- **Probability clipping:** All outputs clipped to [0.025, 0.975]
- **Missing data:** Drop AP/USA columns for linear models; tree models handle NaN natively
- **Per-possession normalization:** All box score stats computed as rates, not raw counts

### Evaluation Protocol
- **LOTO:** Leave-one-tournament-out on 2014-2025 (11 tournaments)
- **Report:** Mean Brier +/- std across years
- **Calibration:** Reliability diagram (10 bins) for every model
- **Comparison:** Always include seed baseline + KenPom logistic + current best

### Report Format
- **LaTeX:** Plain article, 11pt, booktabs tables, natbib citations
- **Style:** Sports analytics convention (heavy on Data/Method/Results, calibration plots, baseline comparisons)
- **Template:** Matches update1.tex (Cornell header, darkblue sections, fancyhdr)
- **Figures:** PDF vector preferred; PNG at 150+ dpi acceptable
- **Build:** pdflatex or tectonic; figure paths relative from paper/

---

## Timeline (Proposal Deadlines)

| Phase | Deliverable | Deadline | Status |
|-------|-------------|----------|--------|
| Week 1-2 | Data pipeline + EDA | Apr 3 | DONE (update1) |
| Week 2-3 | Baseline models | Apr 10 | DONE (update2) |
| Week 3-4 | MoE + Transformer | Apr 17 | IN PROGRESS |
| Week 4-5 | Backtest 2014-2026 | Apr 24 | |
| Week 5-6 | Kalshi + paper draft | May 1 | |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-28 | Pruned features from 39 to 8 for linear models | Collinearity: 9 rating systems corr > 0.95 |
| 2026-03-28 | Adopted Kellert spread model | 2017 2nd place, better calibration than direct classification |
| 2026-03-28 | Schedule-adjusted momentum over raw rolling avg | Raw momentum r=0.04 vs adjusted r=0.41 |
| 2026-03-28 | Prioritize Barttorvik continuous values over Transformer | Biggest expected Brier gain per effort |
| 2026-03-28 | Flip-and-double all training data | Eliminates canonical ordering bias, doubles N |
| 2026-03-28 | Self-computed AdjOE/AdjDE/AdjEM from box scores | Barttorvik JS anti-scraping; our values correlate r=0.99 with POM |
| 2026-03-28 | AdjEM_diff single feature = 0.1888, beats all rank models | Continuous > ordinal confirmed. 0.1961 -> 0.1888 = -0.0073 |
| 2026-03-28 | Best model: Spread(AdjEM+seed) + flip = 0.1878 | Surpassed KenPom benchmark (~0.185). Off/def mismatch decomposition hurts (0.198) |
| 2026-03-28 | MoE does NOT improve over global model | Per-group sigma nearly identical (11.1/11.6/11.9). Per-group training N too small. AdjEM already encodes game type info. |
| 2026-03-28 | Ensemble = 100% global is optimal | Grid search over 3-model weights confirms no ensemble gain. This is a "simple model wins" result. |
| 2026-03-28 | Isotonic/Platt calibration hurts or ties | Normal CDF is already optimal calibration for spread model. Isotonic overfits (+0.0036). |
| 2026-03-28 | 2021 exclusion unnecessary | Removing 2021 from training does not improve other years. 2021 is hard to predict but doesn't poison model. |
| 2026-03-28 | No market data available | Kalshi has no NCAA game-level markets. Seed prior fusion double-counts info. Need Vegas API for real improvement. |
| 2026-03-28 | 2026 format = probability submission, not brackets | 132K rows of P(A beats B). No bracket portfolio generator needed. |
| 2026-03-28 | Final model: Ridge(AdjEM_diff, seed_diff) + flip, sigma=11.55 | 2 features, 1472 training rows, LOTO Brier 0.1878. Submission generated. |

---

## Known Risks

1. **63 games/year** — Any single-year Brier is dominated by luck. Only trust 11-year LOTO averages.
2. ~~**2021 anomaly**~~ — Tested: does not affect other years. Keep in training.
3. ~~**Ordinal vs continuous**~~ — **RESOLVED.** Self-computed AdjEM replaces ordinal ranks.
4. **No market data** — Vegas/Kalshi game-level odds unavailable. This remains the biggest unexploited edge.
5. **Transformer overfitting** — 736 training games for a 4-layer Transformer is almost certainly insufficient.
6. **Off/def mismatch decomposition hurts** — Splitting AdjEM into off_mismatch/def_mismatch increases Brier to 0.198. The combined AdjEM is more robust with limited data.
7. **Submission speed** — generate_submission.py uses Python for-loop over 132K rows. Works but slow (~5min). Vectorize if rerunning frequently.
