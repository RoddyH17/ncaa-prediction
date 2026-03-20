# NCAA March Madness Prediction System

**Author:** Roddy， Yumeng 
**Timeline:** 2026-03 ~ 2026-05 (1.5 months)
**Target:** Kaggle March ML Mania 2027 + Kalshi Live Trading

## Project Overview

A probabilistic NCAA tournament prediction system combining mixture models, gradient boosting, and deep learning sequence models. Built on 12 years of Kaggle competition data + external rating systems (KenPom, Barttorvik, Massey).

## Directory Structure

```
ncaa_prediction/
├── data/                  # Raw & processed datasets (not committed)
│   ├── kaggle/            # Kaggle competition CSVs
│   ├── kenpom/            # KenPom scraped ratings
│   ├── barttorvik/        # T-Rank data
│   └── odds/              # Kalshi/betting market odds
├── notebooks/             # EDA & experiment notebooks
├── src/                   # Production pipeline
│   ├── data_collection.py # Data fetching & cleaning
│   ├── features.py        # Feature engineering
│   ├── models.py          # Model definitions
│   ├── ensemble.py        # Ensemble & meta-prediction
│   ├── evaluation.py      # Brier score, calibration, backtest
│   └── kalshi_trader.py   # Kalshi API integration
├── output/                # Figures, predictions, backtest results
├── paper/                 # LaTeX writeup
├── PROPOSAL.md            # Project proposal (this document below)
└── README.md              # This file
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download Kaggle data
kaggle competitions download -c march-machine-learning-mania-2026

# Run pipeline
python src/data_collection.py
python src/features.py
python src/models.py
python src/evaluation.py
```
