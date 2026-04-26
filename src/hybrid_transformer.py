"""
Hybrid Transformer: Transformer encoder on game sequences + Barttorvik
static features fed to the matchup MLP.

Idea: keep the temporal model that learns trajectory dynamics, but give the
matchup head explicit access to the cross-team continuous efficiency data
(Barttorvik AdjOE, AdjDE, NetRtg, Barthag, AdjTempo) that drove the
Multi-Feature Logistic to 0.159.

Architecture:
  TeamSeasonEncoder (existing) -> CLS embedding (d_model)
  Concat with Barttorvik vector (5 dims per team)
  Hybrid MatchupPredictor: MLP on [emb_a; emb_b; bart_a; bart_b; bart_diff]
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin

from src.sequence_features import build_team_season_sequence
from src.transformer_model import (
    TeamSeasonEncoder, PretrainHead, RegSeasonDataset,
)
from src.data_collection import DATA_DIR


_BART_COLS = ["AdjOE", "AdjDE", "NetRtg", "Barthag", "AdjTempo"]


def _load_barttorvik(season: int) -> pd.DataFrame | None:
    path = DATA_DIR / "external" / f"barttorvik_{season}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "TeamID" in df.columns:
        return df.set_index("TeamID")
    return None


def _get_bart_vector(bart: pd.DataFrame, team_id: int) -> np.ndarray:
    """Get 5-dim Barttorvik vector, fallback to league average if missing."""
    defaults = np.array([105.0, 105.0, 0.0, 0.5, 67.0], dtype=np.float32)
    if bart is None or team_id not in bart.index:
        return defaults
    row = bart.loc[team_id]
    vec = np.zeros(5, dtype=np.float32)
    for i, c in enumerate(_BART_COLS):
        try:
            vec[i] = float(row[c])
        except (TypeError, ValueError, KeyError):
            vec[i] = defaults[i]
    return vec


class HybridMatchupPredictor(nn.Module):
    """Matchup MLP that takes both Transformer embeddings and static Barttorvik."""

    def __init__(self, d_model=32, d_static=5, dropout=0.2):
        super().__init__()
        # Inputs: emb_a (d_model), emb_b (d_model), bart_a (5), bart_b (5)
        d_input = 2 * d_model + 2 * d_static
        self.mlp = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, emb_a, emb_b, bart_a, bart_b):
        x = torch.cat([emb_a, emb_b, bart_a, bart_b], dim=1)
        return self.mlp(x).squeeze(-1)


class HybridDataset(Dataset):
    def __init__(self, seq_a, len_a, seq_b, len_b, bart_a, bart_b, labels):
        self.seq_a = torch.FloatTensor(seq_a)
        self.len_a = torch.LongTensor(len_a)
        self.seq_b = torch.FloatTensor(seq_b)
        self.len_b = torch.LongTensor(len_b)
        self.bart_a = torch.FloatTensor(bart_a)
        self.bart_b = torch.FloatTensor(bart_b)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.seq_a[idx], self.len_a[idx],
                self.seq_b[idx], self.len_b[idx],
                self.bart_a[idx], self.bart_b[idx],
                self.labels[idx])


class HybridTransformerWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible Hybrid Transformer.

    Trains the encoder on regular-season game outcomes (no Barttorvik signal,
    purely temporal), then trains a matchup MLP that gets both the team
    embeddings and the static Barttorvik vectors.
    """

    def __init__(self, data, d_model=32, n_layers=2, n_heads=2,
                 pretrain_epochs=10, finetune_epochs=30,
                 lr=1e-3, weight_decay=1e-4,
                 max_games=35, dropout=0.15,
                 pretrain_samples_per_season=400):
        self.data = data
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_games = max_games
        self.dropout = dropout
        self.pretrain_samples_per_season = pretrain_samples_per_season
        self.encoder = None
        self.predictor = None
        self.device = torch.device("cpu")
        self._bart_cache: dict[int, pd.DataFrame] = {}
        self._bart_means: np.ndarray | None = None

    def _bart_for_season(self, season: int) -> pd.DataFrame | None:
        if season not in self._bart_cache:
            self._bart_cache[season] = _load_barttorvik(season)
        return self._bart_cache[season]

    def _normalize_bart(self, vec: np.ndarray) -> np.ndarray:
        """Standardize Barttorvik vector using training-set statistics."""
        if self._bart_means is None:
            return vec
        return (vec - self._bart_means) / self._bart_stds

    def _compute_bart_stats(self, seasons):
        """Compute mean/std of Barttorvik features over all training teams."""
        all_vecs = []
        for season in seasons:
            bart = self._bart_for_season(season)
            if bart is None:
                continue
            for tid in bart.index:
                vec = _get_bart_vector(bart, tid)
                all_vecs.append(vec)
        if not all_vecs:
            self._bart_means = np.zeros(5, dtype=np.float32)
            self._bart_stds = np.ones(5, dtype=np.float32)
            return
        arr = np.array(all_vecs)
        self._bart_means = arr.mean(axis=0).astype(np.float32)
        self._bart_stds = arr.std(axis=0).astype(np.float32) + 1e-6

    def _build_pretrain_data(self, seasons):
        """Reuse the existing pretrain logic, no Barttorvik in this phase."""
        detail = self.data.get("regular_detail")
        if detail is None:
            return None

        all_seq_a, all_len_a = [], []
        all_seq_b, all_len_b = [], []
        all_labels = []

        for season in seasons:
            sg = detail[detail["Season"] == season]
            if sg.empty:
                continue

            team_ids = set(sg["WTeamID"].tolist() + sg["LTeamID"].tolist())
            full_seqs = {}
            for tid in team_ids:
                full_seqs[tid] = build_team_season_sequence(
                    self.data, tid, season, self.max_games
                )

            if len(sg) > self.pretrain_samples_per_season:
                sg = sg.sample(n=self.pretrain_samples_per_season, random_state=season)

            for _, g in sg.iterrows():
                w, l = int(g["WTeamID"]), int(g["LTeamID"])
                day = int(g["DayNum"])
                if day < 15:
                    continue

                seq_w, len_w_full = full_seqs.get(w, (np.zeros((self.max_games, 13)), 0))
                seq_l, len_l_full = full_seqs.get(l, (np.zeros((self.max_games, 13)), 0))

                frac = min(day / 154.0, 1.0)
                len_w = max(int(len_w_full * frac) - 1, 1) if len_w_full > 0 else 0
                len_l = max(int(len_l_full * frac) - 1, 1) if len_l_full > 0 else 0
                if len_w == 0 or len_l == 0:
                    continue

                trunc_w = np.zeros_like(seq_w)
                trunc_w[:len_w] = seq_w[:len_w]
                trunc_l = np.zeros_like(seq_l)
                trunc_l[:len_l] = seq_l[:len_l]

                if w < l:
                    all_seq_a.append(trunc_w)
                    all_len_a.append(len_w)
                    all_seq_b.append(trunc_l)
                    all_len_b.append(len_l)
                    all_labels.append(1.0)
                else:
                    all_seq_a.append(trunc_l)
                    all_len_a.append(len_l)
                    all_seq_b.append(trunc_w)
                    all_len_b.append(len_w)
                    all_labels.append(0.0)

        if not all_labels:
            return None

        return (np.array(all_seq_a), np.array(all_len_a),
                np.array(all_seq_b), np.array(all_len_b),
                np.array(all_labels))

    def _pretrain(self, seasons):
        pt_data = self._build_pretrain_data(seasons)
        if pt_data is None:
            return
        seq_a, len_a, seq_b, len_b, labels = pt_data
        print(f"    Pretrain pairs: {len(labels)}")

        dataset = RegSeasonDataset(seq_a, len_a, seq_b, len_b, labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        head = PretrainHead(d_model=self.d_model).to(self.device)
        opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(head.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.pretrain_epochs)
        criterion = nn.BCEWithLogitsLoss()

        self.encoder.train()
        head.train()

        for epoch in range(self.pretrain_epochs):
            for sa, la, sb, lb, lbl in loader:
                sa, sb, lbl = sa.to(self.device), sb.to(self.device), lbl.to(self.device)
                emb_a = self.encoder(sa, la)
                emb_b = self.encoder(sb, lb)
                loss = criterion(head(emb_a, emb_b), lbl)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                opt.step()
            sched.step()

        del head

    def _build_features(self, X):
        """Build sequences and Barttorvik vectors for matchups in X."""
        n = len(X)
        seq_a = np.zeros((n, self.max_games, 13))
        len_a = np.zeros(n, dtype=int)
        seq_b = np.zeros((n, self.max_games, 13))
        len_b = np.zeros(n, dtype=int)
        bart_a = np.zeros((n, 5), dtype=np.float32)
        bart_b = np.zeros((n, 5), dtype=np.float32)

        seq_cache = {}

        for i, (_, row) in enumerate(X.iterrows()):
            season = int(row["Season"])
            ta, tb = int(row["TeamA"]), int(row["TeamB"])

            if (ta, season) not in seq_cache:
                seq_cache[(ta, season)] = build_team_season_sequence(
                    self.data, ta, season, self.max_games
                )
            if (tb, season) not in seq_cache:
                seq_cache[(tb, season)] = build_team_season_sequence(
                    self.data, tb, season, self.max_games
                )

            seq_a[i], len_a[i] = seq_cache[(ta, season)]
            seq_b[i], len_b[i] = seq_cache[(tb, season)]

            bart = self._bart_for_season(season)
            bart_a[i] = self._normalize_bart(_get_bart_vector(bart, ta))
            bart_b[i] = self._normalize_bart(_get_bart_vector(bart, tb))

        return seq_a, len_a, seq_b, len_b, bart_a, bart_b

    def fit(self, X, y):
        train_seasons = sorted(X["Season"].unique().tolist())
        self._compute_bart_stats(train_seasons)

        # Build encoder
        self.encoder = TeamSeasonEncoder(
            d_input=13, d_model=self.d_model,
            n_layers=self.n_layers, n_heads=self.n_heads,
            dropout=self.dropout, max_games=self.max_games,
        ).to(self.device)

        # Phase 1: pretrain encoder
        if self.pretrain_epochs > 0:
            print("    Pretraining encoder...")
            self._pretrain(train_seasons)

        # Phase 2: build hybrid predictor
        self.predictor = HybridMatchupPredictor(
            d_model=self.d_model, d_static=5, dropout=self.dropout,
        ).to(self.device)

        # Phase 3: build matchup features for tournament data
        seq_a, len_a, seq_b, len_b, bart_a, bart_b = self._build_features(X)
        dataset = HybridDataset(seq_a, len_a, seq_b, len_b, bart_a, bart_b, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.BCEWithLogitsLoss()

        # Freeze encoder for first half of finetune
        for p in self.encoder.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(self.predictor.parameters(),
                               lr=self.lr, weight_decay=self.weight_decay)
        self.encoder.eval()
        self.predictor.train()

        freeze_epochs = self.finetune_epochs // 2
        for epoch in range(freeze_epochs):
            for sa, la, sb, lb, ba, bb, lbl in loader:
                sa, sb = sa.to(self.device), sb.to(self.device)
                ba, bb, lbl = ba.to(self.device), bb.to(self.device), lbl.to(self.device)
                with torch.no_grad():
                    emb_a = self.encoder(sa, la)
                    emb_b = self.encoder(sb, lb)
                loss = criterion(self.predictor(emb_a, emb_b, ba, bb), lbl)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Unfreeze with low LR
        for p in self.encoder.parameters():
            p.requires_grad = True
        opt = torch.optim.Adam([
            {"params": self.encoder.parameters(), "lr": self.lr * 0.1},
            {"params": self.predictor.parameters(), "lr": self.lr * 0.5},
        ], weight_decay=self.weight_decay)
        self.encoder.train()
        self.predictor.train()

        unfreeze_epochs = self.finetune_epochs - freeze_epochs
        for epoch in range(unfreeze_epochs):
            for sa, la, sb, lb, ba, bb, lbl in loader:
                sa, sb = sa.to(self.device), sb.to(self.device)
                ba, bb, lbl = ba.to(self.device), bb.to(self.device), lbl.to(self.device)
                emb_a = self.encoder(sa, la)
                emb_b = self.encoder(sb, lb)
                loss = criterion(self.predictor(emb_a, emb_b, ba, bb), lbl)
                opt.zero_grad()
                loss.backward()
                opt.step()

        return self

    def predict_proba(self, X):
        seq_a, len_a, seq_b, len_b, bart_a, bart_b = self._build_features(X)
        self.encoder.eval()
        self.predictor.eval()

        with torch.no_grad():
            sa = torch.FloatTensor(seq_a).to(self.device)
            sb = torch.FloatTensor(seq_b).to(self.device)
            la = torch.LongTensor(len_a)
            lb = torch.LongTensor(len_b)
            ba = torch.FloatTensor(bart_a).to(self.device)
            bb = torch.FloatTensor(bart_b).to(self.device)

            emb_a = self.encoder(sa, la)
            emb_b = self.encoder(sb, lb)
            logits = self.predictor(emb_a, emb_b, ba, bb)
            probs = torch.sigmoid(logits).cpu().numpy()

        return np.column_stack([1 - probs, probs])
