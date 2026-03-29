"""
Transformer sequence model for NCAA tournament prediction.

Architecture:
  1. TeamSeasonEncoder: Transformer encoder on game sequences -> team embedding
  2. MatchupPredictor: MLP on concatenated team embeddings -> P(TeamA wins)
  3. TransformerSklearnWrapper: sklearn-compatible wrapper for LOTO backtest

Training pipeline:
  Phase 1 (pre-train): predict regular-season game outcomes (encoder + simple head)
  Phase 2 (fine-tune): predict tournament outcomes (freeze encoder, train matchup MLP)
  Phase 3 (unfreeze): joint fine-tune with low LR on encoder
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin

from src.sequence_features import build_team_season_sequence, build_team_partial_sequence


class TeamSeasonEncoder(nn.Module):
    """Transformer encoder: game sequence -> team embedding."""

    def __init__(self, d_input=13, d_model=32, n_layers=2, n_heads=2, dropout=0.1, max_games=35):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(d_input, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_games + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, lengths=None):
        """
        x: (batch, seq_len, d_input)
        lengths: (batch,) actual sequence lengths
        Returns: (batch, d_model) CLS token output
        """
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Create padding mask if lengths provided
        mask = None
        if lengths is not None:
            mask = torch.zeros(batch_size, seq_len + 1, dtype=torch.bool, device=x.device)
            for i, l in enumerate(lengths):
                mask[i, l + 1:] = True

        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x[:, 0, :])
        return x


class PretrainHead(nn.Module):
    """Simple head for pretraining: predict game outcome from two team embeddings."""

    def __init__(self, d_model=32):
        super().__init__()
        self.head = nn.Linear(d_model * 2, 1)

    def forward(self, emb_a, emb_b):
        x = torch.cat([emb_a, emb_b], dim=1)
        return self.head(x).squeeze(-1)


class MatchupPredictor(nn.Module):
    """MLP on concatenated team embeddings -> P(TeamA wins)."""

    def __init__(self, d_model=32, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, emb_a, emb_b):
        x = torch.cat([emb_a, emb_b], dim=1)
        return self.mlp(x).squeeze(-1)


class RegSeasonDataset(Dataset):
    """Dataset for regular-season pretraining pairs."""

    def __init__(self, seq_a, len_a, seq_b, len_b, labels):
        self.seq_a = torch.FloatTensor(seq_a)
        self.len_a = torch.LongTensor(len_a)
        self.seq_b = torch.FloatTensor(seq_b)
        self.len_b = torch.LongTensor(len_b)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.seq_a[idx], self.len_a[idx],
                self.seq_b[idx], self.len_b[idx],
                self.labels[idx])


class TourneyMatchupDataset(Dataset):
    """Dataset for tournament matchups with pre-computed sequences."""

    def __init__(self, seq_a, len_a, seq_b, len_b, labels):
        self.seq_a = torch.FloatTensor(seq_a)
        self.len_a = torch.LongTensor(len_a)
        self.seq_b = torch.FloatTensor(seq_b)
        self.len_b = torch.LongTensor(len_b)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.seq_a[idx], self.len_a[idx],
                self.seq_b[idx], self.len_b[idx],
                self.labels[idx])


class TransformerSklearnWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for Transformer tournament predictor.

    Training pipeline:
      1. Pre-train encoder on regular-season game outcomes (~5K pairs per season)
      2. Freeze encoder, train matchup MLP on tournament data
      3. Unfreeze encoder with 0.1x LR for final joint fine-tuning
    """

    def __init__(self, data, d_model=32, n_layers=2, n_heads=2,
                 pretrain_epochs=15, finetune_epochs=30,
                 lr=1e-3, weight_decay=1e-4,
                 max_games=35, dropout=0.15,
                 pretrain_samples_per_season=500):
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

    def _build_pretrain_data(self, seasons):
        """
        Build pretraining pairs from regular-season games.
        Uses full-season sequences and truncates by game index for speed.
        """
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

            # Pre-build full sequences for all teams in this season (fast, cached)
            team_ids = set(sg["WTeamID"].tolist() + sg["LTeamID"].tolist())
            full_seqs = {}
            for tid in team_ids:
                full_seqs[tid] = build_team_season_sequence(
                    self.data, tid, season, self.max_games
                )

            # Sample games
            if len(sg) > self.pretrain_samples_per_season:
                sg = sg.sample(n=self.pretrain_samples_per_season, random_state=season)

            for _, g in sg.iterrows():
                w, l = int(g["WTeamID"]), int(g["LTeamID"])
                day = int(g["DayNum"])

                if day < 15:
                    continue

                # Truncate full sequences to games before this day
                # Full sequence is sorted by day, so find cutoff index
                for tid, label_if_a in [(w, 1.0 if w < l else 0.0), (l, 1.0 if l < w else 0.0)]:
                    pass  # just iterating logic below

                seq_w, len_w_full = full_seqs.get(w, (np.zeros((self.max_games, 13)), 0))
                seq_l, len_l_full = full_seqs.get(l, (np.zeros((self.max_games, 13)), 0))

                # Truncate: day_norm is last feature (idx 12), but we need raw day info
                # Approximate: use fraction of games (game at position k out of N happened
                # before day D if k/N < D/max_day). Simpler: keep first ~60-80% of games
                # as a rough causal cut (most sampled games are mid-to-late season).
                # More precise: use the game index proportional to day fraction.
                frac = min(day / 154.0, 1.0)  # 154 ≈ typical last regular season day
                len_w = max(int(len_w_full * frac) - 1, 1) if len_w_full > 0 else 0
                len_l = max(int(len_l_full * frac) - 1, 1) if len_l_full > 0 else 0

                if len_w == 0 or len_l == 0:
                    continue

                # Create truncated copies
                trunc_w = np.zeros_like(seq_w)
                trunc_w[:len_w] = seq_w[:len_w]
                trunc_l = np.zeros_like(seq_l)
                trunc_l[:len_l] = seq_l[:len_l]

                # Canonical ordering: lower TeamID = TeamA
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
        """Pre-train encoder on regular-season game prediction."""
        print("    Building pretrain data...")
        pt_data = self._build_pretrain_data(seasons)
        if pt_data is None:
            print("    No pretrain data available, skipping.")
            return

        seq_a, len_a, seq_b, len_b, labels = pt_data
        print(f"    Pretrain pairs: {len(labels)}")

        dataset = RegSeasonDataset(seq_a, len_a, seq_b, len_b, labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        pretrain_head = PretrainHead(d_model=self.d_model).to(self.device)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(pretrain_head.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.pretrain_epochs
        )
        criterion = nn.BCEWithLogitsLoss()

        self.encoder.train()
        pretrain_head.train()

        for epoch in range(self.pretrain_epochs):
            total_loss = 0
            n_batches = 0
            for sa, la, sb, lb, lbl in loader:
                sa, sb = sa.to(self.device), sb.to(self.device)
                lbl = lbl.to(self.device)

                emb_a = self.encoder(sa, la)
                emb_b = self.encoder(sb, lb)
                logits = pretrain_head(emb_a, emb_b)
                loss = criterion(logits, lbl)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            if (epoch + 1) % 5 == 0:
                print(f"    Pretrain epoch {epoch+1}/{self.pretrain_epochs}, "
                      f"loss={total_loss/max(n_batches,1):.4f}")

        # Discard pretrain head, keep encoder weights
        del pretrain_head

    def _build_sequences(self, X):
        """Build game sequences for all matchups in X."""
        n = len(X)
        d_input = 13
        all_seq_a = np.zeros((n, self.max_games, d_input))
        all_len_a = np.zeros(n, dtype=int)
        all_seq_b = np.zeros((n, self.max_games, d_input))
        all_len_b = np.zeros(n, dtype=int)

        cache = {}
        for i, (_, row) in enumerate(X.iterrows()):
            season = int(row["Season"])
            ta, tb = int(row["TeamA"]), int(row["TeamB"])

            if (ta, season) not in cache:
                cache[(ta, season)] = build_team_season_sequence(
                    self.data, ta, season, self.max_games
                )
            if (tb, season) not in cache:
                cache[(tb, season)] = build_team_season_sequence(
                    self.data, tb, season, self.max_games
                )

            all_seq_a[i], all_len_a[i] = cache[(ta, season)]
            all_seq_b[i], all_len_b[i] = cache[(tb, season)]

        return all_seq_a, all_len_a, all_seq_b, all_len_b

    def fit(self, X, y):
        d_input = 13
        self.encoder = TeamSeasonEncoder(
            d_input=d_input, d_model=self.d_model,
            n_layers=self.n_layers, n_heads=self.n_heads,
            dropout=self.dropout, max_games=self.max_games,
        ).to(self.device)
        self.predictor = MatchupPredictor(
            d_model=self.d_model, dropout=self.dropout
        ).to(self.device)

        # Phase 1: Pre-train encoder on regular-season data
        train_seasons = sorted(X["Season"].unique().tolist())
        if self.pretrain_epochs > 0:
            self._pretrain(train_seasons)

        # Phase 2: Freeze encoder, train matchup MLP on tournament data
        seq_a, len_a, seq_b, len_b = self._build_sequences(X)
        dataset = TourneyMatchupDataset(seq_a, len_a, seq_b, len_b, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.BCEWithLogitsLoss()

        # Freeze encoder for first phase of fine-tuning
        for param in self.encoder.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=self.lr, weight_decay=self.weight_decay,
        )

        self.encoder.eval()
        self.predictor.train()

        freeze_epochs = self.finetune_epochs // 2
        for epoch in range(freeze_epochs):
            for sa, la, sb, lb, labels in loader:
                sa, sb = sa.to(self.device), sb.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    emb_a = self.encoder(sa, la)
                    emb_b = self.encoder(sb, lb)
                logits = self.predictor(emb_a, emb_b)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Phase 3: Unfreeze encoder with low LR for joint fine-tuning
        for param in self.encoder.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam([
            {"params": self.encoder.parameters(), "lr": self.lr * 0.1},
            {"params": self.predictor.parameters(), "lr": self.lr * 0.5},
        ], weight_decay=self.weight_decay)

        self.encoder.train()
        self.predictor.train()

        unfreeze_epochs = self.finetune_epochs - freeze_epochs
        for epoch in range(unfreeze_epochs):
            for sa, la, sb, lb, labels in loader:
                sa, sb = sa.to(self.device), sb.to(self.device)
                labels = labels.to(self.device)

                emb_a = self.encoder(sa, la)
                emb_b = self.encoder(sb, lb)
                logits = self.predictor(emb_a, emb_b)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.predictor.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

        return self

    def predict_proba(self, X):
        seq_a, len_a, seq_b, len_b = self._build_sequences(X)

        self.encoder.eval()
        self.predictor.eval()

        with torch.no_grad():
            sa = torch.FloatTensor(seq_a).to(self.device)
            sb = torch.FloatTensor(seq_b).to(self.device)
            la = torch.LongTensor(len_a)
            lb = torch.LongTensor(len_b)

            emb_a = self.encoder(sa, la)
            emb_b = self.encoder(sb, lb)
            logits = self.predictor(emb_a, emb_b)
            probs = torch.sigmoid(logits).cpu().numpy()

        return np.column_stack([1 - probs, probs])
