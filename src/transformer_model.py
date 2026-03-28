"""
Transformer sequence model for NCAA tournament prediction.

Architecture:
  1. TeamSeasonEncoder: Transformer encoder on game sequences -> team embedding
  2. MatchupPredictor: MLP on concatenated team embeddings -> P(TeamA wins)
  3. TransformerSklearnWrapper: sklearn-compatible wrapper for LOTO backtest

Pre-training: predict regular-season game outcomes (encoder + simple head)
Fine-tuning: predict tournament outcomes (freeze encoder, train matchup MLP)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin

from src.sequence_features import build_team_season_sequence


class TeamSeasonEncoder(nn.Module):
    """Transformer encoder: game sequence -> team embedding."""

    def __init__(self, d_input=13, d_model=64, n_layers=4, n_heads=4, dropout=0.1, max_games=35):
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
        x = torch.cat([cls, x], dim=1)  # (batch, seq_len+1, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Create padding mask if lengths provided
        mask = None
        if lengths is not None:
            # mask shape: (batch, seq_len+1), True = ignore
            mask = torch.zeros(batch_size, seq_len + 1, dtype=torch.bool, device=x.device)
            for i, l in enumerate(lengths):
                mask[i, l + 1:] = True  # +1 for CLS token

        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x[:, 0, :])  # CLS token
        return x


class MatchupPredictor(nn.Module):
    """MLP on concatenated team embeddings -> P(TeamA wins)."""

    def __init__(self, d_model=64, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, emb_a, emb_b):
        """Returns logits (not probabilities)."""
        x = torch.cat([emb_a, emb_b], dim=1)
        return self.mlp(x).squeeze(-1)


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

    On fit():
      1. Build game sequences for all tournament teams
      2. Pre-train encoder on regular-season game prediction (optional)
      3. Fine-tune matchup MLP on tournament data

    On predict_proba(): build sequences, forward through encoder + MLP.
    """

    def __init__(self, data, d_model=64, n_layers=4, n_heads=4,
                 finetune_epochs=30, lr=1e-3, weight_decay=1e-4,
                 max_games=35, dropout=0.15):
        self.data = data
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_games = max_games
        self.dropout = dropout
        self.encoder = None
        self.predictor = None
        self.device = torch.device("cpu")

    def _build_sequences(self, X):
        """Build game sequences for all matchups in X."""
        n = len(X)
        d_input = 13  # number of features per game in sequence_features
        all_seq_a = np.zeros((n, self.max_games, d_input))
        all_len_a = np.zeros(n, dtype=int)
        all_seq_b = np.zeros((n, self.max_games, d_input))
        all_len_b = np.zeros(n, dtype=int)

        # Cache sequences per (team, season) to avoid recomputation
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

        # Build sequences
        seq_a, len_a, seq_b, len_b = self._build_sequences(X)
        dataset = TourneyMatchupDataset(seq_a, len_a, seq_b, len_b, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Train encoder + predictor jointly on tournament data
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        self.encoder.train()
        self.predictor.train()

        for epoch in range(self.finetune_epochs):
            total_loss = 0
            for sa, la, sb, lb, labels in loader:
                sa, sb = sa.to(self.device), sb.to(self.device)
                labels = labels.to(self.device)

                emb_a = self.encoder(sa, la)
                emb_b = self.encoder(sb, lb)
                logits = self.predictor(emb_a, emb_b)

                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

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
