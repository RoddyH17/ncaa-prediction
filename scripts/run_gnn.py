"""
Graph Neural Network on team-game graph for NCAA tournament prediction.

Construction (per season):
  - Nodes = teams (typically 350+ Division I)
  - Edges = regular season games (undirected, pre-tournament)
  - Node features = pre-tournament Barttorvik AdjOE/AdjDE/Barthag/AdjTempo,
                    + seed (if in tournament, else 17 = unseeded), POM rank
  - Edge features = margin (TeamA - TeamB), location, day_norm

Architecture:
  GraphSAGE encoder (3 layers, hidden=64) -> per-team embedding
  MatchupHead: MLP on concatenated (node_a, node_b, abs_diff)
  Output: P(TeamA wins)

Training: per-season training. Predict tournament games using node embeddings
that were learned from regular-season game graph.

This is, to our knowledge, the first GNN application to NCAA tournament
prediction.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num


def build_graph_for_season(data: dict, season: int) -> tuple:
    """Build a single-season graph from regular season games + node features.

    Returns:
        Data object with x (node features), edge_index, edge_attr
        team_to_idx mapping
    """
    detail = data["regular_detail"]
    sg = detail[detail["Season"] == season]
    if sg.empty:
        return None, None

    teams_in_season = sorted(set(sg["WTeamID"].tolist() + sg["LTeamID"].tolist()))
    team_to_idx = {tid: i for i, tid in enumerate(teams_in_season)}

    # Node features: Barttorvik + seed + POM rank
    bart_path = DATA_DIR / "external" / f"barttorvik_{season}.csv"
    bart = pd.read_csv(bart_path).set_index("TeamID") if bart_path.exists() else None

    seeds = data["seeds"]
    s_season = seeds[seeds["Season"] == season].copy()
    s_season["SeedNum"] = s_season["Seed"].apply(_parse_seed_num)
    seed_map = dict(zip(s_season["TeamID"], s_season["SeedNum"]))

    massey = data["massey"]
    pom_df = massey[(massey["Season"] == season) & (massey["SystemName"] == "POM") &
                     (massey["RankingDayNum"] <= 133)]
    if not pom_df.empty:
        latest = pom_df["RankingDayNum"].max()
        pom_map = dict(zip(pom_df[pom_df["RankingDayNum"] == latest]["TeamID"],
                            pom_df[pom_df["RankingDayNum"] == latest]["OrdinalRank"]))
    else:
        pom_map = {}

    def safe_get(df, tid, col, default):
        if df is None or tid not in df.index:
            return default
        v = df.loc[tid, col]
        if hasattr(v, "iloc"):
            v = v.iloc[0]  # handle duplicate index
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    node_features = []
    for tid in teams_in_season:
        adjoe = safe_get(bart, tid, "AdjOE", 105.0)
        adjde = safe_get(bart, tid, "AdjDE", 105.0)
        barthag = safe_get(bart, tid, "Barthag", 0.5)
        tempo = safe_get(bart, tid, "AdjTempo", 67.0)
        seed = seed_map.get(tid, 17)
        pom = pom_map.get(tid, 365)

        node_features.append([
            (adjoe - 105) / 15,    # standardized AdjOE
            (adjde - 105) / 15,    # standardized AdjDE
            (barthag - 0.5) / 0.3,  # Barthag
            (tempo - 67) / 5,      # tempo
            (17 - seed) / 16,      # seed (higher value = better seed)
            (365 - pom) / 365,     # POM (higher value = better)
        ])

    x = torch.FloatTensor(node_features)

    # Edges: regular season games (undirected)
    edge_src = []
    edge_dst = []
    edge_attrs = []
    for _, g in sg.iterrows():
        i = team_to_idx[g["WTeamID"]]
        j = team_to_idx[g["LTeamID"]]
        margin = (g["WScore"] - g["LScore"]) / 30  # normalized margin
        loc = 1.0 if g.get("WLoc", "N") == "H" else (0.0 if g.get("WLoc", "N") == "N" else -1.0)
        day_norm = (g["DayNum"] - 0) / 132.0
        # Both directions: i->j with +margin (i won), j->i with -margin
        edge_src.extend([i, j])
        edge_dst.extend([j, i])
        edge_attrs.extend([
            [+margin, +loc, day_norm],
            [-margin, -loc, day_norm],
        ])

    edge_index = torch.LongTensor([edge_src, edge_dst])
    edge_attr = torch.FloatTensor(edge_attrs)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return graph, team_to_idx


class TeamGNN(nn.Module):
    """GraphSAGE encoder + matchup head."""

    def __init__(self, in_dim=6, hidden=32, n_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.dropout = dropout

        # Matchup head: input = [emb_a, emb_b, |emb_a - emb_b|]
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def predict_matchup(self, embeddings, idx_a, idx_b):
        emb_a = embeddings[idx_a]
        emb_b = embeddings[idx_b]
        diff = torch.abs(emb_a - emb_b)
        x = torch.cat([emb_a, emb_b, diff], dim=-1)
        return self.head(x).squeeze(-1)


def train_and_evaluate(data: dict, train_seasons: list, test_season: int,
                        n_epochs: int = 50, lr: float = 1e-3) -> tuple:
    """Train GNN on multiple seasons, predict test_season tournament."""
    # Build graphs for all training seasons
    train_graphs = {}
    train_tourney_pairs = []
    train_labels = []
    for s in train_seasons:
        g, t2i = build_graph_for_season(data, s)
        if g is None:
            continue
        train_graphs[s] = (g, t2i)
        # Tournament games for this season
        tourney = data["tourney_compact"]
        sg = tourney[tourney["Season"] == s]
        for _, row in sg.iterrows():
            w, l = row["WTeamID"], row["LTeamID"]
            if w in t2i and l in t2i:
                if w < l:
                    train_tourney_pairs.append((s, t2i[w], t2i[l], 1.0))
                else:
                    train_tourney_pairs.append((s, t2i[l], t2i[w], 0.0))

    # Build test graph
    test_graph, test_t2i = build_graph_for_season(data, test_season)
    test_tourney = data["tourney_compact"]
    test_games = test_tourney[test_tourney["Season"] == test_season]
    test_pairs = []
    test_y = []
    for _, row in test_games.iterrows():
        w, l = row["WTeamID"], row["LTeamID"]
        if w in test_t2i and l in test_t2i:
            if w < l:
                test_pairs.append((test_t2i[w], test_t2i[l]))
                test_y.append(1.0)
            else:
                test_pairs.append((test_t2i[l], test_t2i[w]))
                test_y.append(0.0)
    test_y = np.array(test_y)
    test_pairs = np.array(test_pairs)

    if len(test_pairs) == 0:
        return None, None

    # Train
    model = TeamGNN(in_dim=6, hidden=32, n_layers=3, dropout=0.15)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        # Shuffle training pairs
        rng = np.random.default_rng(epoch)
        shuffled = rng.permutation(len(train_tourney_pairs))
        batch_size = 64

        for i in range(0, len(shuffled), batch_size):
            batch_idx = shuffled[i:i+batch_size]
            opt.zero_grad()
            batch_loss = 0
            # Group by season for graph encoding
            by_season = {}
            for j in batch_idx:
                s, ia, ib, y = train_tourney_pairs[j]
                by_season.setdefault(s, []).append((ia, ib, y))

            for s, items in by_season.items():
                graph, _ = train_graphs[s]
                emb = model.encode(graph.x, graph.edge_index)
                idx_a = torch.LongTensor([it[0] for it in items])
                idx_b = torch.LongTensor([it[1] for it in items])
                ys = torch.FloatTensor([it[2] for it in items])
                logits = model.predict_matchup(emb, idx_a, idx_b)
                batch_loss = batch_loss + criterion(logits, ys)

            batch_loss.backward()
            opt.step()
            total_loss += batch_loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/max(n_batches,1):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        emb = model.encode(test_graph.x, test_graph.edge_index)
        idx_a = torch.LongTensor(test_pairs[:, 0])
        idx_b = torch.LongTensor(test_pairs[:, 1])
        logits = model.predict_matchup(emb, idx_a, idx_b)
        probs = torch.sigmoid(logits).numpy()

    return probs, test_y


def main():
    print("Loading data...")
    data = load_all_mens_data()
    seasons = [s for s in range(2014, 2026) if s != 2020]

    print(f"\nGNN LOTO over {len(seasons)} seasons")
    print("Architecture: GraphSAGE 3 layers, hidden=32, dropout=0.15\n")

    results = []
    for holdout in seasons:
        train_seasons = [s for s in seasons if s != holdout]
        print(f"=== Season {holdout} ===")
        probs, y_true = train_and_evaluate(data, train_seasons, holdout, n_epochs=40)
        if probs is None:
            continue
        bs = brier_score_loss(y_true, probs)
        print(f"  Brier: {bs:.4f} ({len(y_true)} games)")
        results.append({"season": holdout, "brier": bs, "n_games": len(y_true)})

    df = pd.DataFrame(results)
    print(f"\n{'='*60}\n  GNN LOTO SUMMARY\n{'='*60}")
    print(df.to_string(index=False))
    print(f"\nMean Brier: {df['brier'].mean():.4f} ± {df['brier'].std():.4f}")
    df.to_csv("output/loto_gnn.csv", index=False)


if __name__ == "__main__":
    main()
