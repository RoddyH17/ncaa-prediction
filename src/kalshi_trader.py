"""
Kalshi prediction market trading strategy for NCAA tournament.

Components:
  SyntheticMarketPrices: generate proxy market prices from seed win rates
  TradingStrategy: signal generation, Kelly sizing, risk limits
  TradingBacktester: LOTO-style trading simulation with P&L tracking
  KalshiClient: API wrapper with dry_run mode
  LiveTradingPipeline: end-to-end live trading
"""

import os
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic market prices (for backtesting without real market data)
# ---------------------------------------------------------------------------

class SyntheticMarketPrices:
    """Generate plausible market-implied probabilities from seed matchups.

    Uses historical seed-vs-seed win rates as the base, then adds
    calibrated noise and favorite-longshot bias to simulate a real market.
    """

    def __init__(self, data: dict, noise_std: float = 0.04, flb_strength: float = 0.02,
                 market_alpha_std: float = 0.03):
        self.noise_std = noise_std
        self.flb_strength = flb_strength
        self.market_alpha_std = market_alpha_std  # independent info the market has that we don't
        self.seed_win_rates = self._compute_seed_win_rates(data)

    def _compute_seed_win_rates(self, data: dict) -> dict:
        """Compute historical P(lower seed wins) for each seed matchup."""
        from src.pipeline import _parse_seed_num

        tourney = data["tourney_compact"]
        seeds = data["seeds"].copy()
        seeds["SeedNum"] = seeds["Seed"].apply(_parse_seed_num)

        t = tourney.merge(
            seeds[["Season", "TeamID", "SeedNum"]],
            left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"]
        ).rename(columns={"SeedNum": "WSeed"}).drop(columns="TeamID")
        t = t.merge(
            seeds[["Season", "TeamID", "SeedNum"]],
            left_on=["Season", "LTeamID"], right_on=["Season", "TeamID"]
        ).rename(columns={"SeedNum": "LSeed"}).drop(columns="TeamID")

        rates = {}
        for (ws, ls), group in t.groupby(["WSeed", "LSeed"]):
            key_hi = (min(ws, ls), max(ws, ls))  # (better seed, worse seed)
            if key_hi not in rates:
                rates[key_hi] = {"wins": 0, "total": 0}
            rates[key_hi]["total"] += len(group)
            if ws == key_hi[0]:  # better seed won
                rates[key_hi]["wins"] += len(group)

        result = {}
        for (s1, s2), counts in rates.items():
            result[(s1, s2)] = counts["wins"] / max(counts["total"], 1)
        return result

    def _favorite_longshot_bias(self, p: float) -> float:
        """Markets slightly overprice longshots and underprice favorites."""
        # Push probabilities slightly toward 0.5
        return p - self.flb_strength * (p - 0.5)

    def generate(self, seed_a: int, seed_b: int, p_model: float = None,
                 rng: np.random.Generator = None) -> float:
        """Generate a synthetic market price for P(TeamA wins).

        If p_model is given, blends model prediction with seed prior to simulate
        an efficient market that has access to similar (but not identical) info.
        Real Kalshi markets are mostly populated by quant firms using ratings
        similar to ours, so the market price tracks the model closely.
        """
        if rng is None:
            rng = np.random.default_rng()

        better = min(seed_a, seed_b)
        worse = max(seed_a, seed_b)
        seed_p = self.seed_win_rates.get((better, worse), 0.5)
        if seed_a > seed_b:
            seed_p = 1 - seed_p

        if p_model is not None:
            # Market is ~70% model-like, ~30% seed-anchored
            base_p = 0.7 * p_model + 0.3 * seed_p
        else:
            base_p = seed_p

        # Market has its own independent information we can't see
        p = base_p + rng.normal(0, self.market_alpha_std)
        # Plus noise and favorite-longshot bias
        p = self._favorite_longshot_bias(p)
        p += rng.normal(0, self.noise_std)
        return float(np.clip(p, 0.02, 0.98))

    def generate_for_matchups(self, matchups_df: pd.DataFrame,
                              p_model: np.ndarray = None, seed: int = 42) -> np.ndarray:
        """Generate synthetic prices for a DataFrame of matchups."""
        rng = np.random.default_rng(seed)
        prices = []
        for i, (_, row) in enumerate(matchups_df.iterrows()):
            pm = p_model[i] if p_model is not None else None
            prices.append(self.generate(int(row["SeedA"]), int(row["SeedB"]), pm, rng))
        return np.array(prices)


# ---------------------------------------------------------------------------
# Trading strategy
# ---------------------------------------------------------------------------

class TradingStrategy:
    """Signal generation + Kelly sizing + risk management.

    Trades when |p_model - p_market| > threshold.
    Sizes via quarter-Kelly. Caps per-game and total exposure.
    """

    def __init__(self, threshold: float = 0.05, kelly_fraction: float = 0.25,
                 max_per_game: float = 0.05, max_total_exposure: float = 0.20,
                 max_contracts: int = 100, spread: float = 0.03):
        self.threshold = threshold
        self.kelly_fraction = kelly_fraction
        self.max_per_game = max_per_game
        self.max_total_exposure = max_total_exposure
        self.max_contracts = max_contracts
        self.spread = spread  # half-spread cost per contract

    def compute_edge(self, p_model: np.ndarray, p_market: np.ndarray) -> np.ndarray:
        return p_model - p_market

    def kelly_size(self, p_model: float, p_market: float) -> float:
        """Fractional Kelly for a binary contract priced at p_market.

        If p_model > p_market: buy Yes. Risk = p_market, profit = 1 - p_market.
        If p_model < p_market: buy No.  Risk = 1 - p_market, profit = p_market.
        """
        if p_model > p_market:
            # Buy Yes: odds = (1 - p_market) / p_market
            edge = p_model - p_market
            odds = (1 - p_market) / max(p_market, 0.01)
            f_star = edge / (1 - p_market) if (1 - p_market) > 0.01 else 0
        else:
            # Buy No: odds = p_market / (1 - p_market)
            edge = p_market - p_model
            f_star = edge / p_market if p_market > 0.01 else 0

        return max(0, self.kelly_fraction * f_star)

    def generate_signals(self, p_model: np.ndarray, p_market: np.ndarray,
                         game_ids: list, bankroll: float) -> pd.DataFrame:
        """Generate trading signals with position sizing and risk limits."""
        edges = self.compute_edge(p_model, p_market)
        signals = []

        for i in range(len(p_model)):
            if abs(edges[i]) < self.threshold:
                continue

            kelly = self.kelly_size(p_model[i], p_market[i])
            side = "yes" if edges[i] > 0 else "no"
            # Buy at the ask (mid + half-spread)
            mid = p_market[i] if side == "yes" else 1 - p_market[i]
            price = mid + self.spread / 2

            raw_size = kelly * bankroll
            capped_size = min(raw_size, self.max_per_game * bankroll)

            if capped_size < 1.0:  # minimum $1 trade
                continue

            # Each contract costs `price` dollars (price is 0-1, contract pays $1)
            # Cap by max_contracts to simulate real market liquidity
            contracts = int(capped_size / max(price, 0.01))
            contracts = min(contracts, self.max_contracts)
            if contracts < 1:
                continue
            actual_cost = contracts * price

            signals.append({
                "game_id": game_ids[i],
                "side": side,
                "edge": edges[i],
                "p_model": p_model[i],
                "p_market": p_market[i],
                "kelly_frac": kelly,
                "position_size": actual_cost,
                "contracts": contracts,
                "price": price,
            })

        if not signals:
            return pd.DataFrame()

        df = pd.DataFrame(signals).sort_values("edge", key=abs, ascending=False)

        # Enforce total exposure cap
        total_cap = self.max_total_exposure * bankroll
        cumulative = 0
        keep = []
        for _, row in df.iterrows():
            if cumulative + row["position_size"] > total_cap:
                # Scale down last position to fit
                remaining = total_cap - cumulative
                if remaining > 1.0:
                    row = row.copy()
                    row["position_size"] = remaining
                    row["contracts"] = max(1, int(remaining / max(row["price"], 0.01)))
                    keep.append(row)
                break
            keep.append(row)
            cumulative += row["position_size"]

        return pd.DataFrame(keep) if keep else pd.DataFrame()


# ---------------------------------------------------------------------------
# Trading backtester
# ---------------------------------------------------------------------------

class TradingBacktester:
    """Simulate trading P&L over historical tournaments using LOTO predictions."""

    def __init__(self, strategy: TradingStrategy, market_gen: SyntheticMarketPrices,
                 initial_bankroll: float = 10000.0):
        self.strategy = strategy
        self.market_gen = market_gen
        self.initial_bankroll = initial_bankroll

    def run_backtest(self, data: dict, build_features_fn, model_factory,
                     seasons: list[int]) -> pd.DataFrame:
        """Full LOTO trading backtest."""
        from src.pipeline import build_tourney_matchups

        results = []

        for holdout in seasons:
            bankroll = self.initial_bankroll  # reset each season
            train_seasons = [s for s in seasons if s != holdout]
            X_train, y_train = build_features_fn(train_seasons)
            X_test, y_test = build_features_fn([holdout])

            if len(X_test) == 0:
                continue

            # Train model and predict
            model = model_factory()
            model.fit(X_train, y_train)
            p_model = model.predict_proba(X_test)[:, 1]

            # Generate synthetic market prices (blended with model predictions)
            matchups = build_tourney_matchups(data, holdout)
            p_market = self.market_gen.generate_for_matchups(
                matchups, p_model=p_model, seed=holdout
            )

            # Generate signals
            game_ids = list(range(len(p_model)))
            signals = self.strategy.generate_signals(p_model, p_market, game_ids, bankroll)

            if signals.empty:
                results.append({
                    "season": holdout, "n_trades": 0, "gross_pnl": 0,
                    "roi": 0, "bankroll": bankroll, "win_rate": 0,
                })
                continue

            # Resolve trades against actual outcomes
            # Each contract: pay `price` dollars, receive $1 if correct, $0 if wrong
            pnl = 0
            wins = 0
            for _, sig in signals.iterrows():
                idx = int(sig["game_id"])
                outcome = y_test[idx]
                n = sig["contracts"]
                cost = sig["position_size"]  # = contracts * price

                if sig["side"] == "yes":
                    correct = (outcome == 1)
                else:
                    correct = (outcome == 0)

                if correct:
                    pnl += n - cost  # receive $n, paid $cost
                    wins += 1
                else:
                    pnl -= cost  # lose the cost

            n = len(signals)
            total_risked = signals["position_size"].sum()
            results.append({
                "season": holdout,
                "n_trades": n,
                "gross_pnl": pnl,
                "roi": pnl / max(total_risked, 1),
                "bankroll": self.initial_bankroll + pnl,
                "win_rate": wins / n if n > 0 else 0,
            })
            logger.info(f"Season {holdout}: {n} trades, P&L=${pnl:.2f}, "
                        f"win rate={wins}/{n}, bankroll=${bankroll:.2f}")

        df = pd.DataFrame(results)

        # Add aggregate risk metrics
        if len(df) > 1 and df["gross_pnl"].std() > 0:
            df.attrs["sharpe"] = df["gross_pnl"].mean() / df["gross_pnl"].std()
            cumulative = df["gross_pnl"].cumsum()
            running_max = cumulative.cummax()
            drawdown = cumulative - running_max
            df.attrs["max_drawdown"] = drawdown.min()
        return df

    def tune_threshold(self, data, build_features_fn, model_factory,
                       seasons, thresholds=None) -> pd.DataFrame:
        """Grid search over signal thresholds."""
        if thresholds is None:
            thresholds = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15]

        results = []
        for tau in thresholds:
            self.strategy.threshold = tau
            bt = self.run_backtest(data, build_features_fn, model_factory, seasons)
            total_pnl = bt["gross_pnl"].sum()
            total_trades = bt["n_trades"].sum()
            avg_win_rate = bt["win_rate"].mean()
            results.append({
                "threshold": tau,
                "total_pnl": total_pnl,
                "total_trades": int(total_trades),
                "avg_win_rate": avg_win_rate,
                "final_bankroll": bt["bankroll"].iloc[-1] if len(bt) > 0 else self.initial_bankroll,
                "roi_pct": total_pnl / self.initial_bankroll * 100,
            })
            print(f"  tau={tau:.2f}: {int(total_trades)} trades, "
                  f"P&L=${total_pnl:+.2f}, ROI={total_pnl/self.initial_bankroll*100:+.1f}%")

        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Kalshi API client
# ---------------------------------------------------------------------------

class KalshiClient:
    """Thin wrapper around Kalshi Trade API v2.

    Set dry_run=True (default) to log orders without executing.
    Credentials from KALSHI_API_KEY / KALSHI_API_SECRET env vars.
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, api_key: str | None = None, api_secret: str | None = None,
                 dry_run: bool = True):
        self.api_key = api_key or os.environ.get("KALSHI_API_KEY")
        self.api_secret = api_secret or os.environ.get("KALSHI_API_SECRET")
        self.dry_run = dry_run
        self._token = None

    def _get_session(self):
        import httpx
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return httpx.Client(base_url=self.BASE_URL, headers=headers, timeout=15)

    def login(self):
        """Authenticate and cache token."""
        if not self.api_key or not self.api_secret:
            logger.warning("No Kalshi credentials. Running in read-only mode.")
            return False
        session = self._get_session()
        resp = session.post("/login", json={
            "email": self.api_key, "password": self.api_secret
        })
        if resp.status_code == 200:
            self._token = resp.json().get("token")
            return True
        logger.error(f"Kalshi login failed: {resp.status_code}")
        return False

    def get_markets(self, series_ticker: str = "NCAAM", limit: int = 200) -> list[dict]:
        session = self._get_session()
        resp = session.get("/markets", params={"series_ticker": series_ticker, "limit": limit})
        if resp.status_code == 200:
            return resp.json().get("markets", [])
        return []

    def get_balance(self) -> float:
        session = self._get_session()
        resp = session.get("/portfolio/balance")
        if resp.status_code == 200:
            return resp.json().get("balance", 0) / 100  # cents to dollars
        return 0.0

    def get_positions(self) -> list[dict]:
        session = self._get_session()
        resp = session.get("/portfolio/positions")
        if resp.status_code == 200:
            return resp.json().get("market_positions", [])
        return []

    def place_order(self, ticker: str, side: str, quantity: int, price: int,
                    order_type: str = "limit") -> dict | None:
        """Place order. price is in cents (1-99). In dry_run, just logs."""
        order = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "count": quantity,
            "type": order_type,
            "yes_price" if side == "yes" else "no_price": price,
        }

        if self.dry_run:
            logger.info(f"[DRY RUN] Order: {side.upper()} {quantity}x {ticker} @ {price}¢")
            return {"status": "dry_run", "order": order}

        session = self._get_session()
        resp = session.post("/portfolio/orders", json=order)
        if resp.status_code == 200:
            return resp.json()
        logger.error(f"Order failed: {resp.status_code} {resp.text}")
        return None


# ---------------------------------------------------------------------------
# Live trading pipeline
# ---------------------------------------------------------------------------

class LiveTradingPipeline:
    """End-to-end: fetch markets → predict → generate signals → execute."""

    def __init__(self, client: KalshiClient, strategy: TradingStrategy,
                 model, data: dict):
        self.client = client
        self.strategy = strategy
        self.model = model
        self.data = data

    def scan_markets(self) -> pd.DataFrame:
        """Fetch active NCAA markets from Kalshi."""
        markets = self.client.get_markets("NCAAM")
        if not markets:
            print("No NCAA markets found.")
            return pd.DataFrame()

        rows = []
        for m in markets:
            rows.append({
                "ticker": m.get("ticker", ""),
                "title": m.get("title", ""),
                "subtitle": m.get("subtitle", ""),
                "yes_bid": m.get("yes_bid", 0) / 100,
                "yes_ask": m.get("yes_ask", 0) / 100,
                "last_price": m.get("last_price", 0) / 100,
                "volume": m.get("volume", 0),
                "close_time": m.get("close_time", ""),
            })
        return pd.DataFrame(rows)

    def execute(self, bankroll: float = None, confirm: bool = True):
        """Full pipeline: scan → predict → signal → trade."""
        markets = self.scan_markets()
        if markets.empty:
            return []

        if bankroll is None:
            bankroll = self.client.get_balance()
            if bankroll <= 0:
                bankroll = 1000.0
                print(f"Could not fetch balance. Using default ${bankroll:.0f}")

        print(f"\nFound {len(markets)} markets, bankroll=${bankroll:.2f}")
        print(markets[["ticker", "title", "yes_bid", "yes_ask", "volume"]].to_string())

        # TODO: match market teams to TeamIDs, build features, run model
        # For now, print a placeholder
        print("\n[Live prediction pipeline not yet wired to team matching]")
        return []
