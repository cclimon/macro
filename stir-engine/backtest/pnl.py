"""
backtest/pnl.py
---------------
P&L engine for the STIR engine ensemble signal.

Converts the {-1, 0, +1} signal series into a realistic daily P&L
stream, accounting for:

  - DV01-based position sizing (rate-equivalent notional)
  - Transaction costs (bid-ask spread + exchange fees)
  - Slippage (entry at next-day open, not signal-day close)
  - Margin / capital allocation for Sharpe normalisation

Performance metrics produced:
  - Cumulative P&L (USD and bps)
  - Daily / annualised Sharpe ratio
  - Maximum drawdown and drawdown duration
  - Win rate, profit factor
  - Per-signal-component attribution (MR, momentum, OSINT)
  - Per-regime attribution (ranging, trending, crisis)

SOFR futures contract specs (CME SR3)
--------------------------------------
  Contract size  : $2,500 per basis point (DV01 = $25 per 0.01 price move)
  Tick size      : 0.005 price points = $12.50
  Margin (approx): ~$500-800 per contract (varies; use $700 default)
  Exchange fee   : ~$0.85/side (CME non-member)
  Typical spread : 0.5–1 tick in liquid hours

Dependencies
------------
    pip install numpy pandas

Usage
-----
    from backtest.pnl import PnLEngine, PnLConfig

    engine = PnLEngine(ensemble_result, config=PnLConfig())
    results = engine.compute()
    engine.summary(results)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contract specification
# ---------------------------------------------------------------------------

@dataclass
class ContractSpec:
    """
    CME SOFR 3-month futures contract parameters.
    Defaults reflect SR3 (3-month SOFR) specifications.
    """
    # Dollar value per 1 basis point move in the futures price
    # SR3: notional $1M × 90/360 = $250,000 × 0.01% = $25 per 0.01 tick
    # but full DV01 (1bp = 0.01 in price) = $25 × 10 = $2,500 per contract
    dv01_per_contract:     float = 2_500.0   # USD per 1bp (0.01 price move × 100)

    # Tick size in price points
    tick_size:             float = 0.005

    # Dollar value per tick
    tick_value:            float = 12.50

    # Exchange fee per side (USD) — CME non-member approximate
    exchange_fee:          float = 0.85

    # Clearing fee per side (USD)
    clearing_fee:          float = 0.35

    # Typical bid-ask spread in ticks (0.5 = half-tick, common in liquid hours)
    typical_spread_ticks:  float = 1.0

    # Initial margin per contract (USD) — approximate, varies with vol
    initial_margin:        float = 700.0

    @property
    def total_fee_per_side(self) -> float:
        return self.exchange_fee + self.clearing_fee

    @property
    def spread_cost_usd(self) -> float:
        """One-way cost of crossing the bid-ask spread."""
        return self.typical_spread_ticks * self.tick_value * 0.5


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PnLConfig:
    """
    Backtest configuration parameters.
    """
    # Position sizing method:
    #   "fixed"    : always 1 contract per signal
    #   "vol_target": size to a target annualised vol of the position
    sizing_method:         str   = "vol_target"

    # Target annualised P&L volatility in USD (vol_target sizing)
    # $5,000 ≈ modest single-contract equivalent at typical SOFR vol
    vol_target_usd:        float = 5_000.0

    # Volatility estimation window for vol_target sizing (days)
    vol_window:            int   = 20

    # Maximum contracts per position (safety cap)
    max_contracts:         int   = 10

    # Slippage model:
    #   "next_open"  : entry/exit at next day's open price (realistic)
    #   "same_close" : entry/exit at signal-day close (optimistic)
    slippage_model:        str   = "next_open"

    # Apply transaction costs (spread + fees) on each trade
    apply_costs:           bool  = True

    # Risk-free rate for Sharpe calculation (annualised, decimal)
    risk_free_rate:        float = 0.045    # approx 4.5% — update to current

    # Minimum trading days in a subperiod for walk-forward inclusion
    min_oos_days:          int   = 21

    # Contract specification
    contract:              ContractSpec = None

    def __post_init__(self):
        if self.contract is None:
            self.contract = ContractSpec()


# ---------------------------------------------------------------------------
# Core P&L computation
# ---------------------------------------------------------------------------

def compute_daily_pnl(
    ensemble_df:  pd.DataFrame,
    config:       PnLConfig,
) -> pd.DataFrame:
    """
    Compute daily P&L from the ensemble signal.

    The signal is interpreted as:
        +1 → long futures (expecting price to rise / rates to fall)
        -1 → short futures (expecting price to fall / rates to rise)
         0 → flat

    Entry/exit mechanics:
        - If slippage_model="next_open": a signal change on day t is
          executed at day t+1's open price. This is the realistic assumption
          and avoids look-ahead bias.
        - Transaction costs (spread + fees round-trip) applied on each
          position change.

    Parameters
    ----------
    ensemble_df : DataFrame from Ensemble.compute() with columns:
                  close, signal_ensemble, realised_vol (optional),
                  regime (optional), w_mr, w_mom, w_osint (optional)
    config      : PnLConfig instance

    Returns
    -------
    pd.DataFrame with all ensemble columns plus:
        position        : actual held position after slippage shift
        contracts       : number of contracts (vol_target sizing)
        price_change    : daily change in futures price (points)
        gross_pnl       : P&L before costs (USD)
        costs           : transaction costs (USD, always ≤ 0)
        net_pnl         : gross_pnl + costs
        cum_net_pnl     : cumulative net P&L (USD)
        daily_return_pct: net_pnl / (contracts × margin) — for Sharpe
        trade_flag      : bool — True on days a trade was executed
    """
    df  = ensemble_df.copy()
    cfg = config
    cs  = cfg.contract

    # --- Position (with optional 1-day slippage shift) ---
    signal = df["signal_ensemble"].fillna(0).astype(int)
    if cfg.slippage_model == "next_open":
        # Hold position entered on prior day's signal
        position = signal.shift(1).fillna(0).astype(int)
    else:
        position = signal.astype(int)

    # --- Contract sizing ---
    if cfg.sizing_method == "vol_target":
        # Estimate daily P&L vol from recent price returns
        if "log_return" in df.columns:
            ret = df["log_return"]
        else:
            ret = df["close"].pct_change()

        daily_vol_price = ret.rolling(cfg.vol_window, min_periods=5).std()
        # Convert to daily P&L vol per contract (USD)
        daily_vol_usd = daily_vol_price * df["close"] * cs.dv01_per_contract / df["close"]
        # Actually: 1bp move = $25 for SR3, so P&L vol per contract (USD) per day:
        # vol in price points × (dv01_per_contract / 0.01)
        # simplify: vol_usd = daily_price_std × (cs.dv01_per_contract / 0.01)
        daily_vol_usd = daily_vol_price * (cs.dv01_per_contract / 0.01)
        daily_vol_usd = daily_vol_usd.replace(0, np.nan).ffill().bfill()

        # Annualised target → daily target
        daily_target = cfg.vol_target_usd / np.sqrt(252)

        contracts = (daily_target / daily_vol_usd).clip(0, cfg.max_contracts)
        contracts = contracts.round().astype(int).clip(1, cfg.max_contracts)
    else:
        contracts = pd.Series(1, index=df.index)

    # Zero out contracts when flat
    contracts = contracts.where(position != 0, 0)

    # --- Daily price change ---
    if cfg.slippage_model == "next_open" and "open" in df.columns:
        # Use actual open-to-open P&L for entered positions
        price_change = df["open"].diff()
    else:
        price_change = df["close"].diff()

    price_change = price_change.fillna(0)

    # --- Gross P&L ---
    # position × contracts × price_change × dv01_per_bp
    # 1 price point = 100bp, so price_change in points × 100 = bps
    # gross_pnl = position × contracts × price_change_bps × $25
    price_change_bps = price_change * 100
    gross_pnl = position * contracts * price_change_bps * (cs.dv01_per_contract / 100)

    # --- Transaction costs ---
    trade_flag = (position != position.shift(1).fillna(0))
    if cfg.apply_costs:
        round_trip_cost = (cs.spread_cost_usd + cs.total_fee_per_side * 2)
        costs = pd.Series(
            np.where(trade_flag, -contracts * round_trip_cost, 0.0),
            index=df.index,
        )
    else:
        costs = pd.Series(0.0, index=df.index)

    # --- Net P&L ---
    net_pnl    = gross_pnl + costs
    cum_pnl    = net_pnl.cumsum()

    # --- Percentage return (for Sharpe) ---
    capital    = (contracts * cs.initial_margin).replace(0, cs.initial_margin)
    daily_ret  = net_pnl / capital

    # --- Assemble ---
    df["position"]         = position
    df["contracts"]        = contracts
    df["price_change"]     = price_change
    df["price_change_bps"] = price_change_bps
    df["gross_pnl"]        = gross_pnl.round(2)
    df["costs"]            = costs.round(2)
    df["net_pnl"]          = net_pnl.round(2)
    df["cum_net_pnl"]      = cum_pnl.round(2)
    df["daily_return"]     = daily_ret
    df["trade_flag"]       = trade_flag

    logger.info(
        "P&L computed: %d trades  total net P&L = $%.0f  "
        "total costs = $%.0f",
        trade_flag.sum(),
        net_pnl.sum(),
        costs.sum(),
    )

    return df


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_performance_metrics(
    pnl_df:    pd.DataFrame,
    config:    PnLConfig,
    label:     str = "Full period",
) -> dict:
    """
    Compute comprehensive performance metrics from a P&L DataFrame.

    Parameters
    ----------
    pnl_df : DataFrame from compute_daily_pnl()
    config : PnLConfig instance
    label  : string label for this period (e.g. "OOS 2021-2022")

    Returns
    -------
    dict of performance metrics
    """
    net     = pnl_df["net_pnl"].fillna(0)
    ret     = pnl_df["daily_return"].fillna(0)
    cum     = pnl_df["cum_net_pnl"]
    trades  = pnl_df["trade_flag"]
    n_days  = len(pnl_df)

    if n_days < config.min_oos_days:
        logger.warning("Period too short for reliable metrics: %d days", n_days)

    # --- Return metrics ---
    total_pnl       = net.sum()
    mean_daily      = net.mean()
    ann_pnl         = mean_daily * 252

    # --- Risk metrics ---
    std_daily       = ret.std()
    ann_vol         = std_daily * np.sqrt(252)
    rf_daily        = config.risk_free_rate / 252
    excess_ret      = ret - rf_daily
    sharpe          = (excess_ret.mean() / std_daily * np.sqrt(252)
                       if std_daily > 0 else 0.0)

    # Sortino (downside vol only)
    downside        = ret[ret < rf_daily]
    downside_vol    = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino         = (ret.mean() - rf_daily) * 252 / downside_vol

    # --- Drawdown ---
    roll_max        = cum.cummax()
    drawdown        = cum - roll_max
    max_drawdown    = drawdown.min()

    # Drawdown duration (longest consecutive days underwater)
    in_dd           = (drawdown < 0).astype(int)
    dd_runs         = in_dd * (in_dd.groupby(
        (in_dd != in_dd.shift()).cumsum()
    ).cumcount() + 1)
    max_dd_duration = int(dd_runs.max()) if in_dd.any() else 0

    # Calmar ratio
    calmar          = (ann_pnl / abs(max_drawdown)
                       if max_drawdown != 0 else 0.0)

    # --- Trade-level metrics ---
    n_trades        = int(trades.sum())
    trade_pnl       = net[trades]
    win_rate        = float((trade_pnl > 0).mean()) if n_trades > 0 else 0.0
    avg_win         = float(trade_pnl[trade_pnl > 0].mean()) if (trade_pnl > 0).any() else 0.0
    avg_loss        = float(trade_pnl[trade_pnl < 0].mean()) if (trade_pnl < 0).any() else 0.0
    profit_factor   = (
        abs(trade_pnl[trade_pnl > 0].sum() / trade_pnl[trade_pnl < 0].sum())
        if (trade_pnl < 0).any() else np.inf
    )

    metrics = {
        "label":            label,
        "n_days":           n_days,
        "n_trades":         n_trades,
        "total_pnl":        round(total_pnl, 2),
        "ann_pnl":          round(ann_pnl, 2),
        "ann_vol":          round(ann_vol, 4),
        "sharpe":           round(sharpe, 3),
        "sortino":          round(sortino, 3),
        "calmar":           round(calmar, 3),
        "max_drawdown":     round(max_drawdown, 2),
        "max_dd_duration":  max_dd_duration,
        "win_rate":         round(win_rate, 3),
        "avg_win":          round(avg_win, 2),
        "avg_loss":         round(avg_loss, 2),
        "profit_factor":    round(profit_factor, 3),
    }

    return metrics


def attribute_pnl(
    pnl_df: pd.DataFrame,
    config: PnLConfig,
) -> dict:
    """
    Decompose P&L attribution by signal component and by regime.

    For component attribution, we compute what the P&L would have been
    if each signal had been traded in isolation (same sizing, same costs).
    This answers: "how much did MR, momentum, and OSINT each contribute?"

    For regime attribution, we slice the actual P&L by which regime was
    active on each day.

    Parameters
    ----------
    pnl_df : DataFrame from compute_daily_pnl() — must have signal_mr,
             signal_mom, signal_osint, regime columns

    Returns
    -------
    dict with keys: "by_component" and "by_regime"
    """
    attribution = {"by_component": {}, "by_regime": {}}

    price_bps = pnl_df["price_change_bps"].fillna(0)
    cs        = config.contract

    # --- Component attribution ---
    for component, col in [
        ("mean_reversion", "signal_mr"),
        ("momentum",       "signal_mom"),
        ("osint",          "signal_osint"),
        ("ensemble",       "signal_ensemble"),
    ]:
        if col not in pnl_df.columns:
            continue

        sig = pnl_df[col].fillna(0).shift(1).fillna(0)  # apply slippage
        raw_pnl = sig * price_bps * (cs.dv01_per_contract / 100)
        attribution["by_component"][component] = {
            "total_pnl":  round(raw_pnl.sum(), 2),
            "ann_sharpe": round(
                raw_pnl.mean() / raw_pnl.std() * np.sqrt(252)
                if raw_pnl.std() > 0 else 0.0, 3
            ),
            "hit_rate":   round(
                (raw_pnl[sig != 0] > 0).mean()
                if (sig != 0).any() else 0.0, 3
            ),
        }

    # --- Regime attribution ---
    if "regime" in pnl_df.columns:
        for regime in pnl_df["regime"].dropna().unique():
            mask    = pnl_df["regime"] == regime
            sub_pnl = pnl_df.loc[mask, "net_pnl"].fillna(0)
            attribution["by_regime"][str(regime)] = {
                "total_pnl": round(sub_pnl.sum(), 2),
                "n_days":    int(mask.sum()),
                "mean_daily": round(sub_pnl.mean(), 2),
            }

    return attribution


# ---------------------------------------------------------------------------
# Main P&L engine class
# ---------------------------------------------------------------------------

class PnLEngine:
    """
    Full P&L computation and performance measurement engine.

    Parameters
    ----------
    ensemble_df : DataFrame from Ensemble.compute() — must have:
                  close, signal_ensemble
                  Optionally: signal_mr, signal_mom, signal_osint,
                  regime, log_return, open
    config      : PnLConfig instance
    """

    def __init__(
        self,
        ensemble_df: pd.DataFrame,
        config:      PnLConfig = None,
    ):
        self.df     = ensemble_df.copy()
        self.config = config or PnLConfig()

    def compute(self) -> pd.DataFrame:
        """Run P&L engine and return enriched DataFrame."""
        self._results = compute_daily_pnl(self.df, self.config)
        return self._results

    def metrics(self, label: str = "Full period") -> dict:
        """Return performance metrics dict."""
        if not hasattr(self, "_results"):
            raise RuntimeError("Call compute() first.")
        return compute_performance_metrics(self._results, self.config, label)

    def attribution(self) -> dict:
        """Return P&L attribution by component and regime."""
        if not hasattr(self, "_results"):
            raise RuntimeError("Call compute() first.")
        return attribute_pnl(self._results, self.config)

    def summary(self, results: pd.DataFrame = None) -> None:
        """Print a formatted performance summary."""
        if results is None:
            if not hasattr(self, "_results"):
                raise RuntimeError("Call compute() first.")
            results = self._results

        m   = compute_performance_metrics(results, self.config)
        att = attribute_pnl(results, self.config)

        print("\n" + "=" * 52)
        print("  P&L Engine — Performance Summary")
        print("=" * 52)
        print(f"  Period              : {results.index[0].date()} → "
              f"{results.index[-1].date()}")
        print(f"  Trading days        : {m['n_days']}")
        print(f"  Trades executed     : {m['n_trades']}")
        print()
        print(f"  Total net P&L       : ${m['total_pnl']:>10,.0f}")
        print(f"  Annualised P&L      : ${m['ann_pnl']:>10,.0f}")
        print(f"  Annualised vol      : {m['ann_vol']:.2%}")
        print()
        print(f"  Sharpe ratio        : {m['sharpe']:>8.3f}")
        print(f"  Sortino ratio       : {m['sortino']:>8.3f}")
        print(f"  Calmar ratio        : {m['calmar']:>8.3f}")
        print()
        print(f"  Max drawdown        : ${m['max_drawdown']:>10,.0f}")
        print(f"  Max DD duration     : {m['max_dd_duration']} days")
        print()
        print(f"  Win rate            : {m['win_rate']:.1%}")
        print(f"  Avg win             : ${m['avg_win']:>8,.0f}")
        print(f"  Avg loss            : ${m['avg_loss']:>8,.0f}")
        print(f"  Profit factor       : {m['profit_factor']:.3f}")

        if att["by_component"]:
            print()
            print("  Component attribution:")
            for comp, stats in att["by_component"].items():
                print(f"    {comp:<20}: P&L ${stats['total_pnl']:>8,.0f}  "
                      f"Sharpe {stats['ann_sharpe']:>6.3f}  "
                      f"Hit {stats['hit_rate']:.1%}")

        if att["by_regime"]:
            print()
            print("  Regime attribution:")
            for regime, stats in att["by_regime"].items():
                print(f"    {regime:<20}: P&L ${stats['total_pnl']:>8,.0f}  "
                      f"({stats['n_days']} days  "
                      f"avg ${stats['mean_daily']:,.0f}/day)")

        print("=" * 52 + "\n")