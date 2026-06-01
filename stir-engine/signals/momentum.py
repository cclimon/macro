"""
signals/momentum.py
--------------------
Momentum signal for SOFR/FF futures.

Two sub-components are combined into a single normalised signal:

  1. Price momentum  — normalised rolling return (rate-of-change scaled by
                       realised vol). Captures the "market is repricing in
                       one direction and will continue" dynamic around Fed
                       communication windows and data releases.

  2. COT positioning — CFTC Commitments of Traders net positioning for
                       leveraged funds and asset managers in SOFR futures.
                       Captures crowding risk and institutional flow. A
                       market where leveraged funds are historically long is
                       vulnerable to unwind; extreme positioning is a
                       contrary signal at the margin.

The two sub-signals are Z-scored independently then blended via
configurable weights into a single `signal_mom` column: {-1, 0, +1}.

Dependencies:
    pip install numpy pandas requests

COT data source:
    CFTC publishes weekly disaggregated COT reports as CSV at:
    https://www.cftc.gov/dea/newcot/c_disagg.txt
    No API key required.

Usage:
    from signals.momentum import MomentumSignal, MomConfig

    mom = MomentumSignal(futures_df, cot_df)   # cot_df optional
    results = mom.compute_signal()
"""

import io
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MomConfig:
    """
    Tunable parameters for the momentum signal.
    """
    # Price momentum lookback windows (business days)
    fast_window:       int   = 5      # ~1 week:  captures post-FOMC repricing
    slow_window:       int   = 21     # ~1 month: captures medium-term trend

    # Volatility normalisation window for momentum scaling
    vol_window:        int   = 20

    # Z-score window for normalising momentum and COT sub-signals
    zscore_window:     int   = 63     # ~1 quarter rolling normalisation

    # Entry/exit thresholds (applied to the blended Z-score)
    entry_z:           float = 1.2    # slightly lower than MR — momentum fades faster
    exit_z:            float = 0.4

    # COT: weight of positioning sub-signal in final blend (0 = price only)
    cot_weight:        float = 0.30   # 30% COT, 70% price momentum

    # COT: CFTC market code for SOFR 3-month futures
    # Run fetch_cot_data() once and inspect df["market_name"] to confirm code.
    cot_market_code:   str   = "SOFR - CHICAGO MERCANTILE EXCHANGE"

    # COT: which trader category to use as the primary positioning signal
    # Options: "leveraged_funds", "asset_manager", "dealer"
    cot_category:      str   = "leveraged_funds"

    # Minimum observations required before emitting a signal
    min_obs:           int   = 40


# ---------------------------------------------------------------------------
# COT data fetch
# ---------------------------------------------------------------------------

CFTC_DISAGG_URL = (
    "https://www.cftc.gov/dea/newcot/c_disagg.txt"
)

# Column name map from raw CFTC CSV → clean names
_COT_COLS = {
    "Market_and_Exchange_Names":              "market_name",
    "Report_Date_as_MM_DD_YYYY":              "date",
    "CFTC_Contract_Market_Code":              "market_code",
    # Leveraged funds
    "Lev_Money_Positions_Long_All":           "lev_long",
    "Lev_Money_Positions_Short_All":          "lev_short",
    "Lev_Money_Positions_Spread_All":         "lev_spread",
    # Asset managers
    "Asset_Mgr_Positions_Long_All":           "am_long",
    "Asset_Mgr_Positions_Short_All":          "am_short",
    # Dealers
    "Dealer_Positions_Long_All":              "dealer_long",
    "Dealer_Positions_Short_All":             "dealer_short",
    # Open interest
    "Open_Interest_All":                      "open_interest",
}


def fetch_cot_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Download the CFTC disaggregated COT report and return a clean DataFrame.

    The report is published every Friday (for positions as of the prior Tuesday).
    Data covers all futures markets; we filter to SOFR in MomentumSignal.

    Parameters
    ----------
    use_cache : if True, load from local parquet if available

    Returns
    -------
    pd.DataFrame with columns: date, market_name, market_code,
        lev_long, lev_short, lev_spread,
        am_long, am_short,
        dealer_long, dealer_short,
        open_interest
    """
    try:
        from config import DATA_DIR
        cache_path = DATA_DIR / "cot_disagg.parquet"
    except ImportError:
        cache_path = None

    if use_cache and cache_path and cache_path.exists():
        logger.info("Loading COT data from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Fetching CFTC disaggregated COT report from %s", CFTC_DISAGG_URL)

    try:
        import requests
        resp = requests.get(CFTC_DISAGG_URL, timeout=30)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch COT data from CFTC: {exc}\n"
            "Check your internet connection. The CFTC URL occasionally changes "
            "— verify at https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"
        )

    # Keep only columns we care about
    available = [c for c in _COT_COLS if c in raw.columns]
    df = raw[available].rename(columns=_COT_COLS)

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Coerce numeric columns
    numeric_cols = [c for c in df.columns if c not in ("market_name", "date", "market_code")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if cache_path:
        df.to_parquet(cache_path)
        logger.info("COT data cached → %s  (%d rows)", cache_path, len(df))

    return df


def extract_sofr_cot(
    cot_df: pd.DataFrame,
    market_code: str,
    category: str = "leveraged_funds",
) -> pd.Series:
    """
    Filter COT DataFrame to the SOFR futures contract and return a
    weekly net positioning series for the specified trader category.

    Net position = longs - shorts, normalised by open interest.
    This gives a percentage (−1 to +1) comparable across time even as
    contract size and participation grows.

    Parameters
    ----------
    cot_df      : DataFrame from fetch_cot_data()
    market_code : CFTC market name string for SOFR
    category    : "leveraged_funds", "asset_manager", or "dealer"

    Returns
    -------
    pd.Series of net positioning (% of OI), weekly, DatetimeIndex
    """
    # Filter by market name (partial match — CFTC names can vary)
    mask = cot_df["market_name"].str.upper().str.contains(
        market_code.upper().split(" - ")[0],   # e.g. "SOFR"
        na=False,
    )
    sofr = cot_df[mask].copy()

    if sofr.empty:
        available = cot_df["market_name"].unique()[:10]
        raise ValueError(
            f"No COT rows matched market_code='{market_code}'.\n"
            f"Available market names (first 10): {list(available)}\n"
            "Adjust MomConfig.cot_market_code to match."
        )

    # Select long/short columns for the requested category
    col_map = {
        "leveraged_funds": ("lev_long",    "lev_short"),
        "asset_manager":   ("am_long",     "am_short"),
        "dealer":          ("dealer_long", "dealer_short"),
    }
    if category not in col_map:
        raise ValueError(f"Unknown COT category: {category!r}. Choose from {list(col_map)}")

    long_col, short_col = col_map[category]

    sofr = sofr.set_index("date").sort_index()
    net  = sofr[long_col] - sofr[short_col]

    # Normalise by open interest to get a comparable % series
    net_pct = net / sofr["open_interest"].replace(0, np.nan)
    net_pct.name = f"cot_net_{category}"

    logger.info(
        "COT — %s net positioning: %d weekly obs  range [%.3f, %.3f]",
        category, len(net_pct), net_pct.min(), net_pct.max(),
    )

    return net_pct


# ---------------------------------------------------------------------------
# Price momentum helpers
# ---------------------------------------------------------------------------

def compute_price_momentum(
    futures_df: pd.DataFrame,
    fast_window: int = 5,
    slow_window: int = 21,
    vol_window:  int = 20,
) -> pd.DataFrame:
    """
    Compute vol-normalised price momentum at two horizons.

    Momentum is defined on the *implied rate* rather than the futures price
    so that a positive value always means "rates are moving higher" —
    which maps intuitively onto the macro narrative.

    fast_mom  = mean(rate_change, fast_window)  / realised_vol
    slow_mom  = mean(rate_change, slow_window)  / realised_vol

    Both are then sign-flipped so that positive momentum → bullish for rates
    (i.e. rates falling → long futures → signal +1).

    Parameters
    ----------
    futures_df  : DataFrame with implied_rate and log_return columns
    fast_window : short horizon (days)
    slow_window : longer horizon (days)
    vol_window  : window for realised vol normalisation

    Returns
    -------
    pd.DataFrame with columns: fast_mom, slow_mom, blended_mom
    """
    rate = futures_df["implied_rate"]

    # Daily rate change (in basis points for readability)
    rate_chg = rate.diff() * 100      # bps

    # Realised vol of rate changes (annualised bps)
    rv = rate_chg.rolling(vol_window).std() * np.sqrt(252)
    rv = rv.replace(0, np.nan)

    # Vol-normalised momentum (negative = rates rising = bearish)
    fast_mom = rate_chg.rolling(fast_window).mean() / rv
    slow_mom = rate_chg.rolling(slow_window).mean() / rv

    # Sign-flip: falling rates (negative rate_chg) = positive futures signal
    fast_mom = -fast_mom
    slow_mom = -slow_mom

    fast_mom.name = "fast_mom"
    slow_mom.name = "slow_mom"

    # Blended: equal-weight fast and slow
    blended = (fast_mom + slow_mom) / 2
    blended.name = "blended_mom"

    return pd.DataFrame({"fast_mom": fast_mom, "slow_mom": slow_mom, "blended_mom": blended})


# ---------------------------------------------------------------------------
# Z-score helper (shared with MR module pattern)
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu  = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    z   = (series - mu) / std.replace(0, np.nan)
    z.name = series.name + "_z"
    return z


# ---------------------------------------------------------------------------
# Signal generation (with hysteresis — same pattern as MR)
# ---------------------------------------------------------------------------

def _zscore_to_signal(
    zscore: pd.Series,
    entry_z: float,
    exit_z:  float,
    name:    str = "signal_mom",
) -> pd.Series:
    signal   = pd.Series(0, index=zscore.index, dtype=int, name=name)
    position = 0
    for i, z in enumerate(zscore.astype(float)):
        if pd.isna(z):
            signal.iloc[i] = 0
            continue
        if position == 0:
            if z > entry_z:
                position = 1
            elif z < -entry_z:
                position = -1
        else:
            if abs(z) < exit_z:
                position = 0
        signal.iloc[i] = position
    return signal


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MomentumSignal:
    """
    Full momentum signal pipeline for SOFR/FF futures.

    Parameters
    ----------
    futures_df : DataFrame from data.cme.fetch_continuous_front()
                 Must have columns: implied_rate, log_return
    cot_df     : DataFrame from fetch_cot_data() — optional.
                 If None, signal is price momentum only (cot_weight ignored).
    config     : MomConfig instance (optional, uses defaults if not provided)
    """

    def __init__(
        self,
        futures_df: pd.DataFrame,
        cot_df:     Optional[pd.DataFrame] = None,
        config:     MomConfig = None,
    ):
        self.futures = futures_df.copy()
        self.cot_raw = cot_df
        self.config  = config or MomConfig()

    # ------------------------------------------------------------------
    # Internal: build aligned COT series on daily spine
    # ------------------------------------------------------------------

    def _build_cot_daily(self) -> Optional[pd.Series]:
        """
        Extract SOFR COT positioning and forward-fill weekly values onto
        the daily futures spine. Returns None if COT data is unavailable.
        """
        if self.cot_raw is None:
            logger.info("No COT data provided — using price momentum only.")
            return None

        try:
            net = extract_sofr_cot(
                self.cot_raw,
                market_code=self.config.cot_market_code,
                category=self.config.cot_category,
            )
        except (ValueError, KeyError) as exc:
            logger.warning("COT extraction failed: %s. Proceeding without COT.", exc)
            return None

        # Forward-fill weekly COT onto daily futures index
        daily = net.reindex(self.futures.index, method="ffill")
        return daily

    # ------------------------------------------------------------------
    # Compute signal
    # ------------------------------------------------------------------

    def compute_signal(self) -> pd.DataFrame:
        """
        Run the full momentum pipeline and return a results DataFrame.

        Returns
        -------
        pd.DataFrame with columns:
            close, implied_rate,
            fast_mom, slow_mom, blended_mom,   (vol-normalised sub-signals)
            cot_net,                            (COT net positioning % OI, if available)
            zscore_mom,                         (blended Z-score fed to signal)
            signal_mom                          ({-1, 0, +1})
        """
        cfg = self.config

        if len(self.futures) < cfg.min_obs:
            raise ValueError(
                f"Insufficient data: {len(self.futures)} rows < min_obs={cfg.min_obs}"
            )

        # --- 1. Price momentum ---
        mom_df = compute_price_momentum(
            self.futures,
            fast_window=cfg.fast_window,
            slow_window=cfg.slow_window,
            vol_window=cfg.vol_window,
        )

        # Z-score the blended price momentum
        price_z = _rolling_zscore(mom_df["blended_mom"], cfg.zscore_window)

        # --- 2. COT positioning ---
        cot_daily = self._build_cot_daily()
        cot_weight = cfg.cot_weight if cot_daily is not None else 0.0

        if cot_daily is not None:
            # Z-score COT net positioning over same window
            # Note: COT is a *contrary* signal at extremes for mean reversion,
            # but here we use it as a *confirming* signal for momentum direction.
            # Heavily long leveraged funds = momentum is already extended = lower
            # weight near extremes. We handle this via the Z-score itself:
            # extreme positive COT Z → overstretched longs → slight fade.
            cot_z = _rolling_zscore(cot_daily, cfg.zscore_window)
        else:
            cot_z      = pd.Series(0.0, index=self.futures.index)
            cot_weight = 0.0

        # --- 3. Blend price momentum and COT ---
        price_weight = 1.0 - cot_weight
        blended_z    = (price_z * price_weight + cot_z * cot_weight)
        blended_z.name = "zscore_mom"

        # --- 4. Signal generation ---
        signal = _zscore_to_signal(
            blended_z,
            entry_z=cfg.entry_z,
            exit_z=cfg.exit_z,
        )

        # --- 5. Assemble results ---
        result = self.futures[["close", "implied_rate"]].copy()
        result["fast_mom"]   = mom_df["fast_mom"]
        result["slow_mom"]   = mom_df["slow_mom"]
        result["blended_mom"] = mom_df["blended_mom"]

        if cot_daily is not None:
            result["cot_net"] = cot_daily
        else:
            result["cot_net"] = np.nan

        result["zscore_mom"] = blended_z
        result["signal_mom"] = signal

        long_  = (signal == 1).sum()
        short_ = (signal == -1).sum()
        flat_  = (signal == 0).sum()

        logger.info(
            "Momentum signal: %d long  %d short  %d flat  "
            "(price_weight=%.0f%%  cot_weight=%.0f%%)",
            long_, short_, flat_,
            price_weight * 100, cot_weight * 100,
        )

        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self, results: Optional[pd.DataFrame] = None) -> None:
        """Print a concise summary of momentum signal parameters and stats."""
        cfg = self.config
        print("\n" + "=" * 52)
        print("  Momentum Signal — Summary")
        print("=" * 52)
        print(f"  Fast window         : {cfg.fast_window} days")
        print(f"  Slow window         : {cfg.slow_window} days")
        print(f"  Vol window          : {cfg.vol_window} days")
        print(f"  Z-score window      : {cfg.zscore_window} days")
        print(f"  Entry Z threshold   : ±{cfg.entry_z}")
        print(f"  Exit  Z threshold   : ±{cfg.exit_z}")
        print(f"  COT category        : {cfg.cot_category}")
        print(f"  COT weight          : {cfg.cot_weight:.0%}")
        print(f"  Price mom weight    : {1 - cfg.cot_weight:.0%}")
        if results is not None and "signal_mom" in results.columns:
            s = results["signal_mom"]
            print(f"  Long days           : {(s == 1).sum()}")
            print(f"  Short days          : {(s == -1).sum()}")
            print(f"  Flat days           : {(s == 0).sum()}")
        print("=" * 52 + "\n")