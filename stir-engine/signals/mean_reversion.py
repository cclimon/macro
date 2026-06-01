"""
signals/mean_reversion.py
--------------------------
Mean reversion signal for SOFR/FF futures based on the
Ornstein-Uhlenbeck (OU) process.

Pipeline:
    1. Compute spread  : implied_rate - fair_value (Fed Funds target midpoint)
    2. Fit OU process  : regress Δspread_t on spread_{t-1} to get reversion speed
    3. Estimate half-life from the reversion speed coefficient
    4. Compute rolling Z-score of the spread over a window scaled to half-life
    5. Generate signal : +1 (long futures / rates too high),
                        -1 (short futures / rates too low),
                         0 (no position, within threshold)

Dependencies:
    pip install numpy pandas statsmodels

Usage:
    from signals.mean_reversion import MeanReversionSignal

    mr = MeanReversionSignal(futures_df, macro_df)
    mr.fit()
    print(mr.half_life)
    results = mr.compute_signal()
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class MRConfig:
    """
    Tunable parameters for the mean reversion signal.
    Defaults are reasonable starting points for SOFR front-month daily data.
    """
    # Z-score entry/exit thresholds
    entry_z:          float = 1.5    # open position when |Z| > entry_z
    exit_z:           float = 0.5   # close position when |Z| < exit_z

    # OU fitting window (business days). None = use full history (expanding).
    fit_window:       Optional[int] = None

    # Z-score rolling window. None = auto-set to 2× estimated half-life.
    zscore_window:    Optional[int] = None

    # Minimum half-life (days) below which signal is suppressed (likely noise)
    min_half_life:    float = 3.0

    # Maximum half-life (days) above which spread is not mean-reverting
    max_half_life:    float = 120.0

    # Fair value definition:
    #   "fed_midpoint" : midpoint of Fed Funds target range (upper + lower) / 2
    #   "fed_upper"    : upper bound of target range
    #   "sofr_fixing"  : actual SOFR daily fixing (tightest anchor)
    fair_value_source: str = "fed_midpoint"


# ---------------------------------------------------------------------------
# Core OU estimation
# ---------------------------------------------------------------------------

def estimate_ou_params(spread: pd.Series) -> dict:
    """
    Fit an Ornstein-Uhlenbeck process to a spread series via OLS.

    Model: Δspread_t = κ × (μ - spread_{t-1}) + ε_t
    Rewritten as OLS:  Δspread_t = α + β × spread_{t-1} + ε_t
    where β = -κ  (must be negative for mean reversion)
          α = κμ  → μ = -α/β

    Parameters
    ----------
    spread : pd.Series of spread values (implied_rate - fair_value)

    Returns
    -------
    dict with keys:
        kappa      : mean reversion speed (annualised)
        mu         : long-run mean of the spread
        sigma      : residual volatility (annualised)
        beta       : raw OLS coefficient on lagged level
        half_life  : days to revert halfway to mean
        r_squared  : fit quality of the OLS regression
        is_mean_reverting : bool — True if β is negative and significant
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels is required. Run: pip install statsmodels")

    spread_clean = spread.dropna()
    if len(spread_clean) < 30:
        raise ValueError(f"Spread series too short for OU estimation ({len(spread_clean)} obs).")

    delta      = spread_clean.diff().dropna()
    lagged     = spread_clean.shift(1).dropna()
    aligned    = pd.concat([delta, lagged], axis=1).dropna()
    aligned.columns = ["delta", "lagged"]

    X = sm.add_constant(aligned["lagged"])
    model = sm.OLS(aligned["delta"], X).fit()

    beta  = model.params["lagged"]      # speed coefficient (expect < 0)
    alpha = model.params["const"]

    # Mean reversion speed κ = -β (per day)
    # Annualise: κ_ann = κ × 252
    kappa_daily = -beta
    kappa_ann   = kappa_daily * 252

    # Long-run mean: μ = -α / β
    mu = -alpha / beta if beta != 0 else 0.0

    # Half-life in days: t½ = ln(2) / κ_daily
    if kappa_daily > 0:
        half_life = np.log(2) / kappa_daily
    else:
        half_life = np.inf  # not mean-reverting

    # Residual vol (annualised)
    sigma_daily = model.resid.std()
    sigma_ann   = sigma_daily * np.sqrt(252)

    is_mr = (beta < 0) and (model.pvalues["lagged"] < 0.10)

    result = {
        "kappa":             round(kappa_ann, 4),
        "mu":                round(mu, 6),
        "sigma":             round(sigma_ann, 4),
        "beta":              round(beta, 6),
        "half_life":         round(half_life, 1),
        "r_squared":         round(model.rsquared, 4),
        "is_mean_reverting": is_mr,
        "n_obs":             len(aligned),
    }

    logger.info(
        "OU fit: κ=%.4f  μ=%.4f  σ=%.4f  half-life=%.1f days  R²=%.4f  mean-reverting=%s",
        result["kappa"], result["mu"], result["sigma"],
        result["half_life"], result["r_squared"], result["is_mean_reverting"],
    )

    return result


def rolling_ou_params(
    spread: pd.Series,
    window: int = 252,
    step: int = 21,
) -> pd.DataFrame:
    """
    Estimate OU parameters over a rolling window.
    Useful for detecting when mean-reversion breaks down (e.g. during
    a Fed hiking cycle the spread may become non-stationary temporarily).

    Parameters
    ----------
    spread : spread series
    window : lookback window in business days
    step   : how often to re-estimate (every N days). Reduces compute cost.

    Returns
    -------
    pd.DataFrame with columns: half_life, kappa, mu, sigma, is_mean_reverting
    indexed to the last date of each window.
    """
    records = []
    dates   = spread.dropna().index

    for i in range(window, len(dates), step):
        window_spread = spread.iloc[i - window:i]
        try:
            params = estimate_ou_params(window_spread)
            params["date"] = dates[i - 1]
            records.append(params)
        except Exception as exc:
            logger.debug("Rolling OU failed at %s: %s", dates[i - 1].date(), exc)

    if not records:
        raise RuntimeError("Rolling OU estimation produced no results.")

    df = pd.DataFrame(records).set_index("date")
    return df


# ---------------------------------------------------------------------------
# Z-score computation
# ---------------------------------------------------------------------------

def compute_zscore(
    spread: pd.Series,
    window: Optional[int] = None,
    half_life: Optional[float] = None,
) -> pd.Series:
    """
    Compute rolling Z-score of a spread series.

    Window selection logic:
        - If window is provided, use it directly.
        - If half_life is provided, window = max(int(2 × half_life), 20).
        - If neither, use a 60-day expanding window minimum.

    Z = (spread - rolling_mean) / rolling_std

    Parameters
    ----------
    spread    : pd.Series of spread values
    window    : explicit rolling window in days
    half_life : estimated half-life from OU fit (used to set window if not explicit)

    Returns
    -------
    pd.Series of Z-scores, same index as spread
    """
    if window is None:
        if half_life is not None:
            window = max(int(2 * half_life), 20)
        else:
            window = 60
        logger.debug("Z-score window set to %d days", window)

    rolling_mean = spread.rolling(window=window, min_periods=window // 2).mean()
    rolling_std  = spread.rolling(window=window, min_periods=window // 2).std()

    zscore = (spread - rolling_mean) / rolling_std
    zscore.name = "zscore_mr"

    return zscore


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def zscore_to_signal(
    zscore: pd.Series,
    entry_z: float = 1.5,
    exit_z:  float = 0.5,
) -> pd.Series:
    """
    Convert a Z-score series to a {-1, 0, +1} signal with hysteresis.

    Entry/exit logic:
        Z > +entry_z  → short futures (rates priced too high vs fair value)
                         signal = -1
        Z < -entry_z  → long futures (rates priced too low)
                         signal = +1
        |Z| < exit_z  → close position
                         signal = 0
        Between thresholds → hold prior position (hysteresis band)

    The hysteresis prevents excessive churn around the entry threshold.

    Parameters
    ----------
    zscore  : pd.Series of Z-scores
    entry_z : threshold to open a position
    exit_z  : threshold to close a position

    Returns
    -------
    pd.Series of integer signals {-1, 0, +1}, same index as zscore
    """
    signal = pd.Series(0, index=zscore.index, dtype=int, name="signal_mr")
    position = 0

    for i, (date, z) in enumerate(zscore.items()):
        if pd.isna(z):
            signal.iloc[i] = 0
            continue

        if position == 0:
            # Not in a trade — check entry
            if z > entry_z:
                position = -1      # short: market pricing too high
            elif z < -entry_z:
                position = 1       # long: market pricing too low
        else:
            # In a trade — check exit
            if abs(z) < exit_z:
                position = 0

        signal.iloc[i] = position

    return signal


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MeanReversionSignal:
    """
    Full mean reversion signal pipeline for SOFR/FF futures.

    Parameters
    ----------
    futures_df : DataFrame from data.cme.fetch_continuous_front()
                 Must have columns: close, implied_rate
    macro_df   : DataFrame from data.fred.fetch_all_macro()
                 Must have columns dependent on config.fair_value_source
    config     : MRConfig instance (optional, uses defaults if not provided)
    """

    def __init__(
        self,
        futures_df: pd.DataFrame,
        macro_df:   pd.DataFrame,
        config:     MRConfig = None,
    ):
        self.futures = futures_df.copy()
        self.macro   = macro_df.copy()
        self.config  = config or MRConfig()

        # Populated by fit()
        self.spread:    Optional[pd.Series] = None
        self.ou_params: Optional[dict]      = None
        self.half_life: Optional[float]     = None

    # ------------------------------------------------------------------
    # Fair value construction
    # ------------------------------------------------------------------

    def _build_fair_value(self) -> pd.Series:
        """
        Construct a daily fair value series for the futures implied rate.

        The spread (implied_rate - fair_value) is the quantity we test
        for mean reversion. A persistently positive spread means the market
        is pricing rates above the current policy anchor — a dovish bet.

        Sources (configured via MRConfig.fair_value_source):
            "fed_midpoint" : (upper_target + lower_target) / 2
            "fed_upper"    : upper bound of target range
            "sofr_fixing"  : actual SOFR daily fixing
        """
        src = self.config.fair_value_source
        macro = self.macro.reindex(self.futures.index, method="ffill")

        if src == "fed_midpoint":
            if "fed_funds_target_upper" in macro.columns and "fed_funds_target_lower" in macro.columns:
                fv = (macro["fed_funds_target_upper"] + macro["fed_funds_target_lower"]) / 2
            elif "fed_funds_effective" in macro.columns:
                logger.warning("Target bounds not found; using effective FF rate as fair value.")
                fv = macro["fed_funds_effective"]
            else:
                raise KeyError("No Fed Funds rate series found in macro_df.")

        elif src == "fed_upper":
            fv = macro["fed_funds_target_upper"]

        elif src == "sofr_fixing":
            fv = macro["sofr_fixing"]

        else:
            raise ValueError(f"Unknown fair_value_source: {src!r}")

        fv.name = "fair_value"
        return fv

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self) -> "MeanReversionSignal":
        """
        Estimate OU parameters on the full available history (or fit_window).
        Must be called before compute_signal().
        """
        fair_value   = self._build_fair_value()
        self.spread  = (self.futures["implied_rate"] - fair_value).dropna()
        self.spread.name = "spread"

        fit_series = (
            self.spread.iloc[-self.config.fit_window:]
            if self.config.fit_window
            else self.spread
        )

        self.ou_params = estimate_ou_params(fit_series)
        self.half_life = self.ou_params["half_life"]

        if not self.ou_params["is_mean_reverting"]:
            logger.warning(
                "Spread does not appear mean-reverting in the fit window "
                "(β=%.4f). Signal will be suppressed. "
                "Consider a shorter fit_window or different fair_value_source.",
                self.ou_params["beta"],
            )

        return self

    # ------------------------------------------------------------------
    # Compute signal
    # ------------------------------------------------------------------

    def compute_signal(self) -> pd.DataFrame:
        """
        Run the full pipeline and return a results DataFrame.

        Returns
        -------
        pd.DataFrame with columns:
            close        : futures price
            implied_rate : 100 - close
            fair_value   : policy rate anchor
            spread       : implied_rate - fair_value
            zscore_mr    : rolling Z-score of spread
            signal_mr    : {-1, 0, +1} position signal
            half_life    : scalar broadcast (for reference)
        """
        if self.spread is None or self.ou_params is None:
            raise RuntimeError("Call fit() before compute_signal().")

        # Suppress signal if spread is not mean-reverting
        if not self.ou_params["is_mean_reverting"]:
            hl = self.half_life
            if hl < self.config.min_half_life or hl > self.config.max_half_life:
                logger.warning(
                    "Half-life %.1f days outside valid range [%.0f, %.0f]. "
                    "Returning zero signal.",
                    hl, self.config.min_half_life, self.config.max_half_life,
                )
                result = self._build_results_frame()
                result["zscore_mr"] = np.nan
                result["signal_mr"] = 0
                return result

        zscore = compute_zscore(
            self.spread,
            window=self.config.zscore_window,
            half_life=self.half_life,
        )

        signal = zscore_to_signal(
            zscore,
            entry_z=self.config.entry_z,
            exit_z=self.config.exit_z,
        )

        result = self._build_results_frame()
        result["zscore_mr"] = zscore
        result["signal_mr"] = signal
        result["half_life"]  = round(self.half_life, 1)

        logger.info(
            "Signal computed: %d long  %d short  %d flat  (half-life=%.1f days)",
            (signal == 1).sum(),
            (signal == -1).sum(),
            (signal == 0).sum(),
            self.half_life,
        )

        return result

    def _build_results_frame(self) -> pd.DataFrame:
        fair_value = self._build_fair_value()
        df = self.futures[["close", "implied_rate"]].copy()
        df["fair_value"] = fair_value.reindex(df.index, method="ffill")
        df["spread"]     = self.spread.reindex(df.index)
        return df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a concise summary of OU fit and signal parameters."""
        if self.ou_params is None:
            print("Model not yet fitted. Call fit() first.")
            return

        cfg = self.config
        p   = self.ou_params
        hl  = self.half_life

        print("\n" + "=" * 52)
        print("  Mean Reversion Signal — OU Fit Summary")
        print("=" * 52)
        print(f"  Fair value source   : {cfg.fair_value_source}")
        print(f"  Observations        : {p['n_obs']}")
        print(f"  Mean reversion speed: κ = {p['kappa']:.4f} (annualised)")
        print(f"  Long-run mean (μ)   : {p['mu']:.4f}%")
        print(f"  Residual vol (σ)    : {p['sigma']:.4f}% (annualised)")
        print(f"  Half-life           : {hl:.1f} business days")
        print(f"  R²                  : {p['r_squared']:.4f}")
        print(f"  Mean-reverting      : {p['is_mean_reverting']}")
        print(f"  Entry Z threshold   : ±{cfg.entry_z}")
        print(f"  Exit  Z threshold   : ±{cfg.exit_z}")
        zscore_window = cfg.zscore_window or max(int(2 * hl), 20)
        print(f"  Z-score window      : {zscore_window} days")
        print("=" * 52 + "\n")