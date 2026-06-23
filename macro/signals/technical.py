# signals/technical.py
# Technical / Price Action signals — RSI, MACD, SMA, ROC, Bollinger, ADX, Z-score

import pandas as pd
import numpy as np
from config.pairs import LOOKBACK


# ── RSI ────────────────────────────────────────────────────────────────────────

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_zone(val: float) -> str:
    if pd.isna(val):
        return "N/A"
    if val >= 70:
        return "Overbought"
    if val <= 30:
        return "Oversold"
    return "Neutral"


# ── MACD ───────────────────────────────────────────────────────────────────────

def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def macd_direction(histogram: pd.Series) -> str:
    """Bullish if last histogram > 0 and rising, Bearish if < 0 and falling."""
    if len(histogram.dropna()) < 2:
        return "N/A"
    last = histogram.iloc[-1]
    prev = histogram.iloc[-2]
    if last > 0 and last > prev:
        return "Bullish ↑"
    if last < 0 and last < prev:
        return "Bearish ↓"
    if last > 0:
        return "Bullish ~"
    if last < 0:
        return "Bearish ~"
    return "Neutral"


# ── SMA Cross ─────────────────────────────────────────────────────────────────

def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window).mean()


def sma_cross_signal(prices: pd.Series, fast: int = 20, slow: int = 50) -> str:
    if len(prices.dropna()) < slow:
        return "N/A"
    sma_f = compute_sma(prices, fast).iloc[-1]
    sma_s = compute_sma(prices, slow).iloc[-1]
    if pd.isna(sma_f) or pd.isna(sma_s):
        return "N/A"
    if sma_f > sma_s:
        return "Bull"
    if sma_f < sma_s:
        return "Bear"
    return "Flat"


# ── Rate of Change ─────────────────────────────────────────────────────────────

def compute_roc(prices: pd.Series, window: int) -> float:
    """% return over last `window` trading days."""
    if len(prices.dropna()) < window + 1:
        return np.nan
    end = prices.dropna().iloc[-1]
    start = prices.dropna().iloc[-(window + 1)]
    if start == 0:
        return np.nan
    return (end / start - 1) * 100


# ── Bollinger Bands %B ─────────────────────────────────────────────────────────

def compute_bollinger(
    prices: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (prices - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "pct_b": pct_b})


# ── ADX ────────────────────────────────────────────────────────────────────────

def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """True ADX — requires OHLC. Falls back to close-only proxy if H/L unavailable."""
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    dm_plus = (high - high.shift()).clip(lower=0)
    dm_minus = (low.shift() - low).clip(lower=0)
    dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)

    atr = tr.ewm(alpha=1 / window, min_periods=window).mean()
    di_plus = 100 * dm_plus.ewm(alpha=1 / window, min_periods=window).mean() / atr
    di_minus = 100 * dm_minus.ewm(alpha=1 / window, min_periods=window).mean() / atr

    dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / window, min_periods=window).mean()
    return adx


def adx_strength(val: float) -> str:
    if pd.isna(val):
        return "N/A"
    if val >= 40:
        return "Strong"
    if val >= 25:
        return "Trending"
    return "Weak"


# ── 52w Z-Score ───────────────────────────────────────────────────────────────

def compute_zscore(prices: pd.Series, window: int = 252) -> float:
    s = prices.dropna()
    if len(s) < window // 2:
        return np.nan
    roll = s.rolling(window)
    mu = roll.mean().iloc[-1]
    sigma = roll.std().iloc[-1]
    if sigma == 0 or pd.isna(sigma):
        return np.nan
    return (s.iloc[-1] - mu) / sigma


# ── Master technical signal builder ───────────────────────────────────────────

def build_technical_signals(spot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given spot_df with columns = pair names and DatetimeIndex,
    return a DataFrame with one row per pair and all technical signals.
    """
    records = []
    for pair in spot_df.columns:
        px = spot_df[pair].dropna()
        if len(px) < 60:
            continue

        rsi_val = compute_rsi(px).iloc[-1]
        macd_df = compute_macd(px)
        bb_df = compute_bollinger(px)
        adx_val = compute_adx(px, px, px).iloc[-1]   # close-only proxy

        rec = {
            "pair": pair,
            # RSI
            "rsi_14": round(rsi_val, 1),
            "rsi_zone": rsi_zone(rsi_val),
            # MACD
            "macd_signal": macd_direction(macd_df["histogram"]),
            "macd_hist": round(macd_df["histogram"].iloc[-1], 6),
            # SMA cross
            "sma_20_50": sma_cross_signal(px, 20, 50),
            "sma_50_200": sma_cross_signal(px, 50, 200),
            # ROC
            "roc_1m": round(compute_roc(px, LOOKBACK["1m"]), 2),
            "roc_3m": round(compute_roc(px, LOOKBACK["3m"]), 2),
            # Bollinger %B
            "bb_pct_b": round(bb_df["pct_b"].iloc[-1], 3),
            # ADX
            "adx_14": round(adx_val, 1) if not pd.isna(adx_val) else np.nan,
            "adx_strength": adx_strength(adx_val),
            # Z-score
            "zscore_1y": round(compute_zscore(px, 252), 2),
        }
        records.append(rec)

    return pd.DataFrame(records).set_index("pair")
