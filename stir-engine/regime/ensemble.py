"""
regime/ensemble.py
-------------------
Ensemble weighting layer for the STIR engine.

Combines the mean reversion (MR) and momentum (MOM) signals using
regime-dependent weights derived from the Markov-switching classifier.
Applies a crisis override that suppresses technical signals when the
market is in a dislocated state.

Weight matrix (configurable via EnsembleConfig):

  Regime      MR weight   MOM weight   OSINT weight   Notes
  ─────────   ─────────   ──────────   ────────────   ──────────────────────────
  Ranging     0.70        0.20         0.10           Mean reversion dominant
  Trending    0.15        0.75         0.10           Momentum dominant
  Crisis      0.15        0.10         0.75           OSINT dominant; tech suppressed

These are starting defaults — they should be calibrated via walk-forward
optimisation once the OSINT module is complete.

Output:
  blended_z      : weighted sum of individual Z-scores
  signal_ensemble: {-1, 0, +1} final position signal
  is_crisis      : boolean crisis flag (triggers override)
  active_weights : dict showing MR/MOM/OSINT weight at each date

Dependencies:
    pip install numpy pandas

Usage:
    from regime.ensemble import Ensemble, EnsembleConfig

    ens = Ensemble(combined_df, regime_classifier)
    results = ens.compute()
    ens.summary(results)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    """
    Regime-dependent weight matrix and blending parameters.

    Weights for each regime must sum to 1.0.
    OSINT weight is a placeholder (0.10) until the OSINT module is built —
    it is redistributed to MR and MOM in proportion until then.
    """

    # --- Weight matrix: {regime: {signal: weight}} ---
    weights: dict = field(default_factory=lambda: {
        "ranging": {
            "mr":    0.70,
            "mom":   0.20,
            "osint": 0.10,
        },
        "trending": {
            "mr":    0.15,
            "mom":   0.75,
            "osint": 0.10,
        },
        "crisis": {
            "mr":    0.15,
            "mom":   0.10,
            "osint": 0.75,
        },
    })

    # Entry/exit thresholds for the blended Z-score
    entry_z:    float = 1.3
    exit_z:     float = 0.5

    # If True, redistribute OSINT weight to MR and MOM (proportionally)
    # until the OSINT module is wired in. Set to False when OSINT is live.
    osint_stub: bool  = True

    # Crisis override: when True, zero out MR and MOM signals regardless
    # of regime probabilities. Weight goes entirely to OSINT (stubbed as 0
    # until OSINT module is built — effectively a "stay flat" signal).
    apply_crisis_override: bool = True

    # Soft blending: interpolate weights using full regime probability
    # distribution rather than hard-assigning the dominant regime.
    # Recommended True — gives smoother transitions between regimes.
    soft_blend: bool = True


# ---------------------------------------------------------------------------
# Weight resolution
# ---------------------------------------------------------------------------

def _resolve_weights(
    regime_probs: pd.Series,
    config: EnsembleConfig,
    osint_available: bool = False,
) -> dict:
    """
    Compute effective signal weights for a single timestep given regime
    probability distribution.

    If soft_blend=True, weights are the expectation over regime probabilities:
        w_signal = Σ_regime [p(regime) × config.weights[regime][signal]]

    If soft_blend=False, weights are from the dominant (argmax) regime.

    Parameters
    ----------
    regime_probs    : pd.Series with keys "prob_ranging", "prob_trending",
                      "prob_crisis" (or subset for two-state model)
    config          : EnsembleConfig
    osint_available : whether OSINT signal is live

    Returns
    -------
    dict: {"mr": w_mr, "mom": w_mom, "osint": w_osint}  summing to 1.0
    """
    cfg = config
    wm  = cfg.weights

    # Map probability column names to regime keys
    regime_map = {
        "prob_ranging":  "ranging",
        "prob_trending": "trending",
        "prob_crisis":   "crisis",
    }

    if cfg.soft_blend:
        w_mr    = 0.0
        w_mom   = 0.0
        w_osint = 0.0

        for prob_col, regime_label in regime_map.items():
            p = regime_probs.get(prob_col, 0.0)
            if regime_label in wm:
                w_mr    += p * wm[regime_label]["mr"]
                w_mom   += p * wm[regime_label]["mom"]
                w_osint += p * wm[regime_label]["osint"]

    else:
        # Hard assignment to dominant regime
        dominant = max(
            [(k, regime_probs.get(f"prob_{v}", 0.0)) for k, v in
             {"ranging": "ranging", "trending": "trending", "crisis": "crisis"}.items()],
            key=lambda x: x[1],
        )[0]
        dominant_regime = wm.get(dominant, wm["ranging"])
        w_mr    = dominant_regime["mr"]
        w_mom   = dominant_regime["mom"]
        w_osint = dominant_regime["osint"]

    # Redistribute OSINT weight if OSINT is not yet live
    if cfg.osint_stub or not osint_available:
        total_tech = w_mr + w_mom
        if total_tech > 0:
            w_mr  += w_osint * (w_mr  / total_tech)
            w_mom += w_osint * (w_mom / total_tech)
        w_osint = 0.0

    # Normalise to sum to 1.0
    total = w_mr + w_mom + w_osint
    if total > 0:
        w_mr    /= total
        w_mom   /= total
        w_osint /= total

    return {"mr": w_mr, "mom": w_mom, "osint": w_osint}


# ---------------------------------------------------------------------------
# Signal generation with hysteresis
# ---------------------------------------------------------------------------

def _zscore_to_signal(
    zscore: pd.Series,
    entry_z: float,
    exit_z:  float,
    name:    str = "signal_ensemble",
) -> pd.Series:
    """Standard hysteresis signal generator — same pattern as MR and MOM modules."""
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
# Main ensemble class
# ---------------------------------------------------------------------------

class Ensemble:
    """
    Regime-weighted signal ensemble for the STIR engine.

    Parameters
    ----------
    combined_df        : DataFrame output from main.py containing:
                            zscore_mr, signal_mr,
                            zscore_mom, signal_mom
    regime_classifier  : fitted RegimeClassifier instance
    osint_signal       : optional pd.Series of OSINT Z-scores (stub = None)
    config             : EnsembleConfig (optional)
    """

    def __init__(
        self,
        combined_df:       pd.DataFrame,
        regime_classifier,
        osint_signal:      Optional[pd.Series] = None,
        config:            EnsembleConfig = None,
    ):
        self.combined   = combined_df.copy()
        self.rc         = regime_classifier
        self.osint      = osint_signal
        self.config     = config or EnsembleConfig()

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def compute(self) -> pd.DataFrame:
        """
        Compute the ensemble blended signal across the full history.

        Returns
        -------
        pd.DataFrame with all input columns plus:
            prob_ranging, prob_trending, prob_crisis   : smoothed regime probs
            regime                                     : dominant regime label
            w_mr, w_mom, w_osint                       : effective signal weights
            blended_z                                  : weighted Z-score
            signal_ensemble                            : {-1, 0, +1}
            is_crisis                                  : boolean crisis flag
        """
        cfg = self.config

        if self.rc.smoothed_probs is None:
            raise RuntimeError(
                "RegimeClassifier must be fitted before computing ensemble. "
                "Call rc.fit() first."
            )

        # --- Align regime probabilities to combined_df index ---
        probs = self.rc.smoothed_probs.reindex(self.combined.index, method="ffill")
        dominant = self.rc.dominant_regime.reindex(self.combined.index, method="ffill")

        # --- Crisis flag series ---
        crisis_series = self.rc.crisis_series().reindex(self.combined.index, fill_value=False)

        # --- Build weight time series ---
        osint_available = self.osint is not None
        weights_list = []

        for date, row in probs.iterrows():
            w = _resolve_weights(row, cfg, osint_available=osint_available)
            weights_list.append(w)

        weights_df = pd.DataFrame(weights_list, index=probs.index)
        weights_df.columns = ["w_mr", "w_mom", "w_osint"]

        # --- Retrieve Z-scores ---
        z_mr  = self.combined["zscore_mr"].reindex(probs.index).astype(float)
        z_mom = self.combined["zscore_mom"].reindex(probs.index).astype(float)
        if osint_available:
            # Guard: if a DataFrame was passed, extract the zscore_osint column
            osint_ser = self.osint
            if isinstance(osint_ser, pd.DataFrame):
                if "zscore_osint" in osint_ser.columns:
                    osint_ser = osint_ser["zscore_osint"]
                else:
                    logger.warning(
                        "osint_signal is a DataFrame with no zscore_osint column "
                        "— treating as unavailable."
                    )
                    osint_ser = None
            z_osint = (
                osint_ser.reindex(probs.index).fillna(0.0).astype(float)
                if osint_ser is not None
                else pd.Series(0.0, index=probs.index)
            )
        else:
            z_osint = pd.Series(0.0, index=probs.index)

        # --- Apply crisis override ---
        # In crisis: zero out technical signals; blended_z driven by OSINT only.
        # If OSINT is not live (stub), blended_z → 0 in crisis (stay flat).
        if cfg.apply_crisis_override:
            z_mr_eff    = z_mr.where(~crisis_series, 0.0)
            z_mom_eff   = z_mom.where(~crisis_series, 0.0)
            z_osint_eff = z_osint  # OSINT is not suppressed in crisis

            # Log override activations
            crisis_dates = crisis_series[crisis_series].index
            if len(crisis_dates) > 0:
                logger.warning(
                    "Crisis override active on %d dates. "
                    "First: %s  Last: %s  "
                    "Technical signals suppressed.",
                    len(crisis_dates),
                    crisis_dates[0].date(),
                    crisis_dates[-1].date(),
                )
        else:
            z_mr_eff    = z_mr
            z_mom_eff   = z_mom
            z_osint_eff = z_osint

        # --- Compute blended Z-score ---
        blended_z = (
            weights_df["w_mr"]    * z_mr_eff    +
            weights_df["w_mom"]   * z_mom_eff   +
            weights_df["w_osint"] * z_osint_eff
        )
        blended_z.name = "blended_z"

        # --- Generate ensemble signal ---
        signal = _zscore_to_signal(blended_z, cfg.entry_z, cfg.exit_z)

        # --- Assemble output ---
        result = self.combined.reindex(probs.index).copy()

        for col in probs.columns:
            result[col] = probs[col]

        result["regime"]           = dominant
        result["w_mr"]             = weights_df["w_mr"]
        result["w_mom"]            = weights_df["w_mom"]
        result["w_osint"]          = weights_df["w_osint"]
        result["blended_z"]        = blended_z
        result["signal_ensemble"]  = signal
        result["is_crisis"]        = crisis_series

        long_  = (signal == 1).sum()
        short_ = (signal == -1).sum()
        flat_  = (signal == 0).sum()

        logger.info(
            "Ensemble signal: %d long  %d short  %d flat",
            long_, short_, flat_,
        )
        logger.info(
            "Regime distribution — Ranging: %.1f%%  Trending: %.1f%%  Crisis: %.1f%%",
            (dominant == "ranging").mean()  * 100,
            (dominant == "trending").mean() * 100,
            (dominant == "crisis").mean()   * 100,
        )

        return result

    # ------------------------------------------------------------------
    # Weight-at-date query (useful for live use)
    # ------------------------------------------------------------------

    def current_weights(self, result: pd.DataFrame) -> dict:
        """
        Return the most recent effective signal weights.
        Useful for a live dashboard: "what is the engine currently doing?"
        """
        last = result.iloc[-1]
        return {
            "date":    result.index[-1].date(),
            "regime":  last.get("regime", "unknown"),
            "w_mr":    round(last.get("w_mr",    0.0), 3),
            "w_mom":   round(last.get("w_mom",   0.0), 3),
            "w_osint": round(last.get("w_osint", 0.0), 3),
            "blended_z": round(last.get("blended_z", 0.0), 4),
            "signal":  int(last.get("signal_ensemble", 0)),
            "is_crisis": bool(last.get("is_crisis", False)),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self, result: Optional[pd.DataFrame] = None) -> None:
        """Print ensemble configuration and, if result is provided, live state."""
        cfg = self.config
        print("\n" + "=" * 56)
        print("  Ensemble — Configuration Summary")
        print("=" * 56)
        print(f"  Soft blend          : {cfg.soft_blend}")
        print(f"  OSINT stub          : {cfg.osint_stub}")
        print(f"  Crisis override     : {cfg.apply_crisis_override}")
        print(f"  Entry Z             : ±{cfg.entry_z}")
        print(f"  Exit  Z             : ±{cfg.exit_z}")
        print("\n  Weight matrix:")
        for regime, w in cfg.weights.items():
            print(f"    {regime:<10} : MR={w['mr']:.2f}  MOM={w['mom']:.2f}  OSINT={w['osint']:.2f}")

        if result is not None:
            cw = self.current_weights(result)
            print(f"\n  Current state ({cw['date']}):")
            print(f"    Regime          : {cw['regime']}")
            print(f"    Weights         : MR={cw['w_mr']:.2f}  MOM={cw['w_mom']:.2f}  OSINT={cw['w_osint']:.2f}")
            print(f"    Blended Z-score : {cw['blended_z']:.4f}")
            print(f"    Signal          : {cw['signal']:+d}")
            print(f"    Crisis flag     : {cw['is_crisis']}")

        print("=" * 56 + "\n")