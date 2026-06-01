"""
regime/classifier.py
---------------------
Markov-switching regime classifier for the STIR engine.

Fits a Markov-switching autoregression (Hamilton 1989) on futures log-returns
to identify latent market regimes. Outputs smoothed regime probabilities at
every timestep — these drive the ensemble weighting layer.

Two regime configurations are supported:

  Two-state  : Ranging  (low vol, mean-reverting)
               Trending (higher vol, persistent direction)

  Three-state: Ranging  (low vol, slight +autocorr)
               Trending (moderate vol, strong +autocorr)
               Crisis   (high vol, negative autocorr / gap risk)

The three-state model is preferred because it allows the crisis override
to activate cleanly without distorting the ranging/trending distinction.

Model:
    r_t = μ_s + φ_s × r_{t-1} + ε_t,   ε_t ~ N(0, σ_s²)

where s ∈ {0, 1, 2} is the latent regime. Each regime has its own
intercept (μ), AR(1) coefficient (φ), and volatility (σ).

Dependencies:
    pip install statsmodels numpy pandas

Usage:
    from regime.classifier import RegimeClassifier, RegimeConfig

    rc = RegimeClassifier(futures_df)
    rc.fit()
    rc.summary()
    probs = rc.smoothed_probs          # DataFrame: prob_ranging, prob_trending, prob_crisis
    regime = rc.dominant_regime        # Series: 0=ranging, 1=trending, 2=crisis
"""

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
class RegimeConfig:
    """
    Tunable parameters for the Markov-switching classifier.
    """
    # Number of latent regimes: 2 or 3
    n_regimes:          int   = 3

    # AR order within each regime (1 is standard for daily returns)
    ar_order:           int   = 1

    # Minimum observations before the classifier is considered reliable
    min_obs:            int   = 126     # ~6 months daily

    # Rolling refit window (business days). None = expanding window (fit once).
    # Set to 504 (2yr) for walk-forward use; None for initial exploratory fit.
    refit_window:       Optional[int] = None

    # How often to refit in walk-forward mode (business days)
    refit_step:         int   = 21      # monthly

    # Crisis detection thresholds (used by ensemble override layer)
    # These are calibrated on annualised realised vol
    crisis_vol_threshold:   float = 0.25    # 25% annualised — hard circuit breaker
    crisis_prob_threshold:  float = 0.60    # >60% crisis probability → override

    # Regime label assignment heuristic:
    # After fitting, states are assigned labels based on their estimated σ.
    # Lowest σ → ranging, highest σ → crisis (three-state),
    # middle σ → trending.
    # Set to False to use raw state indices (0, 1, 2) instead.
    auto_label:         bool  = True


# ---------------------------------------------------------------------------
# Regime label assignment
# ---------------------------------------------------------------------------

def _assign_regime_labels(
    model_result,
    n_regimes: int,
) -> dict:
    """
    Map fitted state indices to economic regime labels by sorting on
    estimated regime volatility (σ).

    Returns a dict: {state_index: label_string}
    e.g. {0: "trending", 1: "ranging", 2: "crisis"}

    The mapping is:
        lowest  σ → "ranging"
        highest σ → "crisis"   (three-state only)
        middle  σ → "trending"
    """
    # Extract per-regime standard deviations from fitted params
    try:
        # statsmodels stores regime-specific params; sigma is sqrt of variance
        sigmas = {}
        for i in range(n_regimes):
            # Parameter name varies by statsmodels version
            for key in [f"sigma[{i}]", f"sqrt.var[{i}]", f"std[{i}]"]:
                if key in model_result.params.index:
                    sigmas[i] = model_result.params[key]
                    break
            if i not in sigmas:
                # Fallback: compute from smoothed residuals
                sigmas[i] = 0.01 * (i + 1)   # ordinal placeholder

    except Exception:
        # If param extraction fails, fall back to ordinal assignment
        if n_regimes == 2:
            return {0: "ranging", 1: "trending"}
        return {0: "ranging", 1: "trending", 2: "crisis"}

    sorted_states = sorted(sigmas, key=lambda s: sigmas[s])

    if n_regimes == 2:
        return {sorted_states[0]: "ranging", sorted_states[1]: "trending"}

    # Three-state
    return {
        sorted_states[0]: "ranging",
        sorted_states[1]: "trending",
        sorted_states[2]: "crisis",
    }


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

class RegimeClassifier:
    """
    Markov-switching autoregression regime classifier.

    Parameters
    ----------
    futures_df : DataFrame from data.cme.fetch_continuous_front()
                 Must have columns: log_return, realised_vol_{N}d
    config     : RegimeConfig (optional)
    """

    def __init__(
        self,
        futures_df: pd.DataFrame,
        config:     RegimeConfig = None,
    ):
        self.futures        = futures_df.copy()
        self.config         = config or RegimeConfig()
        self.model_result   = None
        self.label_map:     dict = {}

        # Outputs — populated by fit() / compute_probabilities()
        self.smoothed_probs:   Optional[pd.DataFrame] = None
        self.dominant_regime:  Optional[pd.Series]    = None
        self.filtered_probs:   Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        series: Optional[pd.Series] = None,
    ) -> "RegimeClassifier":
        """
        Fit the Markov-switching AR model on log returns.

        Parameters
        ----------
        series : override return series (defaults to futures_df["log_return"])

        Populates:
            self.model_result   : fitted statsmodels result object
            self.smoothed_probs : DataFrame of smoothed regime probabilities
            self.dominant_regime: Series of most-probable regime label per date
            self.label_map      : dict mapping state index → regime label
        """
        try:
            from statsmodels.tsa.regime_switching.markov_autoregression import (
                MarkovAutoregression,
            )
        except ImportError:
            raise ImportError(
                "statsmodels >= 0.14 is required. Run: pip install statsmodels"
            )

        cfg = self.config
        returns = series if series is not None else self.futures["log_return"]
        returns = returns.dropna()

        if len(returns) < cfg.min_obs:
            raise ValueError(
                f"Insufficient data: {len(returns)} obs < min_obs={cfg.min_obs}. "
                "Extend DEFAULT_START_DATE in config.py or reduce min_obs."
            )

        logger.info(
            "Fitting Markov-switching AR(%d) with %d regimes on %d observations...",
            cfg.ar_order, cfg.n_regimes, len(returns),
        )

        model = MarkovAutoregression(
            returns,
            k_regimes=cfg.n_regimes,
            order=cfg.ar_order,
            switching_ar=True,        # AR coefficient varies by regime
            switching_variance=True,  # Variance varies by regime (key for crisis detection)
        )

        # Fit with multiple random starts to avoid local minima.
        # EM convergence can fail on homogeneous windows — try progressively
        # more starts, then fall back to a 2-regime model, then equal weights.
        result = None
        for search_reps in [20, 50, 100]:
            try:
                result = model.fit(
                    search_reps=search_reps,
                    search_scale=1.0,
                    disp=False,
                )
                # Validate transition probabilities
                trans = result.regime_transition
                if np.any(np.isnan(trans)) or np.any(trans < 0) or np.any(trans > 1):
                    raise ValueError("Invalid regime transition probabilities.")
                break   # success
            except Exception as exc:
                logger.warning(
                    "MLE fit attempt (search_reps=%d) failed: %s — retrying.",
                    search_reps, exc,
                )
                result = None

        if result is None and cfg.n_regimes == 3:
            # Fallback: try 2-regime model which is more robust on short windows
            logger.warning(
                "3-regime fit failed — falling back to 2-regime model for this window."
            )
            try:
                model2 = MarkovAutoregression(
                    returns,
                    k_regimes=2,
                    order=cfg.ar_order,
                    switching_ar=True,
                    switching_variance=True,
                )
                result = model2.fit(search_reps=50, search_scale=1.0, disp=False)
                # Remap: treat state with higher vol as both trending and crisis
                # (2-regime fallback: state 0=ranging, state 1=trending/crisis)
                self.label_map = {0: "ranging", 1: "trending"}
                cfg = RegimeConfig(
                    n_regimes=2,
                    ar_order=cfg.ar_order,
                    min_obs=cfg.min_obs,
                    auto_label=False,
                    crisis_vol_threshold=cfg.crisis_vol_threshold,
                    crisis_prob_threshold=cfg.crisis_prob_threshold,
                )
                self.config = cfg
            except Exception as exc2:
                logger.warning("2-regime fallback also failed: %s", exc2)
                result = None

        if result is None:
            # Last resort: uniform equal-weight probabilities
            # The ensemble will use equal MR/MOM weights; crisis override
            # still fires via the vol threshold in crisis_series()
            logger.warning(
                "All MLE fits failed — using uniform regime probabilities. "
                "Crisis override via vol threshold remains active."
            )
            n = len(returns)
            if cfg.n_regimes == 3:
                cols = ["prob_ranging", "prob_trending", "prob_crisis"]
            else:
                cols = ["prob_ranging", "prob_trending"]
            probs_uniform = pd.DataFrame(
                1.0 / cfg.n_regimes,
                index=returns.index,
                columns=cols,
            )
            self.smoothed_probs  = probs_uniform
            self.filtered_probs  = probs_uniform.copy()
            self.model_result    = None
            self._build_dominant_regime()
            return self

        self.model_result = result
        logger.info("Fit complete. Log-likelihood: %.4f", self.model_result.llf)

        # Assign economic labels to state indices
        if cfg.auto_label:
            self.label_map = _assign_regime_labels(self.model_result, cfg.n_regimes)
        else:
            labels = ["ranging", "trending", "crisis"]
            self.label_map = {i: labels[i] for i in range(cfg.n_regimes)}

        logger.info("Regime label map: %s", self.label_map)

        # Extract and label smoothed probabilities
        self._extract_probabilities()

        return self

    # ------------------------------------------------------------------
    # Rolling refit (walk-forward)
    # ------------------------------------------------------------------

    def fit_rolling(self) -> "RegimeClassifier":
        """
        Fit the model in a rolling walk-forward fashion.

        Uses config.refit_window as the lookback and config.refit_step
        as the refit frequency. Stitches together out-of-sample probability
        estimates to avoid lookahead bias.

        This is the correct method to use for backtesting. It is slower
        than fit() — typically 2-5 minutes for 5 years of daily data
        with monthly refits.

        Returns self with smoothed_probs populated from OOS estimates only.
        """
        cfg     = self.config
        returns = self.futures["log_return"].dropna()
        dates   = returns.index

        if cfg.refit_window is None:
            raise ValueError(
                "refit_window must be set for rolling fit. "
                "Suggested: RegimeConfig(refit_window=504)"
            )

        all_probs = []

        for end_idx in range(cfg.refit_window, len(dates), cfg.refit_step):
            start_idx   = end_idx - cfg.refit_window
            window_data = returns.iloc[start_idx:end_idx]
            oos_data    = returns.iloc[end_idx: end_idx + cfg.refit_step]

            if len(oos_data) == 0:
                break

            try:
                rc_window = RegimeClassifier(
                    self.futures.loc[window_data.index],
                    config=RegimeConfig(
                        n_regimes=cfg.n_regimes,
                        ar_order=cfg.ar_order,
                        min_obs=cfg.min_obs,
                        auto_label=cfg.auto_label,
                    ),
                )
                rc_window.fit(series=window_data)

                # Predict filtered probs on the OOS window
                # Use the last filtered state as initial condition
                oos_probs = rc_window._predict_oos(oos_data)
                all_probs.append(oos_probs)

                logger.debug(
                    "Rolling fit: window [%s → %s]  OOS [%s → %s]",
                    window_data.index[0].date(), window_data.index[-1].date(),
                    oos_data.index[0].date(),    oos_data.index[-1].date(),
                )

            except Exception as exc:
                logger.warning("Rolling fit failed at %s: %s", dates[end_idx].date(), exc)
                continue

        if not all_probs:
            raise RuntimeError("Rolling fit produced no results. Check data length and min_obs.")

        self.smoothed_probs = pd.concat(all_probs).sort_index()
        self._build_dominant_regime()

        logger.info(
            "Rolling fit complete: %d OOS probability observations",
            len(self.smoothed_probs),
        )

        return self

    # ------------------------------------------------------------------
    # Probability extraction helpers
    # ------------------------------------------------------------------

    def _extract_probabilities(self) -> None:
        """
        Pull smoothed probabilities from the fitted model result and
        label columns by regime name.
        """
        if self.model_result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # statsmodels stores smoothed_marginal_probabilities as (T × k) array
        raw = self.model_result.smoothed_marginal_probabilities

        if hasattr(raw, "values"):
            raw = raw.values

        index = self.futures["log_return"].dropna().index[-len(raw):]

        cols = {
            f"prob_{self.label_map[i]}": raw[:, i]
            for i in range(self.config.n_regimes)
        }
        self.smoothed_probs = pd.DataFrame(cols, index=index)

        # Also store filtered (real-time) probabilities
        raw_filt = self.model_result.filtered_marginal_probabilities
        if hasattr(raw_filt, "values"):
            raw_filt = raw_filt.values

        cols_filt = {
            f"prob_{self.label_map[i]}_filtered": raw_filt[:, i]
            for i in range(self.config.n_regimes)
        }
        self.filtered_probs = pd.DataFrame(cols_filt, index=index)

        self._build_dominant_regime()

    def _build_dominant_regime(self) -> None:
        """Set dominant_regime Series from smoothed_probs."""
        self.dominant_regime = self.smoothed_probs.idxmax(axis=1).str.replace(
            "prob_", "", regex=False
        )
        self.dominant_regime.name = "regime"

    def _predict_oos(self, oos_returns: pd.Series) -> pd.DataFrame:
        """
        Approximate OOS regime probabilities using the fitted transition
        matrix applied forward from the last in-sample filtered state.
        Used internally by fit_rolling().
        """
        if self.model_result is None:
            raise RuntimeError("fit() must be called before _predict_oos()")

        # Get last filtered state distribution
        last_filt = self.model_result.filtered_marginal_probabilities[-1]
        if hasattr(last_filt, "values"):
            last_filt = last_filt.values

        # Extract transition matrix P (k × k)
        try:
            P = self.model_result.transition
            if hasattr(P, "values"):
                P = P.values
        except AttributeError:
            # Fallback: uniform transition
            k = self.config.n_regimes
            P = np.full((k, k), 1.0 / k)

        # Propagate state distribution forward using transition matrix
        probs_list = []
        state = last_filt.copy()

        for date in oos_returns.index:
            state = P.T @ state          # one-step ahead prediction
            state = np.clip(state, 0, 1)
            state /= state.sum()         # renormalise
            probs_list.append(state.copy())

        raw   = np.array(probs_list)
        cols  = {f"prob_{self.label_map[i]}": raw[:, i] for i in range(self.config.n_regimes)}
        return pd.DataFrame(cols, index=oos_returns.index)

    # ------------------------------------------------------------------
    # Crisis detection
    # ------------------------------------------------------------------

    def is_crisis(
        self,
        date: Optional[pd.Timestamp] = None,
        realised_vol: Optional[float] = None,
    ) -> bool:
        """
        Hard crisis check combining regime probability and realised vol.

        Returns True if EITHER:
          (a) crisis regime probability > crisis_prob_threshold, OR
          (b) realised_vol (annualised) > crisis_vol_threshold

        This is the circuit breaker that triggers the ensemble override.

        Parameters
        ----------
        date         : specific date to check (defaults to last available)
        realised_vol : current annualised realised vol (from cme.py)
        """
        cfg = self.config

        # Vol-based hard override (doesn't require regime model)
        if realised_vol is not None:
            if realised_vol > cfg.crisis_vol_threshold:
                logger.warning(
                    "CRISIS OVERRIDE: realised_vol %.2f%% > threshold %.2f%%",
                    realised_vol * 100, cfg.crisis_vol_threshold * 100,
                )
                return True

        # Regime probability check
        if self.smoothed_probs is not None and "prob_crisis" in self.smoothed_probs.columns:
            if date is None:
                date = self.smoothed_probs.index[-1]
            try:
                crisis_prob = self.smoothed_probs.loc[date, "prob_crisis"]
                if crisis_prob > cfg.crisis_prob_threshold:
                    logger.warning(
                        "CRISIS OVERRIDE: crisis regime prob %.2f > threshold %.2f at %s",
                        crisis_prob, cfg.crisis_prob_threshold, date.date(),
                    )
                    return True
            except KeyError:
                pass

        return False

    def crisis_series(self) -> pd.Series:
        """
        Return a boolean Series indicating crisis dates across the full history.
        Combines vol threshold and regime probability threshold.
        Useful for backtesting — shade crisis periods on charts.
        """
        if self.smoothed_probs is None:
            raise RuntimeError("Call fit() before crisis_series()")

        cfg       = self.config
        crisis    = pd.Series(False, index=self.smoothed_probs.index, name="is_crisis")

        # Regime probability criterion
        if "prob_crisis" in self.smoothed_probs.columns:
            crisis |= self.smoothed_probs["prob_crisis"] > cfg.crisis_prob_threshold

        # Vol criterion (align on common index)
        if "realised_vol_20d" in self.futures.columns:
            vol = self.futures["realised_vol_20d"].reindex(crisis.index)
            crisis |= vol > cfg.crisis_vol_threshold

        return crisis

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print fitted regime parameters and transition matrix."""
        if self.model_result is None:
            print("Model not fitted. Call fit() first.")
            return

        cfg = self.config
        print("\n" + "=" * 56)
        print("  Regime Classifier — Markov-Switching AR Summary")
        print("=" * 56)
        print(f"  Regimes             : {cfg.n_regimes}  {list(self.label_map.values())}")
        print(f"  AR order            : {cfg.ar_order}")
        print(f"  Observations        : {len(self.futures['log_return'].dropna())}")
        print(f"  Log-likelihood      : {self.model_result.llf:.4f}")
        print(f"  AIC                 : {self.model_result.aic:.4f}")

        if self.smoothed_probs is not None:
            print("\n  Average smoothed regime probabilities:")
            for col in self.smoothed_probs.columns:
                label = col.replace("prob_", "")
                mean_p = self.smoothed_probs[col].mean()
                print(f"    {label:<12} : {mean_p:.3f}")

            if self.dominant_regime is not None:
                print("\n  Dominant regime distribution:")
                counts = self.dominant_regime.value_counts(normalize=True)
                for regime, pct in counts.items():
                    print(f"    {regime:<12} : {pct:.1%} of days")

        print(f"\n  Crisis thresholds:")
        print(f"    Vol override       : realised_vol > {cfg.crisis_vol_threshold:.0%}")
        print(f"    Prob override      : crisis_prob  > {cfg.crisis_prob_threshold:.0%}")
        print("=" * 56 + "\n")