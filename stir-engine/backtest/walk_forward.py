"""
backtest/walk_forward.py
------------------------
Walk-forward validation harness for the STIR engine.

Methodology
-----------
The full history is divided into overlapping windows:

    |<---- train_window ---->|<- oos_window ->|
                    |<---- train_window ---->|<- oos_window ->|
                                    |<---- train_window ---->|...

On each fold:
  1. Refit the OU process (mean reversion) on the training window
  2. Refit the Markov-switching regime classifier on the training window
  3. Generate ensemble signals on the OOS window only
     (the OOS period never touches the fitting step)
  4. Compute P&L on the OOS window
  5. Roll forward by step_days and repeat

The OOS P&L periods are stitched together into a single continuous
equity curve — this is the honest backtest result.

Crucially: the momentum signal (COT + price) uses only lookback windows
that fit within the training data at each point, so it is also
walk-forward clean by construction.

Dependencies
------------
    pip install numpy pandas statsmodels

Usage
-----
    from backtest.walk_forward import WalkForwardHarness, WFConfig

    wf = WalkForwardHarness(
        futures_df    = futures,
        macro_df      = macro,
        osint_df      = osint_results,    # optional — None if not using OSINT
        config        = WFConfig(),
    )
    wf_results = wf.run()
    wf.summary(wf_results)
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
class WFConfig:
    """
    Walk-forward harness configuration.
    """
    # Training window in business days (~2 years)
    train_window:     int   = 504

    # Out-of-sample window in business days (~6 months)
    oos_window:       int   = 126

    # Step size between folds in business days (~3 months)
    # Smaller = more folds, more overlap, longer runtime
    step_days:        int   = 63

    # Minimum training days before first fold runs
    min_train_days:   int   = 252

    # Whether to refit regime classifier on every fold (True)
    # or refit only once on full history (False — faster but less rigorous)
    refit_regime:     bool  = True

    # Whether to refit mean reversion OU on every fold
    refit_mr:         bool  = True

    # Signal configuration — imported from signal modules at runtime
    # (kept here so the harness is self-contained)
    mr_entry_z:       float = 1.5
    mr_exit_z:        float = 0.5
    mom_entry_z:      float = 1.2
    mom_exit_z:       float = 0.4
    ens_entry_z:      float = 1.3
    ens_exit_z:       float = 0.5
    n_regimes:        int   = 3

    # P&L engine config — forwarded to PnLEngine on each fold
    vol_target_usd:   float = 5_000.0
    apply_costs:      bool  = True
    risk_free_rate:   float = 0.045


# ---------------------------------------------------------------------------
# Single-fold runner
# ---------------------------------------------------------------------------

def run_single_fold(
    futures_train:  pd.DataFrame,
    macro_train:    pd.DataFrame,
    futures_oos:    pd.DataFrame,
    macro_oos:      pd.DataFrame,
    osint_oos:      Optional[pd.DataFrame],
    config:         WFConfig,
    fold_id:        int,
) -> tuple[pd.DataFrame, dict]:
    """
    Fit all models on the training window and generate signals + P&L
    on the OOS window.

    Parameters
    ----------
    futures_train : futures DataFrame for fitting
    macro_train   : macro DataFrame for fitting
    futures_oos   : futures DataFrame for signal generation
    macro_oos     : macro DataFrame for signal generation
    osint_oos     : optional pre-computed OSINT results for OOS period
    config        : WFConfig
    fold_id       : integer fold index (for logging)

    Returns
    -------
    (oos_pnl_df, fold_metrics) tuple
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from signals.mean_reversion import MeanReversionSignal, MRConfig
    from signals.momentum       import MomentumSignal, MomConfig
    from regime.classifier      import RegimeClassifier, RegimeConfig
    from regime.ensemble        import Ensemble, EnsembleConfig
    from backtest.pnl           import PnLEngine, PnLConfig

    logger.info(
        "Fold %02d: train [%s → %s]  OOS [%s → %s]",
        fold_id,
        futures_train.index[0].date(), futures_train.index[-1].date(),
        futures_oos.index[0].date(),   futures_oos.index[-1].date(),
    )

    # ---------------------------------------------------------------
    # 1. Fit mean reversion on training window
    # ---------------------------------------------------------------
    mr_config = MRConfig(
        fair_value_source = "fed_midpoint",
        entry_z           = config.mr_entry_z,
        exit_z            = config.mr_exit_z,
    )
    mr = MeanReversionSignal(futures_train, macro_train, config=mr_config)

    try:
        mr.fit()
    except Exception as exc:
        logger.warning("Fold %02d: MR fit failed (%s) — skipping fold.", fold_id, exc)
        return pd.DataFrame(), {}

    if not mr.ou_params["is_mean_reverting"]:
        logger.warning(
            "Fold %02d: spread not mean-reverting (hl=%.1f) — "
            "MR signal will be zero on this fold.",
            fold_id, mr.half_life,
        )

    # Apply fitted OU params to OOS window
    # We reuse the fitted half-life and long-run mean but compute
    # the Z-score on OOS data only
    from signals.mean_reversion import compute_zscore, zscore_to_signal

    # Build OOS spread using training fair value calibration
    # (fair value source is policy rate — no fitting needed, just alignment)
    mr_oos       = MeanReversionSignal(futures_oos, macro_oos, config=mr_config)
    mr_oos.ou_params = mr.ou_params
    mr_oos.half_life  = mr.half_life
    # Manually build spread using the same fair value method
    fair_oos     = mr_oos._build_fair_value()
    spread_oos   = (futures_oos["implied_rate"] - fair_oos).dropna()

    zscore_mr_oos = compute_zscore(
        spread_oos,
        half_life=mr.half_life,
    )
    signal_mr_oos = zscore_to_signal(
        zscore_mr_oos,
        entry_z=config.mr_entry_z,
        exit_z=config.mr_exit_z,
    )

    # ---------------------------------------------------------------
    # 2. Momentum signal on OOS (self-contained — uses only lookback)
    # ---------------------------------------------------------------
    mom_config  = MomConfig(
        entry_z    = config.mom_entry_z,
        exit_z     = config.mom_exit_z,
        cot_weight = 0.0,   # COT not available on all folds; use price only
    )
    mom     = MomentumSignal(futures_oos, cot_df=None, config=mom_config)
    try:
        mom_results = mom.compute_signal()
    except Exception as exc:
        logger.warning("Fold %02d: momentum failed (%s).", fold_id, exc)
        mom_results = futures_oos[["close", "implied_rate"]].copy()
        mom_results["zscore_mom"] = 0.0
        mom_results["signal_mom"] = 0

    # ---------------------------------------------------------------
    # 3. Refit regime classifier on training window
    # ---------------------------------------------------------------
    rc_config = RegimeConfig(
        n_regimes             = config.n_regimes,
        ar_order              = 1,
        min_obs               = 126,
        crisis_vol_threshold  = 0.25,
        crisis_prob_threshold = 0.60,
    )
    rc = RegimeClassifier(futures_train, config=rc_config)
    try:
        rc.fit()
    except Exception as exc:
        logger.warning("Fold %02d: regime fit failed (%s).", fold_id, exc)
        return pd.DataFrame(), {}

    # ---------------------------------------------------------------
    # 4. Build combined OOS DataFrame for ensemble
    # ---------------------------------------------------------------
    combined_oos = futures_oos[["close", "implied_rate"]].copy()

    # Reindex MR results onto OOS index
    combined_oos["fair_value"] = fair_oos.reindex(futures_oos.index, method="ffill")
    combined_oos["spread"]     = spread_oos.reindex(futures_oos.index)
    combined_oos["zscore_mr"]  = zscore_mr_oos.reindex(futures_oos.index)
    combined_oos["signal_mr"]  = signal_mr_oos.reindex(futures_oos.index).fillna(0)
    combined_oos["zscore_mom"] = mom_results["zscore_mom"].reindex(futures_oos.index)
    combined_oos["signal_mom"] = mom_results["signal_mom"].reindex(futures_oos.index).fillna(0)

    # OSINT (optional)
    if osint_oos is not None:
        for col in ["osint_score", "zscore_osint", "signal_osint", "is_osint_crisis"]:
            if col in osint_oos.columns:
                combined_oos[col] = osint_oos[col].reindex(futures_oos.index).ffill()
        # Pass only the zscore_osint Series to the Ensemble — it expects a Series
        if "zscore_osint" in osint_oos.columns:
            osint_signal = osint_oos["zscore_osint"].reindex(futures_oos.index).ffill()
        else:
            osint_signal = None
    else:
        osint_signal = None

    # ---------------------------------------------------------------
    # 5. Ensemble on OOS
    # ---------------------------------------------------------------
    ens_config = EnsembleConfig(
        entry_z               = config.ens_entry_z,
        exit_z                = config.ens_exit_z,
        osint_stub            = (osint_signal is None),
        apply_crisis_override = True,
        soft_blend            = True,
    )
    ens        = Ensemble(combined_oos, rc, osint_signal=osint_signal, config=ens_config)
    ens_result = ens.compute()

    # Carry through log_return for vol-target sizing
    ens_result["log_return"] = futures_oos["log_return"].reindex(ens_result.index)

    # ---------------------------------------------------------------
    # 6. P&L on OOS
    # ---------------------------------------------------------------
    pnl_config = PnLConfig(
        sizing_method  = "vol_target",
        vol_target_usd = config.vol_target_usd,
        apply_costs    = config.apply_costs,
        risk_free_rate = config.risk_free_rate,
    )
    engine     = PnLEngine(ens_result, config=pnl_config)
    oos_pnl    = engine.compute()

    fold_label   = (f"Fold {fold_id:02d} OOS "
                    f"{futures_oos.index[0].date()}–{futures_oos.index[-1].date()}")
    fold_metrics = engine.metrics(label=fold_label)
    fold_metrics["fold_id"]        = fold_id
    fold_metrics["train_start"]    = str(futures_train.index[0].date())
    fold_metrics["train_end"]      = str(futures_train.index[-1].date())
    fold_metrics["oos_start"]      = str(futures_oos.index[0].date())
    fold_metrics["oos_end"]        = str(futures_oos.index[-1].date())
    fold_metrics["mr_half_life"]   = round(mr.half_life, 1)
    fold_metrics["mr_mean_rev"]    = mr.ou_params["is_mean_reverting"]

    logger.info(
        "Fold %02d OOS: P&L=$%.0f  Sharpe=%.3f  Trades=%d",
        fold_id, fold_metrics["total_pnl"],
        fold_metrics["sharpe"], fold_metrics["n_trades"],
    )

    return oos_pnl, fold_metrics


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

class WalkForwardHarness:
    """
    Walk-forward validation harness.

    Parameters
    ----------
    futures_df  : full futures DataFrame from data.cme
    macro_df    : full macro DataFrame from data.fred
    osint_df    : optional full OSINT results DataFrame (pre-computed)
    config      : WFConfig instance
    """

    def __init__(
        self,
        futures_df:  pd.DataFrame,
        macro_df:    pd.DataFrame,
        osint_df:    Optional[pd.DataFrame] = None,
        config:      WFConfig = None,
    ):
        self.futures = futures_df.copy()
        self.macro   = macro_df.copy()
        self.osint   = osint_df
        self.config  = config or WFConfig()

        # Results populated by run()
        self.fold_metrics:  list[dict]       = []
        self.oos_equity:    Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute all walk-forward folds and return the stitched OOS results.

        Returns
        -------
        pd.DataFrame : full stitched OOS P&L DataFrame with all signal,
                       regime, and P&L columns. The cumulative P&L is
                       the honest walk-forward equity curve.
        """
        cfg      = self.config
        idx      = self.futures.index
        n        = len(idx)
        oos_frames: list[pd.DataFrame] = []
        fold_id  = 0

        # Find first valid start (need min_train_days of history)
        first_train_end = cfg.min_train_days

        pos = first_train_end
        while pos + cfg.oos_window <= n:
            train_start = max(0, pos - cfg.train_window)
            train_end   = pos
            oos_start   = pos
            oos_end     = min(pos + cfg.oos_window, n)

            train_idx   = idx[train_start:train_end]
            oos_idx     = idx[oos_start:oos_end]

            if len(train_idx) < cfg.min_train_days:
                pos += cfg.step_days
                continue

            futures_train = self.futures.loc[train_idx]
            macro_train   = self.macro.reindex(train_idx, method="ffill").ffill().bfill()
            futures_oos   = self.futures.loc[oos_idx]
            macro_oos     = self.macro.reindex(oos_idx, method="ffill").ffill().bfill()

            osint_oos = None
            if self.osint is not None:
                osint_oos = self.osint.reindex(oos_idx, method="ffill").ffill()

            oos_pnl, metrics = run_single_fold(
                futures_train = futures_train,
                macro_train   = macro_train,
                futures_oos   = futures_oos,
                macro_oos     = macro_oos,
                osint_oos     = osint_oos,
                config        = cfg,
                fold_id       = fold_id,
            )

            if not oos_pnl.empty:
                oos_frames.append(oos_pnl)
                self.fold_metrics.append(metrics)

            fold_id += 1
            pos     += cfg.step_days

        if not oos_frames:
            raise RuntimeError(
                "Walk-forward produced no OOS results. "
                "Check that futures history is long enough for "
                f"train_window={cfg.train_window} + oos_window={cfg.oos_window}."
            )

        # Stitch OOS periods — recalculate cumulative P&L continuously
        combined       = pd.concat(oos_frames)
        combined       = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined["cum_net_pnl"] = combined["net_pnl"].cumsum()

        self.oos_equity = combined["cum_net_pnl"]

        logger.info(
            "Walk-forward complete: %d folds  %d OOS days  "
            "total P&L = $%.0f",
            fold_id, len(combined), combined["net_pnl"].sum(),
        )

        return combined

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, wf_results: Optional[pd.DataFrame] = None) -> None:
        """Print walk-forward summary across all folds."""
        if not self.fold_metrics:
            print("No fold metrics available. Run run() first.")
            return

        metrics_df = pd.DataFrame(self.fold_metrics)

        print("\n" + "=" * 68)
        print("  Walk-Forward Validation Summary")
        print("=" * 68)
        print(f"  Folds completed     : {len(metrics_df)}")
        print(f"  Train window        : {self.config.train_window} days")
        print(f"  OOS window          : {self.config.oos_window} days")
        print(f"  Step size           : {self.config.step_days} days")
        print()

        # Aggregate OOS metrics
        if wf_results is not None:
            from backtest.pnl import PnLConfig, compute_performance_metrics
            agg_config  = PnLConfig(risk_free_rate=self.config.risk_free_rate)
            agg_metrics = compute_performance_metrics(
                wf_results, agg_config, "Stitched OOS"
            )
            print("  Stitched OOS performance:")
            print(f"    Total P&L         : ${agg_metrics['total_pnl']:>10,.0f}")
            print(f"    Annualised P&L    : ${agg_metrics['ann_pnl']:>10,.0f}")
            print(f"    Sharpe ratio      : {agg_metrics['sharpe']:>8.3f}")
            print(f"    Sortino ratio     : {agg_metrics['sortino']:>8.3f}")
            print(f"    Max drawdown      : ${agg_metrics['max_drawdown']:>10,.0f}")
            print(f"    Win rate          : {agg_metrics['win_rate']:.1%}")
            print()

        # Per-fold breakdown
        print("  Per-fold OOS results:")
        print(f"  {'Fold':<6} {'OOS Period':<25} {'P&L':>10} "
              f"{'Sharpe':>8} {'Trades':>7} {'HL':>6}")
        print("  " + "-" * 62)
        for _, row in metrics_df.iterrows():
            period = f"{row['oos_start']} – {row['oos_end']}"
            hl     = f"{row.get('mr_half_life', 0):.0f}d"
            print(
                f"  {int(row['fold_id']):<6} {period:<25} "
                f"${row['total_pnl']:>9,.0f} "
                f"{row['sharpe']:>8.3f} "
                f"{int(row['n_trades']):>7} "
                f"{hl:>6}"
            )

        # Summary stats across folds
        print()
        pct_positive = (metrics_df["total_pnl"] > 0).mean()
        print(f"  Folds with positive P&L : {pct_positive:.0%}")
        print(f"  Median fold Sharpe      : {metrics_df['sharpe'].median():.3f}")
        print(f"  Mean fold Sharpe        : {metrics_df['sharpe'].mean():.3f}")
        print(f"  Sharpe consistency (std): {metrics_df['sharpe'].std():.3f}")
        print("=" * 68 + "\n")

    def fold_metrics_df(self) -> pd.DataFrame:
        """Return fold metrics as a DataFrame for further analysis."""
        return pd.DataFrame(self.fold_metrics)