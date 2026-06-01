"""
main.py
-------
Orchestration entry point for the STIR engine.
Run this to fetch, validate, and cache all data for the first time.

Usage:
    python main.py
"""

import io
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LOG_LEVEL, LOG_FILE
from data.fred import fetch_all_macro, save_macro_cache
from data.cme  import (
    fetch_continuous_front,
    compute_realised_vol,
    compute_autocorrelation,
    save_futures_cache,
)

from signals.mean_reversion import MeanReversionSignal, MRConfig
from signals.momentum import MomentumSignal, MomConfig, fetch_cot_data
from regime.classifier import RegimeClassifier, RegimeConfig
from regime.ensemble import Ensemble, EnsembleConfig
from signals.osint import OsintSignal, OsintConfig, create_manual_events_template
from backtest.pnl import PnLEngine, PnLConfig
from backtest.walk_forward import WalkForwardHarness, WFConfig

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
 
# Wrap stdout.buffer so Unicode characters don't crash on Windows terminals
_stream = logging.StreamHandler(
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
)
_stream.setFormatter(_fmt)
 
_file = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file.setFormatter(_fmt)
 
logging.basicConfig(level=getattr(logging, LOG_LEVEL), handlers=[_stream, _file])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_data_pipeline():
    logger.info("=" * 60)
    logger.info("STIR Engine — data pipeline starting")
    logger.info("=" * 60)

    # --- Step 1: SOFR / FF futures ---
    logger.info("Step 1/2: Fetching SOFR futures (yfinance backend)")
    futures = fetch_continuous_front()

    # Attach regime-classifier observables
    futures["realised_vol"]  = compute_realised_vol(futures, window=20)
    futures["autocorr_1d"]   = compute_autocorrelation(futures, window=20, lag=1)

    save_futures_cache(futures)

    logger.info("Futures sample (last 5 rows):")
    logger.info("\n%s", futures[["close", "implied_rate", "realised_vol", "autocorr_1d"]].tail())

    # --- Step 2: FRED macro ---
    logger.info("Step 2/2: Fetching FRED macro series")
    macro = fetch_all_macro()
    save_macro_cache(macro)

    logger.info("Macro sample (last 5 rows):")
    logger.info("\n%s", macro[["fed_funds_effective", "core_cpi_yoy", "core_pce_yoy"]].tail())

    # --- Step 3: Mean reversion signal ---
    logger.info("Step 3: Fitting mean reversion signal (OU process)")
    config = MRConfig(
        fair_value_source = "fed_midpoint",
        entry_z           = 1.5,
        exit_z            = 0.5,
    )
    mr = MeanReversionSignal(futures, macro, config=config)
    mr.fit()
    mr.summary()
 
    results = mr.compute_signal()
    logger.info("Signal sample (last 10 rows):")
    logger.info(
        "\n%s",
        results[["close", "implied_rate", "fair_value", "spread", "zscore_mr", "signal_mr"]].tail(10)
    )

    logger.info("=" * 60)
    logger.info("Data pipeline complete. Caches written to data/cache/")
    logger.info("=" * 60)

    # --- Step 4: Momentum signal ---
    logger.info("Step 4: Computing momentum signal (price + COT)")
 
    # Attempt to fetch COT data; continue gracefully if unavailable
    cot_df = None
    try:
        cot_df = fetch_cot_data(use_cache=True)
        logger.info("COT data loaded: %d rows", len(cot_df))
    except Exception as exc:
        logger.warning("COT fetch failed (%s) — momentum will use price only.", exc)
 
    mom_config = MomConfig(
        fast_window  = 5,
        slow_window  = 21,
        vol_window   = 20,
        zscore_window= 63,
        entry_z      = 1.2,
        exit_z       = 0.4,
        cot_weight   = 0.30 if cot_df is not None else 0.0,
    )
 
    mom = MomentumSignal(futures, cot_df=cot_df, config=mom_config)
    mom_results = mom.compute_signal()
    mom.summary(mom_results)
 
    logger.info("Momentum signal sample (last 10 rows):")
    logger.info(
        "\n%s",
        mom_results[["close", "implied_rate", "fast_mom", "slow_mom",
                     "zscore_mom", "signal_mom"]].tail(10)
    )
 
    # --- Combine MR and momentum signals into a single DataFrame ---
    combined = results[["close", "implied_rate", "fair_value",
                         "spread", "zscore_mr", "signal_mr"]].copy()
    combined["zscore_mom"] = mom_results["zscore_mom"]
    combined["signal_mom"] = mom_results["signal_mom"]
    combined["cot_net"]    = mom_results["cot_net"]
 
    logger.info("=" * 60)
    logger.info("Days 3 & 4 complete. Both signals computed and combined.")
    logger.info("Combined columns: %s", list(combined.columns))
    logger.info("=" * 60)

    return futures, macro, combined

def run_regime_ensemble(futures, combined):
    """
    Day 5: Fit the Markov-switching regime classifier and compute the
    ensemble-weighted blended signal.
 
    Parameters
    ----------
    futures  : DataFrame from run_data_pipeline() with log_return,
               realised_vol_20d, autocorr_1d
    combined : DataFrame from run_data_pipeline() with zscore_mr,
               signal_mr, zscore_mom, signal_mom
 
    Returns
    -------
    ensemble_result : full output DataFrame with regime probs,
                      weights, blended_z, signal_ensemble, is_crisis
    rc              : fitted RegimeClassifier (inspect / reuse)
    ens             : Ensemble instance (for current_weights() queries)
    """
    logger.info("=" * 60)
    logger.info("Day 5: Regime classifier + ensemble")
    logger.info("=" * 60)
 
    # --- Step 5a: Fit Markov-switching regime classifier ---
    logger.info("Step 5a: Fitting Markov-switching AR(1) — 3 regimes")
 
    rc_config = RegimeConfig(
        n_regimes             = 3,
        ar_order              = 1,
        min_obs               = 126,
        refit_window          = None,    # expanding window for initial fit
        crisis_vol_threshold  = 0.25,   # 25% annualised vol -> hard override
        crisis_prob_threshold = 0.60,   # >60% crisis prob -> override
    )
 
    rc = RegimeClassifier(futures, config=rc_config)
    rc.fit()
    rc.summary()
 
    # --- Step 5b: Build ensemble ---
    logger.info("Step 5b: Computing regime-weighted ensemble signal")
 
    ens_config = EnsembleConfig(
        entry_z               = 1.3,
        exit_z                = 0.5,
        osint_stub            = True,   # OSINT not yet live -- weight redistributed
        apply_crisis_override = True,
        soft_blend            = True,
    )
 
    ens = Ensemble(combined, rc, osint_signal=None, config=ens_config)
    ensemble_result = ens.compute()
    ens.summary(ensemble_result)
 
    # --- Log current engine state ---
    cw = ens.current_weights(ensemble_result)
    logger.info(
        "Current engine state:\n"
        "  Date      : %s\n"
        "  Regime    : %s\n"
        "  Weights   : MR=%.2f  MOM=%.2f  OSINT=%.2f\n"
        "  Blended Z : %.4f\n"
        "  Signal    : %+d\n"
        "  Crisis    : %s",
        cw["date"], cw["regime"],
        cw["w_mr"], cw["w_mom"], cw["w_osint"],
        cw["blended_z"], cw["signal"], cw["is_crisis"],
    )
 
    # --- Sample output ---
    display_cols = [
        "close", "implied_rate",
        "prob_ranging", "prob_trending", "prob_crisis",
        "regime", "w_mr", "w_mom",
        "blended_z", "signal_ensemble", "is_crisis",
    ]
    available = [c for c in display_cols if c in ensemble_result.columns]
    logger.info(
        "Ensemble result sample (last 10 rows):\n%s",
        ensemble_result[available].tail(10),
    )
 
    logger.info("=" * 60)
    logger.info("Day 5 complete. Full engine pipeline operational.")
    logger.info("=" * 60)
 
    return ensemble_result, rc, ens

def run_osint_signal(date_index, events_path="data/manual_events.csv"):
    """
    OSINT signal pipeline.
 
    Loads the manual event log (prototype mode) and computes the
    cumulative decay score, Z-score, and circuit breaker flag on the
    provided daily date spine.
 
    Parameters
    ----------
    date_index  : pd.DatetimeIndex — business-day index from futures DataFrame
    events_path : path to the manual events CSV
 
    Returns
    -------
    osint_results : DataFrame with osint_score, zscore_osint,
                    signal_osint, is_osint_crisis
    osint         : OsintSignal instance (for .summary() and .event_log())
    """
    from pathlib import Path
 
    logger.info("=" * 60)
    logger.info("OSINT signal pipeline")
    logger.info("=" * 60)
 
    # Create template if it doesn't exist yet
    if not Path(events_path).exists():
        logger.info("No event log found — creating template at %s", events_path)
        create_manual_events_template(events_path)
 
    osint_config = OsintConfig(
        backend       = "manual",
        zscore_window = 63,
        entry_z       = 1.0,
        exit_z        = 0.3,
        min_confidence= 0.40,
    )
 
    osint = OsintSignal(config=osint_config)
    osint.load_manual_events(events_path)
    osint_results = osint.compute_signal(date_index)
    osint.summary(osint_results)
 
    logger.info("OSINT event log:\n%s", osint.event_log()[
        ["event_date", "event_class", "direction",
         "severity_tier", "impulse", "is_crisis_override"]
    ].to_string())
 
    logger.info("OSINT score sample (last 10 rows):\n%s",
        osint_results[["osint_score", "zscore_osint",
                        "signal_osint", "is_osint_crisis"]].tail(10))
 
    logger.info("=" * 60)
    logger.info("OSINT pipeline complete.")
    logger.info("=" * 60)
 
    return osint_results, osint

def run_backtest(futures, macro, osint_results=None):
    """
    Backtest pipeline: full P&L engine on the complete history, then
    walk-forward validation producing an honest OOS equity curve.
 
    Parameters
    ----------
    futures       : full futures DataFrame
    macro         : full macro DataFrame
    osint_results : optional OSINT signal DataFrame
 
    Returns
    -------
    wf_results : stitched OOS P&L DataFrame
    wf         : WalkForwardHarness instance (for .summary() and .fold_metrics_df())
    """
    logger.info("=" * 60)
    logger.info("Backtest pipeline")
    logger.info("=" * 60)
 
    wf_config = WFConfig(
        train_window   = 504,    # ~2 years
        oos_window     = 126,    # ~6 months
        step_days      = 63,     # ~3 months — 4 folds per year
        refit_regime   = True,
        refit_mr       = True,
        vol_target_usd = 5_000.0,
        apply_costs    = True,
        risk_free_rate = 0.045,
    )
 
    wf = WalkForwardHarness(
        futures_df = futures,
        macro_df   = macro,
        osint_df   = osint_results,
        config     = wf_config,
    )
 
    logger.info(
        "Running walk-forward: train=%dd  OOS=%dd  step=%dd",
        wf_config.train_window, wf_config.oos_window, wf_config.step_days,
    )
 
    wf_results = wf.run()
    wf.summary(wf_results)
 
    # Per-component attribution on stitched OOS
    pnl_cfg = PnLConfig(
        risk_free_rate = wf_config.risk_free_rate,
        apply_costs    = wf_config.apply_costs,
    )
    engine     = PnLEngine(wf_results, config=pnl_cfg)
    engine.compute()
    att = engine.attribution()
 
    logger.info("Component attribution (stitched OOS):")
    for comp, stats in att["by_component"].items():
        logger.info(
            "  %-20s P&L=$%.0f  Sharpe=%.3f  Hit=%.1f%%",
            comp, stats["total_pnl"], stats["ann_sharpe"],
            stats["hit_rate"] * 100,
        )
 
    if att["by_regime"]:
        logger.info("Regime attribution (stitched OOS):")
        for regime, stats in att["by_regime"].items():
            logger.info(
                "  %-20s P&L=$%.0f  (%d days  avg $%.0f/day)",
                regime, stats["total_pnl"], stats["n_days"], stats["mean_daily"],
            )
 
    logger.info("=" * 60)
    logger.info("Backtest complete.")
    logger.info("=" * 60)
 
    return wf_results, wf

if __name__ == "__main__":
    futures, macro, combined = run_data_pipeline()
    osint_results, osint = run_osint_signal(futures.index)
    ensemble_result, rc, ens = run_regime_ensemble(futures, combined)
    wf_results, wf = run_backtest(futures, macro, osint_results)