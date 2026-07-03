# main.py
# EOD Orchestrator — fetches all data and builds signal DataFrames

import pandas as pd
import numpy as np
import logging
from datetime import datetime

from config.pairs import (
    SPOT_TICKERS,
    RATE_3M_TICKERS, OIS_TICKERS, CPI_TICKERS,
    PMI_TICKERS, POLICY_RATE_TICKERS,
    HIST_DAYS,
)
from data.bloomberg import (
    BloombergSession,
    fetch_spot_history,
    fetch_latest_rates,
    fetch_macro_latest,
    fetch_macro_history,
)
from signals.technical import build_technical_signals
from signals.carry import build_carry_signals
from signals.macro import build_macro_signals
from signals.positioning import POSITIONING_TICKERS, HIST_YEARS, build_positioning_tag
from data.cache import save_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_eod() -> dict:
    """
    Full EOD data pull and signal computation.
    Returns dict of DataFrames keyed by pillar.
    """
    logger.info("=== G10 FX Signal Dashboard — EOD Run %s ===", datetime.today().date())

    with BloombergSession() as bbg:

        # ── 1. Spot history ───────────────────────────────────────────────────
        logger.info("Fetching spot history …")
        spot_data = fetch_spot_history(
            bbg,
            list(SPOT_TICKERS.values()),
            days=HIST_DAYS,
        )
        spot_close = spot_data["close"]
        spot_high  = spot_data["high"]
        spot_low   = spot_data["low"]
        spot_df = spot_close   # alias used by carry, charts, and return dict

        # Latest spot
        # ── 2. Rates ──────────────────────────────────────────────────────────
        logger.info("Fetching 3m rates …")
        rates_3m = fetch_latest_rates(bbg, RATE_3M_TICKERS)

        logger.info("Fetching policy rates …")
        policy_rate = fetch_latest_rates(bbg, POLICY_RATE_TICKERS)

        # ── 4. CPI ────────────────────────────────────────────────────────────
        logger.info("Fetching CPI …")
        cpi_latest = fetch_macro_latest(bbg, CPI_TICKERS)
        cpi_hist = fetch_macro_history(bbg, CPI_TICKERS, days=HIST_DAYS, periodicity="MONTHLY")

        # ── 5. PMI ────────────────────────────────────────────────────────────
        logger.info("Fetching PMI …")
        pmi_latest = fetch_macro_latest(bbg, PMI_TICKERS)
        pmi_hist = fetch_macro_history(bbg, PMI_TICKERS, days=HIST_DAYS, periodicity="MONTHLY")

        # ── 6. Positioning ────────────────────────────────────────────────────
        logger.info("Fetching positioning data …")
        pos_days = HIST_YEARS * 365  # fetch_macro_history adds 1.5x buffer internally
        pos_data = {}
        for category, tickers in POSITIONING_TICKERS.items():
            hist = fetch_macro_history(bbg, tickers, days=pos_days, periodicity="DAILY")
            pos_data[category] = {ccy: hist[ccy] for ccy in tickers if ccy in hist.columns}

        # ── 7. Build signals ──────────────────────────────────────────────────
        logger.info("Building technical signals …")
        tech_df = build_technical_signals(spot_close, spot_high, spot_low)

        logger.info("Building carry signals …")
        carry_df = build_carry_signals(
            spot_df=spot_df,
            rates_3m=rates_3m,
            cpi_latest=cpi_latest,
        )

        logger.info("Building macro signals …")
        macro_df = build_macro_signals(
            pmi=pmi_latest,
            cpi_latest=cpi_latest,
            policy_rate=policy_rate,
            pmi_hist=pmi_hist,
            cpi_hist=cpi_hist,
        )

        logger.info("Building positioning signals …")
        positioning_df = build_positioning_tag(pos_data)

    logger.info("All signals built. Pairs: %d", len(tech_df))

    signals = {
        "spot": spot_df,
        "technical": tech_df,
        "carry": carry_df,
        "macro": macro_df,
        "positioning": positioning_df,
        "as_of": datetime.today(),
    }
    save_signals(signals)
    logger.info("Signals saved to cache.")
    return signals


if __name__ == "__main__":
    signals = run_eod()
    print("\n-- TECHNICAL --")
    print(signals["technical"].to_string())
    print("\n-- CARRY --")
    print(signals["carry"].to_string())
    print("\n-- MACRO --")
    print(signals["macro"].to_string())
    print("\n-- POSITIONING --")
    print(signals["positioning"].to_string())
