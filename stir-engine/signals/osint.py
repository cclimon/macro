"""
signals/osint.py
----------------
OSINT / news factor signal for the STIR engine.

Pipeline
--------
1. Ingest    — fetch raw events from GDELT and Fed/Reuters RSS feeds
2. Classify  — LLM call (Claude claude-sonnet-4-20250514) maps each headline to:
               event_class, direction, severity_tier, confidence, reasoning
3. Impulse   — severity_tier × confidence × direction_sign → signed impact
4. Decay     — double-exponential decay accumulates impulses into a running score
               S(t) = Σ [ A_i·exp(-λ_fast·Δt) + B_i·exp(-λ_slow·Δt) ]
5. Signal    — score normalised to Z-score → {-1, 0, +1}
6. Override  — if |score| > threshold OR a crisis_override event fired,
               set is_osint_crisis=True for the ensemble circuit breaker

Backends
--------
  "live"   : GDELT API + RSS polling (production / scheduled runner)
  "manual" : event log loaded from a local CSV (backtesting / prototype)

Dependencies
------------
    pip install anthropic requests feedparser pandas numpy

Usage
-----
    from signals.osint import OsintSignal, OsintConfig

    osint = OsintSignal(config=OsintConfig(backend="manual"))
    osint.load_manual_events("data/manual_events.csv")   # see schema below
    results = osint.compute_signal(date_index=futures_df.index)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OsintConfig:
    """
    Tunable parameters for the OSINT signal.
    """
    # Data backend: "live" (GDELT + RSS) or "manual" (CSV event log)
    backend:               str   = "manual"

    # GDELT: max articles per query window
    gdelt_max_results:     int   = 20

    # GDELT: search terms for STIR-relevant events
    gdelt_query:           str   = (
        '"Federal Reserve" OR "FOMC" OR "interest rate" OR '
        '"inflation" OR "Strait of Hormuz" OR "oil supply" OR '
        '"nonfarm payroll" OR "CPI" OR "PCE"'
    )

    # LLM: model to use for classification
    llm_model:             str   = "claude-sonnet-4-20250514"

    # LLM: temperature (0 = deterministic, good for classification)
    llm_temperature:       float = 0.0

    # LLM: minimum confidence score to include an event (0–1)
    min_confidence:        float = 0.40

    # Z-score normalisation window for the final score series (days)
    zscore_window:         int   = 63

    # Signal entry/exit thresholds
    entry_z:               float = 1.0    # lower than MR/MOM — OSINT fires more readily
    exit_z:                float = 0.3

    # Deduplication: ignore events within this many hours of an identical headline
    dedup_window_hours:    int   = 6

    # RSS poll interval (seconds) — used by the real-time runner
    rss_poll_interval:     int   = 300    # 5 minutes


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a macro fixed income analyst specialising in
US short-term interest rate (STIR) futures. Your task is to classify a
news headline or event and estimate its directional impact on SOFR or
Fed Funds futures prices.

Return ONLY a JSON object with exactly these fields:
{
  "event_class":   one of ["geopolitical_shock", "cb_communication",
                            "scheduled_release", "financial_stress", "noise"],
  "direction":     one of ["hawkish", "dovish", "neutral"],
  "severity_tier": integer 1-5 (1=noise, 5=structural shock),
  "confidence":    float 0.0-1.0 (your certainty in this classification),
  "reasoning":     string, max 2 sentences explaining the rate impact
}

Direction conventions:
  hawkish  → implies higher rates → futures price FALLS  → signal = -1
  dovish   → implies lower rates  → futures price RISES  → signal = +1
  neutral  → no clear rate implication                   → signal =  0

Severity tiers:
  1 = background noise, no market impact expected
  2 = moderate, minor repricing likely
  3 = significant, clear market-moving event
  4 = major, e.g. surprise Fed action or severe data miss
  5 = structural shock, e.g. war, pandemic, financial system stress

Be conservative with severity — tier 4-5 should be rare.
Return only the JSON, no preamble or markdown fences."""


def classify_event_llm(
    headline: str,
    source:   str = "",
    context:  str = "",
    config:   OsintConfig = None,
) -> dict:
    """
    Send a headline to Claude for event classification.

    Parameters
    ----------
    headline : raw news headline or event description
    source   : news source (e.g. "Reuters", "Fed Reserve")
    context  : optional additional context (article snippet, max 200 chars)
    config   : OsintConfig instance

    Returns
    -------
    dict with keys: event_class, direction, severity_tier, confidence, reasoning
    On failure, returns a 'noise' classification with confidence=0.
    """
    if config is None:
        config = OsintConfig()

    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        )

    from config import ANTHROPIC_API_KEY
    if ANTHROPIC_API_KEY == "YOUR_ANTHROPIC_API_KEY_HERE":
        raise ValueError(
            "Anthropic API key not set. "
            "Set the ANTHROPIC_API_KEY environment variable."
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_content = f"Headline: {headline}"
    if source:
        user_content += f"\nSource: {source}"
    if context:
        user_content += f"\nContext: {context[:200]}"

    try:
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=256,
            temperature=config.llm_temperature,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if the model adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)

        # Validate required fields
        required = {"event_class", "direction", "severity_tier",
                    "confidence", "reasoning"}
        missing = required - set(result.keys())
        if missing:
            raise ValueError(f"LLM response missing fields: {missing}")

        # Clamp confidence to [0, 1]
        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
        result["severity_tier"] = int(result["severity_tier"])
        result["headline"] = headline
        result["source"]   = source

        logger.debug(
            "Classified: [%s|%s|tier=%d|conf=%.2f] %s",
            result["event_class"], result["direction"],
            result["severity_tier"], result["confidence"], headline[:60],
        )

        return result

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("LLM classification failed for '%s': %s", headline[:60], exc)
        return {
            "event_class":   "noise",
            "direction":     "neutral",
            "severity_tier": 1,
            "confidence":    0.0,
            "reasoning":     "Classification failed.",
            "headline":      headline,
            "source":        source,
        }


# ---------------------------------------------------------------------------
# GDELT ingestion
# ---------------------------------------------------------------------------

def fetch_gdelt_events(
    query:       str,
    start_dt:    datetime,
    end_dt:      datetime,
    max_results: int = 20,
) -> list[dict]:
    """
    Query the GDELT Document API for news articles matching the query.

    Returns a list of dicts with keys: headline, source, url, datetime
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests not installed. Run: pip install requests")

    from config import GDELT_API_BASE, GDELT_RATE_LIMIT

    params = {
        "query":      query,
        "mode":       "artlist",
        "maxrecords": max_results,
        "format":     "json",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   end_dt.strftime("%Y%m%d%H%M%S"),
        "sort":       "DateDesc",
    }

    logger.info("GDELT query: %s  [%s → %s]", query[:60], start_dt.date(), end_dt.date())

    time.sleep(GDELT_RATE_LIMIT)  # polite rate limiting

    try:
        resp = requests.get(GDELT_API_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("GDELT fetch failed: %s", exc)
        return []

    articles = data.get("articles", [])
    events = []
    for art in articles:
        events.append({
            "datetime": art.get("seendate", ""),
            "headline": art.get("title", ""),
            "source":   art.get("domain", ""),
            "url":      art.get("url", ""),
        })

    logger.info("GDELT returned %d articles", len(events))
    return events


# ---------------------------------------------------------------------------
# RSS ingestion
# ---------------------------------------------------------------------------

def fetch_rss_events(feeds: dict[str, str] = None) -> list[dict]:
    """
    Fetch recent items from RSS feeds (Fed communications, Reuters economy).

    Parameters
    ----------
    feeds : dict mapping feed_name → url. Defaults to config.OSINT_RSS_FEEDS.

    Returns
    -------
    list of dicts with keys: datetime, headline, source, url
    """
    try:
        import feedparser
    except ImportError:
        raise ImportError("feedparser not installed. Run: pip install feedparser")

    if feeds is None:
        from config import OSINT_RSS_FEEDS
        feeds = OSINT_RSS_FEEDS

    events = []
    for feed_name, url in feeds.items():
        logger.info("Fetching RSS: %s", feed_name)
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:10]:  # cap at 10 per feed
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    dt = datetime(*published[:6])
                else:
                    dt = datetime.utcnow()

                events.append({
                    "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "headline": entry.get("title", ""),
                    "source":   feed_name,
                    "url":      entry.get("link", ""),
                })
        except Exception as exc:
            logger.warning("RSS fetch failed for %s: %s", feed_name, exc)

    logger.info("RSS fetched %d total items", len(events))
    return events


# ---------------------------------------------------------------------------
# Impulse and decay engine
# ---------------------------------------------------------------------------

def compute_direction_sign(direction: str) -> int:
    """
    Convert LLM direction string to numeric sign for futures signal.
    Dovish → lower rates → long futures → +1
    Hawkish → higher rates → short futures → -1
    """
    return {"dovish": 1, "hawkish": -1, "neutral": 0}.get(direction, 0)


def compute_initial_impulse(
    severity_tier: int,
    confidence:    float,
    direction:     str,
    event_class:   str,
) -> float:
    """
    Compute the signed initial impulse magnitude for a single event.

    impulse = base_magnitude(tier) × confidence × direction_sign
    Capped at max_magnitude for the event class.

    Parameters
    ----------
    severity_tier : 1-5 integer from LLM classification
    confidence    : 0-1 float from LLM classification
    direction     : "dovish", "hawkish", or "neutral"
    event_class   : taxonomy key from OSINT_EVENT_TAXONOMY

    Returns
    -------
    float: signed impulse (negative = hawkish, positive = dovish)
    """
    from config import OSINT_SEVERITY_TIERS, OSINT_EVENT_TAXONOMY

    base       = OSINT_SEVERITY_TIERS.get(severity_tier, 0.25)
    sign       = compute_direction_sign(direction)
    tax        = OSINT_EVENT_TAXONOMY.get(event_class, OSINT_EVENT_TAXONOMY["noise"])
    max_mag    = tax["max_magnitude"]

    impulse = base * confidence * sign
    impulse = max(-max_mag, min(max_mag, impulse))   # clip to class ceiling

    return round(impulse, 4)


def compute_decayed_score(
    events_df:    pd.DataFrame,
    date_index:   pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Accumulate all event impulses onto a daily time series with
    double-exponential decay.

    For each event i at time t_i:
        contribution(t) = A_i × exp(-λ_fast × Δt)
                        + B_i × exp(-λ_slow × Δt)
    where:
        A_i = impulse_i × weight_fast
        B_i = impulse_i × (1 - weight_fast)
        λ_fast = ln(2) / half_life_fast  (converts half-life to decay rate)
        λ_slow = ln(2) / half_life_slow
        Δt = (t - t_i) in calendar days (≥ 0)

    The total score S(t) = Σ_i contribution_i(t)

    Parameters
    ----------
    events_df  : DataFrame with columns:
                   event_date   (datetime),  event_class (str),
                   impulse      (float),     is_crisis_override (bool)
    date_index : DatetimeIndex for the output series (business days)

    Returns
    -------
    pd.DataFrame with columns:
        osint_score         : cumulative decayed impulse sum
        osint_score_fast    : fast-decay component only (immediate repricing)
        osint_score_slow    : slow-decay component only (uncertainty premium)
        is_osint_crisis     : bool — True if crisis override is active
    """
    from config import OSINT_EVENT_TAXONOMY, OSINT_CRISIS_SCORE_THRESHOLD

    score_fast  = np.zeros(len(date_index))
    score_slow  = np.zeros(len(date_index))
    crisis_flag = np.zeros(len(date_index), dtype=bool)

    if events_df.empty:
        logger.warning("No events to decay — returning zero score series.")
        return pd.DataFrame({
            "osint_score":      score_fast,
            "osint_score_fast": score_fast,
            "osint_score_slow": score_slow,
            "is_osint_crisis":  crisis_flag,
        }, index=date_index)

    dates_float = np.array([d.timestamp() for d in date_index])
    secs_per_day = 86_400.0

    for _, ev in events_df.iterrows():
        ev_class  = ev["event_class"]
        impulse   = float(ev["impulse"])
        ev_dt     = pd.Timestamp(ev["event_date"])
        is_crisis = bool(ev.get("is_crisis_override", False))

        if impulse == 0.0:
            continue

        tax = OSINT_EVENT_TAXONOMY.get(ev_class, OSINT_EVENT_TAXONOMY["noise"])
        wf  = tax["weight_fast"]
        lf  = np.log(2) / tax["half_life_fast"]   # fast decay rate (per day)
        ls  = np.log(2) / tax["half_life_slow"]   # slow decay rate (per day)

        ev_ts  = ev_dt.timestamp()
        dt_arr = (dates_float - ev_ts) / secs_per_day  # Δt in days

        # Only apply impulse at t ≥ event_date (causal — no lookahead)
        mask = dt_arr >= 0

        A = impulse * wf
        B = impulse * (1.0 - wf)

        score_fast[mask]  += A * np.exp(-lf * dt_arr[mask])
        score_slow[mask]  += B * np.exp(-ls * dt_arr[mask])

        # Crisis override: flag all dates from event onward while
        # the slow-decay component retains > 20% of its initial magnitude
        if is_crisis:
            decay_remaining = np.exp(-ls * dt_arr[mask])
            crisis_mask     = np.where(mask)[0][decay_remaining > 0.20]
            crisis_flag[crisis_mask] = True

    total_score = score_fast + score_slow

    # Also flag dates where the total score exceeds the threshold
    crisis_flag |= (np.abs(total_score) > OSINT_CRISIS_SCORE_THRESHOLD)

    return pd.DataFrame({
        "osint_score":      total_score,
        "osint_score_fast": score_fast,
        "osint_score_slow": score_slow,
        "is_osint_crisis":  crisis_flag,
    }, index=date_index)


# ---------------------------------------------------------------------------
# Manual event log (prototype / backtesting)
# ---------------------------------------------------------------------------

_MANUAL_SCHEMA = [
    "event_date",          # YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
    "headline",            # raw headline string
    "source",              # e.g. "Reuters", "Federal Reserve", "Manual"
    "event_class",         # taxonomy key
    "direction",           # "dovish", "hawkish", "neutral"
    "severity_tier",       # 1-5
    "confidence",          # 0.0-1.0
    "is_crisis_override",  # True/False
    "notes",               # optional free text
]


def create_manual_events_template(path: str = "data/manual_events.csv") -> None:
    """
    Write a blank manual events CSV with the correct schema and a few
    pre-populated example rows for context.
    """
    examples = [
        {
            "event_date":         "2025-02-15",
            "headline":           "US and Israeli forces strike Iranian nuclear facilities",
            "source":             "Reuters",
            "event_class":        "geopolitical_shock",
            "direction":          "dovish",
            "severity_tier":      5,
            "confidence":         0.85,
            "is_crisis_override": True,
            "notes":              "Strait of Hormuz risk; oil supply disruption; growth negative",
        },
        {
            "event_date":         "2026-03-04",
            "headline":           "Iran declares Strait of Hormuz closed to commercial shipping",
            "source":             "Reuters",
            "event_class":        "geopolitical_shock",
            "direction":          "dovish",
            "severity_tier":      5,
            "confidence":         0.95,
            "is_crisis_override": True,
            "notes":              "Formal closure; oil supply shock; stagflation risk; rate cut pressure builds",
        },
        {
            "event_date":         "2026-03-19",
            "headline":           "Fed holds rates at 3.50-3.75%, flags two-sided risks",
            "source":             "Federal Reserve",
            "event_class":        "cb_communication",
            "direction":          "neutral",
            "severity_tier":      3,
            "confidence":         0.90,
            "is_crisis_override": False,
            "notes":              "Hold unanimous; two-sided language confirms paralysis",
        },
        {
            "event_date":         "2026-04-07",
            "headline":           "US-Iran ceasefire announced; Hormuz reopening expected within weeks",
            "source":             "Reuters",
            "event_class":        "geopolitical_shock",
            "direction":          "hawkish",
            "severity_tier":      4,
            "confidence":         0.70,
            "is_crisis_override": False,
            "notes":              "Partial de-escalation; hawkish because removes rate cut pressure; confidence lower due to fragility",
        },
        {
            "event_date":         "2026-04-10",
            "headline":           "US and Iran exchange fire in Strait despite ceasefire",
            "source":             "Reuters",
            "event_class":        "geopolitical_shock",
            "direction":          "dovish",
            "severity_tier":      4,
            "confidence":         0.80,
            "is_crisis_override": True,
            "notes":              "Ceasefire violated; partially cancels the April 7 hawkish impulse",
        },
    ]

    df = pd.DataFrame(examples, columns=_MANUAL_SCHEMA)

    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Manual events template written → %s  (%d rows)", path, len(df))
    print(f"Template created at {path}. Edit it to add your own events.")


def load_manual_events(path: str) -> pd.DataFrame:
    """
    Load and validate a manual events CSV.

    Returns
    -------
    pd.DataFrame with columns matching _MANUAL_SCHEMA, plus an 'impulse'
    column computed from severity_tier, confidence, and direction.
    """
    from pathlib import Path
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Manual events file not found: {path}\n"
            "Run create_manual_events_template() to generate a template."
        )

    df = pd.read_csv(path)
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Compute impulse for each row
    df["impulse"] = df.apply(
        lambda r: compute_initial_impulse(
            severity_tier=int(r["severity_tier"]),
            confidence=float(r["confidence"]),
            direction=str(r["direction"]),
            event_class=str(r["event_class"]),
        ),
        axis=1,
    )

    df["is_crisis_override"] = df["is_crisis_override"].astype(bool)

    logger.info(
        "Manual events loaded: %d rows  impulse range [%.3f, %.3f]",
        len(df), df["impulse"].min(), df["impulse"].max(),
    )

    return df


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_events(
    events: list[dict],
    window_hours: int = 6,
) -> list[dict]:
    """
    Remove near-duplicate headlines within a time window.
    Uses simple headline prefix matching (first 40 chars).
    """
    seen: dict[str, datetime] = {}
    deduped = []

    for ev in events:
        key = ev.get("headline", "")[:40].lower().strip()
        ev_dt = pd.Timestamp(ev.get("datetime", datetime.utcnow())).to_pydatetime()

        if key in seen:
            if (ev_dt - seen[key]) < timedelta(hours=window_hours):
                continue   # duplicate within window — skip

        seen[key] = ev_dt
        deduped.append(ev)

    logger.debug("Dedup: %d → %d events", len(events), len(deduped))
    return deduped


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OsintSignal:
    """
    Full OSINT signal pipeline.

    Parameters
    ----------
    config : OsintConfig instance (optional, defaults used if not provided)
    """

    def __init__(self, config: OsintConfig = None):
        self.config     = config or OsintConfig()
        self.events_df: Optional[pd.DataFrame] = None    # classified events log
        self._score_df: Optional[pd.DataFrame] = None    # decayed score series

    # ------------------------------------------------------------------
    # Loading / ingestion
    # ------------------------------------------------------------------

    def load_manual_events(self, path: str) -> "OsintSignal":
        """Load a pre-classified manual event log (for backtesting / prototype)."""
        self.events_df = load_manual_events(path)
        return self

    def fetch_and_classify(
        self,
        start_dt: datetime,
        end_dt:   datetime,
    ) -> "OsintSignal":
        """
        Live backend: fetch events from GDELT + RSS and classify each
        with the LLM. Populates self.events_df.

        Parameters
        ----------
        start_dt : start of fetch window
        end_dt   : end of fetch window (typically now)
        """
        cfg = self.config

        raw_events: list[dict] = []

        # GDELT
        gdelt_events = fetch_gdelt_events(
            cfg.gdelt_query, start_dt, end_dt, cfg.gdelt_max_results
        )
        raw_events.extend(gdelt_events)

        # RSS
        rss_events = fetch_rss_events()
        raw_events.extend(rss_events)

        # Deduplicate
        raw_events = deduplicate_events(raw_events, cfg.dedup_window_hours)

        logger.info("Classifying %d events via LLM...", len(raw_events))

        records = []
        for ev in raw_events:
            classification = classify_event_llm(
                headline=ev.get("headline", ""),
                source=ev.get("source", ""),
                config=cfg,
            )

            if classification["confidence"] < cfg.min_confidence:
                logger.debug(
                    "Skipping low-confidence event (%.2f): %s",
                    classification["confidence"], ev.get("headline", "")[:60],
                )
                continue

            from config import OSINT_EVENT_TAXONOMY
            tax = OSINT_EVENT_TAXONOMY.get(
                classification["event_class"],
                OSINT_EVENT_TAXONOMY["noise"],
            )

            impulse = compute_initial_impulse(
                severity_tier=classification["severity_tier"],
                confidence=classification["confidence"],
                direction=classification["direction"],
                event_class=classification["event_class"],
            )

            records.append({
                "event_date":         pd.Timestamp(ev.get("datetime", datetime.utcnow())),
                "headline":           ev.get("headline", ""),
                "source":             ev.get("source", ""),
                "event_class":        classification["event_class"],
                "direction":          classification["direction"],
                "severity_tier":      classification["severity_tier"],
                "confidence":         classification["confidence"],
                "reasoning":          classification["reasoning"],
                "impulse":            impulse,
                "is_crisis_override": tax["crisis_override"] and
                                      classification["severity_tier"] >= 4,
                "notes":              "",
            })

        self.events_df = pd.DataFrame(records) if records else pd.DataFrame(
            columns=_MANUAL_SCHEMA + ["impulse", "reasoning"]
        )

        logger.info(
            "Classification complete: %d events  "
            "impulse range [%.3f, %.3f]",
            len(self.events_df),
            self.events_df["impulse"].min() if len(self.events_df) else 0,
            self.events_df["impulse"].max() if len(self.events_df) else 0,
        )

        return self

    # ------------------------------------------------------------------
    # Compute signal
    # ------------------------------------------------------------------

    def compute_signal(
        self,
        date_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Compute the full OSINT signal on the provided date spine.

        Returns
        -------
        pd.DataFrame with columns:
            osint_score       : cumulative decayed impulse (raw)
            osint_score_fast  : fast-decay component
            osint_score_slow  : slow-decay component
            zscore_osint      : rolling Z-score of osint_score
            signal_osint      : {-1, 0, +1}
            is_osint_crisis   : bool circuit breaker flag
        """
        if self.events_df is None:
            raise RuntimeError(
                "No events loaded. Call load_manual_events() or "
                "fetch_and_classify() first."
            )

        # Build decayed score series
        score_df = compute_decayed_score(self.events_df, date_index)
        self._score_df = score_df

        cfg = self.config

        # Z-score normalise for signal generation
        score = score_df["osint_score"]
        mu    = score.rolling(cfg.zscore_window, min_periods=10).mean()
        sigma = score.rolling(cfg.zscore_window, min_periods=10).std()
        zscore = ((score - mu) / sigma.replace(0, np.nan)).fillna(0.0)
        zscore.name = "zscore_osint"

        # Signal with hysteresis
        signal   = pd.Series(0, index=date_index, dtype=int, name="signal_osint")
        position = 0
        for i, z in enumerate(zscore):
            if position == 0:
                if z > cfg.entry_z:
                    position = 1
                elif z < -cfg.entry_z:
                    position = -1
            else:
                if abs(z) < cfg.exit_z:
                    position = 0
            signal.iloc[i] = position

        result = score_df.copy()
        result["zscore_osint"]  = zscore
        result["signal_osint"]  = signal

        long_  = (signal == 1).sum()
        short_ = (signal == -1).sum()
        crisis = score_df["is_osint_crisis"].sum()

        logger.info(
            "OSINT signal: %d long  %d short  %d flat  %d crisis-override days",
            long_, short_, len(date_index) - long_ - short_, crisis,
        )

        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def event_log(self) -> pd.DataFrame:
        """Return the classified event log sorted by date."""
        if self.events_df is None or self.events_df.empty:
            return pd.DataFrame()
        return self.events_df.sort_values("event_date").reset_index(drop=True)

    def summary(self, results: Optional[pd.DataFrame] = None) -> None:
        """Print a concise summary of the OSINT signal."""
        cfg = self.config
        print("\n" + "=" * 52)
        print("  OSINT Signal — Summary")
        print("=" * 52)
        print(f"  Backend             : {cfg.backend}")
        print(f"  LLM model           : {cfg.llm_model}")
        print(f"  Min confidence      : {cfg.min_confidence:.0%}")
        print(f"  Z-score window      : {cfg.zscore_window} days")
        print(f"  Entry Z threshold   : ±{cfg.entry_z}")
        print(f"  Exit  Z threshold   : ±{cfg.exit_z}")

        if self.events_df is not None:
            print(f"  Total events        : {len(self.events_df)}")
            if not self.events_df.empty:
                for cls, grp in self.events_df.groupby("event_class"):
                    print(f"    {cls:<24}: {len(grp)} events")

        if results is not None and "signal_osint" in results.columns:
            s = results["signal_osint"]
            c = results["is_osint_crisis"]
            print(f"  Long days           : {(s == 1).sum()}")
            print(f"  Short days          : {(s == -1).sum()}")
            print(f"  Flat days           : {(s == 0).sum()}")
            print(f"  Crisis override days: {c.sum()}")

            if "osint_score" in results.columns:
                score = results["osint_score"]
                print(f"  Score range         : [{score.min():.3f}, {score.max():.3f}]")
                print(f"  Current score       : {score.iloc[-1]:.3f}")

        print("=" * 52 + "\n")