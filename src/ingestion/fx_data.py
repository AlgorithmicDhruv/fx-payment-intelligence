"""
src/ingestion/fx_data.py

Pulls daily FX spot rates from two sources:
  1. FRED (Federal Reserve Economic Data) — authoritative, clean daily series
  2. yfinance — fallback + intraday capability

Merges, validates, and saves to data/raw/.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.settings import (
    FRED_API_KEY, FX_PAIRS, FRED_START_DATE, FRED_END_DATE,
    DATA_RAW, DATA_PROCESSED
)


def _pull_from_fred(pair: str, config: dict) -> pd.Series:
    """Pull FX series from FRED. Returns a daily pd.Series of spot rates."""
    try:
        from fredapi import Fred
        if not FRED_API_KEY:
            raise ValueError("FRED_API_KEY not set in .env")
        fred = Fred(api_key=FRED_API_KEY)
        end = FRED_END_DATE or datetime.today().strftime("%Y-%m-%d")
        series = fred.get_series(
            config["fred_series"],
            observation_start=FRED_START_DATE,
            observation_end=end
        )
        series.name = pair
        # FRED DEXUSEU is USD per EUR (inverted vs market convention)
        # invert=True means FRED gives USD/foreign -> we want foreign/USD convention
        # Actually we keep USD as quote currency throughout for consistency
        series = series.dropna()
        logger.info(f"FRED: pulled {len(series)} rows for {pair} ({config['fred_series']})")
        return series
    except Exception as e:
        logger.warning(f"FRED pull failed for {pair}: {e}")
        return pd.Series(dtype=float, name=pair)


def _pull_from_yfinance(pair: str, config: dict) -> pd.Series:
    """Fallback: pull from yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(config["yf_ticker"])
        df = ticker.history(period="10y", interval="1d")
        series = df["Close"].rename(pair)
        series.index = pd.to_datetime(series.index).tz_localize(None)
        logger.info(f"yfinance: pulled {len(series)} rows for {pair}")
        return series
    except Exception as e:
        logger.warning(f"yfinance pull failed for {pair}: {e}")
        return pd.Series(dtype=float, name=pair)


def pull_fx_data(pairs: dict = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Main entry point. Pulls all FX pairs, merges into a single DataFrame.
    Saves to data/raw/fx_daily.parquet.

    Args:
        pairs: dict of pair configs (defaults to settings.FX_PAIRS)
        force_refresh: re-pull even if cached file exists

    Returns:
        DataFrame with columns = FX pair names, DatetimeIndex (business days)
    """
    pairs = pairs or FX_PAIRS
    cache_path = DATA_RAW / "fx_daily.parquet"

    if cache_path.exists() and not force_refresh:
        logger.info(f"Loading cached FX data from {cache_path}")
        df = pd.read_parquet(cache_path)
        logger.info(f"Cache hit: {df.shape[0]} rows, {df.shape[1]} pairs, "
                    f"from {df.index.min().date()} to {df.index.max().date()}")
        return df

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    series_list = []

    for pair, config in pairs.items():
        logger.info(f"Pulling {pair}...")
        series = _pull_from_fred(pair, config)
        if series.empty:
            logger.info(f"Falling back to yfinance for {pair}")
            series = _pull_from_yfinance(pair, config)
        if not series.empty:
            series_list.append(series)

    if not series_list:
        raise RuntimeError("Failed to pull any FX data. Check API keys and network.")

    df = pd.concat(series_list, axis=1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()

    # Forward-fill weekends/holidays (max 3 days), then drop remaining NaN
    df = df.ffill(limit=3).dropna()

    # Basic sanity checks
    for col in df.columns:
        pct_missing = df[col].isna().mean()
        if pct_missing > 0.05:
            logger.warning(f"{col}: {pct_missing:.1%} missing after ffill")

    df.to_parquet(cache_path)
    logger.success(f"Saved FX data to {cache_path}: {df.shape}")
    return df


def pull_macro_indicators() -> pd.DataFrame:
    """
    Pull macro indicators from FRED that inform FX movements:
    - US 10Y Treasury yield (DGS10)
    - EUR 10Y yield proxy (via German Bund: not on FRED, use spread)
    - VIX (VIXCLS) — risk sentiment
    - US CPI YoY (CPIAUCSL)
    - Fed Funds Rate (FEDFUNDS)
    """
    if not FRED_API_KEY:
        logger.warning("No FRED API key — skipping macro indicators")
        return pd.DataFrame()

    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)

    macro_series = {
        "us_10y":      "DGS10",
        "vix":         "VIXCLS",
        "fed_funds":   "FEDFUNDS",
        "us_cpi":      "CPIAUCSL",
        "us_m2":       "M2SL",
    }

    cache_path = DATA_RAW / "macro_daily.parquet"

    frames = []
    for name, series_id in macro_series.items():
        try:
            s = fred.get_series(series_id, observation_start=FRED_START_DATE)
            s.name = name
            frames.append(s)
            logger.info(f"Macro: pulled {name} ({series_id}), {len(s)} obs")
        except Exception as e:
            logger.warning(f"Could not pull {name} ({series_id}): {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index().ffill(limit=5)

    df.to_parquet(cache_path)
    logger.success(f"Saved macro data: {df.shape}")
    return df


if __name__ == "__main__":
    logger.info("Running FX data ingestion...")
    fx_df = pull_fx_data(force_refresh=True)
    print(fx_df.tail())
    print(fx_df.describe())

    macro_df = pull_macro_indicators()
    if not macro_df.empty:
        print(macro_df.tail())
