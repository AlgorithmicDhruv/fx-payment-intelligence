"""
src/features/engineer.py

Transforms raw FX price series into a rich feature matrix:
  - Log returns (multiple horizons)
  - Rolling volatility (realized, GARCH-fitted)
  - Momentum / mean-reversion signals
  - Regime labels (high vol / low vol / trending)
  - Macro spread features
  - Calendar effects

This is the feature set that feeds BOTH the forecasting model AND the RL agent.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.settings import (
    ROLLING_WINDOWS, VOLATILITY_WINDOW, REGIME_THRESHOLD,
    DATA_PROCESSED, GARCH_P, GARCH_Q, GARCH_DIST
)

TRADING_DAYS_PER_YEAR = 252


def compute_returns(prices: pd.Series, horizons: list = None) -> pd.DataFrame:
    """
    Compute log returns at multiple horizons.
    Log returns are preferred: additive, normally distributed, better for models.
    """
    horizons = horizons or [1, 2, 5, 10, 21]
    out = {}
    for h in horizons:
        out[f"log_ret_{h}d"] = np.log(prices / prices.shift(h))
    return pd.DataFrame(out, index=prices.index)


def compute_rolling_features(prices: pd.Series, windows: list = None) -> pd.DataFrame:
    """
    Rolling statistics: mean, std, z-score, skewness, momentum.
    """
    windows = windows or ROLLING_WINDOWS
    log_ret = np.log(prices / prices.shift(1))
    out = {}

    for w in windows:
        roll = log_ret.rolling(w)
        out[f"vol_{w}d"]      = roll.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        out[f"mean_ret_{w}d"] = roll.mean()
        out[f"skew_{w}d"]     = roll.skew()
        # Z-score of price relative to its rolling window
        price_roll = prices.rolling(w)
        out[f"zscore_{w}d"]   = (prices - price_roll.mean()) / price_roll.std().replace(0, np.nan)
        # Momentum: sign and magnitude
        out[f"mom_{w}d"]      = np.log(prices / prices.shift(w))

    # RSI (14-day) — mean-reversion indicator
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    out["rsi_14d"] = 100 - (100 / (1 + rs))

    # Bollinger Band position
    ma20 = prices.rolling(20).mean()
    std20 = prices.rolling(20).std()
    out["bb_position"] = (prices - ma20) / (2 * std20.replace(0, np.nan))

    return pd.DataFrame(out, index=prices.index)


def fit_garch_volatility(
    prices: pd.Series,
    p: int = GARCH_P,
    q: int = GARCH_Q,
    dist: str = GARCH_DIST
) -> pd.DataFrame:
    """
    Fit a GARCH(p,q) model to extract:
      - Conditional volatility forecast (annualized)
      - Standardized residuals (useful as features)

    Uses the `arch` library. Student-t distribution captures fat tails
    observed in FX returns — important for risk-aware optimization.
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch library not installed. Run: pip install arch")
        return pd.DataFrame(index=prices.index)

    log_ret = np.log(prices / prices.shift(1)).dropna() * 100  # scale for numerical stability

    logger.info(f"Fitting GARCH({p},{q}) with {dist} distribution on {len(log_ret)} observations...")
    model = arch_model(log_ret, vol="Garch", p=p, q=q, dist=dist)

    try:
        result = model.fit(disp="off", show_warning=False)
        cond_vol = result.conditional_volatility  # daily vol in return units
        std_resid = result.std_resid

        # Annualize: daily vol * sqrt(252), convert from % back to decimal
        ann_vol = (cond_vol / 100) * np.sqrt(TRADING_DAYS_PER_YEAR)

        out = pd.DataFrame({
            "garch_cond_vol":  ann_vol,
            "garch_std_resid": std_resid,
        }, index=log_ret.index)

        logger.success(f"GARCH fitted. Mean conditional vol: {ann_vol.mean():.3f}")
        return out

    except Exception as e:
        logger.warning(f"GARCH fitting failed: {e}. Returning realized vol instead.")
        realized_vol = log_ret.rolling(VOLATILITY_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR) / 100
        return pd.DataFrame({"garch_cond_vol": realized_vol}, index=log_ret.index)


def label_regimes(
    prices: pd.Series,
    garch_vol: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Label each trading day with a market regime:
      0 = Low vol / calm trending
      1 = Normal
      2 = High vol / stressed

    Regime is a key feature for the RL agent — it learns different
    execution strategies per regime (e.g., more aggressive in calm markets).
    """
    log_ret = np.log(prices / prices.shift(1))
    realized_vol = log_ret.rolling(VOLATILITY_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    vol_series = garch_vol if garch_vol is not None else realized_vol

    # Percentile-based thresholds (more robust than fixed threshold)
    low_thresh  = vol_series.quantile(0.33)
    high_thresh = vol_series.quantile(0.67)

    regime = pd.cut(
        vol_series,
        bins=[-np.inf, low_thresh, high_thresh, np.inf],
        labels=[0, 1, 2]
    ).astype(float)

    # Trend label: is price above its 63-day (1 quarter) moving average?
    ma63 = prices.rolling(63).mean()
    trend = (prices > ma63).astype(int)

    return pd.DataFrame({
        "regime":       regime,
        "trend_up":     trend,
        "realized_vol": realized_vol,
    }, index=prices.index)


def add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calendar effects matter in FX — month-end flows, quarter-end rebalancing,
    US/EU holiday effects all create systematic rate movements.
    """
    out = pd.DataFrame(index=index)
    out["day_of_week"]   = index.dayofweek          # 0=Mon, 4=Fri
    out["month"]         = index.month
    out["quarter"]       = index.quarter
    out["is_month_end"]  = index.is_month_end.astype(int)
    out["is_quarter_end"]= index.is_quarter_end.astype(int)
    # Month-end window: last 3 days of month (heavy corporate FX flows)
    out["month_end_window"] = (
        (index + pd.offsets.BusinessDay(3)).month != index.month
    ).astype(int)
    return out


def build_feature_matrix(
    prices: pd.Series,
    macro_df: Optional[pd.DataFrame] = None,
    fit_garch: bool = True,
    save: bool = True
) -> pd.DataFrame:
    """
    Master function: assembles the full feature matrix for a single FX pair.

    Args:
        prices:    Daily closing prices (pd.Series)
        macro_df:  Optional macro indicators from FRED
        fit_garch: Whether to fit GARCH (slower but better vol estimate)
        save:      Whether to save to data/processed/

    Returns:
        DataFrame with all features, aligned to prices index, NaN rows dropped
    """
    pair_name = prices.name or "FX"
    logger.info(f"Building feature matrix for {pair_name}...")

    frames = []

    # 1. Price itself (normalized for model input)
    frames.append(pd.DataFrame({"price": prices}))

    # 2. Returns at multiple horizons
    frames.append(compute_returns(prices))

    # 3. Rolling statistical features
    frames.append(compute_rolling_features(prices))

    # 4. GARCH conditional volatility
    if fit_garch:
        garch_feats = fit_garch_volatility(prices)
        frames.append(garch_feats)
        garch_vol = garch_feats.get("garch_cond_vol")
    else:
        garch_vol = None

    # 5. Regime labels
    frames.append(label_regimes(prices, garch_vol=garch_vol))

    # 6. Calendar features
    frames.append(add_calendar_features(prices.index))

    # 7. Macro features (if available) — align to FX index
    if macro_df is not None and not macro_df.empty:
        macro_aligned = macro_df.reindex(prices.index, method="ffill")
        # Compute yield differential (key FX driver via interest rate parity)
        if "us_10y" in macro_aligned.columns:
            macro_aligned["yield_diff"] = macro_aligned["us_10y"]  # placeholder: EUR 10Y not on FRED
        frames.append(macro_aligned)

    # Combine all features
    df = pd.concat(frames, axis=1)
    df = df.loc[prices.index]  # ensure alignment

    # Drop rows where critical features are missing
    initial_len = len(df)
    df = df.dropna(subset=["log_ret_1d", "vol_21d"])
    logger.info(f"Feature matrix: {len(df)}/{initial_len} rows after dropping NaN "
                f"({df.shape[1]} features)")

    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out_path = DATA_PROCESSED / f"{pair_name}_features.parquet"
        df.to_parquet(out_path)
        logger.success(f"Saved feature matrix to {out_path}")

    return df


if __name__ == "__main__":
    # Quick test with synthetic data if no API key available
    logger.info("Testing feature engineering with synthetic data...")
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="B")
    # Simulate GBM for EURUSD around 1.10
    returns = np.random.normal(0, 0.006, len(dates))
    prices = pd.Series(1.10 * np.exp(np.cumsum(returns)), index=dates, name="EURUSD")

    features = build_feature_matrix(prices, fit_garch=True, save=False)
    print(features.tail())
    print(f"\nFeatures: {list(features.columns)}")
    print(f"Shape: {features.shape}")
