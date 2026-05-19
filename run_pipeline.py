"""
run_pipeline.py — Master orchestrator for FX Payment Intelligence Engine

Usage:
  python run_pipeline.py --steps all              # full pipeline
  python run_pipeline.py --steps data             # ingestion + features only
  python run_pipeline.py --steps data,forecast    # data then train model
  python run_pipeline.py --steps optimize         # DP + RL only
  python run_pipeline.py --steps rag              # build vector store only
  python run_pipeline.py --demo                   # synthetic data, no API keys needed
  python run_pipeline.py --demo --steps data      # demo mode, data step only
"""

import argparse
import sys
import time
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from configs.settings import (
    PRIMARY_PAIR, DATA_PROCESSED, OPT_HORIZON, FRED_API_KEY, GROQ_API_KEY
)

VALID_STEPS = ["data", "forecast", "optimize", "rag", "all"]


# ─────────────────────────── synthetic data fallback ──────────────────────────
def _make_synthetic(pair: str = PRIMARY_PAIR) -> pd.DataFrame:
    """Realistic synthetic EURUSD — works with zero API keys."""
    logger.info("Generating synthetic FX data (demo mode)...")
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="B")
    n = len(dates)
    vol = np.ones(n) * 0.006
    vol[1300:1450] = 0.013   # COVID
    vol[1800:2000] = 0.010   # Fed tightening
    log_ret = np.random.normal(0.00005, vol)
    prices = pd.Series(1.10 * np.exp(np.cumsum(log_ret)), index=dates, name=pair)
    return pd.DataFrame({pair: prices})


# ─────────────────────────── step functions ────────────────────────────────────
def step_data(demo=False, force_refresh=False):
    logger.info("=" * 55)
    logger.info("STEP 1: Data Ingestion + Feature Engineering")
    logger.info("=" * 55)

    if demo or not FRED_API_KEY:
        if not FRED_API_KEY:
            logger.warning("No FRED_API_KEY in .env — using synthetic data")
        fx_df = _make_synthetic()
    else:
        from src.ingestion.fx_data import pull_fx_data, pull_macro_indicators
        fx_df    = pull_fx_data(force_refresh=force_refresh)
        macro_df = pull_macro_indicators()

    from src.features.engineer import build_feature_matrix

    pair   = PRIMARY_PAIR if PRIMARY_PAIR in fx_df.columns else fx_df.columns[0]
    prices = fx_df[pair]

    macro_df = None
    if not demo and FRED_API_KEY:
        try:
            from src.ingestion.fx_data import pull_macro_indicators
            macro_df = pull_macro_indicators()
            if macro_df.empty:
                macro_df = None
        except Exception as e:
            logger.warning(f"Macro pull failed: {e}")

    feature_df = build_feature_matrix(
        prices, macro_df=macro_df, fit_garch=True, save=True
    )
    logger.success(f"Features: {feature_df.shape[0]} rows x {feature_df.shape[1]} cols")
    return feature_df


def step_forecast(feature_df):
    logger.info("=" * 55)
    logger.info("STEP 2: Transformer Forecasting Model")
    logger.info("=" * 55)
    from src.forecasting.transformer_model import train
    result = train(feature_df, save_model=True)
    logger.success("Metrics:")
    for k, v in result["metrics"].items():
        logger.info(f"  {k}: {v}")
    return result


def step_optimize(feature_df):
    logger.info("=" * 55)
    logger.info("STEP 3: DP Optimal Stopping + RL Agent")
    logger.info("=" * 55)
    from src.optimization.fx_optimizer import (
        estimate_gbm_params, dp_optimal_stopping,
        backtest_strategies, train_rl_agent
    )

    log_ret     = feature_df["log_ret_1d"].dropna()
    mu, sigma   = estimate_gbm_params(log_ret)
    logger.info(f"GBM params  mu={mu:.4f}  sigma={sigma:.4f}")

    dp_result = dp_optimal_stopping(
        horizon=OPT_HORIZON, mu=mu, sigma=sigma,
        current_price=float(feature_df["price"].iloc[-1])
    )

    # RL — optional, skip cleanly if sb3 not installed
    rl_model = None
    try:
        import stable_baselines3
        logger.info("Training RL PPO agent (50k steps for quick test)...")
        rl_out   = train_rl_agent(feature_df, timesteps=50_000, save=True)
        rl_model = rl_out.get("model")
    except ImportError:
        logger.warning("stable-baselines3 not installed — skipping RL. "
                       "pip install stable-baselines3 gymnasium")

    logger.info("Running backtest (200 episodes)...")
    bt_df = backtest_strategies(
        feature_df, dp_result, rl_model=rl_model, n_episodes=200
    )

    logger.success("Backtest summary (bps vs naive):")
    for col in ["dp_bps", "rl_bps"]:
        if col in bt_df.columns:
            s = bt_df[col]
            logger.info(f"  {col}: mean={s.mean():+.2f}  "
                        f"std={s.std():.2f}  "
                        f"win={( s > 0).mean():.0%}")
    return dp_result


def step_rag():
    logger.info("=" * 55)
    logger.info("STEP 4: RAG Treasury Knowledge Base")
    logger.info("=" * 55)

    if not GROQ_API_KEY:
        logger.warning("No GROQ_API_KEY — vector store will build but LLM "
                       "answers won't work. Add key to .env to enable chat.")

    try:
        from src.rag.treasury_rag import TreasuryAssistant
        assistant = TreasuryAssistant(force_rebuild=False)
        logger.success(f"Vector store ready: {assistant.collection.count()} chunks")

        if GROQ_API_KEY:
            result = assistant.chat(
                "What is the optimal FX execution strategy for a 21-day deadline?"
            )
            logger.info(f"RAG test answer: {result['answer'][:200]}...")
        return assistant
    except Exception as e:
        logger.error(f"RAG step failed: {e}")
        logger.info("Ensure HF_TOKEN is set in .env and huggingface_hub is installed")
        return None


# ─────────────────────────── main ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FX Payment Intelligence Engine",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma-separated: data, forecast, optimize, rag, all\n"
             "Examples:\n"
             "  --steps all\n"
             "  --steps data\n"
             "  --steps data,forecast\n"
             "  --steps optimize"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic data — no API keys required"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download data even if cached"
    )
    args = parser.parse_args()

    steps    = [s.strip().lower() for s in args.steps.split(",")]
    run_all  = "all" in steps
    t0       = time.time()

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  FX PAYMENT INTELLIGENCE ENGINE          ║")
    logger.info("╚══════════════════════════════════════════╝")

    if args.demo:
        logger.info("Mode: DEMO (synthetic data, no API keys needed)")
    else:
        logger.info(f"FRED key : {'✓ set' if FRED_API_KEY else '✗ missing — will use synthetic'}")
        logger.info(f"Groq key : {'✓ set' if GROQ_API_KEY else '✗ missing — RAG chat disabled'}")

    feature_df = None

    # ── load cached features if we're skipping the data step ──
    if not (run_all or "data" in steps):
        cache = DATA_PROCESSED / f"{PRIMARY_PAIR}_features.parquet"
        if cache.exists():
            feature_df = pd.read_parquet(cache)
            logger.info(f"Loaded cached features: {feature_df.shape}")
        else:
            logger.warning("No cached features found — running data step first")
            steps.append("data")

    # ── run requested steps ──
    if run_all or "data" in steps:
        feature_df = step_data(demo=args.demo, force_refresh=args.force_refresh)

    if run_all or "forecast" in steps:
        if feature_df is None:
            logger.error("Need feature data first. Run --steps data first.")
            sys.exit(1)
        step_forecast(feature_df)

    if run_all or "optimize" in steps:
        if feature_df is None:
            logger.error("Need feature data first. Run --steps data first.")
            sys.exit(1)
        step_optimize(feature_df)

    if run_all or "rag" in steps:
        step_rag()

    elapsed = time.time() - t0
    logger.success(f"\nDone in {elapsed:.1f}s")
    logger.info("Launch dashboard:  streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()