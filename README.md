# FX Payment Intelligence Engine

An end-to-end quantitative system that solves the corporate treasury FX execution problem: *given a payment deadline of T trading days, on which day should you convert currency to maximise the rate received?*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://fx-payment-intelligence.streamlit.app/) 

The system combines statistical modelling, deep learning, stochastic optimisation, reinforcement learning, and retrieval-augmented generation into a single production-grade pipeline with two Streamlit interfaces.

---

## Live Deployment
The interactive platform is fully deployed and accessible on Streamlit Cloud:
👉 **[https://fx-payment-intelligence.streamlit.app/](https://fx-payment-intelligence.streamlit.app/)**

---
## The Problem

A corporate treasurer needs to convert USD → EUR within 21 trading days. They observe the spot rate each day and decide: execute now, or wait? Executing too early means leaving a better rate on the table. Waiting too long risks a worse rate as the deadline forces execution.

This is a classical **optimal stopping problem** — and the core of what payment and treasury teams at major financial institutions work on daily.

---

## Architecture

```
fx_intelligence/
├── configs/
│   └── settings.py              # All hyperparameters, paths, API config
├── src/
│   ├── ingestion/
│   │   └── fx_data.py           # FRED + yfinance data pipeline
│   ├── features/
│   │   └── engineer.py          # GARCH, regime detection, 39-feature matrix
│   ├── forecasting/
│   │   └── transformer_model.py # Multi-horizon quantile Transformer
│   ├── optimization/
│   │   └── fx_optimizer.py      # DP optimal stopping + PPO RL agent + backtest
│   ├── rag/
│   │   └── treasury_rag.py      # HF embeddings + JSON store + Groq LLM
│   └── dashboard/
│       └── app.py               # Production Streamlit dashboard
├── run_pipeline.py              # Master orchestration script
├── interview_demo.py            # Interactive Streamlit demo
├── requirements.txt
└── .env                         # API keys (never committed)
```

---

## Five Modules

### 1. Feature Engineering (`src/features/engineer.py`)

Transforms raw daily FX spot rates into a **39-feature matrix** consumed by both the Transformer and the RL agent.

| Group | Features |
|---|---|
| Returns | Log returns at 1, 2, 5, 10, 21 day horizons |
| Rolling stats | Volatility, momentum, z-score, RSI, Bollinger Band position at 4 windows (1W/2W/1M/1Q) |
| GARCH(1,1)-t | Conditional volatility and standardised residuals |
| Regimes | Low / Normal / High vol labels via 33rd/67th percentile cutoffs on GARCH vol |
| Calendar | Month-end, quarter-end flags (corporate FX flow patterns) |

**Why GARCH(1,1) with Student-t errors?** FX returns exhibit volatility clustering and fat tails — extreme moves occur roughly 3× more often than a Gaussian distribution predicts. The Student-t distribution's degrees-of-freedom parameter adapts to actual tail heaviness, giving better-calibrated risk estimates for the optimiser.

**Why log returns, not prices?** Prices are non-stationary. Log returns are stationary — a model trained on 2015 data generalises to 2024.

---

### 2. Transformer Forecasting Model (`src/forecasting/transformer_model.py`)

A Transformer encoder that forecasts the full return distribution, not just a point estimate.

| Parameter | Value |
|---|---|
| Input | 60-day sliding window × 39 features |
| Architecture | 4-head self-attention, 2 encoder layers, 107K parameters |
| Output | q10 / q50 / q90 quantiles for 1-day and 5-day horizons |
| Loss | Pinball (quantile) loss — not MSE |
| Training split | 70% train / 15% val / 15% test (chronological, no leakage) |

**Why Transformer over LSTM?** Self-attention captures non-local temporal dependencies. Month-end corporate FX flow patterns from 30 days prior influence today's rate — a Transformer attends to them directly. An LSTM must carry that signal across 30 hidden-state updates.

**Why quantile loss?** The DP optimiser needs a full predictive distribution to compute the expected value of waiting. A point forecast is insufficient. The 80% interval (q10–q90) should contain ~80% of actual returns — calibration target: 0.78–0.82 coverage.

**Directional accuracy** of 52–56% at 1-day horizon is statistically significant. In a random walk it would be 50%.

---

### 3. DP Optimal Stopping (`src/optimization/fx_optimizer.py`)

Solves the treasurer's decision problem **exactly** under Geometric Brownian Motion assumptions via backward induction (Bellman equation).

**Bellman equation:**
```
V(t=0, S) = S × (1 − cost)                     ← deadline: must execute
V(t,   S) = max(S × (1−cost),  E[V(t−1, S')])  ← execute or wait
```

Solved on a 200-point price grid using Gauss–Hermite quadrature (20 nodes) for the expectation integral. Produces a **threshold curve**: the minimum rate to accept at each day remaining.

As the deadline approaches, the threshold drops — you become less selective because the option value of waiting diminishes.

**GBM parameters** are estimated from historical log returns:
- Drift μ fitted from sample mean × 252
- Volatility σ fitted from sample std × √252
- Mild risk-free discounting at 2% annually

**Limitations of DP:** GBM assumes constant volatility and no regime switches. Real FX violates both. This motivates the RL agent.

---

### 4. RL PPO Agent (`src/optimization/fx_optimizer.py`)

A Proximal Policy Optimisation agent trained on historical FX trajectories with no distributional assumptions.

**State vector (8 dimensions):**
```
[price_change_norm, days_remaining_norm, vol_21d,
 regime, h1d_q50_forecast, h5d_q50_forecast,
 h1d_uncertainty, rsi_14d]
```

**Action:** 0 = wait, 1 = execute now

**Reward:** basis points captured vs executing immediately at t=0 (sparse, end-of-episode)

**Training:** 200,000 environment steps on historical FX data. Each episode is a 21-day window starting at a randomly sampled historical date.

**Why PPO over DQN?** The action space is discrete (execute/wait) but the reward is sparse — PPO's clipped surrogate objective handles this better than Q-learning at this episode length.

**What the RL agent learns that DP cannot:**
- In high-vol regimes: execute *earlier* than the DP threshold (fat tails make waiting riskier than GBM assumes)
- In trending markets: wait *longer* than DP (momentum persists more than GBM assumes)

**Backtest methodology:** N non-overlapping walk-forward episodes. All three strategies (Naive, DP, RL) start from the same historical price. No lookahead bias.

---

### 5. RAG Treasury Assistant (`src/rag/treasury_rag.py`)

A retrieval-augmented Q&A system grounded in curated FX and payments documents.

| Component | Implementation |
|---|---|
| Embeddings | HuggingFace Inference API — BAAI/bge-small-en-v1.5 (67MB, MTEB top-5 retrieval at its size class) |
| Vector store | Plain JSON file — pure-Python cosine similarity |
| Generation | Groq llama-3.3-70b-versatile (temperature 0.1) |
| Retrieval | Top-5 chunks by cosine similarity |

**Knowledge corpus (6 documents, 11 chunks):**
- FX Market Overview — market structure, T+2 settlement, payment stream economics
- FX Rate Drivers — interest rate parity, risk sentiment, economic data surprises
- Corporate FX Execution Guide — execution strategies, optimal stopping, backtesting
- SWIFT Payment Infrastructure — MT103/MT300, ISO 20022, CLS settlement
- FX Volatility Regimes — GARCH regime identification, execution implications
- FX Risk Management — VaR, CVaR, Basel III, stress scenarios

**Why HuggingFace API over local sentence-transformers?**
Loading sentence-transformers locally requires ~500MB RAM and caused OOM on 4GB machines. BAAI/bge-small-en-v1.5 via HF Inference API is the same model running on HF's hardware — identical quality, zero local RAM.

**Why plain JSON over chromadb?**
chromadb 0.4.x uses `np.float_` which was removed in NumPy 2.0. For an 11-chunk corpus, a sorted list with cosine similarity is sub-millisecond and has no dependency conflicts.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/AlgorithmicDhruv/fx-payment-intelligence.git
cd fx-payment-intelligence

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
FRED_API_KEY=your_key_here     # free at fred.stlouisfed.org/docs/api/api_key.html
GROQ_API_KEY=your_key_here     # free at console.groq.com
HF_TOKEN=your_token_here       # free at huggingface.co/settings/tokens
```

### 3. Run the pipeline

```bash
# Full pipeline (data → features → train → RL → RAG)
python run_pipeline.py --steps all

# Individual steps
python run_pipeline.py --steps data
python run_pipeline.py --steps forecast
python run_pipeline.py --steps optimize
python run_pipeline.py --steps rag

# No API keys needed (synthetic data)
python run_pipeline.py --demo
```

### 4. Launch dashboards

```bash
# Production dashboard (4 tabs)
streamlit run src/dashboard/app.py

# Interactive interview demo (5 pages, live inputs)
streamlit run interview_demo.py
```

---

## Dashboards

### Production Dashboard (`src/dashboard/app.py`)

| Tab | Content |
|---|---|
| Live Rates & Forecast | Spot rate chart with Transformer confidence bands (q10/q50/q90), volatility regime bar chart, RSI |
| Execution Optimizer | DP solver — adjust drift, vol, deadline, transaction cost live; instant EXECUTE/WAIT decision |
| Backtest Results | Distribution of P&L across 300 episodes for Naive vs DP vs RL |
| Treasury Assistant | Multi-turn RAG chat with source attribution |

### Interview Demo (`interview_demo.py`)

Five-page Streamlit app where every input is interactive — interviewers can change parameters and test scenarios live without running any code:

- **Features page:** Date picker → instant feature table for any historical date
- **Forecast page:** Historical date → predicted vs actual price, error in bps, direction correct or not
- **DP page:** Sliders for drift, vol, deadline, transaction cost → live threshold curve + EXECUTE/WAIT
- **Backtest page:** Configurable episodes, horizon, transaction cost → P&L histogram
- **RAG page:** Dropdown of suggested questions or free-text → answer with source attribution

---

## Data Sources

| Source | Data | Access |
|---|---|---|
| FRED (Federal Reserve) | EURUSD, GBPUSD, USDJPY, USDCNY daily spot rates from 2014 | Free API key |
| FRED | US 10Y yield, VIX, Fed Funds Rate, CPI, M2 | Free API key |
| Yahoo Finance | Fallback spot rates if FRED unavailable | No key required |
| HuggingFace | BAAI/bge-small-en-v1.5 embeddings | Free token |
| Groq | llama-3.3-70b-versatile LLM generation | Free tier |

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| Data | FRED API, yfinance, pandas | Authoritative FX data, clean daily series |
| Features | arch (GARCH), scikit-learn, NumPy | GARCH vol clustering, rolling stats |
| Forecasting | PyTorch, quantile loss | Full distribution forecast, not point estimate |
| Optimisation | NumPy/SciPy DP, stable-baselines3 PPO | Exact solution + model-free learned policy |
| RAG | HuggingFace Hub, Groq, pure Python | Zero local RAM, no version conflicts |
| Storage | Parquet (features), PyTorch checkpoint, JSON (vectors) | Fast I/O, no heavy DB dependencies |
| Dashboard | Streamlit, Plotly | Rapid deployment, interactive |

---

## Key Design Decisions

**Chronological train/val/test splits** — FX data is never shuffled. The test set is always the most recent 15% of data. Shuffling would leak future information into training.

**Quantile loss over MSE** — MSE optimises for a point estimate. The DP optimiser needs the full predictive distribution (q10/q50/q90) to compute the expected value of waiting. Coverage calibration (target 0.78–0.82) ensures the uncertainty bands are statistically honest.

**DP as analytical baseline** — Before introducing RL, we solve the problem exactly under GBM assumptions. This gives an interpretable benchmark. The RL agent is then evaluated against it, not just against a naive baseline.

**No chromadb, no sentence-transformers** — Both caused environment-specific failures (NumPy 2.x incompatibility and OOM respectively). The replacement — HF Inference API + JSON store + pure-Python cosine — is more robust and has identical retrieval quality at this corpus size.

---

## Environment Variables

| Variable | Required | Source |
|---|---|---|
| `FRED_API_KEY` | Yes (for real data) | fred.stlouisfed.org/docs/api/api_key.html |
| `GROQ_API_KEY` | Yes (for RAG answers) | console.groq.com |
| `HF_TOKEN` | Yes (for embeddings) | huggingface.co/settings/tokens |

All three have free tiers. No credit card required.

---

## Author

**Dhruvkumar Mayurkumar Patel** ·
M.S. Data Science ·
[LinkedIn](https://linkedin.com/in/dhruvkumar-mayurkumar-patel-94b745210) · [Live Demo](https://fx-payment-intelligence.streamlit.app/)