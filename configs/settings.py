"""
configs/settings.py — Central configuration for FX Payment Intelligence Engine
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"

# ── API Keys ──────────────────────────────────────────────────────────────────
FRED_API_KEY   = os.getenv("FRED_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
HF_TOKEN       = os.getenv("HF_TOKEN", "")

# ── Currency pairs ────────────────────────────────────────────────────────────
FX_PAIRS = {
    "EURUSD": {"fred_series": "DEXUSEU", "yf_ticker": "EURUSD=X", "invert": True},
    "GBPUSD": {"fred_series": "DEXUSUK", "yf_ticker": "GBPUSD=X", "invert": True},
    "USDJPY": {"fred_series": "DEXJPUS", "yf_ticker": "JPY=X",    "invert": False},
    "USDCNY": {"fred_series": "DEXCHUS", "yf_ticker": "CNY=X",    "invert": False},
}
PRIMARY_PAIR = "EURUSD"

# ── Data ingestion ────────────────────────────────────────────────────────────
HISTORY_YEARS   = 10
FRED_START_DATE = "2014-01-01"
FRED_END_DATE   = None

# ── Feature engineering ───────────────────────────────────────────────────────
ROLLING_WINDOWS   = [5, 10, 21, 63]
VOLATILITY_WINDOW = 21
REGIME_THRESHOLD  = 0.015

# ── GARCH ─────────────────────────────────────────────────────────────────────
GARCH_P    = 1
GARCH_Q    = 1
GARCH_DIST = "t"

# ── Transformer ───────────────────────────────────────────────────────────────
SEQ_LEN        = 60
PRED_HORIZONS  = [1, 5]
HIDDEN_DIM     = 64
N_HEADS        = 4
N_LAYERS       = 2
DROPOUT        = 0.1
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 32
MAX_EPOCHS     = 100
EARLY_STOP_PAT = 10
TRAIN_SPLIT    = 0.7
VAL_SPLIT      = 0.15

# ── Optimization / RL ─────────────────────────────────────────────────────────
OPT_HORIZON      = 21
TRANSACTION_COST = 0.0002
RL_TIMESTEPS     = 200_000
RL_ALGO          = "PPO"

# ── RAG ───────────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"

# Embeddings: Hugging Face Inference API (BAAI/bge-small-en-v1.5, 67MB)
# Free tier, no local model loaded, no RAM spike, real semantic search
HF_EMBED_MODEL  = "BAAI/bge-small-en-v1.5"

# Vector store: plain JSON (no chromadb, no NumPy version conflicts)
VECTOR_STORE_PATH = MODELS_DIR / "rag" / "vector_store.json"
RAG_DOCS_DIR      = DATA_RAW / "rag_docs"
CHUNK_SIZE        = 600
CHUNK_OVERLAP     = 60
TOP_K_RETRIEVAL   = 5

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_TITLE  = "FX Payment Intelligence Engine"
REFRESH_INTERVAL = 300