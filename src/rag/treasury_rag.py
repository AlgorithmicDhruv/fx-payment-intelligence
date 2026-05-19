"""
src/rag/treasury_rag.py

RAG pipeline for FX / Treasury Q&A.

Architecture:
  Embeddings:   Hugging Face Inference API — BAAI/bge-small-en-v1.5 (67MB model,
                top-5 MTEB retrieval benchmark at its size class). No local model
                loaded — zero RAM impact. Free tier, official HF SDK.
  Vector store: Plain JSON file. No chromadb, no NumPy version conflicts.
  Generation:   Groq llama-3.3-70b-versatile (chat completion).

Interview Q: "Why HF embeddings over sentence-transformers locally?"
A: sentence-transformers loads a 90MB PyTorch model into RAM, which caused OOM
   on a 4GB machine. BAAI/bge-small-en-v1.5 via HF Inference API is identical
   quality (same model), zero local RAM, and actually faster since HF runs it
   on optimized inference hardware. The JSON store with pure-Python cosine
   similarity handles our 11-chunk corpus with sub-millisecond retrieval.

Interview Q: "Why not chromadb?"
A: chromadb 0.4.x uses np.float_ which was removed in NumPy 2.0. Since the
   project uses NumPy 2.x, chromadb crashes on import. For an 11-chunk corpus,
   a sorted list with cosine similarity is more than sufficient — chromadb adds
   complexity without any benefit at this scale.
"""

import sys, json, math, time, requests, os
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from huggingface_hub import InferenceClient

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.settings import (
    GROQ_API_KEY, GROQ_MODEL, HF_TOKEN, HF_EMBED_MODEL,
    VECTOR_STORE_PATH, RAG_DOCS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL,
)


# ── Cosine similarity (pure Python, no NumPy) ─────────────────────────────────
def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-10)


# ── HuggingFace embeddings ────────────────────────────────────────────────────
def _hf_embed(texts: List[str]) -> List[List[float]]:
    """
    Embed texts via HuggingFace Inference API.
    Model: BAAI/bge-small-en-v1.5 — 67MB, MTEB top-5 retrieval at its size.
    Runs on HF hardware. Zero local RAM. Free tier available.
    """
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN not set in .env — get a free token at huggingface.co/settings/tokens"
        )
    client = InferenceClient(api_key=HF_TOKEN)
    result = []
    for text in texts:
        emb = client.feature_extraction(text, model=HF_EMBED_MODEL)
        # HF SDK returns ndarray or nested list depending on model
        if hasattr(emb, "tolist"):
            vec = emb.tolist()
        else:
            vec = list(emb)
        # Flatten single nesting: [[...]] -> [...]
        if vec and isinstance(vec[0], list):
            vec = vec[0]
        result.append([float(v) for v in vec])
        time.sleep(0.05)   # light rate-limit pacing for free tier
    return result


# ── Groq LLM (generation only) ───────────────────────────────────────────────
def _groq_chat(system: str, messages: List[Dict]) -> str:
    """Generate an answer via Groq chat completion."""
    if not GROQ_API_KEY:
        return "GROQ_API_KEY not set in .env"
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": GROQ_MODEL,
              "messages": [{"role": "system", "content": system}] + messages,
              "max_tokens": 1024, "temperature": 0.1},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ── Text chunking ─────────────────────────────────────────────────────────────
def _chunk_text(text: str,
                size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    advance = max(1, size - overlap) guarantees termination regardless of input.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += max(1, size - overlap)
    return chunks


# ── Knowledge base ────────────────────────────────────────────────────────────
def _get_documents() -> List[Dict]:
    """
    Curated FX/payments knowledge base.
    6 documents covering: market structure, rate drivers, execution strategies,
    SWIFT infrastructure, volatility regimes, and risk management.
    """
    return [
        {
            "source": "FX Market Overview",
            "topic":  "market_structure",
            "content": (
                "The global FX market turns over $7.5 trillion per day (BIS 2022). "
                "Key participants: central banks, commercial banks, corporates, asset managers. "
                "Standard settlement is T+2. For SWIFT-based international payments, "
                "conversion timing affects the spot rate received, settlement risk "
                "(Herstatt risk), and liquidity requirements. "
                "Optimal conversion timing depends on forecast rate direction, "
                "volatility regime (high vol means wider bid-ask spreads), "
                "deadline pressure, and transaction size. "
                "For a $1M cross-border payment, converting at sender vs beneficiary "
                "bank can differ by 10-50 bps, or $1,000-$5,000."
            ),
        },
        {
            "source": "FX Rate Drivers",
            "topic":  "forecasting",
            "content": (
                "Short-term FX movements (1-5 days) are driven by: "
                "1. Interest Rate Differentials: F/S = (1+r_domestic)/(1+r_foreign). "
                "EURUSD is heavily influenced by the Fed vs ECB rate differential. "
                "2. Risk Sentiment: USD typically strengthens in risk-off environments. "
                "VIX spikes correlate with USD strength and EUR weakness. "
                "3. Economic Data Surprises: Non-Farm Payrolls, CPI, PMI releases "
                "cause sharp intraday moves. Implied volatility spikes around releases. "
                "4. Technical Levels: Round numbers (1.10, 1.15) act as support/resistance. "
                "5. Order Flow: COT data shows speculative positioning. "
                "Large short positions create potential for short squeezes. "
                "Models achieve directional accuracy of 52-56% at 1-day horizon."
            ),
        },
        {
            "source": "Corporate FX Execution Guide",
            "topic":  "optimization",
            "content": (
                "Given a payment deadline, when is the optimal time to execute? "
                "Strategy 1 - Spot Execution: execute immediately, zero optionality. "
                "Best in high-volatility environments when the rate is already favorable. "
                "Strategy 2 - TWAP: split order over deadline window. "
                "Reduces timing risk. Best for large orders with market impact concerns. "
                "Strategy 3 - Optimal Stopping (DP): wait until rate exceeds a threshold "
                "OR deadline approaches. Solved exactly via backward induction under GBM. "
                "Produces a threshold curve: the minimum rate to accept per day remaining. "
                "As the deadline approaches, the threshold drops. "
                "Strategy 4 - RL Policy: learns from historical trajectories, no GBM "
                "assumptions. Adapts to fat tails and regime switches that DP misses. "
                "In high-vol regimes the RL agent correctly executes earlier than DP. "
                "Backtesting: optimal stopping outperforms immediate execution by "
                "3-8 bps in normal markets, 10-20 bps in trending markets. "
                "Transaction costs of 2-5 bps must be subtracted."
            ),
        },
        {
            "source": "SWIFT Payment Infrastructure",
            "topic":  "payments",
            "content": (
                "SWIFT processes over 42 million messages per day across 200+ countries. "
                "Key FX message types: MT103 customer credit transfer, "
                "MT300/304 FX confirmation. ISO 20022 MX format migration completing 2025. "
                "Payment stream optimization: banks convert FX at the sending bank "
                "rather than the beneficiary bank. "
                "This reduces reliance on correspondent banks whose rates are often "
                "10-50 bps less favorable. For a $1M payment this means $1,000-$5,000 saved. "
                "G20 cross-border payment targets: cost under 3%, speed under 1 hour by 2027. "
                "CLS (Continuous Linked Settlement) settles $6 trillion per day "
                "payment-vs-payment, eliminating Herstatt settlement risk."
            ),
        },
        {
            "source": "FX Volatility Regimes",
            "topic":  "volatility",
            "content": (
                "FX volatility follows distinct regimes identified by GARCH(1,1)-t: "
                "Low vol: annualized vol under 6% (calm periods). "
                "Normal: 6-10% annualized vol. "
                "High vol: above 10% (crises, major policy shifts). "
                "EURUSD history: COVID 2020 peaked at 15%+, "
                "Fed tightening 2022 at 8-12%, 2023-2024 back to 5-8%. "
                "In LOW VOL: spreads tightest (0.5-1 pip), "
                "optimal stopping most effective, use threshold strategy. "
                "In HIGH VOL: spreads widen (2-5+ pips), "
                "execute sooner, reduce optionality. "
                "Why Student-t not Normal for GARCH: FX returns have fat tails — "
                "extreme moves occur 3x more than Gaussian predicts. "
                "Student-t degrees-of-freedom parameter adapts to tail heaviness."
            ),
        },
        {
            "source": "FX Risk Management",
            "topic":  "risk",
            "content": (
                "VaR approaches for FX: "
                "Parametric VaR assumes normality and underestimates tail risk. "
                "Historical simulation VaR uses the actual return distribution (preferred). "
                "Monte Carlo VaR simulates thousands of scenarios (most flexible). "
                "Example: 1-day 99% VaR on $10M EURUSD, daily vol 0.6%: "
                "VaR = $10M x 0.006 x 2.326 = $139,560. "
                "Expected Shortfall (CVaR): mean loss beyond VaR threshold. "
                "More informative for tail risk. Required under Basel III. "
                "FX-specific risks: settlement risk (mitigated by CLS), "
                "rollover risk (rate changes between trade and T+2 settlement), "
                "model risk (GBM assumptions break in crisis periods). "
                "Stress tests: 2008 GFC EURUSD moved 20% in 6 months, "
                "2015 CHF unpegging 15-20% in minutes, 2016 Brexit GBP fell 10% overnight."
            ),
        },
    ]


# ── Vector store ──────────────────────────────────────────────────────────────
def build_vector_store(force_rebuild: bool = False) -> List[Dict]:
    """
    Build or load the vector store.
    Uses HF Inference API for embeddings — no local model, no RAM spike.
    Stored as plain JSON for maximum compatibility.
    """
    VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if VECTOR_STORE_PATH.exists() and not force_rebuild:
        store = json.loads(VECTOR_STORE_PATH.read_text(encoding="utf-8"))
        logger.info(f"Loaded vector store: {len(store)} chunks")
        return store

    docs   = _get_documents()
    chunks, metas = [], []
    for doc in docs:
        for chunk in _chunk_text(doc["content"]):
            chunks.append(chunk)
            metas.append({"source": doc["source"], "topic": doc["topic"]})

    logger.info(
        f"Embedding {len(chunks)} chunks via HF API "
        f"(model: {HF_EMBED_MODEL}, zero local RAM)..."
    )
    embeddings = _hf_embed(chunks)

    store = [
        {"text": t, "embedding": e,
         "source": m["source"], "topic": m["topic"]}
        for t, e, m in zip(chunks, embeddings, metas)
    ]
    VECTOR_STORE_PATH.write_text(json.dumps(store), encoding="utf-8")
    logger.success(f"Vector store saved: {len(store)} chunks → {VECTOR_STORE_PATH}")
    return store


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(store: List[Dict], query: str,
             k: int = TOP_K_RETRIEVAL) -> List[Dict]:
    """
    Embed query via HF API and rank chunks by cosine similarity.
    Pure-Python cosine: no NumPy, no version conflicts.
    Sub-millisecond for 11 chunks.
    """
    q_emb  = _hf_embed([query])[0]
    scored = sorted(store,
                    key=lambda c: _cosine(q_emb, c["embedding"]),
                    reverse=True)
    return [
        {"text": c["text"], "source": c["source"],
         "relevance_score": round(_cosine(q_emb, c["embedding"]), 3)}
        for c in scored[:k]
    ]


# ── Answer generation ─────────────────────────────────────────────────────────
def answer(query: str, store: List[Dict],
           history: Optional[List[Dict]] = None) -> Dict:
    """Retrieve relevant chunks then generate an answer via Groq LLM."""
    chunks  = retrieve(store, query)
    context = "\n---\n".join(
        f"[{c['source']} | relevance {c['relevance_score']}]\n{c['text']}"
        for c in chunks
    )
    system = (
        "You are an expert FX markets and treasury operations analyst. "
        "Answer using the provided context. Be precise, quantitative where "
        "possible, and cite your sources. "
        "If the context is insufficient, say so clearly."
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {query}"
    messages = (history or [])[-6:] + [{"role": "user", "content": user_msg}]
    try:
        ans = _groq_chat(system, messages)
    except Exception as e:
        ans = f"Generation error: {e}"
    return {
        "answer":  ans,
        "sources": list({c["source"] for c in chunks}),
        "chunks":  chunks,
    }


# ── Stateful assistant ────────────────────────────────────────────────────────
class TreasuryAssistant:
    """Multi-turn RAG assistant. Used by both Streamlit apps."""

    def __init__(self, force_rebuild: bool = False):
        logger.info("Initializing Treasury RAG Assistant...")
        self.store   = build_vector_store(force_rebuild=force_rebuild)
        self.history: List[Dict] = []
        logger.success(f"Ready — {len(self.store)} chunks")

    # Compatibility shim for code that calls .collection.count()
    class _Col:
        def __init__(self, n): self._n = n
        def count(self): return self._n

    @property
    def collection(self):
        return self._Col(len(self.store))

    def chat(self, query: str) -> Dict:
        result = answer(query, self.store, self.history)
        self.history += [
            {"role": "user",      "content": query},
            {"role": "assistant",  "content": result["answer"]},
        ]
        return result

    def reset(self):
        self.history = []


if __name__ == "__main__":
    a = TreasuryAssistant()
    r = a.chat("What is optimal stopping for FX execution with a 21-day deadline?")
    print(f"\nAnswer:\n{r['answer'][:500]}")
    print(f"\nSources: {r['sources']}")