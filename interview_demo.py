"""
interview_demo.py
Run: streamlit run interview_demo.py
"""

import sys, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
from configs.settings import (
    PRIMARY_PAIR, DATA_PROCESSED, OPT_HORIZON,
    TRANSACTION_COST, PRED_HORIZONS, FX_PAIRS, GROQ_API_KEY
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FX Intelligence",
    page_icon="🏦", layout="wide",
)

st.markdown("""
<style>
/* Strip Streamlit wrapper background so custom divs show through */
[data-testid="stMarkdownContainer"] { background: transparent !important; }

.info-bar {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-size: 13.5px;
    line-height: 1.7;
    margin-bottom: 14px;
    color: #1e3a8a;
}
.go {
    background: #d1fae5;
    border: 1px solid #6ee7b7;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    color: #065f46;
    margin: 10px 0;
}
.wait {
    background: #fef9c3;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    color: #78350f;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def note(html):
    st.markdown(f'<div class="info-bar">{html}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_features(pair):
    cache = DATA_PROCESSED / f"{pair}_features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="B")
    n = len(dates)
    v = np.where((np.arange(n)>1300)&(np.arange(n)<1450),0.013,
        np.where((np.arange(n)>1800)&(np.arange(n)<2000),0.010,0.006))
    lr = np.random.normal(0.00005, v)
    px = pd.Series(1.10*np.exp(np.cumsum(lr)), index=dates, name=pair)
    df = pd.DataFrame({"price":px})
    df["log_ret_1d"]     = np.log(px/px.shift(1))
    df["log_ret_5d"]     = np.log(px/px.shift(5))
    df["vol_21d"]        = df["log_ret_1d"].rolling(21).std()*np.sqrt(252)
    df["garch_cond_vol"] = df["vol_21d"]*(1+0.08*np.random.randn(n))
    df["mom_5d"]         = np.log(px/px.shift(5))
    df["mom_21d"]        = np.log(px/px.shift(21))
    df["rsi_14d"]        = np.clip(50+np.random.normal(0,12,n),10,90)
    df["bb_position"]    = np.random.normal(0,0.45,n)
    df["zscore_21d"]     = (px-px.rolling(21).mean())/px.rolling(21).std()
    q33,q67 = df["vol_21d"].quantile(0.33), df["vol_21d"].quantile(0.67)
    df["regime"] = pd.cut(df["vol_21d"],bins=[-np.inf,q33,q67,np.inf],
                          labels=[0,1,2]).astype(float)
    return df.dropna()

def smart_truncate(text, max_chars=400):
    """Truncates text safely at a word boundary so words aren't cut in half."""
    if len(text) <= max_chars:
        return text
    
    # Slice up to the max character threshold
    truncated = text[:max_chars]
    
    # Find the last space character in the sliced string to avoid word-splitting
    last_space = truncated.rfind(" ")
    
    if last_space != -1:
        return truncated[:last_space] + "..."
    return truncated + "..."

@st.cache_resource(show_spinner=False)
def get_model():
    import sys
    import os
    import torch
    
    # Force add project root directory to python path for Linux compatibility
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    
    mp = Path("models/forecasting/fx_transformer.pt")
    if not mp.exists(): 
        st.error("DEBUG: The physical file path does not exist on the server!")
        return None
        
    try:
        # Save the original torch.load function
        original_torch_load = torch.load
        
        # Define a wrapper that forces weights_only=False for PyTorch 2.6+ compatibility
        def custom_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            # Ensure it maps to CPU since Streamlit Cloud doesn't have a GPU
            if 'map_location' not in kwargs:
                kwargs['map_location'] = 'cpu'
            return original_torch_load(*args, **kwargs)
            
        # Temporarily overwrite torch.load with our wrapper
        torch.load = custom_torch_load
        
        from src.forecasting.transformer_model import load_model
        model = load_model(mp)
        
        # Restore the original torch.load function to keep the environment clean
        torch.load = original_torch_load
        
        return model
        
    except Exception as e:
        st.error(f"DEBUG: Model found, but loading failed! Error: {e}")
        return None
    
# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 FX Intelligence")
    st.caption("Quantitative FX Execution & Risk Platform")
    st.divider()

    PAGE = st.radio("", [
        "Overview",
        "1. Feature Engineering",
        "2. Transformer Forecast",
        "3. DP Optimal Stopping",
        "4. RL Agent & Backtest",
        "5. RAG Treasury Assistant",
    ], label_visibility="collapsed")

    st.divider()
    pair = st.selectbox("Currency pair", list(FX_PAIRS.keys()))


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if PAGE == "Overview":
    st.title("FX Payment Intelligence Engine")
    st.markdown("by Dhruvkumar Patel")
    st.divider()

    note(
    '<b>The core problem:</b> A corporate client needs to convert $10M USD→EUR within 21 days. '
    'They can execute any day — <b>which day gives the best rate?</b><br><br>'
    'This is the problem payment and treasury teams at major financial institutions solve: '
    '<i>Convert FX earlier in the payment chain to avoid relying on beneficiary banks.</i>'
)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 5 Modules")
        st.markdown("""
| # | Module | What it does |
|---|---|---|
| 1 | Feature Engineering | GARCH vol, regime detection, 39 features |
| 2 | Transformer Forecast | Quantile predictions (q10/q50/q90) |
| 3 | DP Optimal Stopping | Exact threshold curve — backward induction |
| 4 | RL PPO Agent | Learned policy, no distributional assumptions |
| 5 | RAG Assistant | GenAI Q&A over FX/payments documents |
        """)

    with c2:
        st.markdown("#### Tech Stack")
        st.markdown("""
| Layer | Technology |
|---|---|
| Data | FRED API · yfinance |
| Features | arch (GARCH) · scikit-learn |
| Forecast | PyTorch Transformer · quantile loss |
| Optimize | scipy DP · stable-baselines3 PPO |
| RAG | HuggingFace API · Groq LLM |
| UI | Streamlit · Plotly |
        """)

    st.divider()
    st.info("Navigate using the sidebar. Each page has live inputs — change values and results update instantly.")


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURES
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "1. Feature Engineering":
    st.title("1. Feature Engineering")
    note("""
    Transforms raw FX prices into <b>39 features</b> consumed by both the Transformer and RL agent.<br>
    <b>GARCH(1,1)-t</b> captures volatility clustering and fat tails (Student-t errors).
    <b>Regime labels</b> (Low/Normal/High vol) let the RL agent learn different policies per market state.
    """)

    df = get_features(pair)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Trading days",   f"{len(df):,}")
    c2.metric("Features",       df.shape[1])
    c3.metric("Years of data",  f"{df.index.min().year}–{df.index.max().year}")
    c4.metric("Mean GARCH vol", f"{df['garch_cond_vol'].mean():.1%}")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("**Volatility regime distribution**")
        rc = df["regime"].value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=["Low Vol","Normal","High Vol"],
            y=[rc.get(0.0,0),rc.get(1.0,0),rc.get(2.0,0)],
            marker_color=["#22c55e","#f59e0b","#ef4444"],
            text=[rc.get(0.0,0),rc.get(1.0,0),rc.get(2.0,0)],
            textposition="outside"
        ))
        fig.update_layout(height=220, template="plotly_white",
                          margin=dict(t=10,b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**GARCH conditional volatility (2 years)**")
        r2 = df.tail(504)
        fig2 = go.Figure(go.Scatter(
            x=r2.index, y=r2["garch_cond_vol"],
            fill="tozeroy", line=dict(color="#3b82f6",width=1.2),
            fillcolor="rgba(59,130,246,0.15)"
        ))
        fig2.update_layout(height=180, template="plotly_white",
                           margin=dict(t=5,b=5), yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown("**Inspect any date**")
        d = st.date_input("", value=datetime(2020,3,20),
                          min_value=datetime(2015,1,5),
                          max_value=datetime(2024,12,30),
                          label_visibility="collapsed")
        idx = min(df.index.searchsorted(str(d)), len(df)-1)
        row = df.iloc[idx]
        reg_map = {0.0:"🟢 Low Vol", 1.0:"🟡 Normal", 2.0:"🔴 High Vol"}
        show = ["price","log_ret_1d","vol_21d","garch_cond_vol","regime","rsi_14d","bb_position","mom_21d"]
        show = [c for c in show if c in df.columns]
        rows = []
        for c in show:
            v = float(row[c])
            rows.append({"Feature":c,
                         "Value": reg_map.get(v,f"{v:.6f}") if c=="regime" else f"{v:.6f}"})
        st.caption(f"Actual date shown: {df.index[idx].date()}")
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True, height=310)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRANSFORMER FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "2. Transformer Forecast":
    st.title("2. Transformer Forecasting Model")
    note("""
    <b>Input:</b> 60-day window × 39 features &nbsp;·&nbsp;
    <b>Output:</b> q10 / q50 / q90 for 1-day and 5-day ahead<br>
    Quantile loss (not MSE) gives a full probability distribution — essential for the DP optimizer.
    Target: 80% of actuals inside the q10–q90 band (coverage = 0.78–0.82 = well calibrated).
    """)

    df     = get_features(pair)
    bundle = get_model()

    if not bundle:
        st.warning("No trained model found. Run: `python run_pipeline.py --steps forecast`")
    else:
        m = bundle["metrics"]
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Test set metrics**")
            rows = [
                ("1-Day MAE",          f"{m.get('h1d_mae',0):.4f}",      "< 0.006"),
                ("1-Day Directional",  f"{m.get('h1d_dir_acc',0):.4f}",  "≥ 0.52"),
                ("1-Day Coverage",     f"{m.get('h1d_coverage',0):.4f}", "target 0.78–0.82"),
                ("5-Day MAE",          f"{m.get('h5d_mae',0):.4f}",      "< 0.008"),
                ("5-Day Directional",  f"{m.get('h5d_dir_acc',0):.4f}",  "≥ 0.52"),
                ("5-Day Coverage",     f"{m.get('h5d_coverage',0):.4f}", "target 0.78–0.82"),
            ]
            st.dataframe(pd.DataFrame(rows,columns=["Metric","Value","Target"]),
                         use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**Latest forecast**")
            try:
                from src.forecasting.transformer_model import predict
                pred_df = predict(df, bundle)
                last  = pred_df.iloc[-1]
                price = float(df["price"].iloc[-1])
                st.metric("Current price", f"{price:.5f}")
                for h in PRED_HORIZONS:
                    cc = st.columns(3)
                    cc[0].metric(f"{h}D q10", f"{price*(1+last.get(f'h{h}d_q10',0)):.5f}")
                    cc[1].metric(f"{h}D q50", f"{price*(1+last.get(f'h{h}d_q50',0)):.5f}")
                    cc[2].metric(f"{h}D q90", f"{price*(1+last.get(f'h{h}d_q90',0)):.5f}")
            except Exception as e:
                st.error(str(e))

        st.divider()
        st.markdown("**Test any historical date — see predicted vs actual**")
        c1,c2 = st.columns([2,1])
        test_d = c1.date_input("Date", value=datetime(2022,9,28),
                               min_value=datetime(2015,6,1),
                               max_value=datetime(2024,11,1))
        if c2.button("Run", type="primary"):
            try:
                from src.forecasting.transformer_model import predict
                pred_df = predict(df, bundle)
                idx  = min(pred_df.index.searchsorted(str(test_d)), len(pred_df)-1)
                row  = pred_df.iloc[idx]
                price= float(df["price"].iloc[idx])
                rows = []
                for h in PRED_HORIZONS:
                    q50 = row.get(f"h{h}d_q50",0)
                    q10 = row.get(f"h{h}d_q10",0)
                    q90 = row.get(f"h{h}d_q90",0)
                    try:
                        ar  = float(df[f"log_ret_{h}d"].iloc[idx])
                        ap  = price*(1+ar)
                        err = (price*(1+q50)-ap)/ap*10_000
                        rows.append({
                            "Horizon":         f"{h}D",
                            "Predicted (q50)": round(price*(1+q50),5),
                            "Actual":          round(ap,5),
                            "Error (bps)":     round(err,1),
                            "Direction":       "✓" if q50*ar>0 else "✗",
                            "In 80% band":     "✓" if q10<=ar<=q90 else "✗",
                        })
                    except: pass
                if rows:
                    st.dataframe(pd.DataFrame(rows),
                                 use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 3. DP OPTIMAL STOPPING
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "3. DP Optimal Stopping":
    st.title("3. DP Optimal Stopping")
    note("""
    Solves the treasurer's problem: <i>execute now or wait?</i><br>
    Backward induction from deadline → today gives a <b>threshold curve</b>:
    the minimum rate to accept per day remaining.
    As deadline approaches, threshold drops — you become less selective.
    """)

    df = get_features(pair)
    from src.optimization.fx_optimizer import (
        estimate_gbm_params, dp_optimal_stopping, dp_execute_decision
    )
    mu_def, sig_def = estimate_gbm_params(df["log_ret_1d"].dropna())
    cur = float(df["price"].iloc[-1])

    st.markdown("**Adjust inputs — results update on click**")
    c1,c2,c3 = st.columns(3)
    with c1:
        mu      = st.slider("Annual drift μ",        -0.10,0.15,float(round(mu_def,3)),0.005)
        sigma   = st.slider("Annual vol σ",           0.02,0.30,float(round(sig_def,3)),0.005)
    with c2:
        horizon = st.slider("Deadline (trading days)", 3,63,OPT_HORIZON,1)
        tx_bps  = st.slider("Transaction cost (bps)",  0,20,int(TRANSACTION_COST*10_000),1)
    with c3:
        test_px   = st.number_input("Spot rate to test", value=float(round(cur,5)),
                                    format="%.5f", step=0.001)
        test_days = st.slider("Days remaining", 1, horizon, horizon//2)

    if st.button("▶  Solve", type="primary"):
        t0  = time.time()
        res = dp_optimal_stopping(horizon=horizon, mu=mu, sigma=sigma,
                                  transaction_cost=tx_bps/10_000,
                                  current_price=test_px)
        curve = res["threshold_curve"]
        st.caption(f"Solved in {time.time()-t0:.2f}s")

        go_flag, thresh = dp_execute_decision(test_px, test_days, res)
        gap = (test_px - thresh)/thresh*10_000
        if go_flag:
            st.markdown(f'<div class="go">✅ EXECUTE — {test_px:.5f} is {gap:+.1f} bps above threshold {thresh:.5f} with {test_days}d left</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="wait">⏳ WAIT — {test_px:.5f} is {gap:+.1f} bps below threshold {thresh:.5f} with {test_days}d left</div>',
                        unsafe_allow_html=True)

        cl, cr = st.columns(2)
        with cl:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(curve))), y=curve.values,
                                     line=dict(color="#ef4444",width=2.5),name="Threshold"))
            fig.add_hline(y=test_px, line_dash="dash", line_color="#22c55e",
                          annotation_text=f"Test rate {test_px:.5f}")
            fig.add_vline(x=len(curve)-test_days, line_dash="dot",
                          line_color="#64748b", annotation_text=f"{test_days}d left")
            fig.update_layout(height=300, template="plotly_white",
                              margin=dict(t=10,b=10),
                              xaxis_title="Days elapsed",
                              yaxis_title="Min rate to execute")
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            rows = []
            for d in sorted({horizon,horizon*3//4,horizon//2,10,7,5,3,1}):
                if 0 < d <= len(curve):
                    t = curve.iloc[d-1]
                    rows.append({
                        "Days left": d,
                        "Threshold": round(t,5),
                        "Gap (bps)": round((t-test_px)/test_px*10_000,1),
                        "Decision":  "✅ GO" if test_px>=t else "⏳ WAIT",
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True, height=290)
    else:
        st.info("👆 Adjust inputs then click **Solve**")


# ══════════════════════════════════════════════════════════════════════════════
# 4. RL AGENT & BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "4. RL Agent & Backtest":
    st.title("4. RL Agent & Backtest")
    note("""
    DP is optimal under GBM — but real FX has fat tails and regime switches.
    The <b>PPO agent</b> learns directly from trajectories with no distributional assumptions.<br>
    Backtest compares 3 strategies across N walk-forward episodes from the same start price.
    """)

    df = get_features(pair)

    c1,c2,c3 = st.columns(3)
    n_ep   = c1.slider("Episodes",              50,400,200,50)
    bt_hor = c2.slider("Horizon (days)",         5, 63,OPT_HORIZON,1)
    bt_tx  = c3.slider("Transaction cost (bps)", 0, 20,int(TRANSACTION_COST*10_000),1)
    use_rl = st.checkbox("Include RL agent", value=True)

    if st.button("▶  Run Backtest", type="primary"):
        from src.optimization.fx_optimizer import (
            estimate_gbm_params, dp_optimal_stopping, backtest_strategies
        )
        mu, sigma = estimate_gbm_params(df["log_ret_1d"].dropna())
        dp_res = dp_optimal_stopping(horizon=bt_hor, mu=mu, sigma=sigma,
                                     transaction_cost=bt_tx/10_000,
                                     current_price=float(df["price"].iloc[-1]))
        rl_model = None
        if use_rl:
            mp = Path("models/optimization/ppo_fx_agent.zip")
            if mp.exists():
                try:
                    from stable_baselines3 import PPO
                    rl_model = PPO.load(str(mp.with_suffix("")))
                except Exception as e:
                    st.warning(f"RL load failed: {e}")
            else:
                st.warning("No saved RL model. Run: `python run_pipeline.py --steps optimize`")

        with st.spinner(f"Running {n_ep} episodes…"):
            bt = backtest_strategies(df, dp_res, rl_model=rl_model,
                                     n_episodes=n_ep, horizon=bt_hor,
                                     transaction_cost=bt_tx/10_000)

        cols = st.columns(3)
        cols[0].metric("Naive", "0.00 bps", "baseline")
        for i,(cn,lbl) in enumerate([("dp_bps","DP"),("rl_bps","RL PPO")]):
            if cn in bt.columns:
                s = bt[cn].dropna()
                cols[i+1].metric(lbl, f"{s.mean():+.2f} bps",
                                 f"Win rate {(s>0).mean():.0%}")

        fig = go.Figure()
        for cn,col,nm in [("dp_bps","#3b82f6","DP"),("rl_bps","#22c55e","RL PPO")]:
            if cn in bt.columns:
                fig.add_trace(go.Histogram(x=bt[cn],name=nm,
                                           opacity=0.7,marker_color=col,nbinsx=40))
        fig.add_vline(x=0,line_dash="dash",line_color="#ef4444",annotation_text="Naive")
        fig.update_layout(barmode="overlay",height=300,template="plotly_white",
                          margin=dict(t=10,b=10),
                          xaxis_title="Bps vs naive",yaxis_title="Episodes",
                          legend=dict(orientation="h",y=1.05))
        st.plotly_chart(fig, use_container_width=True)

        rows = [{"Strategy":"Naive","Mean":0,"Std":0,"Win%":"—","P25":"—","P75":"—"}]
        for cn,lbl in [("dp_bps","DP Optimal"),("rl_bps","RL PPO")]:
            if cn in bt.columns:
                s = bt[cn].dropna()
                rows.append({"Strategy":lbl,"Mean":round(s.mean(),2),
                             "Std":round(s.std(),2),"Win%":f"{(s>0).mean():.0%}",
                             "P25":round(s.quantile(.25),2),"P75":round(s.quantile(.75),2)})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    else:
        st.info("👆 Set parameters then click **Run Backtest**")


# ══════════════════════════════════════════════════════════════════════════════
# 5. RAG
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "5. RAG Treasury Assistant":
    st.title("5. RAG Treasury Assistant")
    note("""
    RAG-powered Q&A grounded in curated FX/payments documents.<br>
    <b>Embed:</b> HuggingFace API (BAAI/bge-small-en-v1.5, zero local RAM) &nbsp;·&nbsp; <b>Store:</b> Plain JSON &nbsp;·&nbsp; <b>Generate:</b> Groq llama-3.3-70b
    """)

    # Look for the key in Streamlit secrets first (Cloud), then fall back to configs (Local)
    groq_key = st.secrets.get("GROQ_API_KEY", GROQ_API_KEY)
    
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not set in .env")
        st.stop()

    suggestions = [
        "What drives short-term EURUSD movements?",
        "How does volatility regime affect execution strategy?",
        "What is CVaR and how is it used in FX risk management?",
        "Explain optimal stopping for a 21-day FX deadline",
        "What is SWIFT T+2 settlement and why does timing matter?",
        "How does the RL agent differ from DP?",
    ]

    c1,c2 = st.columns([3,1])
    with c1:
        picked = st.selectbox("Suggested question", ["— type your own below —"]+suggestions)
        custom = st.text_input("Or type your own", placeholder="Ask anything about FX…")
        query  = custom if custom else (picked if "type" not in picked else "")
    with c2:
        show_chunks = st.checkbox("Show retrieved chunks", value=False)

    if st.button("▶  Ask", type="primary", disabled=not bool(query)):
        with st.spinner("Retrieving + generating…"):
            try:
                from src.rag.treasury_rag import TreasuryAssistant
                if "rag" not in st.session_state:
                    st.session_state.rag = TreasuryAssistant(force_rebuild=False)
                result = st.session_state.rag.chat(query)
                st.markdown(result["answer"])
                st.caption(f"Sources: {', '.join(result['sources'])}")
                if show_chunks:
                    for i,ch in enumerate(result.get("chunks",[])[:3],1):
                        with st.expander(f"Chunk {i} — {ch['source']} (relevance {ch['relevance_score']})"):
                            st.text(ch["text"][:500])
            except Exception as e:
                st.error(f"{e}")
                st.info("Run `python run_pipeline.py --steps rag` first")

    if st.session_state.get("rag") and st.session_state.rag.history:
        st.divider()
        st.markdown("**Conversation History**")
    
    # Display the last 6 turns safely
        for msg in st.session_state.rag.history[-6:]:
            if msg["role"] == "user":
                prefix = "🧑 **You:**"
                # Truncate user query elegantly at a word boundary if it's too long
                content = smart_truncate(msg["content"], max_chars=400)
            else:
                prefix = "🤖 **Assistant:**"
                # Show the complete response so the interviewer can read the full answer
                content = msg["content"]
            
            st.markdown(f"{prefix} {content}")
        
        if st.button("Clear History"):
            st.session_state.rag.reset()
            st.rerun()

