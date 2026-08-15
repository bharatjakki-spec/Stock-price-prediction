"""
===================================================================================
  QuantPulse AI — Streamlit Mobile Stock Prediction App
===================================================================================
  Run with:
      streamlit run streamlit_app.py

  Features:
  • Live data from Yahoo Finance (yfinance)
  • XGBoost & Random Forest classifiers
  • 17 technical indicator features (RSI, MACD, Bollinger, SMA, EMA, ATR, lags)
  • Time-series split (no data leakage)
  • Interactive Plotly charts (touch-friendly for mobile)
  • Backtest equity curve vs Buy & Hold
  • Feature importance ranking
  • Mobile-first responsive layout
===================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)
import xgboost as xgb

# ─── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="QuantPulse AI",
    page_icon="📈",
    layout="wide",          # use full width on desktop; collapses nicely on mobile
    initial_sidebar_state="expanded",
    menu_items={
        "About": "QuantPulse AI — ML-powered stock market direction forecasting."
    }
)

# ─── Global CSS — mobile-first polish ────────────────────────────────────────
st.markdown("""
<style>
    /* ── Fonts ─────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Background ────────────────────── */
    .stApp {
        background: #07090f;
        background-image:
            radial-gradient(ellipse 70% 40% at 10% 5%, rgba(61,142,248,0.08) 0%, transparent 60%),
            radial-gradient(ellipse 60% 50% at 90% 95%, rgba(0,200,150,0.06) 0%, transparent 60%);
    }

    /* ── Sidebar ─────────────────────── */
    [data-testid="stSidebar"] {
        background: #0e1119 !important;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    [data-testid="stSidebar"] * {
        color: #eef0f5 !important;
    }

    /* ── Metric cards ───────────────── */
    [data-testid="stMetric"] {
        background: #0e1119;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px 16px !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #4f576d !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.5rem !important;
        color: #eef0f5 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
    }

    /* ── Headings ────────────────────── */
    h1, h2, h3 { color: #eef0f5 !important; }

    /* ── Buttons ─────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #0d9f75, #00c896) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 4px 18px rgba(0,200,150,0.25) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        box-shadow: 0 6px 26px rgba(0,200,150,0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Selectbox / Input ───────────── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stDateInput > div > div > input {
        background: #131722 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        color: #eef0f5 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Expander ────────────────────── */
    .streamlit-expanderHeader {
        background: #0e1119 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: #eef0f5 !important;
        font-weight: 600 !important;
    }

    .streamlit-expanderContent {
        background: #0e1119 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* ── Info / Success / Warning boxes ─ */
    .stAlert {
        border-radius: 10px !important;
        font-size: 0.85rem !important;
    }

    /* ── Divider ─────────────────────── */
    hr { border-color: rgba(255,255,255,0.07) !important; }

    /* ── Tab bar ─────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: #0e1119 !important;
        border-radius: 10px !important;
        gap: 4px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 7px !important;
        color: #8b92a8 !important;
        font-weight: 500 !important;
    }

    .stTabs [aria-selected="true"] {
        background: #1a2030 !important;
        color: #eef0f5 !important;
    }

    /* ── Progress bar ────────────────── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3d8ef8, #00c896) !important;
        border-radius: 4px;
    }

    /* ── Spinner ─────────────────────── */
    .stSpinner > div {
        border-top-color: #00c896 !important;
    }

    /* ── Dataframe ───────────────────── */
    .dataframe {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Plotly dark theme ────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#07090f",
    plot_bgcolor="#0e1119",
    font=dict(family="Inter", color="#8b92a8", size=11),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(
        bgcolor="rgba(14,17,25,0.8)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(size=10)
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(family="JetBrains Mono", size=9)
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(family="JetBrains Mono", size=9)
    )
)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.
    Falls back to synthetic Geometric Brownian Motion data if offline.
    Results are cached for 1 hour to avoid repeated API calls.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError("Empty response from Yahoo Finance.")
        # Handle MultiIndex columns from newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Prefer Adj Close for corporate-action-adjusted prices
        df["Price"] = df.get("Adj Close", df.get("Close"))
        return df.dropna()
    except Exception as e:
        st.warning(f"⚠️ Yahoo Finance unavailable ({e}). Using synthetic GBM data for demo.")
        return _synthetic_data(start, end)


def _synthetic_data(start: str, end: str) -> pd.DataFrame:
    """
    Generate realistic stock price path using Geometric Brownian Motion.
    Used as offline fallback for testing/demo purposes.
    """
    np.random.seed(42)
    dates = pd.date_range(start=start, end=end, freq="B")
    n = len(dates)
    dt = 1 / 252
    mu, sigma = 0.12, 0.25
    returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
    price = 150.0 * np.exp(np.cumsum(returns))
    df = pd.DataFrame(index=dates)
    df["Price"] = price
    df["Adj Close"] = price
    df["Close"]  = price * (1 + np.random.normal(0, 0.002, n))
    df["Open"]   = df["Close"] * (1 + np.random.normal(0, 0.005, n))
    df["High"]   = np.maximum(df["Open"], df["Close"]) * (1 + np.abs(np.random.normal(0, 0.008, n)))
    df["Low"]    = np.minimum(df["Open"], df["Close"]) * (1 - np.abs(np.random.normal(0, 0.008, n)))
    df["Volume"] = np.random.lognormal(16, 0.5, n).astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 17 technical indicator features from OHLCV data.
    Uses strictly past data at each point in time (no lookahead bias).

    Feature groups:
    1. Log returns (stationarization)
    2. Moving average trend ratios (SMA 10, SMA 50, EMA 20)
    3. MACD & Signal line momentum
    4. RSI 14-day momentum oscillator
    5. Bollinger Bands %B and bandwidth
    6. Realized volatility (20-day rolling std of log returns)
    7. Volume momentum
    8. Lagged return features (t-1, t-2, t-3, t-5, t-10)
    """
    d = df.copy()

    # 1. Log Returns — converts non-stationary price to stationary return series
    d["log_ret"] = np.log(d["Price"] / d["Price"].shift(1))

    # 2. Moving Averages & Ratio Features
    d["sma_10"]       = d["Price"].rolling(10).mean()
    d["sma_50"]       = d["Price"].rolling(50).mean()
    d["ema_20"]       = d["Price"].ewm(span=20, adjust=False).mean()
    d["sma_10_ratio"] = (d["Price"] / d["sma_10"]) - 1.0
    d["sma_50_ratio"] = (d["Price"] / d["sma_50"]) - 1.0
    d["ema_20_ratio"] = (d["Price"] / d["ema_20"]) - 1.0

    # 3. MACD — trend momentum crossover indicator
    ema12 = d["Price"].ewm(span=12, adjust=False).mean()
    ema26 = d["Price"].ewm(span=26, adjust=False).mean()
    d["macd"]        = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]

    # 4. RSI — measures speed and change of price movements
    delta    = d["Price"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    d["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # 5. Bollinger Bands — volatility envelope around 20-day SMA
    ma20  = d["Price"].rolling(20).mean()
    std20 = d["Price"].rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    d["bollinger_pct_b"]    = (d["Price"] - lower) / (upper - lower + 1e-9)
    d["bollinger_bandwidth"] = (upper - lower) / (ma20 + 1e-9)

    # 6. Realized Volatility & Volume Momentum
    d["volatility_20"]      = d["log_ret"].rolling(20).std()
    d["volume_pct_change"]  = d["Volume"].pct_change()
    d["volume_sma_ratio"]   = d["Volume"] / (d["Volume"].rolling(20).mean() + 1e-9) - 1.0

    # 7. Lagged Return Features — capture serial autocorrelation patterns
    for lag in [1, 2, 3, 5]:
        d[f"log_ret_lag_{lag}"] = d["log_ret"].shift(lag)

    # Targets: next-day direction (1=UP, 0=DOWN) and next-day return (regression)
    d["target_return"]    = d["log_ret"].shift(-1)
    d["target_direction"] = (d["target_return"] > 0).astype(int)

    return d.dropna()


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "log_ret", "sma_10_ratio", "sma_50_ratio", "ema_20_ratio",
    "macd", "macd_signal", "macd_hist", "rsi_14",
    "bollinger_pct_b", "bollinger_bandwidth", "volatility_20",
    "volume_pct_change", "volume_sma_ratio",
    "log_ret_lag_1", "log_ret_lag_2", "log_ret_lag_3", "log_ret_lag_5"
]


def train_model(df: pd.DataFrame, model_type: str, train_ratio: float = 0.80):
    """
    Perform strict chronological time-series split, scale features,
    and train the selected classification model.

    Important: StandardScaler is fitted ONLY on training data to prevent
    data leakage from future test observations into the training distribution.
    """
    X = df[FEATURE_COLS]
    y = df["target_direction"]

    split = int(len(df) * train_ratio)

    X_train_raw = X.iloc[:split]
    X_test_raw  = X.iloc[split:]
    y_train     = y.iloc[:split]
    y_test      = y.iloc[split:]

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # Initialize model
    if model_type == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss", verbosity=0
        )
    else:  # Random Forest
        model = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_split=5,
            random_state=42, n_jobs=-1
        )

    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Classification metrics
    acc    = accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    # Feature importances
    fi = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    # Backtest simulation (long/short strategy)
    prices_test    = df["Price"].iloc[split:]
    actual_returns = np.log(prices_test / prices_test.shift(1)).dropna()
    signals        = pd.Series(y_pred[:-1], index=actual_returns.index)
    positions      = np.where(signals > 0, 1.0, -1.0)
    strat_returns  = positions * actual_returns
    cum_strat      = np.exp(np.cumsum(strat_returns))
    cum_bench      = np.exp(np.cumsum(actual_returns))

    sharpe_strat  = float(np.sqrt(252) * strat_returns.mean() / (strat_returns.std() + 1e-9))
    sharpe_bench  = float(np.sqrt(252) * actual_returns.mean() / (actual_returns.std() + 1e-9))
    total_strat   = float((cum_strat.iloc[-1] - 1) * 100)
    total_bench   = float((cum_bench.iloc[-1] - 1) * 100)

    return {
        "model": model,
        "scaler": scaler,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": round(acc, 2),
        "precision": round(report["1"]["precision"], 4),
        "recall": round(report["1"]["recall"], 4),
        "f1": round(report["1"]["f1-score"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "cm": cm,
        "fi": fi,
        "dates_test": df.index[split:],
        "prices_test": prices_test,
        "cum_strat": cum_strat,
        "cum_bench": cum_bench,
        "sharpe_strat": round(sharpe_strat, 2),
        "sharpe_bench": round(sharpe_bench, 2),
        "total_strat": round(total_strat, 2),
        "total_bench": round(total_bench, 2),
        "next_dir": "📈 UP (Bullish)" if int(y_pred[-1]) == 1 else "📉 DOWN (Bearish)",
        "next_conf": round(float(y_proba[-1]) * 100, 1),
        "train_size": split,
        "test_size": len(X_test),
        "df_clean": df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def chart_price(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick + SMA 10 + SMA 50 overlay chart."""
    fig = go.Figure()

    # Candlestick bars (only if High/Low/Open available)
    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color="#00c896", decreasing_line_color="#f84d4d",
            name=ticker, showlegend=True
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Price"], name=ticker,
            line=dict(color="#3d8ef8", width=1.6), fill="tozeroy",
            fillcolor="rgba(61,142,248,0.06)"
        ))

    # SMA overlays
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_10"], name="SMA 10",
                             line=dict(color="#00c896", width=1.3, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50",
                             line=dict(color="#f5a623", width=1.3, dash="dash")))

    fig.update_layout(**PLOTLY_LAYOUT, title=f"{ticker} — Price & Moving Averages",
                      xaxis_rangeslider_visible=False)
    return fig


def chart_macd(df: pd.DataFrame) -> go.Figure:
    """MACD line, signal line, and histogram."""
    colors = ["#00c896" if v >= 0 else "#f84d4d" for v in df["macd_hist"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"],
                         marker_color=colors, name="Histogram", opacity=0.6))
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"],
                             line=dict(color="#3d8ef8", width=1.5), name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
                             line=dict(color="#f84d4d", width=1.2, dash="dot"), name="Signal"))
    fig.update_layout(**PLOTLY_LAYOUT, title="MACD — Momentum Indicator")
    fig.update_layout(
    **PLOTLY_LAYOUT, 
    title="MACD" 
    )
    return fig


def chart_rsi(df: pd.DataFrame) -> go.Figure:
    """RSI with overbought/oversold reference bands."""
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,77,77,0.07)",
                  layer="below", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,200,150,0.07)",
                  layer="below", line_width=0)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(248,77,77,0.4)", line_width=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,200,150,0.4)", line_width=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"],
                             line=dict(color="#f5a623", width=1.6), name="RSI 14"))
    fig.update_layout(**PLOTLY_LAYOUT, title="RSI 14-Day Momentum Oscillator",
                      yaxis_range=[0, 100])
    return fig


def chart_bollinger(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Bollinger Bands envelope with price."""
    ma20  = df["Price"].rolling(20).mean()
    std20 = df["Price"].rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=upper, name="Upper Band",
                             line=dict(color="rgba(61,142,248,0.4)", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=lower, name="Lower Band",
                             line=dict(color="rgba(61,142,248,0.4)", width=1),
                             fill="tonexty", fillcolor="rgba(61,142,248,0.05)"))
    fig.add_trace(go.Scatter(x=df.index, y=ma20, name="SMA 20",
                             line=dict(color="#f5a623", width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["Price"], name=ticker,
                             line=dict(color="#3d8ef8", width=1.5)))
    fig.update_layout(**PLOTLY_LAYOUT, title="Bollinger Bands (20-Day, 2σ)")
    return fig


def chart_volume(df: pd.DataFrame) -> go.Figure:
    """Volume bar chart with 20-day SMA overlay."""
    vol_avg = df["Volume"].rolling(20).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                         marker_color="rgba(61,142,248,0.4)", name="Volume"))
    fig.add_trace(go.Scatter(x=df.index, y=vol_avg,
                             line=dict(color="#00c896", width=1.4), name="SMA 20"))
    fig.update_layout(**PLOTLY_LAYOUT, title="Trading Volume & 20-Day Average")
    return fig


def chart_backtest(result: dict) -> go.Figure:
    """Cumulative equity curve: model strategy vs buy & hold."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result["cum_strat"].index, y=result["cum_strat"],
        name=f"Model Strategy (Sharpe: {result['sharpe_strat']})",
        line=dict(color="#00c896", width=2),
        fill="tozeroy", fillcolor="rgba(0,200,150,0.06)"
    ))
    fig.add_trace(go.Scatter(
        x=result["cum_bench"].index, y=result["cum_bench"],
        name=f"Buy & Hold (Sharpe: {result['sharpe_bench']})",
        line=dict(color="#3d8ef8", width=1.5, dash="dash")
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="Strategy Equity Curve vs. Buy & Hold ($1.00 invested)")
    return fig


def chart_feature_importance(fi: pd.Series) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    top = fi.head(15)
    colors = [
        f"rgba(61,142,248,{0.5 + 0.5 * v / top.max()})"
        for v in top.values
    ]
    fig = go.Figure(go.Bar(
        x=top.values[::-1], y=top.index[::-1],
        orientation="h", marker_color=colors[::-1],
        text=[f"{v*100:.2f}%" for v in top.values[::-1]],
        textposition="outside", textfont=dict(size=9, color="#8b92a8")
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Feature Importance Ranking",
                      xaxis_title="Importance Score",
                      margin=dict(l=120, r=60, t=36, b=10))
    return fig


def chart_confusion(cm: np.ndarray) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    labels = ["DOWN (0)", "UP (1)"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Predicted DOWN", "Predicted UP"],
        y=["Actual DOWN", "Actual UP"],
        colorscale=[[0, "#0e1119"], [1, "#3d8ef8"]],
        showscale=False,
        text=cm, texttemplate="%{text}",
        textfont=dict(size=18, color="white")
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title="Directional Confusion Matrix",
                      xaxis=dict(side="bottom"))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.2rem;">
        <div style="font-size:2.2rem;">📈</div>
        <div style="font-size:1.3rem; font-weight:700; color:#eef0f5;">QuantPulse <span style='color:#00c896'>AI</span></div>
        <div style="font-size:0.72rem; color:#4f576d; margin-top:2px;">ML Stock Forecasting Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Configuration")

    # Ticker with quick-picks
    ticker = st.text_input("Ticker Symbol", value="AAPL",
                           placeholder="e.g. AAPL, MSFT, NVDA").upper().strip()

    QUICK = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "SPY", "QQQ", "BTC-USD"]
    st.markdown("<div style='font-size:0.7rem; color:#4f576d; margin-bottom:4px;'>Quick picks:</div>",
                unsafe_allow_html=True)
    cols = st.columns(5)
    for i, t in enumerate(QUICK):
        if cols[i % 5].button(t, key=f"qp_{t}",
                              help=f"Load {t}",
                              use_container_width=True):
            ticker = t
            st.session_state["ticker_override"] = t

    if "ticker_override" in st.session_state:
        ticker = st.session_state["ticker_override"]

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.Timestamp("2016-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.Timestamp("2024-01-01"))

    model_type = st.selectbox(
        "Algorithm",
        ["XGBoost", "Random Forest"],
        help="XGBoost: gradient boosted trees with L1/L2 regularization.\nRandom Forest: bagged decision tree ensemble."
    )

    train_ratio = st.slider(
        "Train / Test Split",
        min_value=0.60, max_value=0.90, value=0.80, step=0.05,
        format="%.0f%%",
        help="Fraction of data used for training. Remainder is held out for testing."
    )

    st.divider()
    run_clicked = st.button("⚡  Run Forecast", use_container_width=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem; color:#4f576d; line-height:1.7;">
    <b style="color:#8b92a8;">About</b><br>
    Uses XGBoost / Random Forest trained on 17 technical features to predict next-day market direction.
    Strict chronological train/test split prevents data leakage.
    <br><br>
    <b style="color:#8b92a8;">Disclaimer</b><br>
    For educational purposes only. Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ── Main Page ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<h1 style="margin-bottom:0; font-size:1.6rem;">
    📈 QuantPulse <span style="color:#00c896;">AI</span>
    <span style="font-size:0.85rem; color:#4f576d; font-weight:400; margin-left:8px;">Stock Market Forecasting</span>
</h1>
<p style="color:#4f576d; font-size:0.82rem; margin-top:4px; margin-bottom:1.2rem;">
    ML-powered directional prediction using {model_type} · {ticker}
</p>
""", unsafe_allow_html=True)

# ── State Management ──────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None

# ── Run Pipeline ──────────────────────────────────────────────────────────────
if run_clicked:
    st.session_state.result = None
    st.session_state.df_clean = None

    with st.spinner(f"Fetching {ticker} data and training {model_type}..."):
        progress = st.progress(0)

        # Step 1: Fetch
        progress.progress(15, "📥 Downloading market data...")
        raw_df = fetch_data(ticker, str(start_date), str(end_date))
        st.session_state.raw_df = raw_df
        progress.progress(40, "⚙️ Engineering features...")

        # Step 2: Features
        df_clean = engineer_features(raw_df)
        st.session_state.df_clean = df_clean
        progress.progress(65, f"🤖 Training {model_type}...")

        # Step 3: Train & Evaluate
        result = train_model(df_clean, model_type, train_ratio)
        st.session_state.result = result
        progress.progress(90, "📊 Computing backtest...")
        progress.progress(100, "✅ Done!")
        progress.empty()

    st.success(
        f"✅ **{ticker}** — {len(df_clean):,} trading days · "
        f"Train: {result['train_size']:,} · Test: {result['test_size']:,}"
    )

# ── Display Results ───────────────────────────────────────────────────────────
result   = st.session_state.result
df_clean = st.session_state.df_clean

if result is None:
    # Welcome screen
    st.markdown("""
    <div style="
        background: #0e1119;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 2.5rem;
        text-align: center;
        margin-top: 1rem;
    ">
        <div style="font-size:3rem; margin-bottom:0.8rem;">🚀</div>
        <h2 style="font-size:1.3rem; color:#eef0f5; margin-bottom:0.5rem;">Ready to Forecast</h2>
        <p style="color:#8b92a8; font-size:0.88rem; max-width:480px; margin:0 auto 1.2rem;">
            Configure your ticker and model in the sidebar, then tap
            <strong style="color:#00c896;">⚡ Run Forecast</strong> to train the ML model
            and get a directional prediction.
        </p>
        <div style="display:flex; justify-content:center; gap:1rem; flex-wrap:wrap; font-size:0.8rem; color:#4f576d;">
            <span>📥 Yahoo Finance data</span>
            <span>⚙️ 17 technical features</span>
            <span>🤖 XGBoost / Random Forest</span>
            <span>📈 Interactive Plotly charts</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── KPI Cards ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    dir_color = "normal" if "UP" in result["next_dir"] else "inverse"
    acc_delta = f"+{result['accuracy'] - 50:.1f}% vs random" if result["accuracy"] >= 50 else f"{result['accuracy'] - 50:.1f}% vs random"

    k1.metric("🎯 Directional Accuracy", f"{result['accuracy']}%", acc_delta)
    k2.metric("📊 Strategy Return",
              f"{'+' if result['total_strat'] > 0 else ''}{result['total_strat']}%",
              f"Benchmark: {result['total_bench']}%",
              delta_color="normal" if result["total_strat"] >= result["total_bench"] else "inverse")
    k3.metric("⚡ Sharpe Ratio",
              result["sharpe_strat"],
              f"vs Benchmark: {result['sharpe_bench']}",
              delta_color="normal" if result["sharpe_strat"] >= result["sharpe_bench"] else "inverse")
    k4.metric("🔮 Next-Day Signal",
              result["next_dir"],
              f"Confidence: {result['next_conf']}%",
              delta_color=dir_color)

    st.divider()

    # ── Tabs for Charts ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📉 Price", "⚡ Momentum", "📈 Backtest",
        "🔍 Features", "📊 Metrics", "📚 Theory"
    ])

    # ── TAB 1: Price Charts ────────────────────────────────────────────────────
    with tab1:
        st.plotly_chart(chart_price(df_clean, ticker),
                        use_container_width=True, config={"displaylogo": False})
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_bollinger(df_clean, ticker),
                            use_container_width=True, config={"displaylogo": False})
        with c2:
            st.plotly_chart(chart_volume(df_clean),
                            use_container_width=True, config={"displaylogo": False})

    # ── TAB 2: Momentum Indicators ────────────────────────────────────────────
    with tab2:
        st.plotly_chart(chart_rsi(df_clean),
                        use_container_width=True, config={"displaylogo": False})
        st.plotly_chart(chart_macd(df_clean),
                        use_container_width=True, config={"displaylogo": False})

    # ── TAB 3: Backtest ───────────────────────────────────────────────────────
    with tab3:
        st.plotly_chart(chart_backtest(result),
                        use_container_width=True, config={"displaylogo": False})

        b1, b2, b3, b4 = st.columns(4)
        alpha = round(result["total_strat"] - result["total_bench"], 2)
        b1.metric("Strategy Return", f"{result['total_strat']}%")
        b2.metric("Benchmark Return", f"{result['total_bench']}%")
        b3.metric("Alpha (Excess Return)", f"{'+' if alpha>0 else ''}{alpha}%",
                  delta_color="normal" if alpha > 0 else "inverse")
        b4.metric("Strategy Sharpe", result["sharpe_strat"])

    # ── TAB 4: Feature Importance ─────────────────────────────────────────────
    with tab4:
        st.plotly_chart(chart_feature_importance(result["fi"]),
                        use_container_width=True, config={"displaylogo": False})

        fi_df = result["fi"].reset_index()
        fi_df.columns = ["Feature", "Importance Score"]
        fi_df["Rank"] = range(1, len(fi_df) + 1)
        fi_df["Importance %"] = (fi_df["Importance Score"] * 100).round(3).astype(str) + "%"
        st.dataframe(
            fi_df[["Rank", "Feature", "Importance %"]].set_index("Rank"),
            use_container_width=True
        )

    # ── TAB 5: Metrics & Diagnostics ──────────────────────────────────────────
    with tab5:
        col_m, col_cm = st.columns([1, 1])

        with col_m:
            st.markdown("##### Classification Report")
            metrics_data = {
                "Metric": ["Directional Accuracy", "Precision (UP)", "Recall (UP)",
                           "F1-Score (UP)", "Macro F1-Score"],
                "Value": [f"{result['accuracy']}%", result["precision"],
                          result["recall"], result["f1"], result["macro_f1"]]
            }
            st.dataframe(pd.DataFrame(metrics_data).set_index("Metric"),
                         use_container_width=True)

            st.markdown("##### Dataset Summary")
            ds_data = {
                "Property": ["Ticker", "Algorithm", "Total Days",
                             "Training Days", "Test Days", "Features"],
                "Value": [ticker, model_type, len(df_clean),
                         result["train_size"], result["test_size"], len(FEATURE_COLS)]
            }
            st.dataframe(pd.DataFrame(ds_data).set_index("Property"),
                         use_container_width=True)

        with col_cm:
            st.plotly_chart(chart_confusion(result["cm"]),
                            use_container_width=True, config={"displaylogo": False})

            # Model signal distribution
            up_pct = float(np.mean(result["y_pred"] == 1)) * 100
            st.markdown("##### Signal Distribution")
            st.markdown(f"""
            <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-top:0.5rem;">
                <div style="background:#0e1119; border:1px solid rgba(0,200,150,0.3);
                     border-radius:8px; padding:0.6rem 1rem; text-align:center; flex:1;">
                    <div style="color:#4f576d; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.8px;">UP Signals</div>
                    <div style="color:#00c896; font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:600;">{up_pct:.1f}%</div>
                </div>
                <div style="background:#0e1119; border:1px solid rgba(248,77,77,0.3);
                     border-radius:8px; padding:0.6rem 1rem; text-align:center; flex:1;">
                    <div style="color:#4f576d; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.8px;">DOWN Signals</div>
                    <div style="color:#f84d4d; font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:600;">{100-up_pct:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 6: Theory & Documentation ─────────────────────────────────────────
    with tab6:
        st.markdown("### 📚 Quantitative Forecasting Framework")

        with st.expander("1. Why Log Returns? (Non-Stationarity)"):
            st.markdown("""
**Problem:** Raw stock prices are non-stationary — their mean and variance change over time,
violating assumptions of most ML models.

**Solution:** We convert prices to **log returns**, which are approximately stationary:

```
r_t = ln(P_t / P_{t-1})
```

Log returns also have nice properties: they're additive over time and bounded near zero,
making them more normally distributed than raw price changes.
            """)

        with st.expander("2. Technical Indicator Feature Engineering"):
            st.markdown("""
| Feature Group | Features | Purpose |
|---|---|---|
| **Trend** | SMA 10/50 ratios, EMA 20 ratio | Identify price vs moving average position |
| **Momentum** | RSI 14, MACD, Signal, Histogram | Measure speed & direction of price change |
| **Volatility** | Bollinger %B, Bandwidth, 20-day vol | Capture expansion/contraction cycles |
| **Volume** | Volume %, Volume SMA ratio | Confirm trend with participation data |
| **Lags** | Log return lags t-1, t-2, t-3, t-5 | Serial autocorrelation patterns |

All features use strictly past data up to day *t* — no look-ahead bias.
            """)

        with st.expander("3. Data Leakage Prevention (Critical!)"):
            st.markdown("""
**The Problem:** Standard K-fold cross-validation randomly shuffles data, causing
future data to leak into training folds. This produces optimistically biased metrics
that fall apart in real trading.

**Our Solution:**
- ✅ **Strict chronological split** — training data comes before test data in time
- ✅ **StandardScaler fitted only on training data** — test statistics don't influence scaling
- ✅ **No shuffle** — data order is preserved to maintain temporal integrity

```
Timeline: ──────────────────[TRAIN 80%]────────────────── | ──[TEST 20%]──▶
                                                               ↑
                                                         Scaler fit only here
```
            """)

        with st.expander("4. Algorithm Comparison"):
            st.markdown("""
| Algorithm | Key Strength | Key Weakness | Best For |
|---|---|---|---|
| **XGBoost** | L1/L2 regularization, gradient boosting | Needs careful tuning | Primary tabular classifier |
| **Random Forest** | Robust, interpretable, low overfitting | No extrapolation | Benchmark + feature ranking |
| **LSTM** *(future)* | Long-range temporal patterns | Data hungry, hard to tune | Sequential multi-step forecasting |
            """)

        with st.expander("5. Backtest Strategy Logic"):
            st.markdown("""
The backtest simulates a simple **long/short strategy**:
- If model predicts **UP** → take a **long (+1)** position
- If model predicts **DOWN** → take a **short (−1)** position

```python
strategy_return = position × actual_return
sharpe = √252 × mean(returns) / std(returns)   # Annualized
```

**Limitations:**
- Does not account for transaction costs or slippage
- Real-world execution would widen the performance gap
- Assumes perfect signal-to-trade execution at daily close
            """)

        with st.expander("6. Key Formulas"):
            st.latex(r"r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)")
            st.latex(r"RSI = 100 - \frac{100}{1 + \frac{\text{AvgGain}_{14}}{\text{AvgLoss}_{14}}}")
            st.latex(r"MACD = EMA_{12}(P) - EMA_{26}(P)")
            st.latex(r"\%B = \frac{P_t - \text{Lower Band}}{\text{Upper Band} - \text{Lower Band}}")
            st.latex(r"Sharpe = \sqrt{252} \cdot \frac{\mu_r}{\sigma_r}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#4f576d; font-size:0.72rem; padding-bottom:1rem;">
    QuantPulse AI · Built with Streamlit, XGBoost, scikit-learn & Plotly ·
    <span style="color:#f84d4d;">⚠️ For educational purposes only — not financial advice.</span>
</div>
""", unsafe_allow_html=True)
