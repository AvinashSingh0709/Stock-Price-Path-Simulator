import sys

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data import fetch_stock_data, estimate_parameters
from simulation import simulate_gbm, get_confidence_intervals
from black_scholes import bs_call, bs_put
from greeks import all_greeks
from risk import compute_risk_metrics

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Path Simulator & Greeks Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #0d1117; color: #e6edf3; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px 20px; text-align: center;
  }
  .metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: .08em; }
  .metric-value { font-size: 1.6rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #58a6ff; }
  .metric-sub { font-size: 0.8rem; color: #8b949e; }
  .greek-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 20px; margin-bottom: 12px;
  }
  .greek-name { font-size: 1.1rem; font-weight: 600; color: #f0f6fc; }
  .greek-val { font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono'; color: #3fb950; }
  .greek-formula { font-family: 'JetBrains Mono'; font-size: 0.78rem; color: #8b949e; margin: 6px 0; background:#0d1117; padding:6px 10px; border-radius:5px; }
  .greek-desc { font-size: 0.82rem; color: #8b949e; line-height: 1.5; }
  .section-header { font-size: 1rem; font-weight: 600; color: #8b949e; letter-spacing:.1em; text-transform:uppercase; margin-bottom:16px; }
  div[data-testid="stTabs"] button { font-family:'Inter'; font-weight:500; }
  .stButton > button { background:#1f6feb; color:#fff; border:none; border-radius:6px; font-weight:500; }
  .stButton > button:hover { background:#388bfd; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  Parameters")
    ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter any valid Yahoo Finance ticker").upper().strip()
    st.markdown("---")
    st.markdown("### Option Parameters")
    strike_pct = st.slider("Strike (% of current price)", 70, 130, 100)
    T_days_input = st.slider("Time to Maturity (days)", 7, 365, 90)
    r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.5) / 100
    option_type = st.radio("Option Type", ["call", "put"])
    st.markdown("---")
    st.markdown("### Simulation")
    sim_horizon = st.selectbox("Simulation Horizon", [30, 90, 252], index=1, format_func=lambda x: f"{x} days")
    run_btn = st.button(" Run Analysis", use_container_width=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("#  Stock Price Path Simulator")
st.markdown("##### Monte Carlo · Black-Scholes · Greeks · Risk Metrics")
st.markdown("---")

# ── Run analysis ─────────────────────────────────────────────────────────────
if run_btn or st.session_state.results is None:
    with st.spinner(f"Fetching data for {ticker}…"):
        try:
            df = fetch_stock_data(ticker)
            params = estimate_parameters(df)
            S0 = params["current_price"]
            K = S0 * strike_pct / 100
            T_years = T_days_input / 365

            paths = simulate_gbm(S0, params["annual_drift"], params["annual_vol"], sim_horizon)
            ci = get_confidence_intervals(paths)

            call_price = bs_call(S0, K, T_years, r, params["annual_vol"])
            put_price = bs_put(S0, K, T_years, r, params["annual_vol"])
            greeks = all_greeks(S0, K, T_years, r, params["annual_vol"], option_type)
            risk = compute_risk_metrics(paths, S0, K)

            st.session_state.results = {
                "df": df, "params": params, "paths": paths, "ci": ci,
                "K": K, "T_years": T_years,
                "call_price": call_price, "put_price": put_price,
                "greeks": greeks, "risk": risk,
                "ticker": ticker, "S0": S0,
                "sim_horizon": sim_horizon,
            }
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

res = st.session_state.results
if res is None:
    st.info("Click **Run Analysis** to start.")
    st.stop()

df = res["df"]
params = res["params"]
paths = res["paths"]
ci = res["ci"]
K = res["K"]
S0 = res["S0"]
greeks_vals = res["greeks"]
risk = res["risk"]

# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
def metric(col, label, value, sub=""):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

metric(c1, "Current Price", f"${S0:.2f}", res["ticker"])
metric(c2, "Annual Volatility", f"{params['annual_vol']*100:.1f}%", "σ annualized")
metric(c3, "Annual Drift", f"{params['annual_drift']*100:.1f}%", "μ annualized")
metric(c4, "Call Price", f"${res['call_price']:.2f}", f"K=${K:.2f}")
metric(c5, "Put Price", f"${res['put_price']:.2f}", f"T={T_days_input}d")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor="#2351b4", plot_bgcolor="#0d1117",
    font_color="#1eb0bb", font_family="Inter",
    xaxis=dict(gridcolor="#ad22c9", linecolor="#30363d"),
    yaxis=dict(gridcolor="#b48240", linecolor="#d2dd36"),
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Historical", " Monte Carlo", " Option Pricing", " Greeks", " Risk Metrics"]
)

# ── TAB 1: Historical ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Historical Price & Returns</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#58a6ff", width=1.5)), row=1, col=1)
    log_ret = params["log_returns"]
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in log_ret]
    fig.add_trace(go.Bar(x=log_ret.index, y=log_ret.values, name="Log Return", marker_color=colors, showlegend=False), row=2, col=1)
    fig.update_layout(**PLOT_THEME, height=500, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(bgcolor="#161b22", bordercolor="#30363d"))
    fig.update_yaxes(title_text="Price ($)", row=1)
    fig.update_yaxes(title_text="Log Return", row=2)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Return Distribution**")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=log_ret, nbinsx=60, name="Returns",
                                    marker_color="#58a6ff", opacity=0.8))
        fig2.update_layout(**PLOT_THEME, height=280, margin=dict(l=0,r=0,t=10,b=0),
                           xaxis_title="Log Return", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown("**Key Statistics**")
        stats = {
            "Mean Daily Return": f"{log_ret.mean()*100:.4f}%",
            "Daily Volatility": f"{log_ret.std()*100:.4f}%",
            "Annual Volatility": f"{params['annual_vol']*100:.2f}%",
            "Annual Drift (μ)": f"{params['annual_drift']*100:.2f}%",
            "Skewness": f"{log_ret.skew():.4f}",
            "Kurtosis": f"{log_ret.kurt():.4f}",
            "Observations": f"{len(log_ret):,}",
        }
        for k, v in stats.items():
            st.markdown(f"**{k}:** `{v}`")

# ── TAB 2: Monte Carlo ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Geometric Brownian Motion Simulation · 1,000 Paths</div>', unsafe_allow_html=True)
    st.latex(r"dS = \mu S \, dt + \sigma S \, dW_t \quad \Rightarrow \quad S_t = S_0 \exp\!\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right]")

    fig3 = go.Figure()
    # Plot sample of paths
    for i in range(0, min(150, paths.shape[1]), 1):
        fig3.add_trace(go.Scatter(x=list(range(paths.shape[0])), y=paths[:, i],
                                   mode="lines", line=dict(width=0.4, color="rgba(88,166,255,0.15)"),
                                   showlegend=False, hoverinfo="skip"))
    fig3.add_trace(go.Scatter(x=list(range(paths.shape[0])), y=ci["p95"], name="95th pct",
                               line=dict(color="#f0883e", width=2, dash="dash")))
    fig3.add_trace(go.Scatter(x=list(range(paths.shape[0])), y=ci["p50"], name="Median",
                               line=dict(color="#3fb950", width=2.5)))
    fig3.add_trace(go.Scatter(x=list(range(paths.shape[0])), y=ci["p5"], name="5th pct",
                               line=dict(color="#f85149", width=2, dash="dash"),
                               fill="tonexty", fillcolor="rgba(88,166,255,0.04)"))
    fig3.add_hline(y=S0, line_color="#8b949e", line_dash="dot", annotation_text="Current Price")
    fig3.update_layout(**PLOT_THEME, height=420, margin=dict(l=0,r=0,t=10,b=0),
                       xaxis_title="Trading Days", yaxis_title="Price ($)",
                       legend=dict(bgcolor="#161b22", bordercolor="#30363d"))
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        final = paths[-1]
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(x=final, nbinsx=60, marker_color="#58a6ff", opacity=0.85, name="Final Price"))
        fig4.add_vline(x=K, line_color="#f0883e", line_dash="dash", annotation_text=f"Strike ${K:.0f}")
        fig4.add_vline(x=S0, line_color="#8b949e", line_dash="dot", annotation_text=f"Current ${S0:.0f}")
        fig4.update_layout(**PLOT_THEME, height=300, margin=dict(l=0,r=0,t=10,b=0),
                           title="Final Price Distribution", xaxis_title="Price ($)", yaxis_title="Count")
        st.plotly_chart(fig4, use_container_width=True)
    with col2:
        st.markdown(f"**Simulation Summary ({res['sim_horizon']} days)**")
        d = {
            "Starting Price": f"${S0:.2f}",
            "Median Ending Price": f"${np.median(final):.2f}",
            "Mean Ending Price": f"${final.mean():.2f}",
            "5th Percentile": f"${np.percentile(final,5):.2f}",
            "95th Percentile": f"${np.percentile(final,95):.2f}",
            "Paths Above Strike": f"{(final>K).mean()*100:.1f}%",
            "Paths Below Strike": f"{(final<=K).mean()*100:.1f}%",
        }
        for k, v in d.items():
            st.markdown(f"**{k}:** `{v}`")

# ── TAB 3: Option Pricing ─────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Black-Scholes Option Pricing</div>', unsafe_allow_html=True)
    st.latex(r"C = S_0 N(d_1) - K e^{-rT} N(d_2) \qquad P = K e^{-rT} N(-d_2) - S_0 N(-d_1)")
    st.latex(r"d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}} \qquad d_2 = d_1 - \sigma\sqrt{T}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Call Price", f"${res['call_price']:.4f}")
    col2.metric("Put Price", f"${res['put_price']:.4f}")
    col3.metric("Strike Price", f"${K:.2f}")

    st.markdown("---")
    # Price vs Strike chart
    strikes = np.linspace(S0 * 0.5, S0 * 1.5, 100)
    T_y = res["T_years"]
    calls = [bs_call(S0, k, T_y, r, params["annual_vol"]) for k in strikes]
    puts = [bs_put(S0, k, T_y, r, params["annual_vol"]) for k in strikes]

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=strikes, y=calls, name="Call", line=dict(color="#3fb950", width=2)))
    fig5.add_trace(go.Scatter(x=strikes, y=puts, name="Put", line=dict(color="#f85149", width=2)))
    fig5.add_vline(x=K, line_color="#f0883e", line_dash="dash", annotation_text=f"Strike=${K:.0f}")
    fig5.add_vline(x=S0, line_color="#8b949e", line_dash="dot", annotation_text=f"S₀=${S0:.0f}")
    fig5.update_layout(**PLOT_THEME, height=350, margin=dict(l=0,r=0,t=10,b=0),
                       title="Option Price vs Strike", xaxis_title="Strike ($)", yaxis_title="Option Price ($)",
                       legend=dict(bgcolor="#161b22", bordercolor="#30363d"))
    st.plotly_chart(fig5, use_container_width=True)

    # Price vs Vol surface
    vols = np.linspace(0.05, 0.8, 50)
    ts = np.linspace(7/365, 1.0, 50)
    ZC = np.array([[bs_call(S0, K, t, r, v) for v in vols] for t in ts])
    fig6 = go.Figure(go.Surface(x=vols*100, y=[int(t*365) for t in ts], z=ZC,
                                 colorscale="Blues", showscale=False))
    fig6.update_layout(**PLOT_THEME, height=380, margin=dict(l=0,r=0,t=30,b=0),
                       title="Call Price Surface (Vol × Maturity)",
                       scene=dict(xaxis_title="Vol (%)", yaxis_title="Days", zaxis_title="Call ($)",
                                  bgcolor="#0d1117",
                                  xaxis=dict(backgroundcolor="#0d1117", gridcolor="#21262d"),
                                  yaxis=dict(backgroundcolor="#0d1117", gridcolor="#21262d"),
                                  zaxis=dict(backgroundcolor="#0d1117", gridcolor="#21262d")))
    st.plotly_chart(fig6, use_container_width=True)

# ── TAB 4: Greeks ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Option Greeks Dashboard</div>', unsafe_allow_html=True)

    GREEKS_INFO = {
        "Delta": {
            "formula": "∂V/∂S  =  N(d₁)  [call]  or  N(d₁)−1  [put]",
            "desc": "Rate of change of option price per $1 move in the stock. A delta of 0.6 means the option gains ~$0.60 when the stock rises $1.",
            "color": "#58a6ff",
        },
        "Gamma": {
            "formula": "∂²V/∂S²  =  N'(d₁) / (S·σ·√T)",
            "desc": "Rate of change of Delta per $1 move. High Gamma means Delta changes fast — useful near expiry or at-the-money.",
            "color": "#3fb950",
        },
        "Vega": {
            "formula": "∂V/∂σ  =  S·N'(d₁)·√T / 100",
            "desc": "Sensitivity to a 1% change in implied volatility. Positive for both calls and puts — all options benefit from rising vol.",
            "color": "#f0883e",
        },
        "Theta": {
            "formula": "∂V/∂T  =  [−S·N'(d₁)·σ/(2√T) − r·K·e^{−rT}·N(d₂)] / 365",
            "desc": "Time decay per day. Usually negative — the option loses value as expiry approaches (all else equal).",
            "color": "#d2a8ff",
        },
        "Rho": {
            "formula": "∂V/∂r  =  K·T·e^{−rT}·N(d₂) / 100  [call]",
            "desc": "Sensitivity to a 1% change in interest rates. Calls gain value as rates rise (positive Rho); puts lose value.",
            "color": "#ffa657",
        },
    }

    cols = st.columns(5)
    for col, (name, val) in zip(cols, greeks_vals.items()):
        info = GREEKS_INFO[name]
        col.markdown(f"""
        <div class="greek-card">
          <div class="greek-name" style="color:{info['color']}">{name}</div>
          <div class="greek-val">{val:.5f}</div>
          <div class="greek-formula">{info['formula']}</div>
          <div class="greek-desc">{info['desc']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    # Greeks vs Stock Price
    spot_range = np.linspace(S0 * 0.5, S0 * 1.5, 200)
    from greeks import delta as calc_delta, gamma as calc_gamma, vega as calc_vega, theta as calc_theta, rho as calc_rho

    deltas = [calc_delta(s, K, res["T_years"], r, params["annual_vol"], option_type) for s in spot_range]
    gammas = [calc_gamma(s, K, res["T_years"], r, params["annual_vol"]) for s in spot_range]
    vegas  = [calc_vega(s, K, res["T_years"], r, params["annual_vol"]) for s in spot_range]
    thetas = [calc_theta(s, K, res["T_years"], r, params["annual_vol"], option_type) for s in spot_range]

    fig7 = make_subplots(rows=2, cols=2,
                         subplot_titles=["Delta vs Spot", "Gamma vs Spot", "Vega vs Spot", "Theta vs Spot"])
    data_pairs = [(deltas,"#58a6ff","Delta"), (gammas,"#3fb950","Gamma"),
                  (vegas,"#f0883e","Vega"), (thetas,"#d2a8ff","Theta")]
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for (row,col), (vals,color,name) in zip(positions, data_pairs):
        fig7.add_trace(go.Scatter(x=spot_range, y=vals, name=name, line=dict(color=color, width=2)), row=row, col=col)
        fig7.add_vline(x=S0, line_color="#8b949e", line_dash="dot", row=row, col=col)
        fig7.add_vline(x=K, line_color="#f0883e", line_dash="dash", row=row, col=col)
    fig7.update_layout(**PLOT_THEME, height=450, margin=dict(l=0,r=0,t=40,b=0),
                       showlegend=False)
    st.plotly_chart(fig7, use_container_width=True)
    st.caption("Grey dotted = current price · Orange dashed = strike price")

# ── TAB 5: Risk Metrics ───────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Risk Metrics from Monte Carlo</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    metric(col1, "VaR 95%", f"{risk['VaR_95']*100:.2f}%", "Max loss (5% chance)")
    metric(col2, "Expected Shortfall", f"{risk['ES_95']*100:.2f}%", "Avg tail loss")
    metric(col3, "P(above strike)", f"{risk['prob_above_strike']*100:.1f}%", f"S > ${K:.0f}")
    metric(col4, "P(below strike)", f"{risk['prob_below_strike']*100:.1f}%", f"S ≤ ${K:.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        final = paths[-1]
        ret_dist = (final - S0) / S0 * 100
        var_val = risk["VaR_95"] * 100
        fig8 = go.Figure()
        fig8.add_trace(go.Histogram(x=ret_dist, nbinsx=60, marker_color="#58a6ff", opacity=0.8, name="Return %"))
        # shade tail
        tail_x = ret_dist[ret_dist <= var_val]
        if len(tail_x):
            fig8.add_trace(go.Histogram(x=tail_x, nbinsx=20, marker_color="#f85149", opacity=0.9, name="Tail (5%)"))
        fig8.add_vline(x=var_val, line_color="#f85149", line_dash="dash",
                       annotation_text=f"VaR 95%: {var_val:.1f}%")
        fig8.update_layout(**PLOT_THEME, height=320, margin=dict(l=0,r=0,t=10,b=0),
                           title="Return Distribution with VaR",
                           xaxis_title="Return (%)", yaxis_title="Count",
                           legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
                           barmode="overlay")
        st.plotly_chart(fig8, use_container_width=True)

    with col2:
        st.markdown("**What do these metrics mean?**")
        st.markdown(f"""
**Value at Risk (VaR 95%):** `{risk['VaR_95']*100:.2f}%`
> With 95% confidence, your loss over {res['sim_horizon']} days will not exceed **{abs(risk['VaR_95'])*100:.2f}%** of the stock value.

**Expected Shortfall (CVaR):** `{risk['ES_95']*100:.2f}%`
> On the worst 5% of days, the average loss is **{abs(risk['ES_95'])*100:.2f}%**. More conservative than VaR.

**Probability above strike:** `{risk['prob_above_strike']*100:.1f}%`
> Chance the stock ends above **${K:.2f}** (strike) — useful for call option holders.

**Probability below strike:** `{risk['prob_below_strike']*100:.1f}%`
> Chance the stock ends below **${K:.2f}** — useful for put option holders.
        """)

        st.markdown("---")
        st.markdown("**Simulation Statistics**")
        for k, v in {
            "Mean Final Price": f"${risk['mean_final']:.2f}",
            "Std Dev Final Price": f"${risk['std_final']:.2f}",
            "Min Final Price": f"${risk['min_final']:.2f}",
            "Max Final Price": f"${risk['max_final']:.2f}",
        }.items():
            st.markdown(f"**{k}:** `{v}`")

    # Cumulative loss probability
    sorted_ret = np.sort(ret_dist)
    cum_prob = np.arange(1, len(sorted_ret)+1) / len(sorted_ret) * 100
    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=sorted_ret, y=cum_prob, name="CDF",
                               line=dict(color="#58a6ff", width=2)))
    fig9.add_hline(y=5, line_color="#f85149", line_dash="dash", annotation_text="5% (VaR threshold)")
    fig9.add_vline(x=0, line_color="#8b949e", line_dash="dot", annotation_text="0%")
    fig9.update_layout(**PLOT_THEME, height=300, margin=dict(l=0,r=0,t=10,b=0),
                       title="Cumulative Loss Probability",
                       xaxis_title="Return (%)", yaxis_title="Cumulative Probability (%)",
                       legend=dict(bgcolor="#161b22", bordercolor="#30363d"))
    st.plotly_chart(fig9, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit · Black-Scholes · GBM Monte Carlo · 1,000 simulated paths · Educational purposes only.")
